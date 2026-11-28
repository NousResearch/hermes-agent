"""LINE adapter: send a file (e.g. an .xlsx report) to the user.

Why this exists
---------------
LINE's Messaging API has **no ``document`` push message type**. The push endpoint
accepts only::

    text, textV2, image, video, audio, location, sticker, template, imagemap, flex

A ``{"type": "document", ...}`` payload is rejected with HTTP 400 ("Must be one
of the following values: ..."). The ``file`` type exists only in the *inbound*
direction (a file the user sends the bot).

So the supported way to hand a file to a LINE user is a **flex message with a
download button**: the bubble carries a ``uri`` action pointing at the gateway's
``/line/media/<token>/<name>`` route, which serves the local file over the same
public host the webhook uses.

Schema traps this covers:

1. A button action label is capped at **20 characters** — a longer label makes
   the API reject the *entire* flex payload.
2. ``altText`` is required on a flex message.

Separately, the cron delivery layer drops MEDIA attachments on LINE unless
``line`` is registered in ``gateway.outbound.send._CHUNKED_ROUTES`` (covered by
``tests/tools/test_line_media_delivery.py``), and the out-of-process (standalone)
sender needs the loopback media-publish endpoint (covered here).
"""

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from plugins.platforms.line.adapter import LineAdapter

UID = "U0123456789abcdef0123456789abcdef"


class _FakeResponse:
    status = 200

    async def json(self):
        return {"id": "fake-message-id"}

    async def text(self):
        return '{"id": "fake-message-id"}'

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _RecordingClient:
    """Captures the messages the adapter pushes to LINE."""

    def __init__(self):
        self.messages: list[dict] = []

    async def push(self, chat_id, messages):
        self.messages.extend(messages)

    async def reply(self, reply_token, messages):
        self.messages.extend(messages)


def _adapter(tmp_path, public_url="https://example.test"):
    """A LineAdapter with just the fields send_document touches."""
    ad = LineAdapter.__new__(LineAdapter)
    ad._client = _RecordingClient()
    ad.public_base_url = public_url
    ad._media_tokens = {}
    ad._media_ttl = 86400
    ad._media_temp_paths = set()
    ad._reply_tokens = {}
    ad._last_sent = None
    ad.webhook_host = "127.0.0.1"
    ad.webhook_port = 8646
    ad.channel_secret = "s3cret"
    return ad


def _xlsx(tmp_path, name="零部件價格_2026-11-13.xlsx") -> Path:
    p = Path(tmp_path) / name
    p.write_bytes(b"PK\x03\x04" + b"\x00" * 64)
    return p


def _walk(node, fn):
    if isinstance(node, dict):
        fn(node)
        for value in node.values():
            _walk(value, fn)
    elif isinstance(node, list):
        for value in node:
            _walk(value, fn)


def _collect(node, node_type):
    found = []
    _walk(node, lambda n: found.append(n) if n.get("type") == node_type else None)
    return found


class TestSendDocumentProducesFlex:
    """The pushed payload must be a flex message the LINE API accepts."""

    def test_push_contains_a_flex_message(self, tmp_path):
        ad = _adapter(tmp_path)
        result = asyncio.run(ad.send_document(UID, str(_xlsx(tmp_path))))
        assert result.success, result.error
        assert ad._client.messages, "nothing was pushed to LINE"

    def test_no_document_type_is_ever_pushed(self, tmp_path):
        """LINE rejects type=document with HTTP 400 — it must never appear."""
        ad = _adapter(tmp_path)
        asyncio.run(ad.send_document(UID, str(_xlsx(tmp_path))))
        types = {m.get("type") for m in ad._client.messages}
        assert "document" not in types

    def test_all_pushed_types_are_valid_for_the_push_api(self, tmp_path):
        allowed = {
            "text", "textV2", "image", "video", "audio", "location",
            "sticker", "template", "imagemap", "flex",
        }
        ad = _adapter(tmp_path)
        asyncio.run(ad.send_document(UID, str(_xlsx(tmp_path))))
        for message in ad._client.messages:
            assert message.get("type") in allowed, message.get("type")

    def test_flex_has_required_fields(self, tmp_path):
        ad = _adapter(tmp_path)
        asyncio.run(ad.send_document(UID, str(_xlsx(tmp_path))))
        flex = next(m for m in ad._client.messages if m.get("type") == "flex")
        assert flex.get("altText"), "flex requires altText"
        assert flex.get("contents"), "flex requires contents"

    def test_download_button_points_at_the_media_route(self, tmp_path):
        ad = _adapter(tmp_path)
        asyncio.run(ad.send_document(UID, str(_xlsx(tmp_path))))
        flex = next(m for m in ad._client.messages if m.get("type") == "flex")
        buttons = _collect(flex["contents"], "button")
        assert buttons, "no download button in the flex bubble"
        action = buttons[0]["action"]
        assert action["type"] == "uri"
        assert action["uri"].startswith("https://")
        assert "/line/media/" in action["uri"]

    def test_download_button_survives_a_cjk_filename(self, tmp_path):
        from urllib.parse import unquote

        ad = _adapter(tmp_path)
        asyncio.run(ad.send_document(UID, str(_xlsx(tmp_path))))
        flex = next(m for m in ad._client.messages if m.get("type") == "flex")
        uri = _collect(flex["contents"], "uri")[0]["uri"]
        assert "零部件價格_2026-11-13.xlsx" in unquote(uri)

    def test_button_label_stays_within_the_line_limit(self, tmp_path):
        """LINE caps a button action label at 20 characters."""
        ad = _adapter(tmp_path)
        asyncio.run(ad.send_document(UID, str(_xlsx(tmp_path))))
        flex = next(m for m in ad._client.messages if m.get("type") == "flex")
        for action in _collect(flex["contents"], "uri"):
            assert len(action["label"]) <= 20, action["label"]

    def test_caption_is_also_pushed_as_plain_text(self, tmp_path):
        """The caption should be readable in the chat body without tapping."""
        ad = _adapter(tmp_path)
        asyncio.run(ad.send_document(UID, str(_xlsx(tmp_path)), caption="組裝建議"))
        texts = [m.get("text", "") for m in ad._client.messages if m.get("type") == "text"]
        assert any("組裝建議" in t for t in texts)


class TestSendDocumentGuards:
    """A broken file must fail loudly instead of sending a dead link."""

    def test_missing_file_fails(self, tmp_path):
        ad = _adapter(tmp_path)
        result = asyncio.run(ad.send_document(UID, str(tmp_path / "missing.xlsx")))
        assert not result.success
        assert "not found" in (result.error or "").lower()

    def test_without_a_public_url_it_refuses_to_send(self, tmp_path):
        """Without LINE_PUBLIC_URL there is no fetchable HTTPS URL."""
        ad = _adapter(tmp_path, public_url="")
        ad.webhook_host = "0.0.0.0"  # wildcard bind -> no public hostname
        result = asyncio.run(ad.send_document(UID, str(_xlsx(tmp_path))))
        assert not result.success
        assert "PUBLIC_URL" in (result.error or "").upper()


class TestMediaTokenLifetime:
    """Tokens are permanent — a link dies when its file does, not on a clock."""

    def test_a_registered_token_survives_long_past_any_ttl(self, tmp_path):
        ad = _adapter(tmp_path)
        path = _xlsx(tmp_path)
        url = ad._serve_file(path)
        token = url.split("/line/media/")[1].split("/")[0]
        # Simulate a link opened much later than any reasonable TTL.
        ad._media_tokens[token] = (ad._media_tokens[token][0], time.time() + 10 ** 9)
        assert token in ad._media_tokens

    def test_a_token_whose_file_is_gone_is_dropped(self, tmp_path):
        """The file disappearing is what invalidates a permanent token."""
        ad = _adapter(tmp_path)
        path = _xlsx(tmp_path)
        url = ad._serve_file(path)
        token = url.split("/line/media/")[1].split("/")[0]
        path.unlink()
        assert not Path(ad._media_tokens[token][0]).is_file()


class TestStandaloneLanePublishesMedia:
    """The out-of-process cron lane must also be able to hand over a file.

    The standalone sender builds its own ``_LineClient`` (it has no adapter),
    so the test intercepts that class to capture what would be pushed.
    """

    @pytest.fixture
    def captured(self, monkeypatch):
        import plugins.platforms.line.adapter as adapter_module

        pushed: list[dict] = []

        class _FakeLineClient:
            def __init__(self, token, *a, **kw):
                self.token = token

            async def push(self, chat_id, messages):
                pushed.extend(messages)

        monkeypatch.setattr(adapter_module, "_LineClient", _FakeLineClient)
        monkeypatch.setattr(
            adapter_module, "_credentials", lambda pconfig: ("test-token", "test-secret"))
        return pushed

    def test_standalone_sender_asks_the_gateway_to_publish(self, tmp_path, monkeypatch, captured):
        import plugins.platforms.line.adapter as adapter_module

        published = {}

        def fake_publish(path):
            published["path"] = path
            return {"url": "https://example.test/line/media/tok/x.xlsx",
                    "path": path}

        monkeypatch.setattr(
            "gateway.media_publish.publish_media_file", fake_publish, raising=False)

        pconfig = type("Cfg", (), {"extra": {}})()
        result = asyncio.run(adapter_module._standalone_send(
            pconfig, UID, "價格表", media_files=[str(_xlsx(tmp_path))]))

        assert "error" not in result, result
        assert published.get("path"), "the file was never published"
        flexes = [m for m in captured if m.get("type") == "flex"]
        assert flexes, "no flex download bubble was pushed"

    def test_standalone_sender_normalises_tuple_media_entries(self, tmp_path, monkeypatch, captured):
        """The cron delivery layer passes [(path, is_voice)] tuples."""
        import plugins.platforms.line.adapter as adapter_module

        seen = {}

        def fake_publish(path):
            seen["path"] = path
            return {"url": "https://example.test/line/media/tok/x.xlsx", "path": path}

        monkeypatch.setattr(
            "gateway.media_publish.publish_media_file", fake_publish, raising=False)

        pconfig = type("Cfg", (), {"extra": {}})()
        asyncio.run(adapter_module._standalone_send(
            pconfig, UID, "價格表",
            media_files=[(str(_xlsx(tmp_path)), False)]))

        assert seen.get("path"), "tuple media entry was not normalised"

    def test_standalone_sender_reports_when_publish_fails(self, tmp_path, monkeypatch, captured):
        """A failed publish must be visible, not silently dropped."""
        import plugins.platforms.line.adapter as adapter_module

        def fake_publish(path):
            raise RuntimeError("gateway not reachable")

        monkeypatch.setattr(
            "gateway.media_publish.publish_media_file", fake_publish, raising=False)

        pconfig = type("Cfg", (), {"extra": {}})()
        asyncio.run(adapter_module._standalone_send(
            pconfig, UID, "價格表", media_files=[str(_xlsx(tmp_path))]))

        texts = [m.get("text", "") for m in captured if m.get("type") == "text"]
        # The fallback comes from the i18n catalog, so assert on the *shape*
        # (a notice was added) rather than on a translated literal.
        assert len(texts) >= 2, f"expected the fallback notice, got {texts}"


class TestMediaPublishEndpointIsLoopbackOnly:
    """The publish endpoint mints publicly reachable URLs — lock it down."""

    def test_endpoint_rejects_non_loopback_peers(self, tmp_path):
        ad = _adapter(tmp_path)

        class _Request:
            remote = "203.0.113.9"
            headers = {"Authorization": "Bearer s3cret"}

            async def json(self):
                return {"path": str(_xlsx(tmp_path))}

        response = asyncio.run(ad._handle_media_publish(_Request()))
        assert response.status == 403

    def test_endpoint_requires_the_channel_secret(self, tmp_path):
        ad = _adapter(tmp_path)

        class _Request:
            remote = "127.0.0.1"
            headers = {"Authorization": "Bearer wrong"}

            async def json(self):
                return {"path": str(_xlsx(tmp_path))}

        response = asyncio.run(ad._handle_media_publish(_Request()))
        assert response.status == 401

    def test_endpoint_returns_a_public_url_for_a_valid_request(self, tmp_path):
        ad = _adapter(tmp_path)

        class _Request:
            remote = "127.0.0.1"
            headers = {"Authorization": "Bearer s3cret"}

            async def json(self):
                return {"path": str(_xlsx(tmp_path))}

        response = asyncio.run(ad._handle_media_publish(_Request()))
        assert response.status == 200
        payload = json.loads(response.text)
        assert payload["url"].startswith("https://")
        assert "/line/media/" in payload["url"]


class TestMediaTokenRegistryPersists:
    """A gateway restart must not invalidate links the user already holds."""

    def test_registry_survives_a_reload(self, tmp_path):
        ad = _adapter(tmp_path)
        registry = Path(tmp_path) / "media_tokens.json"
        ad._media_tokens_path = str(registry)
        ad._save_media_tokens = LineAdapter._save_media_tokens.__get__(ad)

        url = ad._serve_file(_xlsx(tmp_path))
        token = url.split("/line/media/")[1].split("/")[0]

        # A fresh adapter reads the same file back.
        restored = _adapter(tmp_path)
        restored._media_tokens_path = str(registry)
        restored._load_media_tokens = LineAdapter._load_media_tokens.__get__(restored)
        restored._load_media_tokens()
        assert token in restored._media_tokens

    def test_saved_registry_is_strict_json(self, tmp_path):
        """Permanent tokens must not serialise as the non-standard Infinity."""
        ad = _adapter(tmp_path)
        path = Path(tmp_path) / "media_tokens.json"
        ad._media_tokens_path = str(path)
        ad._save_media_tokens = LineAdapter._save_media_tokens.__get__(ad)
        ad._media_tokens["tok"] = (str(_xlsx(tmp_path)), float("inf"))
        ad._save_media_tokens()

        raw = path.read_text(encoding="utf-8")
        json.loads(raw, parse_constant=lambda c: pytest.fail(f"non-finite JSON: {c}"))

    def test_entries_whose_file_is_gone_are_not_restored(self, tmp_path):
        ad = _adapter(tmp_path)
        ad._media_tokens_path = str(Path(tmp_path) / "media_tokens.json")
        Path(ad._media_tokens_path).write_text(json.dumps({
            "tokens": {"gone": {"path": str(tmp_path / "nope.xlsx"), "expires_at": None}},
        }), encoding="utf-8")
        ad._load_media_tokens = LineAdapter._load_media_tokens.__get__(ad)
        ad._load_media_tokens()
        assert "gone" not in ad._media_tokens


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
