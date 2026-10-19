"""Every Slack post stays within ``MAX_MESSAGE_LENGTH`` (39,000) after mrkdwn formatting.

Formatting can grow text (``&`` becomes ``&amp;``), so a split on the raw text does not bound
the post. Each lane (standalone text, standalone media text, live adapter, and the cron delivery
chain wrapper -> redaction -> format) is checked at the outermost fake: the HTTP session or the
SDK client.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from plugins.platforms.slack.adapter import SlackAdapter, _standalone_send

LIMIT = SlackAdapter.MAX_MESSAGE_LENGTH
# Synthetic vendor-prefixed key, long enough to trip the redactor; never a real credential.
FAKE_SECRET = "sk-" + "Q" * 32


class _Response:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


class _PostContext:
    def __init__(self, payload):
        self._payload = payload

    async def __aenter__(self):
        return _Response(self._payload)

    async def __aexit__(self, *exc):
        return False


class _Session:
    """Fake aiohttp session: ``good-token`` posts succeed unless ``fail_at`` names the call."""

    def __init__(self, fail_at=None):
        self.calls: list[tuple[str, dict]] = []
        self.fail_at = fail_at

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, *, headers, json, **kwargs):
        token = headers["Authorization"].removeprefix("Bearer ")
        self.calls.append((token, json))
        if token != "good-token":
            return _PostContext({"ok": False, "error": "invalid_auth"})
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            return _PostContext({"ok": False, "error": "ratelimited"})
        return _PostContext({"ok": True, "ts": f"171.{len(self.calls)}"})

    def texts(self, token="good-token"):
        return [payload["text"] for tok, payload in self.calls if tok == token]


@pytest.fixture
def session(monkeypatch):
    fake = _Session()
    monkeypatch.setattr("aiohttp.ClientSession", lambda *a, **kw: fake)
    return fake


def _pconfig(token="good-token"):
    return SimpleNamespace(enabled=True, token=token, extra={})


def _send(message, **kwargs):
    return asyncio.run(_standalone_send(_pconfig(kwargs.pop("token", "good-token")), "C123",
                                        message, **kwargs))


def test_standalone_text_that_grows_when_formatted_is_split_under_the_limit(session):
    result = _send("&" * LIMIT)
    texts = session.texts()
    assert result.get("success") is True
    assert len(texts) > 1 and max(map(len, texts)) <= LIMIT
    assert sum(text.count("&amp;") for text in texts) == LIMIT


def test_standalone_small_and_boundary_text_is_one_unchanged_post(session):
    assert _send("hello")["success"] is True
    assert _send("x" * LIMIT)["success"] is True
    assert session.texts() == ["hello", "x" * LIMIT]


def test_standalone_blank_text_posts_nothing(session):
    assert _send("   ")["skipped"] == "empty_text"
    assert session.calls == []


def test_later_chunks_reuse_the_token_that_took_the_first(session):
    result = _send("&" * LIMIT, token="bad-token,good-token")
    assert result.get("success") is True
    assert [tok for tok, _ in session.calls] == ["bad-token"] + ["good-token"] * len(session.texts())


def test_a_failed_later_chunk_stops_and_reports_what_was_delivered(monkeypatch):
    fake = _Session(fail_at=2)
    monkeypatch.setattr("aiohttp.ClientSession", lambda *a, **kw: fake)
    result = _send("&" * (LIMIT * 2))
    assert "ratelimited" in result["error"]
    assert result["delivered_chunks"] == 1 and result["total_chunks"] > 2
    assert len(fake.calls) == 2


@contextlib.contextmanager
def _fake_slack_sdk(client):
    async_client = ModuleType("slack_sdk.web.async_client")
    async_client.AsyncWebClient = MagicMock(return_value=client)
    modules = {"slack_sdk": ModuleType("slack_sdk"), "slack_sdk.web": ModuleType("slack_sdk.web"),
               "slack_sdk.web.async_client": async_client}
    with patch.dict(sys.modules, modules):
        yield


def _sdk_client():
    client = MagicMock()
    client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "1.1"})
    client.files_upload_v2 = AsyncMock(return_value={"ok": True, "file": {"timestamp": 1}})
    return client


@pytest.fixture
def media_file():
    handle = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    handle.write(b"%PDF-1.4 test")
    handle.close()
    yield handle.name
    os.unlink(handle.name)


@pytest.mark.parametrize("as_caption", [False, True])
def test_standalone_media_text_is_split_under_the_limit(media_file, as_caption):
    client = _sdk_client()
    text = "&" * LIMIT
    kwargs = {"caption": text} if as_caption else {}
    with _fake_slack_sdk(client):
        result = _send("" if as_caption else text, media_files=[(media_file, False)], **kwargs)
    posted = [call.kwargs["text"] for call in client.chat_postMessage.await_args_list]
    assert result.get("success") is True
    assert len(posted) > 1 and max(map(len, posted)) <= LIMIT
    assert client.files_upload_v2.await_args.kwargs["initial_comment"] == ""


def test_a_failed_later_media_text_chunk_reports_what_was_delivered(media_file):
    client = _sdk_client()
    client.chat_postMessage = AsyncMock(side_effect=[
        {"ok": True, "ts": "1.1"}, {"ok": False, "error": "ratelimited"}])
    with _fake_slack_sdk(client):
        result = _send("&" * (LIMIT * 2), media_files=[(media_file, False)])
    assert "ratelimited" in result["error"]
    assert result["delivered_chunks"] == 1 and result["total_chunks"] > 2
    client.files_upload_v2.assert_not_awaited()


def test_short_caption_still_rides_the_upload(media_file):
    client = _sdk_client()
    with _fake_slack_sdk(client):
        result = _send("", media_files=[(media_file, False)], caption="see this & that")
    assert result.get("success") is True
    client.chat_postMessage.assert_not_awaited()
    assert client.files_upload_v2.await_args.kwargs["initial_comment"] == "see this &amp; that"


def test_live_adapter_posts_stay_under_the_limit_for_fenced_and_escaped_text():
    adapter = SlackAdapter(SimpleNamespace(enabled=True, token="x", extra={}))
    adapter._app = MagicMock()
    adapter._app.client = AsyncMock()
    adapter._app.client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "1.0"})
    body = "```\n" + ("a & b\n" * 9000) + "```\n" + ("tail & more " * 4000)
    result = asyncio.run(adapter.send("C123", body))
    posted = [c.kwargs["text"] for c in adapter._app.client.chat_postMessage.await_args_list]
    assert result.success is True
    assert len(posted) > 1 and max(map(len, posted)) <= LIMIT


def _slack_cfg():
    from gateway.config import Platform

    cfg = MagicMock()
    cfg.platforms = {Platform.SLACK: _pconfig()}
    return cfg


def _leaks(texts) -> int:
    return sum(FAKE_SECRET in text for text in texts)


@pytest.mark.parametrize("body_len", [0, 10, LIMIT - 200, LIMIT, LIMIT * 3])
def test_cron_delivery_chain_posts_stay_under_the_limit(session, body_len):
    """``_deliver_result`` -> wrapper -> redaction -> real ``_send_to_platform`` -> the Slack
    plugin's standalone sender: every post fits, nothing is lost, the secret never leaves."""
    from cron.scheduler_delivery import _deliver_result
    from hermes_cli.plugins import discover_plugins

    discover_plugins()
    body = f"token {FAKE_SECRET} " + "&" * body_len + " END-MARKER"
    job = {"id": "limit-job", "name": "limit", "deliver": "origin",
           "origin": {"platform": "slack", "chat_id": "C123"}}
    with patch("gateway.config.load_gateway_config", return_value=_slack_cfg()), \
            patch("sys.is_finalizing", return_value=False):
        error = _deliver_result(job, body)

    texts = session.texts()
    assert error is None
    assert texts and max(map(len, texts)) <= LIMIT
    joined = "".join(texts)
    assert "Cronjob Response: limit" in joined and "END-MARKER" in joined
    assert joined.count("&amp;") == body_len
    assert _leaks(texts) == 0


class _ResponseUrlSession:
    """Fake aiohttp session for ``response_url``: every POST succeeds and is recorded."""

    def __init__(self):
        self.payloads: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, *, json, **kwargs):
        self.payloads.append(json)
        response = MagicMock(status=200)
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=False)
        return response


@pytest.mark.parametrize("body", ["x" * (LIMIT * 6), "&" * (LIMIT * 3), "word " * LIMIT],
                         ids=["hard-cuts", "escaped", "words"])
def test_slash_reply_over_five_posts_keeps_every_payload_under_the_limit(monkeypatch, body):
    """The truncation notice rides the fifth POST: that payload, notice included, still fits."""
    import aiohttp

    fake = _ResponseUrlSession()
    # The adapter binds aiohttp only when the Slack SDK imports; the real module is enough here.
    monkeypatch.setattr("plugins.platforms.slack.adapter.aiohttp", aiohttp)
    monkeypatch.setattr("aiohttp.ClientSession", lambda *a, **kw: fake)
    adapter = SlackAdapter(SimpleNamespace(enabled=True, token="x", extra={}))
    result = asyncio.run(adapter._send_slash_ephemeral(
        {"response_url": "https://hooks.slack.com/commands/long"}, body))
    texts = [payload["text"] for payload in fake.payloads]
    assert result.success is True
    assert len(texts) == 5 and max(map(len, texts)) <= LIMIT
    assert texts[-1].endswith("]_")
