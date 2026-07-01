"""Tests for the Chatwoot Agent Bot plugin adapter.

Covers doc §9: config/connection, chat-id round-trip, the inbound filtering
matrix, the webhook HTTP handler + idempotency, authorization/registration
wiring, scheduled-delivery wiring, the private-note trace, and attachments.

Behavior/invariants only — no snapshots, no live Chatwoot.
"""

import json

import pytest

from gateway.config import Platform, PlatformConfig
from plugins.platforms.chatwoot import adapter as cw


# ── fakes ────────────────────────────────────────────────────────────────────


class _FakeResp:
    def __init__(self, status: int, payload=None, text_body: str = ""):
        self.status = status
        self._payload = payload
        self._text = text_body or (json.dumps(payload) if payload is not None else "")

    async def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    async def text(self):
        return self._text

    async def read(self):
        return self._text.encode("utf-8")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None


class _FakeSession:
    """Records POST/GET calls and replays scripted responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []  # list of (method, url, kwargs)

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        if not self._responses:
            raise AssertionError(f"No scripted response for POST {url}")
        return self._responses.pop(0)

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        if not self._responses:
            raise AssertionError(f"No scripted response for GET {url}")
        return self._responses.pop(0)

    async def close(self):
        return None


class _FakeRequest:
    def __init__(self, body: bytes = b"", token: str = None, content_length=None):
        self._body = body
        self.query = {} if token is None else {"token": token}
        self.content_length = content_length if content_length is not None else len(body)

    async def read(self):
        return self._body


def _make_adapter(**extra) -> cw.ChatwootAdapter:
    base_extra = {"base_url": "https://cw.example.com", "account_id": "1"}
    base_extra.update(extra)
    cfg = PlatformConfig(enabled=True, token="bot-tok", extra=base_extra)
    a = cw.ChatwootAdapter(cfg)
    a._running = True
    return a


def _msg_created(**over):
    payload = {
        "event": "message_created",
        "message_type": "incoming",
        "private": False,
        "content": "hello there",
        "id": 555,
        "conversation": {"id": 42, "display_id": 42, "status": "pending"},
        "account": {"id": 1},
        "sender": {"id": 88, "name": "Jane Doe", "email": "jane@example.com"},
    }
    payload.update(over)
    return payload


# ── chat-id helpers ──────────────────────────────────────────────────────────


class TestChatId:
    def test_format(self):
        assert cw._format_chat_id(1, 42) == "1:42"

    def test_parse_round_trip(self):
        assert cw._parse_chat_id("1:42") == ("1", "42")

    def test_parse_bare_conversation_uses_default_account(self):
        assert cw._parse_chat_id("42", "7") == ("7", "42")

    def test_parse_bare_without_default_raises(self):
        with pytest.raises(ValueError):
            cw._parse_chat_id("42")

    def test_adapter_parse_uses_configured_account(self):
        a = _make_adapter(account_id="9")
        assert a._parse_chat_id("55") == ("9", "55")


# ── config / connection ──────────────────────────────────────────────────────


class TestConfig:
    def test_connected_requires_base_url_and_token(self):
        assert cw.is_connected(PlatformConfig(enabled=True, token="t", extra={"base_url": "u"}))
        assert not cw.is_connected(PlatformConfig(enabled=True, token="t", extra={}))
        assert not cw.is_connected(PlatformConfig(enabled=True, extra={"base_url": "u"}))

    def test_trailing_slash_stripped(self):
        a = _make_adapter(base_url="https://cw.example.com/")
        assert a._base_url == "https://cw.example.com"

    def test_env_enablement_none_when_unconfigured(self, monkeypatch):
        for k in ("CHATWOOT_BASE_URL", "CHATWOOT_TOKEN"):
            monkeypatch.delenv(k, raising=False)
        assert cw._env_enablement() is None

    def test_env_enablement_seeds_extra(self, monkeypatch):
        monkeypatch.setenv("CHATWOOT_BASE_URL", "https://cw.example.com/")
        monkeypatch.setenv("CHATWOOT_TOKEN", "bot-tok")
        monkeypatch.setenv("CHATWOOT_AGENT_TOKEN", "agent-tok")
        monkeypatch.setenv("CHATWOOT_HOME_CHANNEL", "1:42")
        monkeypatch.setenv("CHATWOOT_PRIVATE_NOTE_TRACE", "true")
        seed = cw._env_enablement()
        assert seed["base_url"] == "https://cw.example.com"  # stripped
        assert seed["token"] == "bot-tok"
        assert seed["agent_token"] == "agent-tok"
        assert seed["private_note_trace"] is True
        assert seed["home_channel"]["chat_id"] == "1:42"

    def test_requirements(self):
        # aiohttp is a test dependency, so this should be True in CI.
        assert cw.check_chatwoot_requirements() is True


# ── direction filter ─────────────────────────────────────────────────────────


class TestIsIncoming:
    @pytest.mark.parametrize("val", ["incoming", "INCOMING", 0])
    def test_incoming_accepted(self, val):
        assert cw._is_incoming(val) is True

    @pytest.mark.parametrize("val", ["outgoing", 1, 2, 3, True, None, ""])
    def test_non_incoming_rejected(self, val):
        assert cw._is_incoming(val) is False


# ── conversation id resolution ───────────────────────────────────────────────


class TestConversationId:
    def test_prefers_conversation_id(self):
        assert cw._resolve_conversation_id({"conversation": {"id": 5, "display_id": 9}}) == "5"

    def test_falls_back_to_display_id(self):
        assert cw._resolve_conversation_id({"conversation": {"display_id": 9}}) == "9"

    def test_top_level_fallback(self):
        assert cw._resolve_conversation_id({"conversation_id": 12}) == "12"

    def test_none_when_absent(self):
        assert cw._resolve_conversation_id({}) is None


# ── inbound converter (filter matrix) ────────────────────────────────────────


class TestConvert:
    def test_incoming_string_accepted(self):
        a = _make_adapter()
        ev = a._convert(_msg_created(message_type="incoming"))
        assert ev is not None
        assert ev.text == "hello there"
        assert ev.source.chat_id == "1:42"
        assert ev.source.chat_type == "direct"
        assert ev.source.user_id == "88"

    def test_incoming_int_accepted(self):
        a = _make_adapter()
        assert a._convert(_msg_created(message_type=0)) is not None

    @pytest.mark.parametrize("val", ["outgoing", 1])
    def test_outgoing_ignored(self, val):
        a = _make_adapter()
        assert a._convert(_msg_created(message_type=val)) is None

    def test_private_note_ignored(self):
        a = _make_adapter()
        assert a._convert(_msg_created(private=True)) is None

    def test_status_open_ignored(self):
        a = _make_adapter()
        payload = _msg_created()
        payload["conversation"]["status"] = "open"
        assert a._convert(payload) is None

    def test_status_pending_accepted(self):
        a = _make_adapter()
        payload = _msg_created()
        payload["conversation"]["status"] = "pending"
        assert a._convert(payload) is not None

    def test_wrong_event_ignored(self):
        a = _make_adapter()
        assert a._convert(_msg_created(event="conversation_updated")) is None

    def test_empty_content_no_attachment_ignored(self):
        a = _make_adapter()
        assert a._convert(_msg_created(content="")) is None

    def test_display_id_fallback_used(self):
        a = _make_adapter()
        payload = _msg_created()
        payload["conversation"] = {"display_id": 77, "status": "pending"}
        ev = a._convert(payload)
        assert ev is not None and ev.source.chat_id == "1:77"

    def test_account_from_payload(self):
        a = _make_adapter(account_id="")
        ev = a._convert(_msg_created(account={"id": 3}))
        assert ev is not None and ev.source.chat_id == "3:42"


# ── outbound send ────────────────────────────────────────────────────────────


class TestSend:
    @pytest.mark.asyncio
    async def test_reply_posts_correct_url_body_header(self):
        a = _make_adapter()
        a._session = _FakeSession([_FakeResp(200, {"id": 999})])
        res = await a.send("1:42", "hi")
        assert res.success and res.message_id == "999"
        method, url, kwargs = a._session.calls[0]
        assert method == "POST"
        assert url == "https://cw.example.com/api/v1/accounts/1/conversations/42/messages"
        assert kwargs["json"]["content"] == "hi"
        assert kwargs["json"]["message_type"] == "outgoing"
        assert kwargs["json"]["private"] is False
        assert kwargs["headers"]["api_access_token"] == "bot-tok"

    @pytest.mark.asyncio
    async def test_long_message_chunks(self):
        a = _make_adapter()
        a.MAX_MESSAGE_LENGTH = 10
        a._session = _FakeSession([_FakeResp(200, {"id": i}) for i in range(3)])
        res = await a.send("1:42", "x" * 25)  # 10 + 10 + 5 → 3 chunks
        assert res.success
        assert len(a._session.calls) == 3

    @pytest.mark.asyncio
    async def test_5xx_retryable(self):
        a = _make_adapter()
        a._session = _FakeSession([_FakeResp(503, text_body="boom")])
        res = await a.send("1:42", "hi")
        assert not res.success and res.retryable

    @pytest.mark.asyncio
    async def test_4xx_terminal(self):
        a = _make_adapter()
        a._session = _FakeSession([_FakeResp(400, text_body="bad")])
        res = await a.send("1:42", "hi")
        assert not res.success and not res.retryable

    @pytest.mark.asyncio
    async def test_bad_chat_id_returns_error(self):
        a = _make_adapter(account_id="")
        res = await a.send("nope", "hi")
        assert not res.success


# ── private-note trace ───────────────────────────────────────────────────────


class TestPrivateNoteTrace:
    @pytest.mark.asyncio
    async def test_marked_send_becomes_private_note_with_agent_token(self):
        a = _make_adapter(private_note_trace=True, agent_token="agent-tok")
        a._session = _FakeSession([_FakeResp(200, {"id": 1})])
        await a.send("1:42", "thinking…", metadata={"non_conversational": True})
        _, _, kwargs = a._session.calls[0]
        assert kwargs["json"]["private"] is True
        assert kwargs["headers"]["api_access_token"] == "agent-tok"

    @pytest.mark.asyncio
    async def test_trace_off_never_private(self):
        a = _make_adapter(private_note_trace=False)
        a._session = _FakeSession([_FakeResp(200, {"id": 1})])
        await a.send("1:42", "thinking…", metadata={"non_conversational": True})
        _, _, kwargs = a._session.calls[0]
        assert kwargs["json"]["private"] is False
        assert kwargs["headers"]["api_access_token"] == "bot-tok"

    @pytest.mark.asyncio
    async def test_customer_reply_unaffected_by_trace(self):
        a = _make_adapter(private_note_trace=True, agent_token="agent-tok")
        a._session = _FakeSession([_FakeResp(200, {"id": 1})])
        await a.send("1:42", "final answer")  # no marker
        _, _, kwargs = a._session.calls[0]
        assert kwargs["json"]["private"] is False
        assert kwargs["headers"]["api_access_token"] == "bot-tok"

    @pytest.mark.asyncio
    async def test_missing_agent_token_warns_once(self, caplog):
        a = _make_adapter(private_note_trace=True)  # no agent token
        a._session = _FakeSession([_FakeResp(401, text_body="no"), _FakeResp(401, text_body="no")])
        await a.send("1:42", "trace1", metadata={"non_conversational": True})
        await a.send("1:42", "trace2", metadata={"non_conversational": True})
        assert a._private_note_warned is True
        # Uses bot token as a fallback attempt; 401 is not treated as retryable-crash.
        _, _, kwargs = a._session.calls[0]
        assert kwargs["json"]["private"] is True


# ── webhook handler ──────────────────────────────────────────────────────────


class TestWebhookHandler:
    @pytest.mark.asyncio
    async def test_not_running_404(self):
        a = _make_adapter()
        a._running = False
        resp = await a._handle_webhook(_FakeRequest(b"{}"))
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_oversized_413(self):
        a = _make_adapter()
        a._max_body_bytes = 10
        resp = await a._handle_webhook(_FakeRequest(b"x" * 50, content_length=50))
        assert resp.status == 413

    @pytest.mark.asyncio
    async def test_bad_secret_403(self):
        a = _make_adapter(webhook_secret="s3cret")
        resp = await a._handle_webhook(_FakeRequest(b"{}", token="wrong"))
        assert resp.status == 403

    @pytest.mark.asyncio
    async def test_good_secret_passes(self):
        a = _make_adapter(webhook_secret="s3cret")
        resp = await a._handle_webhook(_FakeRequest(b"{}", token="s3cret"))
        assert resp.status == 200  # empty dict → non-actionable, still acked

    @pytest.mark.asyncio
    async def test_bad_json_400(self):
        a = _make_adapter()
        resp = await a._handle_webhook(_FakeRequest(b"not json"))
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_valid_incoming_dispatched(self, monkeypatch):
        a = _make_adapter()
        dispatched = []

        async def _fake_handle(event):
            dispatched.append(event)

        monkeypatch.setattr(a, "handle_message", _fake_handle)
        body = json.dumps(_msg_created()).encode()
        resp = await a._handle_webhook(_FakeRequest(body))
        assert resp.status == 200
        # allow the fire-and-forget task to run
        import asyncio

        await asyncio.sleep(0)
        assert len(dispatched) == 1

    @pytest.mark.asyncio
    async def test_duplicate_not_redispatched(self, monkeypatch):
        a = _make_adapter()
        dispatched = []

        async def _fake_handle(event):
            dispatched.append(event)

        monkeypatch.setattr(a, "handle_message", _fake_handle)
        body = json.dumps(_msg_created(id=777)).encode()
        r1 = await a._handle_webhook(_FakeRequest(body))
        r2 = await a._handle_webhook(_FakeRequest(body))
        import asyncio

        await asyncio.sleep(0)
        assert r1.status == 200 and r2.status == 200
        assert len(dispatched) == 1  # second is a dedup no-op

    @pytest.mark.asyncio
    async def test_non_actionable_not_dispatched(self, monkeypatch):
        a = _make_adapter()
        dispatched = []

        async def _fake_handle(event):
            dispatched.append(event)

        monkeypatch.setattr(a, "handle_message", _fake_handle)
        body = json.dumps(_msg_created(message_type="outgoing", id=1234)).encode()
        resp = await a._handle_webhook(_FakeRequest(body))
        import asyncio

        await asyncio.sleep(0)
        assert resp.status == 200 and len(dispatched) == 0


# ── attachments ──────────────────────────────────────────────────────────────


class TestAttachments:
    def test_inbound_image_cached(self, monkeypatch):
        a = _make_adapter()
        monkeypatch.setattr(a, "_download_and_cache", lambda url, ft: ("/tmp/x.jpg", "image/jpeg"))
        payload = _msg_created(attachments=[{"file_type": "image", "data_url": "https://x/p.png"}])
        ev = a._convert(payload)
        assert ev is not None
        assert ev.media_urls == ["/tmp/x.jpg"]
        assert ev.media_types == ["image/jpeg"]

    def test_inbound_attachment_failure_still_yields_text(self, monkeypatch):
        a = _make_adapter()

        def _boom(url, ft):
            raise RuntimeError("network")

        monkeypatch.setattr(a, "_download_and_cache", _boom)
        payload = _msg_created(attachments=[{"file_type": "image", "data_url": "https://x/p.png"}])
        ev = a._convert(payload)
        assert ev is not None and ev.text == "hello there" and ev.media_urls == []

    @pytest.mark.asyncio
    async def test_outbound_multipart_builds_attachments_field(self, tmp_path):
        a = _make_adapter()
        a._session = _FakeSession([_FakeResp(200, {"id": 1})])
        f = tmp_path / "pic.jpg"
        f.write_bytes(b"jpegbytes")
        res = await a.send_image("1:42", str(f), caption="look")
        assert res.success
        _, url, kwargs = a._session.calls[0]
        assert url.endswith("/conversations/42/messages")
        assert "data" in kwargs  # FormData multipart, not json
        assert kwargs["headers"]["api_access_token"] == "bot-tok"


# ── standalone send + cron wiring ────────────────────────────────────────────


class TestStandaloneAndCron:
    @pytest.mark.asyncio
    async def test_standalone_send_posts_and_succeeds(self, monkeypatch):
        cfg = PlatformConfig(enabled=True, token="bot-tok", extra={"base_url": "https://cw.example.com", "account_id": "1"})
        captured = {}

        class _Sess:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return None

            def post(self, url, **kwargs):
                captured["url"] = url
                captured["kwargs"] = kwargs
                return _FakeResp(200, {"id": 1})

        monkeypatch.setattr(cw.aiohttp, "ClientSession", lambda *a, **k: _Sess())
        res = await cw._standalone_send(cfg, "1:42", "cron hi")
        assert res == {"success": True}
        assert captured["url"].endswith("/conversations/42/messages")
        assert captured["kwargs"]["headers"]["api_access_token"] == "bot-tok"

    @pytest.mark.asyncio
    async def test_standalone_send_error(self, monkeypatch):
        cfg = PlatformConfig(enabled=True, token="bot-tok", extra={"base_url": "https://cw.example.com"})

        class _Sess:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return None

            def post(self, url, **kwargs):
                return _FakeResp(404, text_body="nope")

        monkeypatch.setattr(cw.aiohttp, "ClientSession", lambda *a, **k: _Sess())
        res = await cw._standalone_send(cfg, "1:42", "cron hi")
        assert "error" in res

    @pytest.mark.asyncio
    async def test_standalone_send_unconfigured(self):
        cfg = PlatformConfig(enabled=True, extra={})
        res = await cw._standalone_send(cfg, "1:42", "hi")
        assert "error" in res


# ── registration / authorization wiring ──────────────────────────────────────


class _MockCtx:
    class _M:
        name = "chatwoot-platform"

    manifest = _M()
    _manager = type("_Mgr", (), {"_plugin_platform_names": set()})()

    def register_platform(self, **kwargs):
        from gateway.platform_registry import PlatformEntry

        self.entry = PlatformEntry(source="plugin", **kwargs)


class TestRegistration:
    def test_register_builds_entry_with_expected_hooks(self):
        ctx = _MockCtx()
        cw.register(ctx)
        e = ctx.entry
        assert e.name == "chatwoot"
        assert e.allowed_users_env == "CHATWOOT_ALLOWED_USERS"
        assert e.allow_all_env == "CHATWOOT_ALLOW_ALL_USERS"
        assert e.cron_deliver_env_var == "CHATWOOT_HOME_CHANNEL"
        assert e.standalone_sender_fn is cw._standalone_send
        assert e.platform_hint and "Chatwoot" in e.platform_hint
        assert e.max_message_length == cw.MAX_MESSAGE_LENGTH

    def test_registered_in_registry_and_cron_wiring(self):
        from gateway.platform_registry import platform_registry, PlatformEntry

        # Register into the real registry so cron helpers can resolve it.
        class _RegCtx(_MockCtx):
            def register_platform(self, **kwargs):
                platform_registry.register(PlatformEntry(source="plugin", **kwargs))

        cw.register(_RegCtx())
        assert platform_registry.is_registered("chatwoot")

        from cron.scheduler import _is_known_delivery_platform, _resolve_home_env_var

        assert _is_known_delivery_platform("chatwoot") is True
        assert _resolve_home_env_var("chatwoot") == "CHATWOOT_HOME_CHANNEL"


# ── core marker edit (Discord-safe regression) ───────────────────────────────


class TestNonConversationalMarker:
    def test_chatwoot_marked(self):
        from gateway.run import _non_conversational_metadata

        out = _non_conversational_metadata({"x": 1}, platform=Platform("chatwoot"))
        assert out["non_conversational"] is True and out["x"] == 1

    def test_discord_unchanged(self):
        from gateway.run import _non_conversational_metadata

        out = _non_conversational_metadata({}, platform=Platform("discord"))
        assert out["non_conversational"] is True

    def test_other_platform_passthrough(self):
        from gateway.run import _non_conversational_metadata

        meta = {"k": "v"}
        out = _non_conversational_metadata(meta, platform=Platform("telegram"))
        assert out == meta and "non_conversational" not in out


# ── redaction ────────────────────────────────────────────────────────────────


class TestRedaction:
    def test_masks_token(self):
        assert cw._redact("supersecrettoken") == "***oken"

    def test_masks_short_and_none(self):
        assert cw._redact("abc") == "***"
        assert cw._redact(None) == "<none>"
