"""Tests for the LINE platform adapter plugin.

Covers LINE adapter behavior from the PR review:

1. webhook signature verification (HMAC-SHA256, base64) + tampering rejection
2. inbound chat-id resolution for user / group / room sources
3. three-allowlist gating (users / groups / rooms / allow_all)
4. inbound dedup via webhookEventId
5. RequestCache state machine (PENDING → READY → DELIVERED, ERROR)
6. Markdown stripping with URL preservation + LINE-sized chunking
7. inbound media normalization to gateway message types and MIME metadata
8. send routing: reply token preferred → push fallback → batched at 5/call
9. register() metadata + standalone_send shape
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

# Load plugins/platforms/line/adapter.py under plugin_adapter_line so it
# cannot collide with sibling platform-plugin tests in the same process.
_line = load_plugin_adapter("line")

verify_line_signature = _line.verify_line_signature
strip_markdown_preserving_urls = _line.strip_markdown_preserving_urls
split_for_line = _line.split_for_line
build_postback_button_message = _line.build_postback_button_message
_resolve_chat = _line._resolve_chat
_allowed_for_source = _line._allowed_for_source
RequestCache = _line.RequestCache
State = _line.State
LineAdapter = _line.LineAdapter
register = _line.register
check_requirements = _line.check_requirements
validate_config = _line.validate_config
_standalone_send = _line._standalone_send
_env_enablement = _line._env_enablement


# ---------------------------------------------------------------------------
# 1. Signature verification
# ---------------------------------------------------------------------------

class TestSignature:

    def _sign(self, body: bytes, secret: str) -> str:
        digest = hmac.new(secret.encode(), body, hashlib.sha256).digest()
        return base64.b64encode(digest).decode()


    def test_wrong_secret_rejected(self):
        body = b'{"events": []}'
        sig = self._sign(body, "secret")
        assert not verify_line_signature(body, sig, "different")


    def test_empty_secret_rejected(self):
        assert not verify_line_signature(b"x", "AAAA", "")


# ---------------------------------------------------------------------------
# 2. Chat-id / source resolution
# ---------------------------------------------------------------------------

class TestSourceResolution:

    def test_user_source(self):
        chat_id, ctype = _resolve_chat({"type": "user", "userId": "U123"})
        assert chat_id == "U123"
        assert ctype == "dm"

    def test_group_source(self):
        chat_id, ctype = _resolve_chat({"type": "group", "groupId": "C456", "userId": "U123"})
        assert chat_id == "C456"
        assert ctype == "group"


# ---------------------------------------------------------------------------
# 3. Three-allowlist gating
# ---------------------------------------------------------------------------

class TestAllowlist:

    def test_allow_all_short_circuits(self):
        for src in [
            {"type": "user", "userId": "Ufoo"},
            {"type": "group", "groupId": "Cfoo"},
            {"type": "room", "roomId": "Rfoo"},
        ]:
            assert _allowed_for_source(src, allow_all=True, user_ids=set(), group_ids=set(), room_ids=set())

    def test_user_in_allowlist_passes(self):
        src = {"type": "user", "userId": "Uok"}
        assert _allowed_for_source(src, allow_all=False, user_ids={"Uok"}, group_ids=set(), room_ids=set())


# ---------------------------------------------------------------------------
# 4. Inbound dedup
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# 5. RequestCache state machine
# ---------------------------------------------------------------------------

class TestRequestCache:


    def test_mark_delivered_from_ready(self):
        c = RequestCache()
        rid = c.register_pending("Uchat")
        c.set_ready(rid, "x")
        c.mark_delivered(rid)
        assert c.get(rid).state is State.DELIVERED


    def test_set_ready_on_delivered_is_noop(self):
        c = RequestCache()
        rid = c.register_pending("Uchat")
        c.set_ready(rid, "first")
        c.mark_delivered(rid)
        c.set_ready(rid, "second")
        # DELIVERED is terminal — no further mutation
        assert c.get(rid).payload == "first"
        assert c.get(rid).state is State.DELIVERED


# ---------------------------------------------------------------------------
# 6. Markdown stripping + chunking
# ---------------------------------------------------------------------------

class TestMarkdownAndChunking:

    def test_bold_stripped(self):
        assert strip_markdown_preserving_urls("**hello**") == "hello"

    def test_italic_stripped(self):
        assert strip_markdown_preserving_urls("*hello*") == "hello"


    def test_split_long_chunks_at_paragraph_boundary(self):
        text = "para1\n\npara2\n\npara3"
        chunks = split_for_line(text, max_chars=8)
        assert all(len(c) <= 8 for c in chunks), chunks
        assert len(chunks) >= 2

    def test_split_caps_at_five_chunks(self):
        # 1000 paragraphs of 100 chars each — must cap at 5 LINE bubbles.
        text = "\n\n".join(["x" * 100 for _ in range(1000)])
        chunks = split_for_line(text)
        assert len(chunks) <= 5


# ---------------------------------------------------------------------------
# 7. Inbound media normalization
# ---------------------------------------------------------------------------

class TestInboundMedia:

    @pytest.fixture
    def adapter(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        from gateway.config import PlatformConfig

        cfg = PlatformConfig(enabled=True, extra={
            "channel_access_token": "tok",
            "channel_secret": "sec",
        })
        ad = LineAdapter(cfg)
        ad._client = MagicMock()
        ad._client.fetch_content = AsyncMock(return_value=b"line-bytes")
        ad.handle_message = AsyncMock()
        return ad

    def _event(self, msg_type, **message):
        payload = {"type": msg_type, "id": f"{msg_type}-1"}
        payload.update(message)
        return {
            "type": "message",
            "replyToken": "reply-token",
            "source": {"type": "group", "groupId": "Cline", "userId": "Uline"},
            "message": payload,
        }

    def _captured_event(self, adapter):
        adapter.handle_message.assert_awaited_once()
        return adapter.handle_message.await_args.args[0]

    def test_image_message_uses_photo_type_and_image_mime(self, adapter):
        with patch.object(
            _line,
            "cache_image_from_bytes_async",
            new=AsyncMock(return_value="/cache/image.jpg"),
        ) as cache:
            asyncio.run(adapter._handle_message_event(self._event("image")))

        cache.assert_awaited_once_with(b"line-bytes", ext=".jpg")
        event = self._captured_event(adapter)
        assert event.message_type is _line.MessageType.PHOTO
        assert event.media_urls == ["/cache/image.jpg"]
        assert event.media_types == ["image/jpeg"]


# ---------------------------------------------------------------------------
# 8. Send routing (reply -> push fallback, batching, system-bypass)
# ---------------------------------------------------------------------------

class TestSendRouting:

    @pytest.fixture
    def adapter(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True, extra={
            "channel_access_token": "tok",
            "channel_secret": "sec",
        })
        ad = LineAdapter(cfg)
        ad._client = MagicMock()
        ad._client.reply = AsyncMock()
        ad._client.push = AsyncMock()
        return ad



    def test_send_caps_messages_per_call_at_five(self, adapter):
        # Build a payload that would naturally split into more than 5 LINE
        # bubbles; the chunker should cap at 5 + truncate.
        big = "\n\n".join(["x" * 4500 for _ in range(20)])
        result = asyncio.run(adapter.send("Uchat", big))
        assert result.success
        call_kwargs = adapter._client.push.call_args
        # call_args is (args, kwargs); for our send the messages are the 2nd positional
        sent_messages = call_kwargs.args[1] if call_kwargs.args else call_kwargs.kwargs.get("messages")
        # Without args, fall back to inspecting the call shape
        if sent_messages is None:
            # We invoked client.push(chat_id, messages) — check first batch
            sent_messages = adapter._client.push.call_args.args[1]
        assert len(sent_messages) <= 5

    def test_format_message_strips_markdown(self, adapter):
        out = adapter.format_message("**bold** [link](https://x.com)")
        assert "**" not in out
        assert "https://x.com" in out


# ---------------------------------------------------------------------------
# 9. Register() metadata + plugin entry points
# ---------------------------------------------------------------------------

class TestRegister:

    class _FakeCtx:
        def __init__(self):
            self.kwargs = None

        def register_platform(self, **kw):
            self.kwargs = kw





    def test_max_message_length_below_line_per_bubble_limit(self):
        ctx = self._FakeCtx()
        register(ctx)
        # LINE per-bubble limit is 5000; we register 4500 to leave headroom.
        assert ctx.kwargs["max_message_length"] <= 5000


class TestSlowLLMPostbackRegression:
    """#106446 — the two reporter reproductions, driven through the real ``send()`` /
    ``_handle_postback_event()`` paths with only the HTTP client mocked."""

    @pytest.fixture
    def adapter(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True, extra={"channel_access_token": "tok", "channel_secret": "sec"})
        ad = LineAdapter(cfg)
        ad._client = MagicMock()
        ad._client.reply = AsyncMock()
        ad._client.push = AsyncMock()
        return ad

    def test_repro_a_heartbeat_does_not_become_the_cached_answer(self, adapter):
        # The gateway's periodic heartbeat (stamped ``_interim_send`` by ``_interim_metadata``)
        # arrives while the postback button is PENDING; the real answer follows.
        rid = adapter._cache.register_pending("Uchat")
        adapter._pending_buttons["Uchat"] = rid
        asyncio.run(adapter.send("Uchat", "⏳ Working — 3 min — iteration 1/60, research_market",
                                 metadata={"_interim_send": True}))
        assert adapter._cache.get(rid).state is State.PENDING, "heartbeat must not fill the cache"
        assert adapter._client.push.await_count == 1, "heartbeat lands as a visible bubble"
        asyncio.run(adapter.send("Uchat", "Final answer: task completed"))
        assert adapter._cache.get(rid).payload == "Final answer: task completed"

    def test_repro_b_stale_mapping_does_not_swallow_the_next_answer(self, adapter):
        # A READY entry still mapped to the chat, then a tap on a button whose entry is gone:
        # neither may absorb a later send or leave the chat stuck.
        rid = adapter._cache.register_pending("Uchat")
        adapter._cache.set_ready(rid, "Previous answer")
        adapter._pending_buttons["Uchat"] = rid
        result = asyncio.run(adapter.send("Uchat", "Answer to the NEXT user message"))
        assert result.success and adapter._client.push.await_count == 1, "next answer reaches the wire"
        assert adapter._cache.get(rid).payload == "Previous answer"
        assert "Uchat" not in adapter._pending_buttons
        # Expired button tap: user gets a notice and the dead mapping is dropped.
        adapter._pending_buttons["Uchat"] = "ghost-rid"
        asyncio.run(adapter._handle_postback_event({
            "replyToken": "reply-token", "source": {"type": "user", "userId": "Uchat"},
            "postback": {"data": json.dumps({"action": "show_response", "request_id": "ghost-rid"})}}))
        adapter._client.reply.assert_awaited_once()
        assert adapter._client.reply.await_args.args[1][0]["text"] == adapter.expired_text
        assert "Uchat" not in adapter._pending_buttons

    @staticmethod
    def _tap(adapter, chat_id, rid):
        asyncio.run(adapter._handle_postback_event({
            "replyToken": f"reply-{chat_id}", "source": {"type": "user", "userId": chat_id},
            "postback": {"data": json.dumps({"action": "show_response", "request_id": rid})}}))

    @staticmethod
    def _entry_in(adapter, state):
        rid = adapter._cache.register_pending("Uchat-a")
        if state is State.ERROR:
            adapter._cache.set_error(rid, "private error for A")
        elif state is not State.PENDING:
            adapter._cache.set_ready(rid, "private answer for A")
            if state is State.DELIVERED:
                adapter._cache.mark_delivered(rid)
        return rid

    @pytest.mark.parametrize("state", list(State))
    def test_request_id_from_another_chat_is_treated_as_unknown(self, adapter, state):
        # #38628: a request id is a routing handle — chat B replaying chat A's id
        # must not receive A's cached output nor settle A's entry.
        rid = self._entry_in(adapter, state)
        adapter._pending_buttons["Uchat-a"] = rid
        self._tap(adapter, "Uchat-b", rid)
        adapter._client.push.assert_not_awaited()
        adapter._client.reply.assert_awaited_once()
        assert adapter._client.reply.await_args.args == ("reply-Uchat-b", [{"type": "text", "text": adapter.expired_text}])
        assert adapter._cache.get(rid).state is state
        assert adapter._pending_buttons == {"Uchat-a": rid}

    def test_originating_chat_still_receives_its_cached_answer(self, adapter):
        rid = self._entry_in(adapter, State.READY)
        adapter._pending_buttons["Uchat-a"] = rid
        self._tap(adapter, "Uchat-a", rid)
        assert adapter._client.reply.await_args.args[0] == "reply-Uchat-a"
        assert adapter._client.reply.await_args.args[1][0]["text"] == "private answer for A"
        assert adapter._cache.get(rid).state is State.DELIVERED
        assert "Uchat-a" not in adapter._pending_buttons


class TestEnvEnablement:

    def test_returns_none_without_credentials(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        assert _env_enablement() is None


class TestStandaloneSend:

    def test_missing_token_returns_error(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        from gateway.config import PlatformConfig
        for extra in ({}, {"channel_access_token": "   "}):  # a blank token never reaches the push API
            result = asyncio.run(_standalone_send(PlatformConfig(enabled=True, extra=extra), "Uchat", "hi"))
            assert "missing token" in result["error"]


class TestPostbackButtonShape:

    def test_template_buttons_structure(self):
        msg = build_postback_button_message("hi", "Tap me", "rid-1")
        assert msg["type"] == "template"
        assert msg["template"]["type"] == "buttons"
        assert msg["template"]["text"] == "hi"
        actions = msg["template"]["actions"]
        assert len(actions) == 1
        assert actions[0]["type"] == "postback"
        data = json.loads(actions[0]["data"])
        assert data == {"action": "show_response", "request_id": "rid-1"}


class TestCheckRequirements:


    def test_rejects_without_secret(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "t")
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        assert not check_requirements()
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "   ")
        assert not check_requirements()


class TestValidateConfig:

    def test_validates_from_extra(self):
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(
            enabled=True,
            extra={"channel_access_token": "t", "channel_secret": "s"},
        )
        assert validate_config(cfg)


class TestAdapterInit:

    @pytest.mark.asyncio
    async def test_connect_fails_when_channel_lock_held(self, monkeypatch):
        """``acquire_scoped_lock`` returns ``(acquired, existing)``; a live foreign holder must stop
        connect() before the LINE client is built (the tuple is truthy, so a bare ``if not`` never fired)."""
        import gateway.status as gateway_status
        from gateway.config import PlatformConfig

        monkeypatch.setattr(
            gateway_status, "acquire_scoped_lock",
            lambda scope, identity, metadata=None: (False, {"pid": 4242, "profile": "other"}))
        ad = LineAdapter(PlatformConfig(enabled=True, extra={"channel_access_token": "tok", "channel_secret": "sec"}))
        assert await ad.connect() is False
        assert ad._fatal_error_code == "line_lock"
        assert "other" in ad._fatal_error_message
        assert ad._client is None

    def test_init_from_config_extra(self, monkeypatch):
        for k in ("LINE_CHANNEL_ACCESS_TOKEN", "LINE_CHANNEL_SECRET", "LINE_PORT"):
            monkeypatch.delenv(k, raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(
            enabled=True,
            extra={
                "channel_access_token": "tok",
                "channel_secret": "sec",
                "port": 7777,
                "public_url": "https://x.example.com",
                "allowed_users": ["U1", "U2"],
            },
        )
        ad = LineAdapter(cfg)
        assert ad.channel_access_token == "tok"
        assert ad.channel_secret == "sec"
        # A whitespace-only secret reads as unset, so connect fails closed instead of keying HMAC with blanks.
        # So do null and other non-strings, which must not become guessable keys like "None" or "True".
        for blank in ("   ", None, True, 0):
            assert LineAdapter(PlatformConfig(enabled=True, extra={"channel_access_token": "tok",
                                                                   "channel_secret": blank})).channel_secret == ""
            assert LineAdapter(PlatformConfig(enabled=True, extra={"channel_access_token": blank,
                                                                   "channel_secret": "sec"})).channel_access_token == ""
        assert ad.webhook_port == 7777
        assert ad.public_base_url == "https://x.example.com"
        assert ad.allowed_users == {"U1", "U2"}


# ---------------------------------------------------------------------------
# 9. Inbound message-type classification
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# 10. Dual-stack bind default (NS-603)
# ---------------------------------------------------------------------------

class TestDualStackBind:
    """The LINE webhook server's default bind must serve BOTH IPv4 and IPv6.

    Regression guard for the hosted LINE 502 (NS-603): Fly.io 6PN — the
    private network the edge router reverse-proxies LINE ingest over — is
    IPv6-only (``<app>.internal`` resolves to an ``fdaa:…`` address). The
    adapter used to default to ``host="0.0.0.0"`` (IPv4 only), so the
    router's dial to ``<app>.internal:8646`` hit an address nothing was
    listening on → connection refused → 502 on webhook verification.

    Mirrors gateway/platforms/webhook.py's fix (commit d542894ad):
    ``DEFAULT_HOST = None`` → asyncio binds one socket per address family.
    ``"::"`` is NOT a valid substitute (bindv6only=1 on Fly machines makes
    it IPv6-only, breaking IPv4 loopback health probes).
    """

    def _cfg(self, **extra):
        from gateway.config import PlatformConfig
        base = {"channel_access_token": "tok", "channel_secret": "sec"}
        base.update(extra)
        return PlatformConfig(enabled=True, extra=base)


    def test_empty_host_normalises_to_none(self, monkeypatch):
        monkeypatch.delenv("LINE_HOST", raising=False)
        ad = LineAdapter(self._cfg(host=""))
        assert ad.webhook_host is None


    @pytest.mark.asyncio
    async def test_default_bind_serves_both_families(self, monkeypatch):
        """Behavioural proof: host=None opens v4 AND v6 listening sockets."""
        # Guard: on IPv4-only hosts (CI runners with IPv6 disabled) the
        # dual-stack bind legitimately yields no v6 socket — that's the
        # environment, not a regression. Probe an actual ::1 bind rather
        # than trusting socket.has_ipv6 (compile-time constant).
        import socket as _socket
        if not _socket.has_ipv6:
            pytest.skip("IPv6 not supported by this Python build")
        try:
            _probe = _socket.socket(_socket.AF_INET6, _socket.SOCK_STREAM)
            try:
                _probe.bind(("::1", 0))
            finally:
                _probe.close()
        except OSError:
            pytest.skip("IPv6 stack unavailable on this host (cannot bind ::1)")
        monkeypatch.delenv("LINE_HOST", raising=False)
        ad = LineAdapter(self._cfg(port=0))
        ad._client = MagicMock()
        ad._client.get_bot_user_id = AsyncMock(return_value="Ubot")

        # Skip credential/network preamble — drive the aiohttp bind directly
        # the same way connect() does.
        from aiohttp import web
        ad._app = web.Application()
        ad._runner = web.AppRunner(ad._app)
        await ad._runner.setup()
        site = web.TCPSite(ad._runner, ad.webhook_host, 0)
        try:
            await site.start()
            addrs = list(ad._runner.addresses)
            has_v6 = any(len(a) == 4 for a in addrs)
            has_v4 = any(len(a) == 2 for a in addrs)
            assert has_v4, f"IPv4 bind missing — got {addrs}"
            assert has_v6, (
                f"IPv6 bind missing (the 6PN reachability bug, NS-603) — got {addrs}"
            )
        finally:
            await ad._runner.cleanup()


class TestMediaPublicUrlGuard:
    """Outbound media requires LINE_PUBLIC_URL whenever the bind host is not
    a publicly fetchable address — including the new dual-stack ``None``
    default and the legacy wildcard strings."""

    def _adapter(self, monkeypatch, **extra):
        from gateway.config import PlatformConfig
        monkeypatch.delenv("LINE_HOST", raising=False)
        monkeypatch.delenv("LINE_PUBLIC_URL", raising=False)
        base = {"channel_access_token": "tok", "channel_secret": "sec"}
        base.update(extra)
        return LineAdapter(PlatformConfig(enabled=True, extra=base))




    def test_send_image_blocked_without_public_url(self, monkeypatch, tmp_path):
        ad = self._adapter(monkeypatch)
        ad._client = MagicMock()
        img = tmp_path / "x.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n123")
        result = asyncio.run(ad.send_image_file("Uchat", str(img)))
        assert not result.success
        assert "LINE_PUBLIC_URL" in (result.error or "")


# ---------------------------------------------------------------------------
# Gateway second gate — LINE group/room chat allowlists
# ---------------------------------------------------------------------------

_LINE_AUTH_ENVS = (
    "LINE_ALLOW_ALL_USERS",
    "LINE_ALLOWED_USERS",
    "LINE_ALLOWED_GROUPS",
    "LINE_ALLOWED_ROOMS",
    "GATEWAY_ALLOW_ALL_USERS",
    "GATEWAY_ALLOWED_USERS",
)


class TestLineGatewayChatAllowlist:
    """The adapter admits a group/room listed in ``LINE_ALLOWED_GROUPS`` /
    ``LINE_ALLOWED_ROOMS`` even when the sender is absent from
    ``LINE_ALLOWED_USERS``. The gateway then re-checks authorization and, without
    those chat lists declared, default-denies the same sender
    (``Unauthorized user: … on line``).

    ``group_allowed_chats_env`` covers ``chat_type == "group"`` only.
    ``"room"`` is not a group/forum type and lives in a different env var, so
    rooms use ``room_allowed_chats_env``. DMs stay on ``LINE_ALLOWED_USERS``.
    """

    @pytest.fixture
    def line_auth(self, monkeypatch):
        for key in _LINE_AUTH_ENVS:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("LINE_ALLOW_ALL_USERS", "false")
        from gateway.platform_registry import platform_registry
        from hermes_cli.plugins import discover_plugins

        discover_plugins()
        entry = platform_registry.get("line")
        assert entry is not None, "LINE platform plugin did not register"
        assert entry.allowed_users_env == "LINE_ALLOWED_USERS"
        assert entry.allow_all_env == "LINE_ALLOW_ALL_USERS"
        assert entry.group_allowed_chats_env == "LINE_ALLOWED_GROUPS"
        assert entry.room_allowed_chats_env == "LINE_ALLOWED_ROOMS"
        return entry

    @staticmethod
    def _runner():
        from types import SimpleNamespace

        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
        return runner

    @staticmethod
    def _source(chat_type: str, chat_id: str, user_id: str):
        from gateway.session import Platform, SessionSource

        return SessionSource(
            platform=Platform("line"),
            chat_id=chat_id,
            chat_type=chat_type,
            user_id=user_id,
            user_name=user_id,
        )

    def test_group_member_outside_dm_allowlist_is_authorized(self, line_auth, monkeypatch):
        monkeypatch.setenv("LINE_ALLOWED_USERS", "Uowner")
        monkeypatch.setenv("LINE_ALLOWED_GROUPS", "Callowed")
        source = self._source("group", "Callowed", "Umember")
        assert self._runner()._is_user_authorized(source) is True

    def test_unlisted_group_is_rejected(self, line_auth, monkeypatch):
        monkeypatch.setenv("LINE_ALLOWED_USERS", "Uowner")
        monkeypatch.setenv("LINE_ALLOWED_GROUPS", "Callowed")
        monkeypatch.setenv("LINE_ALLOWED_ROOMS", "Rallowed")
        source = self._source("group", "Cother", "Umember")
        assert self._runner()._is_user_authorized(source) is False

    def test_room_member_outside_dm_allowlist_is_authorized(self, line_auth, monkeypatch):
        monkeypatch.setenv("LINE_ALLOWED_USERS", "Uowner")
        monkeypatch.setenv("LINE_ALLOWED_ROOMS", "Rallowed")
        source = self._source("room", "Rallowed", "Umember")
        assert self._runner()._is_user_authorized(source) is True

    def test_unlisted_room_is_rejected(self, line_auth, monkeypatch):
        monkeypatch.setenv("LINE_ALLOWED_USERS", "Uowner")
        monkeypatch.setenv("LINE_ALLOWED_GROUPS", "Callowed")
        monkeypatch.setenv("LINE_ALLOWED_ROOMS", "Rallowed")
        source = self._source("room", "Rother", "Umember")
        assert self._runner()._is_user_authorized(source) is False

    def test_group_list_does_not_authorize_a_room(self, line_auth, monkeypatch):
        monkeypatch.setenv("LINE_ALLOWED_GROUPS", "Rallowed")
        source = self._source("room", "Rallowed", "Umember")
        assert self._runner()._is_user_authorized(source) is False

    def test_room_list_does_not_authorize_a_group(self, line_auth, monkeypatch):
        monkeypatch.setenv("LINE_ALLOWED_ROOMS", "Callowed")
        source = self._source("group", "Callowed", "Umember")
        assert self._runner()._is_user_authorized(source) is False

    def test_dm_requires_user_allowlist(self, line_auth, monkeypatch):
        monkeypatch.setenv("LINE_ALLOWED_USERS", "Uowner")
        monkeypatch.setenv("LINE_ALLOWED_GROUPS", "Callowed")
        monkeypatch.setenv("LINE_ALLOWED_ROOMS", "Rallowed")
        runner = self._runner()
        assert runner._is_user_authorized(self._source("dm", "Uowner", "Uowner")) is True
        assert runner._is_user_authorized(self._source("dm", "Umember", "Umember")) is False

    def test_chat_allowlists_do_not_suppress_dm_pairing(self, line_auth, monkeypatch):
        from gateway.session import Platform

        monkeypatch.setenv("LINE_ALLOWED_GROUPS", "Callowed")
        monkeypatch.setenv("LINE_ALLOWED_ROOMS", "Rallowed")
        assert self._runner()._get_unauthorized_dm_behavior(Platform("line")) == "pair"


class TestLineDispatchGatewayAuth:
    """``_dispatch_event`` builds the ``MessageEvent`` the gateway re-checks.

    ``allow_adapter_delegation=False`` is the native re-check: a role flag does
    not count. A listed group or room must still be authorized by its chat id,
    and an unlisted group or DM never becomes that event.
    """

    @staticmethod
    def _register_line():
        from gateway.platform_registry import platform_registry
        from hermes_cli.plugins import discover_plugins

        discover_plugins()
        entry = platform_registry.get("line")
        assert entry is not None
        assert entry.group_allowed_chats_env == "LINE_ALLOWED_GROUPS"
        assert entry.room_allowed_chats_env == "LINE_ALLOWED_ROOMS"
        return entry

    @staticmethod
    def _adapter(monkeypatch, **env):
        for key in _LINE_AUTH_ENVS:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("LINE_ALLOW_ALL_USERS", "false")
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        from gateway.config import PlatformConfig

        return LineAdapter(PlatformConfig(
            enabled=True,
            extra={"channel_access_token": "tok", "channel_secret": "sec"},
        ))

    @staticmethod
    def _dispatch(adapter, source: dict, webhook_id: str):
        captured = {}

        async def _capture(event):
            captured["event"] = event

        adapter.handle_message = _capture
        asyncio.run(adapter._dispatch_event({
            "type": "message",
            "source": source,
            "message": {"type": "text", "id": "m1", "text": "hi"},
            "replyToken": "rt",
            "webhookEventId": webhook_id,
        }))
        return captured.get("event")

    @staticmethod
    def _strictly_authorized(event) -> bool:
        from types import SimpleNamespace

        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
        return runner._is_user_authorized(event.source, allow_adapter_delegation=False) is True

    def test_allowlisted_group_sender_is_authorized_without_delegation(self, monkeypatch):
        self._register_line()
        adapter = self._adapter(
            monkeypatch, LINE_ALLOWED_USERS="Uowner", LINE_ALLOWED_GROUPS="Callowed")
        event = self._dispatch(
            adapter, {"type": "group", "groupId": "Callowed", "userId": "Umember"}, "group-ok")
        assert event is not None
        assert event.source.chat_type == "group"
        assert event.source.chat_id == "Callowed"
        assert event.source.user_id == "Umember"
        assert event.source.role_authorized is False
        assert self._strictly_authorized(event) is True

    def test_allowlisted_room_sender_is_authorized_without_delegation(self, monkeypatch):
        self._register_line()
        adapter = self._adapter(
            monkeypatch, LINE_ALLOWED_USERS="Uowner", LINE_ALLOWED_ROOMS="Rallowed")
        event = self._dispatch(
            adapter, {"type": "room", "roomId": "Rallowed", "userId": "Umember"}, "room-ok")
        assert event is not None
        assert event.source.chat_type == "room"
        assert event.source.chat_id == "Rallowed"
        assert event.source.user_id == "Umember"
        assert event.source.role_authorized is False
        assert self._strictly_authorized(event) is True

    def test_unlisted_group_and_dm_sender_are_rejected(self, monkeypatch):
        self._register_line()
        adapter = self._adapter(
            monkeypatch,
            LINE_ALLOWED_USERS="Uowner",
            LINE_ALLOWED_GROUPS="Callowed",
            LINE_ALLOWED_ROOMS="Rallowed",
        )
        # The adapter drops these before handle_message, so no MessageEvent
        # reaches the gateway's strict re-check.
        assert self._dispatch(
            adapter, {"type": "group", "groupId": "Cother", "userId": "Umember"}, "group-no") is None
        assert self._dispatch(
            adapter, {"type": "user", "userId": "Umember"}, "dm-no") is None

