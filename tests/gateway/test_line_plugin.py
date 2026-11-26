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


_LINE_AUTH_ENVS = (
    "LINE_ALLOW_ALL_USERS",
    "LINE_ALLOWED_USERS",
    "LINE_ALLOWED_GROUPS",
    "LINE_ALLOWED_ROOMS",
    "LINE_REQUIRE_MENTION_CHATS",
    "LINE_REPLY_WITHOUT_MENTION_MEDIA_TYPES",
    "LINE_IGNORE_GROUP_MESSAGE_TYPES",
    "LINE_OBSERVED_UNMENTIONED_MESSAGES",
    "GATEWAY_ALLOW_ALL_USERS",
    "GATEWAY_ALLOWED_USERS",
)

# ---------------------------------------------------------------------------
# Group/room @mention gate and message-type filters
# ---------------------------------------------------------------------------

def _group(chat_id="Cneed", user_id="Umember"):
    return {"type": "group", "groupId": chat_id, "userId": user_id}


def _room(chat_id="Rneed", user_id="Umember"):
    return {"type": "room", "roomId": chat_id, "userId": user_id}


def _dm(user_id="Udm"):
    return {"type": "user", "userId": user_id}


def _text(body, mentionees=None):
    message = {"type": "text", "id": "m1", "text": body}
    if mentionees is not None:
        message["mention"] = {"mentionees": mentionees}
    return message


def _bot_mention(index, length, *, is_self=True, user_id="Ubot"):
    mentionee = {"index": index, "length": length, "type": "user", "userId": user_id}
    if is_self is not None:
        mentionee["isSelf"] = is_self
    return mentionee


def _emoji(body="(laugh)", spans=None):
    if spans is None:
        spans = [(0, len(body))]
    return {
        "type": "text",
        "id": "e1",
        "text": body,
        "emojis": [
            {"index": start, "length": length, "productId": "p", "emojiId": "001"}
            for start, length in spans
        ],
    }


class _LineGroupHarness:
    def _adapter(self, monkeypatch, **env):
        for key in _LINE_AUTH_ENVS:
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        from gateway.config import PlatformConfig

        adapter = LineAdapter(PlatformConfig(
            enabled=True,
            extra={"channel_access_token": "tok", "channel_secret": "sec"},
        ))
        adapter._bot_user_id = "Ubot"
        adapter._bot_display_name = "Hermes"
        adapter._download_media = AsyncMock(return_value=("/cache/line.bin", "application/octet-stream"))
        return adapter

    def _gated(self, monkeypatch, **extra):
        env = {
            "LINE_ALLOW_ALL_USERS": "false",
            "LINE_ALLOWED_USERS": "Uowner,Udm",
            "LINE_ALLOWED_GROUPS": "Cneed,Cfree",
            "LINE_ALLOWED_ROOMS": "Rneed,Rfree",
            "LINE_REQUIRE_MENTION_CHATS": "Cneed,Rneed",
        }
        env.update(extra)
        return self._adapter(monkeypatch, **env)

    @staticmethod
    def _dispatch(adapter, source, message, webhook_id):
        captured = {}

        async def _capture(event):
            captured["event"] = event

        adapter.handle_message = _capture
        asyncio.run(adapter._dispatch_event({
            "type": "message",
            "source": source,
            "message": message,
            "replyToken": "rt",
            "webhookEventId": webhook_id,
        }))
        return captured.get("event")


class TestLineGroupFilters(_LineGroupHarness):
    """``LINE_REQUIRE_MENTION_CHATS``, media bypass, and group/room type ignore.

    DMs are never mention-gated and never type-filtered. Ignore runs before
    the mention gate. Postbacks stay on the slow-LLM path.
    """

    def test_defaults_are_empty_and_do_not_filter(self, monkeypatch):
        adapter = self._adapter(
            monkeypatch, LINE_ALLOW_ALL_USERS="false", LINE_ALLOWED_GROUPS="Cfree")
        assert adapter.require_mention_chats == set()
        assert adapter.reply_without_mention_media_types == set()
        assert adapter.ignore_group_message_types == set()
        event = self._dispatch(adapter, _group("Cfree"), _text("hello"), "default-text")
        assert event is not None and event.text == "hello"
        sticker = self._dispatch(
            adapter, _group("Cfree"),
            {"type": "sticker", "id": "s1", "keywords": ["fun"]}, "default-sticker")
        assert sticker is not None and sticker.text == "[sticker: fun]"

    @pytest.mark.parametrize("source,webhook", [(_group(), "g-plain"), (_room(), "r-plain")])
    def test_unmentioned_text_dropped_in_required_chats(self, monkeypatch, source, webhook):
        adapter = self._gated(monkeypatch)
        assert self._dispatch(adapter, source, _text("hello"), webhook) is None
        adapter._download_media.assert_not_awaited()

    @pytest.mark.parametrize("source,webhook", [(_group(), "g-self"), (_room(), "r-self")])
    def test_is_self_mention_strips_span_and_dispatches(self, monkeypatch, source, webhook):
        adapter = self._gated(monkeypatch)
        event = self._dispatch(
            adapter, source,
            _text("@Hermes hello", [_bot_mention(0, 7)]), webhook)
        assert event is not None
        assert event.text == "hello"
        assert event.source.chat_type == source["type"]

    def test_user_id_mention_matches_when_is_self_is_absent(self, monkeypatch):
        adapter = self._gated(monkeypatch)
        event = self._dispatch(
            adapter, _group(),
            _text("@Hermes ping", [_bot_mention(0, 7, is_self=None)]), "g-uid")
        assert event is not None and event.text == "ping"

    def test_is_self_matches_even_if_cached_user_id_differs(self, monkeypatch):
        adapter = self._gated(monkeypatch)
        adapter._bot_user_id = "Uother"
        event = self._dispatch(
            adapter, _room(),
            _text("@Hermes hi", [_bot_mention(0, 7, user_id="Ubot")]), "r-isself")
        assert event is not None and event.text == "hi"

    def test_user_id_alone_does_not_match_without_cached_bot_id(self, monkeypatch):
        adapter = self._gated(monkeypatch)
        adapter._bot_user_id = None
        assert self._dispatch(
            adapter, _group(),
            _text("@Hermes hi", [_bot_mention(0, 7, is_self=None)]), "g-nobot") is None

    def test_at_all_and_other_users_do_not_open_the_gate(self, monkeypatch):
        adapter = self._gated(monkeypatch)
        assert self._dispatch(
            adapter, _group(),
            _text("@All hello", [{"index": 0, "length": 4, "type": "all"}]), "g-all") is None
        assert self._dispatch(
            adapter, _room(),
            _text("@Bob hi", [_bot_mention(0, 4, is_self=False, user_id="Usomeone")]),
            "r-other") is None

    def test_mention_span_in_the_middle_is_stripped(self, monkeypatch):
        adapter = self._gated(monkeypatch)
        event = self._dispatch(
            adapter, _group("Cfree"),
            _text("hey @Hermes there", [_bot_mention(4, 7)]), "g-mid")
        assert event is not None and event.text == "hey there"

    def test_other_users_mention_is_left_in_place(self, monkeypatch):
        adapter = self._gated(monkeypatch)
        event = self._dispatch(
            adapter, _group("Cfree"),
            _text("@Bob hi", [_bot_mention(0, 4, is_self=False, user_id="Usomeone")]),
            "g-keep")
        assert event is not None and event.text == "@Bob hi"

    @pytest.mark.parametrize("source,webhook", [(_group("Cfree"), "g-free"), (_room("Rfree"), "r-free")])
    def test_unlisted_allowlisted_chat_responds_without_mention(self, monkeypatch, source, webhook):
        adapter = self._gated(monkeypatch)
        event = self._dispatch(adapter, source, _text("hello"), webhook)
        assert event is not None and event.text == "hello"

    def test_allow_all_still_mention_gates_listed_chats_only(self, monkeypatch):
        adapter = self._adapter(
            monkeypatch,
            LINE_ALLOW_ALL_USERS="true",
            LINE_REQUIRE_MENTION_CHATS="Cneed",
        )
        assert self._dispatch(adapter, _group("Cneed"), _text("hello"), "all-drop") is None
        event = self._dispatch(adapter, _group("Cfree"), _text("hello"), "all-free")
        assert event is not None and event.text == "hello"
        mentioned = self._dispatch(
            adapter, _group("Cneed"),
            _text("@Hermes hi", [_bot_mention(0, 7)]), "all-hit")
        assert mentioned is not None and mentioned.text.endswith("hi")
        assert "Umember: hello" in mentioned.text

    def test_help_bypasses_mention_gate_and_strips_the_tag(self, monkeypatch):
        adapter = self._gated(monkeypatch)
        bare = self._dispatch(adapter, _group(), _text("/help"), "g-help")
        assert bare is not None and bare.text == "/help"
        tagged = self._dispatch(
            adapter, _room(),
            _text("@Hermes /help", [_bot_mention(0, 7)]), "r-help")
        assert tagged is not None and tagged.text == "/help"

    def test_unknown_or_cli_only_slash_does_not_bypass(self, monkeypatch):
        adapter = self._gated(monkeypatch)
        assert self._dispatch(adapter, _group(), _text("/not-a-real-command"), "g-nope") is None
        assert self._dispatch(adapter, _room(), _text("/clear"), "r-clear") is None

    def test_dm_is_never_mention_gated(self, monkeypatch):
        adapter = self._gated(monkeypatch, LINE_REQUIRE_MENTION_CHATS="Cneed,Rneed,Udm")
        event = self._dispatch(adapter, _dm(), _text("hello"), "dm-text")
        assert event is not None and event.text == "hello"
        assert event.source.chat_type == "dm"

    def test_mention_list_does_not_authorize_an_unlisted_chat(self, monkeypatch):
        adapter = self._gated(monkeypatch, LINE_REQUIRE_MENTION_CHATS="Cneed,Cother")
        assert self._dispatch(
            adapter, _group("Cother"),
            _text("@Hermes hi", [_bot_mention(0, 7)]), "g-unauth") is None

    def test_require_list_splits_whitespace(self, monkeypatch):
        adapter = self._adapter(
            monkeypatch,
            LINE_ALLOW_ALL_USERS="false",
            LINE_ALLOWED_GROUPS="Cneed",
            LINE_ALLOWED_ROOMS="Rneed",
            LINE_REQUIRE_MENTION_CHATS=" Cneed , Rneed ",
        )
        assert adapter.require_mention_chats == {"Cneed", "Rneed"}
        assert self._dispatch(adapter, _room(), _text("hello"), "r-ws") is None

    def test_configured_media_bypasses_mention_and_other_types_do_not(self, monkeypatch):
        adapter = self._gated(
            monkeypatch, LINE_REPLY_WITHOUT_MENTION_MEDIA_TYPES="image, video")
        assert adapter.reply_without_mention_media_types == {"image", "video"}
        image = self._dispatch(
            adapter, _group(), {"type": "image", "id": "img1"}, "g-image")
        assert image is not None and image.text == "[image]"
        assert image.media_urls == ["/cache/line.bin"]
        video = self._dispatch(
            adapter, _room(), {"type": "video", "id": "vid1"}, "r-video")
        assert video is not None and video.text == "[video]"
        assert self._dispatch(
            adapter, _group(), {"type": "audio", "id": "aud1"}, "g-audio") is None
        assert self._dispatch(adapter, _room(), _text("hello"), "r-text") is None
        assert self._dispatch(
            adapter, _group(), {"type": "sticker", "id": "st1"}, "g-sticker-block") is None
        # audio/text/sticker never reach download; image and video do.
        assert adapter._download_media.await_count == 2

    def test_voice_alias_lets_audio_through(self, monkeypatch):
        adapter = self._gated(monkeypatch, LINE_REPLY_WITHOUT_MENTION_MEDIA_TYPES="voice")
        assert adapter.reply_without_mention_media_types == {"audio"}
        event = self._dispatch(
            adapter, _group(), {"type": "audio", "id": "aud2"}, "g-voice")
        assert event is not None and event.text == "[audio]"

    def test_text_token_does_not_bypass_mention(self, monkeypatch):
        adapter = self._gated(
            monkeypatch, LINE_REPLY_WITHOUT_MENTION_MEDIA_TYPES="text,image")
        assert "text" not in adapter.reply_without_mention_media_types
        assert "image" in adapter.reply_without_mention_media_types
        assert self._dispatch(adapter, _group(), _text("hello"), "g-text-no") is None

    @pytest.mark.parametrize("source,webhook", [(_group("Cfree"), "g-emo"), (_room("Rfree"), "r-emo")])
    def test_sticker_and_pure_emoji_ignored_in_groups_and_rooms(self, monkeypatch, source, webhook):
        adapter = self._gated(monkeypatch, LINE_IGNORE_GROUP_MESSAGE_TYPES="sticker,emoji")
        assert self._dispatch(
            adapter, source, {"type": "sticker", "id": "st", "keywords": ["fun"]}, webhook + "-st") is None
        assert self._dispatch(adapter, source, _emoji(), webhook + "-em") is None
        # Whitespace around a LINE emoji placeholder is still emoji-only.
        padded = _emoji(" (laugh) ", spans=[(1, 7)])
        assert self._dispatch(adapter, source, padded, webhook + "-pad") is None

    def test_mixed_text_and_emoji_is_not_ignored(self, monkeypatch):
        adapter = self._gated(monkeypatch, LINE_IGNORE_GROUP_MESSAGE_TYPES="sticker,emoji")
        event = self._dispatch(
            adapter, _group("Cfree"), _emoji("hi (laugh)", spans=[(3, 7)]), "g-mixed")
        assert event is not None
        assert event.text == "hi (laugh)"

    def test_ignore_runs_before_mention_and_media_bypass(self, monkeypatch):
        adapter = self._gated(
            monkeypatch,
            LINE_IGNORE_GROUP_MESSAGE_TYPES="image,sticker",
            LINE_REPLY_WITHOUT_MENTION_MEDIA_TYPES="image,video",
        )
        assert self._dispatch(
            adapter, _group(),
            {"type": "sticker", "id": "st", "mention": {"mentionees": [_bot_mention(0, 1)]}},
            "g-st-first") is None
        assert self._dispatch(
            adapter, _room(), {"type": "image", "id": "img"}, "r-img-first") is None
        video = self._dispatch(
            adapter, _group(), {"type": "video", "id": "vid"}, "g-vid-ok")
        assert video is not None and video.text.endswith("[video]")
        assert "Umember: [sticker]" in video.text
        assert "[image]" not in video.text
        adapter._download_media.assert_awaited_once()

    def test_photo_alias_ignores_images(self, monkeypatch):
        adapter = self._gated(monkeypatch, LINE_IGNORE_GROUP_MESSAGE_TYPES="photo")
        assert adapter.ignore_group_message_types == {"image"}
        assert self._dispatch(
            adapter, _group("Cfree"), {"type": "image", "id": "img"}, "g-photo") is None

    def test_text_token_does_not_ignore_chat(self, monkeypatch):
        adapter = self._gated(monkeypatch, LINE_IGNORE_GROUP_MESSAGE_TYPES="text,sticker")
        assert adapter.ignore_group_message_types == {"sticker"}
        event = self._dispatch(adapter, _group("Cfree"), _text("hello"), "g-text-keep")
        assert event is not None and event.text == "hello"

    def test_dm_is_never_type_filtered(self, monkeypatch):
        adapter = self._gated(monkeypatch, LINE_IGNORE_GROUP_MESSAGE_TYPES="sticker,emoji,image")
        sticker = self._dispatch(
            adapter, _dm(), {"type": "sticker", "id": "st", "keywords": ["fun"]}, "dm-st")
        assert sticker is not None and sticker.text == "[sticker: fun]"
        emoji = self._dispatch(adapter, _dm(), _emoji(), "dm-em")
        assert emoji is not None and emoji.text == "(laugh)"
        image = self._dispatch(adapter, _dm(), {"type": "image", "id": "img"}, "dm-img")
        assert image is not None and image.text == "[image]"

    def test_postback_in_mention_required_group_still_delivers(self, monkeypatch):
        adapter = self._gated(monkeypatch, LINE_IGNORE_GROUP_MESSAGE_TYPES="sticker,emoji")
        adapter._client = MagicMock()
        adapter._client.reply = AsyncMock()
        adapter._client.push = AsyncMock()
        rid = adapter._cache.register_pending("Cneed")
        adapter._cache.set_ready(rid, "cached answer")
        asyncio.run(adapter._dispatch_event({
            "type": "postback",
            "source": _group(),
            "replyToken": "rt",
            "postback": {"data": json.dumps({"action": "show_response", "request_id": rid})},
            "webhookEventId": "pb-mention",
        }))
        adapter._client.reply.assert_awaited_once()
        assert adapter._client.reply.await_args.args[1][0]["text"] == "cached answer"
        assert adapter._cache.get(rid).state is State.DELIVERED


class TestLineObservation(_LineGroupHarness):
    def _mention(self, adapter, source, webhook):
        event = self._dispatch(adapter, source, _text("@Hermes update?", [_bot_mention(0, 7)]), webhook)
        assert event.text.endswith("update?")
        return event.text

    @pytest.mark.parametrize("value,limit,warn", [
        (None, 50, False), ("  ", 50, False), ("0", 0, False), ("2", 2, False),
        ("501", 500, True), ("-1", 50, True), ("2.5", 50, True), ("many", 50, True),
    ])
    def test_limit_controls_retained_messages(self, monkeypatch, caplog, value, limit, warn):
        key = "LINE_OBSERVED_UNMENTIONED_MESSAGES"
        env = {} if value is None else {key: value}
        adapter = self._gated(monkeypatch, **env)
        assert any(key in r.getMessage() and r.levelname == "WARNING" for r in caplog.records) == warn
        if limit == 0:
            adapter._client = AsyncMock()
        source = _group(user_id="U")
        for i in range(limit + 1):
            body = f"msg-{i:03}"
            assert self._dispatch(adapter, source, _text(body), body) is None
        text = self._mention(adapter, source, "trigger")
        assert text.count("msg-") == limit
        assert "msg-000" not in text
        if limit:
            assert f"msg-{limit:03}" in text
        else:
            assert text == "update?"
            adapter._client.get_member_name.assert_not_called()

    @pytest.mark.parametrize("per_user", [False, True])
    def test_cursor_follows_session_across_triggers(self, monkeypatch, per_user):
        adapter = self._gated(monkeypatch)
        adapter.config.extra["group_sessions_per_user"] = per_user
        a, b = _room(user_id="UA"), _room(user_id="UB")
        assert self._dispatch(adapter, b, _text("early news"), "one") is None
        assert "UB: early news" in self._mention(adapter, a, "a1")
        assert self._mention(adapter, a, "a2") == "update?"
        assert self._dispatch(adapter, a, _text("late news"), "two") is None
        text = self._mention(adapter, b, "b1")
        assert text.count("early news") == int(per_user)
        assert text.count("UA: late news") == 1
        assert text.count("update?") == 1
        assert self._mention(adapter, b, "b2") == "update?"
        assert self._mention(adapter, a, "a3").count("late news") == int(per_user)

    def test_context_budget_preserves_recent_lines_and_the_request(self, monkeypatch):
        adapter = self._gated(monkeypatch)
        source = _group()
        for i in range(8):
            self._dispatch(adapter, source, _text(f"paste-{i} " + "x" * 2200), str(i))
        text = self._mention(adapter, source, "trigger")
        assert len(text.removesuffix("update?")) <= 8000
        assert "paste-4" not in text
        assert text.index("paste-5") < text.index("paste-6") < text.index("paste-7")
        self._dispatch(adapter, source, _text("longest " + "x" * 10000), "long")
        text = self._mention(adapter, source, "trigger2")
        assert "Umember: longest" in text and len(text.removesuffix("update?")) <= 8000
        assert self._mention(adapter, source, "trigger3") == "update?"

    def test_passive_media_and_ignored_types_never_download(self, monkeypatch):
        adapter = self._gated(monkeypatch, LINE_IGNORE_GROUP_MESSAGE_TYPES="sticker,emoji",
                              LINE_REPLY_WITHOUT_MENTION_MEDIA_TYPES="sticker")
        source = _room()
        for kind in ("image", "video", "audio", "file", "sticker", "location"):
            assert self._dispatch(adapter, source, {"type": kind, "title": "Pier"}, kind) is None
        assert self._dispatch(adapter, source, _emoji(), "emoji") is None
        adapter._download_media.assert_not_awaited()
        text = self._mention(adapter, source, "trigger")
        for marker in ("[image]", "[video]", "[audio]", "[file]", "[sticker]", "[location: Pier", "(laugh)"):
            assert text.count(marker) == 1

    @pytest.mark.asyncio
    async def test_media_trigger_takes_only_context_received_before_download(self, monkeypatch):
        adapter = self._gated(monkeypatch, LINE_REPLY_WITHOUT_MENTION_MEDIA_TYPES="image")
        adapter.handle_message = AsyncMock()
        downloading, release = asyncio.Event(), asyncio.Event()

        async def download(*_args, **_kwargs):
            downloading.set()
            await release.wait()
            return "/cache/picture", "image/png"

        adapter._download_media.side_effect = download

        async def dispatch(message):
            await adapter._dispatch_event({"type": "message", "source": _group(), "message": message})

        await dispatch(_text("before photo"))
        photo = asyncio.create_task(dispatch({"type": "image", "id": "photo"}))
        await asyncio.wait_for(downloading.wait(), timeout=5)
        await dispatch(_text("after photo"))
        release.set()
        await photo
        event = adapter.handle_message.await_args.args[0]
        assert event.media_urls == ["/cache/picture"]
        assert event.text.endswith("[image]") and "before photo" in event.text and "after photo" not in event.text
        await dispatch(_text("@Hermes hi", [_bot_mention(0, 7)]))
        text = adapter.handle_message.await_args.args[0].text
        assert "after photo" in text and "before photo" not in text and "[image]" not in text

    def test_image_bypass_injects_unread_context_and_advances_the_cursor(self, monkeypatch):
        """An unmentioned image still consumes the session cursor after it replies.

        When ``LINE_REPLY_WITHOUT_MENTION_MEDIA_TYPES=image``, a photo with no
        text in a mention-gated chat is a trigger: unread background is
        prepended to ``[image]``, the cursor advances, and a later @mention or
        image sees only newer unmentioned messages.
        """
        adapter = self._gated(monkeypatch, LINE_REPLY_WITHOUT_MENTION_MEDIA_TYPES="image")
        source = _group()
        assert self._dispatch(adapter, source, _text("hello before photo"), "pre1") is None
        assert self._dispatch(adapter, source, _text("more chatter"), "pre2") is None
        image = self._dispatch(adapter, source, {"type": "image", "id": "img-only"}, "img1")
        assert image is not None
        assert image.text.endswith("[image]")
        assert "hello before photo" in image.text and "more chatter" in image.text
        assert image.media_urls == ["/cache/line.bin"]
        assert self._mention(adapter, source, "after-image") == "update?"
        assert self._dispatch(adapter, source, _text("after the photo"), "post") is None
        later = self._dispatch(adapter, source, {"type": "image", "id": "img-two"}, "img2")
        assert later is not None and later.text.endswith("[image]")
        assert "after the photo" in later.text
        assert "hello before photo" not in later.text and "more chatter" not in later.text
        assert adapter._download_media.await_count == 2


    @pytest.mark.parametrize("fails,sender", [(False, "Ada"), (True, "Umember")])
    @pytest.mark.asyncio
    async def test_member_lookup_is_best_effort_and_cached(self, monkeypatch, fails, sender):
        adapter = self._gated(monkeypatch)
        release, finished = asyncio.Event(), asyncio.Event()

        async def resolve(*_args):
            await release.wait()
            finished.set()
            if fails:
                raise OSError("profile unavailable")
            return "Ada"

        adapter._client = AsyncMock()
        adapter._client.get_member_name.side_effect = resolve
        adapter.handle_message = AsyncMock()

        async def dispatch(msg):
            await adapter._dispatch_event({"type": "message", "source": _room(), "message": msg})

        await dispatch(_text("first"))
        await dispatch(_text("second"))
        adapter.handle_message.assert_not_awaited()
        await dispatch(_text("@Hermes hi", [_bot_mention(0, 7)]))
        assert "Umember: first" in adapter.handle_message.await_args.args[0].text

        release.set()
        await asyncio.wait_for(finished.wait(), timeout=5)
        await dispatch(_text("third"))
        await dispatch(_text("@Hermes again", [_bot_mention(0, 7)]))
        assert f"{sender}: third" in adapter.handle_message.await_args.args[0].text
        adapter._client.get_member_name.assert_awaited_once_with("room", "Rneed", "Umember")

    @pytest.mark.parametrize("command", ["/new", "/reset"])
    def test_commands_preserve_or_reset_only_the_callers_cursor(self, monkeypatch, command):
        adapter = self._gated(monkeypatch)
        a, b = _group(user_id="UA"), _group(user_id="UB")
        self._dispatch(adapter, a, _text("/not-a-real-command"), "unknown")
        assert self._dispatch(adapter, a, _text("/help"), "help").text == "/help"
        text = self._mention(adapter, a, "after-help")
        assert "/not-a-real-command" in text and "/help" not in text
        self._dispatch(adapter, b, _text("before reset"), "before")
        tagged = _text(f"@Hermes {command}", [_bot_mention(0, 7)])
        assert self._dispatch(adapter, a, tagged, "reset").text == command
        self._dispatch(adapter, b, _text("after reset"), "after")
        text = self._mention(adapter, a, "fresh")
        assert "after reset" in text and "before reset" not in text
        text = self._mention(adapter, b, "independent")
        assert "before reset" in text and "after reset" in text and command not in text

    @pytest.mark.parametrize("source", [_group("Cother"), _room("Rother")])
    def test_authorization_and_mention_gate_precede_recording(self, monkeypatch, source):
        chat_id, _ = _resolve_chat(source)
        adapter = self._gated(monkeypatch, LINE_REQUIRE_MENTION_CHATS=chat_id,
                              LINE_IGNORE_GROUP_MESSAGE_TYPES="sticker")
        assert self._dispatch(adapter, source, _text("unauthorized"), "denied") is None
        adapter.allow_all = True
        adapter.require_mention_chats.clear()
        assert self._dispatch(adapter, source, {"type": "sticker"}, "free") is None
        adapter.require_mention_chats.add(chat_id)
        assert self._mention(adapter, source, "allowed") == "update?"

    def test_dm_postback_and_lifecycle_do_not_enter_observations(self, monkeypatch):
        adapter = self._gated(monkeypatch, LINE_ALLOW_ALL_USERS="true")
        # Reuse a chat ID to make accidental recording visible to the group trigger.
        assert self._dispatch(adapter, _dm("Cneed"), _text("private"), "dm").text == "private"
        for kind in ("join", "postback"):
            asyncio.run(adapter._dispatch_event({"type": kind, "source": _group(), "webhookEventId": kind}))
        assert self._mention(adapter, _group(), "trigger") == "update?"


class TestLineFilterDashboardEnv:
    """The group and room controls are Dashboard fields beside the group/room allowlists."""

    def test_plugin_yaml_and_catalog_expose_the_four_vars(self):
        from pathlib import Path

        from hermes_cli.config import OPTIONAL_ENV_VARS
        from hermes_cli.plugins import discover_plugins
        from hermes_cli.plugins_manifest import parse_manifest_file
        from hermes_cli.web_routers.messaging import _messaging_env_info
        from hermes_cli.web_server_messaging import _messaging_platform_catalog
        from gateway.platform_registry import platform_registry

        manifest_path = Path(__file__).resolve().parents[2] / "plugins" / "platforms" / "line" / "plugin.yaml"
        manifest = parse_manifest_file(manifest_path, manifest_path.parent, "bundled", "platforms")
        assert manifest is not None
        names = manifest.dashboard_env_vars
        rooms = names.index("LINE_ALLOWED_ROOMS")
        assert names[rooms - 1] == "LINE_ALLOWED_GROUPS"
        assert names[rooms + 1:rooms + 5] == (
            "LINE_REQUIRE_MENTION_CHATS",
            "LINE_REPLY_WITHOUT_MENTION_MEDIA_TYPES",
            "LINE_IGNORE_GROUP_MESSAGE_TYPES",
            "LINE_OBSERVED_UNMENTIONED_MESSAGES",
        )

        ignore = OPTIONAL_ENV_VARS["LINE_IGNORE_GROUP_MESSAGE_TYPES"]
        media = OPTIONAL_ENV_VARS["LINE_REPLY_WITHOUT_MENTION_MEDIA_TYPES"]
        mention = OPTIONAL_ENV_VARS["LINE_REQUIRE_MENTION_CHATS"]
        assert ignore["password"] is False and ignore["placeholder"] == "Ex: sticker,emoji"
        assert "sticker,emoji" in ignore["help"]
        assert media["placeholder"] == "Ex: image,video"
        assert mention["is_list"] is True and mention["placeholder"] == "Ex: Cxxxx,Rxxxx"
        observed = OPTIONAL_ENV_VARS["LINE_OBSERVED_UNMENTIONED_MESSAGES"]
        assert observed["password"] is False and observed["placeholder"] == "Ex: 50"
        assert observed["help"]
        for meta in (ignore, media, mention, observed):
            assert meta["category"] == "messaging"
            assert meta["description"] and meta["prompt"]

        assert _messaging_env_info("LINE_IGNORE_GROUP_MESSAGE_TYPES")["placeholder"] == "Ex: sticker,emoji"
        assert _messaging_env_info("LINE_IGNORE_GROUP_MESSAGE_TYPES")["is_list"] is False
        assert _messaging_env_info("LINE_REQUIRE_MENTION_CHATS")["is_list"] is True
        assert _messaging_env_info("LINE_REPLY_WITHOUT_MENTION_MEDIA_TYPES")["help"]

        discover_plugins()
        entry = platform_registry.get("line")
        assert entry is not None
        assert entry.dashboard_env_vars[rooms + 1:rooms + 5] == names[rooms + 1:rooms + 5]
        card = next(item for item in _messaging_platform_catalog() if item["id"] == "line")
        env_vars = card["env_vars"]
        card_rooms = env_vars.index("LINE_ALLOWED_ROOMS")
        assert env_vars[card_rooms - 1] == "LINE_ALLOWED_GROUPS"
        assert env_vars[card_rooms + 1:card_rooms + 5] == names[rooms + 1:rooms + 5]
        assert env_vars.count("LINE_REQUIRE_MENTION_CHATS") == 1
        assert env_vars.count("LINE_IGNORE_GROUP_MESSAGE_TYPES") == 1

    def test_saved_filter_values_are_plain_text_not_redacted(self):
        """Channels payload returns these settings unmasked, like the allowlists.

        Non-list fields used to ship only ``redacted_value`` (``«redacted:…``),
        and the form showed that preview as the placeholder of a saved field.
        """
        from hermes_cli.plugins import discover_plugins
        from hermes_cli.web_routers.messaging import _messaging_platform_payload
        from hermes_cli.web_server_messaging import _messaging_platform_catalog

        discover_plugins()
        entry = next(item for item in _messaging_platform_catalog() if item["id"] == "line")
        saved = {
            "LINE_CHANNEL_ACCESS_TOKEN": "secret-channel-token",
            "LINE_ALLOWED_GROUPS": "Callowed",
            "LINE_REQUIRE_MENTION_CHATS": "Cneed,Rneed",
            "LINE_REPLY_WITHOUT_MENTION_MEDIA_TYPES": "image,video",
            "LINE_IGNORE_GROUP_MESSAGE_TYPES": "sticker,emoji",
            "LINE_OBSERVED_UNMENTIONED_MESSAGES": "37",
        }
        payload = _messaging_platform_payload(entry, saved, runtime={}, scoped=True)
        fields = {field["key"]: field for field in payload["env_vars"]}

        groups = fields["LINE_ALLOWED_GROUPS"]
        assert groups["is_password"] is False
        assert groups["value"] == "Callowed"
        assert "redacted" not in (groups["value"] or "")

        expected = {
            "LINE_REQUIRE_MENTION_CHATS": "Cneed,Rneed",
            "LINE_REPLY_WITHOUT_MENTION_MEDIA_TYPES": "image,video",
            "LINE_IGNORE_GROUP_MESSAGE_TYPES": "sticker,emoji",
            "LINE_OBSERVED_UNMENTIONED_MESSAGES": "37",
        }
        for key, plain in expected.items():
            field = fields[key]
            assert field["is_password"] is False
            assert field["is_set"] is True
            assert field["value"] == plain
            assert "redacted" not in field["value"]
            assert field["value"] != field["redacted_value"]
            assert field["placeholder"].startswith("Ex: ")

        observed = fields["LINE_OBSERVED_UNMENTIONED_MESSAGES"]
        assert observed["placeholder"] == "Ex: 50" and observed["is_list"] is False

        token = fields["LINE_CHANNEL_ACCESS_TOKEN"]
        assert token["is_password"] is True
        assert token["value"] is None
        assert token["redacted_value"]
        assert "secret-channel-token" not in str(token["redacted_value"])


class TestLineBotInfo:
    def test_get_bot_info_reads_user_id_and_display_name(self, monkeypatch):
        import aiohttp

        class _Resp:
            def __init__(self, status, payload):
                self.status = status
                self._payload = payload

            async def json(self):
                return self._payload

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

        class _Session:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            def get(self, *_args, **_kwargs):
                return _Resp(200, {"userId": "Ubot", "displayName": "Hermes", "basicId": "@hermes"})

        monkeypatch.setattr(aiohttp, "ClientSession", _Session)
        info = asyncio.run(_line._LineClient("tok").get_bot_info())
        assert info == {"userId": "Ubot", "displayName": "Hermes"}

    def test_get_bot_info_returns_empty_on_http_error(self, monkeypatch):
        import aiohttp

        class _Resp:
            status = 401

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

        class _Session:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            def get(self, *_args, **_kwargs):
                return _Resp()

        monkeypatch.setattr(aiohttp, "ClientSession", _Session)
        assert asyncio.run(_line._LineClient("tok").get_bot_info()) == {}

    @pytest.mark.asyncio
    async def test_connect_caches_bot_user_id_and_display_name(self, monkeypatch):
        from gateway.config import PlatformConfig

        async def _bind(*_args, **_kwargs):
            return MagicMock()

        monkeypatch.setattr("gateway.platforms.shared_ingress.bind_listener", _bind)
        monkeypatch.setattr(
            "gateway.status.acquire_scoped_lock", lambda *_a, **_k: (True, None))
        monkeypatch.setattr(
            _line._LineClient, "get_bot_info",
            AsyncMock(return_value={"userId": "Ubot", "displayName": "Hermes"}))
        adapter = LineAdapter(PlatformConfig(
            enabled=True, extra={"channel_access_token": "tok", "channel_secret": "sec"}))
        assert await adapter.connect() is True
        assert adapter._bot_user_id == "Ubot"
        assert adapter._bot_display_name == "Hermes"
        await adapter.disconnect()

