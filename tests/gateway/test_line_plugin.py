"""Tests for the LINE platform adapter plugin.

Covers the seven synthesis areas from the PR review:

1. webhook signature verification (HMAC-SHA256, base64) + tampering rejection
2. inbound chat-id resolution for user / group / room sources
3. three-allowlist gating (users / groups / rooms / allow_all)
4. inbound dedup via webhookEventId
5. RequestCache state machine (PENDING → READY → DELIVERED, ERROR)
6. Markdown stripping with URL preservation + LINE-sized chunking
7. send routing: reply token preferred → push fallback → batched at 5/call
8. register() metadata + standalone_send shape
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import base64
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

# Load plugins/platforms/line/adapter.py under plugin_adapter_line so it
# cannot collide with sibling platform-plugin tests in the same xdist worker.
_line = load_plugin_adapter("line")

verify_line_signature = _line.verify_line_signature
strip_markdown_preserving_urls = _line.strip_markdown_preserving_urls
split_for_line = _line.split_for_line
build_postback_button_message = _line.build_postback_button_message
_resolve_chat = _line._resolve_chat
_allowed_for_source = _line._allowed_for_source
_is_system_bypass = _line._is_system_bypass
RequestCache = _line.RequestCache
State = _line.State
LineAdapter = _line.LineAdapter
register = _line.register
check_requirements = _line.check_requirements
validate_config = _line.validate_config
_standalone_send = _line._standalone_send
_env_enablement = _line._env_enablement
_MessageDeduplicator = _line._MessageDeduplicator


# ---------------------------------------------------------------------------
# 1. Signature verification
# ---------------------------------------------------------------------------

class TestSignature:

    def _sign(self, body: bytes, secret: str) -> str:
        digest = hmac.new(secret.encode(), body, hashlib.sha256).digest()
        return base64.b64encode(digest).decode()

    def test_valid_signature_passes(self):
        body = b'{"events": []}'
        sig = self._sign(body, "secret")
        assert verify_line_signature(body, sig, "secret")

    def test_tampered_body_rejected(self):
        body = b'{"events": []}'
        sig = self._sign(body, "secret")
        assert not verify_line_signature(body + b" ", sig, "secret")

    def test_wrong_secret_rejected(self):
        body = b'{"events": []}'
        sig = self._sign(body, "secret")
        assert not verify_line_signature(body, sig, "different")

    def test_empty_signature_rejected(self):
        assert not verify_line_signature(b"x", "", "secret")

    def test_empty_secret_rejected(self):
        assert not verify_line_signature(b"x", "AAAA", "")

    def test_garbage_signature_rejected(self):
        assert not verify_line_signature(b"hello", "not base64 at all!!", "s")


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

    def test_room_source(self):
        chat_id, ctype = _resolve_chat({"type": "room", "roomId": "R789", "userId": "U123"})
        assert chat_id == "R789"
        assert ctype == "room"

    def test_unknown_source_falls_back_to_dm(self):
        chat_id, ctype = _resolve_chat({"type": "weird"})
        assert chat_id == ""
        assert ctype == "dm"

    def test_empty_source(self):
        chat_id, ctype = _resolve_chat({})
        assert chat_id == ""
        assert ctype == "dm"


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

    def test_user_not_in_allowlist_rejected(self):
        src = {"type": "user", "userId": "Uother"}
        assert not _allowed_for_source(src, allow_all=False, user_ids={"Uok"}, group_ids=set(), room_ids=set())

    def test_group_uses_group_list_not_user_list(self):
        src = {"type": "group", "groupId": "Cok", "userId": "Uany"}
        assert _allowed_for_source(src, allow_all=False, user_ids={"Uany"}, group_ids={"Cok"}, room_ids=set())
        assert not _allowed_for_source(src, allow_all=False, user_ids={"Uany"}, group_ids=set(), room_ids=set())

    def test_room_uses_room_list(self):
        src = {"type": "room", "roomId": "Rok"}
        assert _allowed_for_source(src, allow_all=False, user_ids=set(), group_ids=set(), room_ids={"Rok"})
        assert not _allowed_for_source(src, allow_all=False, user_ids=set(), group_ids=set(), room_ids=set())

    def test_unknown_type_rejected(self):
        src = {"type": "weird"}
        assert not _allowed_for_source(src, allow_all=False, user_ids=set(), group_ids=set(), room_ids=set())


# ---------------------------------------------------------------------------
# 4. Inbound dedup
# ---------------------------------------------------------------------------

class TestDedup:

    def test_first_event_not_duplicate(self):
        d = _MessageDeduplicator()
        assert not d.is_duplicate("evt1")

    def test_repeat_event_marked_duplicate(self):
        d = _MessageDeduplicator()
        d.is_duplicate("evt1")
        assert d.is_duplicate("evt1")

    def test_blank_id_not_treated_as_duplicate(self):
        d = _MessageDeduplicator()
        # Blank IDs should always pass through (don't lock out unidentifiable events).
        assert not d.is_duplicate("")
        assert not d.is_duplicate("")

    def test_lru_eviction_under_pressure(self):
        d = _MessageDeduplicator(max_size=10)
        for i in range(20):
            d.is_duplicate(f"evt{i}")
        # Exact eviction order isn't specified, but the cap must be enforced.
        # Insert one more and assert the bookkeeping doesn't grow without bound.
        d.is_duplicate("evt20")
        assert len(d._seen) <= 20  # bounded — exact cap depends on eviction policy


# ---------------------------------------------------------------------------
# 5. RequestCache state machine
# ---------------------------------------------------------------------------

class TestRequestCache:

    def test_register_pending_is_pending(self):
        c = RequestCache()
        rid = c.register_pending("Uchat")
        assert c.get(rid).state is State.PENDING
        assert c.get(rid).chat_id == "Uchat"

    def test_set_ready_transitions(self):
        c = RequestCache()
        rid = c.register_pending("Uchat")
        c.set_ready(rid, "the answer")
        assert c.get(rid).state is State.READY
        assert c.get(rid).payload == "the answer"

    def test_set_error_transitions(self):
        c = RequestCache()
        rid = c.register_pending("Uchat")
        c.set_error(rid, "boom")
        assert c.get(rid).state is State.ERROR
        assert c.get(rid).payload == "boom"

    def test_mark_delivered_from_ready(self):
        c = RequestCache()
        rid = c.register_pending("Uchat")
        c.set_ready(rid, "x")
        c.mark_delivered(rid)
        assert c.get(rid).state is State.DELIVERED

    def test_mark_delivered_from_error(self):
        c = RequestCache()
        rid = c.register_pending("Uchat")
        c.set_error(rid, "x")
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

    def test_find_pending_for_chat(self):
        c = RequestCache()
        rid_a = c.register_pending("Ua")
        rid_b = c.register_pending("Ub")
        assert c.find_pending_for_chat("Ua") == rid_a
        assert c.find_pending_for_chat("Ub") == rid_b
        assert c.find_pending_for_chat("Uc") is None
        c.set_ready(rid_a, "x")
        # No longer PENDING — should not be found
        assert c.find_pending_for_chat("Ua") is None


# ---------------------------------------------------------------------------
# 6. Markdown stripping + chunking
# ---------------------------------------------------------------------------

class TestMarkdownAndChunking:

    def test_bold_stripped(self):
        assert strip_markdown_preserving_urls("**hello**") == "hello"

    def test_italic_stripped(self):
        assert strip_markdown_preserving_urls("*hello*") == "hello"

    def test_inline_code_unfenced(self):
        assert strip_markdown_preserving_urls("run `ls -la`") == "run ls -la"

    def test_link_preserved_with_url(self):
        out = strip_markdown_preserving_urls("see [here](https://x.com)")
        assert "https://x.com" in out
        assert "here (https://x.com)" in out

    def test_heading_prefix_stripped(self):
        out = strip_markdown_preserving_urls("# Title\n## Sub")
        assert out == "Title\nSub"

    def test_bullet_marker_replaced(self):
        out = strip_markdown_preserving_urls("- a\n- b")
        assert out == "• a\n• b"

    def test_code_fence_content_kept(self):
        # Source files often contain code snippets — the agent should still
        # see the content as plain text, just without backticks.
        md = "```python\nprint('hi')\n```"
        out = strip_markdown_preserving_urls(md)
        assert "print('hi')" in out
        assert "```" not in out

    def test_split_short_returns_single_chunk(self):
        assert split_for_line("hi") == ["hi"]

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
# 7. Send routing (reply -> push fallback, batching, system-bypass)
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

    def test_system_bypass_recognized(self):
        assert _is_system_bypass("⚡ Interrupting current run")
        assert _is_system_bypass("⏳ Queued — agent is busy")
        assert _is_system_bypass("⏩ Steered toward new task")
        assert not _is_system_bypass("Hello world")
        assert not _is_system_bypass("")

    def test_send_uses_reply_when_token_present(self, adapter):
        import time as _time
        adapter._reply_tokens["Uchat"] = ("rt-token", _time.time() + 30)
        result = asyncio.run(adapter.send("Uchat", "hello"))
        assert result.success
        adapter._client.reply.assert_called_once()
        adapter._client.push.assert_not_called()
        # Token consumed (single-use)
        assert "Uchat" not in adapter._reply_tokens

    def test_send_falls_back_to_push_when_no_token(self, adapter):
        result = asyncio.run(adapter.send("Uchat", "hello"))
        assert result.success
        adapter._client.push.assert_called_once()
        adapter._client.reply.assert_not_called()

    def test_send_falls_back_to_push_when_reply_fails(self, adapter):
        import time as _time
        adapter._reply_tokens["Uchat"] = ("rt-token", _time.time() + 30)
        adapter._client.reply.side_effect = RuntimeError("expired")
        result = asyncio.run(adapter.send("Uchat", "hello"))
        assert result.success
        adapter._client.reply.assert_called_once()
        adapter._client.push.assert_called_once()

    def test_send_returns_failure_when_push_fails(self, adapter):
        adapter._client.push.side_effect = RuntimeError("network")
        result = asyncio.run(adapter.send("Uchat", "hello"))
        assert not result.success
        assert "network" in result.error

    def test_send_pending_button_caches_response(self, adapter):
        # Simulate that the slow-LLM postback button has fired.
        rid = adapter._cache.register_pending("Uchat")
        adapter._pending_buttons["Uchat"] = rid
        result = asyncio.run(adapter.send("Uchat", "the answer"))
        assert result.success
        # Response must have been cached, not pushed/replied.
        adapter._client.reply.assert_not_called()
        adapter._client.push.assert_not_called()
        assert adapter._cache.get(rid).state is State.READY
        assert adapter._cache.get(rid).payload == "the answer"

    def test_send_system_bypass_skips_postback_cache(self, adapter):
        # Even with a pending button, system busy-acks must surface visibly.
        rid = adapter._cache.register_pending("Uchat")
        adapter._pending_buttons["Uchat"] = rid
        result = asyncio.run(adapter.send("Uchat", "⚡ Interrupting current run"))
        assert result.success
        # Bypass goes through push (no reply token stored)
        adapter._client.push.assert_called_once()
        # And the cache entry is unchanged (still PENDING for the eventual answer)
        assert adapter._cache.get(rid).state is State.PENDING

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
# 8. Register() metadata + plugin entry points
# ---------------------------------------------------------------------------

class TestRegister:

    class _FakeCtx:
        def __init__(self):
            self.kwargs = None

        def register_platform(self, **kw):
            self.kwargs = kw

    def test_register_calls_register_platform(self):
        ctx = self._FakeCtx()
        register(ctx)
        assert ctx.kwargs is not None
        assert ctx.kwargs["name"] == "line"
        assert ctx.kwargs["label"] == "LINE"

    def test_register_advertises_required_env(self):
        ctx = self._FakeCtx()
        register(ctx)
        assert set(ctx.kwargs["required_env"]) == {
            "LINE_CHANNEL_ACCESS_TOKEN",
            "LINE_CHANNEL_SECRET",
        }

    def test_register_wires_allowlist_envs(self):
        ctx = self._FakeCtx()
        register(ctx)
        assert ctx.kwargs["allowed_users_env"] == "LINE_ALLOWED_USERS"
        assert ctx.kwargs["allow_all_env"] == "LINE_ALLOW_ALL_USERS"

    def test_register_wires_cron_home_channel(self):
        ctx = self._FakeCtx()
        register(ctx)
        assert ctx.kwargs["cron_deliver_env_var"] == "LINE_HOME_CHANNEL"

    def test_register_provides_standalone_sender(self):
        ctx = self._FakeCtx()
        register(ctx)
        assert callable(ctx.kwargs["standalone_sender_fn"])

    def test_register_provides_env_enablement(self):
        ctx = self._FakeCtx()
        register(ctx)
        assert callable(ctx.kwargs["env_enablement_fn"])

    def test_register_factory_yields_line_adapter(self):
        ctx = self._FakeCtx()
        register(ctx)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True, extra={
            "channel_access_token": "tok",
            "channel_secret": "sec",
        })
        ad = ctx.kwargs["adapter_factory"](cfg)
        assert isinstance(ad, LineAdapter)

    def test_max_message_length_below_line_per_bubble_limit(self):
        ctx = self._FakeCtx()
        register(ctx)
        # LINE per-bubble limit is 5000; we register 4500 to leave headroom.
        assert ctx.kwargs["max_message_length"] <= 5000


class TestEnvEnablement:

    def test_returns_none_without_credentials(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        assert _env_enablement() is None

    def test_returns_dict_with_credentials(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "tok")
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "sec")
        assert _env_enablement() == {}

    def test_seeds_port_from_env(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "tok")
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "sec")
        monkeypatch.setenv("LINE_PORT", "8080")
        assert _env_enablement() == {"port": 8080}

    def test_seeds_public_url(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "tok")
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "sec")
        monkeypatch.setenv("LINE_PUBLIC_URL", "https://my-tunnel.example.com")
        result = _env_enablement()
        assert result["public_url"] == "https://my-tunnel.example.com"


class TestStandaloneSend:

    def test_missing_token_returns_error(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True, extra={})
        result = asyncio.run(_standalone_send(cfg, "Uchat", "hi"))
        assert "error" in result

    def test_missing_chat_id_returns_error(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "tok")
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True, extra={})
        result = asyncio.run(_standalone_send(cfg, "", "hi"))
        assert "error" in result

    def test_pushes_via_client_when_credentials_present(self, monkeypatch):
        from gateway.config import PlatformConfig

        push_calls = []

        class _FakeClient:
            def __init__(self, *a, **kw):
                pass

            async def push(self, chat_id, messages):
                push_calls.append((chat_id, messages))

        monkeypatch.setattr(_line, "_LineClient", _FakeClient)
        cfg = PlatformConfig(
            enabled=True,
            extra={"channel_access_token": "tok"},
        )
        result = asyncio.run(_standalone_send(cfg, "Uchat", "hello"))
        assert result.get("success") is True
        assert len(push_calls) == 1
        assert push_calls[0][0] == "Uchat"
        # Message wraps as text bubble
        assert push_calls[0][1][0]["type"] == "text"


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

    def test_text_truncated_to_160(self):
        long = "x" * 200
        msg = build_postback_button_message(long, "Tap", "rid")
        assert len(msg["template"]["text"]) <= 160

    def test_alt_text_truncated_to_400(self):
        long = "x" * 500
        msg = build_postback_button_message(long, "Tap", "rid")
        assert len(msg["altText"]) <= 400


class TestCheckRequirements:

    def test_rejects_without_token(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "s")
        assert not check_requirements()

    def test_rejects_without_secret(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "t")
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        assert not check_requirements()


class TestValidateConfig:

    def test_validates_from_extra(self):
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(
            enabled=True,
            extra={"channel_access_token": "t", "channel_secret": "s"},
        )
        assert validate_config(cfg)

    def test_rejects_empty_config(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True, extra={})
        assert not validate_config(cfg)


class TestAdapterInit:

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
        assert ad.webhook_port == 7777
        assert ad.public_base_url == "https://x.example.com"
        assert ad.allowed_users == {"U1", "U2"}

    def test_env_overrides_extra(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "env-tok")
        monkeypatch.setenv("LINE_PORT", "1234")
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(
            enabled=True,
            extra={"channel_access_token": "extra-tok", "channel_secret": "s", "port": 5555},
        )
        ad = LineAdapter(cfg)
        assert ad.channel_access_token == "env-tok"
        assert ad.webhook_port == 1234

    def test_csv_allowlist_parsed(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "t")
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "s")
        monkeypatch.setenv("LINE_ALLOWED_USERS", "U1, U2,U3")
        monkeypatch.setenv("LINE_ALLOWED_GROUPS", "C1")
        from gateway.config import PlatformConfig
        ad = LineAdapter(PlatformConfig(enabled=True))
        assert ad.allowed_users == {"U1", "U2", "U3"}
        assert ad.allowed_groups == {"C1"}

    def test_get_chat_info_infers_type_from_prefix(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "t")
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "s")
        from gateway.config import PlatformConfig
        ad = LineAdapter(PlatformConfig(enabled=True))
        assert asyncio.run(ad.get_chat_info("U123"))["type"] == "dm"
        assert asyncio.run(ad.get_chat_info("C123"))["type"] == "group"
        assert asyncio.run(ad.get_chat_info("R123"))["type"] == "channel"
