"""Tests for the LINE platform adapter plugin."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter


# Load plugins/platforms/line/adapter.py under a unique module name
# (plugin_adapter_line) so it cannot collide with other plugin adapters
# loaded by sibling tests in the same xdist worker.
_line_mod = load_plugin_adapter("line")

LineAdapter = _line_mod.LineAdapter
_validate_line_signature = _line_mod._validate_line_signature
_resolve_chat = _line_mod._resolve_chat
validate_config = _line_mod.validate_config
check_requirements = _line_mod.check_requirements
register = _line_mod.register


# ── Signature validation ──────────────────────────────────────────────────────


class TestSignatureValidation:

    def _sign(self, body: bytes, secret: str) -> str:
        return base64.b64encode(
            hmac.new(secret.encode(), body, hashlib.sha256).digest()
        ).decode()

    def test_valid_signature_accepted(self):
        body = b'{"events":[]}'
        secret = "abc123"
        sig = self._sign(body, secret)
        assert _validate_line_signature(body, sig, secret) is True

    def test_wrong_secret_rejected(self):
        body = b'{"events":[]}'
        sig = self._sign(body, "abc123")
        assert _validate_line_signature(body, sig, "wrong-secret") is False

    def test_tampered_body_rejected(self):
        secret = "abc123"
        sig = self._sign(b'{"events":[]}', secret)
        assert (
            _validate_line_signature(b'{"events":[{"foo":1}]}', sig, secret) is False
        )

    def test_missing_signature_rejected(self):
        assert _validate_line_signature(b"x", "", "secret") is False

    def test_missing_secret_rejected(self):
        assert _validate_line_signature(b"x", "anysig", "") is False

    def test_garbage_signature_rejected(self):
        # Base64-decoded length mismatch must not raise.
        assert (
            _validate_line_signature(b"x", "definitely-not-base64-***", "secret")
            is False
        )


# ── Source / chat-id resolution ───────────────────────────────────────────────


class TestResolveChat:

    def test_user_source(self):
        assert _resolve_chat({"type": "user", "userId": "U123"}) == ("U123", "dm")

    def test_group_source(self):
        assert _resolve_chat({"type": "group", "groupId": "C456"}) == ("C456", "group")

    def test_room_source(self):
        # Multi-person rooms — LINE distinguishes them from groups, but for
        # session routing they behave like groups.
        assert _resolve_chat({"type": "room", "roomId": "R789"}) == ("R789", "group")

    def test_unknown_source_returns_empty(self):
        assert _resolve_chat({"type": "wat"}) == ("", "dm")

    def test_missing_id_returns_empty(self):
        assert _resolve_chat({"type": "group"}) == ("", "group")


# ── validate_config ───────────────────────────────────────────────────────────


class TestValidateConfig:

    def test_valid_with_extra(self):
        cfg = MagicMock()
        cfg.extra = {"channel_access_token": "tok", "channel_secret": "sec"}
        assert validate_config(cfg) is True

    def test_valid_with_env(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "tok")
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "sec")
        cfg = MagicMock()
        cfg.extra = {}
        assert validate_config(cfg) is True

    def test_missing_token_invalid(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        cfg = MagicMock()
        cfg.extra = {"channel_secret": "sec"}
        assert validate_config(cfg) is False

    def test_missing_secret_invalid(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        cfg = MagicMock()
        cfg.extra = {"channel_access_token": "tok"}
        assert validate_config(cfg) is False


# ── Reply token TTL ───────────────────────────────────────────────────────────


def _make_adapter(monkeypatch) -> "LineAdapter":
    monkeypatch.delenv("LINE_PORT", raising=False)
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "sec")
    from gateway.config import PlatformConfig

    return LineAdapter(PlatformConfig(enabled=True))


class TestReplyTokenLifecycle:

    def test_unknown_chat_returns_none(self, monkeypatch):
        a = _make_adapter(monkeypatch)
        assert a._consume_reply_token("Uunknown") is None

    def test_fresh_token_consumed_once(self, monkeypatch):
        a = _make_adapter(monkeypatch)
        a._reply_tokens["Uchat"] = ("rt-fresh", time.time())
        assert a._consume_reply_token("Uchat") == "rt-fresh"
        # Reply tokens are single-use — second call gets None.
        assert a._consume_reply_token("Uchat") is None

    def test_expired_token_dropped(self, monkeypatch):
        a = _make_adapter(monkeypatch)
        # Token captured 5 minutes ago — well past LINE's 60s validity
        # window and our internal 50s cap.
        a._reply_tokens["Uchat"] = ("rt-stale", time.time() - 300)
        assert a._consume_reply_token("Uchat") is None
        # Stale entry was popped, not left behind.
        assert "Uchat" not in a._reply_tokens


# ── Outbound: reply with push fallback ────────────────────────────────────────


class TestSendMessages:

    @pytest.mark.asyncio
    async def test_reply_used_when_token_fresh(self, monkeypatch):
        a = _make_adapter(monkeypatch)
        a._reply_tokens["Uchat"] = ("rt-fresh", time.time())
        a._post = AsyncMock(return_value=(True, ""))

        result = await a._send_messages(
            "Uchat", [{"type": "text", "text": "hi"}]
        )

        assert result.success is True
        a._post.assert_awaited_once()
        path, payload = a._post.call_args.args
        assert path == "/message/reply"
        assert payload["replyToken"] == "rt-fresh"
        assert payload["messages"] == [{"type": "text", "text": "hi"}]

    @pytest.mark.asyncio
    async def test_push_used_when_no_token(self, monkeypatch):
        a = _make_adapter(monkeypatch)
        a._post = AsyncMock(return_value=(True, ""))

        result = await a._send_messages(
            "Uchat", [{"type": "text", "text": "hi"}]
        )

        assert result.success is True
        path, payload = a._post.call_args.args
        assert path == "/message/push"
        assert payload["to"] == "Uchat"

    @pytest.mark.asyncio
    async def test_reply_failure_falls_back_to_push(self, monkeypatch):
        a = _make_adapter(monkeypatch)
        a._reply_tokens["Uchat"] = ("rt-already-used", time.time())
        # First call (reply) fails; second call (push) succeeds.
        a._post = AsyncMock(
            side_effect=[(False, "HTTP 400: Invalid reply token"), (True, "")]
        )

        result = await a._send_messages(
            "Uchat", [{"type": "text", "text": "hi"}]
        )

        assert result.success is True
        assert a._post.await_count == 2
        assert a._post.call_args_list[0].args[0] == "/message/reply"
        assert a._post.call_args_list[1].args[0] == "/message/push"

    @pytest.mark.asyncio
    async def test_push_failure_returns_retryable_error(self, monkeypatch):
        a = _make_adapter(monkeypatch)
        a._post = AsyncMock(return_value=(False, "HTTP 500: oh no"))

        result = await a._send_messages(
            "Uchat", [{"type": "text", "text": "hi"}]
        )

        assert result.success is False
        assert result.retryable is True
        assert "500" in (result.error or "")

    @pytest.mark.asyncio
    async def test_messages_batched_in_groups_of_five(self, monkeypatch):
        a = _make_adapter(monkeypatch)
        a._post = AsyncMock(return_value=(True, ""))

        msgs = [{"type": "text", "text": str(i)} for i in range(12)]
        result = await a._send_messages("Uchat", msgs)

        assert result.success is True
        # 12 messages → 3 batches (5 + 5 + 2).
        assert a._post.await_count == 3
        sizes = [len(call.args[1]["messages"]) for call in a._post.call_args_list]
        assert sizes == [5, 5, 2]


# ── send_typing (loading animation API) ───────────────────────────────────────


class TestSendTyping:

    @pytest.mark.asyncio
    async def test_typing_calls_loading_animation_for_user(self, monkeypatch):
        a = _make_adapter(monkeypatch)
        a._http = MagicMock()  # presence-only; _post is mocked below
        a._post = AsyncMock(return_value=(True, ""))

        await a.send_typing("Uchat")

        a._post.assert_awaited_once()
        path, payload = a._post.call_args.args
        assert path == "/chat/loading/start"
        assert payload["chatId"] == "Uchat"
        assert payload["loadingSeconds"] in range(5, 65, 5)

    @pytest.mark.asyncio
    async def test_typing_skipped_for_group_chat(self, monkeypatch):
        # LINE's loading animation endpoint only works for 1:1 user chats —
        # calling it with a group/room ID would 400.  We skip it client-side.
        a = _make_adapter(monkeypatch)
        a._http = MagicMock()
        a._post = AsyncMock(return_value=(True, ""))

        await a.send_typing("Cgroup-id")
        await a.send_typing("Rroom-id")

        a._post.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_typing_swallows_failures(self, monkeypatch):
        # A 400 from LINE shouldn't break the inbound dispatch path.
        a = _make_adapter(monkeypatch)
        a._http = MagicMock()
        a._post = AsyncMock(return_value=(False, "HTTP 400: bad"))

        await a.send_typing("Uchat")  # must not raise


# ── register() smoke test ─────────────────────────────────────────────────────


def test_register_emits_expected_metadata():
    """register() should call ctx.register_platform with the right keys."""
    captured = {}

    def fake_register(**kwargs):
        captured.update(kwargs)

    ctx = MagicMock()
    ctx.register_platform.side_effect = fake_register
    register(ctx)

    assert captured["name"] == "line"
    assert captured["label"] == "LINE"
    assert "LINE_CHANNEL_ACCESS_TOKEN" in captured["required_env"]
    assert "LINE_CHANNEL_SECRET" in captured["required_env"]
    assert captured["allowed_users_env"] == "LINE_ALLOWED_USERS"
    assert captured["allow_all_env"] == "LINE_ALLOW_ALL_USERS"
