"""Tests for the Blooio (iMessage) platform adapter plugin.

Covers:

1. webhook signature verification (Stripe-style HMAC-SHA256) + tamper/replay rejection
2. inbound chat-id resolution (1:1 vs group) and allowlist gating
3. inbound dedup on message_id
4. Markdown stripping (iMessage renders plain text)
5. outbound send routing: text-array chunking, from-number inference, sent-id tracking
6. reaction normalization + agent-facing add_reaction default targeting
7. inbound reaction routing (only surfaced on the bot's own messages)
8. register() metadata + standalone_send + env_enablement shapes
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_blooio = load_plugin_adapter("blooio")

verify_blooio_signature = _blooio.verify_blooio_signature
BlooioAdapter = _blooio.BlooioAdapter
_MessageDeduplicator = _blooio._MessageDeduplicator
register = _blooio.register
check_requirements = _blooio.check_requirements
validate_config = _blooio.validate_config
_env_enablement = _blooio._env_enablement
_standalone_send = _blooio._standalone_send
MAX_TEXT_LENGTH = _blooio.MAX_TEXT_LENGTH


def _sign(body: bytes, secret: str, ts: int) -> str:
    payload = f"{ts}.".encode() + body
    return hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


def _header(body: bytes, secret: str, ts: int) -> str:
    return f"t={ts},v1={_sign(body, secret, ts)}"


@pytest.fixture
def cfg():
    c = MagicMock()
    c.extra = {}
    return c


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("BLOOIO_"):
            monkeypatch.delenv(key, raising=False)
    yield


# ---------------------------------------------------------------------------
# 1. Signature verification
# ---------------------------------------------------------------------------

class TestSignature:
    SECRET = "whsec_deadbeef"

    def test_valid_signature_passes(self):
        body = b'{"event":"message.received"}'
        ts = int(time.time())
        assert verify_blooio_signature(body, _header(body, self.SECRET, ts), self.SECRET)

    def test_tampered_body_fails(self):
        body = b'{"event":"message.received"}'
        ts = int(time.time())
        header = _header(body, self.SECRET, ts)
        assert not verify_blooio_signature(b'{"event":"tampered"}', header, self.SECRET)

    def test_wrong_secret_fails(self):
        body = b'{"a":1}'
        ts = int(time.time())
        assert not verify_blooio_signature(body, _header(body, self.SECRET, ts), "whsec_other")

    def test_stale_timestamp_fails(self):
        body = b'{"a":1}'
        ts = int(time.time()) - 3600
        assert not verify_blooio_signature(body, _header(body, self.SECRET, ts), self.SECRET)

    def test_malformed_header_fails(self):
        body = b'{"a":1}'
        assert not verify_blooio_signature(body, "garbage", self.SECRET)
        assert not verify_blooio_signature(body, "", self.SECRET)
        assert not verify_blooio_signature(body, "t=123", self.SECRET)


# ---------------------------------------------------------------------------
# 2. Chat resolution + allowlist gating
# ---------------------------------------------------------------------------

class TestResolveAndAllow:
    def test_resolve_dm(self, cfg):
        a = BlooioAdapter(cfg)
        chat_id, chat_type, user_id = a._resolve_chat(
            {"is_group": False, "external_id": "+15551234567"}
        )
        assert (chat_id, chat_type, user_id) == ("+15551234567", "dm", "+15551234567")

    def test_resolve_group(self, cfg):
        a = BlooioAdapter(cfg)
        chat_id, chat_type, user_id = a._resolve_chat(
            {"is_group": True, "group_id": "grp_abc", "sender": "+15559999999"}
        )
        assert (chat_id, chat_type, user_id) == ("grp_abc", "group", "+15559999999")

    def test_allowlist_dm(self, cfg, monkeypatch):
        monkeypatch.setenv("BLOOIO_ALLOWED_USERS", "+15551234567")
        a = BlooioAdapter(cfg)
        assert a._is_allowed("dm", "+15551234567", "+15551234567")
        assert not a._is_allowed("dm", "+19998887777", "+19998887777")

    def test_allow_all(self, cfg, monkeypatch):
        monkeypatch.setenv("BLOOIO_ALLOW_ALL_USERS", "true")
        a = BlooioAdapter(cfg)
        assert a._is_allowed("dm", "+1", "+1")
        assert a._is_allowed("group", "grp_x", "+2")

    def test_group_allowlist(self, cfg, monkeypatch):
        monkeypatch.setenv("BLOOIO_ALLOWED_GROUPS", "grp_ok")
        a = BlooioAdapter(cfg)
        assert a._is_allowed("group", "grp_ok", "+1")
        assert not a._is_allowed("group", "grp_no", "+1")


# ---------------------------------------------------------------------------
# 3. Dedup
# ---------------------------------------------------------------------------

class TestDedup:
    def test_dedup(self):
        d = _MessageDeduplicator(max_size=100)
        assert not d.is_duplicate("m1")
        assert d.is_duplicate("m1")
        assert not d.is_duplicate("m2")

    def test_empty_never_duplicate(self):
        d = _MessageDeduplicator()
        assert not d.is_duplicate("")
        assert not d.is_duplicate("")


# ---------------------------------------------------------------------------
# 4. Markdown stripping
# ---------------------------------------------------------------------------

class TestFormat:
    def test_strips_markdown(self, cfg):
        a = BlooioAdapter(cfg)
        out = a.format_message("**bold** and `code`")
        assert "**" not in out
        assert "`" not in out


# ---------------------------------------------------------------------------
# 5. Outbound send routing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSend:
    async def test_send_single_text(self, cfg):
        a = BlooioAdapter(cfg)
        a._client = MagicMock()
        a._client.send_message = AsyncMock(return_value={"message_id": "msg_1", "status": "queued"})
        res = await a.send("+15551234567", "hello world")
        assert res.success and res.message_id == "msg_1"
        _, body = a._client.send_message.call_args[0]
        assert body["text"] == "hello world"
        # Sent id is tracked so reactions on our own message route back.
        assert "msg_1" in a._sent_message_ids

    async def test_send_uses_inferred_from_number(self, cfg):
        a = BlooioAdapter(cfg)
        a._client = MagicMock()
        a._client.send_message = AsyncMock(return_value={"message_id": "m"})
        a._reply_from_by_chat["+1555"] = "+15550000000"
        await a.send("+1555", "hi")
        _, body = a._client.send_message.call_args[0]
        assert body["from_number"] == "+15550000000"

    async def test_send_long_text_becomes_array(self, cfg):
        a = BlooioAdapter(cfg)
        a._client = MagicMock()
        a._client.send_message = AsyncMock(return_value={"message_ids": ["a", "b"]})
        res = await a.send("+1555", "x" * (MAX_TEXT_LENGTH * 2))
        _, body = a._client.send_message.call_args[0]
        assert isinstance(body["text"], list) and len(body["text"]) >= 2
        assert res.success and res.message_id == "b"

    async def test_send_failure_returns_error(self, cfg):
        a = BlooioAdapter(cfg)
        a._client = MagicMock()
        a._client.send_message = AsyncMock(side_effect=RuntimeError("boom"))
        res = await a.send("+1555", "hi")
        assert not res.success and "boom" in res.error

    async def test_remote_image_url_passthrough(self, cfg):
        a = BlooioAdapter(cfg)
        a._client = MagicMock()
        a._client.send_message = AsyncMock(return_value={"message_id": "m"})
        await a.send_image("+1555", "https://cdn.example.com/cat.jpg", caption="cat")
        _, body = a._client.send_message.call_args[0]
        assert body["attachments"] == ["https://cdn.example.com/cat.jpg"]
        assert body["text"] == "cat"


# ---------------------------------------------------------------------------
# 6. Reactions
# ---------------------------------------------------------------------------

class TestReactionNormalize:
    def test_normalize(self):
        assert BlooioAdapter._normalize_reaction("love") == "+love"
        assert BlooioAdapter._normalize_reaction("+👍") == "+👍"
        assert BlooioAdapter._normalize_reaction("-🔥") == "-🔥"


@pytest.mark.asyncio
class TestAddReaction:
    async def test_add_reaction_defaults_to_last_inbound(self, cfg):
        a = BlooioAdapter(cfg)
        a._client = MagicMock()
        a._client.react = AsyncMock(return_value={})
        a._last_inbound_by_chat["+1555"] = "msg_in"
        res = await a.add_reaction("+1555", "love")
        assert res["success"] and res["message_id"] == "msg_in"
        args = a._client.react.call_args
        assert args[0][1] == "msg_in"
        assert args[0][2] == "+love"

    async def test_add_reaction_relative_index_fallback(self, cfg):
        a = BlooioAdapter(cfg)
        a._client = MagicMock()
        a._client.react = AsyncMock(return_value={})
        res = await a.add_reaction("+1555", "👍")
        assert res["message_id"] == "-1"
        assert a._client.react.call_args.kwargs["direction"] == "inbound"


@pytest.mark.asyncio
class TestInboundReaction:
    async def test_reaction_on_own_message_is_dispatched(self, cfg):
        a = BlooioAdapter(cfg)
        a._record_sent_message("msg_out")
        a.handle_message = AsyncMock()
        await a._handle_inbound_reaction(
            {
                "event": "message.reaction",
                "message_id": "msg_out",
                "external_id": "+15551234567",
                "reaction": "love",
                "action": "added",
                "timestamp": 1,
            }
        )
        assert a.handle_message.await_count == 1
        evt = a.handle_message.await_args[0][0]
        assert evt.text == "reaction:added:love"
        assert evt.reply_to_is_own_message is True

    async def test_reaction_on_foreign_message_ignored(self, cfg):
        a = BlooioAdapter(cfg)
        a.handle_message = AsyncMock()
        await a._handle_inbound_reaction(
            {
                "event": "message.reaction",
                "message_id": "not_ours",
                "external_id": "+15551234567",
                "reaction": "love",
                "action": "added",
                "timestamp": 1,
            }
        )
        assert a.handle_message.await_count == 0


# ---------------------------------------------------------------------------
# 7. Inbound message dispatch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestInboundMessage:
    async def test_authorized_dm_dispatched(self, cfg, monkeypatch):
        monkeypatch.setenv("BLOOIO_ALLOW_ALL_USERS", "true")
        a = BlooioAdapter(cfg)
        a.handle_message = AsyncMock()
        await a._handle_inbound_message(
            {
                "event": "message.received",
                "message_id": "m1",
                "external_id": "+15551234567",
                "internal_id": "+15550000000",
                "text": "hey",
                "is_group": False,
            }
        )
        assert a.handle_message.await_count == 1
        evt = a.handle_message.await_args[0][0]
        assert evt.text == "hey"
        assert evt.source.chat_id == "+15551234567"
        # Reply-from is inferred from the number that received the message.
        assert a._reply_from_by_chat["+15551234567"] == "+15550000000"

    async def test_unauthorized_dropped(self, cfg):
        a = BlooioAdapter(cfg)  # no allowlist → nothing allowed
        a.handle_message = AsyncMock()
        await a._handle_inbound_message(
            {"event": "message.received", "message_id": "m", "external_id": "+1", "text": "x"}
        )
        assert a.handle_message.await_count == 0

    async def test_duplicate_dropped(self, cfg, monkeypatch):
        monkeypatch.setenv("BLOOIO_ALLOW_ALL_USERS", "true")
        a = BlooioAdapter(cfg)
        a.handle_message = AsyncMock()
        event = {
            "event": "message.received",
            "message_id": "dup",
            "external_id": "+1",
            "text": "x",
        }
        await a._handle_inbound_message(dict(event))
        await a._handle_inbound_message(dict(event))
        assert a.handle_message.await_count == 1

    async def test_group_require_mention_gate(self, cfg, monkeypatch):
        monkeypatch.setenv("BLOOIO_ALLOW_ALL_USERS", "true")
        monkeypatch.setenv("BLOOIO_REQUIRE_MENTION", "true")
        a = BlooioAdapter(cfg)
        a.handle_message = AsyncMock()
        # No wake word → dropped.
        await a._handle_inbound_message(
            {
                "event": "message.received",
                "message_id": "g1",
                "is_group": True,
                "group_id": "grp_x",
                "sender": "+1",
                "text": "just chatting",
            }
        )
        assert a.handle_message.await_count == 0
        # Wake word → dispatched, prefix stripped.
        await a._handle_inbound_message(
            {
                "event": "message.received",
                "message_id": "g2",
                "is_group": True,
                "group_id": "grp_x",
                "sender": "+1",
                "text": "hermes what's up",
            }
        )
        assert a.handle_message.await_count == 1
        assert a.handle_message.await_args[0][0].text == "what's up"


# ---------------------------------------------------------------------------
# 8. Registration + standalone + env
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_metadata(self):
        captured = {}

        class Ctx:
            def register_platform(self, **kw):
                captured.update(kw)

        register(Ctx())
        assert captured["name"] == "blooio"
        assert captured["required_env"] == ["BLOOIO_API_KEY"]
        assert captured["allowed_users_env"] == "BLOOIO_ALLOWED_USERS"
        assert captured["cron_deliver_env_var"] == "BLOOIO_HOME_CHANNEL"
        assert captured["pii_safe"] is True
        assert captured["max_message_length"] == MAX_TEXT_LENGTH
        assert callable(captured["standalone_sender_fn"])

    def test_validate_config_env(self, cfg, monkeypatch):
        monkeypatch.setenv("BLOOIO_API_KEY", "sk_x")
        assert validate_config(cfg)

    def test_env_enablement(self, monkeypatch):
        assert _env_enablement() is None
        monkeypatch.setenv("BLOOIO_API_KEY", "sk_x")
        monkeypatch.setenv("BLOOIO_PUBLIC_URL", "https://x.example.com")
        monkeypatch.setenv("BLOOIO_HOME_CHANNEL", "+15551234567")
        seeded = _env_enablement()
        assert seeded["public_url"] == "https://x.example.com"
        assert seeded["home_channel"]["chat_id"] == "+15551234567"


@pytest.mark.asyncio
class TestStandaloneSend:
    async def test_missing_key(self):
        pconfig = MagicMock()
        pconfig.extra = {}
        res = await _standalone_send(pconfig, "+1", "hi")
        assert "error" in res

    async def test_sends_text(self, monkeypatch):
        monkeypatch.setenv("BLOOIO_API_KEY", "sk_x")
        pconfig = MagicMock()
        pconfig.extra = {}
        sent = {}

        async def fake_send(self, chat_id, body):
            sent["chat_id"] = chat_id
            sent["body"] = body
            return {"message_id": "m1"}

        monkeypatch.setattr(_blooio._BlooioClient, "send_message", fake_send)
        res = await _standalone_send(pconfig, "+15551234567", "**hi** there")
        assert res["success"] and res["message_id"] == "m1"
        assert "**" not in sent["body"]["text"]
