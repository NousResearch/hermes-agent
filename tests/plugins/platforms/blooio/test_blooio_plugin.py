"""Tests for the Blooio (iMessage via Blooio) platform plugin — v4 + OAuth.

Covers:

1. webhook signature verification (Stripe-style HMAC-SHA256) + tamper/replay
2. v4 inbound chat resolution (chat_id / group_id / sender) + allowlist gating
3. inbound dedup on message_id
4. Markdown stripping (iMessage renders plain text)
5. outbound send routing: reply into chat_… vs. addressed POST /messages,
   chunking, sent-id tracking
6. reaction normalization + agent-facing add_reaction targeting
7. v4 typed-envelope dispatch + inbound reaction routing (own-message gate)
8. register() metadata (+ CLI command) / standalone_send / env_enablement
9. OAuth: PKCE generation, auth resolution precedence, org header,
   access-token refresh
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

import plugins.platforms.blooio.adapter as _blooio
import plugins.platforms.blooio.auth as _auth

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
    # No stored OAuth tokens by default (avoid reading a real ~/.hermes).
    monkeypatch.setattr(_auth, "_load_tokens", lambda: None)
    yield


# ---------------------------------------------------------------------------
# 1. Signature verification
# ---------------------------------------------------------------------------

class TestSignature:
    SECRET = "whsec_deadbeef"

    def test_valid_signature_passes(self):
        body = b'{"type":"message.received"}'
        ts = int(time.time())
        assert verify_blooio_signature(body, _header(body, self.SECRET, ts), self.SECRET)

    def test_tampered_body_fails(self):
        body = b'{"type":"message.received"}'
        ts = int(time.time())
        header = _header(body, self.SECRET, ts)
        assert not verify_blooio_signature(b'{"type":"tampered"}', header, self.SECRET)

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
# 2. v4 chat resolution + allowlist gating
# ---------------------------------------------------------------------------

class TestResolveAndAllow:
    def test_resolve_dm(self, cfg):
        a = BlooioAdapter(cfg)
        chat_id, chat_type, user_id, group_id = a._resolve_chat(
            {"chat_id": "chat_abc", "sender": "+15551234567"}
        )
        assert (chat_id, chat_type, user_id, group_id) == (
            "chat_abc", "dm", "+15551234567", "",
        )

    def test_resolve_dm_contact_identifier(self, cfg):
        a = BlooioAdapter(cfg)
        _, chat_type, user_id, _ = a._resolve_chat(
            {"chat_id": "chat_abc", "contact": {"identifier": "user@example.com"}}
        )
        assert chat_type == "dm" and user_id == "user@example.com"

    def test_resolve_group(self, cfg):
        a = BlooioAdapter(cfg)
        chat_id, chat_type, user_id, group_id = a._resolve_chat(
            {"chat_id": "chat_g", "group_id": "grp_abc", "sender": "+15559999999"}
        )
        assert (chat_id, chat_type, user_id, group_id) == (
            "chat_g", "group", "+15559999999", "grp_abc",
        )

    def test_allowlist_dm(self, cfg, monkeypatch):
        monkeypatch.setenv("BLOOIO_ALLOWED_USERS", "+15551234567")
        a = BlooioAdapter(cfg)
        assert a._is_allowed("dm", "", "+15551234567")
        assert not a._is_allowed("dm", "", "+19998887777")

    def test_allow_all(self, cfg, monkeypatch):
        monkeypatch.setenv("BLOOIO_ALLOW_ALL_USERS", "true")
        a = BlooioAdapter(cfg)
        assert a._is_allowed("dm", "", "+1")
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
    async def test_reply_into_chat_uses_chat_route(self, cfg):
        a = BlooioAdapter(cfg)
        a._client = MagicMock()
        a._client.send_to_chat = AsyncMock(return_value={"id": "msg_1", "status": "queued"})
        a._client.send_agnostic = AsyncMock()
        res = await a.send("chat_abc", "hello world")
        assert res.success and res.message_id == "msg_1"
        chat_id, body = a._client.send_to_chat.call_args[0]
        assert chat_id == "chat_abc" and body["text"] == "hello world"
        a._client.send_agnostic.assert_not_called()
        assert "msg_1" in a._sent_message_ids

    async def test_addressed_send_uses_messages_route(self, cfg, monkeypatch):
        monkeypatch.setenv("BLOOIO_CHANNEL", "ch_xyz")
        a = BlooioAdapter(cfg)
        a._client = MagicMock()
        a._client.send_to_chat = AsyncMock()
        a._client.send_agnostic = AsyncMock(return_value={"id": "m"})
        await a.send("+15551234567", "hi")
        (body,) = a._client.send_agnostic.call_args[0]
        assert body["to"] == "+15551234567" and body["from"] == "ch_xyz"
        a._client.send_to_chat.assert_not_called()

    async def test_reply_to_is_message_id(self, cfg):
        a = BlooioAdapter(cfg)
        a._client = MagicMock()
        a._client.send_to_chat = AsyncMock(return_value={"id": "m"})
        await a.send("chat_abc", "yo", reply_to="msg_prev")
        _, body = a._client.send_to_chat.call_args[0]
        assert body["reply_to"] == "msg_prev"

    async def test_send_long_text_chunks_sequentially(self, cfg):
        a = BlooioAdapter(cfg)
        a._client = MagicMock()
        a._client.send_to_chat = AsyncMock(return_value={"id": "b"})
        res = await a.send("chat_abc", "x " * MAX_TEXT_LENGTH)
        assert a._client.send_to_chat.await_count >= 2
        assert res.success and res.message_id == "b"

    async def test_send_failure_returns_error(self, cfg):
        a = BlooioAdapter(cfg)
        a._client = MagicMock()
        a._client.send_to_chat = AsyncMock(side_effect=RuntimeError("boom"))
        res = await a.send("chat_abc", "hi")
        assert not res.success and "boom" in res.error

    async def test_remote_image_url_passthrough(self, cfg):
        a = BlooioAdapter(cfg)
        a._client = MagicMock()
        a._client.send_to_chat = AsyncMock(return_value={"id": "m"})
        await a.send_image("chat_abc", "https://cdn.example.com/cat.jpg", caption="cat")
        _, body = a._client.send_to_chat.call_args[0]
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
        a._last_inbound_by_chat["chat_abc"] = "msg_in"
        res = await a.add_reaction("chat_abc", "love")
        assert res["success"] and res["message_id"] == "msg_in"
        args = a._client.react.call_args[0]
        assert args[1] == "msg_in" and args[2] == "+love"

    async def test_add_reaction_without_target_errors(self, cfg):
        a = BlooioAdapter(cfg)
        a._client = MagicMock()
        a._client.react = AsyncMock(return_value={})
        res = await a.add_reaction("chat_abc", "👍")
        assert res["success"] is False
        a._client.react.assert_not_called()


@pytest.mark.asyncio
class TestInboundReaction:
    async def test_reaction_on_own_message_is_dispatched(self, cfg):
        a = BlooioAdapter(cfg)
        a._record_sent_message("msg_out")
        a.handle_message = AsyncMock()
        await a._handle_inbound_reaction(
            {
                "message_id": "msg_out",
                "chat_id": "chat_abc",
                "sender": "+15551234567",
                "reaction": "love",
                "action": "add",
                "timestamp": 1,
            }
        )
        assert a.handle_message.await_count == 1
        evt = a.handle_message.await_args[0][0]
        assert evt.text == "reaction:add:love"
        assert evt.reply_to_is_own_message is True

    async def test_reaction_on_foreign_message_ignored(self, cfg):
        a = BlooioAdapter(cfg)
        a.handle_message = AsyncMock()
        await a._handle_inbound_reaction(
            {
                "message_id": "not_ours",
                "chat_id": "chat_abc",
                "sender": "+15551234567",
                "reaction": "love",
                "action": "add",
                "timestamp": 1,
            }
        )
        assert a.handle_message.await_count == 0


# ---------------------------------------------------------------------------
# 7. Inbound dispatch (v4 typed envelope)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestInboundMessage:
    async def test_v4_envelope_dispatched(self, cfg, monkeypatch):
        monkeypatch.setenv("BLOOIO_ALLOW_ALL_USERS", "true")
        a = BlooioAdapter(cfg)
        a.handle_message = AsyncMock()
        await a._dispatch_event(
            {
                "type": "message.received",
                "created_at": 1000,
                "organization_id": "org_1",
                "data": {
                    "message_id": "m1",
                    "chat_id": "chat_abc",
                    "sender": "+15551234567",
                    "text": "hey",
                },
            }
        )
        assert a.handle_message.await_count == 1
        evt = a.handle_message.await_args[0][0]
        assert evt.text == "hey"
        assert evt.source.chat_id == "chat_abc"

    async def test_unauthorized_dropped(self, cfg):
        a = BlooioAdapter(cfg)  # no allowlist → nothing allowed
        a.handle_message = AsyncMock()
        await a._handle_inbound_message(
            {"message_id": "m", "chat_id": "chat_x", "sender": "+1", "text": "x"}
        )
        assert a.handle_message.await_count == 0

    async def test_duplicate_dropped(self, cfg, monkeypatch):
        monkeypatch.setenv("BLOOIO_ALLOW_ALL_USERS", "true")
        a = BlooioAdapter(cfg)
        a.handle_message = AsyncMock()
        event = {"message_id": "dup", "chat_id": "chat_x", "sender": "+1", "text": "x"}
        await a._handle_inbound_message(dict(event))
        await a._handle_inbound_message(dict(event))
        assert a.handle_message.await_count == 1

    async def test_group_require_mention_gate(self, cfg, monkeypatch):
        monkeypatch.setenv("BLOOIO_ALLOW_ALL_USERS", "true")
        monkeypatch.setenv("BLOOIO_REQUIRE_MENTION", "true")
        a = BlooioAdapter(cfg)
        a.handle_message = AsyncMock()
        await a._handle_inbound_message(
            {
                "message_id": "g1",
                "chat_id": "chat_g",
                "group_id": "grp_x",
                "sender": "+1",
                "text": "just chatting",
            }
        )
        assert a.handle_message.await_count == 0
        await a._handle_inbound_message(
            {
                "message_id": "g2",
                "chat_id": "chat_g",
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
        cli = {}

        class Ctx:
            def register_platform(self, **kw):
                captured.update(kw)

            def register_cli_command(self, **kw):
                cli.update(kw)

        register(Ctx())
        assert captured["name"] == "blooio"
        assert captured["required_env"] == []
        assert captured["allowed_users_env"] == "BLOOIO_ALLOWED_USERS"
        assert captured["cron_deliver_env_var"] == "BLOOIO_HOME_CHANNEL"
        assert captured["pii_safe"] is True
        assert captured["max_message_length"] == MAX_TEXT_LENGTH
        assert callable(captured["standalone_sender_fn"])
        # `hermes blooio ...` CLI is wired up.
        assert cli["name"] == "blooio"
        assert callable(cli["setup_fn"]) and callable(cli["handler_fn"])

    def test_validate_config_api_key(self, cfg, monkeypatch):
        monkeypatch.setenv("BLOOIO_API_KEY", "api_x")
        assert validate_config(cfg)

    def test_validate_config_oauth_token(self, cfg, monkeypatch):
        monkeypatch.setattr(_auth, "_load_tokens", lambda: {"access_token": "t"})
        assert validate_config(cfg)

    def test_env_enablement(self, monkeypatch):
        assert _env_enablement() is None
        monkeypatch.setenv("BLOOIO_API_KEY", "api_x")
        monkeypatch.setenv("BLOOIO_PUBLIC_URL", "https://x.example.com")
        monkeypatch.setenv("BLOOIO_HOME_CHANNEL", "chat_abc")
        seeded = _env_enablement()
        assert seeded["public_url"] == "https://x.example.com"
        assert seeded["home_channel"]["chat_id"] == "chat_abc"


@pytest.mark.asyncio
class TestStandaloneSend:
    async def test_missing_auth(self, monkeypatch):
        pconfig = MagicMock()
        pconfig.extra = {}
        res = await _standalone_send(pconfig, "chat_abc", "hi")
        assert "error" in res

    async def test_sends_into_chat(self, monkeypatch):
        monkeypatch.setenv("BLOOIO_API_KEY", "api_x")
        pconfig = MagicMock()
        pconfig.extra = {}
        sent = {}

        async def fake_send_to_chat(self, chat_id, body):
            sent["chat_id"] = chat_id
            sent["body"] = body
            return {"id": "m1"}

        monkeypatch.setattr(_blooio._BlooioClient, "send_to_chat", fake_send_to_chat)
        res = await _standalone_send(pconfig, "chat_abc", "**hi** there")
        assert res["success"] and res["message_id"] == "m1"
        assert sent["chat_id"] == "chat_abc"
        assert "**" not in sent["body"]["text"]

    async def test_sends_addressed(self, monkeypatch):
        monkeypatch.setenv("BLOOIO_API_KEY", "api_x")
        monkeypatch.setenv("BLOOIO_CHANNEL", "ch_z")
        pconfig = MagicMock()
        pconfig.extra = {}
        sent = {}

        async def fake_send_agnostic(self, body):
            sent["body"] = body
            return {"id": "m2"}

        monkeypatch.setattr(_blooio._BlooioClient, "send_agnostic", fake_send_agnostic)
        res = await _standalone_send(pconfig, "+15551234567", "hello")
        assert res["success"] and res["message_id"] == "m2"
        assert sent["body"]["to"] == "+15551234567" and sent["body"]["from"] == "ch_z"


# ---------------------------------------------------------------------------
# 9. OAuth / auth
# ---------------------------------------------------------------------------

class TestAuthResolution:
    def test_pkce_shape(self):
        verifier, challenge = _auth.generate_pkce()
        assert 43 <= len(verifier) <= 128
        assert "=" not in challenge and "+" not in challenge and "/" not in challenge

    def test_api_key_precedence(self, cfg, monkeypatch):
        monkeypatch.setenv("BLOOIO_API_KEY", "api_x")
        monkeypatch.setenv("BLOOIO_ORG_ID", "org_9")
        au = _auth.resolve_auth(cfg)
        assert au.mode == "api_key" and au.organization_id == "org_9"

    def test_oauth_when_no_key(self, cfg, monkeypatch):
        monkeypatch.setattr(
            _auth, "_load_tokens",
            lambda: {"access_token": "t", "organization_id": "org_5"},
        )
        au = _auth.resolve_auth(cfg)
        assert au.mode == "oauth" and au.organization_id == "org_5"

    def test_no_credentials_returns_none(self, cfg):
        assert _auth.resolve_auth(cfg) is None


@pytest.mark.asyncio
class TestAuthRuntime:
    async def test_org_header_present(self):
        au = _auth.BlooioAuth("api_key", api_key="k", organization_id="org_1")
        client = _blooio._BlooioClient(au, "https://api.blooio.com/v4")
        headers = await client._headers()
        assert headers["Authorization"] == "Bearer k"
        assert headers["X-Organization-Id"] == "org_1"

    async def test_org_header_absent_when_unscoped(self):
        au = _auth.BlooioAuth("api_key", api_key="k")
        client = _blooio._BlooioClient(au, "https://api.blooio.com/v4")
        headers = await client._headers()
        assert "X-Organization-Id" not in headers

    async def test_access_token_refresh(self, monkeypatch):
        stored = {}
        expired = {
            "access_token": "old",
            "refresh_token": "r1",
            "expires_at": time.time() - 10,  # already expired
            "code_verifier": "v1",
            "organization_id": "org_1",
        }
        monkeypatch.setattr(_auth, "_load_tokens", lambda: dict(expired))
        monkeypatch.setattr(_auth, "_store_tokens", lambda rec: stored.update(rec))

        async def fake_post_token(form):
            assert form["grant_type"] == "refresh_token"
            assert form["refresh_token"] == "r1"
            assert form["code_verifier"] == "v1"  # public-client refresh needs it
            return {"access_token": "new", "refresh_token": "r2", "expires_in": 3600}

        monkeypatch.setattr(_auth, "_post_token", fake_post_token)
        au = _auth.BlooioAuth("oauth", organization_id="org_1")
        token = await au.bearer()
        assert token == "new"
        assert stored["access_token"] == "new" and stored["refresh_token"] == "r2"

    async def test_valid_access_token_not_refreshed(self, monkeypatch):
        calls = {"post": 0}
        fresh = {"access_token": "good", "expires_at": time.time() + 3600}
        monkeypatch.setattr(_auth, "_load_tokens", lambda: dict(fresh))

        async def fake_post_token(form):
            calls["post"] += 1
            return {}

        monkeypatch.setattr(_auth, "_post_token", fake_post_token)
        au = _auth.BlooioAuth("oauth")
        assert await au.bearer() == "good"
        assert calls["post"] == 0
