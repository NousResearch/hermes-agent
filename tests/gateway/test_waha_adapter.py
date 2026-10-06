"""Tests for the WAHA (unofficial WhatsApp bridge) platform adapter plugin.

Covers the two load-bearing design decisions from the WAHA adapter brief: (1) LID-robust
identity resolution (WAHA sends both the opaque "<digits>@lid" form and the real phone-JID
form on every event; the adapter must always resolve to the phone form so chat identity is
stable across messages) and (2) multi-sender-one-account routing (one adapter instance, several
allowed senders resolving to DIFFERENT chat_ids so gateway.profile_routes can split them to
different Hermes profiles) -- plus webhook HMAC verification, allowlist/broadcast gating reused
from WhatsAppBehaviorMixin, and outbound send/chunking.

No change-detector tests: every assertion is a behavior contract (two pieces of data agreeing),
proven red on the unfixed code.
"""

import hashlib
import hmac
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig


def _make_adapter(**extra_overrides):
    from plugins.platforms.waha.adapter import WahaAdapter

    extra = {"base_url": "http://localhost:3000", "session": "default", **extra_overrides}
    env = {"WAHA_ALLOWED_USERS": "", "WAHA_API_KEY": "", "WAHA_HMAC_SECRET": ""}
    with patch.dict(os.environ, env, clear=False):
        return WahaAdapter(PlatformConfig(enabled=True, extra=extra))


# ── LID-robust identity resolution (design decision #3) ──────────────────────

class TestPreferPhoneJid:
    def test_prefers_non_lid_candidate(self):
        from plugins.platforms.waha.adapter import _prefer_phone_jid

        result = _prefer_phone_jid(["27842233445@lid", "27842233445@s.whatsapp.net"])
        assert result == "27842233445@s.whatsapp.net"

    def test_order_independent(self):
        """The phone form wins regardless of which field WAHA happened to populate first."""
        from plugins.platforms.waha.adapter import _prefer_phone_jid

        assert _prefer_phone_jid(["27842233445@s.whatsapp.net", "27842233445@lid"]) == "27842233445@s.whatsapp.net"

    def test_falls_back_to_lid_when_nothing_else_present(self):
        from plugins.platforms.waha.adapter import _prefer_phone_jid

        assert _prefer_phone_jid(["999999@lid", None, ""]) == "999999@lid"

    def test_empty_candidates_yield_empty_string(self):
        from plugins.platforms.waha.adapter import _prefer_phone_jid

        assert _prefer_phone_jid([None, "", None]) == ""


class TestBuildMessageEventIdentity:
    """_build_message_event must resolve chat_id to the phone-JID form even though WAHA's
    primary `from` field is the opaque @lid address -- the exact shape discovered live
    debugging webhook_filter_rob_whatsapp.py."""

    def _dm_payload(self, *, lid="27842233445@lid", phone="27842233445@s.whatsapp.net", body="hi", from_me=False):
        return {
            "id": "msg1", "from": lid, "fromMe": from_me, "body": body,
            "_data": {"key": {"remoteJid": lid, "remoteJidAlt": phone}},
        }

    def test_dm_chat_id_resolves_to_phone_jid_not_lid(self):
        adapter = _make_adapter(dm_policy="allowlist", allow_from=["27842233445"])
        event = adapter._build_message_event(self._dm_payload())
        assert event is not None
        assert event.source.chat_id == "27842233445@s.whatsapp.net"
        assert "@lid" not in event.source.chat_id

    def test_dm_without_remote_jid_alt_falls_back_to_lid(self):
        """No phone form anywhere in the payload (older engine/edge case): degrade to the lid,
        never crash or drop the message."""
        adapter = _make_adapter(dm_policy="allowlist", allow_from=["999999"])
        payload = {"id": "msg1", "from": "999999@lid", "fromMe": False, "body": "hi", "_data": {"key": {}}}
        event = adapter._build_message_event(payload)
        assert event is not None
        assert event.source.chat_id == "999999@lid"

    def test_group_sender_resolves_participant_to_phone_jid(self):
        adapter = _make_adapter(dm_policy="disabled", group_policy="open")
        payload = {
            "id": "msg1", "from": "120000@g.us", "fromMe": False, "body": "hello",
            "participant": "27800000002@lid",
            "_data": {"key": {"remoteJid": "120000@g.us", "participant": "27800000002@lid",
                              "participantAlt": "27800000002@s.whatsapp.net"}},
        }
        event = adapter._build_message_event(payload)
        assert event is not None
        assert event.source.chat_id == "120000@g.us"  # group JID itself, no LID ambiguity
        assert event.source.user_id == "27800000002@s.whatsapp.net"

    def test_broadcast_chat_is_dropped(self):
        adapter = _make_adapter(dm_policy="open")
        payload = {"id": "m1", "from": "status@broadcast", "fromMe": False, "body": "story", "_data": {}}
        assert adapter._build_message_event(payload) is None

    def test_unauthorized_dm_sender_is_dropped(self):
        adapter = _make_adapter(dm_policy="allowlist")  # empty allow_from -> nobody admitted
        event = adapter._build_message_event(self._dm_payload())
        assert event is None

    def test_allowlisted_dm_sender_is_admitted(self):
        adapter = _make_adapter(dm_policy="allowlist", allow_from=["27842233445"])
        event = adapter._build_message_event(self._dm_payload())
        assert event is not None
        assert event.text == "hi"


class TestMultiSenderOneAccountRouting:
    """Design decision #4: one WAHA adapter instance serves several senders off one WhatsApp
    account; distinct senders MUST resolve to distinct chat_ids so gateway.profile_routes (core,
    untouched) can split them to different profiles by chat_id."""

    def test_two_distinct_senders_resolve_to_distinct_chat_ids(self):
        adapter = _make_adapter(dm_policy="allowlist", allow_from=["27800000001", "27800000002"])
        rob = adapter._build_message_event({
            "id": "m1", "from": "27800000001@lid", "fromMe": False, "body": "hi",
            "_data": {"key": {"remoteJidAlt": "27800000001@s.whatsapp.net"}},
        })
        ashleigh = adapter._build_message_event({
            "id": "m2", "from": "27800000002@lid", "fromMe": False, "body": "hi",
            "_data": {"key": {"remoteJidAlt": "27800000002@s.whatsapp.net"}},
        })
        assert rob is not None and ashleigh is not None
        assert rob.source.chat_id != ashleigh.source.chat_id
        assert rob.source.chat_id == "27800000001@s.whatsapp.net"
        assert ashleigh.source.chat_id == "27800000002@s.whatsapp.net"


# ── Webhook HMAC verification ────────────────────────────────────────────────

class TestHmacVerification:
    def test_valid_sha512_signature_accepted(self):
        adapter = _make_adapter(hmac_secret="")
        adapter._hmac_secret = "my-secret-key"
        body = b'{"event":"message","session":"default","engine":"WEBJS"}'
        expected = hmac.new(b"my-secret-key", body, hashlib.sha512).hexdigest()
        headers = {"X-Webhook-Hmac": expected, "X-Webhook-Hmac-Algorithm": "sha512"}
        assert adapter._verify_hmac(headers, body) is True

    def test_wrong_secret_rejected(self):
        adapter = _make_adapter()
        adapter._hmac_secret = "my-secret-key"
        body = b"some body"
        wrong = hmac.new(b"wrong-key", body, hashlib.sha512).hexdigest()
        headers = {"X-Webhook-Hmac": wrong, "X-Webhook-Hmac-Algorithm": "sha512"}
        assert adapter._verify_hmac(headers, body) is False

    def test_missing_header_rejected_when_secret_configured(self):
        adapter = _make_adapter()
        adapter._hmac_secret = "my-secret-key"
        assert adapter._verify_hmac({}, b"body") is False

    def test_wrong_algorithm_rejected(self):
        adapter = _make_adapter()
        adapter._hmac_secret = "my-secret-key"
        body = b"body"
        digest = hmac.new(b"my-secret-key", body, hashlib.sha256).hexdigest()
        headers = {"X-Webhook-Hmac": digest, "X-Webhook-Hmac-Algorithm": "sha256"}
        assert adapter._verify_hmac(headers, body) is False

    def test_no_secret_configured_trusts_request(self):
        """connect() enforces loopback-or-secret; _verify_hmac alone just reflects 'no secret set'."""
        adapter = _make_adapter()
        adapter._hmac_secret = ""
        assert adapter._verify_hmac({}, b"anything") is True


# ── connect() fail-closed guards ──────────────────────────────────────────────

class TestConnectGuards:
    @pytest.mark.asyncio
    async def test_non_loopback_host_without_hmac_secret_refuses_to_start(self):
        adapter = _make_adapter(host="0.0.0.0", hmac_secret="")
        result = await adapter.connect()
        assert result is False
        assert adapter.has_fatal_error is True
        assert adapter.fatal_error_retryable is False
        assert adapter.fatal_error_code == "config_missing"

    @pytest.mark.asyncio
    async def test_session_not_working_is_retryable_fatal(self):
        adapter = _make_adapter(host="127.0.0.1")

        class _Resp:
            status = 200
            async def json(self):
                return {"status": "SCAN_QR_CODE", "me": None}
            async def __aenter__(self):
                return self
            async def __aexit__(self, *exc):
                return False

        class _Session:
            def get(self, *a, **k):
                return _Resp()
            async def close(self):
                pass

        with patch("aiohttp.ClientSession", return_value=_Session()):
            result = await adapter.connect()
        assert result is False
        assert adapter.fatal_error_code == "session_not_working"
        assert adapter.fatal_error_retryable is True


# ── chat id normalization + outbound send ─────────────────────────────────────

class TestWahaChatId:
    def test_bare_digits_become_c_us_jid(self):
        from plugins.platforms.waha.adapter import _waha_chat_id
        assert _waha_chat_id("27842233445") == "27842233445@c.us"

    def test_existing_jid_passes_through_unchanged(self):
        from plugins.platforms.waha.adapter import _waha_chat_id
        assert _waha_chat_id("27842233445@s.whatsapp.net") == "27842233445@s.whatsapp.net"


class TestOutboundSend:
    @pytest.mark.asyncio
    async def test_send_posts_to_sendtext_with_session_and_chat_id(self):
        adapter = _make_adapter()
        captured = {}

        class _Resp:
            status = 200
            async def text(self):
                return json.dumps({"id": "msg-123"})
            async def __aenter__(self):
                return self
            async def __aexit__(self, *exc):
                return False

        class _Session:
            def post(self, url, json=None, headers=None, timeout=None):
                captured["url"], captured["json"] = url, json
                return _Resp()

        adapter._http_session = _Session()
        result = await adapter.send("27842233445@c.us", "hello there")
        assert result.success is True
        assert result.message_id == "msg-123"
        assert captured["url"] == "http://localhost:3000/api/sendText"
        assert captured["json"] == {"session": "default", "chatId": "27842233445@c.us", "text": "hello there"}

    @pytest.mark.asyncio
    async def test_send_error_response_is_not_success(self):
        adapter = _make_adapter()

        class _Resp:
            status = 500
            async def text(self):
                return "boom"
            async def __aenter__(self):
                return self
            async def __aexit__(self, *exc):
                return False

        class _Session:
            def post(self, *a, **k):
                return _Resp()

        adapter._http_session = _Session()
        result = await adapter.send("27842233445@c.us", "hi")
        assert result.success is False
        assert "500" in (result.error or "")


# ── get_chat_info contract ────────────────────────────────────────────────────

class TestGetChatInfo:
    @pytest.mark.asyncio
    async def test_group_jid_reports_group_type(self):
        adapter = _make_adapter()
        info = await adapter.get_chat_info("120000@g.us")
        assert info == {"name": "120000@g.us", "type": "group", "chat_id": "120000@g.us"}

    @pytest.mark.asyncio
    async def test_dm_jid_reports_dm_type(self):
        adapter = _make_adapter()
        info = await adapter.get_chat_info("27842233445@c.us")
        assert info["type"] == "dm"


# ── Plugin registration ───────────────────────────────────────────────────────

class TestPluginRegistration:
    def test_register_wires_allowlist_env_names(self):
        from plugins.platforms.waha.adapter import register

        ctx = MagicMock()
        register(ctx)
        ctx.register_platform.assert_called_once()
        kwargs = ctx.register_platform.call_args.kwargs
        assert kwargs["name"] == "waha"
        assert kwargs["allowed_users_env"] == "WAHA_ALLOWED_USERS"
        assert kwargs["allow_all_env"] == "WAHA_ALLOW_ALL_USERS"
        assert kwargs["cron_deliver_env_var"] == "WAHA_HOME_CHANNEL"

    def test_platform_enum_resolves_dynamically_from_the_bundled_plugin_dir(self):
        """``Platform(\"waha\")`` must resolve via the bundled-plugin directory scan (no core enum
        change) and be stable across calls -- zero-core-change contract for the plugin path."""
        from gateway.config import Platform

        first = Platform("waha")
        assert first.value == "waha"
        assert Platform("waha") is first

    def test_check_requirements_is_passive(self):
        """check_fn must never attempt a network call (register_platform's documented contract)."""
        from plugins.platforms.waha.adapter import check_waha_requirements

        with patch("aiohttp.ClientSession", side_effect=AssertionError("must not touch the network")):
            assert isinstance(check_waha_requirements(), bool)


# ── Standalone send (out-of-process cron delivery) ───────────────────────────

class TestStandaloneSend:
    @pytest.mark.asyncio
    async def test_standalone_send_posts_sendtext(self):
        from plugins.platforms.waha.adapter import _standalone_send

        captured = {}

        class _Resp:
            status = 200
            async def text(self):
                return json.dumps({"id": "cron-1"})
            async def __aenter__(self):
                return self
            async def __aexit__(self, *exc):
                return False

        class _Session:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *exc):
                return False
            def post(self, url, headers=None, json=None):
                captured["url"], captured["json"] = url, json
                return _Resp()

        with patch("aiohttp.ClientSession", return_value=_Session()):
            result = await _standalone_send(
                PlatformConfig(enabled=True, extra={"base_url": "http://localhost:3000", "session": "default"}),
                "27842233445", "cron message")
        assert result["success"] is True
        assert captured["url"] == "http://localhost:3000/api/sendText"
        assert captured["json"]["chatId"] == "27842233445@c.us"
