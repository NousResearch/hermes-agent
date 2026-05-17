"""Tests for the Inkbox gateway adapter (email + SMS + voice)."""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_sdk(monkeypatch, *, lookup_result=None):
    """Stub out the lazy ``inkbox`` SDK imports inside the adapter module.

    ``lookup_result`` is the list returned by ``client.contacts.lookup`` for
    every call; defaults to a single-contact fake.
    """
    import gateway.platforms.inkbox as inkbox_mod

    if lookup_result is None:
        lookup_result = [SimpleNamespace(
            id="contact-uuid-123",
            preferred_name="Alex",
            given_name="Alex",
            emails=[SimpleNamespace(value="alex@example.com", is_primary=True)],
            phones=[SimpleNamespace(value="+15555550101", is_primary=True)],
        )]

    fake_client = MagicMock()
    fake_client.contacts.lookup.return_value = lookup_result
    fake_client.contacts.get.return_value = lookup_result[0] if lookup_result else None
    fake_client.get_identity.return_value = SimpleNamespace(
        agent_handle="inkbox-on-call-agent",
        mailbox=SimpleNamespace(email_address="agent@inkboxmail.com"),
        phone_number=SimpleNamespace(id="phone-uuid", number="+18005550100"),
        send_email=MagicMock(return_value=SimpleNamespace(id="msg-1")),
        send_text=MagicMock(return_value=SimpleNamespace(id="sms-1")),
    )

    InkboxClass = MagicMock(return_value=fake_client)
    InkboxClass.return_value.__enter__ = MagicMock(return_value=fake_client)
    InkboxClass.return_value.__exit__ = MagicMock(return_value=False)

    monkeypatch.setattr(inkbox_mod, "Inkbox", InkboxClass, raising=False)
    monkeypatch.setattr(
        inkbox_mod, "verify_webhook", MagicMock(return_value=True), raising=False,
    )
    monkeypatch.setattr(inkbox_mod, "INKBOX_AVAILABLE", True, raising=False)
    return fake_client


def _make_adapter(monkeypatch, *, include_sms_text_batch_delay=True, **extra):
    _patch_sdk(monkeypatch)
    from gateway.platforms.inkbox import InkboxAdapter
    extra_config = {
        "api_key": "ApiKey_test",
        "identity": "inkbox-on-call-agent",
        "base_url": "https://inkbox.ai",
    }
    if include_sms_text_batch_delay:
        extra_config["sms_text_batch_delay_seconds"] = 0
    extra_config.update(extra)

    cfg = PlatformConfig(
        enabled=True,
        api_key="ApiKey_test",
        extra=extra_config,
    )
    adapter = InkboxAdapter(cfg)
    # Pre-construct an SDK client so methods that access self._inkbox work
    # without needing to call connect().
    from gateway.platforms.inkbox import Inkbox
    adapter._inkbox = Inkbox()
    adapter._public_host = "tunnel.example"
    return adapter


async def _drain_background(adapter):
    for task in list(adapter._background_tasks):
        await task


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestInkboxConfigLoading:
    def test_apply_env_overrides_inkbox(self, monkeypatch):
        monkeypatch.setenv("INKBOX_API_KEY", "ApiKey_abc")
        monkeypatch.setenv("INKBOX_IDENTITY", "test-agent")
        monkeypatch.setenv("INKBOX_LISTEN_PORT", "9999")
        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.INKBOX in config.platforms
        ic = config.platforms[Platform.INKBOX]
        assert ic.enabled is True
        assert ic.extra["api_key"] == "ApiKey_abc"
        assert ic.extra["identity"] == "test-agent"
        assert ic.extra["port"] == 9999

    def test_home_channel_set_from_env(self, monkeypatch):
        monkeypatch.setenv("INKBOX_API_KEY", "ApiKey_abc")
        monkeypatch.setenv("INKBOX_IDENTITY", "test-agent")
        monkeypatch.setenv("INKBOX_HOME_CHANNEL", "contact-uuid-home")
        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)
        hc = config.platforms[Platform.INKBOX].home_channel
        assert hc is not None
        assert hc.chat_id == "contact-uuid-home"

    def test_not_connected_without_identity(self, monkeypatch):
        monkeypatch.setenv("INKBOX_API_KEY", "ApiKey_abc")
        monkeypatch.delenv("INKBOX_IDENTITY", raising=False)
        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.INKBOX not in config.get_connected_platforms()

    def test_connected_with_api_key_and_identity(self, monkeypatch):
        monkeypatch.setenv("INKBOX_API_KEY", "ApiKey_abc")
        monkeypatch.setenv("INKBOX_IDENTITY", "test-agent")
        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.INKBOX in config.get_connected_platforms()

    def test_require_signature_boolean_false_honored(self, monkeypatch):
        # Regression: `extra.get("require_signature") or os.getenv(...)`
        # silently coalesced a config-level boolean False into the env
        # default ("true"), so verification could not be disabled via config.
        monkeypatch.delenv("INKBOX_REQUIRE_SIGNATURE", raising=False)
        adapter = _make_adapter(monkeypatch, require_signature=False)
        assert adapter._require_signature is False


# ---------------------------------------------------------------------------
# Webhook routing
# ---------------------------------------------------------------------------

class _FakeRequest:
    """Minimal aiohttp.web.Request stand-in for the webhook handler."""

    def __init__(self, body: bytes, headers: dict | None = None, query: dict | None = None):
        self._body = body
        self.headers = headers or {}
        self.query = query or {}

    async def read(self):
        return self._body


class TestWebhookRouting:
    @pytest.mark.asyncio
    async def test_mail_webhook_routes_to_message_event(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)

        captured = []

        async def fake_handle_message(event):
            captured.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)

        envelope = {
            "event_type": "message.received",
            "data": {"message": {
                "id": "msg-uuid",
                "from_address": "alex@example.com",
                "subject": "Hi there",
                "snippet": "Hello world",
                "thread_id": "thread-7",
            }},
        }
        body = json.dumps(envelope).encode()
        req = _FakeRequest(body, headers={"X-Inkbox-Request-Id": "rid-1"})
        resp = await adapter._handle_webhook(req)
        assert resp.status == 200

        # Drain spawned background task.
        for task in list(adapter._background_tasks):
            await task

        assert len(captured) == 1
        ev = captured[0]
        assert ev.text.endswith("\nHello world")
        assert ev.text.startswith("[inkbox:email")
        assert ev.source.platform == Platform.INKBOX
        # contact_id resolved via lookup() should win over the raw email.
        assert ev.source.chat_id == "contact-uuid-123"
        # email threads mint a sub-session.
        assert ev.source.thread_id == "email:thread-7"
        assert ev.source.chat_topic == "Hi there"

    @pytest.mark.asyncio
    async def test_text_webhook_routes_to_message_event(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)

        captured = []

        async def fake_handle_message(event):
            captured.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)

        envelope = {
            "event_type": "text.received",
            "data": {"text_message": {
                "id": "sms-uuid",
                "remote_phone_number": "+15555550101",
                "local_phone_number": "+18005550100",
                "text": "ping",
                "direction": "inbound",
                "created_at": "2026-04-27T20:00:00Z",
            }},
        }
        body = json.dumps(envelope).encode()
        req = _FakeRequest(body)
        await adapter._handle_webhook(req)

        for task in list(adapter._background_tasks):
            await task

        assert len(captured) == 1
        ev = captured[0]
        assert ev.text.endswith("\nping")
        assert ev.text.startswith("[inkbox:sms")
        assert ev.source.chat_id == "contact-uuid-123"
        # SMS does NOT mint a sub-session — same chat_id, no thread_id.
        assert ev.source.thread_id is None

    @pytest.mark.asyncio
    async def test_incoming_call_webhook_returns_answer_with_ws_url(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)

        envelope = {
            "id": "call-uuid",
            "phone_number_id": "phone-uuid",
            "remote_phone_number": "+15555550101",
            "local_phone_number": "+18005550100",
            "direction": "inbound",
            "status": "ringing",
            "created_at": "2026-04-27T20:00:00Z",
        }
        body = json.dumps(envelope).encode()
        req = _FakeRequest(body)
        resp = await adapter._handle_webhook(req)

        assert resp.status == 200
        payload = json.loads(resp.body.decode())
        assert payload["action"] == "answer"
        assert payload["client_websocket_url"].startswith("wss://tunnel.example")
        assert "call_id=call-uuid" in payload["client_websocket_url"]

    @pytest.mark.asyncio
    async def test_duplicate_request_id_is_ignored(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        captured = []

        async def fake_handle_message(event):
            captured.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)

        envelope = {
            "event_type": "text.received",
            "data": {"text_message": {
                "id": "sms-1",
                "remote_phone_number": "+15555550101",
                "local_phone_number": "+18005550100",
                "text": "ping",
                "direction": "inbound",
                "created_at": "2026-04-27T20:00:00Z",
            }},
        }
        body = json.dumps(envelope).encode()
        for _ in range(2):
            await adapter._handle_webhook(
                _FakeRequest(body, headers={"X-Inkbox-Request-Id": "rid-dup"}),
            )

        for task in list(adapter._background_tasks):
            await task

        assert len(captured) == 1

    @pytest.mark.asyncio
    async def test_duplicate_text_id_is_ignored_without_request_id(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        captured = []

        async def fake_handle_message(event):
            captured.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)

        envelope = {
            "event_type": "text.received",
            "data": {"text_message": {
                "id": "sms-same-id",
                "remote_phone_number": "+15555550101",
                "local_phone_number": "+18005550100",
                "text": "ping",
                "direction": "inbound",
            }},
        }
        body = json.dumps(envelope).encode()
        await adapter._handle_webhook(_FakeRequest(body))
        await adapter._handle_webhook(_FakeRequest(body))

        for task in list(adapter._background_tasks):
            await task

        assert len(captured) == 1

    @pytest.mark.asyncio
    async def test_sms_batching_is_opt_in_by_default(self, monkeypatch):
        monkeypatch.delenv("INKBOX_SMS_TEXT_BATCH_DELAY_SECONDS", raising=False)
        adapter = _make_adapter(monkeypatch, include_sms_text_batch_delay=False)
        captured = []

        async def fake_handle_message(event):
            captured.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)

        envelope = {
            "event_type": "text.received",
            "data": {"text_message": {
                "id": "sms-default-immediate",
                "remote_phone_number": "+15555550101",
                "local_phone_number": "+18005550100",
                "text": "ping",
                "direction": "inbound",
                "created_at": "2026-04-27T20:00:00Z",
            }},
        }
        await adapter._handle_webhook(_FakeRequest(json.dumps(envelope).encode()))
        await _drain_background(adapter)

        assert adapter._sms_text_batch_delay_seconds == 0
        assert len(captured) == 1
        assert captured[0].text.endswith("\nping")
        assert adapter._pending_sms_text_batches == {}

    @pytest.mark.asyncio
    async def test_text_webhook_buffers_rapid_fragments_into_timestamped_burst(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, sms_text_batch_delay_seconds=60)
        captured = []

        async def fake_handle_message(event):
            captured.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)

        def envelope(text_id, text, created_at):
            return {
                "event_type": "text.received",
                "data": {"text_message": {
                    "id": text_id,
                    "remote_phone_number": "+15555550101",
                    "local_phone_number": "+18005550100",
                    "text": text,
                    "direction": "inbound",
                    "created_at": created_at,
                }},
            }

        for payload in [
            envelope("sms-burst-1", "first fragment", "2026-04-27T20:00:00Z"),
            envelope("sms-burst-2", "correction", "2026-04-27T20:00:06Z"),
            envelope("sms-burst-3", "extra detail", "2026-04-27T20:00:11Z"),
        ]:
            await adapter._handle_webhook(_FakeRequest(json.dumps(payload).encode()))

        assert captured == []
        assert len(adapter._pending_sms_text_batches) == 1

        key = next(iter(adapter._pending_sms_text_batches))
        await adapter._flush_sms_text_batch_now(key)
        await _drain_background(adapter)

        assert len(captured) == 1
        ev = captured[0]
        assert ev.text.startswith("[inkbox:sms_burst messages=3 ")
        assert "first_at=2026-04-27T20:00:00Z" in ev.text
        assert "last_at=2026-04-27T20:00:11Z" in ev.text
        assert "[+0s] first fragment" in ev.text
        assert "[+6s] correction" in ev.text
        assert "[+11s] extra detail" in ev.text
        assert ev.message_id == "sms-burst-3"
        assert ev.source.message_id == "sms-burst-3"

    @pytest.mark.asyncio
    async def test_sms_slash_command_bypasses_batching_and_marker(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, sms_text_batch_delay_seconds=60)
        captured = []

        async def fake_handle_message(event):
            captured.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)

        envelope = {
            "event_type": "text.received",
            "data": {"text_message": {
                "id": "sms-command",
                "remote_phone_number": "+15555550101",
                "local_phone_number": "+18005550100",
                "text": "/stop",
                "direction": "inbound",
                "created_at": "2026-04-27T20:00:00Z",
            }},
        }
        await adapter._handle_webhook(_FakeRequest(json.dumps(envelope).encode()))
        await _drain_background(adapter)

        assert len(captured) == 1
        ev = captured[0]
        assert ev.text == "/stop"
        assert ev.message_type.value == "command"
        assert ev.get_command() == "stop"
        assert adapter._last_inbound_modality["contact-uuid-123"] == "sms"
        assert adapter._pending_sms_text_batches == {}

    @pytest.mark.parametrize(
        "control_word",
        [
            "START",
            "STOP",
            "UNSTOP",
            "HELP",
            "CANCEL",
            "END",
            "QUIT",
            "UNSUBSCRIBE",
            "YES",
            "SUBSCRIBE",
            "INFO",
        ],
    )
    @pytest.mark.asyncio
    async def test_sms_control_word_does_not_enqueue_or_change_modality(
        self, monkeypatch, control_word,
    ):
        adapter = _make_adapter(monkeypatch, sms_text_batch_delay_seconds=60)
        adapter._last_inbound_modality["contact-uuid-123"] = "email"
        captured = []

        async def fake_handle_message(event):
            captured.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)

        envelope = {
            "event_type": "text.received",
            "data": {"text_message": {
                "id": f"sms-{control_word.lower()}-control",
                "remote_phone_number": "+15555550101",
                "local_phone_number": "+18005550100",
                "text": control_word,
                "direction": "inbound",
                "created_at": "2026-04-27T20:00:00Z",
            }},
        }
        await adapter._handle_webhook(_FakeRequest(json.dumps(envelope).encode()))
        await _drain_background(adapter)

        assert captured == []
        assert adapter._last_inbound_modality["contact-uuid-123"] == "email"
        assert adapter._pending_sms_text_batches == {}

    @pytest.mark.asyncio
    async def test_sms_flush_limit_preserves_concurrent_repopulated_batch(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch,
            sms_text_batch_delay_seconds=60,
            sms_text_batch_max_messages=1,
        )

        first = {
            "event_type": "text.received",
            "data": {"text_message": {
                "id": "sms-limit-first",
                "remote_phone_number": "+15555550101",
                "local_phone_number": "+18005550100",
                "text": "first",
                "direction": "inbound",
                "created_at": "2026-04-27T20:00:00Z",
            }},
        }
        second = {
            "event_type": "text.received",
            "data": {"text_message": {
                "id": "sms-limit-second",
                "remote_phone_number": "+15555550101",
                "local_phone_number": "+18005550100",
                "text": "second",
                "direction": "inbound",
                "created_at": "2026-04-27T20:00:01Z",
            }},
        }
        await adapter._handle_webhook(_FakeRequest(json.dumps(first).encode()))
        assert len(adapter._pending_sms_text_batches) == 1

        async def fake_flush(key):
            existing = adapter._pending_sms_text_batches.pop(key)
            fragment = dict(existing["fragments"][0])
            fragment["text"] = "concurrent"
            fragment["message_id"] = "sms-limit-concurrent"
            adapter._pending_sms_text_batches[key] = {
                "marker": existing["marker"],
                "fragments": [fragment],
                "raw_messages": list(existing["raw_messages"]),
                "last_event": existing["last_event"],
            }

        monkeypatch.setattr(adapter, "_flush_sms_text_batch_now", fake_flush)

        await adapter._handle_webhook(_FakeRequest(json.dumps(second).encode()))

        batch = next(iter(adapter._pending_sms_text_batches.values()))
        assert [fragment["text"] for fragment in batch["fragments"]] == [
            "concurrent",
            "second",
        ]

        for task in list(adapter._pending_sms_text_batch_tasks.values()):
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    @pytest.mark.asyncio
    async def test_duplicate_text_id_is_ignored_while_sms_batch_pending(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, sms_text_batch_delay_seconds=60)
        captured = []

        async def fake_handle_message(event):
            captured.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)

        envelope = {
            "event_type": "text.received",
            "data": {"text_message": {
                "id": "sms-pending-dup",
                "remote_phone_number": "+15555550101",
                "local_phone_number": "+18005550100",
                "text": "ping",
                "direction": "inbound",
                "created_at": "2026-04-27T20:00:00Z",
            }},
        }
        body = json.dumps(envelope).encode()
        await adapter._handle_webhook(_FakeRequest(body))
        await adapter._handle_webhook(_FakeRequest(body))

        assert len(adapter._pending_sms_text_batches) == 1
        batch = next(iter(adapter._pending_sms_text_batches.values()))
        assert len(batch["fragments"]) == 1

        key = next(iter(adapter._pending_sms_text_batches))
        await adapter._flush_sms_text_batch_now(key)
        await _drain_background(adapter)

        assert len(captured) == 1
        assert captured[0].text.endswith("\nping")

    @pytest.mark.asyncio
    async def test_text_lifecycle_event_does_not_enqueue_agent_turn(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        captured = []

        async def fake_handle_message(event):
            captured.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)

        envelope = {
            "event_type": "text.delivered",
            "data": {"text_message": {
                "id": "sms-queued-1",
                "remote_phone_number": "+15555550101",
                "direction": "outbound",
                "delivery_status": "delivered",
            }},
        }
        resp = await adapter._handle_webhook(_FakeRequest(json.dumps(envelope).encode()))

        assert resp.status == 200
        assert captured == []


# ---------------------------------------------------------------------------
# Contact-resolution cache
# ---------------------------------------------------------------------------

class TestContactCache:
    @pytest.mark.asyncio
    async def test_lookup_cached_within_ttl(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        cid1, name1 = await adapter._resolve_contact(kind="email", value="alex@example.com")
        cid2, name2 = await adapter._resolve_contact(kind="email", value="alex@example.com")
        assert cid1 == cid2 == "contact-uuid-123"
        assert name1 == name2 == "Alex"
        # Two calls, but only one SDK round-trip.
        assert adapter._inkbox.contacts.lookup.call_count == 1

    @pytest.mark.asyncio
    async def test_lookup_negative_result_cached(self, monkeypatch):
        # Override SDK to return zero contacts.
        _patch_sdk(monkeypatch, lookup_result=[])
        from gateway.platforms.inkbox import InkboxAdapter, Inkbox

        cfg = PlatformConfig(extra={
            "api_key": "ApiKey_test",
            "identity": "inkbox-on-call-agent",
            "signing_key": "whsec_test",
        })
        adapter = InkboxAdapter(cfg)
        adapter._inkbox = Inkbox()

        cid, name = await adapter._resolve_contact(kind="phone", value="+15555550999")
        assert cid is None and name is None
        # Repeat — should be served from the negative cache.
        await adapter._resolve_contact(kind="phone", value="+15555550999")
        assert adapter._inkbox.contacts.lookup.call_count == 1


# ---------------------------------------------------------------------------
# Authorization + send-tool integration
# ---------------------------------------------------------------------------

class TestPlatformWiring:
    def test_inkbox_in_send_message_platform_map(self):
        # Inkbox is one of the dispatcher branches in _send_to_platform.
        import inspect
        from tools import send_message_tool

        src = inspect.getsource(send_message_tool._send_to_platform)
        assert "Platform.INKBOX" in src

    def test_inkbox_in_known_delivery_platforms(self):
        from cron.scheduler import _KNOWN_DELIVERY_PLATFORMS
        assert "inkbox" in _KNOWN_DELIVERY_PLATFORMS

    def test_inkbox_in_home_target_env_vars(self):
        from cron.scheduler import _HOME_TARGET_ENV_VARS
        assert _HOME_TARGET_ENV_VARS["inkbox"] == "INKBOX_HOME_CHANNEL"

    def test_hermes_inkbox_toolset_exists(self):
        from toolsets import TOOLSETS
        assert "hermes-inkbox" in TOOLSETS
        assert "hermes-inkbox" in TOOLSETS["hermes-gateway"]["includes"]

    def test_inkbox_prompt_hint_exists(self):
        from agent.prompt_builder import PLATFORM_HINTS
        assert "inkbox" in PLATFORM_HINTS

    def test_inkbox_in_platform_registry(self):
        # placeholder — see body below
        pass

    def _placeholder_satisfies_collector(self):  # pragma: no cover
        pass


# Tunnel-supervision tests are intentionally absent here. The hand-rolled
# tunnel client (and our adapter-side watchdog) was replaced by
# ``inkbox.tunnels.client`` which owns its own supervisor; tunnel-runtime
# behavior is now covered by the SDK's own test suite.


# Inkbox is registered in the PLATFORMS map.
def test_inkbox_in_platforms_registry():
    from hermes_cli.platforms import PLATFORMS
    assert "inkbox" in PLATFORMS
    assert PLATFORMS["inkbox"].default_toolset == "hermes-inkbox"


# ---------------------------------------------------------------------------
# Send (outbound)
# ---------------------------------------------------------------------------

class TestSend:
    @pytest.mark.asyncio
    async def test_send_suppresses_todo_tool_progress_sms(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        result = await adapter.send(
            "+155****0101",
            '📋 todo: "planning 3 task(s)"',
            metadata={"mode": "sms", "to_phone": "+155****0101"},
        )
        assert result.success is True
        assert result.message_id == "suppressed-admin-notice"
        identity = adapter._inkbox.get_identity.return_value
        identity.send_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_sms_uses_e164_chat_id_directly(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        result = await adapter.send(
            "+15555550101", "hello", metadata={"mode": "sms", "to_phone": "+15555550101"},
        )
        assert result.success is True
        identity = adapter._inkbox.get_identity.return_value
        identity.send_text.assert_called_once_with(to="+15555550101", text="hello")
        assert result.raw_response["mode"] == "sms"

    @pytest.mark.asyncio
    async def test_send_sms_structures_provider_error(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        identity = adapter._inkbox.get_identity.return_value

        class FakeInkboxAPIError(Exception):
            status_code = 409
            detail = {
                "detail": {
                    "error": "messaging_profile_disabled",
                    "message": "Messaging profile is disabled.",
                },
            }

        identity.send_text.side_effect = FakeInkboxAPIError("conflict")

        result = await adapter.send(
            "+15555550101",
            "hello",
            metadata={"mode": "sms", "to_phone": "+15555550101"},
        )

        assert result.success is False
        assert result.retryable is False
        assert result.fallback_allowed is False
        assert "messaging_profile_disabled" in (result.error or "")
        assert result.raw_response["status_code"] == 409
        assert result.raw_response["error_code"] == "messaging_profile_disabled"
        assert result.raw_response["category"] == "sender_provisioning"

    @pytest.mark.asyncio
    async def test_send_sms_marks_server_error_retryable(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        identity = adapter._inkbox.get_identity.return_value

        class FakeInkboxAPIError(Exception):
            status_code = 502
            detail = {"detail": {"error": "carrier_unavailable", "message": "Carrier unavailable."}}

        identity.send_text.side_effect = FakeInkboxAPIError("unavailable")

        result = await adapter.send(
            "+15555550101",
            "hello",
            metadata={"mode": "sms", "to_phone": "+15555550101"},
        )

        assert result.success is False
        assert result.retryable is True
        assert result.fallback_allowed is False
        assert result.raw_response["category"] == "transient"

    @pytest.mark.asyncio
    async def test_send_email_resolves_address_from_contact_when_chat_id_is_uuid(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        result = await adapter.send(
            "contact-uuid-123",
            "hi from hermes",
            metadata={"mode": "email", "subject": "Greetings"},
        )
        assert result.success is True
        identity = adapter._inkbox.get_identity.return_value
        identity.send_email.assert_called_once()
        kwargs = identity.send_email.call_args.kwargs
        assert kwargs["to"] == ["alex@example.com"]
        assert kwargs["subject"] == "Greetings"
        assert kwargs["body_text"] == "hi from hermes"

    @pytest.mark.asyncio
    async def test_send_voice_without_active_ws_fails_cleanly(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        result = await adapter.send(
            "contact-uuid-123", "spoken reply", metadata={"mode": "voice"},
        )
        assert result.success is False
        assert "active call" in (result.error or "").lower()
