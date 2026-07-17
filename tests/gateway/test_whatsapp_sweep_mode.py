"""Executable contracts for opt-in WhatsApp inbox sweeps."""

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

REPO = Path(__file__).resolve().parents[2]


def test_fixed_window_is_not_extended_by_inbound_receipts() -> None:
    script = """
import assert from 'node:assert/strict';
import { createInboxSweepController } from './scripts/whatsapp-bridge/inbox_sweep.js';
let now = 0, next = 0, closed = 0; const timers = new Map();
const setTimeoutFn = (fn, delay) => { const id = ++next; timers.set(id, [now + delay, fn]); return id; };
const clearTimeoutFn = (id) => timers.delete(id);
const advance = (ms) => { const target = now + ms; while (true) { const due = [...timers.entries()].filter(([, value]) => value[0] <= target).sort((a,b) => a[1][0] - b[1][0])[0]; if (!due) break; timers.delete(due[0]); now = due[1][0]; due[1][1](); } now = target; };
const sweep = createInboxSweepController({ reconnectIntervalMs: 120000, setTimeoutFn, clearTimeoutFn, closeSocket: () => closed++, reconnect: () => {} });
sweep.connected(); advance(2999); sweep.receivedInbound(); assert.equal(closed, 0); advance(1); assert.equal(closed, 1);
"""
    result = subprocess.run(["node", "--input-type=module", "--eval", script], cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_sweep_cycle_stays_anchored_to_open_and_buffer_is_bounded() -> None:
    script = """
import assert from 'node:assert/strict';
import { createInboxReceiptBuffer, createInboxSweepController } from './scripts/whatsapp-bridge/inbox_sweep.js';
let now = 0, next = 0, closed = 0; const timers = new Map();
const setTimeoutFn = (fn, delay) => { const id = ++next; timers.set(id, [now + delay, fn]); return id; };
const clearTimeoutFn = (id) => timers.delete(id);
const advance = (ms) => { const target = now + ms; while (true) { const due = [...timers.entries()].filter(([, value]) => value[0] <= target).sort((a,b) => a[1][0] - b[1][0])[0]; if (!due) break; timers.delete(due[0]); now = due[1][0]; due[1][1](); } now = target; };
const sweep = createInboxSweepController({ reconnectIntervalMs: 120000, setTimeoutFn, clearTimeoutFn, closeSocket: () => closed++, reconnect: () => {} });
sweep.connected(); advance(3000); assert.equal(closed, 1); advance(500); sweep.closed({ intentional: true });
assert.deepEqual([...timers.values()].map(([at]) => at - now), [116500]);
const delivered = []; const buffer = createInboxReceiptBuffer({ maxEntries: 1, deliver: receipt => delivered.push(receipt) });
assert.equal(buffer.capture({ id: 'first' }), true); assert.equal(buffer.capture({ id: 'second' }), false); assert.deepEqual(delivered, []); buffer.release(); assert.deepEqual(delivered, [{ id: 'first' }]);
"""
    result = subprocess.run(["node", "--input-type=module", "--eval", script], cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_inbox_config_is_opt_in_and_rejects_sender_reply_target() -> None:
    disabled = WhatsAppAdapter(PlatformConfig(enabled=True, extra={}))
    assert disabled._inbox_sweep_enabled is False
    enabled = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"inbox_sweep": {"enabled": True, "reconnect_interval_seconds": 120, "delivery_platform": "telegram"}}))
    assert enabled._inbox_sweep_enabled is True
    assert enabled._inbox_sweep_interval_ms == 120_000
    with pytest.raises(ValueError, match="inbox_sweep"):
        WhatsAppAdapter(PlatformConfig(enabled=True, extra={"inbox_sweep": {"enabled": True, "reconnect_interval_seconds": 120, "delivery_platform": "whatsapp"}}))


def _event() -> MessageEvent:
    return MessageEvent(text="Can we meet today?", message_id="wamid.test", metadata={"whatsapp_inbox_sweep": True, "whatsapp_inbox_delivery_platform": "telegram"}, source=SessionSource(platform=Platform.WHATSAPP, chat_id="15551234567", user_id="15551234567", user_name="Test sender", chat_type="dm"))


@pytest.mark.asyncio
async def test_high_priority_receipt_routes_home_and_never_calls_whatsapp_send() -> None:
    runner = cast(Any, GatewayRunner.__new__(GatewayRunner))
    runner.config = SimpleNamespace(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)}, get_home_channel=lambda p: SimpleNamespace(chat_id="operator-home", thread_id=None) if p == Platform.TELEGRAM else None)
    telegram, whatsapp = SimpleNamespace(send=AsyncMock(return_value=SimpleNamespace(success=True))), SimpleNamespace(send=AsyncMock())
    runner.adapters = {Platform.TELEGRAM: telegram, Platform.WHATSAPP: whatsapp}
    runner._triage_whatsapp_inbox_sweep_event = AsyncMock(return_value="## WhatsApp triage\n**Priority:** high")
    assert await runner._forward_whatsapp_inbox_sweep_event(_event()) is True
    telegram.send.assert_awaited_once_with("operator-home", "## WhatsApp triage\n**Priority:** high")
    whatsapp.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_sweep_receipt_skips_pre_dispatch_plugins() -> None:
    runner = cast(Any, GatewayRunner.__new__(GatewayRunner))
    runner._startup_restore_in_progress = False
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner.config = SimpleNamespace(platforms={}, get_home_channel=lambda p: None)
    runner.session_store = object()
    runner.adapters = {Platform.WHATSAPP: SimpleNamespace(send=AsyncMock())}
    runner._forward_whatsapp_inbox_sweep_event = AsyncMock(return_value=True)

    with patch("hermes_cli.plugins.invoke_hook") as invoke_hook:
        assert await runner._handle_message(_event()) is None
    invoke_hook.assert_not_called()


@pytest.mark.asyncio
async def test_inbox_mode_refuses_sender_facing_text_before_http() -> None:
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"inbox_sweep": {"enabled": True, "reconnect_interval_seconds": 120, "delivery_platform": "telegram"}}))
    adapter._running = True
    adapter._http_session = SimpleNamespace(post=MagicMock())
    result = await adapter.send("15551234567", "do not send")
    assert result.success is False
    assert result.error == "WhatsApp inbox sweep mode is receive-only"
    adapter._http_session.post.assert_not_called()


@pytest.mark.asyncio
async def test_triage_never_rehydrates_sender_state_or_persists_receipt() -> None:
    runner = cast(Any, GatewayRunner.__new__(GatewayRunner))
    runtime = {"api_key": "test-key"}
    runner._resolve_session_agent_runtime = MagicMock(return_value=("test/model", runtime))
    runner._resolve_turn_agent_config = MagicMock(return_value={"model": "test/model", "runtime": {}, "request_overrides": None})
    runner._provider_routing = {}
    runner._reasoning_config = {}
    runner._service_tier = None
    runner._fallback_model = None
    runner._load_provider_routing = lambda: {}
    runner._load_reasoning_config = lambda: {}
    runner._load_service_tier = lambda: None
    runner._load_fallback_model = lambda: None
    runner._cleanup_agent_resources = lambda agent: None

    async def _run_inline(fn):
        return fn()

    runner._run_in_executor_with_context = _run_inline
    created = []

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._persist_disabled = False
            self._session_json_enabled = True
            created.append(self)

        def run_conversation(self, **kwargs):
            assert self._persist_disabled is True
            assert self._session_json_enabled is False
            return {"final_response": "**Priority:** low"}

    with patch("gateway.run._load_gateway_config", return_value={}), patch("run_agent.AIAgent", FakeAgent):
        assert await runner._triage_whatsapp_inbox_sweep_event(_event()) == "**Priority:** low"

    runner._resolve_session_agent_runtime.assert_called_once()
    assert runner._resolve_session_agent_runtime.call_args.kwargs["source"] is None
    assert created[0].kwargs["session_db"] is None
    assert created[0].kwargs["skip_memory"] is True
    assert created[0].kwargs["skip_context_files"] is True


def test_blank_home_channel_is_rejected_at_startup() -> None:
    runner = cast(Any, GatewayRunner.__new__(GatewayRunner))
    runner.config = SimpleNamespace(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)},
        get_home_channel=lambda platform: SimpleNamespace(chat_id="") if platform == Platform.TELEGRAM else None,
    )
    adapter = SimpleNamespace(_inbox_sweep_enabled=True, _inbox_sweep_delivery_platform="telegram")
    assert runner._validate_whatsapp_inbox_sweep_target(adapter) is False
