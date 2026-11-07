"""Turn attribution follows response readiness across native FIFO reconstruction."""
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def _lines(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("turn timing:")]


def test_formatter_omits_unknowns_and_clamps_setup():
    from gateway.turn_timing import TurnTiming, format_turn_timing_line
    timing = TurnTiming(handler_start=1.0, hygiene_s=2.0, agent_start=2.0, agent_end=5.0, finished=7.0)
    line = format_turn_timing_line(timing, platform="slack", session_key="fixture", outcome="ok", api_calls=None)
    assert "handler=6.0s setup=0.0s hygiene=2.0s agent=3.0s" in line
    assert all(f"{field}=" not in line for field in
               ("model", "tools", "compaction", "ttfb", "api_calls", "since_msg", "delivery", "queue"))


@pytest.mark.parametrize("age", [None, -1, 86401, "epoch", "mismatch"])
def test_since_message_omits_missing_future_stale_and_mismatched_timestamps(age):
    from gateway.turn_timing import since_message_seconds
    now = datetime(2026, 10, 27, tzinfo=UTC)
    stamp = now - timedelta(seconds=age) if isinstance(age, int) else None
    if age == "epoch":
        stamp = datetime.fromtimestamp(0, UTC)
    if age == "mismatch":
        stamp = now.replace(tzinfo=None)
    assert since_message_seconds(stamp, now) is None


@pytest.mark.parametrize("aware", [False, True])
def test_since_message_uses_wall_clock_for_matching_timestamps(aware):
    from gateway.turn_timing import since_message_seconds
    now = datetime(2026, 10, 27, tzinfo=UTC if aware else None)
    assert since_message_seconds(now - timedelta(seconds=5), now) == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_fields", [None, {"tools_s": 2.5, "compaction_s": 0.0, "model_s": 1.0, "ttfb_s": 0.5}])
@pytest.mark.parametrize("status", ["ok", "failed", "interrupted", "partial"])
async def test_real_handler_logs_once_at_response_ready(monkeypatch, caplog, agent_fields, status):
    from tests.gateway.test_internal_event_pin_wiring import _make_runner
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    runner = _make_runner(monkeypatch)
    clock = SimpleNamespace(now=2.0)
    monkeypatch.setattr("time.monotonic", lambda: clock.now)
    async def hygiene(*args):
        clock.now = 4.0
        return []
    async def notes(*args, **kwargs):
        clock.now = 5.0
    async def run(**kwargs):
        clock.now = 10.0
        return {"final_response": "PRIVATE_RESPONSE", "messages": [], "tools": [], "api_calls": 3,
                "history_offset": 0, "last_prompt_tokens": 0,
                **({status: True} if status != "ok" else {}),
                **({"turn_timing": agent_fields} if agent_fields else {})}
    runner._hmwa_run_session_hygiene, runner._hmwa_first_contact_notes, runner._run_agent = hygiene, notes, run
    source = SessionSource(platform=Platform.DISCORD, chat_id="PRIVATE_CHAT", user_id="PRIVATE_USER", chat_type="dm")
    event = MessageEvent(text="PRIVATE_BODY", source=source, timestamp=datetime.now() - timedelta(seconds=20))
    tail = runner._deliver_turn_response_then_limit_offer
    async def deliver(*args):
        assert len(_lines(caplog)) == 1
        clock.now = 30.0  # Delivery must not change the frozen handler duration.
        return await tail(*args)
    runner._deliver_turn_response_then_limit_offer = deliver
    with caplog.at_level(logging.INFO):
        await runner._handle_message_with_agent(event, source, "fixture-key", 1)
    line, = _lines(caplog)
    assert f"outcome={status} handler=8.0s setup=1.0s hygiene=2.0s agent=5.0s" in line
    assert "since_msg=" in line and "api_calls=3" in line
    assert ("tools=2.5s compaction=0.0s model=1.0s ttfb=0.5s" in line) == bool(agent_fields)
    assert all(value not in line for value in (event.text, source.user_id, source.chat_id, "PRIVATE_RESPONSE"))
    assert all(f"{name}=" not in line for name in ("delivery", "queue", "total"))
    messages = [r.getMessage() for r in caplog.records]
    assert messages[messages.index(line) - 1].startswith("response ready:")
    print(line)


@pytest.mark.asyncio
@pytest.mark.parametrize("stamp", [None, datetime.fromtimestamp(0, UTC), datetime(2099, 1, 1)])
async def test_real_handler_omits_unusable_message_timestamp(monkeypatch, caplog, stamp):
    from tests.gateway.test_internal_event_pin_wiring import _make_runner, _capture, _human_source
    from gateway.platforms.event import MessageEvent
    runner = _make_runner(monkeypatch)
    _capture(runner, [])
    event = MessageEvent(text="fixture", source=_human_source(), timestamp=stamp)
    with caplog.at_level(logging.INFO):
        await runner._handle_message_with_agent(event, event.source, "fixture-key", 1)
    line, = _lines(caplog)
    assert "since_msg=" not in line


@pytest.mark.asyncio
async def test_fifo_admitted_restored_event_logs_once(tmp_path, monkeypatch, caplog):
    import gateway.run as gr
    import gateway.session_envelope as envelope
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.session_authority import initialize_session_authority
    from hermes_state_runtime import list_session_admissions
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    from tests.waiting import async_wait_until
    from tests.gateway.test_internal_event_pin_wiring import _capture

    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "15551234567")
    monkeypatch.setattr(gr, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    runner = gr.GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={
        "text_batch_delay_seconds": 0, "session_path": str(tmp_path / "wa")}))
    adapter.gateway_runner, runner.adapters = runner, {Platform.WHATSAPP: adapter}
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="timing")
    runner._wire_adapter_handlers(adapter)
    calls, restored, executed = [], [], []
    _capture(runner, calls)
    restore, handle = envelope.restore_native, runner._handle_message_with_agent
    def record_restore(*args, **kwargs):
        event = restore(*args, **kwargs)
        assert not hasattr(event, "_turn_timing")
        restored.append(event)
        return event
    async def record_handle(event, *args, **kwargs):
        executed.append(event)
        return await handle(event, *args, **kwargs)
    monkeypatch.setattr(envelope, "restore_native", record_restore)
    runner._handle_message_with_agent = record_handle
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="sent"))
    source = adapter.build_source(chat_id="15551234567@s.whatsapp.net", user_id="15551234567@s.whatsapp.net",
                                  chat_type="dm", message_id="fixture-message")
    event = MessageEvent(text="PRIVATE_FIFO_BODY", source=source, message_id="fixture-message",
                         timestamp=datetime.now() - timedelta(seconds=10))
    try:
        with caplog.at_level(logging.INFO):
            await adapter.handle_message(event)
            await async_wait_until(lambda: calls and not adapter._background_tasks and not adapter._session_tasks,
                                   timeout=30, what="FIFO timing turn")
        assert len(executed) == 1 and executed[0] is not event
        assert any(item is executed[0] for item in restored)
        rows = list_session_admissions(authority.db, session_id=calls[0]["session_id"], pending_only=False)
        assert len(rows) == 1 and rows[0]["request_id"] == "fixture-message"
        line, = _lines(caplog)
        assert "outcome=ok handler=" in line and "since_msg=" in line
        assert "PRIVATE_FIFO_BODY" not in line
        assert "model=" not in line and "delivery=" not in line
    finally:
        authority.db.close()


def test_runner_projects_agent_timing(monkeypatch):
    from tests.gateway.test_runner_compressed_receipt import _CompressingAgent
    from tests.gateway.test_compression_failure_session_sync import (
        _install_compression_failure_agent, _run_compression_failure_turn, _runner, _SessionStore)
    from gateway.config import Platform
    from gateway.session import SessionSource
    original = _CompressingAgent.run_conversation
    def run(self, *args, **kwargs):
        return {**original(self, *args, **kwargs), "turn_timing": {"tools_s": 2.5}}
    monkeypatch.setattr(_CompressingAgent, "run_conversation", run)
    _install_compression_failure_agent(monkeypatch, _CompressingAgent)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="fixture", chat_type="dm", user_id="fixture")
    result = _run_compression_failure_turn(_runner(_SessionStore()), source)
    assert result["turn_timing"] == {"tools_s": 2.5}
