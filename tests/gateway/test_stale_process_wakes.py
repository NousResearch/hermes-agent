"""Queued process wakes are revalidated when they reach the head of the conversation."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from plugins.platforms.discord.adapter import DiscordAdapter
from tools import process_registry as processes
from tools.process_registry_notifications import format_process_notification


def setup_gateway(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "42")
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    registry = processes.ProcessRegistry()
    monkeypatch.setattr(processes, "process_registry", registry)
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    source = SessionSource(platform=Platform.DISCORD, chat_type="dm", chat_id="42", user_id="42")
    key = build_session_key(source)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    adapter._active_sessions[key] = asyncio.Event()
    return registry, runner, adapter, source, key


def notice(registry, key, name, kind="completion"):
    process = processes.ProcessSession(id=name, command="build", session_key=key,
                                      exited=kind == "completion", exit_code=1,
                                      output_buffer=f"{name} result\n")
    registry._running[name] = process
    event = {"type": kind, "session_id": name, "session_key": key,
             "command": process.command, "exit_code": process.exit_code,
             "output": process.output_buffer, "started_at": process.started_at, "seq": 1, "elapsed": 180, "interval": 180}
    return process, event


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["completion", "heartbeat", "replaced-heartbeat", "missing-heartbeat"])
@pytest.mark.parametrize("route", ["recursive", "handler"])
async def test_obsolete_wake_never_starts_a_turn_or_strands_human_followup(monkeypatch, kind, route):
    registry, runner, adapter, source, key = setup_gateway(monkeypatch)
    adapter.set_message_handler(runner._handle_message)
    process, raw = notice(registry, key, "proc_obsolete", "completion" if kind == "completion" else "heartbeat")
    assert await runner._inject_watch_notification(format_process_notification(raw), raw) is True
    # It was useful at admission; it becomes stale while the original turn is running.
    if kind == "heartbeat":
        process.exited = True
    elif kind == "replaced-heartbeat":
        registry._running[process.id] = processes.ProcessSession(id=process.id, command="new build", started_at=99)
    elif kind == "missing-heartbeat":
        del registry._running[process.id]
    else:
        registry.read_log(process.id)
    queued = adapter._pending_messages[key]
    human = MessageEvent(text="Next real question", source=source, metadata=dict(queued.metadata))
    await adapter.handle_message(human)
    if route == "recursive":
        event, text = await runner._run_agent_drain_pending({"final_response": "done"}, adapter, source, key)
        assert event is human and text == human.text
        assert not runner._overflow_queue(key)
    else:
        # The adapter's next background task calls the normal handler, bypassing recursive drain.
        stale = adapter.get_pending_message(key)
        runner._hm_estop_gate = lambda *_: "UNEXPECTED TURN"
        assert await runner._handle_message(stale) is None
        assert adapter.get_pending_message(key) is human
        assert not runner._overflow_queue(key)


@pytest.mark.asyncio
@pytest.mark.parametrize("consumed", [0, 1, 2])
async def test_completion_batch_preserves_unread_failures_and_other_notifications(monkeypatch, consumed):
    registry, runner, adapter, source, key = setup_gateway(monkeypatch)
    adapter.set_message_handler(AsyncMock())
    entries = [notice(registry, key, f"proc_result{i}")[1] for i in range(2)]
    assert all(await asyncio.gather(*[
        runner._enqueue_process_completion_notification(format_process_notification(raw), raw)
        for raw in entries
    ]))
    for raw in entries[:consumed]:
        registry.read_log(raw["session_id"])
    # Poll is read-only: it must never suppress autonomous reporting of an unread failure.
    for raw in entries[consumed:]:
        registry.poll(raw["session_id"])
    process, heartbeat = notice(registry, key, "proc_live", "heartbeat")
    assert await runner._inject_watch_notification(format_process_notification(heartbeat), heartbeat) is True
    _, watch = notice(registry, key, "proc_watch", "watch_match")
    watch["pattern"] = "READY"
    registry._running[watch["session_id"]].exited = True
    assert await runner._inject_watch_notification(format_process_notification(watch), watch) is True
    texts = []
    while True:
        event, text = await runner._run_agent_drain_pending({"final_response": "done"}, adapter, source, key)
        if event is None:
            break
        texts.append(text)
    assert len(texts) == (1 if consumed < 2 else 0) + 2
    for raw in entries[:consumed]:
        assert all(raw["session_id"] not in text for text in texts)
    for raw in entries[consumed:]:
        assert raw["session_id"] in texts[0]
    assert "heartbeat" in texts[-2] and "READY" in texts[-1]
    await runner._cancel_process_completion_batch_tasks()
