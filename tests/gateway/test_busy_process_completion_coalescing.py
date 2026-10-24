"""Busy-session completion wakes should not cascade into one model turn per process.

A real shell child exits while the adapter's first turn is blocked. The gateway
watcher admits its completion, but the model-facing turn must wait for the busy
turn to end. Completions arriving outside the short producer batch window can
still share one queued turn; an intervening human message must keep its place.
"""

import asyncio
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.run_notifications import INTERNAL_NOTIFICATION_FOOTER
from gateway.session import SessionSource, build_session_key
from hermes_constants import get_hermes_home
from plugins.platforms.discord.adapter import DiscordAdapter
from tools import process_registry as registry_module


@pytest.mark.asyncio
async def test_real_processes_finishing_during_busy_turn_coalesce_after_release(
    tmp_path, monkeypatch, record_property,
):
    registry = registry_module.ProcessRegistry()
    monkeypatch.setattr(registry_module, "process_registry", registry)
    monkeypatch.setattr(registry_module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    adapter.send = AsyncMock()
    source = SessionSource(platform=Platform.DISCORD, chat_type="dm", chat_id="42", user_id="42")
    key = build_session_key(source)
    started, release = asyncio.Event(), asyncio.Event()
    received = []
    timeline = []
    inject = runner._inject_watch_notification

    async def record_injection(text, event, **kwargs):
        admitted = await inject(text, event, **kwargs)
        if admitted is True:
            timeline.append(("injected", time.time(), event["session_id"]))
        return admitted

    monkeypatch.setattr(runner, "_inject_watch_notification", record_injection)

    async def handler(event):
        if event.text == "long-running turn":
            started.set()
            await release.wait()
        else:
            received.append(event)
            timeline.append(("model_turn", time.time(), event.text))
            # The fake handler stands in for a GatewayRunner turn, including the
            # normal post-turn FIFO promotion before the adapter's next drain.
            if key not in adapter._pending_messages:
                next_event = runner._promote_queued_event(key, adapter, None)
                if next_event is not None:
                    adapter._pending_messages[key] = next_event

    adapter.set_message_handler(handler)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    try:
        await adapter.handle_message(MessageEvent(text="long-running turn", source=source))
        await asyncio.wait_for(started.wait(), timeout=5)
        ids = []
        starts = []
        receipts = []
        for index in range(3):
            argv = [sys.executable, "-c", f"print('FINISHED_{index}')"]
            # The registry runs this through the host shell, so quote it for the
            # platform like the rest of the tree does (POSIX quoting would reach
            # cmd.exe unchanged on Windows).
            command = subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)
            # Avoid unrelated parent-manifest probing in the isolated checkout;
            # the child still runs through the real local process registry.
            with patch("tools.environments.local._resolve_hermes_bin_dir", return_value=None):
                session = registry.spawn_local(command, cwd=str(tmp_path), session_key=key)
            session.notify_on_complete = True
            ids.append(session.id)
            starts.append(session.started_at)
            watcher = {
                "session_id": session.id, "check_interval": 0.05,
                "session_key": key, "platform": "discord", "chat_type": "dm",
                "chat_id": "42", "notify_on_complete": True,
            }
            await asyncio.wait_for(runner._run_process_watcher(watcher), timeout=10)
            receipt = Path(get_hermes_home()) / "logs" / "process-results" / f"{session.id}.json"
            assert receipt.exists(), f"missing completion receipt for {session.id}"
            receipts.append(receipt.stat().st_mtime)
            timeline.append(("watcher_admitted", time.time(), session.id))
            # Space the completions past the producer's 0.1 s same-tick batch window: that
            # separation is the scenario under test here (wakes spread over time, #117555),
            # not an arbitrary delay.
            await asyncio.sleep(0.15)  # health: allow HX013 -- the window boundary is the case under test

        assert received == [], "a synthetic completion must not interrupt a busy turn"
        released_at = time.time()
        release.set()
        await asyncio.wait_for(asyncio.gather(*list(adapter._background_tasks)), timeout=10)
        # The session task can hand off to a fresh task during the gather above.
        while adapter._background_tasks:
            await asyncio.wait_for(asyncio.gather(*list(adapter._background_tasks)), timeout=10)
            await asyncio.sleep(0)
        assert len(received) == 1, f"one queued completion turn expected, got {len(received)}"
        assert all(session_id in received[0].text for session_id in ids), (
            f"model-facing text={received[0].text!r}; ids={ids!r}; "
            f"pending={adapter._pending_messages!r}; overflow={runner._overflow_queue(key)!r}"
        )
        assert timeline[-1][1] >= released_at
        assert all(when < released_at for when in receipts)
        injections = [(when, process_id) for kind, when, process_id in timeline if kind == "injected"]
        assert len(injections) == len(ids)
        assert all(receipt <= injected < released_at for receipt, (injected, _) in zip(receipts, injections))
        for process_id, started_at, receipt, (injected_at, _) in zip(ids, starts, receipts, injections):
            record_property(f"process_timeline_{process_id}",
                            f"start={started_at:.6f} result_mtime={receipt:.6f} "
                            f"injected={injected_at:.6f} model_turn={timeline[-1][1]:.6f} "
                            f"released={released_at:.6f}")
    finally:
        release.set()
        await adapter.cancel_background_tasks()
        await runner._cancel_process_completion_batch_tasks()


@pytest.mark.asyncio
async def test_busy_completion_coalescing_preserves_human_order_and_separate_targets():
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    source = SessionSource(platform=Platform.DISCORD, chat_type="dm", chat_id="42", user_id="42")
    key = build_session_key(source)
    adapter._pending_messages[key] = MessageEvent(
        text="first completion\n\n" + INTERNAL_NOTIFICATION_FOOTER, source=source, internal=True,
        metadata={"notification_origin": "process_registry_synthetic", "notification_event_type": "completion",
                  "gateway_session_id": "first-parent"},
    )
    human = MessageEvent(text="human instruction", source=source)
    other_parent = MessageEvent(
        text="other parent", source=source, internal=True,
        metadata={"notification_origin": "process_registry_synthetic", "notification_event_type": "completion",
                  "gateway_session_id": "other-parent"},
    )
    same_parent = MessageEvent(
        text="third completion", source=source, internal=True,
        metadata={"notification_origin": "process_registry_synthetic", "notification_event_type": "completion",
                  "gateway_session_id": "first-parent"},
    )
    second = MessageEvent(
        text="second completion\n\n" + INTERNAL_NOTIFICATION_FOOTER, source=source, internal=True,
        metadata={"notification_origin": "process_registry_synthetic", "notification_event_type": "completion",
                  "gateway_session_id": "first-parent"},
    )
    runner._queue_or_replace_pending_event(key, second)
    assert runner._queue_depth(key, adapter=adapter) == 1
    assert "first completion" in adapter._pending_messages[key].text
    assert "second completion" in adapter._pending_messages[key].text
    assert adapter._pending_messages[key].metadata["coalesced_process_wakes"] == 2
    assert adapter._pending_messages[key].text.count(INTERNAL_NOTIFICATION_FOOTER) == 1
    for event in (human, same_parent, other_parent):
        runner._queue_or_replace_pending_event(key, event)
    assert runner._queue_depth(key, adapter=adapter) == 4
    queued = [adapter._pending_messages[key], *(runner._overflow_queue(key) or [])]
    assert queued[1].text == "human instruction"
    assert queued[2].text == "third completion"
    assert queued[3].text == "other parent"
    assert all(event._gateway_accepted for event in (second, human, same_parent, other_parent))


@pytest.mark.asyncio
async def test_busy_completion_can_join_overflow_tail_behind_human_at_queue_cap():
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    source = SessionSource(platform=Platform.DISCORD, chat_type="dm", chat_id="42", user_id="42")
    key = build_session_key(source)
    runner._BUSY_QUEUE_MAX_PENDING = 2  # type: ignore[assignment]
    human = MessageEvent(text="human first", source=source)
    adapter._pending_messages[key] = human
    first = MessageEvent(
        text="first completion\n\n" + INTERNAL_NOTIFICATION_FOOTER, source=source, internal=True,
        metadata={"notification_origin": "process_registry_synthetic", "notification_event_type": "completion"},
    )
    second = MessageEvent(
        text="second completion\n\n" + INTERNAL_NOTIFICATION_FOOTER, source=source, internal=True,
        metadata={"notification_origin": "process_registry_synthetic", "notification_event_type": "completion"},
    )
    runner._queue_or_replace_pending_event(key, first)
    runner._queue_or_replace_pending_event(key, second)
    assert adapter._pending_messages[key] is human
    overflow = runner._overflow_queue(key)
    assert overflow is not None and len(overflow) == 1
    assert all(label in overflow[0].text for label in ("first completion", "second completion"))
    assert overflow[0].text.count(INTERNAL_NOTIFICATION_FOOTER) == 1
    assert first._gateway_accepted and second._gateway_accepted


@pytest.mark.asyncio
async def test_completion_without_pending_slot_does_not_merge_orphaned_overflow():
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    source = SessionSource(platform=Platform.DISCORD, chat_type="dm", chat_id="42", user_id="42")
    key = build_session_key(source)
    old = MessageEvent(
        text="orphaned completion", source=source, internal=True,
        metadata={"notification_origin": "process_registry_synthetic", "notification_event_type": "completion"},
    )
    runner._session_state(key).conversation.queued_events.append(old)
    new = MessageEvent(
        text="fresh completion", source=source, internal=True,
        metadata={"notification_origin": "process_registry_synthetic", "notification_event_type": "completion"},
    )
    runner._queue_or_replace_pending_event(key, new)
    assert old.text == "orphaned completion"
    assert adapter._pending_messages[key] is new
