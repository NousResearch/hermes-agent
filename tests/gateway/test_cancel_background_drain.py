"""Regression test: cancel_background_tasks must drain late-arrival tasks.

During gateway shutdown, a message arriving while
cancel_background_tasks is mid-await can spawn a fresh
_process_message_background task via handle_message, which is added
to self._background_tasks.  Without the re-drain loop, the subsequent
_background_tasks.clear() drops the reference; the task runs
untracked against a disconnecting adapter.
"""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key

class _StubAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        return None

    async def get_chat_info(self, chat_id):
        return {}

def _make_adapter():
    adapter = _StubAdapter(PlatformConfig(enabled=True, token="t"), Platform.TELEGRAM)
    adapter._send_with_retry = AsyncMock(return_value=None)
    return adapter

def _event(text, cid="42"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id=cid, chat_type="dm"),
    )

@pytest.mark.asyncio
async def test_cancel_background_tasks_drains_late_arrivals():
    """A message that arrives during the gather window must be picked
    up by the re-drain loop, not leaked as an untracked task."""
    adapter = _make_adapter()
    sk = build_session_key(
        SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm")
    )

    m1_started = asyncio.Event()
    m1_cleanup_running = asyncio.Event()
    m2_started = asyncio.Event()
    m2_cancelled = asyncio.Event()

    async def handler(event):
        if event.text == "M1":
            m1_started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                m1_cleanup_running.set()
                # Widen the gather window with a shielded cleanup
                # delay so M2 can get injected during it.
                await asyncio.shield(asyncio.sleep(0.2))
                raise
        else:  # M2 — the late arrival
            m2_started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                m2_cancelled.set()
                raise

    adapter._message_handler = handler

    # Spawn M1.
    await adapter.handle_message(_event("M1"))
    await asyncio.wait_for(m1_started.wait(), timeout=1.0)

    # Kick off shutdown.  This will cancel M1 and await its cleanup.
    cancel_task = asyncio.create_task(adapter.cancel_background_tasks())

    # Wait until M1's cleanup is running (inside the shielded sleep).
    # This is the race window: cancel_task is awaiting gather, M1 is
    # shielded in cleanup, the _active_sessions entry has been cleared
    # by M1's own finally.
    await asyncio.wait_for(m1_cleanup_running.wait(), timeout=1.0)

    # Clear the active-session entry (M1's finally hasn't fully run yet,
    # but in production the platform dispatcher would deliver a new
    # message that takes the no-active-session spawn path).  For this
    # repro, make it deterministic.
    adapter._active_sessions.pop(sk, None)

    # Inject late arrival — spawns a fresh _process_message_background
    # task and adds it to _background_tasks while cancel_task is still
    # in gather.
    await adapter.handle_message(_event("M2"))
    await asyncio.wait_for(m2_started.wait(), timeout=1.0)

    # Let cancel_task finish.  Round 1's gather completes when M1's
    # shielded cleanup finishes.  Round 2 should pick up M2.
    await asyncio.wait_for(cancel_task, timeout=5.0)

    # Assert M2 was drained, not leaked.
    assert m2_cancelled.is_set(), (
        "Late-arrival M2 was NOT cancelled by cancel_background_tasks — "
        "the re-drain loop is missing and the task leaked"
    )
    assert adapter._background_tasks == set()


@pytest.mark.asyncio
@pytest.mark.parametrize("synthetic_head", [False, True], ids=("human-control", "synthetic-head"))
async def test_cancel_background_tasks_persists_incompatible_debounce(
    tmp_path, monkeypatch, synthetic_head
):
    """Shutdown must persist an accepted debounce event even when prompt identity blocks its flush."""
    import gateway.shutdown_flush as shutdown_flush

    flush_dir = tmp_path / "pending_messages"
    flush_dir.mkdir()
    monkeypatch.setattr(shutdown_flush, "_get_flush_dir", lambda: flush_dir)
    monkeypatch.setenv("TELEGRAM_ALLOW_ALL_USERS", "true")

    adapter = _make_adapter()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=False)
    runner.adapters, runner._profile_adapters = {Platform.TELEGRAM: adapter}, {}
    runner._primary_profile_name, runner._sessions, runner._draining = "default", {}, False
    runner._busy_input_mode = runner._busy_text_mode = adapter._busy_text_mode = "queue"
    adapter.gateway_runner = runner
    adapter._busy_session_handler = runner._handle_active_session_busy_message

    async def held_turn(_event):
        raise AssertionError("the pre-existing active turn remains held")

    adapter.set_message_handler(held_turn)
    source = adapter.build_source(
        chat_id="1001", chat_type="dm", user_id="101", message_id="201",
    )
    head = (
        runner._synthetic_prompt_event(source, "pending-continuation")
        if synthetic_head
        else MessageEvent(text="pending-human", source=source, message_id="201")
    )
    key = adapter._event_session_key(head)
    adapter._active_sessions[key] = asyncio.Event()
    runner._enqueue_fifo(key, head, adapter)
    adapter._spawn_drain_task(head, key, delay=adapter._REQUEUE_BACKOFF_MAX_SECONDS)

    human = MessageEvent(
        text="human-must-survive-shutdown",
        source=adapter.build_source(
            chat_id="1001", chat_type="dm", user_id="101", message_id="202",
        ),
        message_id="202",
    )
    await adapter.handle_message(human)
    assert key in adapter._text_debounce

    await asyncio.wait_for(adapter._text_debounce[key].task, timeout=5)
    assert human._gateway_accepted is True
    assert not adapter._session_tasks[key].done()

    await adapter.cancel_background_tasks()

    payloads = [json.loads(path.read_text()) for path in flush_dir.glob("*.json")]
    persisted_text = "\n".join(
        str((payload.get("data") or {}).get("text", "")) for payload in payloads
    )
    assert "human-must-survive-shutdown" in persisted_text
    if synthetic_head:
        assert "pending-continuation" in persisted_text
