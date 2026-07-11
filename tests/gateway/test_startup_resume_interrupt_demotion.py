"""Boot-resume recovery turns must not be killed by a fast user follow-up.

2026-07-10 live incident: a gateway restart auto-resumed an interrupted
session; the user sent "how's it going?" ~16s later, while the recovery
turn was at iteration 1/300.  busy_input_mode='interrupt' aborted the
recovery turn and the ack said only "⚡ Interrupting current task" — the
user was never told a restart had happened, and the recovery turn's
handoff checklist died at iteration 1.

Contract (same demotion pattern as subagents/#30170 and
compression/#56391):
- while a session's running turn is a boot-resume turn, 'interrupt'
  demotes to 'queue' (explicit /stop remains the escape hatch);
- the busy ack names the restart and the recovery, not a generic
  interrupt.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_tg = types.ModuleType("telegram")
_tg.constants = types.ModuleType("telegram.constants")
_ct = MagicMock()
_ct.SUPERGROUP = "supergroup"
_ct.GROUP = "group"
_ct.PRIVATE = "private"
_tg.constants.ChatType = _ct
sys.modules.setdefault("telegram", _tg)
sys.modules.setdefault("telegram.constants", _tg.constants)
sys.modules.setdefault("telegram.ext", types.ModuleType("telegram.ext"))

from gateway.platforms.base import (  # noqa: E402
    MessageEvent,
    MessageType,
    SessionSource,
    build_session_key,
)
from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL  # noqa: E402


def _make_event(text: str = "hello", chat_id: str = "123") -> MessageEvent:
    source = SessionSource(
        platform=MagicMock(value="telegram"),
        chat_id=chat_id,
        chat_type="private",
        user_id="user1",
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg1",
    )


def _make_runner(*, session_id: str = "parent-session") -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._busy_ack_ts = {}
    runner._draining = False
    runner.adapters = {}
    runner.config = MagicMock()
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda _source: True
    runner._busy_input_mode = "interrupt"
    session_key = build_session_key(_make_event().source)
    entry = SimpleNamespace(session_key=session_key, session_id=session_id)
    session_store = SimpleNamespace(
        _lock=threading.Lock(),
        _entries={session_key: entry},
        switch_session=MagicMock(),
    )
    session_store._ensure_loaded_locked = lambda: None
    runner.session_store = session_store
    runner._session_db = MagicMock()
    runner._session_db._db = MagicMock()
    runner._session_db._db.get_compression_lock_holder.return_value = None
    return runner


def _make_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter._pending_messages = {}
    adapter._send_with_retry = AsyncMock()
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter.platform = MagicMock(value="telegram")
    return adapter


def _make_running_parent() -> MagicMock:
    parent = MagicMock()
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent.get_activity_summary.return_value = {
        "api_call_count": 1,
        "max_iterations": 300,
        "current_tool": "terminal",
    }
    return parent


class TestSessionInStartupResume:
    def test_false_when_never_marked(self) -> None:
        runner = _make_runner()
        assert runner._session_in_startup_resume("sk") is False

    def test_true_while_marked_false_after(self) -> None:
        runner = _make_runner()
        runner._startup_resume_active = {"sk"}
        assert runner._session_in_startup_resume("sk") is True
        runner._startup_resume_active.discard("sk")
        assert runner._session_in_startup_resume("sk") is False


class TestBusyHandlerProtectsBootResume:
    @pytest.mark.asyncio
    async def test_does_not_interrupt_recovery_turn(self) -> None:
        """THE incident: user message during a boot-resume turn must queue,
        not abort the recovery."""
        runner = _make_runner()
        adapter = _make_adapter()
        event = _make_event(text="how's it going?")
        sk = build_session_key(event.source)
        parent = _make_running_parent()
        runner._running_agents[sk] = parent
        runner.adapters[event.source.platform] = adapter
        runner._startup_resume_active = {sk}

        handled = await runner._handle_active_session_busy_message(event, sk)

        assert handled is True
        parent.interrupt.assert_not_called()
        assert adapter._pending_messages.get(sk) is event

    @pytest.mark.asyncio
    async def test_ack_names_the_restart_not_generic_interrupt(self) -> None:
        runner = _make_runner()
        adapter = _make_adapter()
        event = _make_event(text="did you restart?")
        sk = build_session_key(event.source)
        parent = _make_running_parent()
        runner._running_agents[sk] = parent
        runner._running_agents_ts[sk] = time.time() - 60
        runner.adapters[event.source.platform] = adapter
        runner._startup_resume_active = {sk}

        with patch("gateway.run.merge_pending_message_event"):
            await runner._handle_active_session_busy_message(event, sk)

        adapter._send_with_retry.assert_called_once()
        content = adapter._send_with_retry.call_args.kwargs.get("content", "")
        assert "restarted" in content.lower()
        assert "resum" in content.lower()  # resuming/resume
        assert "queued" in content.lower()
        assert "/stop" in content
        assert "Interrupting" not in content

    @pytest.mark.asyncio
    async def test_interrupt_still_fires_for_normal_turns(self) -> None:
        """No boot-resume in flight -> interrupt semantics unchanged."""
        runner = _make_runner()
        adapter = _make_adapter()
        event = _make_event(text="please stop")
        sk = build_session_key(event.source)
        parent = _make_running_parent()
        runner._running_agents[sk] = parent
        runner.adapters[event.source.platform] = adapter

        with patch("gateway.run.merge_pending_message_event"):
            await runner._handle_active_session_busy_message(event, sk)

        parent.interrupt.assert_called_once_with("please stop")

    @pytest.mark.asyncio
    async def test_other_sessions_resume_does_not_demote_this_one(self) -> None:
        """The marker is per-session: session X resuming must not demote
        interrupts for session Y."""
        runner = _make_runner()
        adapter = _make_adapter()
        event = _make_event(text="interrupt me")
        sk = build_session_key(event.source)
        parent = _make_running_parent()
        runner._running_agents[sk] = parent
        runner.adapters[event.source.platform] = adapter
        runner._startup_resume_active = {"some:other:session"}

        with patch("gateway.run.merge_pending_message_event"):
            await runner._handle_active_session_busy_message(event, sk)

        parent.interrupt.assert_called_once_with("interrupt me")


class TestRunStartupResumeEventMarksSession:
    @pytest.mark.asyncio
    async def test_marker_set_during_turn_and_cleared_after(self) -> None:
        runner = _make_runner()
        sk = "agent:main:telegram:dm:123:user1"
        observed = {}

        class _Adapter:
            _session_tasks: dict = {}

            async def handle_message(self, event):
                observed["during"] = runner._session_in_startup_resume(sk)

        await runner._run_startup_resume_event(_Adapter(), _make_event(), sk)

        assert observed["during"] is True
        assert runner._session_in_startup_resume(sk) is False

    @pytest.mark.asyncio
    async def test_marker_cleared_even_when_handle_message_raises(self) -> None:
        runner = _make_runner()
        sk = "agent:main:telegram:dm:123:user1"
        runner._release_running_agent_state = MagicMock()

        class _Adapter:
            _session_tasks: dict = {}

            async def handle_message(self, event):
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await runner._run_startup_resume_event(_Adapter(), _make_event(), sk)

        assert runner._session_in_startup_resume(sk) is False
