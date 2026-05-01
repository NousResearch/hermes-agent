"""Regression tests for approval-state cleanup on session boundaries."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key
from tools import approval as approval_mod
from tools.approval import (
    _ApprovalEntry,
    approve_session,
    enable_session_yolo,
    is_approved,
    is_session_yolo_enabled,
)


@pytest.fixture(autouse=True)
def _clear_approval_state():
    approval_mod._gateway_queues.clear()
    approval_mod._gateway_notify_cbs.clear()
    approval_mod._session_approved.clear()
    approval_mod._session_yolo.clear()
    approval_mod._permanent_approved.clear()
    approval_mod._pending.clear()
    yield
    approval_mod._gateway_queues.clear()
    approval_mod._gateway_notify_cbs.clear()
    approval_mod._session_approved.clear()
    approval_mod._session_yolo.clear()
    approval_mod._permanent_approved.clear()
    approval_mod._pending.clear()


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_entry(session_id: str, source: SessionSource | None = None) -> SessionEntry:
    source = source or _make_source()
    return SessionEntry(
        session_key=build_session_key(source),
        session_id=session_id,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=source.platform,
        chat_type=source.chat_type,
    )


def _make_resume_runner():
    from gateway.run import GatewayRunner

    source = _make_source()
    session_key = build_session_key(source)
    current_entry = _make_entry("current-session", source)
    resumed_entry = _make_entry("resumed-session", source)

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner._pending_approvals = {}
    runner._update_prompt_pending = {}
    runner._agent_cache_lock = None
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = current_entry
    runner.session_store.switch_session.return_value = resumed_entry
    runner.session_store.load_transcript.return_value = []
    runner._session_db = MagicMock()
    runner._session_db.resolve_session_by_title.return_value = "resumed-session"
    runner._session_db.get_session_title.return_value = "Resumed Work"
    return runner, session_key


def _make_branch_runner():
    from gateway.run import GatewayRunner

    source = _make_source()
    session_key = build_session_key(source)
    current_entry = _make_entry("current-session", source)
    branched_entry = _make_entry("branched-session", source)

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner.config = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner._pending_approvals = {}
    runner._update_prompt_pending = {}
    runner._agent_cache_lock = None
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = current_entry
    runner.session_store.load_transcript.return_value = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    runner.session_store.switch_session.return_value = branched_entry
    runner._session_db = MagicMock()
    runner._session_db.get_session_title.return_value = "Current Work"
    runner._session_db.get_next_title_in_lineage.return_value = "Current Work #2"
    return runner, session_key


@pytest.mark.asyncio
async def test_resume_clears_session_scoped_approval_and_yolo_state():
    runner, session_key = _make_resume_runner()
    other_key = "agent:main:telegram:dm:other-chat"

    approve_session(session_key, "recursive delete")
    approve_session(other_key, "recursive delete")
    enable_session_yolo(session_key)
    enable_session_yolo(other_key)
    runner._pending_approvals[session_key] = {"command": "rm -rf /tmp/demo"}
    runner._pending_approvals[other_key] = {"command": "rm -rf /tmp/other"}
    runner._update_prompt_pending[session_key] = True
    runner._update_prompt_pending[other_key] = True

    result = await runner._handle_resume_command(_make_event("/resume Resumed Work"))

    assert "Resumed session" in result
    assert is_approved(session_key, "recursive delete") is False
    assert is_session_yolo_enabled(session_key) is False
    assert session_key not in runner._pending_approvals
    assert session_key not in runner._update_prompt_pending
    assert is_approved(other_key, "recursive delete") is True
    assert is_session_yolo_enabled(other_key) is True
    assert other_key in runner._pending_approvals
    assert other_key in runner._update_prompt_pending


@pytest.mark.asyncio
async def test_branch_clears_session_scoped_approval_and_yolo_state():
    runner, session_key = _make_branch_runner()
    other_key = "agent:main:telegram:dm:other-chat"

    approve_session(session_key, "recursive delete")
    approve_session(other_key, "recursive delete")
    enable_session_yolo(session_key)
    enable_session_yolo(other_key)
    runner._pending_approvals[session_key] = {"command": "rm -rf /tmp/demo"}
    runner._pending_approvals[other_key] = {"command": "rm -rf /tmp/other"}
    runner._update_prompt_pending[session_key] = True
    runner._update_prompt_pending[other_key] = True

    result = await runner._handle_branch_command(_make_event("/branch"))

    assert "Branched to" in result
    assert is_approved(session_key, "recursive delete") is False
    assert is_session_yolo_enabled(session_key) is False
    assert session_key not in runner._pending_approvals
    assert session_key not in runner._update_prompt_pending
    assert is_approved(other_key, "recursive delete") is True
    assert is_session_yolo_enabled(other_key) is True
    assert other_key in runner._pending_approvals
    assert other_key in runner._update_prompt_pending


@pytest.mark.asyncio
async def test_branch_preserves_persisted_assistant_metadata():
    runner, _session_key = _make_branch_runner()
    runner.session_store.load_transcript.return_value = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": "world",
            "finish_reason": "stop",
            "reasoning": "thinking",
            "reasoning_content": "provider scratchpad",
            "reasoning_details": [{"type": "summary", "text": "step"}],
            "codex_reasoning_items": [{"id": "r1", "type": "reasoning"}],
            "codex_message_items": [{"id": "m1", "type": "message"}],
        },
    ]

    result = await runner._handle_branch_command(_make_event("/branch"))

    assert "Branched to" in result
    append_calls = runner._session_db.append_message.call_args_list
    assert len(append_calls) == 2
    assistant_kwargs = append_calls[1].kwargs
    assert assistant_kwargs["role"] == "assistant"
    assert assistant_kwargs["finish_reason"] == "stop"
    assert assistant_kwargs["reasoning"] == "thinking"
    assert assistant_kwargs["reasoning_content"] == "provider scratchpad"
    assert assistant_kwargs["reasoning_details"] == [{"type": "summary", "text": "step"}]
    assert assistant_kwargs["codex_reasoning_items"] == [{"id": "r1", "type": "reasoning"}]
    assert assistant_kwargs["codex_message_items"] == [{"id": "m1", "type": "message"}]


def test_clear_session_boundary_security_state_is_scoped():
    """The helper must wipe only the target session's approval/yolo state.

    Also exercises the /new reset path indirectly: /new calls this helper,
    so if the helper is scoped correctly, /new's clearing is correct too.
    """
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._pending_approvals = {}
    runner._update_prompt_pending = {}

    source = _make_source()
    session_key = build_session_key(source)
    other_key = "agent:main:telegram:dm:other-chat"

    approve_session(session_key, "recursive delete")
    approve_session(other_key, "recursive delete")
    enable_session_yolo(session_key)
    enable_session_yolo(other_key)
    runner._pending_approvals[session_key] = {"command": "rm -rf /tmp/demo"}
    runner._pending_approvals[other_key] = {"command": "rm -rf /tmp/other"}
    runner._update_prompt_pending[session_key] = True
    runner._update_prompt_pending[other_key] = True

    runner._clear_session_boundary_security_state(session_key)

    # Target session cleared
    assert is_approved(session_key, "recursive delete") is False
    assert is_session_yolo_enabled(session_key) is False
    assert session_key not in runner._pending_approvals
    assert session_key not in runner._update_prompt_pending
    # Other session untouched
    assert is_approved(other_key, "recursive delete") is True
    assert is_session_yolo_enabled(other_key) is True
    assert other_key in runner._pending_approvals
    assert other_key in runner._update_prompt_pending

    # Empty session_key is a no-op
    runner._clear_session_boundary_security_state("")
    assert is_approved(other_key, "recursive delete") is True
    assert other_key in runner._update_prompt_pending


def test_clear_session_boundary_security_state_wakes_blocked_approvals():
    """Boundary cleanup must cancel blocked approval waiters immediately."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._pending_approvals = {}
    runner._update_prompt_pending = {}

    source = _make_source()
    session_key = build_session_key(source)
    other_key = "agent:main:telegram:dm:other-chat"

    target_entry = _ApprovalEntry({"command": "rm -rf /tmp/demo"})
    other_entry = _ApprovalEntry({"command": "rm -rf /tmp/other"})
    approval_mod._gateway_queues[session_key] = [target_entry]
    approval_mod._gateway_queues[other_key] = [other_entry]

    runner._clear_session_boundary_security_state(session_key)

    assert target_entry.event.is_set()
    assert target_entry.result == "deny"
    assert other_entry.event.is_set() is False
    assert other_entry.result is None
    assert session_key not in approval_mod._gateway_queues
    assert other_key in approval_mod._gateway_queues
