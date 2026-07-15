from hermes_state import AsyncSessionDB
"""Regression tests for approval-state cleanup on session boundaries."""

import threading
from contextvars import Context
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import (
    CapabilityEpochRotationBlocked,
    SessionEntry,
    SessionSource,
    build_session_key,
)
from tools import approval as approval_mod
from tools import slash_confirm as slash_confirm_mod
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
    approval_mod._session_approved_generations.clear()
    approval_mod._session_yolo.clear()
    approval_mod._session_yolo_generations.clear()
    approval_mod._session_authority_generations.clear()
    approval_mod._retired_session_capability_epochs.clear()
    approval_mod._plan_capabilities.clear()
    approval_mod._permanent_approved.clear()
    approval_mod._pending.clear()
    slash_confirm_mod._pending.clear()
    yield
    approval_mod._gateway_queues.clear()
    approval_mod._gateway_notify_cbs.clear()
    approval_mod._session_approved.clear()
    approval_mod._session_approved_generations.clear()
    approval_mod._session_yolo.clear()
    approval_mod._session_yolo_generations.clear()
    approval_mod._session_authority_generations.clear()
    approval_mod._retired_session_capability_epochs.clear()
    approval_mod._plan_capabilities.clear()
    approval_mod._permanent_approved.clear()
    approval_mod._pending.clear()
    slash_confirm_mod._pending.clear()


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
    runner.session_store.load_transcript.return_value = []
    runner._session_db = AsyncSessionDB(MagicMock())
    runner._session_db._db.resolve_session_by_title.return_value = "resumed-session"
    runner._session_db._db.get_session_title.return_value = "Resumed Work"
    # The resumed session is live and shares the caller's origin, so the
    # /resume IDOR guard authorizes it (this test covers the post-resume
    # security-state clearing, not the ownership check).
    runner._gateway_session_origin_for_id = lambda sid: source
    original_clear = runner._clear_session_boundary_security_state
    runner._clear_session_boundary_security_state = MagicMock(
        wraps=original_clear
    )

    def _switch_with_writer_callback(*_args, **_kwargs):
        runner._clear_session_boundary_security_state(session_key)
        return resumed_entry

    runner.session_store.switch_session.side_effect = _switch_with_writer_callback
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
    runner._session_db = AsyncSessionDB(MagicMock())
    runner._session_db._db.get_session_title.return_value = "Current Work"
    runner._session_db._db.get_next_title_in_lineage.return_value = "Current Work #2"
    runner._session_db._db.set_session_title.return_value = True
    runner._session_db._db.delete_session.return_value = True
    original_clear = runner._clear_session_boundary_security_state
    runner._clear_session_boundary_security_state = MagicMock(
        wraps=original_clear
    )

    def _switch_with_writer_callback(*_args, **_kwargs):
        runner._clear_session_boundary_security_state(session_key)
        return branched_entry

    runner.session_store.switch_session.side_effect = _switch_with_writer_callback
    return runner, session_key


@pytest.mark.asyncio
async def test_resume_clears_session_scoped_approval_and_yolo_state():
    runner, session_key = _make_resume_runner()
    other_key = "agent:main:telegram:dm:other-chat"

    runner._pending_skills_reload_notes = {
        session_key: "[USER INITIATED SKILLS RELOAD: target]",
        other_key: "[USER INITIATED SKILLS RELOAD: other]",
    }
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
    assert session_key not in runner._pending_skills_reload_notes
    assert is_approved(other_key, "recursive delete") is True
    assert is_session_yolo_enabled(other_key) is True
    assert other_key in runner._pending_approvals
    assert other_key in runner._update_prompt_pending
    assert other_key in runner._pending_skills_reload_notes
    # SessionStore's privileged pre-publication callback owns the only clear;
    # the handler must not perform a racy late clear after publication.
    runner._clear_session_boundary_security_state.assert_called_once_with(
        session_key
    )


@pytest.mark.asyncio
async def test_branch_clears_session_scoped_approval_and_yolo_state():
    runner, session_key = _make_branch_runner()
    other_key = "agent:main:telegram:dm:other-chat"

    runner._pending_skills_reload_notes = {
        session_key: "[USER INITIATED SKILLS RELOAD: target]",
        other_key: "[USER INITIATED SKILLS RELOAD: other]",
    }
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
    assert session_key not in runner._pending_skills_reload_notes
    assert is_approved(other_key, "recursive delete") is True
    assert is_session_yolo_enabled(other_key) is True
    assert other_key in runner._pending_approvals
    assert other_key in runner._update_prompt_pending
    assert other_key in runner._pending_skills_reload_notes
    runner._clear_session_boundary_security_state.assert_called_once_with(
        session_key
    )


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
    append_calls = runner._session_db._db.append_message.call_args_list
    assert len(append_calls) == 2
    assistant_kwargs = append_calls[1].kwargs
    assert assistant_kwargs["role"] == "assistant"
    assert assistant_kwargs["finish_reason"] == "stop"
    assert assistant_kwargs["reasoning"] == "thinking"
    assert assistant_kwargs["reasoning_content"] == "provider scratchpad"
    assert assistant_kwargs["reasoning_details"] == [{"type": "summary", "text": "step"}]
    assert assistant_kwargs["codex_reasoning_items"] == [{"id": "r1", "type": "reasoning"}]
    assert assistant_kwargs["codex_message_items"] == [{"id": "m1", "type": "message"}]


@pytest.mark.asyncio
async def test_branch_writer_outage_removes_staged_artifact():
    runner, session_key = _make_branch_runner()
    runner.session_store.switch_session.side_effect = (
        CapabilityEpochRotationBlocked("writer unavailable")
    )

    result = await runner._handle_branch_command(_make_event("/branch"))

    assert "temporarily blocked" in result
    created_id = runner._session_db._db.create_session.call_args.kwargs[
        "session_id"
    ]
    runner._session_db._db.delete_session.assert_called_once_with(created_id)
    assert session_key not in runner._running_agents


@pytest.mark.asyncio
async def test_branch_reports_unconfirmed_staged_artifact_cleanup():
    runner, _session_key = _make_branch_runner()
    runner.session_store.switch_session.side_effect = (
        CapabilityEpochRotationBlocked("writer unavailable")
    )
    runner._session_db._db.delete_session.return_value = False

    result = await runner._handle_branch_command(_make_event("/branch"))

    assert "operator cleanup is required" in result
    runner._session_db._db.delete_session.assert_called_once()


@pytest.mark.asyncio
async def test_branch_does_not_clear_a_concurrent_turn_slot():
    runner, session_key = _make_branch_runner()
    active_agent = MagicMock()
    runner._running_agents[session_key] = active_agent

    result = await runner._handle_branch_command(_make_event("/branch"))

    assert "blocked" in result.lower()
    assert runner._running_agents[session_key] is active_agent
    runner._session_db._db.create_session.assert_not_called()
    runner.session_store.switch_session.assert_not_called()


def test_clear_session_boundary_security_state_is_scoped():
    """The helper must wipe only the target session's approval/yolo state.

    Also exercises the /new reset path indirectly: /new calls this helper,
    so if the helper is scoped correctly, /new's clearing is correct too.
    """
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._pending_approvals = {}
    runner._update_prompt_pending = {}
    runner._pending_skills_reload_notes = {}

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
    runner._pending_skills_reload_notes[session_key] = (
        "[USER INITIATED SKILLS RELOAD: target]"
    )
    runner._pending_skills_reload_notes[other_key] = (
        "[USER INITIATED SKILLS RELOAD: other]"
    )

    async def _target_handler(choice):
        return f"target:{choice}"

    async def _other_handler(choice):
        return f"other:{choice}"

    slash_confirm_mod.register(session_key, "confirm-target", "reload-mcp", _target_handler)
    slash_confirm_mod.register(other_key, "confirm-other", "reload-mcp", _other_handler)

    runner._clear_session_boundary_security_state(session_key)

    # Target session cleared
    assert is_approved(session_key, "recursive delete") is False
    assert is_session_yolo_enabled(session_key) is False
    assert session_key not in runner._pending_approvals
    assert session_key not in runner._update_prompt_pending
    assert session_key not in runner._pending_skills_reload_notes
    assert slash_confirm_mod.get_pending(session_key) is None
    # Other session untouched
    assert is_approved(other_key, "recursive delete") is True
    assert is_session_yolo_enabled(other_key) is True
    assert other_key in runner._pending_approvals
    assert other_key in runner._update_prompt_pending
    assert other_key in runner._pending_skills_reload_notes
    assert slash_confirm_mod.get_pending(other_key) is not None

    # Empty session_key is a no-op
    runner._clear_session_boundary_security_state("")
    assert is_approved(other_key, "recursive delete") is True
    assert other_key in runner._update_prompt_pending
    assert other_key in runner._pending_skills_reload_notes
    assert slash_confirm_mod.get_pending(other_key) is not None


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


def test_resolver_pop_race_cannot_reauthorize_after_boundary(monkeypatch):
    """A resolver detached just before clear must remain bound to old generation."""

    session_key = "agent:main:discord:thread:channel-1:thread-1"
    notified = threading.Event()
    resolver_popped = threading.Event()
    allow_resolver_signal = threading.Event()
    decision_holder = {}

    class _PausingEvent:
        def __init__(self):
            self._event = threading.Event()

        def set(self):
            resolver_popped.set()
            assert allow_resolver_signal.wait(timeout=2)
            self._event.set()

        def wait(self, timeout=None):
            return self._event.wait(timeout)

        def is_set(self):
            return self._event.is_set()

    monkeypatch.setattr(
        approval_mod,
        "_get_approval_config",
        lambda: {"gateway_timeout": 5},
    )

    def _waiter():
        decision_holder["decision"] = approval_mod._await_gateway_decision(
            session_key,
            lambda _data: notified.set(),
            {
                "command": "rm -rf /tmp/demo",
                "pattern_key": "recursive delete",
                "pattern_keys": ["recursive delete"],
                "description": "dangerous command",
            },
            include_authority_fence=True,
        )

    waiter = threading.Thread(target=_waiter)
    waiter.start()
    assert notified.wait(timeout=2)
    with approval_mod._lock:
        entry = approval_mod._gateway_queues[session_key][0]
        entry.event = _PausingEvent()
        old_generation = entry.authority_generation

    resolver = threading.Thread(
        target=approval_mod.resolve_gateway_approval,
        args=(session_key, "session"),
    )
    resolver.start()
    assert resolver_popped.wait(timeout=2)

    # The resolver already removed the entry, reproducing the original race.
    approval_mod.clear_session_local(session_key)
    allow_resolver_signal.set()
    resolver.join(timeout=2)
    waiter.join(timeout=2)

    assert not resolver.is_alive()
    assert not waiter.is_alive()
    decision = decision_holder["decision"]
    assert decision["authority_stale"] is True
    assert decision["choice"] == "deny"
    assert approval_mod.approve_session(
        session_key,
        "recursive delete",
        expected_generation=old_generation,
    ) is False
    assert approval_mod.is_approved(session_key, "recursive delete") is False


def test_retired_old_context_cannot_read_successor_authority_or_grant_local_plan(
    monkeypatch,
):
    """Successor state with the same keys is invisible to the retired worker."""

    from gateway.session_context import set_session_vars

    session_key = "agent:main:discord:thread:channel-2:thread-2"
    old_epoch = "a" * 64
    new_epoch = "b" * 64
    pattern_key = "recursive delete"
    command = "rm -rf /tmp/demo"
    old_context = Context()
    new_context = Context()

    def _bind(epoch):
        set_session_vars(
            session_key=session_key,
            session_id="session-old" if epoch == old_epoch else "session-new",
            capability_epoch_sha256=epoch,
            user_id="owner-1",
        )

    old_context.run(_bind, old_epoch)
    old_generation, _ = old_context.run(
        approval_mod.capture_session_authority_fence,
        session_key,
    )
    old_context.run(
        approval_mod.clear_session_local,
        session_key,
        retire_capability_epoch_sha256=old_epoch,
    )

    new_context.run(_bind, new_epoch)
    new_generation, _ = new_context.run(
        approval_mod.capture_session_authority_fence,
        session_key,
    )
    assert new_generation != old_generation
    assert new_context.run(
        approval_mod.approve_session,
        session_key,
        pattern_key,
        expected_generation=new_generation,
    ) is True
    assert new_context.run(
        approval_mod.enable_session_yolo,
        session_key,
        expected_generation=new_generation,
    ) is True
    assert new_context.run(
        approval_mod.is_approved,
        session_key,
        pattern_key,
    ) is True
    assert new_context.run(
        approval_mod.is_session_yolo_enabled,
        session_key,
    ) is True

    assert old_context.run(
        approval_mod.is_approved,
        session_key,
        pattern_key,
    ) is False
    assert old_context.run(
        approval_mod.is_session_yolo_enabled,
        session_key,
    ) is False
    with pytest.raises(PermissionError, match="retired"):
        old_context.run(
            approval_mod.capture_session_authority_fence,
            session_key,
        )

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"approvals": {"plan_owner_user_ids": ["owner-1"]}},
    )
    with pytest.raises(PermissionError, match="retired"):
        old_context.run(
            approval_mod.grant_plan_capability,
            session_key=session_key,
            plan_id="plan:old-worker",
            exact_commands=[command],
            approved_by_user_id="owner-1",
        )

    successor_grant = new_context.run(
        approval_mod.grant_plan_capability,
        session_key=session_key,
        plan_id="plan:successor",
        exact_commands=[command],
        approved_by_user_id="owner-1",
    )
    assert successor_grant["state"] == "granted"
    assert old_context.run(
        approval_mod.consume_plan_capability,
        session_key,
        command,
    ) is None
    assert new_context.run(
        approval_mod.consume_plan_capability,
        session_key,
        command,
    ) == "plan:successor"
