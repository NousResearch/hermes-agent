"""Regression coverage for the active TodoStore stop gate.

The guard protects Hermes-owned execution state, rather than relying on a
model's wording or provider-specific finish reason.  This is the case that
the intent-ack detector alone cannot cover: ``Need commit.`` after a test
passes, while the commit/CI todo remains in progress.
"""

import json
from types import SimpleNamespace

from agent.agent_runtime_helpers import (
    build_todo_completion_exhausted_message,
    build_todo_completion_nudge,
    should_continue_for_active_todos,
    todo_completion_continuation_enabled,
)
from tools.todo_tool import TodoStore


def _agent(*, enabled=True, todos=None):
    store = TodoStore()
    if todos is not None:
        store.write(todos)
    return SimpleNamespace(
        _todo_completion_continuation=enabled,
        _todo_store=store,
    )


ACTIVE = [
    {"id": "s4-ci", "content": "Commit, push, and verify CI", "status": "in_progress"},
]


def test_active_todo_turn_catches_short_need_commit_stop():
    agent = _agent(todos=ACTIVE)
    messages = [{"role": "user", "content": "finish the release"}]

    assert should_continue_for_active_todos(agent, "Need commit.", messages)
    assert "s4-ci" in build_todo_completion_nudge(agent)


def test_pending_todo_also_blocks_a_premature_stop():
    agent = _agent(todos=[
        {"id": "release", "content": "publish the release", "status": "pending"},
    ])
    messages = [{"role": "user", "content": "finish the release"}]

    assert should_continue_for_active_todos(agent, "Ready for the next step.", messages)


def test_active_todo_turn_catches_ci_queued_wait_stop():
    agent = _agent(todos=ACTIVE)
    messages = [{"role": "user", "content": "finish the release"}]

    assert should_continue_for_active_todos(agent, "CI queued, wait.", messages)


def test_notified_background_work_is_an_intentional_pause():
    agent = _agent(todos=ACTIVE)
    messages = [
        {"role": "user", "content": "wait for CI"},
        {
            "role": "tool",
            "content": json.dumps(
                {"output": "Background process started", "notify_on_complete": True}
            ),
        },
    ]

    assert not should_continue_for_active_todos(agent, "CI is running.", messages)


def test_historical_background_work_does_not_disable_later_todo_guard():
    agent = _agent(todos=ACTIVE)
    messages = [
        {
            "role": "tool",
            "content": json.dumps(
                {"output": "Background process started", "notify_on_complete": True}
            ),
        },
        {"role": "user", "content": "continue the release"},
    ]

    assert should_continue_for_active_todos(agent, "Need commit.", messages)


def test_direct_user_approval_or_input_request_is_a_valid_stop():
    agent = _agent(todos=ACTIVE)
    messages = [{"role": "user", "content": "finish the release"}]

    assert not should_continue_for_active_todos(
        agent, "I need your approval before pushing the release branch.", messages
    )
    assert not should_continue_for_active_todos(
        agent, "需要你的确认才能推送到生产分支。", messages
    )


def test_only_active_todos_trigger_and_config_can_opt_out():
    completed = [
        {"id": "done", "content": "already verified", "status": "completed"},
        {"id": "cancel", "content": "not needed", "status": "cancelled"},
    ]
    messages = [{"role": "user", "content": "finish the release"}]

    assert not should_continue_for_active_todos(_agent(todos=completed), "Need commit.", messages)
    assert not should_continue_for_active_todos(_agent(enabled=False, todos=ACTIVE), "Need commit.", messages)
    assert todo_completion_continuation_enabled(SimpleNamespace())


def test_exhaustion_message_reports_unresolved_todo_ids():
    agent = _agent(todos=ACTIVE)

    message = build_todo_completion_exhausted_message(agent)

    assert "incomplete" in message.lower()
    assert "s4-ci" in message
