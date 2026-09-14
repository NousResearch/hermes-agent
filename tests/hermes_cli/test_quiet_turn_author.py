"""``hermes chat -Q`` passes the dispatcher's HERMES_TURN_AUTHOR to ``run_conversation`` as ``turn_author``.

A bot-to-bot delivery runs the recipient's turn as a ``-Q`` subprocess with that variable set.
A human's ``-Q`` run has it unset and the turn stays unattributed.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

import cli
from agent.turn_author import TURN_AUTHOR_ENV

AUTHOR = {"id": "bot:coder", "name": "coder", "is_bot": True}


@pytest.fixture(autouse=True)
def _plain_one_shot_env(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)


def _run(monkeypatch, env_value, run_conversation=None):
    """One quiet turn with HERMES_TURN_AUTHOR set to ``env_value`` (unset for None); returns the recorded call kwargs."""
    if env_value is None:
        monkeypatch.delenv(TURN_AUTHOR_ENV, raising=False)
    else:
        monkeypatch.setenv(TURN_AUTHOR_ENV, env_value)
    recorded = []

    def record(**kwargs):
        recorded.append(kwargs)
        return {"final_response": "ok"}

    agent = SimpleNamespace(run_conversation=run_conversation or record, session_id="s-1")
    with pytest.raises(SystemExit) as exc:
        cli._run_quiet_single_query(SimpleNamespace(agent=agent, conversation_history=[], session_id="s-1"), "hello")
    assert exc.value.code == 0
    return recorded


def test_quiet_one_shot_passes_turn_author_from_env(monkeypatch, capsys):
    recorded = _run(monkeypatch, json.dumps(AUTHOR))
    assert recorded == [{"user_message": "hello", "conversation_history": [], "turn_author": AUTHOR}]
    assert capsys.readouterr().out.strip() == "ok"


def test_quiet_one_shot_consumes_the_variable_before_the_turn(monkeypatch):
    """Tool subprocesses spawned during the turn must not see the dispatcher's author."""
    seen = {}

    def run_conversation(**kwargs):
        seen["env"] = os.environ.get(TURN_AUTHOR_ENV)
        return {"final_response": "ok"}

    _run(monkeypatch, json.dumps(AUTHOR), run_conversation)
    assert seen["env"] is None
    assert TURN_AUTHOR_ENV not in os.environ


def test_quiet_one_shot_resumes_nested_notify_on_this_session_not_parent(monkeypatch, capsys):
    """A nested Bot Mode completion keyed to B wakes B even when the parent env still names A.

    ``chat -Q`` used to inherit the dispatcher's HERMES_SESSION_KEY and exit after the
    dispatch ack, so C's reply was saved but B never resumed.
    """
    from tools.approval_context import get_current_session_key
    from tools.process_registry import process_registry

    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_SESSION_KEY", "session-A")
    calls = []

    def run_conversation(**kwargs):
        calls.append({"key": get_current_session_key(), "msg": kwargs["user_message"]})
        if len(calls) == 1:
            process_registry.completion_queue.put({
                "type": "completion",
                "session_id": "proc-c",
                "session_key": "session-B",
                "command": "message_agent C",
                "exit_code": 0,
                "output": "marker-from-C",
            })
            return {"final_response": "sent, finish your turn"}
        return {"final_response": "C said marker-from-C"}

    agent = SimpleNamespace(run_conversation=run_conversation, session_id="session-B")
    try:
        with pytest.raises(SystemExit) as exc:
            cli._run_quiet_single_query(
                SimpleNamespace(agent=agent, conversation_history=[], session_id="session-B"),
                "ask C",
            )
        assert exc.value.code == 0
        assert calls[0] == {"key": "session-B", "msg": "ask C"}
        assert calls[1]["key"] == "session-B"
        assert "marker-from-C" in calls[1]["msg"]
        assert capsys.readouterr().out.strip() == "C said marker-from-C"
    finally:
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()


def test_quiet_notify_loop_shares_one_linger_budget(monkeypatch):
    """One stuck notify_on_complete child is waited on once, not once per round plus finalize.

    The loop shares a single deadline across rounds and stops after draining once a
    wait times out; the finalize pass is skipped entirely because the loop ran.
    """
    from hermes_cli import quiet_single_query as qsq
    from tools import process_registry as pr

    waits = []

    def fake_wait(task_id=None, *, timeout=None, poll_interval=1.0):
        waits.append(timeout)
        # First wait: one stuck process times out; second wait (same run): nothing pending.
        return {"waited": ["proc-stuck"] if len(waits) == 1 else [], "completed": [], "timed_out": ["proc-stuck"] if len(waits) == 1 else []}

    monkeypatch.setattr(pr.process_registry, "wait_for_pending_completions", fake_wait)
    monkeypatch.setattr(pr.process_registry, "drain_notifications", lambda *a, **k: [])
    monkeypatch.setattr(qsq.time, "monotonic", lambda: 100.0)

    calls = []
    result = qsq.continue_quiet_notify_completions(
        "session-B", lambda text: calls.append(text) or {"final_response": text}, linger_budget=600.0,
    )
    # Round 1: full budget; timed out -> drained (empty) -> break. Exactly one wait call.
    assert waits == [600.0]
    assert calls == []
    assert result is None
