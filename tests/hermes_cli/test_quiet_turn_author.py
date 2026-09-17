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
    """Round 2 waits only the REMAINING budget, not a fresh one per round.

    A fresh per-round deadline (the pre-fix bug) would pass round 1 as 600 and
    round 2 as 600 again; the shared deadline yields 600 then the remainder.
    """
    from hermes_cli import quiet_single_query as qsq
    from tools import process_registry as pr

    waits = []

    def fake_wait(task_id=None, *, timeout=None, poll_interval=1.0):
        waits.append(timeout)
        return {"waited": [], "completed": [], "timed_out": []}

    # Round 1 drains one text so a round 2 happens; the clock advances 300s across
    # the follow-up turn, so round 2 must wait only the remaining 300s.
    rounds = [[({"type": "completion", "session_key": "session-B"}, "[IMPORTANT: reply from C]")], []]

    def fake_drain(*a, **k):
        return rounds.pop(0) if rounds else []

    clock = iter([100.0, 100.0, 400.0, 400.0])
    monkeypatch.setattr(qsq.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(pr.process_registry, "wait_for_pending_completions", fake_wait)
    monkeypatch.setattr(pr.process_registry, "drain_notifications", fake_drain)

    calls = []
    result = qsq.continue_quiet_notify_completions(
        "session-B", lambda text: calls.append(text) or {"final_response": text}, linger_budget=600.0,
    )
    assert waits == [600.0, 300.0], "round 2 must wait the remaining budget, not a fresh one"
    assert len(calls) == 1
    assert result == {"final_response": calls[0]}


def test_quiet_notify_loop_stops_after_timeout_with_drained_texts(monkeypatch):
    """A timed-out process is waited on no further: drained texts still run, then the loop stops."""
    from hermes_cli import quiet_single_query as qsq
    from tools import process_registry as pr

    waits = []

    def fake_wait(task_id=None, *, timeout=None, poll_interval=1.0):
        waits.append(timeout)
        return {"waited": ["proc-stuck"], "completed": [], "timed_out": ["proc-stuck"]}

    monkeypatch.setattr(pr.process_registry, "wait_for_pending_completions", fake_wait)
    monkeypatch.setattr(
        pr.process_registry, "drain_notifications",
        lambda *a, **k: [({"type": "completion", "session_key": "session-B"}, "[IMPORTANT: reply from C]")],
    )
    monkeypatch.setattr(qsq.time, "monotonic", lambda: 100.0)

    calls = []
    result = qsq.continue_quiet_notify_completions(
        "session-B", lambda text: calls.append(text) or {"final_response": text}, linger_budget=600.0,
    )
    # One wait only: timed out -> drained text ran -> break (no second 600s wait).
    assert waits == [600.0]
    assert calls == ["[IMPORTANT: reply from C]"]
    assert result == {"final_response": "[IMPORTANT: reply from C]"}


def test_finalize_linger_skipped_after_quiet_notify_loop(monkeypatch):
    """_wait_for_oneshot_background_completions must not re-wait once the notify loop consumed the budget."""
    from tools import process_registry as pr

    waited = []
    monkeypatch.setattr(pr.process_registry, "wait_for_pending_completions",
                        lambda *a, **k: waited.append(1) or {"waited": [], "completed": [], "timed_out": []})

    cli = SimpleNamespace(_quiet_notify_linger_done=True)
    cli._wait_for_oneshot_background_completions(cli) if hasattr(cli, "_wait_for_oneshot_background_completions") else None
    # The real entrypoint is the module function; call it through the CLI module.
    import cli as cli_mod

    cli_mod._wait_for_oneshot_background_completions(cli)
    assert waited == [], "finalize must skip the re-wait when the quiet notify loop already ran"


def test_quiet_notify_loop_injects_owned_async_delegation_events(monkeypatch):
    """Owned async_delegation results must not be consumed-and-dropped by the loop.

    drain_notifications pops owned events of every type; the quiet loop is this
    one-shot's only consumer, so an owned async_delegation text is injected as a
    follow-up turn exactly like a completion.
    """
    from hermes_cli import quiet_single_query as qsq
    from tools import process_registry as pr

    monkeypatch.setattr(pr.process_registry, "wait_for_pending_completions",
                        lambda *a, **k: {"waited": [], "completed": [], "timed_out": []})
    # A real drain pops owned events; round 2 finds the queue empty and the loop returns.
    events = [({"type": "async_delegation", "session_key": "session-B"}, "[IMPORTANT: delegated reply from C]")]

    def fake_drain(*a, **k):
        return [events.pop(0)] if events else []

    monkeypatch.setattr(pr.process_registry, "drain_notifications", fake_drain)

    seen = []
    result = qsq.continue_quiet_notify_completions(
        "session-B", lambda text: seen.append(text) or {"final_response": text}, linger_budget=0.0,
    )
    assert seen == ["[IMPORTANT: delegated reply from C]"]
    assert result == {"final_response": "[IMPORTANT: delegated reply from C]"}
