"""``hermes chat -Q`` as a dispatcher's re-run of a failed bot delivery (#111721).

A failed delivery turn persists its user row before the provider call; the policy-gated re-run
replays the same session and payload in a fresh process. Told so via HERMES_RESUME_UNANSWERED_TURN,
the re-run resumes that row instead of appending a second copy of the DM.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import cli
from agent.context_compressor import _DB_PERSISTED_MARKER
from tools.bot_relay import RESUME_UNANSWERED_TURN_ENV


def _quiet_turn(monkeypatch, history, marker):
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    if marker is None:
        monkeypatch.delenv(RESUME_UNANSWERED_TURN_ENV, raising=False)
    else:
        monkeypatch.setenv(RESUME_UNANSWERED_TURN_ENV, marker)
    seen = {}

    def run_conversation(**kwargs):
        seen["history"] = list(kwargs["conversation_history"])
        seen["env"] = os.environ.get(RESUME_UNANSWERED_TURN_ENV)
        return {"final_response": "ok"}

    agent = SimpleNamespace(run_conversation=run_conversation, session_id="s-1")
    the_cli = SimpleNamespace(agent=agent, conversation_history=history, session_id="s-1")
    try:
        cli._run_quiet_single_query(the_cli, "hello")
    except SystemExit as exc:
        assert exc.code == 0
    return agent, seen


def test_rerun_resumes_the_unanswered_user_row_it_already_persisted(monkeypatch):
    tail = {"role": "user", "content": "hello"}
    agent, seen = _quiet_turn(monkeypatch, [{"role": "assistant", "content": "earlier"}, tail], "1")
    # The persisted row leaves the history and becomes this turn's staged user dict, already
    # durable — so the agent reuses it as the turn's user message and the flush writes no new row.
    assert seen["history"] == [{"role": "assistant", "content": "earlier"}]
    assert agent._pending_cli_user_message is tail
    assert tail[_DB_PERSISTED_MARKER] is True
    assert seen["env"] is None, "the marker is consumed before the turn so tool subprocesses never inherit it"


def test_without_the_marker_or_an_unanswered_tail_nothing_is_adopted(monkeypatch):
    """A person re-sending the same word after a failed turn has sent a second real message; an
    answered tail is not a re-run either. Only the dispatcher's explicit marker plus an identical
    unanswered tail adopts a row."""
    tail = {"role": "user", "content": "hello"}
    agent, seen = _quiet_turn(monkeypatch, [tail], None)
    assert seen["history"] == [tail] and not hasattr(agent, "_pending_cli_user_message")
    assert _DB_PERSISTED_MARKER not in tail

    answered = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "done"}]
    agent, seen = _quiet_turn(monkeypatch, list(answered), "1")
    assert seen["history"] == answered and not hasattr(agent, "_pending_cli_user_message")


def test_rerun_adopts_the_dm_behind_the_failed_attempts_tool_scaffolding(monkeypatch):
    """A 503 after a tool round persisted user + assistant(tool_calls) + tool before the failure text,
    and the dispatcher retries it. The DM is still unanswered: the re-run adopts it and drops the
    failed attempt's scaffolding from the in-memory turn instead of appending a second copy."""
    tail = {"role": "user", "content": "hello"}
    scaffolding = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
                                                                "function": {"name": "t", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
    ]
    agent, seen = _quiet_turn(monkeypatch, [{"role": "assistant", "content": "earlier"}, tail, *scaffolding], "1")
    assert seen["history"] == [{"role": "assistant", "content": "earlier"}]
    assert agent._pending_cli_user_message is tail and tail[_DB_PERSISTED_MARKER] is True


def test_turn_report_is_written_before_the_exit_linger_and_the_path_is_not_inherited(monkeypatch, tmp_path):
    """A spawner that bounds only the turn (cron Bot Chat lane, #113608) reads the outcome from
    HERMES_QUIET_TURN_REPORT_FILE: written the moment the turn ends — before the one-shot exit
    linger — stamped with this pid, and the variable is popped before the turn spawns anything."""
    from hermes_cli import quiet_single_query as qsq

    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    report = tmp_path / "turn.json"
    monkeypatch.setenv(qsq.TURN_REPORT_FILE_ENV, str(report))
    seen = {}

    def run_conversation(**kwargs):
        seen["env_during_turn"] = os.environ.get(qsq.TURN_REPORT_FILE_ENV)
        seen["report_during_turn"] = report.exists()
        return {"final_response": "ok"}

    def linger(*args, **kwargs):
        seen["report_at_linger"] = qsq.read_turn_report(str(report))
        return {"waited": [], "completed": [], "timed_out": []}

    monkeypatch.setattr("tools.process_registry.process_registry.wait_for_pending_completions", linger)
    agent = SimpleNamespace(run_conversation=run_conversation, session_id="s-1")
    try:
        cli._run_quiet_single_query(SimpleNamespace(agent=agent, conversation_history=[], session_id="s-1"), "hello")
    except SystemExit as exc:
        assert exc.code == 0
    assert seen["env_during_turn"] is None and seen["report_during_turn"] is False
    assert seen["report_at_linger"] == {"pid": os.getpid(), "exit_code": 0, "error": ""}
    # Identity is the path, not the writer's pid: the spawner owns a fresh per-delivery temp
    # name and pops it before the turn, and a launcher that re-execs makes the pid unknowable.
    # What must still fail closed is a record that is not a turn report.
    report.write_text('{"pid": 1}', encoding="utf-8")
    assert qsq.read_turn_report(str(report)) is None
    report.write_text("not json", encoding="utf-8")
    assert qsq.read_turn_report(str(report)) is None
    report.unlink()
    assert qsq.read_turn_report(str(report)) is None
