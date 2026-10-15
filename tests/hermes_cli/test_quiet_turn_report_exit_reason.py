"""``-Q`` turn report carries the loop's typed ``turn_exit_reason``.

A budget-exhausted ``hermes chat --query-file … --oneshot -Q`` child is
indistinguishable from any other rc!=0 death to its spawner: the loop's typed
``turn_exit_reason`` reaches the result dict but the only machine-readable
export that carries it, ``--usage-file``, is wired solely into the top-level
``-z`` contract. The quiet path owns the spawner-facing channel —
``HERMES_QUIET_TURN_REPORT_FILE``, written before the linger exit, pid-stamped,
consumed via ``_report_turn`` / ``run_reported_turn`` — so the typed reason
rides it: one additive field, no new surface.

(The lane's older bot-replay coverage lived in test_quiet_single_query.py,
retired upstream in 8637219a9; these are the report-contract tests only.)
"""

from __future__ import annotations

import os
import threading
from types import SimpleNamespace

import cli


def test_write_turn_report_carries_the_typed_turn_exit_reason(tmp_path):
    """Budget exhaustion is invisible to a `-q` spawner (the usage-file report is -z-only), so
    the turn report carries the loop's typed outcome verbatim."""
    from hermes_cli.quiet_single_query import write_turn_report, read_turn_report
    report = tmp_path / "turn.json"
    write_turn_report(str(report), exit_code=1, error="boom", reply="",
                      turn_exit_reason="max_iterations_reached(60/60)")
    assert read_turn_report(str(report), os.getpid()) == {
        "pid": os.getpid(), "exit_code": 1, "error": "boom", "reply": "",
        "turn_exit_reason": "max_iterations_reached(60/60)"}
    # A writer that knows nothing about the field still produces a valid record.
    other = tmp_path / "old.json"
    write_turn_report(str(other), exit_code=0)
    assert read_turn_report(str(other), os.getpid())["turn_exit_reason"] == ""


def test_the_quiet_run_stamps_the_report_with_the_failing_turns_exit_reason(monkeypatch, tmp_path):
    """The typed reason survives from the turn result into the report the spawner reads: a
    max_turns death arrives as failed + max_iterations_reached(...), not bare unknown."""
    from hermes_cli import quiet_single_query as qsq

    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    report = tmp_path / "turn.json"
    monkeypatch.setenv(qsq.TURN_REPORT_FILE_ENV, str(report))

    def run_conversation(**kwargs):
        return {"final_response": "", "failed": True, "error": "agent turn did not run cleanly",
                "turn_exit_reason": "max_iterations_reached(3/3)"}

    monkeypatch.setattr("tools.process_registry.process_registry.wait_for_pending_completions",
                        lambda *a, **k: {"waited": [], "completed": [], "timed_out": []})
    agent = SimpleNamespace(run_conversation=run_conversation, session_id="s-1")
    try:
        cli._run_quiet_single_query(SimpleNamespace(agent=agent, conversation_history=[], session_id="s-1"), "hello")
    except SystemExit as exc:
        assert exc.code == 1, "a failed turn still exits 1; the reason is additive"
    assert qsq.read_turn_report(str(report), os.getpid())["turn_exit_reason"] == "max_iterations_reached(3/3)"


def test_run_reported_turn_carries_the_typed_reason_on_the_return(monkeypatch, tmp_path):
    """The report file is the child's channel; spawners only see the CompletedProcess (the relay
    lane unlinks the report, the cron lane branches on returncode). The typed reason must ride the
    return on BOTH bookings: the lingering-child booking and the child-that-exited booking."""
    import subprocess as sp
    from hermes_cli import quiet_single_query as qsq

    # Fake child runs at *this* pid so read_turn_report's pid check passes against real files.
    # --- lingering booking: the child never exits; the report carries the turn.
    live = SimpleNamespace(pid=os.getpid(), returncode=None, kill=lambda: None,
                           communicate=lambda: ("", ""))
    monkeypatch.setattr(qsq.subprocess, "Popen", lambda *a, **k: live)
    monkeypatch.setattr(threading, "Thread", lambda *a, **k: SimpleNamespace(
        start=lambda: None, is_alive=lambda: True, join=lambda timeout=None: None))
    path = str(tmp_path / "t.json")
    qsq.write_turn_report(path, exit_code=1, error="out of budget", reply="",
                          turn_exit_reason="max_iterations_reached(3/3)")
    booked = qsq.run_reported_turn(["hermes", "chat", "-Q"], env={}, report_path=path,
                                   timeout=5.0, exit_grace=0.0)
    assert booked.returncode == 1 and booked.stdout == "" and booked.stderr == "out of budget"
    assert booked.turn_exit_reason == "max_iterations_reached(3/3)"

    # --- exited booking: the child exited; the reason comes from the report read after the drain.
    exited = SimpleNamespace(pid=os.getpid(), returncode=1, kill=lambda: None,
                             communicate=lambda: ("answer", "tail"))
    monkeypatch.setattr(qsq.subprocess, "Popen", lambda *a, **k: exited)
    monkeypatch.setattr(threading, "Thread",
                        lambda target=None, **k: SimpleNamespace(
                            start=target, is_alive=lambda: False, join=lambda timeout=None: None))
    done_path = str(tmp_path / "e.json")
    qsq.write_turn_report(done_path, exit_code=1, error="", reply="answer",
                          turn_exit_reason="max_iterations_reached(60/60)")
    done = qsq.run_reported_turn(["hermes", "chat", "-Q"], env={}, report_path=done_path,
                                 timeout=5.0)
    assert isinstance(done, sp.CompletedProcess)
    assert done.returncode == 1 and done.stdout == "answer"
    assert done.turn_exit_reason == "max_iterations_reached(60/60)"
    # A child that died before stamping any report still returns cleanly with the unknown.
    blank = qsq.run_reported_turn(["hermes", "chat", "-Q"], env={}, report_path=str(tmp_path / "n.json"),
                                  timeout=5.0)
    assert blank.turn_exit_reason == ""
