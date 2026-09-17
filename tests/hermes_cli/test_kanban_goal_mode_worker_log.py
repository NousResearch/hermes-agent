"""A goal_mode Kanban worker keeps the live tool feed in its worker log.

The dispatcher used to force ``-Q`` on goal_mode cards because the judge loop only ran in
cli.py's fully-quiet branch; ``-Q`` strips every tool callback, so the dashboard's Worker log
stayed blank for the whole run while non-goal cards logged normally (Discord report, Sep 2026).
The loop now also runs on the ``-q`` path, driving follow-up turns through ``cli.chat``.
"""

from __future__ import annotations

from types import SimpleNamespace

import cli
from hermes_cli import goals
from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli.kanban_db import Task


def _task(**overrides) -> Task:
    base = dict(
        id="t_goal1", title="ship it", body="acceptance: tests pass", assignee="worker",
        status="running", priority=0, created_by=None, created_at=1, started_at=None,
        completed_at=None, workspace_kind="scratch", workspace_path=None,
        claim_lock=None, claim_expires=None, tenant=None,
    )
    base.update(overrides)
    return Task(**base)


def test_goal_mode_worker_takes_the_same_stdout_rich_path_as_a_one_shot_worker(monkeypatch):
    monkeypatch.setattr(dispatch, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(dispatch, "_resolve_worker_cli_toolsets", lambda home: None)
    plain = dispatch._worker_argv(_task(goal_mode=False), "worker", None)
    goal = dispatch._worker_argv(_task(goal_mode=True), "worker", None)
    # The mode travels in HERMES_KANBAN_GOAL_MODE; the argv must not opt the worker out of
    # the tool feed that the Worker log is made of.
    assert goal == plain
    assert "-Q" not in goal


def test_non_quiet_one_shot_runs_the_goal_loop_through_cli_chat(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_goal1")
    monkeypatch.setenv("HERMES_KANBAN_GOAL_MODE", "1")
    monkeypatch.setattr(cli, "_should_seed_interactive", lambda *a, **k: False)
    monkeypatch.setattr(cli, "_collect_query_images", lambda q, i: (q, []))
    monkeypatch.setattr(cli, "_collect_kanban_task_images", lambda imgs: [])
    monkeypatch.setattr(cli, "_finalize_single_query", lambda c: None)

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("hermes_cli.kanban_db_connect.connect_closing", lambda: _Conn())
    monkeypatch.setattr("hermes_cli.kanban_db.get_task", lambda conn, tid: _task(goal_mode=True))

    def _fake_loop(*, run_turn, first_response, log, **_kw):
        # One continuation turn: it must go through cli.chat (tool feed on stdout), and the
        # verdict line must reach stdout too, not only agent.log.
        assert first_response == "first answer"
        assert run_turn("keep going") == "second answer"
        log("kanban goal loop: turn 1/20 verdict=continue reason=x")
        return {"turns": 1}

    monkeypatch.setattr(goals, "run_kanban_goal_loop", _fake_loop)

    chats: list[str] = []

    def _chat(message, images=None):
        chats.append(message)
        return "first answer" if len(chats) == 1 else "second answer"

    stub = SimpleNamespace(
        _single_query_mode=False,
        _claim_active_session=lambda *a, **k: True,
        console=SimpleNamespace(print=lambda *a, **k: None),
        _show_security_advisories=lambda: None,
        chat=_chat,
        _print_exit_summary=lambda **k: None,
        _last_turn_result={"final_response": "second answer", "completed": True},
    )
    printed: list[str] = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(" ".join(str(x) for x in a)))
    try:
        cli._run_single_query_mode(stub, "work kanban task t_goal1", None, False, True)
    except SystemExit as exc:
        assert exc.code == 0
    assert chats == ["work kanban task t_goal1", "keep going"]
    assert any("verdict=continue" in line for line in printed)
