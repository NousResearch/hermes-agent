"""A goal_mode Kanban worker keeps the live tool feed in its worker log.

The dispatcher used to force ``-Q`` on goal_mode cards because the judge loop only ran in
cli.py's fully-quiet branch; ``-Q`` strips every tool callback, so the dashboard's Worker log
stayed blank for the whole run while non-goal cards logged normally (Discord report, Sep 2026).
The loop now also runs on the ``-q`` path, driving follow-up turns through ``cli.chat``.
"""

from __future__ import annotations

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
