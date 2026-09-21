"""Review-lane starvation hardening: non-spawnable reviewers are surfaced.

Live incident (t_e649600f): an implementer passed ``reviewer="sdlc-review"``
(a SKILL name, not a profile) so the review task sat in ``status=review`` for
36 days while the dispatcher bucketed it in ``skipped_nonspawnable`` every
tick with no alert — unlike the ready lane, nothing ever pulls a review task,
so the silent skip is starvation, not "correctly idle".

Contract pinned here:
* A review task whose reviewer is not a spawnable profile is reported in
  ``DispatchResult.skipped_review_nonspawnable`` (task id + reviewer name)
  and logged as a WARNING — it must NOT land in the ready-lane
  ``skipped_nonspawnable`` bucket (that bucket means "terminal lane, OK").
* The dispatcher surfaces only; it never auto-reassigns: swapping the
  reviewer silently changes who signs off on the work (Sahil, 2026-09-20).
* ``dry_run`` and real ticks report identically (no DB write in either).
* Healthy review lanes (real reviewer profile) behave exactly as before.

Proven with the REAL dispatch path (stub spawn_fn, not mocked dispatch).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture()
def review_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with one spawnable reviewer profile.

    ``profile_exists`` requires an identity marker: a bare directory is a
    ghost shell, so each synthetic profile gets a ``config.yaml`` (same
    pattern as ``home_with_protected_config``). ``sdlc-review`` gets NO
    directory — it is a skill name, which is exactly the live failure mode.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    reviewer_dir = home / "profiles" / "kensei-review"
    reviewer_dir.mkdir(parents=True)
    (reviewer_dir / "config.yaml").write_text("tier: 1\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return kb, home


@pytest.fixture()
def no_stagger(monkeypatch):
    """No-op the same-profile stagger sleep so real-spawn tests stay fast."""
    monkeypatch.setattr(kbd._kb.time, "sleep", lambda *a, **k: None)


def _recorder():
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append((task.id, task.assignee))
        return 4321  # pretend PID

    return spawns, fake_spawn


def _make_review(kb, conn, reviewer="sdlc-review", title="starved"):
    """A task in the review column owned by *reviewer* (real dispatch path)."""
    tid = kb.create_task(conn, title=title, assignee=reviewer)
    assert kb.request_review(conn, tid, summary="handoff")
    row = conn.execute(
        "SELECT status, assignee FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["status"] == "review"
    assert row["assignee"] == reviewer
    return tid


def test_starved_review_task_is_surfaced_not_silently_skipped(
    review_home, caplog,
):
    """AC1: a review task with a non-existent reviewer appears in dispatch
    diagnostics with its task id and the unresolvable profile name — and is
    NOT misfiled into the ready-lane ``skipped_nonspawnable`` "OK" bucket."""
    _kb, home = review_home
    with _kb.connect() as conn:
        tid = _make_review(_kb, conn)
        res = _kb.dispatch_once(conn, dry_run=True)
    assert res.spawned == []
    assert res.skipped_review_nonspawnable == [(tid, "sdlc-review")]
    assert res.skipped_nonspawnable == []
    with _kb.connect() as conn:
        row = conn.execute(
            "SELECT status, assignee FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
    assert row["status"] == "review"
    assert row["assignee"] == "sdlc-review"
    assert any(
        "REVIEW TASK STARVED" in r.getMessage() and tid in r.getMessage()
        and "sdlc-review" in r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING
    )


def test_real_tick_reports_without_mutating(review_home, no_stagger):
    """The dispatcher surfaces only — a real (non-dry-run) tick reports the
    starved reviewer but never reassigns: who signs off is an operator
    decision, not a dispatcher side effect."""
    _kb, home = review_home
    spawns, fake_spawn = _recorder()
    with _kb.connect() as conn:
        tid = _make_review(_kb, conn)
        res = _kb.dispatch_once(conn, spawn_fn=fake_spawn)
        row = conn.execute(
            "SELECT assignee FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        kinds = [e.kind for e in kb.list_events(conn, tid)]
    assert res.skipped_review_nonspawnable == [(tid, "sdlc-review")]
    assert res.spawned == []
    assert row["assignee"] == "sdlc-review"
    assert "assigned" not in kinds


def test_healthy_review_lane_unchanged(review_home, no_stagger):
    """AC3: a real reviewer profile dispatches exactly as before — no new
    buckets fire."""
    _kb, home = review_home
    spawns, fake_spawn = _recorder()
    with _kb.connect() as conn:
        tid = _make_review(_kb, conn, reviewer="kensei-review")
        res = _kb.dispatch_once(conn, spawn_fn=fake_spawn)
    assert [(t, a) for t, a in spawns] == [(tid, "kensei-review")]
    assert res.skipped_review_nonspawnable == []
    assert res.skipped_nonspawnable == []


def test_ready_lane_nonspawnable_semantics_unchanged(review_home):
    """The ready lane keeps its own contract: a non-spawnable READY assignee
    (terminal lane) still lands in ``skipped_nonspawnable`` untouched."""
    _kb, home = review_home
    with _kb.connect() as conn:
        tid = _kb.create_task(conn, title="terminal lane", assignee="orion-cc")
        res = _kb.dispatch_once(conn, dry_run=True)
    assert res.skipped_nonspawnable == [tid]
    assert res.skipped_review_nonspawnable == []