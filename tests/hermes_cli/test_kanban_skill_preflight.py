"""Dispatcher skill preflight: a required skill the assignee profile cannot
load holds the card BEFORE claim (#121652) instead of surfacing as a
post-claim worker crash or a silent partial load.

The preflight resolves through ``build_preloaded_skills_prompt`` — the same
loader the worker's ``--skills`` preload runs — under the assignee's profile
home, so a verdict can never disagree with what the spawned worker sees.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _install_skill(home: Path, name: str) -> None:
    skill_dir = home / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test fixture skill\n---\n\nDo the thing.\n",
        encoding="utf-8",
    )


def _events_of(conn, task_id: str, kind: str) -> list:
    return [e for e in kb.list_events(conn, task_id) if e.kind == kind]


def test_missing_skill_holds_card_before_claim(kanban_home, all_assignees_spawnable):
    """A ready card naming a skill the profile home cannot load is held:
    no claim, no spawn, one skill_preflight event."""
    conn = kbc.connect()
    spawned = []
    try:
        tid = kb.create_task(
            conn,
            title="needs missing skill",
            assignee="default",
            workspace_kind="scratch",
            skills=["__no_such_skill__"],
        )
        result = kbd.dispatch_once(
            conn,
            spawn_fn=lambda task, workspace: spawned.append(task.id) or 4242,
        )
        assert not spawned
        assert result.skill_preflight_held == [(tid, "__no_such_skill__")]
        assert kb.get_task(conn, tid).status == "ready"
        events = _events_of(conn, tid, "skill_preflight")
        assert len(events) == 1
        assert events[0].payload == {"lane": "ready", "missing": ["__no_such_skill__"]}
    finally:
        conn.close()


def test_hold_event_written_once_per_distinct_missing_set(
    kanban_home, all_assignees_spawnable
):
    """A held card re-checks every tick; an unchanged verdict must not
    re-write task_events (the respawn_guarded noise shape, #121651)."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(
            conn,
            title="still missing",
            assignee="default",
            workspace_kind="scratch",
            skills=["__no_such_skill__"],
        )
        kbd.dispatch_once(conn, spawn_fn=lambda task, workspace: 4242)
        kbd.dispatch_once(conn, spawn_fn=lambda task, workspace: 4242)
        kbd.dispatch_once(conn, spawn_fn=lambda task, workspace: 4242)
        assert len(_events_of(conn, tid, "skill_preflight")) == 1
    finally:
        conn.close()


def test_installed_skill_spawns_normally(kanban_home, all_assignees_spawnable):
    """A card whose skills all resolve from the profile home spawns unchanged."""
    _install_skill(kanban_home, "demo")
    conn = kbc.connect()
    spawned = []
    try:
        tid = kb.create_task(
            conn,
            title="has skill",
            assignee="default",
            workspace_kind="scratch",
            skills=["demo"],
        )
        result = kbd.dispatch_once(
            conn,
            spawn_fn=lambda task, workspace: spawned.append(task.id) or 4242,
        )
        assert spawned == [tid]
        assert not result.skill_preflight_held
        assert kb.get_task(conn, tid).status == "running"
    finally:
        conn.close()


def test_held_card_recovers_once_skill_installed(kanban_home, all_assignees_spawnable):
    """The hold is not a deadlock: installing the skill lets the next tick
    claim and spawn with no retry budget spent."""
    conn = kbc.connect()
    spawned = []
    spawn = lambda task, workspace: spawned.append(task.id) or 4242  # noqa: E731
    try:
        tid = kb.create_task(
            conn,
            title="later installed",
            assignee="default",
            workspace_kind="scratch",
            skills=["late-skill"],
        )
        kbd.dispatch_once(conn, spawn_fn=spawn)
        assert not spawned
        _install_skill(kanban_home, "late-skill")
        kbd.dispatch_once(conn, spawn_fn=spawn)
        assert spawned == [tid]
        assert kb.get_task(conn, tid).consecutive_failures == 0
    finally:
        conn.close()


def test_disabled_skill_counts_as_missing(kanban_home, all_assignees_spawnable):
    """skills.disabled uses the worker's own membership semantics (disabled =
    missing for the preload path), so the dispatcher holds the same card the
    worker would have skipped."""
    _install_skill(kanban_home, "demo")
    (kanban_home / "config.yaml").write_text(
        'skills:\n  disabled: ["demo"]\n',
        encoding="utf-8",
    )
    conn = kbc.connect()
    try:
        tid = kb.create_task(
            conn,
            title="disabled skill",
            assignee="default",
            workspace_kind="scratch",
            skills=["demo"],
        )
        result = kbd.dispatch_once(conn, spawn_fn=lambda task, workspace: 4242)
        assert result.skill_preflight_held == [(tid, "demo")]
        assert kb.get_task(conn, tid).status == "ready"
    finally:
        conn.close()


def test_review_lane_holds_without_sdlc_review(kanban_home, all_assignees_spawnable):
    """The review lane force-loads sdlc-review after claim today; the
    preflight models that, so a reviewer home without it is held too."""
    conn = kbc.connect()
    spawned = []
    try:
        tid = kb.create_task(
            conn,
            title="review without bundle",
            assignee="default",
            workspace_kind="scratch",
        )
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='review' WHERE id=?", (tid,))
        result = kbd.dispatch_once(
            conn,
            spawn_fn=lambda task, workspace: spawned.append(task.id) or 4242,
        )
        assert not spawned
        assert result.skill_preflight_held == [(tid, "sdlc-review")]
        events = _events_of(conn, tid, "skill_preflight")
        assert events[0].payload == {"lane": "review", "missing": ["sdlc-review"]}
    finally:
        conn.close()


def test_review_lane_spawns_when_sdlc_review_available(
    kanban_home, all_assignees_spawnable
):
    """With sdlc-review resolvable, a review card spawns with the skill
    appended exactly as before the preflight existed."""
    _install_skill(kanban_home, "sdlc-review")
    conn = kbc.connect()
    spawned = []
    try:
        tid = kb.create_task(
            conn,
            title="review with bundle",
            assignee="default",
            workspace_kind="scratch",
        )
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='review' WHERE id=?", (tid,))
        kbd.dispatch_once(
            conn,
            spawn_fn=lambda task, workspace: spawned.append(task.skills) or 4242,
        )
        assert spawned == [["sdlc-review"]]
    finally:
        conn.close()


def test_card_without_skills_spawns_unchecked(kanban_home, all_assignees_spawnable):
    """Skills-less cards (the common case) never touch the resolver."""
    conn = kbc.connect()
    spawned = []
    try:
        tid = kb.create_task(
            conn, title="plain", assignee="default", workspace_kind="scratch"
        )
        kbd.dispatch_once(
            conn,
            spawn_fn=lambda task, workspace: spawned.append(task.id) or 4242,
        )
        assert spawned == [tid]
    finally:
        conn.close()


def test_unresolvable_profile_fails_open(kanban_home, all_assignees_spawnable):
    """When the assignee home cannot be resolved, the preflight steps aside
    (fail open) — the worker's own error reporting stays the backstop."""
    conn = kbc.connect()
    spawned = []
    try:
        tid = kb.create_task(
            conn,
            title="ghost profile",
            assignee="ghost-profile",
            workspace_kind="scratch",
            skills=["anything"],
        )
        result = kbd.dispatch_once(
            conn,
            spawn_fn=lambda task, workspace: spawned.append(task.id) or 4242,
        )
        assert spawned == [tid]
        assert not result.skill_preflight_held
    finally:
        conn.close()


def test_any_spawnable_review_false_when_skill_held(
    kanban_home, all_assignees_spawnable
):
    """A skill-held review row must not consume the ready-lane capacity
    reservation (one such row would pin ready_budget to 0)."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(
            conn,
            title="held review",
            assignee="default",
            workspace_kind="scratch",
        )
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='review' WHERE id=?", (tid,))
        assert not kbd._any_spawnable_review(conn, kbd._lane_rows(conn, "review"))
        _install_skill(kanban_home, "sdlc-review")
        assert kbd._any_spawnable_review(conn, kbd._lane_rows(conn, "review"))
    finally:
        conn.close()


def test_describe_suppression_names_skill_preflight(
    kanban_home, all_assignees_spawnable
):
    """The zero-spawn warning line explains a skill-preflight hold the same
    way it explains respawn-guard reasons (#111910)."""
    conn = kbc.connect()
    try:
        kb.create_task(
            conn,
            title="held",
            assignee="default",
            workspace_kind="scratch",
            skills=["__no_such_skill__"],
        )
        result = kbd.dispatch_once(conn, spawn_fn=lambda task, workspace: 4242)
        line = kbd.describe_suppression([result])
        assert "skill_preflight=1" in line
    finally:
        conn.close()
