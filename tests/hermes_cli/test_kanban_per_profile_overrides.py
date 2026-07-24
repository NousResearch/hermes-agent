"""Tests for per-profile concurrency cap OVERRIDES in the kanban dispatcher.

``kanban.max_in_progress_per_profile_overrides`` is a dict of
``{profile_name: int}`` that overrides the global
``kanban.max_in_progress_per_profile`` for the listed profiles only:

- Override takes priority over the global cap for listed profiles.
- Unlisted profiles fall back to the global cap.
- When the global cap is unset (null), only listed profiles are capped;
  unlisted profiles run unconstrained.
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest


@pytest.fixture()
def isolated_kanban_home_with_profiles(monkeypatch):
    """Spin up a fresh HERMES_HOME with kanban DB + alpha/beta profiles."""
    test_home = tempfile.mkdtemp(prefix="kanban_per_profile_overrides_test_")
    for prof in ("alpha", "beta", "default"):
        os.makedirs(os.path.join(test_home, "profiles", prof), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", test_home)
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db
    yield kanban_db


def _fake_spawn(*args, **kwargs):
    return 12345


def _seed_tasks(kb, alpha: int = 5, beta: int = 5):
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        for i in range(alpha):
            kb.create_task(conn, title=f"a{i}", assignee="alpha")
        for i in range(beta):
            kb.create_task(conn, title=f"b{i}", assignee="beta")


def test_override_caps_only_listed_profile_when_global_unset(
    isolated_kanban_home_with_profiles,
):
    """Global cap null + override {alpha: 2}: alpha capped at 2, beta
    unconstrained (all 5 dispatch)."""
    kb = isolated_kanban_home_with_profiles
    _seed_tasks(kb)
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            max_in_progress_per_profile=None,
            max_in_progress_per_profile_overrides={"alpha": 2},
        )
    spawn_assignees = [s[1] for s in res.spawned]
    capped_assignees = [c[1] for c in res.skipped_per_profile_capped]
    assert spawn_assignees.count("alpha") == 2
    assert spawn_assignees.count("beta") == 5
    assert capped_assignees.count("alpha") == 3
    assert "beta" not in capped_assignees


def test_override_takes_priority_over_global_cap(
    isolated_kanban_home_with_profiles,
):
    """Global cap=4 but override {alpha: 1}: alpha gets 1 (override wins),
    beta falls back to the global cap of 4."""
    kb = isolated_kanban_home_with_profiles
    _seed_tasks(kb)
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            max_in_progress_per_profile=4,
            max_in_progress_per_profile_overrides={"alpha": 1},
        )
    spawn_assignees = [s[1] for s in res.spawned]
    capped_assignees = [c[1] for c in res.skipped_per_profile_capped]
    assert spawn_assignees.count("alpha") == 1
    assert spawn_assignees.count("beta") == 4
    assert capped_assignees.count("alpha") == 4
    assert capped_assignees.count("beta") == 1


def test_invalid_override_entries_are_ignored(
    isolated_kanban_home_with_profiles,
):
    """Non-numeric / <1 override values are ignored; the valid entry still
    applies. Overrides dict entirely invalid == no overrides."""
    kb = isolated_kanban_home_with_profiles
    _seed_tasks(kb, alpha=3, beta=3)
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            max_in_progress_per_profile_overrides={
                "alpha": 2,
                "beta": "not-a-number",
                "gamma": 0,
            },
        )
    spawn_assignees = [s[1] for s in res.spawned]
    assert spawn_assignees.count("alpha") == 2
    # beta's invalid override is ignored -> unconstrained
    assert spawn_assignees.count("beta") == 3


def test_override_counts_pre_existing_running(
    isolated_kanban_home_with_profiles,
):
    """A task already in 'running' status counts toward the override cap."""
    kb = isolated_kanban_home_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        running_alpha = kb.create_task(conn, title="running alpha", assignee="alpha")
        conn.execute(
            "UPDATE tasks SET status = 'running', claim_lock = 'test:1' WHERE id = ?",
            (running_alpha,),
        )
        for i in range(3):
            kb.create_task(conn, title=f"a{i}", assignee="alpha")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            max_in_progress_per_profile_overrides={"alpha": 1},
        )
    assert not [s for s in res.spawned if s[1] == "alpha"]
    assert len(res.skipped_per_profile_capped) == 3
