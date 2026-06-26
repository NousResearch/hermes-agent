"""Tests for the kanban worker-lane registry and dispatcher routing.

Covers:
* the registry itself (register / get / normalize / validate / clear);
* the dispatcher helpers `_assignee_is_spawnable` / `_resolve_spawn_fn`;
* dispatch routing: a registered external-lane assignee is spawnable and
  routes to the lane's spawn_fn, while an unknown assignee is still bucketed
  as skipped_nonspawnable; profile assignees and explicit spawn_fn overrides
  keep working unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import worker_lanes as wl


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB and one real profile dir."""
    home = tmp_path / ".hermes"
    home.mkdir()
    # A real profile directory so profile_exists("alpha") resolves True.
    (home / "profiles" / "alpha").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture(autouse=True)
def _clean_lanes():
    """Keep the process-local lane registry isolated per test."""
    wl.clear_worker_lanes()
    yield
    wl.clear_worker_lanes()


def _stub_lane(name="junie-x", *, max_concurrency=None, calls=None):
    def spawn(task, workspace, *, board=None):
        if calls is not None:
            calls.append(task.id)
        return 4242

    return wl.WorkerLane(
        name=name,
        spawn_fn=spawn,
        kind="stub",
        max_concurrency=max_concurrency,
    )


# ── registry ──────────────────────────────────────────────────────


def test_register_get_and_normalize():
    lane = _stub_lane("junie-x")
    wl.register_worker_lane(lane)
    assert wl.get_worker_lane("junie-x") is lane
    assert wl.get_worker_lane("JUNIE-X") is lane  # name is normalized
    assert wl.is_worker_lane_assignee("junie-x")
    assert wl.get_worker_lane("nope") is None
    assert wl.get_worker_lane("") is None
    assert [l.name for l in wl.list_worker_lanes()] == ["junie-x"]


def test_duplicate_rejected_unless_replace():
    wl.register_worker_lane(_stub_lane("dup"))
    with pytest.raises(ValueError):
        wl.register_worker_lane(_stub_lane("dup"))
    # replace=True overwrites
    wl.register_worker_lane(_stub_lane("dup"), replace=True)


def test_clear_worker_lanes():
    wl.register_worker_lane(_stub_lane("a"))
    wl.register_worker_lane(_stub_lane("b"))
    wl.clear_worker_lanes()
    assert wl.list_worker_lanes() == []


def test_worker_lane_validation():
    ok = lambda *a, **k: None  # noqa: E731
    with pytest.raises(ValueError):
        wl.WorkerLane(name="x", spawn_fn="not-callable")
    with pytest.raises(ValueError):
        wl.WorkerLane(name="x", spawn_fn=ok, max_concurrency=0)
    with pytest.raises(ValueError):
        wl.WorkerLane(name="BAD NAME", spawn_fn=ok)
    # kind is optional now — empty kind is valid
    assert wl.WorkerLane(name="ok", spawn_fn=ok).kind == ""


# ── dispatcher helpers ────────────────────────────────────────────


def test_resolve_spawn_fn_prefers_lane_then_default():
    lane = _stub_lane("junie-x")
    wl.register_worker_lane(lane)
    assert kb._resolve_spawn_fn("junie-x", None) is lane.spawn_fn
    assert kb._resolve_spawn_fn("alpha", None) is kb._default_spawn  # no lane → default
    override = lambda *a, **k: 1  # noqa: E731
    assert kb._resolve_spawn_fn("junie-x", override) is override  # explicit wins


def test_kanban_worker_env_sets_contract(kanban_home, tmp_path):
    import types

    task = types.SimpleNamespace(
        id="t_1", current_run_id=7, claim_lock="host:1:abc", tenant=None
    )
    env = wl.kanban_worker_env(task, str(tmp_path), base={})
    assert env["HERMES_KANBAN_TASK"] == "t_1"
    assert env["HERMES_KANBAN_WORKSPACE"] == str(tmp_path)
    assert env["HERMES_KANBAN_RUN_ID"] == "7"
    assert env["HERMES_KANBAN_CLAIM_LOCK"] == "host:1:abc"
    assert env["HERMES_KANBAN_DB"]  # resolved
    assert env["HERMES_KANBAN_BOARD"]
    assert env["TERMINAL_CWD"] == str(tmp_path)  # real abs dir
    # lane-specific vars are the caller's responsibility, not the helper's
    assert "HERMES_PROFILE" not in env
    assert "HERMES_WORKER_LANE" not in env


def test_assignee_is_spawnable(kanban_home):
    assert kb._assignee_is_spawnable("alpha")       # profile dir exists
    assert not kb._assignee_is_spawnable("ghost")   # neither profile nor lane
    assert not kb._assignee_is_spawnable("")
    wl.register_worker_lane(_stub_lane("junie-x"))
    assert kb._assignee_is_spawnable("junie-x")      # registered lane


# ── dispatch routing ──────────────────────────────────────────────


def test_dispatch_routes_lane_assignee_and_skips_unknown(kanban_home):
    wl.register_worker_lane(_stub_lane("junie-x"))
    with kb.connect() as conn:
        t_lane = kb.create_task(conn, title="lane task", assignee="junie-x")
        t_prof = kb.create_task(conn, title="profile task", assignee="alpha")
        t_ghost = kb.create_task(conn, title="ghost task", assignee="ghost")
    with kb.connect() as conn:
        res = kb.dispatch_once(conn, dry_run=True)

    spawned_ids = [s[0] for s in res.spawned]
    assert t_lane in spawned_ids       # external lane → spawnable
    assert t_prof in spawned_ids       # Hermes profile → unchanged
    assert t_ghost not in spawned_ids
    assert t_ghost in res.skipped_nonspawnable


def test_dispatch_honors_lane_max_concurrency(kanban_home):
    wl.register_worker_lane(_stub_lane("junie-x", max_concurrency=1))
    with kb.connect() as conn:
        for i in range(3):
            kb.create_task(conn, title=f"t{i}", assignee="junie-x")
    with kb.connect() as conn:
        res = kb.dispatch_once(conn, dry_run=True)

    assert len(res.spawned) == 1
    assert len(res.skipped_per_profile_capped) == 2


def test_unregistered_lane_assignee_is_skipped(kanban_home):
    # No lane registered → the same assignee is non-spawnable.
    with kb.connect() as conn:
        t = kb.create_task(conn, title="orphan", assignee="junie-x")
    with kb.connect() as conn:
        res = kb.dispatch_once(conn, dry_run=True)
    assert t in res.skipped_nonspawnable
    assert t not in [s[0] for s in res.spawned]
