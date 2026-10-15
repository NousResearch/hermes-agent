"""kanban.require_model_selection: fail-closed task-level model routing.

With the policy on, the dispatcher must refuse to claim a ready/review card
that lacks a task-level model, provider or reasoning effort, so a worker never
silently runs on the assignee profile's default model. The check sits in the
shared ``_dispatch_lane_task`` gate, BEFORE the claim, so it covers the gateway
tick, ``hermes kanban dispatch``, the dashboard dispatch button, requeued
retries/reclaims and the review lane alike. Policy off = upstream behavior.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_dispatch_gates as kbg

SELECTED = dict(model_override="loop-model", provider_override="custom", reasoning_effort="medium")


@pytest.fixture
def kanban_home(tmp_path, monkeypatch, all_assignees_spawnable):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _policy(home: Path, on: bool) -> None:
    (home / "config.yaml").write_text(
        f"kanban:\n  require_model_selection: {'true' if on else 'false'}\n", encoding="utf-8",
    )


def _spawner(spawned: list):
    def spawn(task, workspace, board=None):
        spawned.append((task.id, task.model_override, task.provider_override, task.reasoning_effort))
        return 4242
    return spawn


def _kinds(conn, tid):
    return [e.kind for e in kb.list_events(conn, tid)]


@pytest.mark.parametrize("fields, missing", [
    ({}, ["model", "provider", "reasoning"]),
    ({"model_override": "loop-model"}, ["provider", "reasoning"]),
    ({"model_override": "loop-model", "provider_override": "custom"}, ["reasoning"]),
    ({"reasoning_effort": "high"}, ["model", "provider"]),
])
def test_missing_selection_is_held_before_claim(kanban_home, fields, missing):
    _policy(kanban_home, True)
    spawned: list = []
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="unrouted", assignee="worker", **fields)
        res = kbd.dispatch_once(conn, spawn_fn=_spawner(spawned))
        task = kb.get_task(conn, tid)
        events = kb.list_events(conn, tid)
    assert spawned == []
    assert res.spawned == []
    assert res.skipped_model_unselected == [(tid, missing)]
    # Never claimed: no run, no lock, still ready.
    assert task.status == "ready" and task.claim_lock is None and task.current_run_id is None
    assert "claimed" not in [e.kind for e in events]
    held = [e.payload for e in events if e.kind == "skipped_model_unselected"]
    assert held == [{"missing": missing}]
    assert "model_unselected=1" in kbd.describe_suppression([res])


def test_full_selection_spawns_with_task_choice(kanban_home):
    _policy(kanban_home, True)
    spawned: list = []
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="routed", assignee="worker", **SELECTED)
        res = kbd.dispatch_once(conn, spawn_fn=_spawner(spawned))
        task = kb.get_task(conn, tid)
    assert res.skipped_model_unselected == []
    assert spawned == [(tid, "loop-model", "custom", "medium")]
    assert task.status == "running"


def test_policy_off_keeps_upstream_behavior(kanban_home):
    """Explicit false and an absent key both spawn an unrouted card."""
    spawned: list = []
    with kbc.connect() as conn:
        absent = kb.create_task(conn, title="absent key", assignee="worker")
        kbd.dispatch_once(conn, spawn_fn=_spawner(spawned))
        _policy(kanban_home, False)
        off = kb.create_task(conn, title="explicit false", assignee="worker")
        res = kbd.dispatch_once(conn, spawn_fn=_spawner(spawned))
    assert [s[0] for s in spawned] == [absent, off]
    assert res.skipped_model_unselected == []


def test_hold_event_written_once_and_dry_run_writes_nothing(kanban_home):
    _policy(kanban_home, True)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="unrouted", assignee="worker")
        dry = kbd.dispatch_once(conn, dry_run=True)
        assert dry.skipped_model_unselected == [(tid, ["model", "provider", "reasoning"])]
        assert dry.spawned == []
        assert "skipped_model_unselected" not in _kinds(conn, tid)
        for _ in range(3):
            kbd.dispatch_once(conn, spawn_fn=_spawner([]))
        assert _kinds(conn, tid).count("skipped_model_unselected") == 1


def test_selecting_after_hold_releases_card(kanban_home):
    """Operator fixes the card with the native setters; next tick spawns it."""
    _policy(kanban_home, True)
    spawned: list = []
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="later routed", assignee="worker")
        kbd.dispatch_once(conn, spawn_fn=_spawner(spawned))
        assert spawned == []
        assert kb.set_model_override(conn, tid, "loop-model", provider="custom")
        assert kb.set_reasoning_effort(conn, tid, "high")
        kbd.dispatch_once(conn, spawn_fn=_spawner(spawned))
    assert spawned == [(tid, "loop-model", "custom", "high")]


def test_review_lane_is_guarded_and_does_not_reserve_capacity(kanban_home):
    """An unrouted review card is held and must not steal the ready lane's
    single slot via the review reservation."""
    _policy(kanban_home, True)
    spawned: list = []
    with kbc.connect() as conn:
        review = kb.create_task(conn, title="unrouted review", assignee="reviewer")
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (review,))
        ready = kb.create_task(conn, title="routed ready", assignee="worker", **SELECTED)
        # Budget 1: without the guard the review reservation would pin the
        # ready lane to 0 and the routed ready card would starve.
        kbd.dispatch_once(conn, spawn_fn=_spawner(spawned), max_spawn=1)
        assert [s[0] for s in spawned] == [ready]
        # Unbudgeted tick reaches the review row and holds it before claim.
        res = kbd.dispatch_once(conn, spawn_fn=_spawner(spawned))
        review_task = kb.get_task(conn, review)
    assert [s[0] for s in spawned] == [ready]
    assert res.skipped_model_unselected == [(review, ["model", "provider", "reasoning"])]
    assert review_task.status == "review" and review_task.claim_lock is None
    assert review_task.current_run_id is None


def test_routed_review_card_spawns(kanban_home):
    _policy(kanban_home, True)
    spawned: list = []
    with kbc.connect() as conn:
        review = kb.create_task(conn, title="routed review", assignee="reviewer", **SELECTED)
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (review,))
        kbd.dispatch_once(conn, spawn_fn=_spawner(spawned))
    assert spawned == [(review, "loop-model", "custom", "medium")]


def test_reclaimed_retry_is_guarded_again(kanban_home):
    """A retry goes back through the same gate: if the selection was cleared
    while the card was running, the requeued card is held, not respawned on
    the profile default."""
    _policy(kanban_home, True)
    spawned: list = []
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="retry", assignee="worker", **SELECTED)
        kbd.dispatch_once(conn, spawn_fn=_spawner(spawned))
        assert kb.get_task(conn, tid).status == "running"
        assert kb.set_model_override(conn, tid, None)  # clears model + provider
        assert kb.reclaim_task(conn, tid, reason="retry", signal_fn=lambda *a, **k: None)
        assert kb.get_task(conn, tid).status == "ready"
        res = kbd.dispatch_once(conn, spawn_fn=_spawner(spawned))
        task = kb.get_task(conn, tid)
    assert len(spawned) == 1
    assert res.skipped_model_unselected == [(tid, ["model", "provider"])]
    assert task.status == "ready" and task.claim_lock is None


def test_running_card_is_untouched_when_policy_turns_on(kanban_home):
    """In-flight work already claimed under policy-off keeps its run."""
    spawned: list = []
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="in flight", assignee="worker")
        kbd.dispatch_once(conn, spawn_fn=_spawner(spawned))
        before = kb.get_task(conn, tid)
        _policy(kanban_home, True)
        res = kbd.dispatch_once(conn, spawn_fn=_spawner(spawned))
        after = kb.get_task(conn, tid)
        kinds = _kinds(conn, tid)
    assert before.status == after.status == "running"
    assert after.current_run_id == before.current_run_id
    assert after.claim_lock == before.claim_lock
    assert res.skipped_model_unselected == []
    assert "skipped_model_unselected" not in kinds
    assert len(spawned) == 1


def test_unreadable_config_fails_closed(kanban_home, monkeypatch):
    import hermes_cli.config_effective as ce

    def boom(*a, **k):
        raise OSError("unreadable")
    monkeypatch.setattr(ce, "load_user_config_effective", boom)
    assert kbg.require_model_selection_enabled() is True


def test_manual_cli_dispatch_reports_hold(kanban_home, capsys):
    """``hermes kanban dispatch`` uses the same gate and names the hold."""
    from hermes_cli import kanban_ops

    _policy(kanban_home, True)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="manual", assignee="worker", model_override="m")
    args = argparse.Namespace(dry_run=True, max=None, json=True, failure_limit=kbd.DEFAULT_FAILURE_LIMIT)
    assert kanban_ops._cmd_dispatch(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["skipped_model_unselected"] == [{"task_id": tid, "missing": ["provider", "reasoning"]}]
    assert out["spawned"] == []


def test_selected_task_reaches_worker_launch_policy(tmp_path, monkeypatch):
    """End to end on the owner side: a task admitted under the policy carries
    its model/provider/effort into the frozen worker launch policy (the
    per-task equivalent of ``-m/--provider/--reasoning``)."""
    from contextlib import closing

    from gateway.session_contract import Principal
    from gateway.session_kanban import build_kanban_policy

    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD", raising=False)
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: True)
    monkeypatch.setattr(
        "hermes_cli.profiles.resolve_profile_env",
        lambda name: str(tmp_path) if name == "default" else str(tmp_path / name),
    )
    _policy(tmp_path, True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kb.create_board("owned")
    spawned: list = []
    with closing(kbc.connect(board="owned")) as conn:
        tid = kb.create_task(conn, title="routed", assignee="default", workspace_kind="dir",
                             workspace_path=str(workspace), **SELECTED)
        kbd.dispatch_once(conn, board="owned", spawn_fn=_spawner(spawned))
        task = kb.get_task(conn, tid)
    assert spawned == [(tid, "loop-model", "custom", "medium")]
    actor = Principal("owner", str(tmp_path), frozenset({"session:create"}), "transport")
    connection = SimpleNamespace(authority=SimpleNamespace(profile_id=str(tmp_path)), actor=actor, native_owner=True)
    params = dict(board="owned", task_id=tid, run_id=task.current_run_id, claim_lock=task.claim_lock)
    config = {"model": {"default": "profile-default", "provider": "other"},
              "platform_toolsets": {"cli": ["terminal"]}}
    policy, _ = build_kanban_policy(connection, params, config)
    assert policy.model == "loop-model" and policy.provider == "custom"
    assert policy.config()["agent"]["reasoning_effort"] == "medium"
