from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _spawnable_profiles(monkeypatch, names: set[str] | None = None):
    from hermes_cli import profiles

    allowed = names
    monkeypatch.setattr(
        profiles,
        "profile_exists",
        lambda name: True if allowed is None else name in allowed,
    )


def _positive_budget(monkeypatch):
    monkeypatch.setattr(
        kb,
        "_load_worker_profile_config",
        lambda _assignee: {"agent": {"max_turns": 5}},
    )


def _explanations_by_kind(res):
    return {item["kind"]: item for item in res.ready_explanations}


def test_explains_global_capacity_limit(kanban_home, monkeypatch):
    _spawnable_profiles(monkeypatch)
    _positive_budget(monkeypatch)
    with kb.connect() as conn:
        running = kb.create_task(conn, title="running", assignee="worker")
        assert kb.claim_task(conn, running) is not None
        ready = kb.create_task(conn, title="ready", assignee="worker")

        res = kb.dispatch_once(conn, dry_run=True, max_in_progress=1)

    by_kind = _explanations_by_kind(res)
    assert not res.spawned
    assert by_kind["capacity_limited_global"] == {
        "kind": "capacity_limited_global",
        "limit": 1,
        "running": 1,
        "ready_count": 1,
        "task_ids": [ready],
    }


def test_explains_per_profile_capacity_limit(kanban_home, monkeypatch):
    _spawnable_profiles(monkeypatch)
    _positive_budget(monkeypatch)
    with kb.connect() as conn:
        running = kb.create_task(conn, title="busy", assignee="worker-a")
        assert kb.claim_task(conn, running) is not None
        capped = kb.create_task(conn, title="same profile", assignee="worker-a")

        res = kb.dispatch_once(conn, dry_run=True, max_in_progress_per_profile=1)

    by_kind = _explanations_by_kind(res)
    assert not res.spawned
    assert (capped, "worker-a", 1) in res.skipped_per_profile_capped
    assert by_kind["capacity_limited_per_profile"]["task_ids"] == [capped]
    assert by_kind["capacity_limited_per_profile"]["profiles"] == {
        "worker-a": {"limit": 1, "running": 1, "task_ids": [capped]}
    }


def test_explains_ready_zero_budget_profiles_without_mutating_dry_run(
    kanban_home, monkeypatch
):
    _spawnable_profiles(monkeypatch)
    monkeypatch.setattr(
        kb,
        "_load_worker_profile_config",
        lambda _assignee: {"agent": {"max_turns": 0}},
    )
    with kb.connect() as conn:
        ready = kb.create_task(conn, title="no budget", assignee="worker")

        res = kb.dispatch_once(conn, dry_run=True)
        task = kb.get_task(conn, ready)

    by_kind = _explanations_by_kind(res)
    assert not res.spawned
    assert ready in res.spawn_blocked_zero_budget
    assert task.status == "ready"
    assert by_kind["zero_budget_profile"]["task_ids"] == [ready]
    assert by_kind["zero_budget_profile"]["profiles"] == {
        "worker": {"budget_key": "agent.max_turns", "budget": 0, "task_ids": [ready]}
    }


def test_explains_missing_unmanaged_profiles(kanban_home, monkeypatch):
    _spawnable_profiles(monkeypatch, names={"managed"})
    _positive_budget(monkeypatch)
    with kb.connect() as conn:
        missing = kb.create_task(conn, title="terminal lane", assignee="orion-cc")

        res = kb.dispatch_once(conn, dry_run=True)

    by_kind = _explanations_by_kind(res)
    assert not res.spawned
    assert missing in res.skipped_nonspawnable
    assert by_kind["missing_profile"] == {
        "kind": "missing_profile",
        "task_ids": [missing],
        "assignees": {"orion-cc": [missing]},
    }


def test_explains_ready_cards_waiting_on_incomplete_dependencies(
    kanban_home, monkeypatch
):
    _spawnable_profiles(monkeypatch)
    _positive_budget(monkeypatch)
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = kb.create_task(conn, title="child", assignee="worker", parents=[parent])
        # Simulate a stale/manual ready bit. dispatch --dry-run must explain it
        # without promoting or spawning the dependency-gated child.
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (child,))
        conn.commit()

        res = kb.dispatch_once(conn, dry_run=True)
        child_after = kb.get_task(conn, child)

    by_kind = _explanations_by_kind(res)
    assert child_after.status == "ready"
    assert child not in {tid for tid, _who, _ws in res.spawned}
    assert by_kind["dependency_waiting"] == {
        "kind": "dependency_waiting",
        "task_ids": [child],
        "parents": {child: [parent]},
    }


def test_explains_true_dispatcher_stuck_when_spawnable_work_makes_no_progress(
    kanban_home, monkeypatch
):
    _spawnable_profiles(monkeypatch)
    _positive_budget(monkeypatch)
    with kb.connect() as conn:
        ready = kb.create_task(conn, title="should spawn", assignee="worker")
        monkeypatch.setattr(kb, "claim_task", lambda *_args, **_kwargs: None)

        res = kb.dispatch_once(conn, board="default", spawn_fn=lambda *_args, **_kwargs: 123)

    by_kind = _explanations_by_kind(res)
    assert not res.spawned
    assert by_kind["dispatcher_stuck"]["task_ids"] == [ready]
    assert any(
        "hermes kanban dispatch --board default --dry-run --explain" in cmd
        for cmd in by_kind["dispatcher_stuck"]["suggested_commands"]
    )


def test_dispatch_explain_cli_supports_board_after_subcommand(
    kanban_home, monkeypatch
):
    from hermes_cli import config

    _spawnable_profiles(monkeypatch)
    _positive_budget(monkeypatch)
    monkeypatch.setattr(config, "load_config", lambda: {"kanban": {"max_in_progress": 1}})
    with kb.connect() as conn:
        running = kb.create_task(conn, title="busy", assignee="worker")
        assert kb.claim_task(conn, running) is not None
        kb.create_task(conn, title="queued", assignee="worker")

    out = kc.run_slash("dispatch --board default --dry-run --explain --json")

    assert not out.startswith("⚠ /kanban usage error")
    payload = json.loads(out)
    assert payload["explanations"][0]["kind"] == "capacity_limited_global"
    assert payload["explanations"][0]["limit"] == 1


def test_gateway_capacity_explanation_is_not_stuck_warning():
    from gateway import kanban_watchers as gw

    capacity = gw._format_dispatch_health_warning(
        "quick-move",
        6,
        [{"kind": "capacity_limited_per_profile", "profiles": {"reels": {"running": 2, "limit": 2}}}],
    )
    stuck = gw._format_dispatch_health_warning(
        "quick-move",
        6,
        [{"kind": "dispatcher_stuck", "task_ids": ["t_ready"]}],
    )

    assert "dispatcher stuck" not in capacity.lower()
    assert "capacity-limited" in capacity
    assert "dispatcher stuck" in stuck.lower()
    assert "hermes kanban dispatch --board quick-move --dry-run --explain" in stuck


def test_parallel_reroute_replacements_use_capacity_model_without_promoting_children(
    kanban_home, monkeypatch
):
    _spawnable_profiles(monkeypatch)
    _positive_budget(monkeypatch)
    with kb.connect() as conn:
        busy = kb.create_task(conn, title="busy worker-a", assignee="worker-a")
        assert kb.claim_task(conn, busy) is not None
        capped_replacement = kb.create_task(
            conn, title="replacement a", assignee="worker-a"
        )
        healthy_replacement = kb.create_task(
            conn, title="replacement b", assignee="worker-b"
        )
        downstream = kb.create_task(
            conn,
            title="synthesis waits",
            assignee="synth",
            parents=[capped_replacement, healthy_replacement],
        )

        res = kb.dispatch_once(conn, dry_run=True, max_in_progress_per_profile=1)
        downstream_after = kb.get_task(conn, downstream)

    spawned_ids = {tid for tid, _who, _ws in res.spawned}
    by_kind = _explanations_by_kind(res)
    assert healthy_replacement in spawned_ids
    assert capped_replacement not in spawned_ids
    assert by_kind["capacity_limited_per_profile"]["task_ids"] == [capped_replacement]
    assert downstream_after.status == "todo"
