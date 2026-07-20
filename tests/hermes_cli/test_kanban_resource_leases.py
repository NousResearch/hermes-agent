"""Cross-board exclusive resource lease behavior for Kanban tasks."""

from __future__ import annotations

from contextlib import contextmanager
import threading
import time
import json
import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db(board="default")
    kb.init_db(board="other")
    return home


def _create(board: str, title: str, keys=()):
    with kb.connect_closing(board=board) as conn:
        return kb.create_task(
            conn,
            title=title,
            assignee="worker",
            resource_keys=keys,
            board=board,
        )


def test_resource_keys_are_normalized_bounded_and_round_trip(kanban_home):
    with kb.connect_closing(board="default") as conn:
        tid = kb.create_task(
            conn,
            title="normalized",
            resource_keys=[" GPU:0 ", "gpu:0", "Control-Plane:CTO"],
        )
        task = kb.get_task(conn, tid)
        assert task.resource_keys == ["control-plane:cto", "gpu:0"]

        with pytest.raises(ValueError, match="at most"):
            kb.set_resource_keys(
                conn,
                tid,
                [f"resource:{i}" for i in range(kb.MAX_RESOURCE_KEYS + 1)],
            )


def test_same_key_is_exclusive_across_two_board_databases(kanban_home):
    first = _create("default", "first", ["control-plane:cto"])
    second = _create("other", "second", ["control-plane:cto"])

    with kb.connect_closing(board="default") as conn:
        assert kb.claim_task(conn, first, board="default") is not None
    with kb.connect_closing(board="other") as conn:
        assert kb.claim_task(conn, second, board="other") is None
        task = kb.get_task(conn, second)
        assert task.status == "ready"
        conflict = kb.get_resource_conflict(conn, second, board="other")
        assert conflict.resource_key == "control-plane:cto"
        assert conflict.holder_board == "default"
        assert conflict.holder_task_id == first


def test_concurrent_cross_board_claim_race_has_one_winner(kanban_home):
    tids = {
        "default": _create("default", "a", ["gpu:0"]),
        "other": _create("other", "b", ["gpu:0"]),
    }
    barrier = threading.Barrier(2)
    outcomes: dict[str, bool] = {}

    def claim(board: str):
        with kb.connect_closing(board=board) as conn:
            barrier.wait()
            outcomes[board] = kb.claim_task(conn, tids[board], board=board) is not None

    threads = [threading.Thread(target=claim, args=(board,)) for board in tids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not any(thread.is_alive() for thread in threads)
    assert sum(outcomes.values()) == 1


def test_opposite_key_order_is_deadlock_free_and_atomic(kanban_home):
    tids = {
        "default": _create("default", "a", ["gpu:0", "browser:shared"]),
        "other": _create("other", "b", ["browser:shared", "gpu:0"]),
    }
    barrier = threading.Barrier(2)
    outcomes: list[bool] = []

    def claim(board: str):
        with kb.connect_closing(board=board) as conn:
            barrier.wait()
            outcomes.append(kb.claim_task(conn, tids[board], board=board) is not None)

    threads = [threading.Thread(target=claim, args=(board,)) for board in tids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not any(thread.is_alive() for thread in threads)
    assert sum(outcomes) == 1
    leases = kb.list_resource_leases()
    assert {lease.resource_key for lease in leases} == {"browser:shared", "gpu:0"}
    assert len({lease.owner_token for lease in leases}) == 1


def test_resource_key_update_racing_claim_cannot_create_unleased_run(
    kanban_home, monkeypatch
):
    task_id = _create("default", "racy", ["gpu:0"])
    original = kb._acquire_resource_leases

    def acquire_then_change(**kwargs):
        conflict = original(**kwargs)
        assert conflict is None
        with kb.connect_closing() as updater:
            assert kb.set_resource_keys(updater, task_id, ["gpu:1"])
        return None

    monkeypatch.setattr(kb, "_acquire_resource_leases", acquire_then_change)
    with kb.connect_closing() as conn:
        assert kb.claim_task(conn, task_id, board="default") is None
        task = kb.get_task(conn, task_id)
        assert task.status == "ready"
        assert task.resource_keys == ["gpu:1"]
    assert kb.list_resource_leases() == []


def test_independent_keys_and_no_key_tasks_keep_parallelism(kanban_home):
    keyed_a = _create("default", "a", ["gpu:0"])
    keyed_b = _create("other", "b", ["gpu:1"])
    plain = _create("other", "plain")

    with kb.connect_closing(board="default") as conn:
        assert kb.claim_task(conn, keyed_a, board="default") is not None
    with kb.connect_closing(board="other") as conn:
        assert kb.claim_task(conn, keyed_b, board="other") is not None
        assert kb.claim_task(conn, plain, board="other") is not None

    assert {lease.resource_key for lease in kb.list_resource_leases()} == {
        "gpu:0",
        "gpu:1",
    }


def test_spawn_failure_releases_lease_without_counting_conflict(
    kanban_home, all_assignees_spawnable
):
    first = _create("default", "first", ["browser:shared"])
    second = _create("other", "second", ["browser:shared"])

    def fail_spawn(task, workspace, board=None):
        raise RuntimeError("spawn failed")

    with kb.connect_closing(board="default") as conn:
        result = kb.dispatch_once(
            conn,
            spawn_fn=fail_spawn,
            board="default",
            failure_limit=5,
        )
        assert kb.get_task(conn, first).consecutive_failures == 1
        assert not result.resource_conflicts

    with kb.connect_closing(board="other") as conn:
        assert kb.claim_task(conn, second, board="other") is not None


def test_stale_reclaim_releases_lease_for_successor(kanban_home):
    first = _create("default", "first", ["gpu:0"])
    second = _create("other", "second", ["gpu:0"])
    with kb.connect_closing(board="default") as conn:
        claimed = kb.claim_task(conn, first, board="default", ttl_seconds=1)
        assert claimed is not None
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET claim_expires=? WHERE id=?",
                (int(time.time()) - 1, first),
            )
        assert kb.release_stale_claims(conn) == 1

    with kb.connect_closing(board="other") as conn:
        assert kb.claim_task(conn, second, board="other") is not None


def test_heartbeat_renews_every_resource_lease_before_board_claim(kanban_home):
    task_id = _create("default", "heartbeat", ["gpu:0", "browser:shared"])
    with kb.connect_closing() as conn:
        claimed = kb.claim_task(
            conn, task_id, board="default", claimer="host:worker", ttl_seconds=60
        )
        assert claimed is not None
        assert kb.heartbeat_claim(
            conn, task_id, claimer="host:worker", ttl_seconds=3600
        )
        renewed = kb.get_task(conn, task_id)
        assert renewed is not None
    leases = kb.list_resource_leases()
    assert len(leases) == 2
    assert {lease.claim_expires for lease in leases} == {renewed.claim_expires}


def test_fenced_release_cannot_delete_successor_lease(kanban_home):
    first = _create("default", "first", ["gpu:0"])
    second = _create("other", "second", ["gpu:0"])
    with kb.connect_closing(board="default") as conn:
        old = kb.claim_task(conn, first, board="default")
        assert old is not None
        old_lock, old_run = old.claim_lock, old.current_run_id
        assert kb.reclaim_task(conn, first)
    with kb.connect_closing(board="other") as conn:
        successor = kb.claim_task(conn, second, board="other")
        assert successor is not None

    assert not kb.release_resource_leases(
        task_id=first,
        owner_token=old_lock,
        run_id=old_run,
        board="default",
    )
    leases = kb.list_resource_leases()
    assert len(leases) == 1
    assert leases[0].holder_task_id == second


def test_archive_and_restart_prune_release_inactive_lease(kanban_home):
    task_id = _create("default", "archived", ["vm:prod/api"])
    with kb.connect_closing() as conn:
        assert kb.claim_task(conn, task_id, board="default") is not None
        assert kb.archive_task(conn, task_id)
    assert kb.list_resource_leases() == []

    task_id = _create("default", "crashed between cleanup phases", ["vm:prod/api"])
    with kb.connect_closing() as conn:
        assert kb.claim_task(conn, task_id, board="default") is not None
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='ready', claim_lock=NULL, "
                "claim_expires=NULL WHERE id=?",
                (task_id,),
            )
    assert kb.prune_inactive_resource_leases() == 1
    assert kb.list_resource_leases() == []


def test_review_claim_honors_cross_board_resource_lease(kanban_home):
    holder = _create("default", "holder", ["review-env:shared"])
    review = _create("other", "review", ["review-env:shared"])
    with kb.connect_closing(board="other") as conn:
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='review' WHERE id=?", (review,))
    with kb.connect_closing(board="default") as conn:
        assert kb.claim_task(conn, holder, board="default") is not None
    with kb.connect_closing(board="other") as conn:
        assert kb.claim_review_task(conn, review, board="other") is None
    with kb.connect_closing(board="default") as conn:
        assert kb.complete_task(conn, holder, summary="released")
    with kb.connect_closing(board="other") as conn:
        assert kb.claim_review_task(conn, review, board="other") is not None


def test_dispatcher_restart_observes_existing_cross_board_lease(kanban_home):
    first = _create("default", "first", ["gpu:0"])
    second = _create("other", "second", ["gpu:0"])
    with kb.connect_closing(board="default") as conn:
        assert kb.claim_task(conn, first, board="default") is not None

    kb._INITIALIZED_PATHS.clear()
    with kb.connect_closing(board="other") as conn:
        assert kb.claim_task(conn, second, board="other") is None


def test_resource_conflict_stays_ready_and_does_not_consume_profile_capacity(
    kanban_home, all_assignees_spawnable
):
    holder = _create("default", "holder", ["gpu:0"])
    blocked = _create("other", "blocked", ["gpu:0"])
    independent = _create("other", "independent", ["gpu:1"])
    with kb.connect_closing(board="default") as conn:
        assert kb.claim_task(conn, holder, board="default") is not None

    with kb.connect_closing(board="other") as conn:
        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda task, workspace, board=None: 999999,
            board="other",
            max_in_progress_per_profile=1,
        )
        assert [item[0] for item in result.resource_conflicts] == [blocked]
        assert [item[0] for item in result.spawned] == [independent]
        task = kb.get_task(conn, blocked)
        assert task.status == "ready"
        assert task.consecutive_failures == 0


def test_cli_create_and_update_resource_keys_round_trip(kanban_home):
    from hermes_cli.kanban import run_slash

    created = json.loads(
        run_slash(
            "create 'leased' --assignee worker --resource GPU:0 "
            "--resource control-plane:cto --json"
        )
    )
    assert created["resource_keys"] == ["control-plane:cto", "gpu:0"]

    assert "Updated" in run_slash(
        f"resources {created['id']} browser:shared control-plane:cto"
    )
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, created["id"]).resource_keys == [
            "browser:shared",
            "control-plane:cto",
        ]


def test_kanban_create_tool_schema_and_handler_round_trip(kanban_home, monkeypatch):
    from tools import kanban_tools as kt

    prop = kt.KANBAN_CREATE_SCHEMA["parameters"]["properties"]["resource_keys"]
    assert prop["type"] == "array"
    assert prop["maxItems"] == kb.MAX_RESOURCE_KEYS

    monkeypatch.setenv("HERMES_PROFILE", "orchestrator")
    payload = json.loads(
        kt._handle_create({
            "title": "tool leased",
            "assignee": "worker",
            "resource_keys": [" GPU:0 ", "gpu:0"],
        })
    )
    assert payload["ok"] is True
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, payload["task_id"]).resource_keys == ["gpu:0"]


def test_expired_registry_ttl_cannot_override_live_authoritative_claim(kanban_home):
    holder = _create("default", "holder", ["gpu:0"])
    contender = _create("other", "contender", ["gpu:0"])
    with kb.connect_closing(board="default") as conn:
        claimed = kb.claim_task(conn, holder, board="default", ttl_seconds=3600)
        assert claimed is not None
    with kb._resource_lease_connect() as leases, kb.write_txn(leases):
        leases.execute(
            "UPDATE resource_leases SET claim_expires = ? WHERE holder_task_id = ?",
            (int(time.time()) - 1, holder),
        )

    with kb.connect_closing(board="other") as conn:
        assert kb.claim_task(conn, contender, board="other") is None
        assert kb.get_task(conn, contender).status == "ready"
    assert kb.list_resource_leases()[0].holder_task_id == holder


def test_live_pid_claim_extension_renews_resource_lease_under_same_lock(kanban_home):
    holder = _create("default", "holder", ["gpu:0"])
    contender = _create("other", "contender", ["gpu:0"])
    claimer = kb._claimer_id()
    expired = int(time.time()) - 1
    with kb.connect_closing(board="default") as conn:
        claimed = kb.claim_task(
            conn, holder, board="default", claimer=claimer, ttl_seconds=1
        )
        assert claimed is not None
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET worker_pid = ?, claim_expires = ?, "
                "last_heartbeat_at = ? WHERE id = ?",
                (os.getpid(), expired, int(time.time()), holder),
            )
        with kb._resource_lease_connect() as leases, kb.write_txn(leases):
            leases.execute(
                "UPDATE resource_leases SET claim_expires = ? "
                "WHERE holder_task_id = ?",
                (expired, holder),
            )
        assert kb.release_stale_claims(conn) == 0
        renewed = kb.get_task(conn, holder)
        assert renewed.status == "running"

    lease = kb.list_resource_leases()[0]
    assert lease.holder_task_id == holder
    assert lease.claim_expires == renewed.claim_expires
    with kb.connect_closing(board="other") as conn:
        assert kb.claim_task(conn, contender, board="other") is None


def test_surviving_worker_reclaim_deferral_renews_resource_lease(kanban_home):
    holder = _create("default", "holder", ["gpu:0"])
    expired = int(time.time()) - 1
    with kb.connect_closing(board="default") as conn:
        claimed = kb.claim_task(conn, holder, board="default")
        assert claimed is not None
        with kb._resource_lease_connect() as leases, kb.write_txn(leases):
            leases.execute(
                "UPDATE resource_leases SET claim_expires = ? "
                "WHERE holder_task_id = ?",
                (expired, holder),
            )
        kb._defer_reclaim_for_live_worker(
            conn,
            holder,
            claimed.claim_lock,
            int(time.time()),
            {
                "termination_attempted": True,
                "host_local": True,
                "terminated": False,
            },
            reason="test",
        )
        deferred = kb.get_task(conn, holder)
        assert deferred is not None
    assert kb.list_resource_leases()[0].claim_expires == deferred.claim_expires


def test_schedule_running_task_releases_resource_lease_immediately(kanban_home):
    holder = _create("default", "holder", ["browser:shared"])
    contender = _create("other", "contender", ["browser:shared"])
    with kb.connect_closing(board="default") as conn:
        assert kb.claim_task(conn, holder, board="default") is not None
        assert kb.schedule_task(conn, holder, reason="later")
    assert kb.list_resource_leases() == []
    with kb.connect_closing(board="other") as conn:
        assert kb.claim_task(conn, contender, board="other") is not None


def test_dashboard_direct_running_transition_releases_resource_lease_immediately(
    kanban_home,
):
    from plugins.kanban.dashboard.plugin_api import _set_status_direct

    holder = _create("default", "holder", ["browser:shared"])
    contender = _create("other", "contender", ["browser:shared"])
    with kb.connect_closing(board="default") as conn:
        assert kb.claim_task(conn, holder, board="default") is not None
        assert _set_status_direct(conn, holder, "ready")
    assert kb.list_resource_leases() == []
    with kb.connect_closing(board="other") as conn:
        assert kb.claim_task(conn, contender, board="other") is not None


def test_review_dry_run_reports_cross_board_resource_conflict(
    kanban_home, all_assignees_spawnable
):
    holder = _create("default", "holder", ["review-env:shared"])
    review = _create("other", "review", ["review-env:shared"])
    with kb.connect_closing(board="other") as conn, kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (review,))
    with kb.connect_closing(board="default") as conn:
        assert kb.claim_task(conn, holder, board="default") is not None
    with kb.connect_closing(board="other") as conn:
        result = kb.dispatch_once(conn, dry_run=True, board="other", max_spawn=2)
        assert result.spawned == []
        assert result.resource_conflicts == [
            (review, "review-env:shared", "default", holder)
        ]


def test_cross_board_regression_uses_only_isolated_board_databases(kanban_home):
    holder = _create("default", "synthetic holder", ["gpu:isolated"])
    with kb.connect_closing(board="default") as conn:
        assert kb.claim_task(conn, holder, board="default") is not None

    lease = kb.list_resource_leases()[0]
    isolated_root = kanban_home.resolve()
    assert kb._resource_leases_path().resolve().is_relative_to(isolated_root)
    assert Path(lease.holder_db_path).resolve().is_relative_to(isolated_root)


def test_timeout_transition_releases_resource_lease_immediately(
    kanban_home, monkeypatch
):
    with kb.connect_closing(board="default") as conn:
        holder = kb.create_task(
            conn,
            title="holder",
            assignee="worker",
            max_runtime_seconds=1,
            resource_keys=["gpu:0"],
            board="default",
        )
        assert kb.claim_task(conn, holder, board="default") is not None
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET worker_pid = ?, started_at = ? WHERE id = ?",
                (999_991, int(time.time()) - 10, holder),
            )
            conn.execute(
                "UPDATE task_runs SET started_at = ? WHERE task_id = ?",
                (int(time.time()) - 10, holder),
            )
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
        assert kb.enforce_max_runtime(conn, signal_fn=lambda _pid, _sig: None) == [holder]
    assert kb.list_resource_leases() == []


def test_stale_transition_releases_resource_lease_immediately(
    kanban_home, monkeypatch
):
    holder = _create("default", "holder", ["gpu:0"])
    old = int(time.time()) - 20_000
    with kb.connect_closing(board="default") as conn:
        assert kb.claim_task(conn, holder, board="default") is not None
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET worker_pid = ?, started_at = ?, "
                "last_heartbeat_at = ? WHERE id = ?",
                (999_992, old, old, holder),
            )
            conn.execute(
                "UPDATE task_runs SET started_at = ? WHERE task_id = ?",
                (old, holder),
            )
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
        assert kb.detect_stale_running(
            conn, stale_timeout_seconds=14_400, signal_fn=lambda _pid, _sig: None
        ) == [holder]
    assert kb.list_resource_leases() == []


def test_crash_transition_releases_resource_lease_immediately(
    kanban_home, monkeypatch
):
    holder = _create("default", "holder", ["gpu:0"])
    with kb.connect_closing(board="default") as conn:
        assert kb.claim_task(conn, holder, board="default") is not None
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET worker_pid = ?, started_at = ? WHERE id = ?",
                (999_993, int(time.time()) - 10, holder),
            )
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
        assert kb.detect_crashed_workers(conn) == [holder]
    assert kb.list_resource_leases() == []


def test_unleased_heartbeat_does_not_depend_on_resource_lock(
    kanban_home, monkeypatch
):
    task_id = _create("default", "ordinary")
    claimer = kb._claimer_id()

    @contextmanager
    def unavailable_lock(_path):
        yield False

    with kb.connect_closing(board="default") as conn:
        assert kb.claim_task(conn, task_id, claimer=claimer) is not None
        monkeypatch.setattr(kb, "_cross_process_init_lock", unavailable_lock)
        assert kb.heartbeat_claim(
            conn, task_id, claimer=claimer, ttl_seconds=3600
        )
