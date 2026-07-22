"""Stable machine ownership tests for distributed Kanban preparation."""

from __future__ import annotations

import os
import time
import uuid

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def machine_home(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    kb._MACHINE_ID_CACHE.clear()
    yield root
    kb._MACHINE_ID_CACHE.clear()


def test_machine_id_is_uuid_and_shared_across_profiles(machine_home, monkeypatch):
    first = kb.get_machine_id()
    assert str(uuid.UUID(first)) == first
    assert kb.machine_id_path() == machine_home / "kanban" / "machine-id"
    assert kb.machine_id_path().read_text(encoding="utf-8").strip() == first

    profile_home = machine_home / "profiles" / "ios-specialist"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    assert kb.get_machine_id() == first
    assert kb.machine_id_path() == machine_home / "kanban" / "machine-id"


def test_machine_id_file_is_created_once_across_threads(machine_home):
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        values = list(pool.map(lambda _n: kb.get_machine_id(), range(48)))

    assert len(set(values)) == 1
    assert kb.machine_id_path().read_text(encoding="utf-8").strip() == values[0]


def test_invalid_machine_id_fails_closed(machine_home):
    path = kb.machine_id_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not-a-uuid\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid Hermes Kanban machine identity"):
        kb.get_machine_id()


def test_claim_stamps_task_and_run_then_clears_active_owner(machine_home):
    kb.init_db()
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="owned", assignee="worker")
        claimed = kb.claim_task(conn, task_id, claimer="opaque-claim-token")
        assert claimed is not None
        local_machine_id = kb.get_machine_id()
        assert claimed.machine_id == local_machine_id

        run = kb.latest_run(conn, task_id)
        assert run is not None
        assert run.machine_id == local_machine_id

        assert kb.complete_task(conn, task_id, result="done") is True
        assert kb.get_task(conn, task_id).machine_id is None
        assert kb.latest_run(conn, task_id).machine_id == local_machine_id


def test_machine_column_not_claim_token_controls_pid_ownership(
    machine_home, monkeypatch,
):
    kb.init_db()
    with kb.connect() as conn:
        local_task = kb.create_task(conn, title="local", assignee="worker")
        # Deliberately misleading token: explicit task.machine_id still makes
        # this local and therefore renewable.
        kb.claim_task(conn, local_task, claimer="foreign-looking:token")
        kb._set_worker_pid(conn, local_task, 12345)
        now = int(time.time())
        conn.execute(
            "UPDATE tasks SET claim_expires = ? WHERE id = ?",
            (now + 1, local_task),
        )
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
        assert kb.renew_owned_leases(conn, now=now) == 1

        remote_task = kb.create_task(conn, title="remote", assignee="worker")
        remote_id = str(uuid.uuid4())
        # Deliberately spoof the local token prefix: explicit remote ownership
        # must win, preventing this process from probing a foreign PID.
        kb.claim_task(
            conn,
            remote_task,
            claimer=f"{kb.get_machine_id()}:spoofed",
            machine_id=remote_id,
        )
        kb._set_worker_pid(conn, remote_task, 54321)
        conn.execute(
            "UPDATE tasks SET started_at = ?, claim_expires = ? WHERE id = ?",
            (now - 300, now + 1, remote_task),
        )
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)

        assert kb.renew_owned_leases(conn, now=now) == 0
        assert kb.detect_crashed_workers(conn) == []
        assert kb.get_task(conn, remote_task).status == "running"


def test_migration_backfills_provably_local_legacy_claim(machine_home):
    path = kb.init_db()
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="legacy", assignee="worker")
        kb.claim_task(conn, task_id)
        kb._set_worker_pid(conn, task_id, os.getpid())
        legacy_lock = f"{kb._legacy_hostname()}:legacy-worker"
        conn.execute(
            "UPDATE tasks SET machine_id = NULL, claim_lock = ? WHERE id = ?",
            (legacy_lock, task_id),
        )
        conn.execute(
            "UPDATE task_runs SET machine_id = NULL, claim_lock = ? "
            "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
            (legacy_lock, task_id),
        )

    kb.init_db(path)

    with kb.connect(path) as conn:
        task = kb.get_task(conn, task_id)
        run = kb.latest_run(conn, task_id)
        assert task.machine_id == kb.get_machine_id()
        assert run is not None
        assert run.machine_id == kb.get_machine_id()


def test_dispatch_leaves_missing_capability_for_another_machine(
    machine_home, monkeypatch,
):
    kb.init_db()
    monkeypatch.setattr(kb, "local_machine_capabilities", lambda: ("linux",))
    monkeypatch.setattr(kb, "local_machine_profiles", lambda: ("default",))
    spawned: list[str] = []

    with kb.connect() as conn:
        restricted = kb.create_task(
            conn,
            title="Needs Xcode",
            assignee="default",
            required_capabilities=["macos", "xcode"],
        )
        ordinary = kb.create_task(
            conn, title="Runs here", assignee="default",
        )

        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda task, _workspace: spawned.append(task.id),
        )

        assert spawned == [ordinary]
        assert (restricted, "capabilities") in result.skipped_routing
        assert kb.get_task(conn, restricted).status == "ready"
        assert kb.task_capabilities(conn, restricted) == ("macos", "xcode")


def test_dispatch_honors_target_machine_pin(machine_home, monkeypatch):
    kb.init_db()
    monkeypatch.setattr(kb, "local_machine_capabilities", lambda: ("linux",))
    monkeypatch.setattr(kb, "local_machine_profiles", lambda: ("default",))

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Pinned elsewhere",
            assignee="default",
            target_machine=str(uuid.uuid4()),
        )
        result = kb.dispatch_once(
            conn, spawn_fn=lambda *_args, **_kwargs: pytest.fail("must not spawn"),
        )

        assert result.skipped_routing == [(task_id, "target_machine")]
        assert kb.get_task(conn, task_id).status == "ready"


def test_claim_routing_check_is_inside_claim_transaction(machine_home, monkeypatch):
    kb.init_db()
    monkeypatch.setattr(kb, "local_machine_capabilities", lambda: ("linux",))
    monkeypatch.setattr(kb, "local_machine_profiles", lambda: ("default",))

    with kb.connect() as conn:
        local_id = kb.register_local_machine(conn)
        task_id = kb.create_task(
            conn,
            title="Foreign capability",
            assignee="default",
            required_capabilities=["macos"],
        )

        assert kb.claim_task(
            conn,
            task_id,
            machine_id=local_id,
            enforce_machine_routing=True,
        ) is None
        assert kb.get_task(conn, task_id).status == "ready"
