"""Negative tests for the G1 Phase 1 security boundary."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from plugins.agentops.control.api import ControlAPI, SocketSecurityError
from plugins.agentops.control.config import load_agentops_config
from plugins.agentops.control.daemon import run_daemon, start_daemon_thread
from plugins.agentops.control.events import EventSpool, EventValidationError
from plugins.agentops.control.models import AuditEvent, EventEnvelope
from plugins.agentops.control.store import StoreMigrationError, StoreRestoreError, inspect_store, open_store


def _config(write_config):
    return load_agentops_config(write_config())


def _audit(index: int) -> AuditEvent:
    return AuditEvent.create(
        actor_type="system",
        actor_id="agentopsd",
        action=f"test.audit.{index}",
        object_type="test",
        object_id=f"audit-{index}",
        timestamp="2026-08-09T12:00:00+00:00",
        metadata={"source": "hardening"},
    )


@pytest.mark.parametrize("field", ["event_id", "event_type", "producer", "target_id", "correlation_id"])
def test_event_string_fields_reject_secret_values(make_event, field):
    kwargs = {
        "schema_version": 1,
        "event_id": "evt-0001",
        "event_type": "signal.observed",
        "occurred_at": make_event().occurred_at,
        "producer": "test.collector.v1",
        "target_id": "hermes:profile:default:gateway",
        "correlation_id": "corr-0001",
        "payload": {"status": "ok"},
        "redaction_version": 1,
    }
    kwargs[field] = "sk-test-canary-secret"
    with pytest.raises(EventValidationError):
        EventEnvelope.create(**kwargs)


@pytest.mark.parametrize(
    "field",
    ["actor_type", "actor_id", "action", "object_type", "object_id", "timestamp", "before_hash", "after_hash"],
)
def test_audit_string_fields_reject_secret_values(field):
    kwargs = {
        "actor_type": "system",
        "actor_id": "agentopsd",
        "action": "event.append",
        "object_type": "event",
        "object_id": "evt-0001",
        "timestamp": "2026-08-09T12:00:00+00:00",
        "metadata": {"source": "hardening"},
        "before_hash": None,
        "after_hash": None,
    }
    kwargs[field] = "sk-test-canary-secret"
    with pytest.raises(ValueError):
        AuditEvent.create(**kwargs)


def test_invalid_utf8_spool_is_hashed_and_never_persisted_verbatim(write_config, capsys):
    config = _config(write_config)
    store = open_store(config)
    spool = EventSpool(config.spool_dir)
    secret = b"sk-test-canary-secret"
    spool.root.mkdir()
    (spool.root / "invalid.json").write_bytes(b'{"payload":"' + secret + b'"}\xff')

    result = spool.replay(store)

    assert result.quarantined == 1
    captured = b"".join(
        path.read_bytes()
        for path in [
            config.sqlite_path,
            Path(f"{config.sqlite_path}-wal"),
            Path(f"{config.sqlite_path}-shm"),
            *config.spool_dir.rglob("*"),
        ]
        if path.is_file()
    )
    assert secret not in captured
    quarantine = json.loads((spool.quarantine_dir / "invalid.json").read_text(encoding="utf-8"))
    assert quarantine["redacted"] is True
    assert quarantine["content_hash"].startswith("sha256:")
    store.close()
    daemon = start_daemon_thread(config.config_path)
    try:
        captured += json.dumps(daemon.health(), sort_keys=True).encode("utf-8")
    finally:
        daemon.stop()
    captured += (capsys.readouterr().out + capsys.readouterr().err).encode("utf-8")
    assert secret not in captured


def test_unmanaged_existing_state_dir_is_rejected(tmp_path):
    state_dir = tmp_path / "foreign-state"
    state_dir.mkdir()
    path = tmp_path / "agentops.yaml"
    path.write_text(f"storage:\n  state_dir: {state_dir}\n", encoding="utf-8")

    config = load_agentops_config(path)

    assert config.state_dir_safe is False
    assert "unmanaged_state_dir_rejected" in config.safe_start_reasons


def test_git_worktree_and_symlink_state_dirs_are_rejected(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").mkdir()
    state_dir = checkout / "agentops"
    path = tmp_path / "agentops.yaml"
    path.write_text(f"storage:\n  state_dir: {state_dir}\n", encoding="utf-8")
    assert "git_worktree_state_dir_rejected" in load_agentops_config(path).safe_start_reasons

    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    path.write_text(f"storage:\n  state_dir: {linked}\n", encoding="utf-8")
    assert "state_dir_symlink_rejected" in load_agentops_config(path).safe_start_reasons


def test_hermes_root_state_dir_is_rejected(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setattr("plugins.agentops.control.config.get_hermes_home", lambda: str(home))
    path = tmp_path / "agentops.yaml"
    path.write_text(f"storage:\n  state_dir: {home}\n", encoding="utf-8")

    config = load_agentops_config(path)

    assert config.state_dir_safe is False
    assert "hermes_root_state_dir_rejected" in config.safe_start_reasons


def test_unrelated_database_is_untouched_when_config_path_is_rejected(write_config):
    config_path = write_config()
    good = load_agentops_config(config_path)
    unrelated = good.state_dir.parent / "unrelated.db"
    with sqlite3.connect(unrelated) as connection:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("CREATE TABLE unrelated(value TEXT)")
        connection.execute("INSERT INTO unrelated VALUES ('unchanged')")
    before = hashlib.sha256(unrelated.read_bytes()).hexdigest()
    config_path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "storage:",
                f"  state_dir: {good.state_dir}",
                f"  sqlite_path: {unrelated}",
                "  spool_dir: event-spool",
                "control_plane:",
                "  socket_path: agentops.sock",
                "safety:",
                "  default_authority: observe_only",
            ]
        ),
        encoding="utf-8",
    )
    rejected = load_agentops_config(config_path)

    assert rejected.state_dir_safe is False
    assert "sqlite_outside_state_dir" in rejected.safe_start_reasons
    assert run_daemon(rejected, threading.Event()) == 1
    assert hashlib.sha256(unrelated.read_bytes()).hexdigest() == before
    with sqlite3.connect(unrelated) as connection:
        assert str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "delete"


@pytest.mark.parametrize("kind", ["wrong_schema", "future_version", "invalid_audit", "truncated"])
def test_restore_rejects_bad_candidates_before_replacing_live_store(write_config, make_event, kind):
    config = _config(write_config)
    store = open_store(config)
    store.append_event(make_event("evt-live"))
    store.backup_to()  # Create the controlled backup directory first.
    candidate = config.backup_dir / f"bad-{kind}.db"
    if kind == "wrong_schema":
        with sqlite3.connect(candidate) as connection:
            connection.execute("CREATE TABLE unrelated(value TEXT)")
    elif kind in {"future_version", "invalid_audit"}:
        baseline = store.backup_to()
        with sqlite3.connect(baseline) as source, sqlite3.connect(candidate) as destination:
            source.backup(destination)
        with sqlite3.connect(candidate) as connection:
            if kind == "future_version":
                connection.execute("UPDATE schema_migrations SET version = 99 WHERE singleton = 1")
            else:
                connection.execute("UPDATE audit_events SET entry_hash = 'tampered' WHERE sequence = 1")
    else:
        candidate.write_bytes(b"not a sqlite database")

    with pytest.raises(StoreRestoreError):
        store.restore_from(candidate)

    assert store.event_count() == 1
    assert store.verify_audit_chain() is True
    store.close()


def test_restore_reopen_failure_rolls_back_to_preserved_snapshot(write_config, make_event, monkeypatch):
    config = _config(write_config)
    store = open_store(config)
    store.append_event(make_event("evt-before"))
    backup = store.backup_to()
    store.append_event(make_event("evt-after"))
    original_open = store._open_existing_writable
    calls = 0

    def fail_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise StoreMigrationError("synthetic reopen failure")
        return original_open()

    monkeypatch.setattr(store, "_open_existing_writable", fail_once)
    with pytest.raises(StoreRestoreError):
        store.restore_from(backup)

    assert calls == 2
    assert store.event_count() == 2
    assert store.verify_audit_chain() is True
    assert list(config.backup_dir.glob("pre-restore-*.db"))
    store.close()


def test_schema_migration_runner_is_singleton_and_monotonic():
    from plugins.agentops.control.store import _run_migrations

    with sqlite3.connect(":memory:") as connection:
        _run_migrations(connection, 0)
        rows = connection.execute("SELECT singleton, version FROM schema_migrations").fetchall()
        assert rows == [(1, 1)]
        with pytest.raises(StoreMigrationError):
            _run_migrations(connection, 2)


@pytest.mark.parametrize("removed_sequence", [1, 2, 3])
def test_audit_chain_rejects_head_middle_and_tail_deletion(write_config, removed_sequence):
    store = open_store(_config(write_config))
    for index in range(1, 4):
        store.append_audit(_audit(index))
    store._connection.execute("DELETE FROM audit_events WHERE sequence = ?", (removed_sequence,))
    store._connection.commit()

    assert store.verify_audit_chain() is False
    store.close()


def test_audit_chain_rejects_metadata_head_mismatch(write_config):
    store = open_store(_config(write_config))
    store.append_audit(_audit(1))
    store._connection.execute("UPDATE metadata SET value = '99' WHERE key = 'audit_head_sequence'")
    store._connection.commit()

    assert store.verify_audit_chain() is False
    store.close()


def test_quarantine_budget_drops_untrusted_raw_input_without_retaining_it(write_config):
    config = _config(write_config)
    store = open_store(config)
    spool = EventSpool(config.spool_dir, max_bytes=128)
    secret = b"sk-test-canary-secret"
    spool.root.mkdir()
    (spool.root / "oversized-invalid.json").write_bytes(secret + b"\xff")

    result = spool.replay(store)

    assert result.dropped == 1
    assert not (spool.root / "oversized-invalid.json").exists()
    assert secret not in b"".join(path.read_bytes() for path in spool.quarantine_dir.glob("*") if path.is_file())
    store.close()


def test_orphaned_quarantine_temp_is_removed_before_replay_and_never_blocks_restart(write_config):
    config = _config(write_config)
    store = open_store(config)
    spool = EventSpool(config.spool_dir)
    secret = b"sk-test-canary-secret"
    spool._ensure_directories()
    orphan = spool.quarantine_dir / ".invalid.json.tmp"
    orphan.write_bytes(secret + b"\xff")
    source = spool.root / "invalid.json"
    source.write_bytes(secret + b"\xff")

    result = EventSpool(config.spool_dir).replay(store)

    assert result.quarantined == 1
    assert result.failed == 0
    assert not orphan.exists()
    assert not source.exists()
    assert secret not in b"".join(path.read_bytes() for path in config.spool_dir.rglob("*") if path.is_file())
    store.close()


def test_quarantine_replace_failure_is_fatal_and_does_not_leave_raw_input(write_config, monkeypatch):
    import plugins.agentops.control.events as events

    config = _config(write_config)
    spool = EventSpool(config.spool_dir)
    spool._ensure_directories()
    secret = b"sk-test-canary-secret"
    source = spool.root / "invalid.json"
    source.write_bytes(secret + b"\xff")
    real_replace = os.replace

    def fail_quarantine_replace(source_path, destination_path):
        if Path(source_path).parent == spool.quarantine_dir:
            raise OSError("synthetic replace interruption")
        return real_replace(source_path, destination_path)

    monkeypatch.setattr(events.os, "replace", fail_quarantine_replace)
    handle = start_daemon_thread(config.config_path)
    try:
        health = handle.health()
        assert health["ready"] is False
        assert "spool_quarantine_failed" in health["safe_start_reasons"]
    finally:
        handle.stop()
    assert not source.exists()
    assert not list(spool.quarantine_dir.glob(".*.tmp"))
    assert secret not in b"".join(path.read_bytes() for path in config.spool_dir.rglob("*") if path.is_file())


def test_unremovable_untrusted_spool_input_keeps_daemon_not_ready(write_config, monkeypatch):
    import plugins.agentops.control.events as events

    config = _config(write_config)
    spool = EventSpool(config.spool_dir)
    spool._ensure_directories()
    source = spool.root / "invalid.json"
    source.write_bytes(b"sk-test-canary-secret\xff")
    real_replace = os.replace

    def fail_quarantine_replace(source_path, destination_path):
        if Path(source_path).parent == spool.quarantine_dir:
            raise OSError("synthetic replace interruption")
        return real_replace(source_path, destination_path)

    monkeypatch.setattr(events.os, "replace", fail_quarantine_replace)
    monkeypatch.setattr(events.EventSpool, "_remove_untrusted_source", lambda _self, _path: False)
    handle = start_daemon_thread(config.config_path)
    try:
        health = handle.health()
        assert health["ready"] is False
        assert "spool_quarantine_failed" in health["safe_start_reasons"]
    finally:
        handle.stop()
    assert source.exists()


def _prepared_restore(write_config, make_event):
    config = _config(write_config)
    store = open_store(config)
    store.append_event(make_event("evt-before"))
    backup = store.backup_to()
    store.append_event(make_event("evt-after"))
    return config, store, backup


@pytest.mark.parametrize(
    "failure_point",
    ["copy", "replace", "fsync", "first_reopen", "rollback_reopen", "rollback_replace", "rollback_fsync"],
)
def test_restore_faults_leave_a_usable_verified_original_store(write_config, make_event, monkeypatch, failure_point):
    import plugins.agentops.control.store as store_module

    config, store, backup = _prepared_restore(write_config, make_event)
    if failure_point == "copy":
        real_copy = store_module._copy_database

        def fail_replacement_copy(source, destination):
            if Path(destination).name.startswith("replace-"):
                raise OSError("synthetic replacement copy failure")
            return real_copy(source, destination)

        monkeypatch.setattr(store_module, "_copy_database", fail_replacement_copy)
    elif failure_point == "replace":
        real_replace = store_module.os.replace

        def fail_replacement_replace(source, destination):
            if Path(source).name.startswith("replace-"):
                raise OSError("synthetic replacement rename failure")
            return real_replace(source, destination)

        monkeypatch.setattr(store_module.os, "replace", fail_replacement_replace)
    elif failure_point in {"fsync", "rollback_fsync"}:
        real_fsync = store_module._fsync_directory
        state_fsync_calls = 0

        def fail_post_replace_fsync(path):
            nonlocal state_fsync_calls
            if Path(path) == config.state_dir:
                state_fsync_calls += 1
                if failure_point == "fsync" and state_fsync_calls == 2:
                    raise OSError("synthetic post-replace fsync failure")
                if failure_point == "rollback_fsync" and state_fsync_calls == 4:
                    raise OSError("synthetic rollback fsync failure")
            return real_fsync(path)

        monkeypatch.setattr(store_module, "_fsync_directory", fail_post_replace_fsync)
        if failure_point == "rollback_fsync":
            original_open = store._open_existing_writable
            calls = 0

            def fail_first_open():
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise StoreMigrationError("synthetic first reopen failure")
                return original_open()

            monkeypatch.setattr(store, "_open_existing_writable", fail_first_open)
    else:
        original_open = store._open_existing_writable
        calls = 0

        def fail_open():
            nonlocal calls
            calls += 1
            if failure_point in {"first_reopen", "rollback_replace"} and calls == 1:
                raise StoreMigrationError("synthetic first reopen failure")
            if failure_point == "rollback_reopen" and calls <= 2:
                raise StoreMigrationError("synthetic rollback reopen failure")
            return original_open()

        monkeypatch.setattr(store, "_open_existing_writable", fail_open)
        if failure_point == "rollback_replace":
            real_replace = store_module.os.replace

            def fail_rollback_replace(source, destination):
                if Path(source).name.startswith("rollback-"):
                    raise OSError("synthetic rollback rename failure")
                return real_replace(source, destination)

            monkeypatch.setattr(store_module.os, "replace", fail_rollback_replace)

    with pytest.raises(StoreRestoreError):
        store.restore_from(backup)

    assert store.event_count() == 2
    assert store.verify_audit_chain() is True
    store.close()


def test_restore_serializes_concurrent_append_across_snapshot_and_replace(write_config, make_event, monkeypatch):
    config, store, backup = _prepared_restore(write_config, make_event)
    entered_snapshot = threading.Event()
    release_snapshot = threading.Event()
    append_complete = threading.Event()
    restore_errors: list[BaseException] = []
    append_errors: list[BaseException] = []
    original_snapshot = store._backup_to_locked

    def blocking_snapshot(destination):
        result = original_snapshot(destination)
        if Path(destination).name.startswith("pre-restore-"):
            entered_snapshot.set()
            assert release_snapshot.wait(timeout=5)
        return result

    monkeypatch.setattr(store, "_backup_to_locked", blocking_snapshot)

    def restore() -> None:
        try:
            store.restore_from(backup)
        except BaseException as exc:
            restore_errors.append(exc)

    def append() -> None:
        try:
            store.append_event(make_event("evt-concurrent"))
            append_complete.set()
        except BaseException as exc:
            append_errors.append(exc)

    restore_thread = threading.Thread(target=restore)
    restore_thread.start()
    assert entered_snapshot.wait(timeout=5)
    append_thread = threading.Thread(target=append)
    append_thread.start()
    assert append_complete.wait(timeout=0.15) is False
    release_snapshot.set()
    restore_thread.join(timeout=10)
    append_thread.join(timeout=10)

    assert restore_errors == []
    assert append_errors == []
    assert append_complete.is_set()
    assert store.event_count() == 2
    assert store.verify_audit_chain() is True
    store.close()


def test_uds_refuses_wide_state_dir_and_chmod_failure(write_config, monkeypatch):
    config = _config(write_config)
    api = ControlAPI(config.socket_path, config.state_dir, lambda: {"ready": True}, allow_stale_reclaim=True)
    os.chmod(config.state_dir, 0o755)
    with pytest.raises(SocketSecurityError):
        api.start()
    os.chmod(config.state_dir, 0o700)

    real_chmod = os.chmod

    def fail_socket_chmod(path, mode):
        if Path(path) == config.socket_path:
            raise OSError("synthetic chmod failure")
        real_chmod(path, mode)

    monkeypatch.setattr("plugins.agentops.control.api.os.chmod", fail_socket_chmod)
    with pytest.raises(OSError):
        api.start()
    assert not os.path.lexists(config.socket_path)


def test_uds_refuses_symlink_socket_occupant(write_config):
    config = _config(write_config)
    target = config.state_dir / "target"
    target.write_text("do not touch", encoding="utf-8")
    config.socket_path.symlink_to(target)
    api = ControlAPI(config.socket_path, config.state_dir, lambda: {"ready": True}, allow_stale_reclaim=True)

    with pytest.raises(SocketSecurityError):
        api.start()

    assert target.read_text(encoding="utf-8") == "do not touch"


def test_second_daemon_is_rejected_and_stale_socket_is_reclaimed_after_kill(write_config):
    config_path = write_config()
    config = load_agentops_config(config_path)
    command = (
        "import sys, threading; from pathlib import Path; "
        "from plugins.agentops.control.config import load_agentops_config; "
        "from plugins.agentops.control.daemon import run_daemon; "
        "raise SystemExit(run_daemon(load_agentops_config(Path(sys.argv[1])), threading.Event()))"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[4])}
    first = subprocess.Popen([sys.executable, "-c", command, str(config_path)], env=env)
    try:
        deadline = time.monotonic() + 5
        while not config.socket_path.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert config.socket_path.exists()
        second = subprocess.run([sys.executable, "-c", command, str(config_path)], env=env, capture_output=True, timeout=10)
        assert second.returncode == 1

        first.kill()
        assert first.wait(timeout=10) != 0
        assert config.socket_path.exists()  # The next lock holder may reclaim only this stale UDS inode.
        restarted = start_daemon_thread(config_path)
        try:
            assert restarted.health()["ready"] is True
        finally:
            restarted.stop()
    finally:
        if first.poll() is None:
            first.kill()
            first.wait(timeout=10)


def test_daemon_does_not_replace_non_socket_occupant(write_config):
    config = _config(write_config)
    config.socket_path.write_text("occupied", encoding="utf-8")

    assert run_daemon(config, threading.Event()) == 1
    assert config.socket_path.read_text(encoding="utf-8") == "occupied"


def test_process_lock_rejects_hard_linked_lockfile(write_config):
    from plugins.agentops.control.daemon import ProcessLockError, _ProcessLock

    config = _config(write_config)
    config.lock_path.write_text("lock", encoding="ascii")
    linked = config.state_dir / "lock-copy"
    os.link(config.lock_path, linked)

    with pytest.raises(ProcessLockError):
        _ProcessLock(config.lock_path).acquire()


def test_real_cli_doctor_exits_nonzero_when_degraded(tmp_path):
    workspace = Path(__file__).resolve().parents[4]
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("plugins:\n  enabled:\n    - agentops\n", encoding="utf-8")
    config_path = tmp_path / "missing-agentops.yaml"
    environment = {
        **os.environ,
        "HERMES_HOME": str(hermes_home),
        "HERMES_BUNDLED_PLUGINS": str(workspace / "plugins"),
        "PYTHONPATH": str(workspace),
    }

    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "agentops", "doctor", "--json", "--config", str(config_path)],
        cwd=workspace,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert json.loads(result.stdout)["status"] == "degraded"
