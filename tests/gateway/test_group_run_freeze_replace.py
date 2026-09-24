"""REPLACE must not erase frozen receipts through either declared unique key."""

import sqlite3
from contextlib import closing

import pytest

from gateway.platforms import api_server_run_idempotency as storage
from gateway.platforms.api_server_run_scope import room_run_scope_key
from tests.gateway.test_group_run_freeze_store import reserve
from tests.gateway.test_group_run_scope import IDENTITY


def _install_v1_guards(conn):
    # Historical DDL, not a replacement store: exercise the real upgrade path.
    for version in (1, 2):
        for name in ("insert", "identity"):
            conn.execute(f"DROP TRIGGER IF EXISTS group_run_frozen_{name}_v{version}")
    conn.execute("""CREATE TRIGGER group_run_frozen_insert_v1
        BEFORE INSERT ON run_idempotency
        WHEN EXISTS (SELECT 1 FROM group_run_freezes WHERE scope=NEW.scope)
        BEGIN SELECT RAISE(ABORT, 'group run scope frozen'); END""")
    conn.execute("""CREATE TRIGGER group_run_frozen_identity_v1
        BEFORE UPDATE ON run_idempotency
        WHEN EXISTS (SELECT 1 FROM group_run_freezes WHERE scope IN (OLD.scope,NEW.scope))
          AND (NEW.scope IS NOT OLD.scope OR NEW.idempotency_key IS NOT OLD.idempotency_key
            OR NEW.fingerprint IS NOT OLD.fingerprint OR NEW.run_id IS NOT OLD.run_id
            OR NEW.owner_pid IS NOT OLD.owner_pid OR NEW.owner_started IS NOT OLD.owner_started)
        BEGIN SELECT RAISE(ABORT, 'group run scope frozen'); END""")
    conn.commit()


def _replace(conn, operation, scope, key, run_id):
    if operation == "insert":
        conn.execute("""INSERT OR REPLACE INTO run_idempotency
            (scope,idempotency_key,fingerprint,run_id,status_json,owner_pid,owner_started,created_at,updated_at)
            SELECT ?,?,fingerprint,?,status_json,owner_pid,owner_started,created_at,updated_at
            FROM run_idempotency WHERE run_id='attacker'""", (scope, key, run_id))
    else:
        conn.execute("""UPDATE OR REPLACE run_idempotency
            SET scope=?, idempotency_key=?, run_id=? WHERE run_id='attacker'""", (scope, key, run_id))


@pytest.mark.parametrize("upgrade", [False, True], ids=["fresh", "v1-upgrade"])
@pytest.mark.parametrize("collision", ["run-id", "primary-key", "both"])
@pytest.mark.parametrize("operation", ["insert", "update"])
def test_replace_preserves_frozen_victims_and_all_statement_rows(tmp_path, upgrade, collision, operation):
    path = tmp_path / "runs.db"
    other = {**IDENTITY, "room_id": "unfrozen"}
    with closing(storage.RunIdempotencyStore(str(path))) as store:
        reserve(store, "victim", status="running")
        reserve(store, "attacker", identity=other)
        reserve(store, "bystander", identity=other)
        before = store.freeze_room_scope(IDENTITY, "stop")
    with closing(sqlite3.connect(path)) as legacy:
        legacy.execute("PRAGMA recursive_triggers=OFF")
        assert legacy.execute("PRAGMA recursive_triggers").fetchone()[0] == 0
        if upgrade:
            _install_v1_guards(legacy)
        rows = legacy.execute("SELECT * FROM run_idempotency ORDER BY run_id").fetchall()
        with closing(storage.RunIdempotencyStore(str(path))) as store:
            scope, key, run_id = room_run_scope_key(other), "replacement", "victim"
            if collision == "primary-key":
                scope, key, run_id = room_run_scope_key(IDENTITY), "key-victim", "replacement"
            elif collision == "both":
                key = "key-bystander"
            denied = False
            try:
                _replace(legacy, operation, scope, key, run_id)
                legacy.commit()
            except sqlite3.IntegrityError as exc:
                legacy.rollback()
                assert str(exc) == "group run scope frozen"
                denied = True
            # Check evidence first: v1 silently returned an empty successful snapshot.
            assert store.room_stop_snapshot("stop") == before
            assert denied
            assert legacy.execute("SELECT * FROM run_idempotency ORDER BY run_id").fetchall() == rows
            assert reserve(store, "victim", status="running")[0] == "reused"
        with closing(storage.RunIdempotencyStore(str(path))) as restarted:
            assert restarted.room_stop_snapshot("stop") == before


@pytest.mark.parametrize("operation", ["insert", "update"])
def test_unfrozen_replacements_and_frozen_bookkeeping_pruning_remain_usable(tmp_path, operation):
    path = tmp_path / "runs.db"
    other = {**IDENTITY, "room_id": "unfrozen"}
    with closing(storage.RunIdempotencyStore(str(path))) as store, closing(sqlite3.connect(path)) as legacy:
        legacy.execute("PRAGMA recursive_triggers=OFF")
        for run_id, status in (("active", "running"), ("terminal", "completed")):
            reserve(store, run_id, status=status)
        store.freeze_room_scope(IDENTITY, "stop")
        for run_id in ("attacker", "by-key", "by-id"):
            reserve(store, run_id, identity=other)
        # Both unique constraints can replace unfrozen victims, even beside a freeze.
        _replace(legacy, operation, room_run_scope_key(other), "key-by-key", "by-id")
        legacy.commit()
        assert store.lookup(room_run_scope_key(other), "key-by-key", "fingerprint-attacker")[0] == "reused"
        assert legacy.execute("SELECT count(*) FROM run_idempotency WHERE run_id='by-key'").fetchone()[0] == 0
        with legacy:
            legacy.execute("""UPDATE OR REPLACE run_idempotency
                SET scope=scope,idempotency_key=idempotency_key,run_id=run_id,
                    status_json='{"status":"stopping"}',retention_until=9999999999,acknowledged_at=1
                WHERE run_id='active'""")
        assert store.room_stop_snapshot("stop")["runs"][0]["status"] == "stopping"
        assert store.extend_retention(room_run_scope_key(IDENTITY), "active", 9999999999)
        with legacy:
            legacy.execute("DELETE FROM run_idempotency")
        snapshot = store.room_stop_snapshot("stop")
        assert snapshot["counts"] == {"total": 2, "terminal": 1, "nonterminal": 0, "unknown": 1}
        assert snapshot["missing_runs"] == 1 and snapshot["truncated"]


def test_guard_migration_is_atomic_and_installs_new_guards_before_removing_v1(tmp_path, monkeypatch):
    path = tmp_path / "runs.db"
    with closing(storage.RunIdempotencyStore(str(path))) as store:
        reserve(store)
        store.freeze_room_scope(IDENTITY, "stop")
    with closing(sqlite3.connect(path)) as legacy:
        _install_v1_guards(legacy)
        connect = sqlite3.connect
        actions = []

        def deny_old_guard_drop(action, name, _table, _database, _trigger):
            if action in (sqlite3.SQLITE_CREATE_TRIGGER, sqlite3.SQLITE_DROP_TRIGGER):
                actions.append((action, name))
            if action == sqlite3.SQLITE_DROP_TRIGGER and name == "group_run_frozen_identity_v1":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        def failing_connect(*args, **kwargs):
            conn = connect(*args, **kwargs)
            conn.set_authorizer(deny_old_guard_drop)
            return conn

        with monkeypatch.context() as patch:
            patch.setattr(storage.sqlite3, "connect", failing_connect)
            with pytest.raises(storage.GroupStopStorageUnavailable):
                storage.RunIdempotencyStore(str(path))
        before_drop = actions[:next(i for i, (action, _) in enumerate(actions) if action == sqlite3.SQLITE_DROP_TRIGGER)]
        for name in ("insert", "identity"):
            assert (sqlite3.SQLITE_CREATE_TRIGGER, f"group_run_frozen_{name}_v2") in before_drop
        names = {row[0] for row in legacy.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
        assert {"group_run_frozen_insert_v1", "group_run_frozen_identity_v1"} <= names
        assert not {"group_run_frozen_insert_v2", "group_run_frozen_identity_v2"} & names
        with pytest.raises(sqlite3.IntegrityError, match="frozen"):
            legacy.execute("UPDATE run_idempotency SET scope='unfrozen' WHERE run_id='run-one'")
        legacy.rollback()
        with closing(storage.RunIdempotencyStore(str(path))) as retried:
            assert retried.room_stop_snapshot("stop")["counts"]["nonterminal"] == 1
        names = {row[0] for row in legacy.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
        assert {"group_run_frozen_insert_v2", "group_run_frozen_identity_v2"} <= names
        assert not {"group_run_frozen_insert_v1", "group_run_frozen_identity_v1"} & names
