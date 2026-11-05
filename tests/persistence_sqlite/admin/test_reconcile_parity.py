"""Declarative schema drift on ``MaintenanceAdmin`` against the legacy module functions: each case builds one lagging
store, copies it into two profile homes, runs the legacy function on one and the admin operation on the other, and
compares the results, the errors and the files."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from persistence.sqlite._state.health import mark_storage_corrupt
from persistence import CorruptStoreError, DeclarativeDrift, PermanentStoreError, ProfileScope
from persistence.core import admin as admin_contract
from persistence.sqlite import SQLiteBackend
from persistence.sqlite.admin import SQLiteStoreAdmin
from persistence.sqlite.admin import _reconcile
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.persistence_sqlite._legacy_store import legacy_store

__all__ = ["legacy_store"]

_ADDED_TABLES = ("runtime_epoch", "session_admissions", "worker_executions", "worker_receipts")
_ADDED_COLUMNS = (("sessions", "runtime_revision"), ("sessions", "runtime_generation"),
                  ("async_delegations", "owner_execution_id"))


@pytest.fixture(autouse=True)
def _volume_headroom(monkeypatch):
    """The backup and snapshot guards reserve 2% of the volume, and the holder scan sees every process on the host."""
    from persistence.sqlite._state import repair
    monkeypatch.setattr(repair, "_REPAIR_BACKUP_FREE_FRACTION", 0.0)
    monkeypatch.setattr(repair, "_live_writer_holds_db", lambda _path: False)


@pytest.fixture
def legacy_reconcile(legacy_store, monkeypatch):
    """Main's reconcile module, its repair module given the same stand-ins."""
    import hermes_state_reconcile
    import hermes_state_repair

    monkeypatch.setattr(hermes_state_repair, "_REPAIR_BACKUP_FREE_FRACTION", 0.0)
    monkeypatch.setattr(hermes_state_repair, "_live_writer_holds_db", lambda _path: False)
    return hermes_state_reconcile


def _lagging_store(path: Path) -> Path:
    """A store built by ``SessionDB`` with four declared tables and three declared columns removed again."""
    from hermes_state import SessionDB

    with SessionDB(db_path=path) as db:
        db.create_session("kept", source="cli")
        db.append_message("kept", role="user", content="history survives")
    return _drop_added_schema(path)


def _store_lagging(path: Path) -> Path:
    """``_lagging_store`` with its rows written through the backend's store."""
    scope = ProfileScope(profile_dir=path.parent, name="default")
    backend = SQLiteBackend({scope.hermes_home_key: path})
    try:
        store = backend.open(scope)
        seed_session(store, "kept", "cli")
        seed_messages(store, "kept", {"role": "user", "content": "history survives"})
    finally:
        backend.close()
    return _drop_added_schema(path)


def _drop_added_schema(path: Path) -> Path:
    """Remove the four declared tables and three declared columns again."""
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("DROP TRIGGER IF EXISTS sessions_runtime_ledger_delete_guard")
        for table in reversed(_ADDED_TABLES):
            conn.execute(f"DROP TABLE {table}")
        conn.execute("DROP INDEX IF EXISTS idx_sessions_local_created_started")
        for table, column in _ADDED_COLUMNS:
            conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
        conn.commit()
    return path


def _current_store(path: Path) -> Path:
    from hermes_state import SessionDB

    with SessionDB(db_path=path) as db:
        db.create_session("kept", source="cli")
    return path


def _homes(tmp_path: Path, build) -> tuple[Path, Path]:
    """The same store in two profile homes: the legacy side's and the admin's."""
    seed = build(tmp_path / "seed.db")
    paths = []
    for name in ("legacy", "admin"):
        home = tmp_path / "profiles" / name
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
        with closing(sqlite3.connect(seed)) as src, closing(sqlite3.connect(home / "state.db")) as dst:
            src.backup(dst)
        paths.append(home / "state.db")
    return paths[0], paths[1]


def _schema(path: Path) -> list:
    with closing(sqlite3.connect(path)) as conn:
        return sorted(conn.execute("SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"))


def _history(path: Path) -> list:
    with closing(sqlite3.connect(path)) as conn:
        return conn.execute("SELECT s.id, m.content FROM sessions s JOIN messages m ON m.session_id = s.id").fetchall()


def _detail(text: str, legacy_path: Path, admin_path: Path) -> str:
    """The message with each side's home and profile label replaced by one placeholder."""
    for home in (legacy_path.parent, admin_path.parent):
        text = text.replace(str(home), "<home>")
        text = text.replace(f"'{home.name}'", "'<label>'").replace(f"-p {home.name} ", "-p <label> ")
    return text


# ── drift detection ────────────────────────────────────────────────────────


@pytest.mark.parametrize("build", [_lagging_store, _current_store], ids=["lagging", "current"])
def test_schema_drift_reason_matches(tmp_path, build, legacy_reconcile):
    legacy_path, admin_path = _homes(tmp_path, build)
    before = admin_path.read_bytes()

    reason = SQLiteStoreAdmin(admin_path).schema_drift_reason()

    assert reason == legacy_reconcile.schema_drift_reason(legacy_path)
    assert (reason is not None) == (build is _lagging_store)
    assert admin_path.read_bytes() == before


# ── explicit repair ────────────────────────────────────────────────────────


@pytest.mark.parametrize("backup", [True, False], ids=["backup", "no-backup"])
def test_repair_declarative_schema_reconciles_like_the_legacy_function(tmp_path, backup, legacy_reconcile):
    legacy_path, admin_path = _homes(tmp_path, _lagging_store)

    legacy = legacy_reconcile.repair_declarative_schema(legacy_path, backup=backup)
    report = SQLiteStoreAdmin(admin_path, read_only=False).repair_declarative_schema(backup=backup)

    assert {**report, "backup_path": None} == {**legacy, "backup_path": None}
    assert report["repaired"] is True and report["strategy"] == "reconcile_declarative_schema"
    assert (report["backup_path"] is not None) == (legacy["backup_path"] is not None) == backup
    if backup:
        # The backup sits at the same place relative to each store.
        backups = [Path(r["backup_path"]) for r in (report, legacy)]
        assert [b.parent.relative_to(p.parent) for b, p in zip(backups, (admin_path, legacy_path))] == [
            backups[1].parent.relative_to(legacy_path.parent)] * 2
        assert all(b.is_file() for b in backups)
    assert _schema(admin_path) == _schema(legacy_path)
    assert set(_ADDED_TABLES) <= {row[1] for row in _schema(admin_path)}
    assert _history(admin_path) == _history(legacy_path) == [("kept", "history survives")]
    assert SQLiteStoreAdmin(admin_path).schema_drift_reason() is None


def test_repair_of_a_current_store_reports_already_healthy(tmp_path, legacy_reconcile):
    legacy_path, admin_path = _homes(tmp_path, _current_store)
    before = admin_path.read_bytes()

    report = SQLiteStoreAdmin(admin_path, read_only=False).repair_declarative_schema()

    assert report == legacy_reconcile.repair_declarative_schema(legacy_path)
    assert report["strategy"] == "already_healthy"
    assert admin_path.read_bytes() == before


def test_repair_under_a_running_gateway_reports_the_ownership_error(tmp_path, legacy_reconcile):
    from gateway.runtime_ownership import ProfileOwnership

    legacy_path, admin_path = _homes(tmp_path, _lagging_store)
    before = {path: path.read_bytes() for path in (legacy_path, admin_path)}
    owner = ProfileOwnership()
    owner.reserve([legacy_path.parent, admin_path.parent])
    try:
        legacy = legacy_reconcile.repair_declarative_schema(legacy_path)
        report = SQLiteStoreAdmin(admin_path, read_only=False).repair_declarative_schema()
    finally:
        owner.close()

    assert report["repaired"] is legacy["repaired"] is False
    assert _detail(report["error"], legacy_path, admin_path) == _detail(legacy["error"], legacy_path, admin_path)
    assert {path: path.read_bytes() for path in before} == before


# ── browse reconcile ───────────────────────────────────────────────────────


def test_browse_reconcile_catches_an_unowned_store_up_once(tmp_path, legacy_reconcile):
    legacy_path, admin_path = _homes(tmp_path, _lagging_store)
    admin = SQLiteStoreAdmin(admin_path, read_only=False)

    assert admin.reconcile_drift_for_browse() is legacy_reconcile.reconcile_drift_for_browse(legacy_path) is True
    assert _schema(admin_path) == _schema(legacy_path)
    assert _history(admin_path) == _history(legacy_path)
    assert admin.schema_drift_reason() is None

    before = admin_path.read_bytes()
    assert admin.reconcile_drift_for_browse() is legacy_reconcile.reconcile_drift_for_browse(legacy_path) is False
    assert admin_path.read_bytes() == before


def test_browse_reconcile_refuses_an_owned_store_by_name(tmp_path, legacy_reconcile):
    from gateway.runtime_ownership import ProfileOwnership

    legacy_path, admin_path = _homes(tmp_path, _lagging_store)
    before = {path: path.read_bytes() for path in (legacy_path, admin_path)}
    owner = ProfileOwnership()
    owner.reserve([legacy_path.parent, admin_path.parent])
    try:
        with pytest.raises(legacy_reconcile.DeclarativeDriftError) as legacy:
            legacy_reconcile.reconcile_drift_for_browse(legacy_path)
        with pytest.raises(DeclarativeDrift) as admin:
            SQLiteStoreAdmin(admin_path, read_only=False).reconcile_drift_for_browse()
    finally:
        owner.close()

    assert admin.value.owned is legacy.value.owned is True
    assert _detail(str(admin.value), legacy_path, admin_path) == _detail(str(legacy.value), legacy_path, admin_path)
    assert "running gateway owns it" in str(admin.value)
    assert {path: path.read_bytes() for path in before} == before


def test_browse_reconcile_refuses_a_store_missing_its_history_tables(tmp_path, legacy_reconcile):
    def build(path):
        _lagging_store(path)
        with closing(sqlite3.connect(path)) as conn:
            conn.execute("PRAGMA foreign_keys=OFF")
            conn.execute("DROP TABLE messages")
            conn.commit()
        return path

    legacy_path, admin_path = _homes(tmp_path, build)
    before = admin_path.read_bytes()
    with pytest.raises(legacy_reconcile.DeclarativeDriftError) as legacy:
        legacy_reconcile.reconcile_drift_for_browse(legacy_path)
    with pytest.raises(DeclarativeDrift) as admin:
        SQLiteStoreAdmin(admin_path, read_only=False).reconcile_drift_for_browse()

    assert str(admin.value) == str(legacy.value) and "sessions recover" in str(admin.value)
    assert admin.value.owned is legacy.value.owned is False
    assert admin_path.read_bytes() == before


def test_browse_reconcile_of_a_latched_corrupt_store_raises_the_corrupt_error(tmp_path, legacy_reconcile):
    from hermes_state_health import mark_storage_corrupt as mark_legacy_corrupt

    legacy_path, admin_path = _homes(tmp_path, _lagging_store)
    mark_legacy_corrupt(legacy_path, "injected damage")
    mark_storage_corrupt(admin_path, "injected damage")
    before = admin_path.read_bytes()

    with pytest.raises(legacy_reconcile.CorruptStoreError) as legacy:
        legacy_reconcile.reconcile_drift_for_browse(legacy_path)
    with pytest.raises(CorruptStoreError) as admin:
        SQLiteStoreAdmin(admin_path, read_only=False).reconcile_drift_for_browse()

    assert str(admin.value) == str(legacy.value) and "malformed" in str(admin.value)
    assert admin_path.read_bytes() == before


def test_browse_constants_match(legacy_reconcile):
    assert _reconcile.BROWSE_RECONCILE_BUDGET_S == legacy_reconcile.BROWSE_RECONCILE_BUDGET_S
    assert _reconcile.BROWSE_RECONCILE_PURPOSE == legacy_reconcile.BROWSE_RECONCILE_PURPOSE
    # The backend runs on the admin contract's constants, the ones the gateway's boot wait reads.
    assert _reconcile.BROWSE_RECONCILE_BUDGET_S is admin_contract.BROWSE_RECONCILE_BUDGET_S
    assert _reconcile.BROWSE_RECONCILE_PURPOSE is admin_contract.BROWSE_RECONCILE_PURPOSE


# ── read-only refusals ─────────────────────────────────────────────────────


@pytest.mark.parametrize("operation", ["repair_declarative_schema", "reconcile_drift_for_browse"])
def test_read_only_handle_refuses_schema_writes(tmp_path, operation):
    _legacy_path, admin_path = _homes(tmp_path, _store_lagging)
    before = admin_path.read_bytes()
    with pytest.raises(PermanentStoreError):
        getattr(SQLiteStoreAdmin(admin_path, read_only=True), operation)()
    assert admin_path.read_bytes() == before
