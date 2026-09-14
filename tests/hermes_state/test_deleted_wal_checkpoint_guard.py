"""Regression for #105670: a halted split-brain/replaced handle must not run ANY checkpoint.

After ``DeletedWalGenerationError`` / ``StateDbReplacedError`` the handle is quarantined
(sticky flags). The close() path and the periodic _try_wal_checkpoint() must skip entirely
for quarantined handles so no stale-generation frames are checkpointed into the main DB.
Additionally, _disable_close_time_checkpoint() must run on first halt (3.12+) so SQLite's
internal last-connection checkpoint is also disabled.

Without the guards: a live split-brain writer halts correctly, but the shutdown checkpoint
converts the contained split-brain into page corruption in the main DB — exactly the #105670
incident's close-time damage.
"""

import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import hermes_state_wal
from hermes_state import (
    DeletedWalGenerationError, SessionDB, _close_time_checkpoint_configurable,
    guard_transient_wal_handle,
)


@pytest.fixture
def force_wal(monkeypatch):
    """Pin WAL so this host's vulnerable SQLite still matches production topology."""
    monkeypatch.setattr(
        hermes_state_wal,
        "is_sqlite_wal_reset_vulnerable",
        lambda version_info=None: False,
    )
    monkeypatch.setattr(hermes_state_wal, "resolve_journal_mode", lambda: "wal")


def _make_db(path: Path, session_id: str, content: str) -> SessionDB:
    db = SessionDB(db_path=path)
    db.create_session(session_id, "cli")
    db.append_message(session_id, role="user", content=content)
    return db


def _require_wal(db: SessionDB) -> Path:
    if not db._wal_active:
        db.close()
        pytest.skip("WAL not active on this filesystem")
    wal = Path(str(db.db_path) + "-wal")
    if not wal.exists():
        db.close()
        pytest.skip("WAL sidecar missing after first write")
    return wal


def _unlink_sidecars(db_path: Path) -> None:
    import os

    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(db_path) + suffix)
        if sidecar.exists():
            os.unlink(sidecar)


@pytest.mark.linux_only  # deleted-WAL write halt uses Linux unlink semantics
def test_close_after_halt_runs_no_checkpoint(tmp_path, force_wal):
    """A writer halted by DeletedWalGenerationError must not checkpoint (periodic, VACUUM or close) nor run FTS repair."""
    path = tmp_path / "state.db"
    db = _make_db(path, "s", "before")
    _require_wal(db)
    _unlink_sidecars(path)

    with pytest.raises(DeletedWalGenerationError):
        db.append_message("s", role="user", content="after-unlink")
    assert db._db_wal_generation_lost is True

    # Neither the periodic checkpoint, the stale-FTS retry, VACUUM, nor close() may touch the file now.
    with patch.object(db._conn, "execute", wraps=db._conn.execute) as mock_execute:
        db._try_wal_checkpoint()
        db._fts_stale = True
        assert db.retry_deferred_fts_recovery() is False
        with pytest.raises(DeletedWalGenerationError):
            db.vacuum()
        assert not [c for c in mock_execute.call_args_list if "vacuum" in str(c).lower()], "VACUUM ran on a quarantined handle"
        db.close()
        # No checkpoint call should have been made.
        checkpoint_calls = [
            call
            for call in mock_execute.call_args_list
            if "wal_checkpoint" in str(call).lower()
        ]
        assert not checkpoint_calls, (
            f"close() ran {len(checkpoint_calls)} checkpoint call(s) on a quarantined handle"
        )


@pytest.mark.linux_only  # deleted-WAL write halt uses Linux unlink semantics
def test_halt_disables_close_time_checkpoint(tmp_path, force_wal):
    """On 3.12+ the halt must also call _disable_close_time_checkpoint()."""
    import sqlite3

    flag = getattr(sqlite3, "SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE", None)
    if flag is None:
        pytest.skip("Python 3.12+ SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE not available")

    path = tmp_path / "state.db"
    db = _make_db(path, "s", "before")
    _require_wal(db)

    setconfig = getattr(db._conn, "setconfig", None)
    if setconfig is None:
        pytest.skip("Connection.setconfig not available")

    _unlink_sidecars(path)

    with patch.object(db._conn, "setconfig", wraps=setconfig) as mock_setconfig:
        with pytest.raises(DeletedWalGenerationError):
            db.append_message("s", role="user", content="after-unlink")

        # _disable_close_time_checkpoint() must have been called during the halt.
        disable_calls = [
            call
            for call in mock_setconfig.call_args_list
            if call[0][0] == flag and call[0][1] is True
        ]
        assert disable_calls, (
            "halt did not call setconfig(SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, True)"
        )
    db.close()


def test_writer_open_disables_close_time_checkpoint(tmp_path, force_wal):
    """On 3.12+ every writer must arm NO_CKPT_ON_CLOSE at open, not only at halt."""
    flag = getattr(sqlite3, "SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE", None)
    db = _make_db(tmp_path / "state.db", "s", "before")
    try:
        if flag is None or not hasattr(db._conn, "setconfig"):
            pytest.skip("Python 3.12+ SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE not available")
        assert db._conn.getconfig(flag) is True
    finally:
        db.close()


def test_writer_close_pins_when_foreign_holders_exist(tmp_path, force_wal, monkeypatch):
    """Python <3.12 must not sqlite3_close a writer while a sibling still holds the DB."""
    if _close_time_checkpoint_configurable():
        pytest.skip("Python 3.12+ disables close-time checkpoint; sqlite3_close is safe")
    db = _make_db(tmp_path / "state.db", "s", "held")
    monkeypatch.setattr(db, "_foreign_state_db_holders", lambda: [(4242, "gateway-writer")])
    with patch.object(SessionDB, "_close_connection_quietly") as quiet_close:
        db.close()
    quiet_close.assert_not_called()
    assert db._conn is None
    assert db._connection_pinned is True


def test_writer_close_without_holders_still_closes(tmp_path, force_wal, monkeypatch):
    """A sole writer must still close for real so tests and one-shots release the file."""
    db = _make_db(tmp_path / "state.db", "s", "solo")
    monkeypatch.setattr(db, "_foreign_state_db_holders", lambda: [])
    with patch.object(SessionDB, "_close_connection_quietly") as quiet_close:
        db.close()
    quiet_close.assert_called_once()
    assert db._connection_pinned is False


def test_guard_noop_on_none_and_read_only():
    guard_transient_wal_handle(None)
    ro = Mock(read_only=True)
    guard_transient_wal_handle(ro)
    ro._disable_close_time_checkpoint.assert_not_called()


def test_guard_disables_close_time_checkpoint_on_writable_handle():
    db = Mock(read_only=False)
    guard_transient_wal_handle(db)
    db._disable_close_time_checkpoint.assert_called_once_with()


def test_guard_swallows_checkpoint_config_failures():
    db = Mock(read_only=False)
    db._disable_close_time_checkpoint.side_effect = RuntimeError("no setconfig")
    guard_transient_wal_handle(db)
