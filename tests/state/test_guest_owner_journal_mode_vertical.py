"""Vertical: configured WAL is the OWNER's prerogative; guests never establish it.

Contract. state.db's journal mode is established ONLY by the owning SessionDB
connection via ``apply_wal_with_fallback`` (driven by the configured
``database.journal_mode``). The helper ledgers that share state.db
(``tools.async_delegation``, ``gateway.delivery_ledger``) are GUESTS: when the
physical file starts in ``journal_mode=DELETE`` — e.g. the WAL-reset-vulnerable
protection path, ``database.journal_mode=delete``, or a fresh file before the
owner opens — a guest operation must succeed AND leave the on-disk mode DELETE.
Ownership is a privilege, not a property of every opener.

This is the mandatory vertical witness for the guest/journal-ownership
reconciliation:
    configured WAL + physical DELETE + guest async_delegation op
        + guest delivery_ledger op  ->  physical journal mode still DELETE
then, separately:
    SessionDB (owner) initialization  ->  MAY establish WAL.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from gateway import delivery_ledger as dl
from tools import async_delegation as ad


@pytest.fixture
def guest_env(tmp_path, monkeypatch):
    """Point both guest ledgers at one throwaway state.db with configured WAL."""
    # Pre-create the physical file in DELETE mode, exactly as a guest would
    # inherit it before the owner ever opens.
    path = tmp_path / "state.db"
    seed = sqlite3.connect(str(path), timeout=10, isolation_level=None)
    try:
        seed.execute("CREATE TABLE IF NOT EXISTS seed_probe (v INTEGER)")
        assert (
            str(seed.execute("PRAGMA journal_mode=DELETE").fetchone()[0]).lower()
            == "delete"
        )
    finally:
        seed.close()

    # Configure the canonical journal-mode setting to WAL (the DEFAULT and the
    # aggressive case: guests must NOT act on it; only the owner may).
    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "load_config_readonly",
        lambda *a, **k: {"database": {"journal_mode": "wal"}},
    )

    monkeypatch.setattr(ad, "_db_path", lambda: path)
    monkeypatch.setattr(dl, "_db_path", lambda: path)
    return path


def _on_disk_mode(path: Path) -> str:
    conn = sqlite3.connect(str(path), timeout=10, isolation_level=None)
    try:
        return str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    finally:
        conn.close()


def test_guests_preserve_delete_even_with_wal_configured(guest_env):
    """configured WAL + physical DELETE + guest ops -> physical remains DELETE."""
    path = guest_env
    assert _on_disk_mode(path) == "delete"

    # Guest 1: async_delegation durable dispatch.
    ad._persist_dispatch({
        "delegation_id": "deleg_vertical",
        "session_key": "test:vertical",
        "origin_ui_session_id": "",
        "parent_session_id": None,
        "dispatched_at": 1.0,
        "origin_session_id": "",
        "goal": "vertical",
        "context": "ctx",
    })
    assert ad.get_durable_delegation("deleg_vertical") is not None

    # Guest 2: delivery ledger obligation.
    dl.record_obligation(
        obligation_id="obl_vertical",
        session_key="sess",
        platform="telegram",
        chat_id="123",
        thread_id=None,
        content="vertical write",
    )
    assert any(
        r["id"] == "obl_vertical" for r in json.loads(dl.debug_rows())
    )

    # The core invariant: neither guest established WAL.
    final_mode = _on_disk_mode(path)
    assert final_mode == "delete", (
        f"guest changed configured-WAL journal mode to {final_mode!r}"
    )


def test_sessiondb_owner_may_establish_wal(tmp_path, monkeypatch):
    """OWNER positive control: SessionDB initialization establishes WAL when the
    runtime supports it and the configuration requires it (journal_mode=wal is
    the canonical default). On a WAL-reset-vulnerable SQLite build the owner
    deliberately falls back to DELETE (mirroring ``SessionDB.test_wal_mode``):
    the owner — not any guest — is the sole authority over journal mode either
    way, so the invariant asserted here is that the mode is decided by the
    owner path, NOT by the guest ledgers.
    """
    import hermes_cli.config as config_mod
    from hermes_state import SessionDB
    from hermes_state_wal import is_sqlite_wal_reset_vulnerable

    monkeypatch.setattr(
        config_mod,
        "load_config_readonly",
        lambda *a, **k: {"database": {"journal_mode": "wal"}},
    )

    path = tmp_path / "state.db"
    seed = sqlite3.connect(str(path), timeout=10, isolation_level=None)
    try:
        seed.execute("CREATE TABLE IF NOT EXISTS seed_probe (v INTEGER)")
        assert (
            str(seed.execute("PRAGMA journal_mode=DELETE").fetchone()[0]).lower()
            == "delete"
        )
    finally:
        seed.close()

    db = SessionDB(db_path=path)
    try:
        mode = str(db._conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    finally:
        db.close()

    if is_sqlite_wal_reset_vulnerable():
        # Owner's safety fallback: configured WAL is refused on purpose
        # (DELETE stays). WAL simply cannot be enabled on this runtime; the
        # contract being proven is OWNERSHIP, not that DELETE is wrong.
        assert mode == "delete", (
            f"vulnerable runtime: owner must fall back to DELETE, got {mode!r}"
        )
    else:
        assert mode == "wal", f"owner failed to establish WAL, mode={mode!r}"
        # WAL is a property of the file, so it must survive the owner close.
        assert _on_disk_mode(path) == "wal"
