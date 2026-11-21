"""Physical admin objects a caller kept from a local activity refuse their use while a hosted scope is bound."""
from __future__ import annotations

import sqlite3

import pytest

import persistence_assembly as pa
from persistence import ScopePinRefused
from tests.contracts.hosted.test_hosted_activity_contract import _hosted


def test_a_kept_repair_context_refuses_when_entered_under_a_hosted_scope(tmp_path, monkeypatch):
    """Creating a lazy repair context locally does not authorize a later hosted entry."""
    import hermes_cli.sqlite_safe_read as safe_read

    scope = pa.profile_scope(home=tmp_path)
    path = tmp_path / "state.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO sessions VALUES ('local-secret')")
    conn.close()
    admin = pa.open_store_admin(scope, read_only=False)
    # Positive local control: the same API works before any hosted binding.
    with admin.exclusive_repair_guard() as guard:
        guard.wal_checkpoint_passive()
    kept = admin.exclusive_repair_guard()
    opened = []
    original = safe_read.connect_tracked

    def observe(*args, **kwargs):
        opened.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(safe_read, "connect_tracked", observe)
    with _hosted("foreign-tenant", "agent-1"):
        with pytest.raises(ScopePinRefused):
            admin.exclusive_repair_guard()
        with pytest.raises(ScopePinRefused):
            with kept as guard:
                guard.wal_checkpoint_passive()
    assert opened == []


def test_an_entered_repair_guard_refuses_its_operations_under_a_hosted_scope(tmp_path):
    """An entered repair guard is a retained service too, not blanket host authority."""
    from persistence.sqlite.admin import SQLiteStoreAdmin

    path = tmp_path / "state.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
    conn.close()
    with SQLiteStoreAdmin(path, read_only=False).exclusive_repair_guard() as guard:
        guard.wal_checkpoint_passive()
        with _hosted("foreign-tenant", "agent-1"):
            with pytest.raises(ScopePinRefused):
                guard.wal_checkpoint_passive()
