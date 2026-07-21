"""#68545: configurable journal_mode (env + config.yaml) + centralized DB openers."""

from __future__ import annotations

import inspect
import os
import sqlite3
from pathlib import Path

import pytest


def test_resolve_journal_mode_defaults_to_wal(monkeypatch):
    from hermes_state import resolve_journal_mode

    monkeypatch.delenv("HERMES_JOURNAL_MODE", raising=False)
    assert resolve_journal_mode() == "wal"


def test_resolve_journal_mode_env_override(monkeypatch):
    from hermes_state import resolve_journal_mode

    monkeypatch.setenv("HERMES_JOURNAL_MODE", "delete")
    assert resolve_journal_mode() == "delete"


def test_resolve_journal_mode_env_truncase(monkeypatch):
    from hermes_state import resolve_journal_mode

    monkeypatch.setenv("HERMES_JOURNAL_MODE", "DELETE")
    assert resolve_journal_mode() == "delete"


def test_resolve_journal_mode_invalid_falls_back_to_wal(monkeypatch):
    from hermes_state import resolve_journal_mode

    monkeypatch.setenv("HERMES_JOURNAL_MODE", "bogus")
    assert resolve_journal_mode() == "wal"


def test_apply_wal_with_fallback_honors_delete_mode(monkeypatch, tmp_path):
    """When HERMES_JOURNAL_MODE=delete, apply_wal_with_fallback must NOT set WAL."""
    from hermes_state import apply_wal_with_fallback

    monkeypatch.setenv("HERMES_JOURNAL_MODE", "delete")
    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    mode = apply_wal_with_fallback(conn, db_label="test.db")
    assert mode == "delete"
    actual = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert actual.lower() == "delete"
    conn.close()


def test_apply_wal_with_fallback_defaults_to_wal(monkeypatch, tmp_path):
    """Without override, apply_wal_with_fallback still sets WAL."""
    from hermes_state import apply_wal_with_fallback

    monkeypatch.delenv("HERMES_JOURNAL_MODE", raising=False)
    # Avoid the WAL-reset vulnerability check on older SQLite versions
    # (our dev env has 3.50.4 which is flagged vulnerable, causing a
    # delete-mode fallback that makes this test environment-dependent).
    monkeypatch.setattr(
        "hermes_state.is_sqlite_wal_reset_vulnerable", lambda **kw: False
    )
    db = tmp_path / "test2.db"
    conn = sqlite3.connect(str(db))
    mode = apply_wal_with_fallback(conn, db_label="test2.db")
    assert mode == "wal"
    conn.close()


def test_direct_setters_use_apply_wal_with_fallback():
    """All 5 bypass openers must route through apply_wal_with_fallback (#68545)."""
    for fpath in [
        "tools/async_delegation.py",
        "gateway/delivery_ledger.py",
        "agent/verification_evidence.py",
        "cron/executions.py",
        "plugins/platforms/discord/recovery.py",
    ]:
        src = Path(fpath).read_text(encoding="utf-8")
        assert "apply_wal_with_fallback" in src, f"{fpath} must use apply_wal_with_fallback"
        assert 'PRAGMA journal_mode=WAL"' not in src, f"{fpath} must not set WAL directly"