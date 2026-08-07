"""Behavioral coverage for #68545's centralized journal-mode setting."""

from __future__ import annotations

import sqlite3

import pytest
import yaml


def _write_config(monkeypatch: pytest.MonkeyPatch, tmp_path, config: object) -> None:
    home = tmp_path / "hermes-home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )


def _configure_mode(monkeypatch: pytest.MonkeyPatch, tmp_path, mode: object) -> None:
    _write_config(monkeypatch, tmp_path, {"database": {"journal_mode": mode}})


def _disable_vulnerable_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "hermes_state.is_sqlite_wal_reset_vulnerable",
        lambda **kwargs: False,
    )


def test_database_journal_mode_has_a_canonical_default():
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["database"]["journal_mode"] == "wal"


def test_resolve_journal_mode_uses_real_database_config(monkeypatch, tmp_path):
    from hermes_state import resolve_journal_mode

    _configure_mode(monkeypatch, tmp_path, "DELETE")
    assert resolve_journal_mode() == "delete"


def test_new_nonsecret_hermes_env_override_is_not_exposed(monkeypatch, tmp_path):
    from hermes_state import resolve_journal_mode

    _configure_mode(monkeypatch, tmp_path, "wal")
    monkeypatch.setenv("HERMES_JOURNAL_MODE", "delete")
    assert resolve_journal_mode() == "wal"


@pytest.mark.parametrize("value", ["bogus", "truncate", None, 42, {"bad": "shape"}])
def test_invalid_config_value_falls_back_to_wal(monkeypatch, tmp_path, value):
    from hermes_state import resolve_journal_mode

    _configure_mode(monkeypatch, tmp_path, value)
    assert resolve_journal_mode() == "wal"


@pytest.mark.parametrize("database", [[], "delete", 42, None])
def test_malformed_database_section_falls_back_to_wal(
    monkeypatch, tmp_path, database
):
    from hermes_state import resolve_journal_mode

    _write_config(monkeypatch, tmp_path, {"database": database})
    assert resolve_journal_mode() == "wal"


def test_apply_wal_with_fallback_honors_delete_config(monkeypatch, tmp_path):
    from hermes_state import apply_wal_with_fallback

    _configure_mode(monkeypatch, tmp_path, "delete")
    _disable_vulnerable_gate(monkeypatch)
    conn = sqlite3.connect(tmp_path / "configured.db")
    try:
        assert apply_wal_with_fallback(conn, db_label="configured.db") == "delete"
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    finally:
        conn.close()


def test_apply_wal_with_fallback_defaults_to_wal(monkeypatch, tmp_path):
    from hermes_state import apply_wal_with_fallback

    _configure_mode(monkeypatch, tmp_path, "wal")
    _disable_vulnerable_gate(monkeypatch)
    conn = sqlite3.connect(tmp_path / "default.db")
    try:
        assert apply_wal_with_fallback(conn, db_label="default.db") == "wal"
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        conn.close()


def test_configured_delete_validates_vulnerable_sqlite_result(monkeypatch, tmp_path):
    """The safety gate must not report DELETE when SQLite returns MEMORY."""
    from hermes_state import apply_wal_with_fallback

    _configure_mode(monkeypatch, tmp_path, "delete")
    monkeypatch.setattr(
        "hermes_state.is_sqlite_wal_reset_vulnerable",
        lambda **kwargs: True,
    )
    conn = sqlite3.connect(":memory:")
    try:
        with pytest.raises(sqlite3.OperationalError, match="configured.*delete"):
            apply_wal_with_fallback(conn, db_label="memory-configured.db")
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "memory"
    finally:
        conn.close()


def test_configured_delete_never_live_downgrades_existing_wal(monkeypatch, tmp_path):
    from hermes_state import apply_wal_with_fallback

    _configure_mode(monkeypatch, tmp_path, "delete")
    db_path = tmp_path / "existing-wal.db"
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() == "wal"
        monkeypatch.setattr(
            "hermes_state.is_sqlite_wal_reset_vulnerable",
            lambda **kwargs: True,
        )
        assert apply_wal_with_fallback(conn, db_label="existing-wal.db") == "wal"
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        conn.close()


def test_real_db_openers_honor_configured_delete(monkeypatch, tmp_path):
    """All helper-routed file-backed openers must behaviorally use DELETE."""
    _configure_mode(monkeypatch, tmp_path, "delete")
    _disable_vulnerable_gate(monkeypatch)

    from agent import verification_evidence
    from cron import executions
    from gateway import delivery_ledger
    from gateway.platforms.api_server import ResponseStore
    from hermes_cli import kanban_db, projects_db
    from hermes_state import SessionDB
    from plugins.memory.holographic.store import MemoryStore
    from plugins.platforms.discord.recovery import DiscordRecoveryStore
    from tools import async_delegation

    observed: dict[str, str] = {}

    for name, connect in (
        ("async_delegation", async_delegation._connect),
        ("delivery_ledger", delivery_ledger._connect),
        ("verification_evidence", verification_evidence._connect),
    ):
        conn = connect()
        try:
            observed[name] = conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
        finally:
            conn.close()

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    executions.EXECUTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    cron_conn = executions._connect()
    try:
        executions._initialize_schema(cron_conn)
        observed["cron_executions"] = cron_conn.execute(
            "PRAGMA journal_mode"
        ).fetchone()[0].lower()
    finally:
        cron_conn.close()

    discord = DiscordRecoveryStore(hermes_home=tmp_path)
    observed["discord_recovery"] = discord.call(
        lambda conn: conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
    )

    session_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        observed["session_db"] = session_db._conn.execute(
            "PRAGMA journal_mode"
        ).fetchone()[0].lower()
    finally:
        session_db.close()

    kanban_conn = kanban_db.connect(db_path=tmp_path / "kanban.db")
    try:
        observed["kanban"] = kanban_conn.execute(
            "PRAGMA journal_mode"
        ).fetchone()[0].lower()
    finally:
        kanban_conn.close()

    projects_conn = projects_db.connect(db_path=tmp_path / "projects.db")
    try:
        observed["projects"] = projects_conn.execute(
            "PRAGMA journal_mode"
        ).fetchone()[0].lower()
    finally:
        projects_conn.close()

    holographic = MemoryStore(db_path=tmp_path / "memory_store.db")
    try:
        observed["holographic"] = holographic._conn.execute(
            "PRAGMA journal_mode"
        ).fetchone()[0].lower()
    finally:
        holographic.close()

    response_store = ResponseStore(db_path=str(tmp_path / "response_store.db"))
    try:
        observed["response_store"] = response_store._conn.execute(
            "PRAGMA journal_mode"
        ).fetchone()[0].lower()
    finally:
        response_store.close()

    assert observed == {
        "async_delegation": "delete",
        "delivery_ledger": "delete",
        "verification_evidence": "delete",
        "cron_executions": "delete",
        "discord_recovery": "delete",
        "session_db": "delete",
        "kanban": "delete",
        "projects": "delete",
        "holographic": "delete",
        "response_store": "delete",
    }
