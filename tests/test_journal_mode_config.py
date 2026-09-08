"""Behavioral coverage for #68545's centralized journal-mode setting."""

from __future__ import annotations

import sqlite3

import pytest

import hermes_state_wal
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
        "hermes_state_wal.is_sqlite_wal_reset_vulnerable",
        lambda **kwargs: False,
    )


@pytest.fixture(autouse=True)
def _reset_configured_delete_override_warned_paths():
    """Reset the configured-delete-override warned-paths set so the
    once-per-process-per-db_label dedup doesn't leak between tests."""
    import hermes_state

    hermes_state_wal._delete_overridden_warned_paths.clear()
    yield
    hermes_state_wal._delete_overridden_warned_paths.clear()


def test_database_journal_mode_has_a_canonical_default():
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["database"]["journal_mode"] == "wal"


def test_resolve_journal_mode_uses_real_database_config(monkeypatch, tmp_path):
    from hermes_state_wal import resolve_journal_mode

    _configure_mode(monkeypatch, tmp_path, "DELETE")
    assert resolve_journal_mode() == "delete"


def test_new_nonsecret_hermes_env_override_is_not_exposed(monkeypatch, tmp_path):
    from hermes_state_wal import resolve_journal_mode

    _configure_mode(monkeypatch, tmp_path, "wal")
    monkeypatch.setenv("HERMES_JOURNAL_MODE", "delete")
    assert resolve_journal_mode() == "wal"


@pytest.mark.parametrize("value", ["bogus", "truncate", None, 42, {"bad": "shape"}])
def test_invalid_config_value_falls_back_to_wal(monkeypatch, tmp_path, value):
    from hermes_state_wal import resolve_journal_mode

    _configure_mode(monkeypatch, tmp_path, value)
    assert resolve_journal_mode() == "wal"


@pytest.mark.parametrize("database", [[], "delete", 42, None])
def test_malformed_database_section_falls_back_to_wal(
    monkeypatch, tmp_path, database
):
    from hermes_state_wal import resolve_journal_mode

    _write_config(monkeypatch, tmp_path, {"database": database})
    assert resolve_journal_mode() == "wal"


def test_apply_wal_with_fallback_honors_delete_config(monkeypatch, tmp_path):
    from hermes_state_wal import apply_wal_with_fallback

    _configure_mode(monkeypatch, tmp_path, "delete")
    _disable_vulnerable_gate(monkeypatch)
    conn = sqlite3.connect(tmp_path / "configured.db")
    try:
        assert apply_wal_with_fallback(conn, db_label="configured.db") == "delete"
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    finally:
        conn.close()


def test_apply_wal_with_fallback_defaults_to_wal(monkeypatch, tmp_path):
    from hermes_state_wal import apply_wal_with_fallback

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
    from hermes_state_wal import apply_wal_with_fallback

    _configure_mode(monkeypatch, tmp_path, "delete")
    monkeypatch.setattr(
        "hermes_state_wal.is_sqlite_wal_reset_vulnerable",
        lambda **kwargs: True,
    )
    conn = sqlite3.connect(":memory:")
    try:
        with pytest.raises(sqlite3.OperationalError, match="configured.*delete"):
            apply_wal_with_fallback(conn, db_label="memory-configured.db")
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "memory"
    finally:
        conn.close()


def test_configured_delete_never_live_downgrades_existing_wal(monkeypatch, tmp_path, caplog):
    """Keeping WAL is correct, but the operator must be told their configured
    delete had no effect (otherwise the DB silently stays WAL and the protection
    they configured never applies)."""
    from hermes_state_wal import apply_wal_with_fallback

    _configure_mode(monkeypatch, tmp_path, "delete")
    db_path = tmp_path / "existing-wal.db"
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() == "wal"
        monkeypatch.setattr(
            "hermes_state_wal.is_sqlite_wal_reset_vulnerable",
            lambda **kwargs: True,
        )
        with caplog.at_level("ERROR", logger="hermes_state"):
            assert apply_wal_with_fallback(conn, db_label="existing-wal.db") == "wal"
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert any(
            "database.journal_mode=delete is configured" in r.message
            and "on-disk" in r.message
            for r in caplog.records
        ), "expected a warning that configured delete was overridden by on-disk WAL"
    finally:
        conn.close()


def test_configured_delete_overridden_warns_on_non_vulnerable_runtime_too(monkeypatch, tmp_path, caplog):
    """When the SQLite runtime is NOT WAL-reset-vulnerable (e.g. after a
    3.51.3+ upgrade), the on-disk WAL + configured-delete case reaches the
    read-only probe path instead of the vulnerability path. That path used to
    return WAL with no signal at all; it must emit the same override warning."""
    from hermes_state_wal import apply_wal_with_fallback

    _configure_mode(monkeypatch, tmp_path, "delete")
    _disable_vulnerable_gate(monkeypatch)
    db_path = tmp_path / "existing-wal.db"
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() == "wal"
        with caplog.at_level("ERROR", logger="hermes_state"):
            assert apply_wal_with_fallback(conn, db_label="existing-wal.db") == "wal"
            assert apply_wal_with_fallback(conn, db_label="existing-wal.db") == "wal"
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        warnings = [
            r for r in caplog.records
            if "database.journal_mode=delete is configured" in r.message
        ]
        assert len(warnings) == 1, (
            "probe path must warn when configured delete is overridden by on-disk WAL, "
            "and dedup to one per process per db_label"
        )
    finally:
        conn.close()


def test_configured_delete_overridden_warning_fires_once_per_db(monkeypatch, tmp_path, caplog):
    """The override warning is deduped per process per db_label (same discipline
    as the WAL-fallback warning), so repeated connections don't flood the log."""
    from hermes_state_wal import apply_wal_with_fallback

    _configure_mode(monkeypatch, tmp_path, "delete")
    monkeypatch.setattr(
        "hermes_state_wal.is_sqlite_wal_reset_vulnerable",
        lambda **kwargs: True,
    )
    db_path = tmp_path / "existing-wal.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL").fetchone()
        with caplog.at_level("ERROR", logger="hermes_state"):
            assert apply_wal_with_fallback(conn, db_label="once.db") == "wal"
            assert apply_wal_with_fallback(conn, db_label="once.db") == "wal"
            assert apply_wal_with_fallback(conn, db_label="once.db") == "wal"
        warnings = [
            r for r in caplog.records
            if "database.journal_mode=delete is configured" in r.message
        ]
        assert len(warnings) == 1, "override warning must fire once per process per db_label"
    finally:
        conn.close()


def test_configured_delete_with_require_wal_and_existing_wal_returns_wal(monkeypatch, tmp_path, caplog):
    """Pin the require_wal=True + configured=delete + on-disk WAL edge case: the
    existing-WAL probe branch returns "wal" unconditionally (require_wal only
    governs the WAL-refusal fallback paths), so the override warning fires and no
    WalUnsupportedError is raised."""
    from hermes_state_wal import apply_wal_with_fallback

    _configure_mode(monkeypatch, tmp_path, "delete")
    _disable_vulnerable_gate(monkeypatch)
    db_path = tmp_path / "existing-wal.db"
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() == "wal"
        with caplog.at_level("ERROR", logger="hermes_state"):
            result = apply_wal_with_fallback(
                conn, db_label="existing-wal.db", require_wal=True
            )
        assert result == "wal"
        assert any(
            "database.journal_mode=delete is configured" in r.message
            for r in caplog.records
        )
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
    from hermes_cli import kanban_db_connect as kbc
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

    kanban_conn = kbc.connect(db_path=tmp_path / "kanban.db")
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
