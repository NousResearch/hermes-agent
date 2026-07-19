"""Configured state-store coverage for CLI runtime helpers."""

from __future__ import annotations

from pathlib import Path


def _write_state_config(home: Path, body: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(body, encoding="utf-8")


def test_cli_helpers_use_the_active_profile_store(tmp_path, monkeypatch):
    from hermes_cli.main import _resolve_last_session
    from hermes_cli.oneshot import _create_session_db_for_oneshot
    from hermes_state import SessionDB

    default_home = tmp_path / ".hermes"
    profile_home = default_home / "profiles" / "worker"
    _write_state_config(
        profile_home,
        "sessions:\n"
        "  state:\n"
        "    sqlite_path: worker-state.db\n",
    )
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    db = SessionDB.for_home(profile_home)
    try:
        db.create_session("worker-session", source="cli")
    finally:
        db.close()

    assert _resolve_last_session("cli") == "worker-session"
    assert not (default_home / "state.db").exists()

    oneshot_db = _create_session_db_for_oneshot()
    assert oneshot_db is not None
    try:
        assert oneshot_db.db_path == profile_home / "worker-state.db"
    finally:
        oneshot_db.close()


def test_cli_helpers_fail_closed_when_active_profile_selects_unavailable_postgres(
    tmp_path, monkeypatch
):
    from hermes_cli.main import _resolve_last_session
    from hermes_cli.oneshot import _create_session_db_for_oneshot

    profile_home = tmp_path / ".hermes" / "profiles" / "worker"
    _write_state_config(
        profile_home,
        "sessions:\n"
        "  state:\n"
        "    backend: postgres\n"
        "    postgres:\n"
        "      dsn_env: HERMES_STATE_POSTGRES_DSN\n"
        "      schema: worker_state\n",
    )
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.delenv("HERMES_STATE_POSTGRES_DSN", raising=False)

    assert _resolve_last_session("cli") is None
    assert _create_session_db_for_oneshot() is None
    assert not (profile_home / "state.db").exists()
