"""Selected PostgreSQL must refuse the wake ownership predicate, not answer False.

``session_owned_by_profile`` is an authorization predicate (wake admission gates
on it).  A selected-PostgreSQL profile home must surface the fail-closed
refusal instead of laundering it into "not owned" via the broad except.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway.wake import session_owned_by_profile
from state_store_runtime_readiness import (
    PostgreSQLRuntimeActivationError,
    trap_state_db_opens,
)

_PG_CONFIG = (
    "state_store:\n"
    "  backend: postgresql\n"
    "  postgresql:\n"
    "    dsn_env: HERMES_STATE_STORE_TEST_DSN\n"
)


class _Homes:
    """Stand-in for ``gateway.run._multiplex_profile_homes(config)``."""

    def __init__(self, homes: dict[str, Path]) -> None:
        self._homes = homes

    def resolve(self, _config):
        return list(self._homes.items())


def _selected_pg_home(tmp_path: Path, monkeypatch, name: str = "pg-wake") -> Path:
    home = tmp_path / ".hermes" / "profiles" / name
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(_PG_CONFIG, encoding="utf-8")
    monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", "postgresql://fixture/only")
    return home


def test_selected_postgresql_profile_refuses_ownership_predicate_before_state_db(
    tmp_path, monkeypatch
):
    home = _selected_pg_home(tmp_path, monkeypatch)
    homes = _Homes({"pg-wake": home})
    monkeypatch.setattr("gateway.run._multiplex_profile_homes", homes.resolve)

    with trap_state_db_opens(home) as opens:
        with pytest.raises(PostgreSQLRuntimeActivationError) as caught:
            session_owned_by_profile(object(), "pg-wake", "raw-session-id")

    assert caught.value.report.selected_backend == "postgresql"
    assert caught.value.report.profile_home == str(home.resolve())
    assert opens == []
    assert not (home / "state.db").exists()


def test_sqlite_home_keeps_best_effort_false_on_unreadable_store(tmp_path, monkeypatch):
    """The broad except still answers False for ordinary failures (no regression)."""
    home = tmp_path / ".hermes" / "profiles" / "plain"
    home.mkdir(parents=True)
    # No config.yaml => resolve_state_store_config defaults to sqlite.
    monkeypatch.setattr(
        "gateway.run._multiplex_profile_homes",
        _Homes({"plain": home}).resolve,
    )

    assert session_owned_by_profile(object(), "plain", "raw-session-id") is False
    # A read-only SessionDB open on a non-existent db still degrades, never raises.
    assert not (home / "state.db").exists()


def test_unserved_profile_still_answers_false_without_touching_storage(tmp_path, monkeypatch):
    home = _selected_pg_home(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "gateway.run._multiplex_profile_homes",
        _Homes({"other": home}).resolve,
    )
    with trap_state_db_opens(home) as opens:
        assert session_owned_by_profile(object(), "pg-wake", "raw-session-id") is False
    assert opens == []
