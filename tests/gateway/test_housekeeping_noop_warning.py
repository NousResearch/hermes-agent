"""A served profile whose ``sessions:`` config resolves to a no-op sweep must not fail silently.

The housekeeping gate returns before touching the profile's state.db when both ``auto_archive``
and ``auto_prune`` resolve false — an explicit opt-out (merged defaults carry ``auto_prune: true``,
#54189) — and no log or dashboard surface says so: a store can accumulate months of unarchived
sessions with zero signal (#132542). The no-op return now counts what a sweep would have archived
(the same ``archive_stale_sessions`` predicate) and WARNs once per process per profile; an empty
backlog is a legitimate deliberate opt-out and stays silent.

Real stores, real ``config.yaml`` files, real ``load_config`` and real ``acquire()``; only the
home is redirected at a temp dir.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest

import gateway.run as gateway_run


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """A profile home at a temp dir whose ``sessions:`` block explicitly disables both sweeps.

    Omitting the block is NOT enough for the no-op gate: merged defaults carry
    ``sessions.auto_prune: true`` (#54189), so only an explicit opt-out resolves to no-op."""
    home = tmp_path / "home"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(
        "model:\n"
        "  provider: nous\n"
        "sessions:\n"
        "  auto_archive: false\n"
        "  auto_prune: false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    # The hermetic conftest pins ``hermes_state.DEFAULT_DB_PATH`` at one sandbox store whenever
    # hermes_state is already imported, and that pin WINS over ``get_hermes_home()`` inside
    # ``_default_db_path()`` — restore the import-time sentinel so an argless ``acquire()``
    # resolves through the scope (same unpinning as the profile-scope housekeeping tests).
    import hermes_state

    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )
    resolved = Path(hermes_state._default_db_path())
    assert resolved.is_relative_to(tmp_path), (
        f"unpinned store escaped the sandbox: {resolved}"
    )
    monkeypatch.setattr(gateway_run, "_noop_housekeeping_warned_homes", set())
    return home


def _store_with_session(home: Path, *, stale_days: float = None) -> None:
    """One unarchived session; ``stale_days`` ages its row cells (no messages, so the freshest-of
    recency expression sees only the session-row cells), None leaves it fresh."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session("s1", "cli")
        if stale_days is not None:
            stale = time.time() - stale_days * 86400
            db._write_sql(
                "UPDATE sessions SET started_at = ?, last_activity_at = ?",
                (stale, stale),
            )
    finally:
        db.close()


def _noop_warnings(caplog) -> list:
    return [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "housekeeping is disabled" in r.getMessage()
    ]


def test_noop_config_with_stale_store_warns_once_per_process(hermes_home, caplog):
    _store_with_session(hermes_home, stale_days=30)

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        gateway_run._housekeeping_state_db_maintenance()
        gateway_run._housekeeping_state_db_maintenance()

    warnings = _noop_warnings(caplog)
    assert len(warnings) == 1, (
        "the disabled-sweep warning must fire exactly once per process"
    )
    message = warnings[0].getMessage()
    assert str(hermes_home) in message
    assert (
        "sessions.auto_archive=False" in message
        and "sessions.auto_prune=False" in message
    )
    assert "1 session(s)" in message


def test_noop_config_with_fresh_store_stays_silent(hermes_home, caplog):
    _store_with_session(hermes_home)  # fresh session: nothing to sweep

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        gateway_run._housekeeping_state_db_maintenance()

    assert not _noop_warnings(caplog), (
        "a deliberate opt-out with nothing to sweep is legitimate and must stay silent"
    )


def test_enabled_config_sweeps_without_noop_warning(tmp_path, monkeypatch, caplog):
    home = tmp_path / "home"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(
        "sessions:\n"
        "  auto_archive: true\n"
        "  auto_archive_days: 3\n"
        "  min_interval_hours: 0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_state

    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )
    resolved = Path(hermes_state._default_db_path())
    assert resolved.is_relative_to(tmp_path), (
        f"unpinned store escaped the sandbox: {resolved}"
    )
    monkeypatch.setattr(gateway_run, "_noop_housekeeping_warned_homes", set())
    _store_with_session(home, stale_days=30)

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        gateway_run._housekeeping_state_db_maintenance()

    assert not _noop_warnings(caplog), (
        "an enabled sweep takes the archive path, not the no-op gate"
    )
    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db")
    try:
        assert bool((db.get_session("s1") or {}).get("archived")), (
            "the stale session was not archived by the enabled sweep"
        )
    finally:
        db.close()
