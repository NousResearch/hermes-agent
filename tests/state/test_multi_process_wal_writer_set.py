"""Cross-process WAL writer-set detection (#100896).

A state.db opened in WAL mode by several processes at once — a gateway
alongside a dashboard and/or webui — is the writer-set shape behind the
corruption cluster #100896 / #90837 / #100313. The in-process
duplicate-handle warning (#98573) cannot see it: it counts handles in ONE
process, and in the #100896 report it fired 7 minutes before onset while the
second writer was a different process entirely.

This module pins the open-time detection: when a ``SessionDB`` lands in WAL
mode, it asks the foreign-holder authority whether another process is holding
the database or its ``-wal`` / ``-shm`` sidecars, and — on positive evidence
only — emits a loud, once-per-process-per-path WARNING that names the set and
points at the durable containment (`database.journal_mode: delete`).

It is diagnostic containment, never a live downgrade: flipping journal_mode
under concurrent openers destroys a peer's committed-but-uncheckpointed WAL
frames, which is precisely the corruption it exists to prevent.
"""

import logging

import pytest

import hermes_state


@pytest.fixture(autouse=True)
def _clear_warned_paths():
    """Isolate the per-process dedup set between tests."""
    with hermes_state._multi_writer_warned_lock:
        hermes_state._multi_writer_warned_paths.clear()
    yield
    with hermes_state._multi_writer_warned_lock:
        hermes_state._multi_writer_warned_paths.clear()


def _warned(caplog):
    return [
        r.getMessage()
        for r in caplog.records
        if "multi-process WAL writer set" in r.getMessage()
    ]


def test_warns_on_foreign_holder(tmp_path, monkeypatch, caplog):
    """A peer holding the DB open is named, with its PID, at detection time."""
    db_path = tmp_path / "state.db"
    monkeypatch.setattr(
        hermes_state,
        "_foreign_state_db_holders",
        lambda p: [(4242, str(db_path))],
    )

    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        hermes_state._warn_if_multi_process_wal_writer_set(db_path)

    messages = _warned(caplog)
    assert messages, "a second writer went unreported"
    assert "4242" in messages[0]


def test_stays_quiet_without_foreign_holder(tmp_path, monkeypatch, caplog):
    """No peers, no warning — the scan is not allowed to alarm on absence."""
    db_path = tmp_path / "state.db"
    monkeypatch.setattr(hermes_state, "_foreign_state_db_holders", lambda p: [])

    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        hermes_state._warn_if_multi_process_wal_writer_set(db_path)

    assert _warned(caplog) == []


def test_scan_failure_does_not_warn(tmp_path, monkeypatch, caplog):
    """A pid<0 sentinel (scan failed) is not positive evidence of a peer."""
    db_path = tmp_path / "state.db"
    monkeypatch.setattr(
        hermes_state,
        "_foreign_state_db_holders",
        lambda p: [(-1, "open-file scan failed: nope")],
    )

    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        hermes_state._warn_if_multi_process_wal_writer_set(db_path)

    assert _warned(caplog) == []


def test_warns_once_per_process_per_path(tmp_path, monkeypatch, caplog):
    """The /proc scan must not ride every open; dedup bounds it to one warning."""
    db_path = tmp_path / "state.db"
    monkeypatch.setattr(
        hermes_state,
        "_foreign_state_db_holders",
        lambda p: [(1, str(db_path)), (2, str(db_path))],
    )

    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        hermes_state._warn_if_multi_process_wal_writer_set(db_path)
        caplog.clear()
        hermes_state._warn_if_multi_process_wal_writer_set(db_path)

    assert _warned(caplog) == []


@pytest.mark.requires_wal
def test_session_db_open_warns_when_joining_wal_writer_set(
    tmp_path, monkeypatch, caplog
):
    """The wiring: opening state.db in WAL with a peer present warns at open."""
    db_path = tmp_path / "state.db"
    monkeypatch.setattr(
        hermes_state,
        "_foreign_state_db_holders",
        lambda p: [(4242, str(db_path))],
    )

    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        db = hermes_state.SessionDB(db_path=db_path)
    try:
        messages = _warned(caplog)
        assert messages, (
            "SessionDB opened in WAL alongside a peer process and did not name "
            "the multi-process writer set"
        )
        assert "journal_mode: delete" in messages[0]
    finally:
        db.close()
