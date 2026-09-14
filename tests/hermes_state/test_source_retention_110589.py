"""#110589 — per-source retention overrides for the auto-prune sweep.

``sessions.source_retention_days`` (``{source: days}`` in config.yaml) ages an
automation-heavy source class (e.g. ``cron``) out faster than the global
``retention_days`` without touching human history. Overrides only tighten: the
global pass has already run, so a source cannot be kept longer than the default.
Unset/invalid maps leave the sweep byte-for-byte unchanged (#110589 back-compat).
"""

from __future__ import annotations

import time

import pytest


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_state as hs

    store = hs.SessionDB(db_path=tmp_path / "state.db")
    yield store
    store.close()


def _seed(store, sid: str, source: str, age_days: float) -> None:
    store.create_session(session_id=sid, source=source)
    store.end_session(sid, end_reason="done")
    store._conn.execute(
        "UPDATE sessions SET started_at = ?, last_activity_at = ? WHERE id = ?",
        (time.time() - age_days * 86400, time.time() - age_days * 86400, sid),
    )
    store._conn.commit()


def _write_config(home, map_yaml: str) -> None:
    (home / "config.yaml").write_text(f"sessions:\n  source_retention_days:\n{map_yaml}")


def test_cron_override_ages_cron_not_human(db, tmp_path, monkeypatch):
    _seed(db, "cron-old", "cron", 60)
    _seed(db, "cron-recent", "cron", 10)
    _seed(db, "human-old", "whatsapp", 60)

    _write_config(tmp_path, "    cron: 30\n")
    result = db.maybe_auto_prune_and_vacuum(
        retention_days=90, min_interval_hours=0, vacuum=False)

    assert db.get_session("cron-old") is None, "cron rows past the 30d override must prune"
    assert db.get_session("cron-recent") is not None
    assert db.get_session("human-old") is not None, "human history keeps the global window"
    assert result["pruned"] >= 1


def test_no_map_keeps_global_behavior(db, tmp_path, monkeypatch):
    _seed(db, "cron-old", "cron", 60)
    _seed(db, "human-old", "whatsapp", 60)

    # No config file at all.
    result = db.maybe_auto_prune_and_vacuum(
        retention_days=90, min_interval_hours=0, vacuum=False)

    assert db.get_session("cron-old") is not None
    assert db.get_session("human-old") is not None


def test_invalid_map_is_ignored(db, tmp_path, monkeypatch):
    _seed(db, "cron-old", "cron", 60)

    (tmp_path / "config.yaml").write_text(
        "sessions:\n  source_retention_days: banana\n")
    result = db.maybe_auto_prune_and_vacuum(
        retention_days=90, min_interval_hours=0, vacuum=False)

    assert db.get_session("cron-old") is not None


def test_global_pass_still_applies_to_unmapped_sources(db, tmp_path, monkeypatch):
    _seed(db, "cron-old", "cron", 200)
    _seed(db, "human-older", "whatsapp", 200)
    _seed(db, "human-recent", "whatsapp", 50)

    _write_config(tmp_path, "    cron: 30\n")
    db.maybe_auto_prune_and_vacuum(
        retention_days=90, min_interval_hours=0, vacuum=False)

    # >90d rows go via the global pass regardless of the map; <90d unmapped stay.
    assert db.get_session("cron-old") is None
    assert db.get_session("human-older") is None
    assert db.get_session("human-recent") is not None
