"""Regressions for #76354 review S1/S2/S4 — activity write budget, watchdog
pre-delivery revalidation, and import/export activity asymmetry.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.session_activity import ActivityProvenance, build_activity_snapshot
from hermes_state import SessionDB


# ── S1: observational activity writes must not ride the 20s patience ────────


def _hold_write_lock(db_path: Path, held: threading.Event, release: threading.Event):
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.execute("BEGIN IMMEDIATE")
        held.set()
        release.wait(timeout=30)
        conn.rollback()
    finally:
        conn.close()


def test_s1_contended_activity_write_gives_up_within_short_budget(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "S1_CONTENDED"
    db.create_session(sid, source="cli")

    held = threading.Event()
    release = threading.Event()
    locker = threading.Thread(
        target=_hold_write_lock, args=(tmp_path / "state.db", held, release)
    )
    locker.start()
    try:
        assert held.wait(timeout=5)
        t0 = time.monotonic()
        with pytest.raises(sqlite3.OperationalError):
            db.touch_session_activity(sid, time.time(), description="working")
        elapsed_touch = time.monotonic() - t0
    finally:
        release.set()
        locker.join(timeout=10)

    # The observational write gave up within the short budget — far below
    # the 20s routine patience the review flagged.
    assert elapsed_touch < 3.0, f"activity touch waited {elapsed_touch:.1f}s"


def test_s1_clear_labels_noop_skips_transaction(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "S1_NOOP"
    db.create_session(sid, source="cli")
    # Fresh session: labels empty → the clear must not open a transaction.
    calls = []
    original = db._execute_write

    def _spy(fn, patience_s=None):
        calls.append(fn)
        return original(fn, patience_s=patience_s)

    monkeypatch.setattr(db, "_execute_write", _spy)
    db.clear_session_activity_labels(sid)
    assert calls == [], "no-op label clear must skip the write transaction"

    # Non-empty labels → clear runs exactly one write.
    db.touch_session_activity(sid, time.time(), description="doing work")
    calls.clear()
    db.clear_session_activity_labels(sid)
    assert len(calls) == 1
    activity = db.get_session_activity(sid)
    assert activity["last_activity_description"] == ""


def test_s1_contended_clear_gives_up_within_short_budget(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "S1_CLEAR_CONTENDED"
    db.create_session(sid, source="cli")
    db.touch_session_activity(sid, time.time(), description="busy")

    held = threading.Event()
    release = threading.Event()
    locker = threading.Thread(
        target=_hold_write_lock, args=(tmp_path / "state.db", held, release)
    )
    locker.start()
    try:
        assert held.wait(timeout=5)
        t0 = time.monotonic()
        with pytest.raises(sqlite3.OperationalError):
            db.clear_session_activity_labels(sid)
        elapsed = time.monotonic() - t0
    finally:
        release.set()
        locker.join(timeout=10)
    assert elapsed < 3.0, f"label clear waited {elapsed:.1f}s under contention"
