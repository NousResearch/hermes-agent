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


# ── S2: watchdog revalidates immediately before /new delivery ────────────────


class _FakeAdapter:
    def __init__(self):
        self._pending_messages = {}
        self.sent = []

    async def send(self, chat_id, content, metadata=None):
        self.sent.append({"chat_id": chat_id, "content": content})


class _RacingAgent:
    """Reports stale activity on the first read, fresh on the second.

    Models an agent that makes progress between the watchdog's candidate
    scan and its delivery attempt.
    """

    def __init__(self):
        self.reads = 0

    def get_activity_summary(self):
        self.reads += 1
        age = 999 if self.reads == 1 else 1
        return build_activity_snapshot(
            last_activity_at=time.time() - age,
            last_activity_description="api call",
            last_activity_provenance=ActivityProvenance.UNKNOWN,
        )


def _runner_for_stall(adapter):
    from gateway.run import GatewayRunner

    r = GatewayRunner.__new__(GatewayRunner)
    r._running = True
    r.adapters = {"fake": adapter}
    r._profile_adapters = {}
    r._running_agents = {}
    r._running_agents_ts = {}
    r._queued_events = {}
    r._session_stall_notified = {}
    r._thread_metadata_for_source = lambda source, *a, **k: {}
    return r


def _pending_event(chat_id="chat-1"):
    source = SimpleNamespace(chat_id=chat_id, thread_id=None, platform=None)
    return SimpleNamespace(text="follow-up", source=source, timestamp=time.time())


@pytest.mark.asyncio
async def test_s2_progress_between_scan_and_send_aborts_delivery():
    adapter = _FakeAdapter()
    runner = _runner_for_stall(adapter)
    session_key = "agent:main:telegram:dm:race"
    adapter._pending_messages[session_key] = _pending_event()
    agent = _RacingAgent()
    runner._running_agents[session_key] = agent

    sent = await runner._check_session_stalls(60)
    # First read said stale; the pre-delivery re-read said fresh → abort.
    assert sent == 0
    assert adapter.sent == []
    assert agent.reads >= 2, "watchdog must re-read activity before delivery"
    # Latch re-armed: a future genuine stall must still notify.
    assert session_key not in runner._session_stall_notified


@pytest.mark.asyncio
async def test_s2_pending_drained_between_scan_and_send_aborts_delivery():
    adapter = _FakeAdapter()
    runner = _runner_for_stall(adapter)
    session_key = "agent:main:telegram:dm:drain"

    class _DrainOnReadAgent:
        def __init__(self):
            self.reads = 0

        def get_activity_summary(self):
            self.reads += 1
            if self.reads == 1:
                # Simulate the queue draining after the candidate scan but
                # before the pre-delivery revalidation.
                adapter._pending_messages.pop(session_key, None)
            return build_activity_snapshot(
                last_activity_at=time.time() - 999,
                last_activity_description="api call",
                last_activity_provenance=ActivityProvenance.UNKNOWN,
            )

    adapter._pending_messages[session_key] = _pending_event()
    runner._running_agents[session_key] = _DrainOnReadAgent()

    sent = await runner._check_session_stalls(60)
    assert sent == 0
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_s2_still_stale_after_revalidation_delivers():
    """Sanity: revalidation must not suppress GENUINE stall notices."""
    adapter = _FakeAdapter()
    runner = _runner_for_stall(adapter)
    session_key = "agent:main:telegram:dm:genuine"
    adapter._pending_messages[session_key] = _pending_event()

    class _StaleAgent:
        def get_activity_summary(self):
            return build_activity_snapshot(
                last_activity_at=time.time() - 999,
                last_activity_description="api call",
                last_activity_provenance=ActivityProvenance.UNKNOWN,
            )

    runner._running_agents[session_key] = _StaleAgent()
    sent = await runner._check_session_stalls(60)
    assert sent == 1
    assert adapter.sent and "/new" in adapter.sent[0]["content"]


# ── S4: export includes activity fields; import resets them ─────────────────


def test_s4_export_includes_activity_import_resets_it(tmp_path):
    src = SessionDB(db_path=tmp_path / "src.db")
    sid = "S4_PORTABILITY"
    src.create_session(sid, source="cli")
    src.append_message(sid, "user", "hello")
    src.touch_session_activity(
        sid,
        time.time(),
        description="working on something",
        provenance=ActivityProvenance.AGENT_COMPRESSION,
    )

    exported = src.export_session(sid)
    # Export INCLUDES the live activity fields (part of the durable row).
    assert exported["last_activity_at"] is not None
    assert exported["last_activity_description"] == "working on something"

    dst = SessionDB(db_path=tmp_path / "dst.db")
    result = dst.import_sessions([exported])
    assert sid in result.get("imported_ids", result.get("imported", [sid]))

    row = dst.get_session(sid)
    # Import RESETS activity: no resurrected "working" label on a machine
    # where no agent is running (explicit contract, #76354 S4).
    assert row.get("last_activity_at") is None
    assert not row.get("last_activity_description")
    assert not row.get("last_activity_provenance")
