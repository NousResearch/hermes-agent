"""
Tests for session_orchestration/relay.py.

Coverage
--------
1. Basic happy-path: relay acquires lock, drives via adapter, releases lock.
2. Handoff path: when detect() returns PAUSED_HANDOFF, relay calls resume()
   instead of drive().
3. LockConflictError: a second relay call with retry_on_conflict=False raises
   immediately when the lock is already held.
4. Real two-party test: relay-send and watcher-style capture issued concurrently
   from two threads BOTH go through acquire_lock; proved they cannot hold the
   lock simultaneously (no interleave window).
5. Crash-while-locked reclaim: a lock held past its TTL is reclaimed by the next
   caller's acquire_lock.
6. Multi-line prompt lands intact via the adapter's load-buffer path (not split
   into separate send-keys calls).
"""

from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, call, patch

import pytest

from session_orchestration.adapters.base import AgentAdapter
from session_orchestration.registry import SessionOrchestrationRegistry
from session_orchestration.relay import LockConflictError, SessionRelay
from session_orchestration.types import Capabilities, SessionHandle, SessionLifecycle


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "state.db"


@pytest.fixture()
def registry(db_path: Path) -> SessionOrchestrationRegistry:
    return SessionOrchestrationRegistry(db_path=db_path)


def _make_handle(session_id: str = "abc123") -> SessionHandle:
    return SessionHandle(
        session_id=session_id,
        tmux_session=f"hermes-cc-{session_id[:8]}",
        pane=f"hermes-cc-{session_id[:8]}:0.0",
        launch_ts=datetime.now(tz=timezone.utc),
    )


def _seed_registry(registry: SessionOrchestrationRegistry, task_id: str) -> None:
    """Insert a minimal RUNNING row so acquire_lock has something to lock."""
    registry.upsert(task_id, agent="claude-code", state="RUNNING")


class _FakeAdapter(AgentAdapter):
    """Stub adapter that records all calls without touching tmux."""

    def __init__(self, lifecycle: SessionLifecycle = SessionLifecycle.WAITING_USER):
        self._lifecycle = lifecycle
        self.drive_calls: List[str] = []
        self.resume_calls: List[str] = []
        self.detect_calls: int = 0

    def capabilities(self) -> Capabilities:
        return Capabilities()

    def launch(self, workdir: str, prompt: str) -> SessionHandle:
        raise NotImplementedError

    def drive(self, handle: SessionHandle, message: str) -> None:
        self.drive_calls.append(message)

    def detect(self, handle: SessionHandle) -> SessionLifecycle:
        self.detect_calls += 1
        return self._lifecycle

    def resume(self, handle: SessionHandle, prompt: str) -> None:
        self.resume_calls.append(prompt)

    def terminate(self, handle: SessionHandle) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. Basic happy-path
# ---------------------------------------------------------------------------


def test_send_message_acquires_and_releases_lock(registry, db_path):
    task_id = str(uuid.uuid4())
    _seed_registry(registry, task_id)
    adapter = _FakeAdapter(lifecycle=SessionLifecycle.WAITING_USER)
    relay = SessionRelay(registry, adapter, ttl_seconds=30.0)
    handle = _make_handle()

    relay.send_message(task_id, handle, "hello world")

    assert adapter.drive_calls == ["hello world"]
    # Lock must be released after the call.
    row = registry.get(task_id)
    assert row["lock_holder"] is None
    assert row["lock_ts"] is None


# ---------------------------------------------------------------------------
# 2. Handoff path — resume() called instead of drive()
# ---------------------------------------------------------------------------


def test_send_message_handoff_calls_resume_not_drive(registry):
    task_id = str(uuid.uuid4())
    _seed_registry(registry, task_id)
    adapter = _FakeAdapter(lifecycle=SessionLifecycle.PAUSED_HANDOFF)
    relay = SessionRelay(registry, adapter, ttl_seconds=30.0)
    handle = _make_handle()

    relay.send_message(task_id, handle, "resume prompt")

    assert adapter.drive_calls == []
    assert adapter.resume_calls == ["resume prompt"]
    # Lock still released after resume.
    row = registry.get(task_id)
    assert row["lock_holder"] is None


# ---------------------------------------------------------------------------
# 3. LockConflictError when lock is already held (no retry)
# ---------------------------------------------------------------------------


def test_send_message_raises_lock_conflict_when_held(registry):
    task_id = str(uuid.uuid4())
    _seed_registry(registry, task_id)
    # Pre-acquire the lock as an external holder with a long TTL.
    acquired = registry.acquire_lock(task_id, "watcher:pid:999", ttl_seconds=300.0)
    assert acquired, "Pre-acquisition should succeed on a fresh row"

    adapter = _FakeAdapter()
    relay = SessionRelay(registry, adapter, ttl_seconds=30.0)
    handle = _make_handle()

    with pytest.raises(LockConflictError):
        relay.send_message(task_id, handle, "should not land")

    # The adapter must NOT have been driven.
    assert adapter.drive_calls == []


# ---------------------------------------------------------------------------
# 4. Real two-party test — concurrent relay-send and watcher-capture
#    CANNOT both hold the lock simultaneously.
# ---------------------------------------------------------------------------


def test_two_party_concurrent_cannot_interleave(db_path):
    """
    Spawn a relay-send thread and a watcher-capture thread concurrently
    against the same task_id + state.db.  Both go through acquire_lock.
    Record the (holder, acquire_time, release_time) intervals and assert
    no overlap: at most one holder can own the lock at any instant.
    """
    task_id = str(uuid.uuid4())

    # Two independent registry instances pointing at the same DB (simulates
    # separate processes — relay and watcher — sharing state.db).
    reg_relay = SessionOrchestrationRegistry(db_path=db_path)
    reg_watcher = SessionOrchestrationRegistry(db_path=db_path)

    reg_relay.upsert(task_id, agent="claude-code", state="RUNNING")

    events: List[dict] = []  # {"holder": str, "event": "acquire"|"release", "t": float}
    errors: List[str] = []
    lock = threading.Lock()

    def _relay_thread():
        holder = "relay:pid:1"
        # Retry loop: if the watcher got there first, spin until available
        deadline = time.monotonic() + 5.0
        acquired = False
        while time.monotonic() < deadline:
            ok = reg_relay.acquire_lock(task_id, holder, ttl_seconds=5.0)
            if ok:
                acquired = True
                break
            time.sleep(0.01)
        if not acquired:
            with lock:
                errors.append("relay: could not acquire lock")
            return
        with lock:
            events.append({"holder": holder, "event": "acquire", "t": time.monotonic()})
        # Simulate a slow drive: hold the lock for a brief period.
        time.sleep(0.05)
        with lock:
            events.append({"holder": holder, "event": "release", "t": time.monotonic()})
        reg_relay.release_lock(task_id, holder)

    def _watcher_thread():
        holder = "watcher:pid:2"
        deadline = time.monotonic() + 5.0
        acquired = False
        while time.monotonic() < deadline:
            ok = reg_watcher.acquire_lock(task_id, holder, ttl_seconds=5.0)
            if ok:
                acquired = True
                break
            time.sleep(0.01)
        if not acquired:
            with lock:
                errors.append("watcher: could not acquire lock")
            return
        with lock:
            events.append({"holder": holder, "event": "acquire", "t": time.monotonic()})
        # Simulate a slow capture: hold the lock for a brief period.
        time.sleep(0.05)
        with lock:
            events.append({"holder": holder, "event": "release", "t": time.monotonic()})
        reg_watcher.release_lock(task_id, holder)

    t1 = threading.Thread(target=_relay_thread)
    t2 = threading.Thread(target=_watcher_thread)

    # Start both at the same time to maximise contention.
    t1.start()
    t2.start()
    t1.join(timeout=10.0)
    t2.join(timeout=10.0)

    assert not errors, f"Lock threads failed: {errors}"
    assert len(events) == 4, f"Expected 4 events (acquire+release x2), got: {events}"

    # Sort events by timestamp to reconstruct the timeline.
    events.sort(key=lambda e: e["t"])

    # Assert no overlap: the acquire of the second holder must come AFTER
    # the release of the first holder.
    # Build intervals per holder.
    intervals: dict[str, dict] = {}
    for ev in events:
        h = ev["holder"]
        if h not in intervals:
            intervals[h] = {}
        intervals[h][ev["event"]] = ev["t"]

    holders = list(intervals.keys())
    assert len(holders) == 2, f"Expected 2 distinct holders, got {holders}"

    h1_start = intervals[holders[0]]["acquire"]
    h1_end = intervals[holders[0]]["release"]
    h2_start = intervals[holders[1]]["acquire"]
    h2_end = intervals[holders[1]]["release"]

    # Intervals [h1_start, h1_end] and [h2_start, h2_end] must NOT overlap.
    overlapping = h1_start < h2_end and h2_start < h1_end
    assert not overlapping, (
        f"Lock intervals overlapped! "
        f"{holders[0]}: [{h1_start:.4f}, {h1_end:.4f}] | "
        f"{holders[1]}: [{h2_start:.4f}, {h2_end:.4f}]"
    )


# ---------------------------------------------------------------------------
# 5. Crash-while-locked reclaim — expired TTL is reclaimed by new caller
# ---------------------------------------------------------------------------


def test_crash_while_locked_reclaim(registry):
    """
    Simulate a crash: pre-set a lock whose TTL has already expired (lock_ts
    is an epoch float in the past).  A new acquire_lock call must reclaim it
    and return True.
    """
    task_id = str(uuid.uuid4())
    _seed_registry(registry, task_id)

    # Acquire with a tiny TTL, then manually expire it by force-writing a
    # past expiry epoch directly so we don't have to sleep.
    import sqlite3

    conn = sqlite3.connect(str(registry._db_path))
    try:
        # Set lock to already-expired: expiry 60 s in the past.
        past_expiry = str(time.time() - 60.0)
        conn.execute(
            "UPDATE session_orchestration SET lock_holder = ?, lock_ts = ? "
            "WHERE task_id = ?",
            ("crashed:pid:666", past_expiry, task_id),
        )
        conn.commit()
    finally:
        conn.close()

    # Verify the lock looks held to a naive reader.
    row = registry.get(task_id)
    assert row["lock_holder"] == "crashed:pid:666"

    # Now the new caller should reclaim it.
    new_holder = "relay:pid:1001"
    reclaimed = registry.acquire_lock(task_id, new_holder, ttl_seconds=30.0)
    assert reclaimed, "Stale lock should be reclaimed by a new caller"

    row = registry.get(task_id)
    assert row["lock_holder"] == new_holder


# ---------------------------------------------------------------------------
# 6. Multi-line prompt lands intact
# ---------------------------------------------------------------------------


def test_multiline_prompt_passed_intact_to_adapter(registry):
    """
    The relay forwards the message to adapter.drive() verbatim (no splitting,
    no newline mangling).  The adapter (ClaudeCodeAdapter in production) handles
    multi-line delivery via load-buffer/paste-buffer.
    """
    task_id = str(uuid.uuid4())
    _seed_registry(registry, task_id)
    adapter = _FakeAdapter(lifecycle=SessionLifecycle.WAITING_USER)
    relay = SessionRelay(registry, adapter, ttl_seconds=30.0)
    handle = _make_handle()

    multiline = "First line.\nSecond line.\nThird line."
    relay.send_message(task_id, handle, multiline)

    assert adapter.drive_calls == [multiline], (
        "Multi-line prompt must reach adapter.drive() unchanged"
    )


# ---------------------------------------------------------------------------
# 7. Lock is released even when adapter.drive() raises
# ---------------------------------------------------------------------------


def test_lock_released_on_drive_exception(registry):
    task_id = str(uuid.uuid4())
    _seed_registry(registry, task_id)

    class _BoomAdapter(_FakeAdapter):
        def drive(self, handle, message):
            raise TimeoutError("pane never ready")

    adapter = _BoomAdapter(lifecycle=SessionLifecycle.WAITING_USER)
    relay = SessionRelay(registry, adapter, ttl_seconds=30.0)
    handle = _make_handle()

    with pytest.raises(TimeoutError):
        relay.send_message(task_id, handle, "trigger boom")

    # Lock must be released even after exception.
    row = registry.get(task_id)
    assert row["lock_holder"] is None, "Lock must be released in finally block on exception"


# ---------------------------------------------------------------------------
# pre_keys threading (answerable menu answer route)
# ---------------------------------------------------------------------------


def test_send_message_threads_pre_keys_to_drive(registry):
    """A menu answer supplies pre_keys=['Escape']; the relay must forward them
    to adapter.drive so the menu is cancelled before the paste."""
    task_id = str(uuid.uuid4())
    _seed_registry(registry, task_id)

    class _PreKeysAdapter(_FakeAdapter):
        def __init__(self) -> None:
            super().__init__(lifecycle=SessionLifecycle.WAITING_USER)
            self.pre_keys_seen = "unset"

        def drive(self, handle, message, *, pre_keys=None):
            self.pre_keys_seen = pre_keys
            self.drive_calls.append(message)

    adapter = _PreKeysAdapter()
    relay = SessionRelay(registry, adapter, ttl_seconds=30.0)
    handle = _make_handle()

    relay.send_message(
        task_id, handle, "The user chose option 2", pre_keys=["Escape"]
    )

    assert adapter.drive_calls == ["The user chose option 2"]
    assert adapter.pre_keys_seen == ["Escape"]
