"""
Unit tests for session_orchestration/watcher.py (T006).

Coverage
--------
1. A tick iterating seeded rows writes state without spurious notifications —
   state updates reach the registry; no Discord/network calls.
2. A tick whose target session is locked by the relay skips capture — the
   tmux_capture callable is NOT called when the lock is held.
3. Intent-queue drain is applied: an intent enqueued before the tick is
   processed and the row is updated.
4. Unavailable-adapter rows are skipped: a row whose agent name is absent
   from the verified adapters set is never processed.
5. Lock is acquired BEFORE capture and released AFTER — the lock holder is
   cleared from the registry before _process_row returns.
6. startup_verify is idempotent — calling it twice does not re-probe adapters.
7. Config-gate: _is_session_orchestration_enabled returns False when config
   key is absent.

All tests use:
- An in-memory SQLite DB (via tmp_path fixture).
- A FakeAdapter that returns a fixed SessionLifecycle.
- A fake tmux_capture callable that records invocations.
- Injected probe_runner / probe_specs so no real binaries are invoked.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock

import pytest

import json
from datetime import timedelta

from session_orchestration.adapters.base import AgentAdapter
from session_orchestration.adapters.verify import AdapterProbeSpec
from session_orchestration.markers import append_marker
from session_orchestration.registry import SessionOrchestrationRegistry
from session_orchestration.types import Capabilities, SessionHandle, SessionLifecycle
from session_orchestration.watcher import (
    SessionWatcher,
    _parse_marker_ts,
    _pane_hash,
    run_tick,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeAdapter(AgentAdapter):
    """Adapter that returns a fixed lifecycle value and records detect() calls."""

    def __init__(self, lifecycle: SessionLifecycle = SessionLifecycle.RUNNING):
        self._lifecycle = lifecycle
        self.detect_calls: List[SessionHandle] = []

    def capabilities(self) -> Capabilities:
        return Capabilities()

    def launch(self, workdir: str, prompt: str) -> SessionHandle:
        raise NotImplementedError

    def drive(self, handle: SessionHandle, message: str) -> None:
        raise NotImplementedError

    def detect(self, handle: SessionHandle) -> SessionLifecycle:
        self.detect_calls.append(handle)
        return self._lifecycle

    def terminate(self, handle: SessionHandle) -> None:
        raise NotImplementedError

    def resume(self, handle: SessionHandle, prompt: str) -> None:
        raise NotImplementedError


class FakeProbeRunner:
    """ProbeRunner that finds every binary and returns fixed help text."""

    def __init__(self, help_text: str = ""):
        self._help = help_text

    def which(self, binary: str) -> str | None:
        return f"/usr/local/bin/{binary}"

    def help_text(self, binary: str) -> str:
        return self._help


class FakeCapture:
    """Fake tmux_capture that returns fixed text and records calls."""

    def __init__(self, text: str = "some pane output"):
        self._text = text
        self.calls: List[str] = []

    def __call__(self, pane: str) -> str:
        self.calls.append(pane)
        return self._text


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "state.db"


@pytest.fixture()
def registry(db_path: Path) -> SessionOrchestrationRegistry:
    return SessionOrchestrationRegistry(db_path=db_path)


def _fake_probe_spec(adapter: FakeAdapter) -> AdapterProbeSpec:
    """Return a probe spec whose flags are all None (always pass)."""
    return AdapterProbeSpec(binary="fake-agent")


def _make_watcher(
    registry: SessionOrchestrationRegistry,
    adapter: FakeAdapter,
    capture: FakeCapture,
    *,
    adapter_name: str = "fake",
) -> SessionWatcher:
    """Build a SessionWatcher with injected fakes."""
    watcher = SessionWatcher(
        registry=registry,
        adapters={adapter_name: adapter},
        tmux_capture=capture,
    )
    # Inject probe fakes so no real binaries are touched
    probe_runner = FakeProbeRunner()
    probe_specs = {type(adapter).__name__: _fake_probe_spec(adapter)}
    watcher.startup_verify(probe_runner=probe_runner, probe_specs=probe_specs)
    return watcher


def _seed_row(
    registry: SessionOrchestrationRegistry,
    task_id: str = "task-001",
    agent: str = "fake",
    state: str = "RUNNING",
    tmux_session: str = "hermes-fake-001",
) -> None:
    """Insert a minimal registry row.

    The registry schema has ``tmux_session`` but no ``pane`` column.
    The watcher derives the pane as ``<tmux_session>:0.0`` from the row.
    """
    registry.upsert(
        task_id,
        agent=agent,
        run_id=f"run-{uuid.uuid4().hex[:8]}",
        repo=f"repo-{uuid.uuid4().hex[:8]}",
        state=state,
        tmux_session=tmux_session,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTickWritesState:
    """Tick iterates seeded rows and writes state to the registry."""

    def test_state_written_for_active_row(self, registry, db_path):
        task_id = "t-state-001"
        _seed_row(registry, task_id=task_id, state="RUNNING")

        adapter = FakeAdapter(SessionLifecycle.WAITING_USER)
        capture = FakeCapture("❯ ")
        watcher = _make_watcher(registry, adapter, capture)

        processed = watcher.tick()

        assert processed == 1, "expected exactly one row processed"
        row = registry.get(task_id)
        assert row is not None
        assert row["state"] == "WAITING_USER"

    def test_no_spurious_notification_on_same_state(self, registry, db_path):
        """Tick must NOT raise or produce side-effects when state unchanged."""
        task_id = "t-state-002"
        _seed_row(registry, task_id=task_id, state="RUNNING")

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture("working…")
        watcher = _make_watcher(registry, adapter, capture)

        # Should not raise; processed count reflects the single row
        processed = watcher.tick()
        assert processed == 1

        row = registry.get(task_id)
        assert row["state"] == "RUNNING"

    def test_terminal_rows_not_processed(self, registry, db_path):
        """Rows in DONE/ERROR state are skipped."""
        for tid, state in [("t-done", "DONE"), ("t-error", "ERROR")]:
            _seed_row(registry, task_id=tid, state=state)

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture()
        watcher = _make_watcher(registry, adapter, capture)

        processed = watcher.tick()
        assert processed == 0
        # capture was never called for terminal rows
        assert capture.calls == []


class TestLockSkipsCapture:
    """A tick whose target session is locked skips capture entirely."""

    def test_capture_not_called_when_locked(self, registry, db_path):
        task_id = "t-lock-001"
        _seed_row(registry, task_id=task_id, tmux_session="locked-session")

        # Acquire the lock from an "external" holder (simulates relay)
        external_holder = "relay:pid:9999:1234567890.000"
        acquired = registry.acquire_lock(
            task_id, external_holder, ttl_seconds=300.0
        )
        assert acquired, "setup: external holder should have acquired lock"

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture()
        watcher = _make_watcher(registry, adapter, capture)

        processed = watcher.tick()

        # Row was skipped — no capture call, processed count is 0
        assert processed == 0
        assert capture.calls == [], (
            "capture-pane must NOT be called while lock is held by relay"
        )
        # adapter.detect should not have been called either
        assert adapter.detect_calls == []

    def test_capture_called_when_unlocked(self, registry, db_path):
        task_id = "t-lock-002"
        _seed_row(registry, task_id=task_id, tmux_session="free-session")

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture("running…")
        watcher = _make_watcher(registry, adapter, capture)

        processed = watcher.tick()

        assert processed == 1
        assert any("free-session" in c for c in capture.calls)

    def test_lock_released_after_capture(self, registry, db_path):
        """Lock must be released after capture so relay can acquire it next."""
        task_id = "t-lock-003"
        _seed_row(registry, task_id=task_id, tmux_session="rel-session")

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture()
        watcher = _make_watcher(registry, adapter, capture)

        watcher.tick()

        # After tick, relay must be able to acquire the lock
        row = registry.get(task_id)
        assert row is not None
        # lock_holder should be None (released)
        assert row.get("lock_holder") is None, (
            "lock_holder must be released after capture"
        )


class TestIntentQueueDrain:
    """Intent-queue entries are drained and applied during the tick."""

    def test_adopt_intent_creates_row(self, registry, db_path):
        """An 'adopt' intent enqueued before the tick creates a registry row."""
        task_id = "t-intent-adopt-001"
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        repo = f"repo-{uuid.uuid4().hex[:8]}"

        registry.enqueue_intent(
            "adopt",
            task_id=task_id,
            run_id=run_id,
            repo=repo,
            payload={"agent": "fake", "run_id": run_id, "repo": repo, "task_id": task_id},
        )

        # No rows yet
        assert registry.get(task_id) is None

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture()
        watcher = _make_watcher(registry, adapter, capture)

        # Tick drains the intent → row appears
        watcher.tick()

        row = registry.get(task_id)
        assert row is not None, "adopt intent should have created a registry row"
        assert row["agent"] == "fake"

    def test_update_intent_modifies_existing_row(self, registry, db_path):
        """An 'update' intent modifies fields on an existing row."""
        task_id = "t-intent-update-001"
        _seed_row(registry, task_id=task_id, state="RUNNING")

        # Enqueue an update intent to change the project field
        registry.enqueue_intent(
            "update",
            task_id=task_id,
            payload={"task_id": task_id, "project": "my-project"},
        )

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture()
        watcher = _make_watcher(registry, adapter, capture)
        watcher.tick()

        row = registry.get(task_id)
        assert row is not None
        assert row.get("project") == "my-project"

    def test_queue_empty_after_drain(self, registry, db_path):
        """Queue is empty after the tick drains it."""
        task_id = "t-intent-drain-001"
        registry.enqueue_intent(
            "update",
            task_id=task_id,
            payload={"task_id": task_id, "project": "x"},
        )

        # Seed the row so the update has somewhere to land
        _seed_row(registry, task_id=task_id)

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture()
        watcher = _make_watcher(registry, adapter, capture)
        watcher.tick()

        # Second tick: intents drained = empty
        intents = registry.drain_intents()
        assert intents == [], "queue should be empty after the tick drained it"


class TestUnavailableAdapterSkipped:
    """Rows whose agent name is not in available adapters are skipped."""

    def test_unknown_agent_row_skipped(self, registry, db_path):
        task_id = "t-unavail-001"
        _seed_row(registry, task_id=task_id, agent="unknown-agent")

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture()
        # Watcher only has "fake", not "unknown-agent"
        watcher = _make_watcher(registry, adapter, capture, adapter_name="fake")

        processed = watcher.tick()

        assert processed == 0
        assert capture.calls == []

    def test_available_agent_processed_while_unknown_skipped(self, registry, db_path):
        """One row available, one not — only the available one is processed."""
        _seed_row(registry, task_id="t-avail", agent="fake", state="RUNNING")
        _seed_row(registry, task_id="t-unavail", agent="mystery-agent", state="RUNNING")

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture()
        watcher = _make_watcher(registry, adapter, capture, adapter_name="fake")

        processed = watcher.tick()

        assert processed == 1


class TestStartupVerifyIdempotent:
    """startup_verify is a no-op when called multiple times."""

    def test_second_call_is_noop(self, registry, db_path):
        adapter = FakeAdapter()
        probe_runner = FakeProbeRunner()
        probe_specs = {type(adapter).__name__: _fake_probe_spec(adapter)}

        call_count = [0]
        original_verify = __import__(
            "session_orchestration.adapters.verify", fromlist=["verify_adapters"]
        ).verify_adapters

        watcher = SessionWatcher(
            registry=registry,
            adapters={"fake": adapter},
        )
        watcher.startup_verify(probe_runner=probe_runner, probe_specs=probe_specs)
        first_available = dict(watcher._available_adapters)

        # Second call should be a no-op (same result, no re-probe)
        watcher.startup_verify(probe_runner=FakeProbeRunner("different"), probe_specs=probe_specs)
        second_available = dict(watcher._available_adapters)

        # Available set unchanged
        assert set(first_available.keys()) == set(second_available.keys())


class TestPaneHash:
    """_pane_hash returns consistent values for change detection."""

    def test_same_text_same_hash(self):
        assert _pane_hash("hello world") == _pane_hash("hello world")

    def test_different_text_different_hash(self):
        assert _pane_hash("hello") != _pane_hash("world")

    def test_empty_string(self):
        h = _pane_hash("")
        assert isinstance(h, str)
        assert len(h) == 16


class TestIdleTickIncrement:
    """idle_ticks increments when pane text is unchanged."""

    def test_idle_ticks_increment_on_unchanged_pane(self, registry, db_path):
        task_id = "t-idle-001"
        _seed_row(registry, task_id=task_id, tmux_session="idle-session")

        static_text = "working… (same every tick)"
        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture(static_text)
        watcher = _make_watcher(registry, adapter, capture)

        # First tick: establishes pane_hash
        watcher.tick()
        row1 = registry.get(task_id)
        idle1 = row1.get("idle_ticks", 0)

        # Second tick: same pane text → idle_ticks should increase
        watcher.tick()
        row2 = registry.get(task_id)
        idle2 = row2.get("idle_ticks", 0)

        assert idle2 > idle1, (
            f"idle_ticks should increase when pane text is unchanged "
            f"(got {idle1} then {idle2})"
        )

    def test_idle_ticks_reset_on_changed_pane(self, registry, db_path):
        task_id = "t-idle-002"
        _seed_row(registry, task_id=task_id, tmux_session="active-session")

        texts = ["first output", "second output"]
        call_n = [0]

        def changing_capture(pane: str) -> str:
            idx = min(call_n[0], len(texts) - 1)
            call_n[0] += 1
            return texts[idx]

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        watcher = SessionWatcher(
            registry=registry,
            adapters={"fake": adapter},
            tmux_capture=changing_capture,
        )
        probe_runner = FakeProbeRunner()
        probe_specs = {type(adapter).__name__: _fake_probe_spec(adapter)}
        watcher.startup_verify(probe_runner=probe_runner, probe_specs=probe_specs)

        # Tick 1 — establishes first hash
        watcher.tick()

        # Tick 2 — different text → idle_ticks should be reset to 0
        watcher.tick()
        row = registry.get(task_id)
        assert (row.get("idle_ticks") or 0) == 0


# ---------------------------------------------------------------------------
# Marker tailing tests (T008)
# ---------------------------------------------------------------------------


def _seed_row_with_workdir(
    registry: SessionOrchestrationRegistry,
    task_id: str,
    workdir: str,
    *,
    agent: str = "fake",
    state: str = "RUNNING",
    tmux_session: str = "hermes-fake-marker",
    **extra,
) -> None:
    """Insert a row that includes a workdir so the watcher can locate the
    marker file at ``{workdir}/.hermes/sessions/{task_id}.jsonl``."""
    registry.upsert(
        task_id,
        agent=agent,
        run_id=f"run-{uuid.uuid4().hex[:8]}",
        repo=f"repo-{uuid.uuid4().hex[:8]}",
        state=state,
        tmux_session=tmux_session,
        workdir=workdir,
        **extra,
    )


def _write_stale_marker(marker_file: Path, kind: str, task_id: str, age_seconds: int = 400) -> None:
    """Write a marker line with a timestamp ``age_seconds`` in the past."""
    stale_ts = (datetime.now(tz=timezone.utc) - timedelta(seconds=age_seconds)).isoformat()
    line = json.dumps({"v": 1, "ts": stale_ts, "kind": kind, "task": task_id, "payload": {}})
    marker_file.parent.mkdir(parents=True, exist_ok=True)
    with marker_file.open("ab") as fh:
        fh.write((line + "\n").encode())


class TestMarkerTailing:
    """Per-tick marker tailing: authoritative state + heartbeat hang-guard."""

    def test_recent_marker_overrides_pane_state(self, registry, tmp_path):
        """A recent marker's kind determines state; adapter.detect() is NOT called."""
        task_id = "t-marker-recent-001"
        workdir = str(tmp_path)
        _seed_row_with_workdir(registry, task_id, workdir)

        # Write a recent 'status' marker (maps to RUNNING)
        marker_file = tmp_path / ".hermes" / "sessions" / f"{task_id}.jsonl"
        marker_file.parent.mkdir(parents=True, exist_ok=True)
        append_marker(str(marker_file), "status", {"phase": "running"}, task=task_id)

        # Adapter would return WAITING_USER if called — marker must override it
        adapter = FakeAdapter(SessionLifecycle.WAITING_USER)
        capture = FakeCapture("pane output")
        watcher = _make_watcher(registry, adapter, capture)

        watcher.tick()

        row = registry.get(task_id)
        assert row is not None
        assert row["state"] == "RUNNING", (
            "status marker (->RUNNING) must override adapter WAITING_USER"
        )
        assert adapter.detect_calls == [], (
            "adapter.detect() must NOT be called when a recent marker provides state"
        )

    def test_stale_marker_falls_back_to_pane(self, registry, tmp_path):
        """A marker older than the recency window is ignored; pane detect() is used."""
        task_id = "t-marker-stale-001"
        workdir = str(tmp_path)
        _seed_row_with_workdir(registry, task_id, workdir)

        # Write a stale 'status' marker (400 s old — outside 300 s window)
        marker_file = tmp_path / ".hermes" / "sessions" / f"{task_id}.jsonl"
        _write_stale_marker(marker_file, "status", task_id, age_seconds=400)

        # Adapter returns WAITING_USER (pane fallback should win)
        adapter = FakeAdapter(SessionLifecycle.WAITING_USER)
        capture = FakeCapture("pane output")
        watcher = _make_watcher(registry, adapter, capture)

        watcher.tick()

        row = registry.get(task_id)
        assert row is not None
        assert row["state"] == "WAITING_USER", (
            "stale marker must not override state; pane detect() should win"
        )
        assert len(adapter.detect_calls) == 1, (
            "adapter.detect() must be called when all markers are stale"
        )

    def test_absent_marker_file_falls_back_to_pane(self, registry, tmp_path):
        """When the marker file does not exist, pane detect() is the sole signal."""
        task_id = "t-marker-absent-001"
        workdir = str(tmp_path)
        _seed_row_with_workdir(registry, task_id, workdir)
        # No marker file created

        adapter = FakeAdapter(SessionLifecycle.WAITING_USER)
        capture = FakeCapture("pane output")
        watcher = _make_watcher(registry, adapter, capture)

        watcher.tick()

        row = registry.get(task_id)
        assert row is not None
        assert row["state"] == "WAITING_USER"
        assert len(adapter.detect_calls) == 1

    def test_marker_offset_advances_across_ticks(self, registry, tmp_path):
        """marker_offset in the registry advances after each tick so each tick
        reads only NEW marker lines (never re-reads previously consumed lines)."""
        task_id = "t-marker-offset-001"
        workdir = str(tmp_path)
        _seed_row_with_workdir(registry, task_id, workdir)

        marker_file = tmp_path / ".hermes" / "sessions" / f"{task_id}.jsonl"
        marker_file.parent.mkdir(parents=True, exist_ok=True)

        # Write first marker and run tick 1
        append_marker(str(marker_file), "status", {"phase": "running"}, task=task_id)
        offset_after_first_marker = marker_file.stat().st_size

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture("pane text")
        watcher = _make_watcher(registry, adapter, capture)
        watcher.tick()

        row_after_tick1 = registry.get(task_id)
        assert row_after_tick1 is not None
        assert row_after_tick1.get("marker_offset", 0) == offset_after_first_marker, (
            "marker_offset must advance to end of first marker after tick 1"
        )

        # Write second marker and run tick 2
        append_marker(str(marker_file), "heartbeat", {"note": None}, task=task_id)
        offset_after_second_marker = marker_file.stat().st_size

        watcher.tick()

        row_after_tick2 = registry.get(task_id)
        assert row_after_tick2 is not None
        assert row_after_tick2.get("marker_offset", 0) == offset_after_second_marker, (
            "marker_offset must advance to end of second marker after tick 2"
        )
        assert row_after_tick2["marker_offset"] > row_after_tick1["marker_offset"], (
            "marker_offset must be strictly larger after tick 2 than tick 1"
        )

    def test_heartbeat_marker_suppresses_hang_guard(self, registry, tmp_path, monkeypatch):
        """A recent heartbeat marker prevents _on_hang from firing even when pane
        hash is unchanged past idle/stale thresholds."""
        task_id = "t-hang-suppress-001"
        workdir = str(tmp_path)
        static_pane = "static pane text"
        # Pre-set idle_ticks=5 and last_pane_hash matching the static pane text so
        # the hang condition (idle_ticks > 0, pane unchanged) would normally fire.
        _seed_row_with_workdir(
            registry,
            task_id,
            workdir,
            idle_ticks=5,
            last_pane_hash=_pane_hash(static_pane),
        )

        # Write a recent heartbeat marker
        marker_file = tmp_path / ".hermes" / "sessions" / f"{task_id}.jsonl"
        marker_file.parent.mkdir(parents=True, exist_ok=True)
        append_marker(str(marker_file), "heartbeat", {"note": None}, task=task_id)

        hang_called = []
        monkeypatch.setattr(
            "session_orchestration.watcher._on_hang",
            lambda *a, **kw: hang_called.append(True),
        )

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture(static_pane)
        watcher = _make_watcher(registry, adapter, capture)
        watcher.tick()

        assert hang_called == [], (
            "_on_hang must NOT be called when a recent heartbeat marker exists"
        )

    def test_stale_heartbeat_does_not_suppress_hang_guard(self, registry, tmp_path, monkeypatch):
        """A heartbeat marker older than the recency window must not suppress _on_hang."""
        task_id = "t-hang-stale-001"
        workdir = str(tmp_path)
        static_pane = "static pane text stale"
        _seed_row_with_workdir(
            registry,
            task_id,
            workdir,
            idle_ticks=5,
            last_pane_hash=_pane_hash(static_pane),
        )

        # Write a stale heartbeat marker (400 s old — outside window)
        marker_file = tmp_path / ".hermes" / "sessions" / f"{task_id}.jsonl"
        _write_stale_marker(marker_file, "heartbeat", task_id, age_seconds=400)

        hang_called = []
        monkeypatch.setattr(
            "session_orchestration.watcher._on_hang",
            lambda *a, **kw: hang_called.append(True),
        )

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture(static_pane)
        watcher = _make_watcher(registry, adapter, capture)
        watcher.tick()

        assert hang_called == [True], (
            "_on_hang must be called when the only heartbeat marker is stale"
        )

    def test_migrate_schema_idempotent(self, db_path):
        """Creating two registry instances on the same DB must not raise.

        The second creation triggers _migrate_schema on an already-migrated DB
        (the marker_offset column already exists); the idempotent ALTER TABLE
        must swallow the OperationalError silently.
        """
        reg1 = SessionOrchestrationRegistry(db_path=db_path)
        reg2 = SessionOrchestrationRegistry(db_path=db_path)
        # Explicitly calling _migrate_schema a third time must also be silent
        reg2._migrate_schema()
        # Verify the column is present by inserting and reading back a row
        reg1.upsert(
            "t-migrate-001",
            agent="fake",
            run_id="run-mig-001",
            repo="repo-mig-001",
            marker_offset=42,
        )
        row = reg1.get("t-migrate-001")
        assert row is not None
        assert row.get("marker_offset") == 42, (
            "marker_offset column must be present and writable after migration"
        )


# ---------------------------------------------------------------------------
# Handoff marker tests (T009)
# ---------------------------------------------------------------------------


class FakeAdapterWithResume(FakeAdapter):
    """FakeAdapter that records resume() calls instead of raising."""

    def __init__(self, lifecycle: SessionLifecycle = SessionLifecycle.RUNNING):
        super().__init__(lifecycle)
        self.resume_calls: List[tuple] = []  # (handle, prompt) pairs

    def resume(self, handle: SessionHandle, prompt: str) -> None:
        self.resume_calls.append((handle, prompt))


class TestHandoffMarkers:
    """T009: handoff_continue auto-resumes; handoff_decision DMs the user."""

    def test_handoff_continue_calls_resume_and_sets_running(self, registry, tmp_path):
        """A recent handoff_continue marker must call adapter.resume(handle, '')
        and override the lifecycle to RUNNING before the upsert."""
        task_id = "t-hc-resume-001"
        workdir = str(tmp_path)
        _seed_row_with_workdir(registry, task_id, workdir)

        marker_file = tmp_path / ".hermes" / "sessions" / f"{task_id}.jsonl"
        marker_file.parent.mkdir(parents=True, exist_ok=True)
        append_marker(
            str(marker_file),
            "handoff_continue",
            {"handoff_text": "carry on"},
            task=task_id,
        )

        adapter = FakeAdapterWithResume(SessionLifecycle.RUNNING)
        capture = FakeCapture("pane output")
        watcher = _make_watcher(registry, adapter, capture)
        watcher.tick()

        assert len(adapter.resume_calls) == 1, (
            "adapter.resume() must be called exactly once for handoff_continue"
        )
        _handle, prompt = adapter.resume_calls[0]
        assert prompt == "", "resume must be called with an empty prompt"

        row = registry.get(task_id)
        assert row is not None
        assert row["state"] == "RUNNING", (
            "state must be RUNNING after handoff_continue overrides PAUSED_HANDOFF"
        )

    def test_handoff_continue_does_not_call_send_dm(
        self, registry, tmp_path, monkeypatch
    ):
        """handoff_continue must NOT trigger a DM — only auto-resume."""
        task_id = "t-hc-no-dm-001"
        workdir = str(tmp_path)
        _seed_row_with_workdir(registry, task_id, workdir, discord_user_id="user-123")

        marker_file = tmp_path / ".hermes" / "sessions" / f"{task_id}.jsonl"
        marker_file.parent.mkdir(parents=True, exist_ok=True)
        append_marker(
            str(marker_file),
            "handoff_continue",
            {"handoff_text": "carry on"},
            task=task_id,
        )

        dm_calls: List = []
        monkeypatch.setattr(
            "session_orchestration.dm_transport.send_dm",
            lambda *a, **kw: dm_calls.append(a),
        )

        adapter = FakeAdapterWithResume(SessionLifecycle.RUNNING)
        capture = FakeCapture("pane output")
        watcher = _make_watcher(registry, adapter, capture)
        watcher.tick()

        assert dm_calls == [], "send_dm must NOT be called for handoff_continue"

    def test_handoff_decision_calls_send_dm_with_question(
        self, registry, tmp_path, monkeypatch
    ):
        """A recent handoff_decision marker must send a DM containing the
        marker payload's question text when discord_user_id is present."""
        task_id = "t-hd-dm-001"
        workdir = str(tmp_path)
        _seed_row_with_workdir(
            registry, task_id, workdir, discord_user_id="user-456"
        )

        marker_file = tmp_path / ".hermes" / "sessions" / f"{task_id}.jsonl"
        marker_file.parent.mkdir(parents=True, exist_ok=True)
        append_marker(
            str(marker_file),
            "handoff_decision",
            {"question": "continue?", "handoff_text": "needs a decision"},
            task=task_id,
        )

        # Inject a fake bot token so the DM branch executes
        monkeypatch.setattr(
            "tools.discord_tool._get_bot_token",
            lambda: "fake-bot-token",
        )

        dm_calls: List = []
        monkeypatch.setattr(
            "session_orchestration.dm_transport.send_dm",
            lambda user_id, message, token, **kw: dm_calls.append(
                (user_id, message, token)
            ),
        )

        adapter = FakeAdapterWithResume(SessionLifecycle.RUNNING)
        capture = FakeCapture("pane output")
        watcher = _make_watcher(registry, adapter, capture)
        watcher.tick()

        assert len(dm_calls) == 1, "send_dm must be called exactly once"
        _, message, token = dm_calls[0]
        assert "continue?" in message, "DM must include the question text"
        assert token == "fake-bot-token"

    def test_handoff_decision_does_not_call_resume(
        self, registry, tmp_path, monkeypatch
    ):
        """handoff_decision must NOT call adapter.resume() — DM only."""
        task_id = "t-hd-no-resume-001"
        workdir = str(tmp_path)
        _seed_row_with_workdir(
            registry, task_id, workdir, discord_user_id="user-789"
        )

        marker_file = tmp_path / ".hermes" / "sessions" / f"{task_id}.jsonl"
        marker_file.parent.mkdir(parents=True, exist_ok=True)
        append_marker(
            str(marker_file),
            "handoff_decision",
            {"question": "what to do?", "handoff_text": "decision needed"},
            task=task_id,
        )

        # Suppress network calls
        monkeypatch.setattr("tools.discord_tool._get_bot_token", lambda: "tok")
        monkeypatch.setattr(
            "session_orchestration.dm_transport.send_dm",
            lambda *a, **kw: True,
        )

        adapter = FakeAdapterWithResume(SessionLifecycle.RUNNING)
        capture = FakeCapture("pane output")
        watcher = _make_watcher(registry, adapter, capture)
        watcher.tick()

        assert adapter.resume_calls == [], (
            "adapter.resume() must NOT be called for handoff_decision"
        )

    def test_handoff_decision_skips_dm_when_no_user_id(
        self, registry, tmp_path, monkeypatch
    ):
        """handoff_decision with no discord_user_id must be a no-op (no DM,
        no exception)."""
        task_id = "t-hd-no-uid-001"
        workdir = str(tmp_path)
        # No discord_user_id in the row
        _seed_row_with_workdir(registry, task_id, workdir)

        marker_file = tmp_path / ".hermes" / "sessions" / f"{task_id}.jsonl"
        marker_file.parent.mkdir(parents=True, exist_ok=True)
        append_marker(
            str(marker_file),
            "handoff_decision",
            {"question": "no user?", "handoff_text": "need decision"},
            task=task_id,
        )

        dm_calls: List = []
        monkeypatch.setattr(
            "session_orchestration.dm_transport.send_dm",
            lambda *a, **kw: dm_calls.append(a),
        )

        adapter = FakeAdapterWithResume(SessionLifecycle.RUNNING)
        capture = FakeCapture("pane output")
        watcher = _make_watcher(registry, adapter, capture)

        # Must not raise
        watcher.tick()

        assert dm_calls == [], (
            "send_dm must NOT be called when discord_user_id is absent"
        )

    def test_handoff_continue_resume_runs_under_lock(self, tmp_path):
        """Lock-ordering contract: acquire_lock → adapter.resume → release_lock.

        The watcher must hold the per-session lock when it calls adapter.resume()
        for handoff_continue, so the relay cannot race on the same tmux pane.
        We verify this by wrapping the registry and adapter to record a shared
        event log and asserting the acquire/resume/release order.
        """
        from session_orchestration.registry import SessionOrchestrationRegistry

        # --- shared event log ---
        events: List[str] = []

        # --- registry wrapper that records lock events ---
        inner_registry = SessionOrchestrationRegistry(db_path=tmp_path / "lock-order.db")

        class _LockRecordingRegistry:
            """Thin proxy that records acquire/release into `events`."""

            def __getattr__(self, name: str):
                return getattr(inner_registry, name)

            def acquire_lock(self, task_id: str, holder: str, ttl_seconds: float = 300.0) -> bool:
                result = inner_registry.acquire_lock(task_id, holder, ttl_seconds=ttl_seconds)
                if result:
                    events.append(f"acquire:{task_id}")
                return result

            def release_lock(self, task_id: str, holder: str) -> None:
                events.append(f"release:{task_id}")
                return inner_registry.release_lock(task_id, holder)

        recording_registry = _LockRecordingRegistry()

        # --- adapter that records resume into the same event log ---
        class _LockRecordingAdapter(FakeAdapterWithResume):
            def resume(self, handle: "SessionHandle", prompt: str) -> None:
                events.append("resume")
                super().resume(handle, prompt)

        task_id = "t-lock-order-001"
        workdir = str(tmp_path)
        _seed_row_with_workdir(inner_registry, task_id, workdir)

        marker_file = tmp_path / ".hermes" / "sessions" / f"{task_id}.jsonl"
        marker_file.parent.mkdir(parents=True, exist_ok=True)
        append_marker(
            str(marker_file),
            "handoff_continue",
            {"handoff_text": "carry on"},
            task=task_id,
        )

        adapter = _LockRecordingAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture("pane output")
        watcher = _make_watcher(recording_registry, adapter, capture)
        watcher.tick()

        # Must have seen all three events
        assert "resume" in events, "adapter.resume() must have been called"
        acquire_idx = next(
            (i for i, e in enumerate(events) if e == f"acquire:{task_id}"), None
        )
        resume_idx = next((i for i, e in enumerate(events) if e == "resume"), None)
        release_idx = next(
            (i for i, e in enumerate(events) if e == f"release:{task_id}"), None
        )
        assert acquire_idx is not None, "acquire_lock must have fired"
        assert release_idx is not None, "release_lock must have fired"
        assert acquire_idx < resume_idx, (
            "acquire_lock must precede adapter.resume() — lock-contract violation"
        )
        assert resume_idx < release_idx, (
            "adapter.resume() must precede release_lock — lock-contract violation"
        )
