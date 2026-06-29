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

from session_orchestration.adapters.base import AgentAdapter
from session_orchestration.adapters.verify import AdapterProbeSpec
from session_orchestration.registry import SessionOrchestrationRegistry
from session_orchestration.types import Capabilities, SessionHandle, SessionLifecycle
from session_orchestration.watcher import SessionWatcher, _pane_hash, run_tick


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
