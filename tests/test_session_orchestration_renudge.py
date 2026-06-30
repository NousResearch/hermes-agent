"""
Unit tests for T-FEED-004 — attention re-nudge after X minutes.

Coverage (tasks (a)-(f) from acceptance criteria):

(a) A row in WAITING_USER with attention_since older than renudge_after_seconds
    → re-nudge DM fires once and last_renudge_at is set.
(b) Within the interval → does NOT re-fire.
(c) After another full interval → fires again.
(d) A state change out of the attention set clears attention_since/last_renudge_at.
(e) renudge_after_seconds=0 disables re-nudging entirely.
(f) No discord_user_id → no DM, no crash.

Plus:
- entering attention sets attention_since, clears last_renudge_at.
- existing watcher tests are not broken (FakeAdapter matches the existing interface).

All tests use in-memory SQLite, injected _now_fn, and injected _send_dm_fn so
no real time passes and no Discord API calls are made.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from session_orchestration.adapters.base import AgentAdapter
from session_orchestration.adapters.verify import AdapterProbeSpec
from session_orchestration.registry import SessionOrchestrationRegistry
from session_orchestration.types import Capabilities, SessionHandle, SessionLifecycle
from session_orchestration.watcher import (
    SessionWatcher,
    _check_renudge,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeAdapter(AgentAdapter):
    """Adapter that returns a fixed lifecycle value."""

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


class FakeCapture:
    """Fake tmux_capture returning fixed text."""

    def __init__(self, text: str = "some pane output"):
        self._text = text

    def __call__(self, pane: str) -> str:
        return self._text


class FakeProbeRunner:
    """ProbeRunner that passes every binary."""

    def which(self, binary: str) -> str | None:
        return f"/usr/local/bin/{binary}"

    def help_text(self, binary: str) -> str:
        return ""


class RecordingDmFn:
    """Injectable send_dm_fn that records calls and returns True."""

    def __init__(self) -> None:
        self.calls: List[tuple] = []  # [(user_id, msg), ...]

    def __call__(self, user_id: str, msg: str) -> bool:
        self.calls.append((user_id, msg))
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXED_NOW = 1_700_000_000.0  # arbitrary fixed epoch
_RENUDGE_AFTER = 1800  # 30 min — matches default


def _fake_probe_spec() -> AdapterProbeSpec:
    return AdapterProbeSpec(binary="fake-agent")


def _make_watcher(
    registry: SessionOrchestrationRegistry,
    adapter: FakeAdapter,
    *,
    now_fn=None,
    send_dm_fn=None,
    adapter_name: str = "fake",
) -> SessionWatcher:
    """Build a SessionWatcher with all network/time dependencies injected."""
    watcher = SessionWatcher(
        registry=registry,
        adapters={adapter_name: adapter},
        tmux_capture=FakeCapture(),
        tmux_liveness_fn=lambda _s: True,  # pane always alive
        _now_fn=now_fn if now_fn is not None else (lambda: _FIXED_NOW),
        _send_dm_fn=send_dm_fn if send_dm_fn is not None else RecordingDmFn(),
    )
    probe_runner = FakeProbeRunner()
    probe_specs = {type(adapter).__name__: _fake_probe_spec()}
    watcher.startup_verify(probe_runner=probe_runner, probe_specs=probe_specs)
    return watcher


def _seed_waiting_user(
    registry: SessionOrchestrationRegistry,
    task_id: str,
    *,
    discord_user_id: Optional[str] = "discord-user-999",
    attention_since: Optional[float] = None,
    last_renudge_at: Optional[float] = None,
) -> None:
    """Insert a WAITING_USER row and optionally pre-stamp attention columns."""
    registry.upsert(
        task_id,
        agent="fake",
        run_id=f"run-{uuid.uuid4().hex[:8]}",
        repo=f"repo-{uuid.uuid4().hex[:8]}",
        state="WAITING_USER",
        tmux_session=f"hermes-fake-{uuid.uuid4().hex[:6]}",
        discord_user_id=discord_user_id,
    )
    if attention_since is not None or last_renudge_at is not None:
        registry.set_attention_stamps(task_id, attention_since, last_renudge_at)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "state.db"


@pytest.fixture()
def registry(db_path: Path) -> SessionOrchestrationRegistry:
    return SessionOrchestrationRegistry(db_path=db_path)


# ---------------------------------------------------------------------------
# Test (a): re-nudge fires when attention_since is older than the interval
# ---------------------------------------------------------------------------


class TestRenudgeFiresAfterInterval:
    def test_dm_fires_once_and_last_renudge_at_set(self, registry: SessionOrchestrationRegistry):
        task_id = "t-renudge-a"
        # attention_since is 2000s ago — past the 1800s threshold
        attention_since = _FIXED_NOW - 2000
        _seed_waiting_user(
            registry, task_id, attention_since=attention_since, last_renudge_at=None
        )

        dm = RecordingDmFn()
        watcher = _make_watcher(
            registry,
            FakeAdapter(SessionLifecycle.WAITING_USER),
            now_fn=lambda: _FIXED_NOW,
            send_dm_fn=dm,
        )

        with patch(
            "session_orchestration.watcher._load_renudge_after_seconds_cfg",
            return_value=_RENUDGE_AFTER,
        ):
            watcher.tick()

        assert len(dm.calls) == 1, "expected exactly one DM"
        user_id, msg = dm.calls[0]
        assert user_id == "discord-user-999"
        assert "still needs your input" in msg
        assert "waiting " in msg

        row = registry.get(task_id)
        assert row is not None
        assert row["last_renudge_at"] is not None
        assert abs(float(row["last_renudge_at"]) - _FIXED_NOW) < 1.0


# ---------------------------------------------------------------------------
# Test (b): within the interval → does NOT fire
# ---------------------------------------------------------------------------


class TestRenudgeWithinInterval:
    def test_no_dm_within_interval(self, registry: SessionOrchestrationRegistry):
        task_id = "t-renudge-b"
        # attention_since is only 500s ago — under the 1800s threshold
        attention_since = _FIXED_NOW - 500
        _seed_waiting_user(
            registry, task_id, attention_since=attention_since, last_renudge_at=None
        )

        dm = RecordingDmFn()
        watcher = _make_watcher(
            registry,
            FakeAdapter(SessionLifecycle.WAITING_USER),
            now_fn=lambda: _FIXED_NOW,
            send_dm_fn=dm,
        )

        with patch(
            "session_orchestration.watcher._load_renudge_after_seconds_cfg",
            return_value=_RENUDGE_AFTER,
        ):
            watcher.tick()

        assert len(dm.calls) == 0, "DM must not fire within the interval"

        row = registry.get(task_id)
        # last_renudge_at should still be NULL (no nudge fired)
        assert row is not None
        assert row["last_renudge_at"] is None

    def test_last_renudge_at_resets_window(self, registry: SessionOrchestrationRegistry):
        """If last_renudge_at was just set, the window uses it as the reference."""
        task_id = "t-renudge-b2"
        attention_since = _FIXED_NOW - 3600  # well past initial threshold
        last_renudge_at = _FIXED_NOW - 500   # but re-nudge was recent
        _seed_waiting_user(
            registry, task_id,
            attention_since=attention_since,
            last_renudge_at=last_renudge_at,
        )

        dm = RecordingDmFn()
        watcher = _make_watcher(
            registry,
            FakeAdapter(SessionLifecycle.WAITING_USER),
            now_fn=lambda: _FIXED_NOW,
            send_dm_fn=dm,
        )

        with patch(
            "session_orchestration.watcher._load_renudge_after_seconds_cfg",
            return_value=_RENUDGE_AFTER,
        ):
            watcher.tick()

        # 500s since last_renudge_at < 1800s → should NOT fire
        assert len(dm.calls) == 0


# ---------------------------------------------------------------------------
# Test (c): after another full interval → fires again
# ---------------------------------------------------------------------------


class TestRenudgeRepeats:
    def test_fires_again_after_second_interval(self, registry: SessionOrchestrationRegistry):
        task_id = "t-renudge-c"
        attention_since = _FIXED_NOW - 4000
        # Simulate: re-nudge already fired 2000s ago (> 1800s → should fire again)
        last_renudge_at = _FIXED_NOW - 2000

        _seed_waiting_user(
            registry, task_id,
            attention_since=attention_since,
            last_renudge_at=last_renudge_at,
        )

        dm = RecordingDmFn()
        watcher = _make_watcher(
            registry,
            FakeAdapter(SessionLifecycle.WAITING_USER),
            now_fn=lambda: _FIXED_NOW,
            send_dm_fn=dm,
        )

        with patch(
            "session_orchestration.watcher._load_renudge_after_seconds_cfg",
            return_value=_RENUDGE_AFTER,
        ):
            watcher.tick()

        assert len(dm.calls) == 1, "expected re-nudge to fire again after second interval"

        row = registry.get(task_id)
        assert row is not None
        # last_renudge_at must be updated to ~now
        assert abs(float(row["last_renudge_at"]) - _FIXED_NOW) < 1.0


# ---------------------------------------------------------------------------
# Test (d): state change out of attention clears attention_since/last_renudge_at
# ---------------------------------------------------------------------------


class TestAttentionStampsClearedOnLeave:
    def test_clearing_on_transition_to_running(self, registry: SessionOrchestrationRegistry):
        task_id = "t-renudge-d"
        # Row starts in WAITING_USER with stamps set
        attention_since = _FIXED_NOW - 2000
        _seed_waiting_user(
            registry, task_id,
            attention_since=attention_since,
            last_renudge_at=_FIXED_NOW - 100,
        )

        # Adapter will now detect RUNNING → transition out of attention
        dm = RecordingDmFn()
        watcher = _make_watcher(
            registry,
            FakeAdapter(SessionLifecycle.RUNNING),
            now_fn=lambda: _FIXED_NOW,
            send_dm_fn=dm,
        )

        with patch("session_orchestration.feed.push_turn_change", return_value=None), \
             patch(
                 "session_orchestration.watcher._load_renudge_after_seconds_cfg",
                 return_value=_RENUDGE_AFTER,
             ):
            watcher.tick()

        row = registry.get(task_id)
        assert row is not None
        assert row["state"] == "RUNNING"
        assert row["attention_since"] is None, "attention_since must be cleared on leave"
        assert row["last_renudge_at"] is None, "last_renudge_at must be cleared on leave"

        # No re-nudge DM should fire (leaving attention, not staying)
        assert len(dm.calls) == 0


class TestAttentionSinceSetOnEnter:
    def test_attention_since_stamped_on_enter(self, registry: SessionOrchestrationRegistry):
        task_id = "t-renudge-enter"
        # Row starts in RUNNING (no attention stamps)
        registry.upsert(
            task_id,
            agent="fake",
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            repo=f"repo-{uuid.uuid4().hex[:8]}",
            state="RUNNING",
            tmux_session="hermes-fake-enter",
            discord_user_id="discord-user-999",
        )

        # Adapter transitions to WAITING_USER → entering attention
        dm = RecordingDmFn()
        watcher = _make_watcher(
            registry,
            FakeAdapter(SessionLifecycle.WAITING_USER),
            now_fn=lambda: _FIXED_NOW,
            send_dm_fn=dm,
        )

        with patch("session_orchestration.feed.push_turn_change", return_value=None), \
             patch(
                 "session_orchestration.watcher._load_renudge_after_seconds_cfg",
                 return_value=_RENUDGE_AFTER,
             ):
            watcher.tick()

        row = registry.get(task_id)
        assert row is not None
        assert row["state"] == "WAITING_USER"
        assert row["attention_since"] is not None, "attention_since must be set on enter"
        assert abs(float(row["attention_since"]) - _FIXED_NOW) < 1.0
        assert row["last_renudge_at"] is None, "last_renudge_at must be NULL on enter"

        # No re-nudge DM on the entering tick (it fires only on staying ticks)
        assert len(dm.calls) == 0


# ---------------------------------------------------------------------------
# Test (e): renudge_after_seconds=0 disables
# ---------------------------------------------------------------------------


class TestRenudgeDisabledByZero:
    def test_zero_disables_renudge(self, registry: SessionOrchestrationRegistry):
        task_id = "t-renudge-e"
        attention_since = _FIXED_NOW - 9999  # very old — would fire if enabled
        _seed_waiting_user(
            registry, task_id, attention_since=attention_since, last_renudge_at=None
        )

        dm = RecordingDmFn()
        watcher = _make_watcher(
            registry,
            FakeAdapter(SessionLifecycle.WAITING_USER),
            now_fn=lambda: _FIXED_NOW,
            send_dm_fn=dm,
        )

        with patch(
            "session_orchestration.watcher._load_renudge_after_seconds_cfg",
            return_value=0,  # disabled
        ):
            watcher.tick()

        assert len(dm.calls) == 0, "DM must not fire when renudge_after_seconds=0"

    def test_negative_also_disables(self, registry: SessionOrchestrationRegistry):
        task_id = "t-renudge-e2"
        attention_since = _FIXED_NOW - 9999
        _seed_waiting_user(
            registry, task_id, attention_since=attention_since, last_renudge_at=None
        )

        dm = RecordingDmFn()
        watcher = _make_watcher(
            registry,
            FakeAdapter(SessionLifecycle.WAITING_USER),
            now_fn=lambda: _FIXED_NOW,
            send_dm_fn=dm,
        )

        with patch(
            "session_orchestration.watcher._load_renudge_after_seconds_cfg",
            return_value=-1,
        ):
            watcher.tick()

        assert len(dm.calls) == 0


# ---------------------------------------------------------------------------
# Test (f): no discord_user_id → no DM, no crash
# ---------------------------------------------------------------------------


class TestRenudgeNoUserId:
    def test_no_dm_when_no_user_id(self, registry: SessionOrchestrationRegistry):
        task_id = "t-renudge-f"
        attention_since = _FIXED_NOW - 2000
        # No discord_user_id
        _seed_waiting_user(
            registry, task_id,
            discord_user_id=None,
            attention_since=attention_since,
            last_renudge_at=None,
        )

        dm = RecordingDmFn()
        watcher = _make_watcher(
            registry,
            FakeAdapter(SessionLifecycle.WAITING_USER),
            now_fn=lambda: _FIXED_NOW,
            send_dm_fn=dm,
        )

        # Should not crash, should not call DM
        with patch(
            "session_orchestration.watcher._load_renudge_after_seconds_cfg",
            return_value=_RENUDGE_AFTER,
        ):
            watcher.tick()

        assert len(dm.calls) == 0, "No DM when discord_user_id is absent"

        # last_renudge_at is still updated (best-effort, independent of DM success)
        row = registry.get(task_id)
        assert row is not None
        assert row["last_renudge_at"] is not None


# ---------------------------------------------------------------------------
# Direct unit test for _check_renudge
# ---------------------------------------------------------------------------


class TestCheckRenudgeDirect:
    """Direct unit tests for the _check_renudge module-level function."""

    def _make_registry(self, tmp_path: Path) -> SessionOrchestrationRegistry:
        return SessionOrchestrationRegistry(db_path=tmp_path / "state.db")

    def test_fires_when_interval_elapsed(self, tmp_path: Path):
        registry = self._make_registry(tmp_path)
        task_id = "t-direct-a"
        registry.upsert(task_id, agent="fake", discord_user_id="u-001", state="WAITING_USER")
        registry.set_attention_stamps(task_id, _FIXED_NOW - 2000, None)

        fresh_row = registry.get(task_id)
        dm = RecordingDmFn()

        with patch(
            "session_orchestration.watcher._load_renudge_after_seconds_cfg",
            return_value=1800,
        ):
            _check_renudge(task_id, fresh_row, _FIXED_NOW, registry=registry, send_dm_fn=dm)

        assert len(dm.calls) == 1
        assert dm.calls[0][0] == "u-001"
        updated_row = registry.get(task_id)
        assert updated_row["last_renudge_at"] is not None

    def test_skips_when_no_attention_since(self, tmp_path: Path):
        registry = self._make_registry(tmp_path)
        task_id = "t-direct-b"
        registry.upsert(task_id, agent="fake", discord_user_id="u-001", state="WAITING_USER")
        # No attention_since stamp set

        fresh_row = registry.get(task_id)
        dm = RecordingDmFn()

        with patch(
            "session_orchestration.watcher._load_renudge_after_seconds_cfg",
            return_value=1800,
        ):
            _check_renudge(task_id, fresh_row, _FIXED_NOW, registry=registry, send_dm_fn=dm)

        assert len(dm.calls) == 0

    def test_paused_handoff_also_renudges(self, tmp_path: Path):
        """Re-nudge fires for PAUSED_HANDOFF as well as WAITING_USER."""
        registry = self._make_registry(tmp_path)
        task_id = "t-direct-c"
        registry.upsert(
            task_id, agent="fake", discord_user_id="u-001", state="PAUSED_HANDOFF"
        )
        registry.set_attention_stamps(task_id, _FIXED_NOW - 2000, None)

        fresh_row = registry.get(task_id)
        dm = RecordingDmFn()

        with patch(
            "session_orchestration.watcher._load_renudge_after_seconds_cfg",
            return_value=1800,
        ):
            _check_renudge(task_id, fresh_row, _FIXED_NOW, registry=registry, send_dm_fn=dm)

        assert len(dm.calls) == 1

    def test_dm_failure_does_not_prevent_last_renudge_at_update(self, tmp_path: Path):
        """Even if send_dm_fn raises, last_renudge_at should be updated (best-effort)."""
        registry = self._make_registry(tmp_path)
        task_id = "t-direct-d"
        registry.upsert(task_id, agent="fake", discord_user_id="u-001", state="WAITING_USER")
        registry.set_attention_stamps(task_id, _FIXED_NOW - 2000, None)

        fresh_row = registry.get(task_id)

        def failing_dm(user_id: str, msg: str) -> bool:
            raise RuntimeError("network error")

        with patch(
            "session_orchestration.watcher._load_renudge_after_seconds_cfg",
            return_value=1800,
        ):
            # Should not raise despite the DM failure
            _check_renudge(
                task_id, fresh_row, _FIXED_NOW, registry=registry, send_dm_fn=failing_dm
            )

        updated_row = registry.get(task_id)
        assert updated_row["last_renudge_at"] is not None
