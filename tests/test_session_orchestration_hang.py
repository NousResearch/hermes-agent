"""
Unit tests for T009 — hang detection + one auto-nudge.

Critical behaviors tested (all 4 from the audit-blocker list):

1. Hang ONLY when state==RUNNING.
   A WAITING_USER session held at the prompt for >N ticks → NO nudge.

2. Stale-accelerant cannot mask a real hang.
   A "heartbeat accelerant" that carries no fresh activity (freshness TTL
   exceeded) must NOT suppress hang detection.  Hang is derived from
   static pane-staleness signals only (idle_ticks + last_output_ts).

3. Exactly one auto-nudge/action per stale episode.
   First tick over threshold → notify + nudge + nudge_count incremented.
   Second tick still hung → no notification and no second nudge.

4. Simulated long build (pane static under active-tool indicator) → no hang.
   An adapter that declares an ``idle_indicator_regex`` matching the static
   pane text must suppress hang detection even when idle_ticks > threshold.

Tests use in-memory SQLite, FakeAdapter / FakeRelay fakes, and monkeypatched
feed so no Discord network calls are made.
"""

from __future__ import annotations

import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from session_orchestration.adapters.base import AgentAdapter
from session_orchestration.adapters.verify import AdapterProbeSpec
from session_orchestration.registry import SessionOrchestrationRegistry
from session_orchestration.types import Capabilities, SessionHandle, SessionLifecycle
from session_orchestration.watcher import (
    SessionWatcher,
    _on_hang,
    _pane_hash,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeAdapter(AgentAdapter):
    """Adapter returning a fixed lifecycle; optionally declares idle_indicator_regex."""

    def __init__(
        self,
        lifecycle: SessionLifecycle = SessionLifecycle.RUNNING,
        idle_indicator_regex: Optional[re.Pattern] = None,
    ):
        self._lifecycle = lifecycle
        self._idle_regex = idle_indicator_regex
        self.detect_calls: List[SessionHandle] = []
        self.drive_calls: List[str] = []

    def capabilities(self) -> Capabilities:
        return Capabilities(idle_indicator_regex=self._idle_regex)

    def launch(self, workdir: str, prompt: str) -> SessionHandle:
        raise NotImplementedError

    def drive(self, handle: SessionHandle, message: str) -> None:
        self.drive_calls.append(message)

    def detect(self, handle: SessionHandle) -> SessionLifecycle:
        self.detect_calls.append(handle)
        return self._lifecycle

    def resume(self, handle: SessionHandle, prompt: str) -> None:
        raise NotImplementedError


class FakeCapture:
    """Fake tmux_capture returning fixed text."""

    def __init__(self, text: str = "some pane output"):
        self._text = text
        self.calls: List[str] = []

    def __call__(self, pane: str) -> str:
        self.calls.append(pane)
        return self._text


class FakeProbeRunner:
    """ProbeRunner that passes every binary."""

    def which(self, binary: str) -> str | None:
        return f"/usr/local/bin/{binary}"

    def help_text(self, binary: str) -> str:
        return ""


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
    return AdapterProbeSpec(binary="fake-agent")


def _make_watcher(
    registry: SessionOrchestrationRegistry,
    adapter: FakeAdapter,
    capture: FakeCapture,
    *,
    adapter_name: str = "fake",
) -> SessionWatcher:
    watcher = SessionWatcher(
        registry=registry,
        adapters={adapter_name: adapter},
        tmux_capture=capture,
    )
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
    idle_ticks: int = 0,
    nudge_count: int = 0,
    last_output_ts: Optional[float] = None,
    last_pane_hash: Optional[str] = None,
) -> None:
    """Insert a registry row with optional pre-seeded hang-relevant fields."""
    registry.upsert(
        task_id,
        agent=agent,
        run_id=f"run-{uuid.uuid4().hex[:8]}",
        repo=f"repo-{uuid.uuid4().hex[:8]}",
        state=state,
        tmux_session=tmux_session,
        idle_ticks=idle_ticks,
        nudge_count=nudge_count,
        last_output_ts=last_output_ts,
        last_pane_hash=last_pane_hash,
    )


# ---------------------------------------------------------------------------
# Hang-config helper — make hang threshold small so tests don't need many ticks
# ---------------------------------------------------------------------------


class _SmallThresholdConfig:
    """Minimal config stub with low thresholds for testing."""
    hang_idle_ticks: int = 2
    hang_stale_seconds: int = 1  # 1 second so tests don't need to wait


def _low_threshold_config():
    """Patch target that returns a config with low hang thresholds."""
    return _SmallThresholdConfig()


def _stale_guard_kwargs(pane_text: str = "stale unchanged pane") -> Dict[str, Any]:
    pane_hash = _pane_hash(pane_text)
    return {
        "pane_text": pane_text,
        "previous_pane_hash": pane_hash,
        "current_pane_hash": pane_hash,
    }


# ---------------------------------------------------------------------------
# Behavior 1: WAITING_USER held >N ticks → NO nudge
# ---------------------------------------------------------------------------


class TestWaitingUserNoHang:
    """A session in WAITING_USER state must never trigger a hang or nudge,
    even if idle_ticks exceeds the threshold and last_output_ts is very old.
    """

    def test_waiting_user_no_nudge_after_many_idle_ticks(self, registry, db_path):
        """WAITING_USER + idle_ticks >> threshold → _on_hang never called."""
        task_id = "t-hang-wu-001"
        # Seed row in WAITING_USER with many idle ticks and very old last_output_ts
        old_ts = time.time() - 9999
        _seed_row(
            registry,
            task_id=task_id,
            state="WAITING_USER",
            idle_ticks=100,
            last_output_ts=old_ts,
        )

        # Adapter returns WAITING_USER — session is at the prompt
        adapter = FakeAdapter(SessionLifecycle.WAITING_USER)
        static_text = "❯ "  # same every tick → pane unchanged
        capture = FakeCapture(static_text)

        hang_called = []

        original_on_hang = __import__(
            "session_orchestration.watcher", fromlist=["_on_hang"]
        )._on_hang

        def spy_on_hang(task_id, row, **kwargs):
            hang_called.append(task_id)
            original_on_hang(task_id, row, **kwargs)

        watcher = _make_watcher(registry, adapter, capture)

        notifications = []
        with (
            patch(
                "session_orchestration.watcher._on_hang",
                side_effect=spy_on_hang,
            ),
            patch(
                "session_orchestration.feed.push_hang_notification",
                side_effect=lambda *a, **kw: notifications.append(a),
            ),
        ):
            watcher.tick()
            watcher.tick()
            watcher.tick()

        assert hang_called == [], (
            "_on_hang must NOT be called for WAITING_USER sessions; "
            f"was called {len(hang_called)} time(s)"
        )
        assert notifications == [], (
            "push_hang_notification must NOT fire for a WAITING_USER session"
        )

    def test_waiting_user_state_is_excluded_from_hang_callsite(self, registry, db_path):
        """The call-site in _process_row gates on state==RUNNING; this test
        verifies the gating via a full watcher.tick() so the invariant is
        proven end-to-end, not just by reading the code."""
        task_id = "t-hang-wu-002"
        old_ts = time.time() - 9999
        _seed_row(
            registry,
            task_id=task_id,
            state="RUNNING",  # seed RUNNING so the row is iterated
            idle_ticks=0,
            last_output_ts=old_ts,
        )

        # Adapter returns WAITING_USER (transition: RUNNING→WAITING_USER).
        # After the transition, state==WAITING_USER → _on_hang must NOT fire.
        adapter = FakeAdapter(SessionLifecycle.WAITING_USER)
        capture = FakeCapture("❯ prompt")

        hang_called = []
        with patch(
            "session_orchestration.watcher._on_hang",
            side_effect=lambda tid, row, **kw: hang_called.append(tid),
        ):
            watcher = _make_watcher(registry, adapter, capture)
            watcher.tick()

        # Verify state is now WAITING_USER
        row = registry.get(task_id)
        assert row["state"] == "WAITING_USER"
        assert hang_called == [], (
            "_on_hang must NOT fire when new_state is WAITING_USER"
        )


# ---------------------------------------------------------------------------
# Behavior 2: Stale accelerant cannot mask a real hang
# ---------------------------------------------------------------------------


class TestStaleAccelerantCannotMaskHang:
    """A 'stale' accelerant (e.g. an old heartbeat signal) must NOT suppress
    detection of a genuinely hung session.  Hang detection uses STATIC thresholds
    (idle_ticks + last_output_ts) only — not any accelerant flag.
    """

    def test_stale_accelerant_does_not_suppress_hang(self, registry, db_path):
        """Simulate a scenario where an accelerant was last seen a long time ago.
        The watcher must still detect the hang via idle_ticks + last_output_ts.

        In the real codebase the accelerant resets idle_ticks / last_output_ts when
        it fires with fresh activity.  A STALE accelerant simply never fires —
        so idle_ticks keeps incrementing and last_output_ts stays old.  _on_hang
        reads these static signals and must declare hang regardless of whether
        an accelerant 'would have' fired if it were active.

        This test verifies that _on_hang reaches the hang-confirmed branch
        (does NOT return early) when thresholds are exceeded, even when there
        is no accelerant flag set on the row.
        """
        task_id = "t-hang-accel-001"
        old_ts = time.time() - 9999  # very old — stale threshold exceeded
        _seed_row(
            registry,
            task_id=task_id,
            state="RUNNING",
            idle_ticks=100,       # well above any reasonable threshold
            nudge_count=0,
            last_output_ts=old_ts,
        )
        row = registry.get(task_id)

        hang_notifications: List[Any] = []
        nudge_increments: List[str] = []

        with (
            patch(
                "session_orchestration.config.load_session_orchestration_config",
                return_value=_low_threshold_config(),
            ),
            patch(
                "session_orchestration.feed.push_hang_notification",
                side_effect=lambda *a, **kw: hang_notifications.append(a),
            ),
            patch.object(
                registry,
                "increment_counter",
                wraps=registry.increment_counter,
                side_effect=lambda tid, col, **kw: nudge_increments.append(col)
                or registry.__class__.increment_counter(registry, tid, col, **kw),
            ),
        ):
            _on_hang(task_id, row, registry=registry, **_stale_guard_kwargs())

        assert hang_notifications, (
            "push_hang_notification must be called when thresholds are exceeded "
            "(stale accelerant cannot mask a real hang)"
        )

    def test_fresh_accelerant_resets_via_pane_change(self, registry, db_path):
        """A FRESH accelerant fires by resetting idle_ticks and last_output_ts
        (because it carries fresh activity — the pane changed).  This test
        verifies the mechanism: a changed pane resets idle_ticks to 0, which
        means the _on_hang call-site condition (idle_ticks > 0) is not met,
        so _on_hang is not called at all.

        This is the CORRECT suppression path — via actual fresh pane activity,
        not via a blanket suppression flag.
        """
        task_id = "t-hang-accel-002"
        # Start with a row that previously had idle_ticks = 5
        _seed_row(
            registry,
            task_id=task_id,
            state="RUNNING",
            idle_ticks=5,
            last_output_ts=time.time() - 9999,
            last_pane_hash=_pane_hash("old pane content"),
        )

        # On this tick the pane changes → idle_ticks resets to 0 → no hang
        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture("NEW pane content — activity!")  # different from old hash

        hang_called = []
        with patch(
            "session_orchestration.watcher._on_hang",
            side_effect=lambda tid, row, **kw: hang_called.append(tid),
        ):
            watcher = _make_watcher(registry, adapter, capture)
            watcher.tick()

        row = registry.get(task_id)
        assert (row.get("idle_ticks") or 0) == 0, "idle_ticks should reset on pane change"
        assert hang_called == [], (
            "_on_hang must NOT be called when the pane changed this tick"
        )


# ---------------------------------------------------------------------------
# Behavior 3: Exactly one auto-nudge/action per stale episode
# ---------------------------------------------------------------------------


class TestExactlyOneNudgePerEpisode:
    """First confirmed hang acts once; later unchanged-pane ticks do nothing."""

    def test_first_hang_sends_nudge(self, registry, db_path):
        """nudge_count==0 → push_hang_notification(escalate=False) + relay nudge."""
        task_id = "t-hang-nudge-001"
        old_ts = time.time() - 9999
        _seed_row(
            registry,
            task_id=task_id,
            state="RUNNING",
            idle_ticks=100,
            nudge_count=0,
            last_output_ts=old_ts,
        )
        row = registry.get(task_id)
        adapter = FakeAdapter(SessionLifecycle.RUNNING)

        notifications: List[Any] = []
        nudge_relay_calls: List[str] = []

        def fake_push_hang(tid, r, escalate=False, **kw):
            notifications.append({"task_id": tid, "escalate": escalate})

        with (
            patch(
                "session_orchestration.config.load_session_orchestration_config",
                return_value=_low_threshold_config(),
            ),
            patch(
                "session_orchestration.feed.push_hang_notification",
                side_effect=fake_push_hang,
            ),
            patch(
                "session_orchestration.watcher._send_auto_nudge",
                side_effect=lambda tid, r, **kw: nudge_relay_calls.append(tid),
            ),
        ):
            _on_hang(task_id, row, registry=registry, adapter=adapter, **_stale_guard_kwargs())

        assert len(notifications) == 1, "Exactly one notification for first hang"
        assert notifications[0]["escalate"] is False, (
            "First hang notification must not be an escalation"
        )
        assert nudge_relay_calls == [task_id], (
            "Auto-nudge must be sent exactly once on first confirmed hang"
        )

        # nudge_count must have been incremented
        fresh = registry.get(task_id)
        assert fresh["nudge_count"] == 1, (
            "nudge_count must be incremented to 1 after the auto-nudge"
        )

    def test_second_hang_does_not_act_again(self, registry, db_path):
        """nudge_count>=1 → no notification and no relay nudge."""
        task_id = "t-hang-nudge-002"
        old_ts = time.time() - 9999
        _seed_row(
            registry,
            task_id=task_id,
            state="RUNNING",
            idle_ticks=100,
            nudge_count=1,  # already nudged once
            last_output_ts=old_ts,
        )
        row = registry.get(task_id)
        adapter = FakeAdapter(SessionLifecycle.RUNNING)

        notifications: List[Any] = []
        nudge_relay_calls: List[str] = []

        def fake_push_hang(tid, r, escalate=False, **kw):
            notifications.append({"task_id": tid, "escalate": escalate})

        with (
            patch(
                "session_orchestration.config.load_session_orchestration_config",
                return_value=_low_threshold_config(),
            ),
            patch(
                "session_orchestration.feed.push_hang_notification",
                side_effect=fake_push_hang,
            ),
            patch(
                "session_orchestration.watcher._send_auto_nudge",
                side_effect=lambda tid, r, **kw: nudge_relay_calls.append(tid),
            ),
        ):
            _on_hang(task_id, row, registry=registry, adapter=adapter, **_stale_guard_kwargs())

        assert notifications == [], (
            "Second hang in the same stale episode must not fire another notification"
        )
        assert nudge_relay_calls == [], (
            "_send_auto_nudge must NOT be called on second hang (already nudged)"
        )

    def test_nudge_count_not_incremented_after_episode_action(self, registry, db_path):
        """nudge_count must stay at 1 after the episode action has fired."""
        task_id = "t-hang-nudge-003"
        old_ts = time.time() - 9999
        _seed_row(
            registry,
            task_id=task_id,
            state="RUNNING",
            idle_ticks=100,
            nudge_count=1,
            last_output_ts=old_ts,
        )
        row = registry.get(task_id)
        adapter = FakeAdapter(SessionLifecycle.RUNNING)

        with (
            patch(
                "session_orchestration.config.load_session_orchestration_config",
                return_value=_low_threshold_config(),
            ),
            patch("session_orchestration.feed.push_hang_notification"),
            patch("session_orchestration.watcher._send_auto_nudge"),
        ):
            _on_hang(task_id, row, registry=registry, adapter=adapter, **_stale_guard_kwargs())

        fresh = registry.get(task_id)
        assert fresh["nudge_count"] == 1, (
            "nudge_count must not be incremented again in the same stale episode"
        )


# ---------------------------------------------------------------------------
# Behavior 4: Simulated long build (static pane + active-tool indicator) → no hang
# ---------------------------------------------------------------------------


class TestActiveBuildIndicatorSuppressesHang:
    """A pane that is static but carries an active-tool indicator regex match
    must NOT trigger a hang alert — the session is busy (building/compiling).
    """

    def test_build_indicator_suppresses_hang(self, registry, db_path):
        """Adapter declares idle_indicator_regex that matches the static pane.
        Even with idle_ticks >> threshold, _on_hang must return early.
        """
        task_id = "t-hang-build-001"
        old_ts = time.time() - 9999
        _seed_row(
            registry,
            task_id=task_id,
            state="RUNNING",
            idle_ticks=100,
            nudge_count=0,
            last_output_ts=old_ts,
        )
        row = registry.get(task_id)

        # Build indicator pattern — matches when compiling
        build_regex = re.compile(r"Building|Compiling|cargo build")
        adapter = FakeAdapter(
            lifecycle=SessionLifecycle.RUNNING,
            idle_indicator_regex=build_regex,
        )
        # Pane text contains the build indicator
        build_pane_text = "  Compiling myproject v0.1.0 (/src/myproject)\n"

        hang_notifications: List[Any] = []
        nudge_calls: List[str] = []

        with (
            patch(
                "session_orchestration.config.load_session_orchestration_config",
                return_value=_low_threshold_config(),
            ),
            patch(
                "session_orchestration.feed.push_hang_notification",
                side_effect=lambda *a, **kw: hang_notifications.append(a),
            ),
            patch(
                "session_orchestration.watcher._send_auto_nudge",
                side_effect=lambda *a, **kw: nudge_calls.append(a),
            ),
        ):
            _on_hang(
                task_id,
                row,
                registry=registry,
                adapter=adapter,
                **_stale_guard_kwargs(build_pane_text),
            )

        assert hang_notifications == [], (
            "push_hang_notification must NOT fire when active-tool indicator matches"
        )
        assert nudge_calls == [], (
            "_send_auto_nudge must NOT be called when active-tool indicator matches"
        )

    def test_no_indicator_regex_falls_through_to_hang(self, registry, db_path):
        """An adapter with no idle_indicator_regex (None) should fall through
        to hang detection — verifying the suppression is opt-in, not always-on.
        """
        task_id = "t-hang-build-002"
        old_ts = time.time() - 9999
        _seed_row(
            registry,
            task_id=task_id,
            state="RUNNING",
            idle_ticks=100,
            nudge_count=0,
            last_output_ts=old_ts,
        )
        row = registry.get(task_id)

        # No idle_indicator_regex — hang must fire
        adapter = FakeAdapter(
            lifecycle=SessionLifecycle.RUNNING,
            idle_indicator_regex=None,
        )
        static_pane_text = "  Compiling myproject v0.1.0 (/src/myproject)\n"

        hang_notifications: List[Any] = []

        with (
            patch(
                "session_orchestration.config.load_session_orchestration_config",
                return_value=_low_threshold_config(),
            ),
            patch(
                "session_orchestration.feed.push_hang_notification",
                side_effect=lambda *a, **kw: hang_notifications.append(a),
            ),
            patch("session_orchestration.watcher._send_auto_nudge"),
        ):
            _on_hang(
                task_id,
                row,
                registry=registry,
                adapter=adapter,
                **_stale_guard_kwargs(static_pane_text),
            )

        assert hang_notifications, (
            "push_hang_notification MUST fire when adapter has no idle_indicator_regex"
        )

    def test_indicator_regex_no_match_falls_through_to_hang(self, registry, db_path):
        """Adapter has an idle_indicator_regex but pane text does not match.
        Hang must fire — suppression only applies on a match.
        """
        task_id = "t-hang-build-003"
        old_ts = time.time() - 9999
        _seed_row(
            registry,
            task_id=task_id,
            state="RUNNING",
            idle_ticks=100,
            nudge_count=0,
            last_output_ts=old_ts,
        )
        row = registry.get(task_id)

        build_regex = re.compile(r"Compiling|Building")
        adapter = FakeAdapter(
            lifecycle=SessionLifecycle.RUNNING,
            idle_indicator_regex=build_regex,
        )
        # Pane text does NOT contain build indicator
        idle_pane_text = "waiting for something...\n"

        hang_notifications: List[Any] = []

        with (
            patch(
                "session_orchestration.config.load_session_orchestration_config",
                return_value=_low_threshold_config(),
            ),
            patch(
                "session_orchestration.feed.push_hang_notification",
                side_effect=lambda *a, **kw: hang_notifications.append(a),
            ),
            patch("session_orchestration.watcher._send_auto_nudge"),
        ):
            _on_hang(
                task_id,
                row,
                registry=registry,
                adapter=adapter,
                **_stale_guard_kwargs(idle_pane_text),
            )

        assert hang_notifications, (
            "push_hang_notification MUST fire when idle_indicator_regex does NOT match"
        )


# ---------------------------------------------------------------------------
# Behavior: Static thresholds — below threshold → no hang
# ---------------------------------------------------------------------------


class TestStaticThresholds:
    """Hang is NOT declared when idle_ticks < hang_idle_ticks or
    elapsed < hang_stale_seconds, regardless of other signals.
    """

    def test_below_idle_tick_threshold_no_hang(self, registry, db_path):
        task_id = "t-hang-thresh-001"
        old_ts = time.time() - 9999
        _seed_row(
            registry,
            task_id=task_id,
            state="RUNNING",
            idle_ticks=1,  # below threshold of 2
            nudge_count=0,
            last_output_ts=old_ts,
        )
        row = registry.get(task_id)
        adapter = FakeAdapter(SessionLifecycle.RUNNING)

        hang_notifications: List[Any] = []

        with (
            patch(
                "session_orchestration.config.load_session_orchestration_config",
                return_value=_low_threshold_config(),
            ),
            patch(
                "session_orchestration.feed.push_hang_notification",
                side_effect=lambda *a, **kw: hang_notifications.append(a),
            ),
        ):
            _on_hang(task_id, row, registry=registry, adapter=adapter, **_stale_guard_kwargs())

        assert hang_notifications == [], (
            "No hang when idle_ticks < hang_idle_ticks threshold"
        )

    def test_below_stale_seconds_threshold_no_hang(self, registry, db_path):
        task_id = "t-hang-thresh-002"
        recent_ts = time.time() - 0.1  # very recent — below 1s stale threshold
        _seed_row(
            registry,
            task_id=task_id,
            state="RUNNING",
            idle_ticks=100,
            nudge_count=0,
            last_output_ts=recent_ts,
        )
        row = registry.get(task_id)
        adapter = FakeAdapter(SessionLifecycle.RUNNING)

        hang_notifications: List[Any] = []

        with (
            patch(
                "session_orchestration.config.load_session_orchestration_config",
                return_value=_low_threshold_config(),
            ),
            patch(
                "session_orchestration.feed.push_hang_notification",
                side_effect=lambda *a, **kw: hang_notifications.append(a),
            ),
        ):
            _on_hang(task_id, row, registry=registry, adapter=adapter, **_stale_guard_kwargs())

        assert hang_notifications == [], (
            "No hang when last_output_ts is within stale_seconds threshold"
        )


# ---------------------------------------------------------------------------
# Integration: full watcher.tick() verifies end-to-end call-site gating
# ---------------------------------------------------------------------------


class TestHangEndToEnd:
    """Integration: full watcher.tick() end-to-end to confirm call-site gating."""

    def test_running_static_pane_triggers_hang_hook(self, registry, db_path):
        """After N ticks of static pane, _on_hang is called (RUNNING row)."""
        task_id = "t-hang-e2e-001"
        old_ts = time.time() - 9999
        # Seed row already with hash matching the capture text so idle_ticks increments
        static_text = "stuck session output"
        _seed_row(
            registry,
            task_id=task_id,
            state="RUNNING",
            idle_ticks=2,  # one static tick keeps this above the patched threshold
            last_output_ts=old_ts,
            last_pane_hash=_pane_hash(static_text),
        )

        adapter = FakeAdapter(SessionLifecycle.RUNNING)
        capture = FakeCapture(static_text)
        watcher = _make_watcher(registry, adapter, capture)

        hang_called = []

        with (
            patch(
                "session_orchestration.config.load_session_orchestration_config",
                return_value=_low_threshold_config(),
            ),
            patch(
                "session_orchestration.watcher._on_hang",
                side_effect=lambda tid, row, **kw: hang_called.append(tid),
            ),
        ):
            watcher.tick()

        assert hang_called == [task_id], (
            "_on_hang must be called for a RUNNING session with static pane"
        )

    def test_paused_handoff_no_hang(self, registry, db_path):
        """PAUSED_HANDOFF sessions must never trigger _on_hang."""
        task_id = "t-hang-e2e-002"
        old_ts = time.time() - 9999
        _seed_row(
            registry,
            task_id=task_id,
            state="RUNNING",
            idle_ticks=100,
            last_output_ts=old_ts,
        )

        # Adapter returns PAUSED_HANDOFF
        adapter = FakeAdapter(SessionLifecycle.PAUSED_HANDOFF)
        capture = FakeCapture("prompt: > ")
        watcher = _make_watcher(registry, adapter, capture)

        hang_called = []
        with patch(
            "session_orchestration.watcher._on_hang",
            side_effect=lambda tid, row, **kw: hang_called.append(tid),
        ):
            watcher.tick()

        row = registry.get(task_id)
        assert row["state"] == "PAUSED_HANDOFF"
        assert hang_called == [], (
            "_on_hang must NOT fire for PAUSED_HANDOFF sessions"
        )
