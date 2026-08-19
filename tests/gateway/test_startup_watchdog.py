"""Startup-liveness watchdog tests (OOF-298).

The watchdog covers the pre-event-loop window: armed at process entry,
disarmed once the gateway's asyncio loop is confirmed live. If neither
happens within the deadline it must dump diagnostics, record a lifecycle
exit, and hard-exit with the service-restart code so the supervisor
respawns the process instead of babysitting a live-PID zombie.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

import gateway.startup_watchdog as sw
from gateway.restart import GATEWAY_SERVICE_RESTART_EXIT_CODE
from gateway.startup_watchdog import (
    StartupWatchdogHandle,
    arm_startup_watchdog,
    disarm_startup_watchdog,
    get_startup_watchdog_dump_path,
    resolve_startup_watchdog_timeout,
    startup_watchdog_disabled,
)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Every test gets a fresh singleton and its own HERMES_HOME."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv(sw.ENV_STARTUP_WATCHDOG, raising=False)
    monkeypatch.delenv(sw.ENV_STARTUP_WATCHDOG_TIMEOUT_S, raising=False)
    sw._reset_for_tests()
    yield
    sw._reset_for_tests()


class _ExitCapture:
    """Replaces StartupWatchdogHandle._exit so _fire() cannot kill pytest."""

    def __init__(self):
        self.codes: list[int] = []
        self.fired = threading.Event()

    def __call__(self, code: int) -> None:
        self.codes.append(code)
        self.fired.set()


@pytest.fixture
def exit_capture(monkeypatch):
    capture = _ExitCapture()
    monkeypatch.setattr(StartupWatchdogHandle, "_exit", staticmethod(capture))
    return capture


class TestConfigResolution:
    def test_default_timeout(self):
        assert resolve_startup_watchdog_timeout() == sw.DEFAULT_STARTUP_WATCHDOG_TIMEOUT_S

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv(sw.ENV_STARTUP_WATCHDOG_TIMEOUT_S, "120")
        assert resolve_startup_watchdog_timeout() == 120.0

    def test_env_override_clamped_to_floor(self, monkeypatch):
        monkeypatch.setenv(sw.ENV_STARTUP_WATCHDOG_TIMEOUT_S, "5")
        assert resolve_startup_watchdog_timeout() == sw._MIN_TIMEOUT_S

    def test_garbage_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(sw.ENV_STARTUP_WATCHDOG_TIMEOUT_S, "soon")
        assert resolve_startup_watchdog_timeout() == sw.DEFAULT_STARTUP_WATCHDOG_TIMEOUT_S

    def test_nonpositive_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(sw.ENV_STARTUP_WATCHDOG_TIMEOUT_S, "-1")
        assert resolve_startup_watchdog_timeout() == sw.DEFAULT_STARTUP_WATCHDOG_TIMEOUT_S

    @pytest.mark.parametrize("raw", ["0", "false", "no", "off", "FALSE", "Off"])
    def test_disabled_values(self, monkeypatch, raw):
        monkeypatch.setenv(sw.ENV_STARTUP_WATCHDOG, raw)
        assert startup_watchdog_disabled() is True

    @pytest.mark.parametrize("raw", ["", "1", "true", "yes"])
    def test_enabled_values(self, monkeypatch, raw):
        monkeypatch.setenv(sw.ENV_STARTUP_WATCHDOG, raw)
        assert startup_watchdog_disabled() is False


class TestArmDisarm:
    def test_arm_returns_live_handle(self):
        handle = arm_startup_watchdog(timeout_s=60)
        assert handle is not None
        assert handle.is_alive()
        assert not handle.disarmed
        disarm_startup_watchdog()
        handle.join(timeout=2)
        assert not handle.is_alive()

    def test_arm_is_idempotent(self):
        first = arm_startup_watchdog(timeout_s=60)
        second = arm_startup_watchdog(timeout_s=60)
        assert first is second
        disarm_startup_watchdog()

    def test_disarm_prevents_fire(self, exit_capture):
        handle = arm_startup_watchdog(timeout_s=0.2)
        assert handle is not None
        disarm_startup_watchdog()
        handle.join(timeout=2)
        assert not exit_capture.fired.is_set()
        assert exit_capture.codes == []

    def test_disarm_without_arm_is_safe(self):
        disarm_startup_watchdog()  # must not raise

    def test_disarm_is_idempotent(self):
        arm_startup_watchdog(timeout_s=60)
        disarm_startup_watchdog()
        disarm_startup_watchdog()  # must not raise

    def test_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv(sw.ENV_STARTUP_WATCHDOG, "0")
        assert arm_startup_watchdog(timeout_s=60) is None

    def test_rearm_after_disarm_starts_fresh_thread(self):
        first = arm_startup_watchdog(timeout_s=60)
        disarm_startup_watchdog()
        first.join(timeout=2)
        second = arm_startup_watchdog(timeout_s=60)
        assert second is not None
        assert second is not first
        assert second.is_alive()
        disarm_startup_watchdog()


class TestFire:
    def test_fires_after_deadline_with_restart_code(self, exit_capture, tmp_path):
        handle = arm_startup_watchdog(timeout_s=0.1)
        assert handle is not None
        assert exit_capture.fired.wait(timeout=5)
        assert exit_capture.codes == [GATEWAY_SERVICE_RESTART_EXIT_CODE]

    def test_fire_writes_dump_record(self, exit_capture, tmp_path):
        arm_startup_watchdog(timeout_s=0.1)
        assert exit_capture.fired.wait(timeout=5)
        dump_path = get_startup_watchdog_dump_path(tmp_path)
        # The record write happens before _exit; poll briefly for the file.
        deadline = time.monotonic() + 2
        while not dump_path.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert dump_path.exists()
        record = json.loads(dump_path.read_text(encoding="utf-8").splitlines()[0])
        assert record["tag"] == "startup_watchdog.fired"
        assert record["exit_code"] == GATEWAY_SERVICE_RESTART_EXIT_CODE
        assert record["timeout_s"] == pytest.approx(0.1)

    def test_fire_marks_lifecycle_exit(self, exit_capture, tmp_path, monkeypatch):
        marked = {}

        def _fake_mark_exited(code, reason=None):
            marked["code"] = code
            marked["reason"] = reason

        import gateway.lifecycle_ledger as ledger

        monkeypatch.setattr(ledger, "mark_exited", _fake_mark_exited)
        arm_startup_watchdog(timeout_s=0.1)
        assert exit_capture.fired.wait(timeout=5)
        # mark_exited runs just before _exit on the same thread; once fired
        # is set the _exit stub has returned, so mark_exited already ran.
        assert marked == {
            "code": GATEWAY_SERVICE_RESTART_EXIT_CODE,
            "reason": "startup_liveness_watchdog",
        }

    def test_custom_exit_code(self, exit_capture):
        arm_startup_watchdog(timeout_s=0.1, exit_code=42)
        assert exit_capture.fired.wait(timeout=5)
        assert exit_capture.codes == [42]


class TestFireTimeoutClamp:
    def test_explicit_timeout_below_floor_still_used_directly(self, exit_capture):
        """arm_startup_watchdog(timeout_s=...) is a trusted caller/test seam —
        it bypasses the env floor clamp so tests stay fast. Only env-provided
        values are clamped (they come from operators)."""
        handle = arm_startup_watchdog(timeout_s=0.1)
        assert handle is not None
        assert handle.timeout_s == pytest.approx(0.1)


class TestDumpPath:
    def test_dump_path_under_home(self, tmp_path):
        assert get_startup_watchdog_dump_path(tmp_path) == (
            tmp_path / "logs" / "gateway-startup-watchdog.log"
        )

    def test_dump_write_failure_is_swallowed(self, monkeypatch):
        # Point the dump at an unwritable location; must not raise.
        monkeypatch.setattr(
            sw, "get_startup_watchdog_dump_path", lambda home=None: Path("/dev/null/nope")
        )
        sw._write_dump_record({"tag": "x"})
