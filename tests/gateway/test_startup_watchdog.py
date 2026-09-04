"""Startup-liveness watchdog tests (OOF-298).

The watchdog covers the pre-event-loop window: armed at process entry
(before the gateway package imports — the implementation is the stdlib-only
top-level module ``hermes_startup_watchdog``), disarmed once the gateway's asyncio loop is confirmed
live. If neither happens within the deadline — and the process shows no CPU
progress, so slow-but-alive schema migrations are exempt — it must dump
diagnostics, record a lifecycle exit, and hard-exit with the service-restart
code so the supervisor respawns the process instead of babysitting a
live-PID zombie.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

import hermes_state_repair

import hermes_startup_watchdog as sw
from hermes_startup_watchdog import (
    SERVICE_RESTART_EXIT_CODE,
    StartupWatchdogHandle,
    arm_startup_watchdog,
    disarm_startup_watchdog,
    get_startup_watchdog_dump_path,
    kick_startup_watchdog,
    report_startup_progress,
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


@pytest.fixture(autouse=True)
def _no_cpu_progress(monkeypatch):
    """Freeze process CPU time so fire tests never see 'progress'.

    Individual tests that exercise the CPU-progress extension override this
    with their own sequence.
    """
    monkeypatch.setattr(
        StartupWatchdogHandle, "_process_cpu_seconds", staticmethod(lambda: 0.0)
    )


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


class TestContracts:
    def test_restart_code_parity_with_gateway_restart(self):
        """The stdlib-only module duplicates the exit-code constant; keep it
        in lockstep with the canonical gateway.restart definition."""
        from gateway.restart import GATEWAY_SERVICE_RESTART_EXIT_CODE

        assert SERVICE_RESTART_EXIT_CODE == GATEWAY_SERVICE_RESTART_EXIT_CODE

    def test_implementation_module_is_stdlib_only(self):
        """Import-lightness is a correctness property (arm-before-imports,
        no import-lock dependence at fire time): the implementation module
        must not import the gateway/agent/hermes_cli graphs at module level."""
        import ast
        import inspect

        source = inspect.getsource(sw)
        tree = ast.parse(source)
        forbidden_roots = {
            "gateway",
            "agent",
            "hermes_cli",
            "hermes_state",
            "hermes_constants",
            "tools",
            "plugins",
        }
        offenders = []
        for node in ast.walk(tree):
            # Only module-level and unconditional imports matter; function-
            # bodied imports (the ledger helper) are deliberate and guarded.
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                root = name.split(".")[0]
                if root in forbidden_roots and node.col_offset == 0:
                    offenders.append(name)
        assert offenders == []

    def test_all_documented_entry_points_arm_the_watchdog(self):
        """Every documented arm site must actually call arm_startup_watchdog,
        so a future entry point can't silently ship unwatched. This is a
        structural contract: the arm sites are the whole point of the
        pre-loop coverage, and they're spread across four files."""
        import ast
        import inspect
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        arm_sites = {
            # (file, reason) — each must contain an arm_startup_watchdog call
            "hermes_cli/main.py": "argv fast-path (standard `hermes gateway run`)",
            "hermes_cli/gateway.py": "run_gateway() config-bridge re-arm",
            "gateway/run.py": "gateway.run.main() backstop arm",
            "cli.py": "legacy `--gateway` entry point",
        }
        for rel, reason in arm_sites.items():
            path = repo_root / rel
            assert path.exists(), f"arm site file missing: {rel} ({reason})"
            source = path.read_text()
            tree = ast.parse(source)
            # Collect both direct calls and aliased imports (main.py uses
            # `from hermes_startup_watchdog import arm_startup_watchdog as _arm_sw`
            # then calls `_arm_sw()` to keep the fast-path import-light).
            aliases = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "hermes_startup_watchdog":
                    for alias in node.names:
                        if alias.name == "arm_startup_watchdog":
                            aliases.add(alias.asname or "arm_startup_watchdog")
            calls = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    fn = node.func
                    name = (
                        fn.id
                        if isinstance(fn, ast.Name)
                        else fn.attr
                        if isinstance(fn, ast.Attribute)
                        else None
                    )
                    if name in aliases or name == "arm_startup_watchdog":
                        calls.append(node.lineno)
            assert calls, f"{rel} has no arm_startup_watchdog call ({reason})"



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

    def test_disarm_after_deadline_expiry_wins_over_fire(self, exit_capture, monkeypatch):
        """The P2 race from review: deadline expires, but disarm lands before
        the fire transition claims the state — the disarm must win. We force
        the interleaving by blocking the watchdog thread inside the CPU probe
        (which runs after deadline expiry, before the fire transition). The
        probe is also called once at thread start for the baseline, so only
        the second call blocks."""
        in_probe = threading.Event()
        release_probe = threading.Event()
        calls = {"n": 0}

        def _blocking_probe():
            calls["n"] += 1
            if calls["n"] >= 2:
                in_probe.set()
                release_probe.wait(timeout=10)
            return 0.0

        monkeypatch.setattr(
            StartupWatchdogHandle,
            "_process_cpu_seconds",
            staticmethod(_blocking_probe),
        )
        handle = arm_startup_watchdog(timeout_s=0.1)
        assert handle is not None
        # Wait until the deadline has expired and the thread is inside the
        # probe (post-expiry, pre-fire-transition).
        assert in_probe.wait(timeout=5)
        disarm_startup_watchdog()
        release_probe.set()
        handle.join(timeout=5)
        assert not exit_capture.fired.is_set()
        assert exit_capture.codes == []


class TestKick:
    def test_kick_extends_deadline(self, exit_capture):
        handle = arm_startup_watchdog(timeout_s=0.3)
        assert handle is not None
        # Kick far enough out that the original 0.3s deadline can't fire
        # while we watch.
        kick_startup_watchdog(extra_s=60)
        time.sleep(0.6)
        assert not exit_capture.fired.is_set()
        disarm_startup_watchdog()

    def test_kick_without_arm_is_safe(self):
        kick_startup_watchdog(extra_s=30)  # must not raise

    def test_kick_with_garbage_extra_is_safe(self):
        arm_startup_watchdog(timeout_s=60)
        kick_startup_watchdog(extra_s="nonsense")  # type: ignore[arg-type]
        disarm_startup_watchdog()


class TestCpuProgressExtension:
    def test_cpu_progress_extends_instead_of_firing(self, exit_capture, monkeypatch):
        """A long schema migration burns CPU: the watchdog must extend, not
        fire (the P1 false-fire/restart-loop case from review). Cap raised
        here to observe pure extension behavior; the cap itself is covered
        by test_cpu_extensions_are_capped."""
        monkeypatch.setattr(sw, "_MAX_CPU_EXTENSIONS", 10_000)
        # Each probe call reports +10s CPU — always 'progress'.
        counter = {"cpu": 0.0}

        def _busy_probe():
            counter["cpu"] += 10.0
            return counter["cpu"]

        monkeypatch.setattr(
            StartupWatchdogHandle, "_process_cpu_seconds", staticmethod(_busy_probe)
        )
        handle = arm_startup_watchdog(timeout_s=0.1)
        assert handle is not None
        time.sleep(0.6)
        assert not exit_capture.fired.is_set()
        assert handle._extensions >= 1
        disarm_startup_watchdog()

    def test_cpu_extensions_are_capped(self, exit_capture, monkeypatch):
        """Adversarial: an unrelated daemon thread burning CPU while the
        startup thread sits parked must NOT hide the deadlock forever.
        Process-wide CPU is bounded evidence — after _MAX_CPU_EXTENSIONS
        the watchdog fires anyway (review blocker #2, false-negative arm)."""
        counter = {"cpu": 0.0}

        def _busy_probe():
            counter["cpu"] += 10.0
            return counter["cpu"]

        monkeypatch.setattr(
            StartupWatchdogHandle, "_process_cpu_seconds", staticmethod(_busy_probe)
        )
        handle = arm_startup_watchdog(timeout_s=0.1)
        assert handle is not None
        # Perpetual CPU "progress" earns exactly _MAX_CPU_EXTENSIONS
        # extensions, then fires.
        assert exit_capture.fired.wait(timeout=10)
        assert handle._extensions == sw._MAX_CPU_EXTENSIONS
        assert exit_capture.codes == [SERVICE_RESTART_EXIT_CODE]

    def test_no_cpu_progress_fires(self, exit_capture):
        # autouse fixture pins CPU time at 0.0 — no progress.
        arm_startup_watchdog(timeout_s=0.1)
        assert exit_capture.fired.wait(timeout=5)
        assert exit_capture.codes == [SERVICE_RESTART_EXIT_CODE]

    def test_probe_failure_fails_toward_firing(self, exit_capture, monkeypatch):
        """If CPU time can't be read the watchdog must still fire on a real
        deadlock rather than extending forever."""
        monkeypatch.setattr(
            StartupWatchdogHandle, "_process_cpu_seconds", staticmethod(lambda: None)
        )
        arm_startup_watchdog(timeout_s=0.1)
        assert exit_capture.fired.wait(timeout=5)


class TestProgressLease:
    """Phase-owned progress leases (review blocker #2): the authoritative
    'startup is alive' signal, owned by the startup path itself — works for
    I/O-bound phases with ~zero CPU, can't be counterfeited by unrelated
    busy threads, and is clamped per call so it can't silence the watchdog
    forever without renewal."""

    def test_lease_prevents_firing_with_zero_cpu(self, exit_capture):
        """Adversarial (false-positive arm): an I/O-bound repair/backup
        phase accrues ~no CPU. Without a lease it would be killed; with one
        it must survive past the deadline."""
        handle = arm_startup_watchdog(timeout_s=0.1)
        assert handle is not None
        report_startup_progress(60.0, phase="state_db_repair")
        time.sleep(0.6)
        assert not exit_capture.fired.is_set()
        disarm_startup_watchdog()

    def test_expired_lease_no_longer_protects(self, exit_capture):
        """A lease is a bounded claim, not a permanent mute: once it expires
        (and no renewal arrives, no CPU progress) the watchdog fires."""
        handle = arm_startup_watchdog(timeout_s=0.1)
        assert handle is not None
        report_startup_progress(0.2, phase="short_phase")
        assert exit_capture.fired.wait(timeout=10)
        assert exit_capture.codes == [SERVICE_RESTART_EXIT_CODE]

    def test_lease_duration_is_clamped(self):
        handle = arm_startup_watchdog(timeout_s=60)
        assert handle is not None
        report_startup_progress(10**9, phase="greedy")
        with handle._state_lock:
            remaining = handle._lease_until - time.monotonic()
        assert remaining <= sw._MAX_LEASE_S + 1
        disarm_startup_watchdog()

    def test_lease_outranks_cpu_extension_cap(self, exit_capture, monkeypatch):
        """A current lease is honored even when the CPU-fallback cap is
        exhausted — the lease is the stronger, owned signal. Cap pinned to
        0 so CPU progress alone can never extend; only the lease can."""
        monkeypatch.setattr(sw, "_MAX_CPU_EXTENSIONS", 0)
        counter = {"cpu": 0.0}

        def _busy_probe():
            counter["cpu"] += 10.0
            return counter["cpu"]

        monkeypatch.setattr(
            StartupWatchdogHandle, "_process_cpu_seconds", staticmethod(_busy_probe)
        )
        handle = arm_startup_watchdog(timeout_s=0.1)
        assert handle is not None
        report_startup_progress(60.0, phase="post_cap_migration")
        time.sleep(0.6)
        assert not exit_capture.fired.is_set()
        disarm_startup_watchdog()

    def test_lease_without_arm_is_safe(self):
        report_startup_progress(30.0, phase="x")  # must not raise

    def test_lease_with_garbage_is_safe(self):
        arm_startup_watchdog(timeout_s=60)
        report_startup_progress("nonsense")  # type: ignore[arg-type]
        report_startup_progress(-5)
        disarm_startup_watchdog()

    def test_lease_visible_in_dump_record(self, exit_capture, tmp_path):
        arm_startup_watchdog(timeout_s=0.1)
        report_startup_progress(0.15, phase="brief_phase")
        assert exit_capture.fired.wait(timeout=10)
        dump_path = get_startup_watchdog_dump_path(tmp_path)
        deadline = time.monotonic() + 2
        while not dump_path.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        record = json.loads(dump_path.read_text(encoding="utf-8").splitlines()[0])
        assert record["lease_count"] >= 1
        assert record["last_lease_phase"] == "brief_phase"

    def test_schema_init_declares_lease(self):
        """hermes_state_schema._init_schema must hold a progress lease so
        multi-GB migrations aren't misread as deadlocks (wiring contract)."""
        import inspect

        import hermes_state_schema

        src = inspect.getsource(hermes_state_schema.SessionSchemaMixin._init_schema)
        assert "report_startup_progress" in src

    def test_repair_declares_lease(self):
        """repair_state_db_schema (I/O-bound, ~zero CPU) must hold a lease."""
        import inspect

        import hermes_state

        src = inspect.getsource(hermes_state_repair.repair_state_db_schema)
        assert "report_startup_progress" in src


class TestFire:
    def test_fires_after_deadline_with_restart_code(self, exit_capture):
        handle = arm_startup_watchdog(timeout_s=0.1)
        assert handle is not None
        assert exit_capture.fired.wait(timeout=5)
        assert exit_capture.codes == [SERVICE_RESTART_EXIT_CODE]

    def test_fire_writes_dump_record_and_stacks(self, exit_capture, tmp_path):
        arm_startup_watchdog(timeout_s=0.1)
        assert exit_capture.fired.wait(timeout=5)
        dump_path = get_startup_watchdog_dump_path(tmp_path)
        deadline = time.monotonic() + 2
        while not dump_path.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert dump_path.exists()
        content = dump_path.read_text(encoding="utf-8")
        record = json.loads(content.splitlines()[0])
        assert record["tag"] == "startup_watchdog.fired"
        assert record["exit_code"] == SERVICE_RESTART_EXIT_CODE
        assert record["timeout_s"] == pytest.approx(0.1)
        # File-based faulthandler dump follows the JSON record (stderr may be
        # absent on detached runs).
        assert "Thread" in content or "Current thread" in content

    def test_fire_marks_lifecycle_exit(self, exit_capture, monkeypatch):
        marked = {}

        def _fake_mark_exited(code, reason=None):
            marked["code"] = code
            marked["reason"] = reason

        import gateway.lifecycle_ledger as ledger

        monkeypatch.setattr(ledger, "mark_exited", _fake_mark_exited)
        arm_startup_watchdog(timeout_s=0.1)
        assert exit_capture.fired.wait(timeout=5)
        # The ledger write runs on a helper thread joined (with timeout)
        # before _exit; once fired is set the join already happened.
        assert marked == {
            "code": SERVICE_RESTART_EXIT_CODE,
            "reason": "startup_liveness_watchdog",
        }

    def test_custom_exit_code(self, exit_capture):
        arm_startup_watchdog(timeout_s=0.1, exit_code=42)
        assert exit_capture.fired.wait(timeout=5)
        assert exit_capture.codes == [42]


class TestBoundedExit:
    """The fire path's forensics (logging, dump record, faulthandler,
    ledger) may themselves wedge — the wedged main thread can hold the
    logging handler lock, the disk can be full or hung. An escort thread
    free of logging/filesystem/application locks must still reach the exit
    seam within _FIRE_EXIT_BOUND_S (review blocker #1)."""

    @pytest.fixture
    def fast_escort(self, monkeypatch):
        """Shrink the escort bound so tests don't wait 10s."""
        monkeypatch.setattr(
            StartupWatchdogHandle, "_sleep", staticmethod(lambda s: time.sleep(0.2))
        )

    def test_exits_even_when_logging_lock_is_held(
        self, exit_capture, fast_escort, monkeypatch
    ):
        """Adversarial: acquire the lock of every handler reachable from
        this module's logger before the deadline expires. logger.critical
        in _fire blocks forever — the escort must exit anyway."""
        import logging

        # Ensure there is at least one handler whose lock we can hold.
        blocker_handler = logging.StreamHandler()
        root = logging.getLogger()
        root.addHandler(blocker_handler)
        held = [h for h in root.handlers if h.lock is not None]
        assert held, "expected at least one lockable logging handler"
        for h in held:
            h.lock.acquire()
        handle = None
        try:
            handle = arm_startup_watchdog(timeout_s=0.1)
            assert handle is not None
            # Normal forensic path is stuck on logger.critical; only the
            # escort can set fired.
            assert exit_capture.fired.wait(timeout=10)
            assert SERVICE_RESTART_EXIT_CODE in exit_capture.codes
        finally:
            for h in held:
                h.lock.release()
            root.removeHandler(blocker_handler)
            # Let the unblocked fire thread finish while _exit is still the
            # capture — otherwise it could reach the REAL os._exit after
            # monkeypatch teardown and kill the test run.
            if handle is not None:
                handle.join(timeout=10)

    def test_exits_even_when_dump_write_hangs(
        self, exit_capture, fast_escort, monkeypatch
    ):
        """Adversarial: filesystem forensics hang (full/hung disk). The
        escort must exit anyway."""
        forever = threading.Event()

        def _hang(record):
            forever.wait(timeout=30)  # bounded only so the test can't leak

        monkeypatch.setattr(sw, "_write_dump_record", _hang)
        handle = arm_startup_watchdog(timeout_s=0.1)
        assert handle is not None
        assert exit_capture.fired.wait(timeout=10)
        assert SERVICE_RESTART_EXIT_CODE in exit_capture.codes
        # Unblock and drain the fire thread before monkeypatch teardown
        # (same real-os._exit hazard as above).
        forever.set()
        handle.join(timeout=10)

    def test_escort_stands_down_when_fire_completes(self, exit_capture, monkeypatch):
        """When forensics complete normally the escort must NOT double-exit:
        it observes _fire_done and returns."""
        monkeypatch.setattr(
            StartupWatchdogHandle, "_sleep", staticmethod(lambda s: time.sleep(0.5))
        )
        handle = arm_startup_watchdog(timeout_s=0.1)
        assert handle is not None
        assert exit_capture.fired.wait(timeout=5)
        # Give the escort time to wake and observe _fire_done.
        time.sleep(0.8)
        assert exit_capture.codes == [SERVICE_RESTART_EXIT_CODE]

    def test_escort_uses_no_logging_or_filesystem(self):
        """Structural guarantee: the escort body must not touch logging,
        the filesystem, or imports — only sleep, an Event check, and the
        exit seam."""
        import inspect

        src = inspect.getsource(StartupWatchdogHandle._exit_escort)
        for banned in ("logger.", "logging", "open(", "Path(", "import ", "mkdir"):
            assert banned not in src, f"escort must not use {banned!r}"


class TestDumpPath:
    def test_dump_path_under_home(self, tmp_path):
        assert get_startup_watchdog_dump_path(tmp_path) == (
            tmp_path / "logs" / "gateway-startup-watchdog.log"
        )

    def test_dump_write_failure_is_swallowed(self, monkeypatch):
        monkeypatch.setattr(
            sw, "get_startup_watchdog_dump_path", lambda home=None: Path("/dev/null/nope")
        )
        sw._write_dump_record({"tag": "x"})
