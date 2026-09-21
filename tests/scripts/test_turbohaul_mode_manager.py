"""Tests for scripts/turbohaul_mode_manager.py (request-scoped GPU-KV/RAM-KV mode switch).

Covers the acceptance criteria in kanban task t_fb7af9a5:

- API allows entering long-context RAM-KV mode and returning to normal GPU-KV mode.
- Restore is guaranteed on success, error, and timeout paths.
- Mode state is isolated per request/route invocation (normal traffic unaffected).
- Mode transitions are logged for diagnosis.
- Fault injection verifies restore on error/timeout/switch-failure paths.

The module is deliberately dependency-free (stdlib only) so tests can load it
with importlib without touching the repo sys.path — same pattern as
tests/scripts/test_p12_offload_gate.py.
"""

import importlib.util
import logging
import threading
import time
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "turbohaul_mode_manager.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "turbohaul_mode_manager", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Recording / fault-injecting switcher seam
# ---------------------------------------------------------------------------


class RecordingSwitcher:
    """Records every physical switch call; can be faulted per target mode.

    Mirrors the injectable ``mode_switcher`` seam the manager uses to perform
    the real sidecar/config change. Tests use this instead of touching the
    live llama-server.
    """

    def __init__(self, fail_on=None):
        self.calls = []          # list of (mode, request_id)
        self.fail_on = set(fail_on or ())

    def __call__(self, mode, request_id):
        self.calls.append((mode, request_id))
        if mode in self.fail_on:
            raise RuntimeError(f"injected switch failure for mode {mode!r}")

    @property
    def entered(self):
        return [c for c in self.calls if c[0] == "ram-kv"]

    @property
    def restored(self):
        return [c for c in self.calls if c[0] == "gpu-kv"]


@pytest.fixture()
def mod():
    return load_module()


@pytest.fixture()
def logger():
    lg = logging.getLogger("test.turbohaul.mode")
    lg.handlers.clear()
    lg.setLevel(logging.DEBUG)
    lg.propagate = False
    return lg


def make_manager(mod, switcher=None, logger=None, timeout_s=None):
    kwargs = {"mode_switcher": switcher, "logger": logger}
    if timeout_s is not None:
        kwargs["timeout_s"] = timeout_s
    return mod.TurbohaulModeManager(**kwargs)


# ---------------------------------------------------------------------------
# Mode constants + idle state
# ---------------------------------------------------------------------------


def test_mode_constants_exist(mod):
    assert mod.MODE_GPU_KV == "gpu-kv"
    assert mod.MODE_RAM_KV == "ram-kv"


def test_idle_manager_is_gpu_kv_and_unaffected(mod):
    mgr = make_manager(mod)
    assert mgr.effective_mode() == mod.MODE_GPU_KV
    assert mgr.is_long_context_active() is False
    # No request ever entered — normal traffic sees no state at all.
    assert mgr.active_requests() == {}


# ---------------------------------------------------------------------------
# Switch + restore (basic lifecycle)
# ---------------------------------------------------------------------------


def test_enter_switches_to_ram_kv_and_restore_returns_gpu_kv(mod):
    sw = RecordingSwitcher()
    mgr = make_manager(mod, switcher=sw)
    rec = mgr.enter_long_context("req-1")
    assert rec.mode == mod.MODE_RAM_KV
    assert rec.request_id == "req-1"
    assert mgr.effective_mode() == mod.MODE_RAM_KV
    assert mgr.is_long_context_active() is True
    assert sw.entered == [("ram-kv", "req-1")]

    mgr.restore_normal("req-1")
    assert mgr.effective_mode() == mod.MODE_GPU_KV
    assert mgr.is_long_context_active() is False
    assert mgr.active_requests() == {}
    assert sw.restored == [("gpu-kv", "req-1")]


def test_enter_requires_request_id(mod):
    mgr = make_manager(mod)
    with pytest.raises(ValueError):
        mgr.enter_long_context("")
    with pytest.raises(ValueError):
        mgr.enter_long_context(None)


def test_restore_without_active_request_is_noop(mod):
    sw = RecordingSwitcher()
    mgr = make_manager(mod, switcher=sw)
    mgr.restore_normal("never-entered")
    assert sw.calls == []
    assert mgr.effective_mode() == mod.MODE_GPU_KV


# ---------------------------------------------------------------------------
# Guaranteed restore: success, error, timeout
# ---------------------------------------------------------------------------


def test_run_success_restores(mod):
    sw = RecordingSwitcher()
    mgr = make_manager(mod, switcher=sw)
    result = mgr.run_long_context("req-s", lambda: 42)
    assert result == 42
    assert mgr.effective_mode() == mod.MODE_GPU_KV
    assert sw.entered == [("ram-kv", "req-s")]
    assert sw.restored == [("gpu-kv", "req-s")]


def test_run_error_restores_and_propagates(mod):
    sw = RecordingSwitcher()
    mgr = make_manager(mod, switcher=sw)

    def boom():
        raise ValueError("injected body failure")

    with pytest.raises(ValueError, match="injected body failure"):
        mgr.run_long_context("req-e", boom)
    # Restore still happened.
    assert mgr.effective_mode() == mod.MODE_GPU_KV
    assert sw.restored == [("gpu-kv", "req-e")]


def test_run_timeout_restores_and_raises_mode_timeout(mod):
    sw = RecordingSwitcher()
    mgr = make_manager(mod, switcher=sw)

    def slow():
        time.sleep(1.0)
        return "too-late"

    with pytest.raises(mod.ModeTimeoutError):
        mgr.run_long_context("req-t", slow, timeout_s=0.05)
    # Restore still happened, even though the body was still running.
    assert mgr.effective_mode() == mod.MODE_GPU_KV
    assert sw.restored == [("gpu-kv", "req-t")]


def test_run_timeout_uses_manager_default_when_not_passed(mod):
    sw = RecordingSwitcher()
    mgr = make_manager(mod, switcher=sw, timeout_s=0.05)

    def slow():
        time.sleep(1.0)
        return "too-late"

    with pytest.raises(mod.ModeTimeoutError):
        mgr.run_long_context("req-t2", slow)
    assert mgr.effective_mode() == mod.MODE_GPU_KV


def test_context_manager_restores_on_body_exception(mod):
    sw = RecordingSwitcher()
    mgr = make_manager(mod, switcher=sw)
    with pytest.raises(RuntimeError):
        with mgr.long_context("req-cm"):
            raise RuntimeError("body blew up")
    assert mgr.effective_mode() == mod.MODE_GPU_KV
    assert sw.restored == [("gpu-kv", "req-cm")]


def test_context_manager_restores_on_normal_exit(mod):
    sw = RecordingSwitcher()
    mgr = make_manager(mod, switcher=sw)
    with mgr.long_context("req-cm2"):
        assert mgr.effective_mode() == mod.MODE_RAM_KV
    assert mgr.effective_mode() == mod.MODE_GPU_KV


# ---------------------------------------------------------------------------
# Repeated transitions + re-entry
# ---------------------------------------------------------------------------


def test_repeated_enter_restore_cycles(mod):
    sw = RecordingSwitcher()
    mgr = make_manager(mod, switcher=sw)
    for i in range(3):
        mgr.enter_long_context("req-cycle")
        assert mgr.effective_mode() == mod.MODE_RAM_KV
        mgr.restore_normal("req-cycle")
        assert mgr.effective_mode() == mod.MODE_GPU_KV
    assert len(sw.entered) == 3
    assert len(sw.restored) == 3


def test_same_request_reenter_is_idempotent(mod):
    sw = RecordingSwitcher()
    mgr = make_manager(mod, switcher=sw)
    mgr.enter_long_context("req-again")
    rec2 = mgr.enter_long_context("req-again")  # already active for this request
    assert rec2.mode == mod.MODE_RAM_KV
    # No double physical switch.
    assert sw.entered == [("ram-kv", "req-again")]
    # Single restore returns to GPU-KV.
    mgr.restore_normal("req-again")
    assert mgr.effective_mode() == mod.MODE_GPU_KV
    assert sw.restored == [("gpu-kv", "req-again")]


def test_run_long_context_reusable_across_calls(mod):
    sw = RecordingSwitcher()
    mgr = make_manager(mod, switcher=sw)
    for i in range(3):
        assert mgr.run_long_context(f"req-r{i}", lambda: i) == i
    assert len(sw.entered) == 3
    assert len(sw.restored) == 3


# ---------------------------------------------------------------------------
# Request isolation / concurrency edges
# ---------------------------------------------------------------------------


def test_second_request_while_busy_is_rejected(mod):
    sw = RecordingSwitcher()
    mgr = make_manager(mod, switcher=sw)
    mgr.enter_long_context("req-a")
    with pytest.raises(mod.ModeBusyError):
        mgr.enter_long_context("req-b")
    # A's mode untouched; B never switched physically.
    assert sw.entered == [("ram-kv", "req-a")]
    assert mgr.effective_mode() == mod.MODE_RAM_KV
    # After A restores, B can enter.
    mgr.restore_normal("req-a")
    mgr.enter_long_context("req-b")
    assert mgr.effective_mode() == mod.MODE_RAM_KV
    assert sw.entered[-1] == ("ram-kv", "req-b")


def test_concurrent_enter_race_closes_with_one_holder(mod):
    """Two threads racing enter_long_context must yield exactly one ok, one
    ModeBusyError, and exactly one physical ram-kv switch.

    Regression for the TOCTOU race: the single-holder slot is now reserved
    under the lock before the physical switch, so a second concurrent request
    sees the reservation and is rejected instead of double-switching (breaks
    AC-3 per-request isolation + AC-2 restore correctness, mirroring
    Turbohaul max_parallel_sidecars=1).
    """
    class SlowSwitcher(RecordingSwitcher):
        def __call__(self, mode, request_id):
            time.sleep(0.05)  # widen the switching seam so both threads race
            super().__call__(mode, request_id)

    sw = SlowSwitcher()
    mgr = make_manager(mod, switcher=sw)
    barrier = threading.Barrier(2)
    results: dict[str, str] = {}

    def _enter(rid: str) -> None:
        barrier.wait()
        try:
            mgr.enter_long_context(rid)
            results[rid] = "ok"
        except mod.ModeBusyError:
            results[rid] = "busy"
        except Exception as exc:  # pragma: no cover - unexpected
            results[rid] = f"err:{type(exc).__name__}"

    t1 = threading.Thread(target=_enter, args=("race-a",))
    t2 = threading.Thread(target=_enter, args=("race-b",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert sorted(results.values()) == ["busy", "ok"], results
    assert len(mgr.active_requests()) == 1, mgr.active_requests()
    assert len(sw.entered) == 1, sw.calls


def test_normal_request_never_touches_mode_state(mod):
    sw = RecordingSwitcher()
    mgr = make_manager(mod, switcher=sw)
    # A normal (non-long-context) request never calls the manager at all —
    # active_requests stays empty and effective mode stays GPU-KV.
    assert mgr.active_requests() == {}
    assert mgr.effective_mode() == mod.MODE_GPU_KV
    assert sw.calls == []


def test_active_requests_reports_holders(mod):
    mgr = make_manager(mod, switcher=RecordingSwitcher())
    mgr.enter_long_context("req-visible")
    assert mgr.active_requests() == {"req-visible": mod.MODE_RAM_KV}
    mgr.restore_normal("req-visible")
    assert mgr.active_requests() == {}


# ---------------------------------------------------------------------------
# Fault injection: switch + restore failures
# ---------------------------------------------------------------------------


def test_enter_switch_failure_raises_and_leaves_state_clean(mod):
    sw = RecordingSwitcher(fail_on={"ram-kv"})
    mgr = make_manager(mod, switcher=sw)
    with pytest.raises(mod.ModeSwitchError):
        mgr.enter_long_context("req-fail")
    # Nothing was recorded as active; effective mode stays GPU-KV.
    assert mgr.active_requests() == {}
    assert mgr.effective_mode() == mod.MODE_GPU_KV


def test_run_restore_failure_surfaces_loudly(mod):
    # Restore is always *attempted*; if the physical restore fails the
    # manager must surface it rather than silently leaving RAM-KV active.
    sw = RecordingSwitcher(fail_on={"gpu-kv"})
    mgr = make_manager(mod, switcher=sw)
    with pytest.raises(mod.ModeSwitchError):
        mgr.run_long_context("req-restore-fail", lambda: 1)
    # The physical restore was attempted (it failed), and the failed request
    # is no longer tracked as holding RAM-KV.
    assert ("gpu-kv", "req-restore-fail") in sw.calls
    assert mgr.active_requests() == {}


def test_restore_failure_on_error_path_keeps_original_context(mod):
    sw = RecordingSwitcher(fail_on={"gpu-kv"})
    mgr = make_manager(mod, switcher=sw)

    def boom():
        raise ValueError("body")

    with pytest.raises(mod.ModeSwitchError):
        mgr.run_long_context("req-rf-e", boom)
    assert mgr.active_requests() == {}


# ---------------------------------------------------------------------------
# Logging: transitions carry request identifiers
# ---------------------------------------------------------------------------


def test_transitions_logged_with_request_id(mod, logger):
    stream = logging.StreamHandler()
    records = []
    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)
    cap = Capture()
    logger.addHandler(cap)
    mgr = make_manager(mod, switcher=RecordingSwitcher(), logger=logger)

    mgr.enter_long_context("req-log")
    mgr.restore_normal("req-log")

    text = "\n".join(r.getMessage() for r in records)
    assert "MODE_TRANSITION" in text
    assert "req-log" in text
    assert "from_mode=gpu-kv" in text
    assert "to_mode=ram-kv" in text
    assert "outcome=entered" in text
    assert "outcome=restored" in text


def test_switch_failure_logged_with_request_id(mod, logger):
    cap = logging.Handler()
    records = []
    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)
    logger.addHandler(Capture())
    sw = RecordingSwitcher(fail_on={"ram-kv"})
    mgr = make_manager(mod, switcher=sw, logger=logger)
    with pytest.raises(mod.ModeSwitchError):
        mgr.enter_long_context("req-log-fail")
    text = "\n".join(r.getMessage() for r in records)
    assert "MODE_TRANSITION" in text
    assert "req-log-fail" in text
    assert "outcome=switch-failed" in text
