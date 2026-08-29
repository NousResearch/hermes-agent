"""Tests for agent/deadline.py — the unified deadline layer (#85125).

Covers:
* clamp_timeout normalization (None / non-positive / oversized / NaN / junk)
* resolve_timeout precedence: config.yaml ``timeouts:`` > legacy env var > default
* run_bounded_sync: completion, exception propagation, timeout + on_timeout
* run_bounded_async: completion, exception propagation, timeout + abandonment
  of cancellation-shielded tasks, on_abandon cleanup
* kill_process_tree: descendants of a session-leader child die with it (POSIX)
* backward-compat contract of tool_executor._resolve_concurrent_tool_timeout
  after its migration onto resolve_timeout
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import threading
import time

import pytest

from agent.deadline import (
    MAX_SAFE_TIMEOUT_S,
    BoundedResult,
    DeadlineExpired,
    clamp_timeout,
    kill_process_tree,
    resolve_timeout,
    run_bounded_async,
    run_bounded_sync,
)


# ---------------------------------------------------------------------------
# clamp_timeout
# ---------------------------------------------------------------------------


class TestClampTimeout:
    def test_none_stays_none(self):
        assert clamp_timeout(None) is None

    def test_zero_and_negative_mean_unbounded(self):
        assert clamp_timeout(0) is None
        assert clamp_timeout(-5) is None

    def test_normal_value_passes_through(self):
        assert clamp_timeout(420.0) == 420.0

    def test_oversized_value_clamped_to_platform_safe_max(self):
        # The #83220 class: >time_t deadlines crash Lock.acquire on macOS.
        assert clamp_timeout(10**18) == MAX_SAFE_TIMEOUT_S

    def test_clamped_value_safe_for_threading_primitives(self):
        # Regression proof for #83220: the clamped value itself must be
        # accepted by the exact primitive that used to overflow. Acquiring an
        # uncontended lock returns immediately regardless of timeout, so
        # passing the full clamped value is safe and actually exercises the
        # time_t conversion.
        big = clamp_timeout(float(10**15))
        assert big is not None
        lock = threading.Lock()
        assert lock.acquire(timeout=big)
        lock.release()

    def test_nan_and_junk_treated_as_unbounded(self):
        assert clamp_timeout(float("nan")) is None
        assert clamp_timeout("not-a-number") is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# resolve_timeout
# ---------------------------------------------------------------------------


class TestResolveTimeout:
    def test_default_wins_when_nothing_configured(self, monkeypatch):
        monkeypatch.setattr("agent.deadline._timeouts_section", lambda: {})
        monkeypatch.delenv("HERMES_TEST_DEADLINE_X", raising=False)
        assert (
            resolve_timeout("a.b", default=42.0, env_var="HERMES_TEST_DEADLINE_X")
            == 42.0
        )

    def test_env_var_beats_default(self, monkeypatch):
        monkeypatch.setattr("agent.deadline._timeouts_section", lambda: {})
        monkeypatch.setenv("HERMES_TEST_DEADLINE_X", "17.5")
        assert (
            resolve_timeout("a.b", default=42.0, env_var="HERMES_TEST_DEADLINE_X")
            == 17.5
        )

    def test_config_beats_env_var(self, monkeypatch):
        monkeypatch.setattr(
            "agent.deadline._timeouts_section", lambda: {"a": {"b": 99}}
        )
        monkeypatch.setenv("HERMES_TEST_DEADLINE_X", "17.5")
        assert (
            resolve_timeout("a.b", default=42.0, env_var="HERMES_TEST_DEADLINE_X")
            == 99.0
        )

    def test_dotted_key_walks_nested_maps(self, monkeypatch):
        monkeypatch.setattr(
            "agent.deadline._timeouts_section",
            lambda: {"tools": {"concurrent_batch": 300}},
        )
        assert resolve_timeout("tools.concurrent_batch", default=420.0) == 300.0

    def test_zero_config_value_means_unbounded(self, monkeypatch):
        monkeypatch.setattr("agent.deadline._timeouts_section", lambda: {"a": {"b": 0}})
        assert resolve_timeout("a.b", default=42.0) is None

    def test_invalid_config_value_falls_through_to_env(self, monkeypatch):
        monkeypatch.setattr(
            "agent.deadline._timeouts_section", lambda: {"a": {"b": "soon"}}
        )
        monkeypatch.setenv("HERMES_TEST_DEADLINE_X", "17.5")
        assert (
            resolve_timeout("a.b", default=42.0, env_var="HERMES_TEST_DEADLINE_X")
            == 17.5
        )

    def test_invalid_env_value_falls_through_to_default(self, monkeypatch):
        monkeypatch.setattr("agent.deadline._timeouts_section", lambda: {})
        monkeypatch.setenv("HERMES_TEST_DEADLINE_X", "banana")
        assert (
            resolve_timeout("a.b", default=42.0, env_var="HERMES_TEST_DEADLINE_X")
            == 42.0
        )

    def test_bool_config_value_rejected(self, monkeypatch):
        # YAML `true` must not silently become a 1-second deadline.
        monkeypatch.setattr(
            "agent.deadline._timeouts_section", lambda: {"a": {"b": True}}
        )
        assert resolve_timeout("a.b", default=42.0) == 42.0

    def test_nan_config_value_falls_through(self, monkeypatch):
        # NaN must fall through to the next source, not resolve as unbounded.
        monkeypatch.setattr(
            "agent.deadline._timeouts_section", lambda: {"a": {"b": float("nan")}}
        )
        assert resolve_timeout("a.b", default=42.0) == 42.0

    def test_broken_config_read_never_breaks_the_protected_path(self, monkeypatch):
        # _timeouts_section swallows config-load failures internally; prove
        # the public contract by making the underlying loader raise.
        import agent.deadline as dl

        def _boom():
            raise RuntimeError("config unreadable")

        monkeypatch.setattr("hermes_cli.config.load_config_readonly", _boom)
        assert dl._timeouts_section() == {}
        assert resolve_timeout("a.b", default=5.0) == 5.0


# ---------------------------------------------------------------------------
# run_bounded_sync
# ---------------------------------------------------------------------------


class TestRunBoundedSync:
    def test_completion_returns_value(self):
        result = run_bounded_sync(lambda: "ok", 5.0, label="t")
        assert result.timed_out is False
        assert result.value == "ok"
        assert result.raise_if_timed_out() == "ok"

    def test_unbounded_when_timeout_none(self):
        result = run_bounded_sync(lambda: 7, None, label="t")
        assert result.timed_out is False and result.value == 7

    def test_exception_propagates_unchanged(self):
        class Boom(RuntimeError):
            pass

        with pytest.raises(Boom):
            run_bounded_sync(lambda: (_ for _ in ()).throw(Boom("x")), 5.0, label="t")

    def test_timeout_abandons_worker_and_reports(self):
        release = threading.Event()

        def _wedged():
            release.wait(30)
            return "late"

        start = time.monotonic()
        result = run_bounded_sync(_wedged, 0.2, label="wedged")
        elapsed = time.monotonic() - start
        assert result.timed_out is True
        assert result.value is None
        assert elapsed < 5.0  # returned near the deadline, not after 30s
        with pytest.raises(DeadlineExpired) as exc_info:
            result.raise_if_timed_out()
        assert "wedged" in str(exc_info.value)
        release.set()

    def test_on_timeout_callback_runs(self):
        release = threading.Event()
        fired = []
        result = run_bounded_sync(
            lambda: release.wait(30),
            0.1,
            label="t",
            on_timeout=lambda: fired.append(True),
        )
        assert result.timed_out and fired == [True]
        release.set()

    def test_on_timeout_callback_failure_is_swallowed(self):
        release = threading.Event()
        result = run_bounded_sync(
            lambda: release.wait(30),
            0.1,
            label="t",
            on_timeout=lambda: (_ for _ in ()).throw(RuntimeError("cleanup boom")),
        )
        assert result.timed_out is True
        release.set()

    def test_deadline_expired_is_a_timeout_error(self):
        # Error-classification contract: our deadline must be catchable as
        # TimeoutError but distinguishable by type from transport timeouts.
        assert issubclass(DeadlineExpired, TimeoutError)


# ---------------------------------------------------------------------------
# run_bounded_async
# ---------------------------------------------------------------------------


class TestRunBoundedAsync:
    def test_completion_returns_value(self):
        async def scenario():
            async def op():
                return "ok"

            return await run_bounded_async(op(), 5.0, label="t")

        result = asyncio.run(scenario())
        assert result.timed_out is False and result.value == "ok"

    def test_unbounded_when_timeout_none(self):
        async def scenario():
            async def op():
                return 7

            return await run_bounded_async(op(), None, label="t")

        result = asyncio.run(scenario())
        assert result.timed_out is False and result.value == 7

    def test_exception_propagates_unchanged(self):
        class Boom(RuntimeError):
            pass

        async def scenario():
            async def op():
                raise Boom("x")

            await run_bounded_async(op(), 5.0, label="t")

        with pytest.raises(Boom):
            asyncio.run(scenario())

    def test_timeout_returns_promptly(self):
        async def scenario():
            async def op():
                await asyncio.sleep(30)

            start = time.monotonic()
            result = await run_bounded_async(op(), 0.2, label="slow")
            return result, time.monotonic() - start

        result, elapsed = asyncio.run(scenario())
        assert result.timed_out is True
        assert elapsed < 5.0

    def test_timeout_abandons_cancellation_shielded_task(self):
        """The family-A killer case: asyncio.wait_for cannot expire a shielded
        scope; the thread-timer deadline must return anyway."""

        async def scenario():
            hung = asyncio.Event()

            async def inner():
                await hung.wait()

            async def shielded():
                # Shield swallows the cancellation run_bounded_async issues.
                await asyncio.shield(asyncio.ensure_future(inner()))

            start = time.monotonic()
            result = await run_bounded_async(shielded(), 0.2, label="shielded")
            elapsed = time.monotonic() - start
            hung.set()  # release the orphan so the loop can drain
            await asyncio.sleep(0)
            return result, elapsed

        result, elapsed = asyncio.run(scenario())
        assert result.timed_out is True
        assert elapsed < 5.0

    def test_on_abandon_cleanup_runs_detached(self):
        async def scenario():
            cleaned = asyncio.Event()

            async def _cleanup():
                cleaned.set()

            async def op():
                await asyncio.sleep(30)

            result = await run_bounded_async(op(), 0.1, label="t", on_abandon=_cleanup)
            await asyncio.wait_for(cleaned.wait(), timeout=5.0)
            return result

        result = asyncio.run(scenario())
        assert result.timed_out is True

    def test_completed_op_never_reports_timeout(self):
        # Race guard: completion just under the deadline must report success.
        async def scenario():
            async def op():
                await asyncio.sleep(0.01)
                return "made it"

            return await run_bounded_async(op(), 5.0, label="t")

        result = asyncio.run(scenario())
        assert result.timed_out is False and result.value == "made it"

    def test_external_cancellation_cancels_inner_task(self):
        # If the CALLER cancels run_bounded_async, the inner task must not be
        # leaked running unobserved.
        async def scenario():
            started = asyncio.Event()
            inner_cancelled = asyncio.Event()

            async def op():
                started.set()
                try:
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    inner_cancelled.set()
                    raise

            outer = asyncio.ensure_future(run_bounded_async(op(), 25.0, label="t"))
            await started.wait()
            outer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await outer
            await asyncio.wait_for(inner_cancelled.wait(), timeout=5.0)
            return True

        assert asyncio.run(scenario()) is True


# ---------------------------------------------------------------------------
# kill_process_tree
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group semantics")
class TestKillProcessTree:
    def test_kills_descendants_of_session_leader(self, tmp_path):
        """A child spawned with start_new_session must die with its own child.

        This is the orphan-tree class (#71148): killing only the direct child
        leaves grandchildren running.
        """
        started = tmp_path / "grandchild_started"
        marker = tmp_path / "grandchild_alive"
        grandchild_py = tmp_path / "grandchild.py"
        grandchild_py.write_text(
            "import pathlib, time\n"
            f"pathlib.Path({str(started)!r}).write_text('x')\n"
            "time.sleep(10)\n"
            f"pathlib.Path({str(marker)!r}).write_text('x')\n"
        )
        parent_py = tmp_path / "parent.py"
        parent_py.write_text(
            "import subprocess, sys, time\n"
            f"subprocess.Popen([sys.executable, {str(grandchild_py)!r}])\n"
            "time.sleep(10)\n"
        )
        proc = subprocess.Popen(
            [sys.executable, str(parent_py)], start_new_session=True
        )
        deadline = time.monotonic() + 10
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert started.exists(), "grandchild never spawned — test harness broken"
        assert kill_process_tree(proc.pid) is True
        proc.wait(timeout=5)
        # Grandchild must be dead too: marker never appears.
        time.sleep(1.5)
        assert not marker.exists()

    def test_kills_descendant_in_its_own_session(self, tmp_path):
        """A descendant that setsid'd out of the parent's group must die too.

        killpg on the parent's group cannot reach it; the psutil descendant
        sweep must (tools/environments/base.py documents user commands doing
        exactly this).
        """
        started = tmp_path / "setsid_grandchild_started"
        marker = tmp_path / "setsid_grandchild_alive"
        grandchild_py = tmp_path / "grandchild.py"
        grandchild_py.write_text(
            "import pathlib, time\n"
            f"pathlib.Path({str(started)!r}).write_text('x')\n"
            "time.sleep(10)\n"
            f"pathlib.Path({str(marker)!r}).write_text('x')\n"
        )
        parent_py = tmp_path / "parent.py"
        parent_py.write_text(
            "import subprocess, sys, time\n"
            # grandchild leaves the parent's session/group entirely
            f"subprocess.Popen([sys.executable, {str(grandchild_py)!r}], start_new_session=True)\n"
            "time.sleep(10)\n"
        )
        proc = subprocess.Popen(
            [sys.executable, str(parent_py)], start_new_session=True
        )
        deadline = time.monotonic() + 10
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert started.exists(), "grandchild never spawned — test harness broken"
        assert kill_process_tree(proc.pid) is True
        proc.wait(timeout=5)
        time.sleep(1.5)
        assert not marker.exists()

    def test_already_dead_pid_returns_false(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
        proc.wait(timeout=10)  # reaped: PID is gone from the process table
        assert kill_process_tree(proc.pid) is False

    def test_non_group_leader_falls_back_to_single_kill(self):
        # Child in OUR process group: killpg would signal the test runner.
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            assert os.getpgid(proc.pid) != proc.pid  # not a leader
            assert kill_process_tree(proc.pid, sig=signal.SIGTERM) is True
            proc.wait(timeout=5)
        finally:
            if proc.poll() is None:
                proc.kill()


# ---------------------------------------------------------------------------
# tool_executor migration contract
# ---------------------------------------------------------------------------


class TestConcurrentToolTimeoutMigration:
    """_resolve_concurrent_tool_timeout keeps its exact legacy contract."""

    def _resolver(self):
        from agent import tool_executor

        return tool_executor._resolve_concurrent_tool_timeout

    def test_default_unchanged(self, monkeypatch):
        monkeypatch.setattr("agent.deadline._timeouts_section", lambda: {})
        monkeypatch.delenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", raising=False)
        assert self._resolver()() == 420.0

    def test_env_var_still_works(self, monkeypatch):
        monkeypatch.setattr("agent.deadline._timeouts_section", lambda: {})
        monkeypatch.setenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", "60")
        assert self._resolver()() == 60.0

    def test_env_zero_still_disables(self, monkeypatch):
        monkeypatch.setattr("agent.deadline._timeouts_section", lambda: {})
        monkeypatch.setenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", "0")
        assert self._resolver()() is None

    def test_env_invalid_still_falls_back_to_default(self, monkeypatch):
        monkeypatch.setattr("agent.deadline._timeouts_section", lambda: {})
        monkeypatch.setenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", "junk")
        assert self._resolver()() == 420.0

    def test_new_config_key_wins(self, monkeypatch):
        monkeypatch.setattr(
            "agent.deadline._timeouts_section",
            lambda: {"tools": {"concurrent_batch": 300}},
        )
        monkeypatch.setenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", "60")
        assert self._resolver()() == 300.0


class TestSequentialToolTimeoutResolver:
    """_resolve_sequential_tool_timeout: own key, inherits concurrent default."""

    def _resolver(self):
        from agent import tool_executor

        return tool_executor._resolve_sequential_tool_timeout

    def test_inherits_concurrent_default(self, monkeypatch):
        monkeypatch.setattr("agent.deadline._timeouts_section", lambda: {})
        monkeypatch.delenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", raising=False)
        assert self._resolver()() == 420.0

    def test_inherits_concurrent_env_bridge(self, monkeypatch):
        # No sequential-specific setting -> concurrent env var flows through.
        monkeypatch.setattr("agent.deadline._timeouts_section", lambda: {})
        monkeypatch.setenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", "60")
        assert self._resolver()() == 60.0

    def test_own_config_key_wins_over_concurrent(self, monkeypatch):
        monkeypatch.setattr(
            "agent.deadline._timeouts_section",
            lambda: {"tools": {"concurrent_batch": 300, "sequential_call": 90}},
        )
        assert self._resolver()() == 90.0

    def test_zero_disables_independently(self, monkeypatch):
        # Sequential bound can be disabled while the concurrent one stays on.
        monkeypatch.setattr(
            "agent.deadline._timeouts_section",
            lambda: {"tools": {"concurrent_batch": 300, "sequential_call": 0}},
        )
        assert self._resolver()() is None

    def test_concurrent_disabled_flows_through(self, monkeypatch):
        # concurrent disabled (None default) + no sequential key -> unbounded.
        monkeypatch.setattr(
            "agent.deadline._timeouts_section",
            lambda: {"tools": {"concurrent_batch": 0}},
        )
        assert self._resolver()() is None


# ---------------------------------------------------------------------------
# Phase 3a (#85125): SuspectableBackend — poisoned-state contract.
# ---------------------------------------------------------------------------


def test_suspectable_backend_name_is_not_shadowed():
    """`agent.deadline.SuspectableBackend` must resolve to the Phase 3a
    Protocol (sync `ensure_healthy(self) -> bool`), not a second, unrelated
    `class SuspectableBackend` defined later in the module. A later Phase 3b
    adopter independently redefined the same name as a concrete async class
    (`ensure_healthy(self, timeout=5.0)`), which silently shadowed the
    Protocol at module scope — nothing subclasses or imports either by name
    today, so the collision caused no runtime breakage yet, but it would
    hand the wrong (and differently-shaped) type to the next `from
    agent.deadline import SuspectableBackend` consumer.
    """
    import inspect

    from agent.deadline import SuspectableBackend

    assert getattr(SuspectableBackend, "_is_protocol", False) is True
    assert not inspect.iscoroutinefunction(SuspectableBackend.ensure_healthy)
    assert "timeout" not in inspect.signature(SuspectableBackend.ensure_healthy).parameters


class _RecordingBackend:
    """Minimal SuspectableBackend: records mark_suspect calls."""

    def __init__(self) -> None:
        self.reasons: list[str] = []

    def mark_suspect(self, reason: str) -> None:
        self.reasons.append(reason)

    def ensure_healthy(self) -> bool:
        return not self.reasons


def test_async_timeout_marks_backend_once():
    from agent.deadline import run_bounded_async

    async def never():
        await asyncio.Event().wait()

    async def drive():
        backend = _RecordingBackend()
        result = await run_bounded_async(
            never(), 0.05, label="phase3a", backend=backend
        )
        assert result.timed_out
        return backend

    backend = asyncio.run(drive())
    assert len(backend.reasons) == 1
    assert "phase3a" in backend.reasons[0]
    assert "timed out after 0.1" in backend.reasons[0]


def test_async_completion_never_marks_backend():
    from agent.deadline import run_bounded_async

    async def quick():
        return "done"

    async def drive():
        backend = _RecordingBackend()
        result = await run_bounded_async(
            quick(), 5.0, label="phase3a-ok", backend=backend
        )
        assert not result.timed_out and result.value == "done"
        return backend

    backend = asyncio.run(drive())
    assert backend.reasons == []


def test_sync_timeout_marks_backend_once():
    from agent.deadline import run_bounded_sync

    def block():
        time.sleep(10)

    backend = _RecordingBackend()
    result = run_bounded_sync(block, 0.05, label="phase3a-sync", backend=backend)
    assert result.timed_out
    assert len(backend.reasons) == 1
    assert "phase3a-sync" in backend.reasons[0]


def test_non_adopting_backend_cannot_weaken_the_bound():
    """A backend without mark_suspect still gets a real timeout result."""

    class PlainBackend:
        pass

    async def never():
        await asyncio.Event().wait()

    async def drive():
        from agent.deadline import run_bounded_async

        return await run_bounded_async(
            never(), 0.05, label="phase3a-plain", backend=PlainBackend()
        )

    result = asyncio.run(drive())
    assert result.timed_out
    assert result.label == "phase3a-plain"


def test_mark_suspect_raising_never_corrupts_the_result():
    """A broken mark_suspect must not eat the timeout or the reason."""

    class ExplodingBackend:
        def mark_suspect(self, reason: str) -> None:
            raise RuntimeError("backend is broken")

    async def never():
        await asyncio.Event().wait()

    async def drive():
        from agent.deadline import run_bounded_async

        return await run_bounded_async(
            never(), 0.05, label="phase3a-boom", backend=ExplodingBackend()
        )

    result = asyncio.run(drive())
    assert result.timed_out
    assert result.label == "phase3a-boom"


def test_sync_completion_never_marks_backend():
    from agent.deadline import run_bounded_sync

    backend = _RecordingBackend()
    result = run_bounded_sync(
        lambda: "ok", 5.0, label="phase3a-sync-ok", backend=backend
    )
    assert not result.timed_out and result.value == "ok"
    assert backend.reasons == []


def test_sync_mark_happens_before_on_timeout():
    """The review-round ordering contract: mark BEFORE owner cleanup, so a
    recycle in on_timeout never sees an unmarked backend (and a healed
    replacement never inherits a stale flag)."""
    from agent.deadline import run_bounded_sync

    backend = _RecordingBackend()
    seen_at_cleanup: list[int] = []

    def on_timeout():
        seen_at_cleanup.append(len(backend.reasons))

    result = run_bounded_sync(
        lambda: time.sleep(10),
        0.05,
        label="phase3a-order-sync",
        on_timeout=on_timeout,
        backend=backend,
    )
    assert result.timed_out
    assert seen_at_cleanup == [1]  # mark already applied when cleanup ran


def test_async_mark_happens_before_on_abandon_cleanup():
    """Pins the scheduling invariant the inline mark relies on: on_abandon
    is ensure_future'd (can't start until the next loop tick), so the
    synchronous mark always lands first. An offloaded (to_thread) mark
    would break this — this test is the guard against that 'fix'."""
    from agent.deadline import run_bounded_async

    backend = _RecordingBackend()
    seen_at_cleanup: list[int] = []
    cleaned = asyncio.Event()

    async def _cleanup():
        seen_at_cleanup.append(len(backend.reasons))
        cleaned.set()

    async def never():
        await asyncio.Event().wait()

    async def drive():
        result = await run_bounded_async(
            never(), 0.05, label="phase3a-order-async",
            on_abandon=_cleanup, backend=backend,
        )
        await asyncio.wait_for(cleaned.wait(), timeout=5.0)
        return result

    result = asyncio.run(drive())
    assert result.timed_out
    assert seen_at_cleanup == [1]  # mark already applied when cleanup started
