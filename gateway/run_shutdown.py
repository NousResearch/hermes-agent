"""Stop/drain/restart, scale-to-zero and active-work accounting methods for GatewayRunner.

Split out of ``gateway/run.py``; bound onto ``GatewayRunner`` via the MRO.
``gateway.run`` internals are imported lazily inside method bodies (import cycle),
so ``patch("gateway.run.X")`` keeps intercepting them at call time.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import shlex
import sys
import threading
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from gateway.config import Platform
from gateway.restart import (
    DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT, GATEWAY_SERVICE_RESTART_EXIT_CODE, resolve_cron_drain_budget
)
from gateway.run_common import _UNSET
from gateway.shutdown_watchdog import arm_shutdown_watchdog, resolve_shutdown_watchdog_delay

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")

# Windows has no bash/setsid chain: a tiny detached Python watcher waits for the gateway PID to
# exit (bounded), then spawns ``hermes gateway restart``.
_WINDOWS_RESTART_WATCHER = """
import os, subprocess, sys, time
from hermes_cli._subprocess_compat import windows_detach_flags_without_breakaway
pid = int(sys.argv[1])
restart_after_s = float(sys.argv[2])
cmd = sys.argv[3:]
deadline = time.monotonic() + restart_after_s

def _alive(p):
    # On Windows, os.kill(pid, 0) is NOT a no-op — it maps to
    # GenerateConsoleCtrlEvent(0, pid) (bpo-14484). Use the
    # Win32 handle-based existence check instead.
    if os.name == 'nt':
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.OpenProcess.restype = ctypes.c_void_p
        k32.WaitForSingleObject.restype = ctypes.c_uint
        k32.GetLastError.restype = ctypes.c_uint
        h = k32.OpenProcess(0x1000 | 0x100000, False, int(p))
        if not h:
            return k32.GetLastError() != 87
        try:
            return k32.WaitForSingleObject(h, 0) == 0x102
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(int(p), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False

while time.monotonic() < deadline:
    if not _alive(pid):
        break
    time.sleep(0.2)
subprocess.Popen(
    cmd,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    creationflags=windows_detach_flags_without_breakaway(),
)
""".strip()


@contextmanager
def _log_suppressed(level: int, msg: str, *args, exc_info: bool = False):
    """``suppress(Exception)`` that logs the swallowed exception on ``gateway.run``.

    Without ``exc_info`` the exception is appended as the last ``%s`` argument (``msg % (*args, exc)``);
    with it the traceback is attached instead. Best-effort seams use this everywhere a failure must be
    visible in the log but must never propagate.
    """
    try:
        yield
    except Exception as exc:
        if exc_info:
            logger.log(level, msg, *args, exc_info=(type(exc), exc, exc.__traceback__))
        else:
            logger.log(level, msg, *args, exc)


def _send_failed(result: Any) -> bool:
    """True when an adapter ``send()`` result explicitly reports failure."""
    return result is not None and getattr(result, "success", True) is False


def _send_error(result: Any) -> str:
    """Error text of a failed ``send()`` result (adapters may omit it)."""
    return getattr(result, "error", "send returned success=False")


def _notice_target_key(platform_value: str, chat_id, thread_id) -> tuple:
    """Dedup key for one notice destination: thread/topic platforms share a chat but route apart."""
    return (platform_value, str(chat_id), str(thread_id) if thread_id else None)


class GatewayShutdownMixin:
    """Stop/drain/restart, scale-to-zero and active-work accounting methods for GatewayRunner."""

    @dataclasses.dataclass
    class _StopContext:
        """State threaded through the ``_stop_*`` phases of one ``stop()`` run."""

        deferred_count: Callable[[], int]
        started_at: Optional[float] = None
        active_agents: dict = dataclasses.field(default_factory=dict)
        timed_out: bool = False
        drain_elapsed: float = 0.0

        def elapsed(self) -> float:
            return time.monotonic() - self.started_at

    # Active-work accounting
    def _active_work_count(self) -> int:
        """All agent work the gateway must expose and drain as one total."""
        return (
            self._running_agent_count()
            + self._active_cron_job_count()
            + self._active_api_run_count()
            + self._active_deferred_agent_worker_count()
        )

    @staticmethod
    def _running_cron_job_count() -> int:
        # The FULL work aggregate, not _running_agent_count(): cron jobs run on the scheduler's own thread
        # pool and API-server runs live on the adapter — both outside _running_agents (the #60432 blind
        # spot), so counting agents alone let a suspend land mid-cron-job. Fail-AWAKE accounting: the shared
        # shutdown-drain counters (_active_cron_job_count/_active_api_run_count) swallow exceptions to 0,
        # which is fine for a drain but unsafe for a suspend predicate — a transient read failure would make
        # live work look idle and reopen the mid-job freeze. Here an unreadable source counts as work
        # (sentinel 1) so the machine stays awake until the source is readable again.
        from cron.scheduler import get_running_job_ids
        return len(get_running_job_ids())

    def _active_cron_job_count(self) -> int:
        """Cron jobs currently executing — they run outside ``_running_agents``; 0 if cron can't import.

        Cron jobs run through a standalone ``AIAgent`` on the scheduler's own thread pool
        (``cron/scheduler.py::run_job``), entirely outside ``self._running_agents`` — the dict every OTHER
        active-work check on this class (``_running_agent_count``, ``_drain_active_agents``) reads. Without
        this, the shutdown drain is structurally blind to in-flight cron work: it can report
        ``active_at_start=0`` and proceed straight to killing tool subprocesses while a cron job's terminal
        command is still running (#60432). Best-effort: returns 0 if the cron module can't be imported (e.g.
        a minimal test double for this class).
        """
        try:
            return self._running_cron_job_count()
        except Exception:
            return 0

    def _api_server_hook(self, name: str, *args: Any) -> int:
        """Call the primary API-server adapter's ``name`` hook, clamped >= 0 (0 when the hook is absent).

        Only the primary API server owns the HTTP listener, so only it is a source of this work.
        """
        helper = getattr(getattr(self, "adapters", {}).get(Platform.API_SERVER), name, None)
        return max(0, int(helper(*args))) if callable(helper) else 0

    def _active_api_run_count(self) -> int:
        """API-server work that is outside ``_running_agents``."""
        try:
            return self._api_server_hook("active_agent_work_count")
        except Exception:
            return 0

    def _interrupt_api_server_runs(self, reason: str) -> int:
        """Interrupt API-server agents not in ``_running_agents`` (same set ``_active_api_run_count`` counts)."""
        try:
            return self._api_server_hook("interrupt_active_runs", reason)
        except Exception as exc:
            logger.debug("Failed interrupting api_server runs during shutdown: %s", exc)
            return 0

    def _active_deferred_agent_worker_count(self) -> int:
        """Executor workers that outlived their gateway turn (e.g. a timed-out hygiene compression)."""
        workers = getattr(self, "_deferred_agent_workers", None)
        if not isinstance(workers, dict):
            return 0
        return sum(1 for future in list(workers) if not future.done())

    def _track_deferred_agent_worker(self, future: asyncio.Future, agent: Any) -> None:
        """Expose an executor worker to drain/interrupt until it really exits."""
        workers = getattr(self, "_deferred_agent_workers", None)
        if workers is None:
            workers = self._deferred_agent_workers = {}
        workers[future] = agent

        def _discard_worker(done_future: asyncio.Future) -> None:
            workers.pop(done_future, None)
            # Workers that outlive their starting coroutine have no later waiter: consume the
            # terminal exception so asyncio emits no unhandled-future warning.
            # See #98973.
            if not done_future.cancelled():
                with suppress(Exception):
                    done_future.exception()

        future.add_done_callback(_discard_worker)

    def _interrupt_deferred_agent_workers(self, reason: str) -> int:
        """Request cancellation of detached executor-backed agent work."""
        from gateway.run import request_hard_interrupt
        workers = getattr(self, "_deferred_agent_workers", None)
        if not isinstance(workers, dict):
            return 0
        interrupted = 0
        seen: set[int] = set()
        for future, agent in list(workers.items()):
            if future.done() or agent is None or id(agent) in seen:
                continue
            seen.add(id(agent))
            try:
                request_hard_interrupt(agent, reason)
                interrupted += 1
            except Exception as exc:
                logger.debug("Failed interrupting deferred agent worker during shutdown: %s", exc)
        return interrupted

    # Scale-to-zero idle detection / dormant-quiesce
    def _scale_to_zero_has_live_background_work(self) -> bool:
        """Live background work (delegations, processes, pending watchers) that must block a suspend.

        PERMANENT supervised watchers (_hermes_supervised_watcher, incl. the scale-to-zero watcher
        itself) are excluded, else this would be True forever and the gateway could never go dormant.
        """
        if any(
            not t.done() and not getattr(t, "_hermes_supervised_watcher", False)
            for t in self._background_tasks
        ):
            return True
        def _delegations_active() -> bool:
            from tools.async_delegation import active_count
            return active_count() > 0

        def _processes_active() -> bool:
            from tools.process_registry import process_registry
            return bool(process_registry.has_any_active() or process_registry.pending_watchers)

        for label, probe in (("async-delegation", _delegations_active), ("bg-work", _processes_active)):
            with _log_suppressed(logging.DEBUG, f"scale-to-zero {label} check failed", exc_info=True):
                if probe():
                    return True
        return False

    @staticmethod
    def _gateway_cfg_section(name: str) -> Optional[dict]:
        """``gateway.<name>`` from the user config when it is a dict, else None (never raises)."""
        from gateway.run import _load_gateway_config
        try:
            user_cfg = _load_gateway_config()
            gw = user_cfg.get("gateway") if isinstance(user_cfg, dict) else None
            section = gw.get(name) if isinstance(gw, dict) else None
        except Exception:  # noqa: BLE001
            return None
        return section if isinstance(section, dict) else None

    def _scale_to_zero_idle_timeout_seconds(self) -> float:
        from gateway.scale_to_zero import parse_idle_timeout_seconds
        stz = self._gateway_cfg_section("scale_to_zero")
        return parse_idle_timeout_seconds(stz.get("idle_timeout_minutes") if stz else None)

    def _restart_loop_guard_config(self) -> tuple:
        """``(max_restarts, window_seconds, max_gap_seconds)`` for the restart-loop breaker.

        ``max_restarts <= 0`` disables it; ``max_gap_seconds`` is the longest spacing between
        restart-interrupted boots that still counts as one loop.
        """
        from gateway import restart_loop_guard as _rlg
        rlg = self._gateway_cfg_section("restart_loop_guard") or {}

        def _int_or(key: str, default: int, positive: bool) -> int:
            value = rlg.get(key)
            if isinstance(value, int) and (value > 0 or not positive):
                return value
            return default

        return (
            _int_or("max_restarts", _rlg.DEFAULT_MAX_RESTARTS, positive=False),
            _int_or("window_seconds", _rlg.DEFAULT_WINDOW_SECONDS, positive=True),
            _int_or("max_gap_seconds", _rlg.DEFAULT_MAX_GAP_SECONDS, positive=True),
        )

    def _scale_to_zero_active_messaging_platforms(self) -> list:
        """ENABLED MESSAGING platforms for the relay-only arm gate.

        config.platforms is pre-seeded with disabled placeholders, and the api_server is force-enabled
        on every hosted container (counting it silently disarmed the feature everywhere).
        """
        if not self.config:
            return []
        non_messaging = {Platform.LOCAL, Platform.API_SERVER, Platform.WEBHOOK}
        try:
            return [
                p for p, pc in self.config.platforms.items()
                if getattr(pc, "enabled", False) and p not in non_messaging
            ]
        except Exception:  # noqa: BLE001
            return []

    @staticmethod
    def _relay_wake_url_or_none():
        from gateway.relay import relay_wake_url
        try:
            return relay_wake_url()
        except Exception:  # noqa: BLE001
            return None

    def _scale_to_zero_should_arm(self) -> bool:
        """Whether to start the idle watcher (D1/D11/§3.4(1))."""
        from gateway.scale_to_zero import messaging_is_relay_only_or_absent, scale_to_zero_enabled, should_arm
        return should_arm(
            enabled=scale_to_zero_enabled(),
            relay_only_or_absent=messaging_is_relay_only_or_absent(self._scale_to_zero_active_messaging_platforms()),
            wake_url=self._relay_wake_url_or_none(),
        )

    def _log_scale_to_zero_not_armed_reason(self) -> None:
        """One INFO line on why the idle watcher did NOT arm — only for an OPTED-IN instance."""
        from gateway.scale_to_zero import messaging_is_relay_only_or_absent, scale_to_zero_enabled
        try:
            if not scale_to_zero_enabled():
                return  # not opted in — normal, stay quiet
            active = [getattr(p, "value", p) for p in self._scale_to_zero_active_messaging_platforms()]
            logger.info(
                "scale-to-zero: NOT armed despite opt-in — relay_only_or_absent=%s (enabled platforms=%s), "
                "wake_url=%s. Need relay-only messaging + a registered wake URL.",
                messaging_is_relay_only_or_absent(active), active or "none",
                "set" if self._relay_wake_url_or_none() else "MISSING",
            )
        except Exception:  # noqa: BLE001 - diagnostics must never block startup
            logger.debug("scale-to-zero: not-armed reason logging failed", exc_info=True)

    def _scale_to_zero_is_idle(self) -> bool:
        from gateway.scale_to_zero import is_idle
        # FULL work aggregate with fail-AWAKE reads: the drain counters swallow errors to 0, which a
        # suspend predicate would read as idle, so an unreadable source counts as work here.

        def _read_or_awake(label: str, fn: Callable[[], Any], busy_sentinel: Any) -> Any:
            try:
                return fn()
            except Exception:  # noqa: BLE001 - unreadable source => assume busy
                logger.debug("scale-to-zero: %s unreadable — staying awake", label, exc_info=True)
                return busy_sentinel

        cron_count = _read_or_awake("cron work count", self._running_cron_job_count, 1)
        api_count = _read_or_awake("api work count", lambda: self._api_server_hook("active_agent_work_count"), 1)
        # An attached dashboard/desktop/TUI client (heartbeat mtime) is inbound activity — folded into
        # the inbound clock, not a conjunct, so a lingering marker cannot pin the box.
        last_inbound = self._last_inbound_at
        from gateway.scale_to_zero import dashboard_client_last_seen
        seen = _read_or_awake("dashboard heartbeat", dashboard_client_last_seen, time.time())
        if seen is not None and seen > last_inbound:
            last_inbound = seen
        return is_idle(
            active_work_count=self._running_agent_count() + cron_count + api_count,
            seconds_since_last_inbound=time.time() - last_inbound,
            idle_timeout_seconds=self._scale_to_zero_idle_timeout_seconds(),
            has_live_background_work=self._scale_to_zero_has_live_background_work(),
        )

    def _scale_to_zero_note_real_inbound(self) -> None:
        """Stamp real inbound and flip status back to running after a dormant wake.

        Internal completion/replay events deliberately do not call this (they must not keep an idle
        gateway awake).
        """
        self._last_inbound_at = time.time()
        if getattr(self, "_scale_to_zero_cooldown_until", 0.0) > 0:
            self._scale_to_zero_status("running", "scale-to-zero: status restore failed")
            self._scale_to_zero_cooldown_until = 0.0

    def _scale_to_zero_status(self, state: str, fail_msg: str) -> None:
        """Best-effort runtime status write; failures are debug-logged with ``fail_msg``."""
        try:
            self._update_runtime_status(state)
        except Exception:  # noqa: BLE001 - status is best-effort
            logger.debug(fail_msg, exc_info=True)

    def _relay_adapter_for_dormancy(self):
        """Return the connected RELAY adapter, if any (the one go_dormant targets)."""
        return self.adapters.get(Platform.RELAY)

    async def _scale_to_zero_watcher(self, interval: float = 30.0) -> None:
        """Watch for idle, drive the relay dormant, then self-suspend. On sustained idle: status
        `draining` (NOT _running=False), relay go_dormant() (socket close, NOT disconnect()), no
        mark_resume_pending (suspend preserves RAM), THEN suspend via the flaps socket — Fly autostop
        sees only INBOUND connections and would freeze mid-job. Without a flaps socket NAS brokers
        the stop through the stamped GATEWAY_RELAY_SLEEP_URL; with no lever at all the watcher
        abstains."""
        await asyncio.sleep(min(interval, 30.0))  # let startup settle
        while self._running:
            try:
                await asyncio.sleep(interval)
                if not self._running:
                    return
                if time.time() < self._scale_to_zero_cooldown_until or not self._scale_to_zero_is_idle():
                    continue
                go_dormant = getattr(self._relay_adapter_for_dormancy(), "go_dormant", None)
                if not callable(go_dormant):
                    continue
                # Quiesce only when a suspend can follow: otherwise the re-dial after the socket
                # close just clears the flip again.
                from gateway.scale_to_zero import suspend_available
                if not suspend_available():
                    if not self._scale_to_zero_no_suspend_logged:
                        self._scale_to_zero_no_suspend_logged = True
                        logger.info(
                            "scale-to-zero: idle, but this platform offers no suspend lever (no "
                            "in-machine API and no brokered sleep URL); staying connected rather "
                            "than quiescing"
                        )
                    continue
                logger.info(
                    "scale-to-zero: gateway idle for >= %.0fs — going dormant "
                    "(relay buffered, socket closed) then self-suspending",
                    self._scale_to_zero_idle_timeout_seconds(),
                )
                self._scale_to_zero_status("draining", "scale-to-zero: status mark failed")
                # Both levers: the 1s dormant re-dial can beat either suspend and clear the flip.
                # Held BEFORE go_dormant, whose close arms it.
                if not self._scale_to_zero_hold_redial(True):
                    # Without the hold the re-dial can clear the flip before the stop lands, so
                    # refuse rather than suspend unprotected.
                    logger.warning(
                        "scale-to-zero: could not hold the relay re-dial — staying awake rather "
                        "than suspending unprotected"
                    )
                    self._scale_to_zero_abandon_suspend()
                    continue
                dormant_ok = True
                try:
                    result = go_dormant()
                    if asyncio.iscoroutine(result):
                        result = await result
                    # The going_idle ack. Without it inbound is NOT buffered, so suspending would
                    # freeze a live destination: the whole bug.
                    if result is not True:
                        dormant_ok = False
                        logger.warning(
                            "scale-to-zero: connector did not ack going_idle — staying awake "
                            "rather than freezing a live destination"
                        )
                except Exception:  # noqa: BLE001 - dormancy is best-effort
                    dormant_ok = False
                    logger.debug("scale-to-zero: go_dormant failed", exc_info=True)
                # After a wake the drained inbound updates _last_inbound_at; give it a window so we
                # don't immediately re-go-dormant on the same idle reading before traffic lands.
                self._scale_to_zero_cooldown_until = time.time() + max(interval, 60.0)
                # Suspend ONLY after an ACKED quiesce (else inbound black-holes while we sleep), and
                # re-check idle — inbound may have landed during the quiesce await.
                if not dormant_ok:
                    self._scale_to_zero_abandon_suspend()
                    continue
                if not self._scale_to_zero_is_idle():
                    logger.info("scale-to-zero: inbound arrived during quiesce — skipping suspend")
                    self._scale_to_zero_abandon_suspend()
                    continue
                await self._scale_to_zero_self_suspend()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - the watcher must never crash the gateway
                logger.debug("scale-to-zero watcher iteration error", exc_info=True)

    async def _scale_to_zero_self_suspend(self) -> None:
        """Suspend this machine, in-guest where possible and via NAS otherwise (fail-awake).

        Called ONLY after a clean, acked go_dormant(), with the re-dial already held.
        """
        from gateway.scale_to_zero import (
            brokered_sleep_url, request_brokered_suspend, self_suspend_available, suspend_self
        )
        try:
            if self_suspend_available():
                accepted = await asyncio.to_thread(suspend_self)
                lever = "self-suspend"
                if accepted:
                    # flaps answers seconds BEFORE the kernel freezes, so the fence has to span
                    # that gap.
                    await self._scale_to_zero_await_freeze_gap()
                    self._scale_to_zero_hold_redial(False)
                else:
                    self._scale_to_zero_abandon_suspend()
            else:
                # No in-guest API (Azure ACA): NAS holds the credential for the stop verb and
                # brokers it for us.
                url = brokered_sleep_url()
                if not url:
                    # The watcher held on our behalf; nothing is coming to freeze the machine, so
                    # a held supervisor would just stay offline.
                    self._scale_to_zero_abandon_suspend()
                    logger.debug(
                        "scale-to-zero: no suspend lever available — dormant without platform suspend"
                    )
                    return
                # The watcher already holds the supervisor across this call.
                accepted = await asyncio.to_thread(request_brokered_suspend, url)
                lever = "brokered suspend"
                if not accepted:
                    self._scale_to_zero_abandon_suspend()
            if not accepted:
                logger.warning(
                    "scale-to-zero: %s not accepted — machine stays awake (fail-awake); will "
                    "retry on the next idle window", lever,
                )
        except Exception:  # noqa: BLE001 - suspend is best-effort, never crash
            logger.debug("scale-to-zero: self-suspend failed", exc_info=True)
            self._scale_to_zero_abandon_suspend()

    async def _scale_to_zero_await_freeze_gap(self) -> None:
        """Hold the re-dial fence across the flaps-2xx -> kernel-freeze gap.

        Sliced on the WALL clock rather than one ``asyncio.sleep`` because a Fly suspend stops
        CLOCK_MONOTONIC while CLOCK_REALTIME keeps tracking host time. Measured on a Fly machine
        (gru, 2026-09-03) across a 252.219s freeze: ``time.monotonic()`` advanced 0.501s,
        ``time.time()`` advanced 252.219s. ``asyncio.sleep`` runs on ``loop.time()`` (monotonic),
        so a single sleep would resume with its REMAINDER after the wake and delay the drain
        re-dial by exactly that much, on every wake of every Fly agent.

        On the wall clock the deadline is already past by the time we resume, so the fence costs
        nothing after a freeze while still spanning the full gap before one. That decoupling is
        what lets FLY_FREEZE_GRACE_S be sized for the slowest (largest-RAM) machine.
        """
        from gateway.scale_to_zero import FLY_FREEZE_GRACE_S, FLY_FREEZE_GRACE_TICK_S
        deadline = time.time() + FLY_FREEZE_GRACE_S
        while time.time() < deadline:
            await asyncio.sleep(FLY_FREEZE_GRACE_TICK_S)

    def _scale_to_zero_abandon_suspend(self) -> None:
        """Undo a quiesce we are not going to follow with a suspend.

        All three together: a released supervisor still advertising `draining` reads as
        mid-shutdown until the next real inbound event, and an abort that skips the cooldown
        re-runs on every tick.
        """
        self._scale_to_zero_hold_redial(False)
        # Same guard as _exit_external_drain: a real shutdown drain must win, so never resurrect
        # a stopping gateway to `running`.
        if not getattr(self, "_draining", False) and self._running:
            self._scale_to_zero_status("running", "scale-to-zero: status restore failed")
        # An abort before the cooldown is set would otherwise retry every tick.
        self._scale_to_zero_cooldown_until = max(
            self._scale_to_zero_cooldown_until, time.time() + 60.0
        )

    def _scale_to_zero_hold_redial(self, held: bool) -> bool:
        """Hold or release the relay's reconnect supervisor. Returns whether the transport actually
        took it, so the caller can refuse to suspend without the protection rather than fail open."""
        try:
            adapter = self._relay_adapter_for_dormancy()
            if adapter is None:
                return False
            method = getattr(adapter, "hold_redial" if held else "release_redial", None)
            if not callable(method):
                return False
            # Trust the adapter's answer rather than the absence of an exception: it deliberately
            # never raises, so "did not throw" proves nothing.
            return method() is True
        except Exception:  # noqa: BLE001 - never blocks the suspend it precedes
            logger.debug("scale-to-zero: redial hold toggle failed", exc_info=True)
            return False

    # External drain control: the dashboard writes/removes ``.drain_request.json`` (gateway/drain_control.py);
    # the watcher flips between accepting and refusing NEW turns WITHOUT exiting (reversible).
    def _enter_external_drain(self) -> None:
        """Begin external drain: refuse NEW turns (in-flight ones are NOT interrupted). Idempotent."""
        if self._external_drain_active:
            return
        self._external_drain_active = True
        logger.info(
            "External drain ENGAGED (.drain_request.json present) — refusing "
            "new turns; %d in-flight turn(s) will finish. Process stays up.", self._active_work_count(),
        )
        # Persist "draining" so /api/status tracks it; active_agents is read-merged, only state changes.
        self._update_runtime_status("draining")

    def _exit_external_drain(self) -> None:
        """Cancel external drain: re-accept new turns. Idempotent; never resurrects a stopping gateway."""
        if not self._external_drain_active:
            return
        self._external_drain_active = False
        if self._draining or not self._running:
            logger.info(
                "External drain marker cleared during shutdown — not reverting "
                "to running (shutdown takes precedence)."
            )
            return
        logger.info(
            "External drain RELEASED (.drain_request.json removed) — "
            "re-accepting new turns; gateway_state -> running."
        )
        self._update_runtime_status("running")

    async def _drain_control_watcher(self, interval: float = 1.0) -> None:
        """Poll ``.drain_request.json`` at 1s: present -> enter drain, absent -> exit; a stale epoch = absent."""
        from gateway.drain_control import drain_requested
        while self._running:
            try:
                # Off-thread: a synchronous marker read at 1s cadence can stall 30s+ under host I/O
                # pressure and take every platform heartbeat down.
                if await asyncio.to_thread(drain_requested):
                    self._enter_external_drain()
                    # API and cron work live outside messaging's _running_agents map; refresh the
                    # aggregate while an external caller polls this reversible drain state.
                    self._persist_active_agents()
                else:
                    self._exit_external_drain()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Drain-control watcher tick error: %s", exc, exc_info=True)
            await asyncio.sleep(interval)

    def _update_platform_runtime_status(
        self, platform: str, *, platform_state: Optional[str] = None,
        error_code: Optional[str] = None, error_message: Optional[str] = None,
        needs_attention: Optional[bool] = None, retrying_since: Any = _UNSET,
    ) -> None:
        from gateway.run import _write_runtime_status_quiet
        extra: Dict[str, Any] = {}
        if needs_attention is not None:
            extra["needs_attention"] = needs_attention
        if retrying_since is not _UNSET:
            extra["retrying_since"] = retrying_since
        _write_runtime_status_quiet(
            platform=platform, platform_state=platform_state, error_code=error_code,
            error_message=error_message, **extra,
        )

    # Per-platform circuit breaker (pause/resume): reconnect watcher + /platform pause|resume.
    def _pause_failed_platform(self, platform, *, reason: str = "") -> None:
        """Pause a queued platform (manual ``/platform pause`` only — the watcher never auto-pauses)."""
        info = getattr(self, "_failed_platforms", {}).get(platform)
        if info is None or info.get("paused"):
            return
        info["paused"] = True
        info["pause_reason"] = reason or "auto-paused after repeated failures"
        # next_retry=inf: a stale code path missing "paused" still never fires.
        info["next_retry"] = float("inf")
        self._update_platform_runtime_status(
            platform.value, platform_state="paused", error_code=None, error_message=info["pause_reason"],
        )
        logger.warning(
            "%s paused after %d consecutive failures (%s) — fix the underlying issue then run `/platform "
            "resume %s` to retry, or `hermes gateway restart` to restart the gateway.",
            platform.value, info.get("attempts", 0), info["pause_reason"], platform.value,
        )

    def _resume_paused_platform(self, platform) -> bool:
        """Unpause a platform (reset attempts, retry on the next watcher tick). True iff it was paused."""
        info = getattr(self, "_failed_platforms", {}).get(platform)
        if info is None or not info.get("paused"):
            return False
        info["paused"] = False
        info.pop("pause_reason", None)
        info["attempts"] = 0
        info["next_retry"] = time.monotonic()  # retry on next watcher tick
        self._update_platform_runtime_status(platform.value, platform_state="retrying")
        logger.info("%s resumed — retrying on next watcher tick", platform.value)
        return True

    # Drain / interrupt
    def _drain_work_counts(self) -> tuple:
        """``(agents, cron, api, deferred)`` — the four sources the drain waits on."""
        return (
            self._running_agent_count(), self._active_cron_job_count(),
            self._active_api_run_count(), self._active_deferred_agent_worker_count(),
        )

    async def _drain_active_agents(
        self, timeout: float, cron_timeout: Optional[float] = None
    ) -> tuple[Dict[str, Any], bool]:
        snapshot = self._snapshot_running_agents()
        loop = asyncio.get_running_loop()
        last_counts = self._drain_work_counts()
        last_status_at = 0.0

        def _maybe_update_status(force: bool = False) -> None:
            nonlocal last_counts, last_status_at
            now = loop.time()
            counts = self._drain_work_counts()
            if force or counts != last_counts or (now - last_status_at) >= 1.0:
                self._update_runtime_status("draining")
                last_counts, last_status_at = counts, now

        # Cron/API/deferred work lives outside ``_running_agents``; fold it in or it is killed unwarned.
        _cron0, _api0, _deferred0 = last_counts[1:]
        _maybe_update_status(force=True)
        if not self._running_agents and not (_cron0 or _api0 or _deferred0):
            return snapshot, False
        # Cron has its own deadline: a chat turn is announced+resumable; a killed cron run is a permanent failure.
        # ``timeout`` (``restart_drain_timeout``) defaults to 0 because interrupting a chat turn is
        # announced and resumable; a cron run killed mid-flight is recorded in jobs.json as a permanent
        # failure nobody is waiting on. Sharing one budget meant the default config could report
        # ``timed_out=True`` after 0.00s with a cron job in flight and kill it — the drain never even
        # entered this loop (#82161).
        started = loop.time()
        deadline = started + timeout
        cron_deadline = started + (timeout if cron_timeout is None else cron_timeout)

        def _still_draining() -> bool:
            now = loop.time()
            agents, cron, api, deferred = self._drain_work_counts()
            return bool(((agents or api or deferred) and now < deadline) or (cron and now < cron_deadline))

        # Both budgets at 0 = an expired deadline (loop unentered), so timed_out still comes from real state.
        while _still_draining():
            _maybe_update_status()
            await asyncio.sleep(0.1)
        timed_out = any(self._drain_work_counts())
        _maybe_update_status(force=True)
        return snapshot, timed_out

    def _interrupt_running_agents(self, reason: str) -> None:
        from gateway.run import _AGENT_PENDING_SENTINEL, request_hard_interrupt
        for session_key, agent in list(self._running_agents.items()):
            if agent is _AGENT_PENDING_SENTINEL:
                continue
            with _log_suppressed(logging.DEBUG, "Failed interrupting agent during shutdown: %s"):
                request_hard_interrupt(agent, reason)
                logger.debug("Interrupted running agent for session %s during shutdown", session_key)
        # API-server / desk turns are adapter-owned and never enter _running_agents, so the loop above
        # cannot see them even though _drain_active_agents() waited for them.
        for count, what in (
            (self._interrupt_api_server_runs(reason), "api_server run(s)"),
            (self._interrupt_deferred_agent_workers(reason), "deferred agent worker(s)"),
        ):
            if count:
                logger.debug("Interrupted %d %s during shutdown", count, what)

    def _shutdown_interrupt_reason(self) -> str:
        from gateway.run import _INTERRUPT_REASON_GATEWAY_RESTART, _INTERRUPT_REASON_GATEWAY_SHUTDOWN
        return _INTERRUPT_REASON_GATEWAY_RESTART if self._restart_requested else _INTERRUPT_REASON_GATEWAY_SHUTDOWN

    async def _mark_running_sessions_resume_pending(self, log_prefix: str) -> list:
        """Mark every non-pending running session resume_pending; returns the keys marked."""
        from gateway.run import _AGENT_PENDING_SENTINEL
        reason = "restart_timeout" if self._restart_requested else "shutdown_timeout"
        marked: list[str] = []
        # Pre-mark sessions as resume_pending BEFORE the drain wait. If the process is killed by the service
        # manager during the drain, the durable marker is already written so the next gateway boot can
        # recover in-flight sessions (#27856).
        for _sk, _agent in list(self._running_agents.items()):
            if _agent is _AGENT_PENDING_SENTINEL:
                continue
            with _log_suppressed(logging.DEBUG, "%s failed for %s: %s", log_prefix, _sk):
                await self.async_session_store.mark_resume_pending(_sk, reason)
                marked.append(_sk)
        return marked

    def _restart_notification_allowed(self, platform: Platform) -> bool:
        """False when the platform config sets ``gateway_restart_notification=false``."""
        platform_cfg = self.config.platforms.get(platform)
        return platform_cfg is None or bool(platform_cfg.gateway_restart_notification)

    def _notice_allowed(self, platform: Platform, what: str) -> bool:
        """``_restart_notification_allowed`` with the INFO suppression line for shutdown notices."""
        if self._restart_notification_allowed(platform):
            return True
        logger.info(
            "Shutdown notification suppressed for %s: %s has gateway_restart_notification=false", what, platform.value,
        )
        return False

    async def _notify_interrupted_cron_jobs(self, job_ids) -> int:
        """Tell the owner of each just-interrupted cron job that its run died; returns notices sent.

        The cron worker can't (its thread reaches ``_deliver_result`` after teardown closed the
        transport), so this runs post-interrupt while adapters are still connected. Best-effort.

        Its thread reaches ``_deliver_result`` asynchronously, and by then ``_bounded_adapter_teardown`` has
        closed the transport — so the notice never leaves the process, and ``_consume_interrupted_flag``
        discards the resulting ``delivery_error`` along with it. The run's only trace is a line in jobs.json
        nobody reads (#82232).
        Must therefore be called from the post-interrupt phase, while adapters are still connected — the
        same window ``_notify_active_sessions_of_shutdown`` relies on for chat sessions, which is blind to
        cron work because cron runs on the scheduler's own thread pool rather than ``self._running_agents``
        (#60432).
        """
        if not job_ids:
            return 0
        try:
            from cron.jobs import get_job
            from cron.scheduler import _resolve_delivery_targets
        except Exception as e:
            logger.debug("Cron interrupt notification unavailable: %s", e)
            return 0
        action = "restarting" if self._restart_requested else "shutting down"
        notified: set = set()
        for job_id in job_ids:
            try:
                job = get_job(job_id)
                if not job:
                    continue
                # deliver=local / unresolvable-origin jobs resolve to zero targets and stay silent (no home-
                # channel fallback). Interrupted notices are failure-category status: honor failure_deliver.
                # See #43014.
                targets = _resolve_delivery_targets(job, for_failure=True)
            except Exception as e:
                logger.debug("Cron interrupt targets unresolved for %s: %s", job_id, e)
                continue
            msg = (
                f"⚠️ Cron job '{job.get('name') or job_id}' was interrupted — "
                f"the gateway is {action} and killed the run before it "
                "finished. No result was produced for this run."
            )
            for target in targets or ():
                try:
                    platform = Platform(str(target.get("platform", "")).lower())
                except Exception:
                    continue
                adapter = self.adapters.get(platform)
                if adapter is None or not self._restart_notification_allowed(platform):
                    continue
                chat_id = str(target.get("chat_id"))
                thread_id = target.get("thread_id")
                dedup_key = (job_id, *_notice_target_key(platform.value, chat_id, thread_id))
                if dedup_key in notified:
                    continue
                with _log_suppressed(logging.DEBUG, "Cron interrupt notice to %s:%s raised: %s", platform.value, chat_id):
                    metadata = self._thread_metadata_for_target(platform, chat_id, thread_id, adapter=adapter)
                    if await self._send_notice_logged(
                        adapter, chat_id, msg, platform.value, "Cron interrupt notice to %s:%s failed: %s",
                        "Cron interrupt notice to %s:%s raised: %s", metadata=metadata,
                    ):
                        notified.add(dedup_key)
        if notified:
            logger.info("Shutdown: delivered %d interrupted-cron-job notice(s)", len(notified))
        return len(notified)

    async def _shutdown_notification_target(self, session_key: str):
        """``(source, platform_str, chat_id, thread_id)``: persisted origin > cached source > parsed key."""
        from gateway.run import _parse_session_key
        source = None
        try:
            if getattr(self, "session_store", None) is not None:
                await self.async_session_store._ensure_loaded()
                entry = self.session_store._entries.get(session_key)
                source = getattr(entry, "origin", None) if entry else None
        except Exception as e:
            logger.debug("Failed to load session origin for shutdown notification %s: %s", session_key, e)
        if source is None:
            source = self._get_cached_session_source(session_key)
        if source is not None:
            return source, source.platform.value, str(source.chat_id), source.thread_id
        _parsed = _parse_session_key(session_key)
        if not _parsed:
            return None
        return None, _parsed["platform"], _parsed["chat_id"], _parsed.get("thread_id")

    async def _send_shutdown_notice(
        self, adapter, chat_id: str, msg: str, kind: str, platform_str: str, **send_kwargs
    ) -> bool:
        """Send one shutdown notice; True when delivered. Failures are debug-logged, never raised."""
        where = "home channel " if kind == "home channel" else ""
        fail_fmt = f"Failed to send shutdown notification to {where}%s:%s: %s"
        if not await self._send_notice_logged(adapter, chat_id, msg, platform_str, fail_fmt, **send_kwargs):
            return False
        logger.info("Sent shutdown notification to %s %s:%s", kind, platform_str, chat_id)
        return True

    @staticmethod
    async def _send_notice_logged(
        adapter, chat_id: str, msg: str, platform_str: str, fail_fmt: str, raise_fmt: Optional[str] = None, **kw
    ) -> bool:
        """``adapter.send`` whose failure is debug-logged as ``fmt % (platform, chat, error)`` — ``fail_fmt``
        for success=False, ``raise_fmt`` (default ``fail_fmt``) for a raise; True only on a delivered send."""
        try:
            result = await adapter.send(chat_id, msg, **kw)
        except Exception as e:
            logger.debug(raise_fmt or fail_fmt, platform_str, chat_id, e)
            return False
        if _send_failed(result):
            logger.debug(fail_fmt, platform_str, chat_id, _send_error(result))
            return False
        return True

    async def _notify_active_sessions_of_shutdown(self) -> None:
        """Send shutdown/restart notifications to active chats and home channels.

        Called at the start of stop() while adapters are connected; send failures never block shutdown.
        """
        restart_source = self._restart_command_source if self._restart_requested else None
        msg = "⚠️ Gateway shutting down — Your current task will be interrupted."
        if self._restart_requested:
            msg = (
                "⚠️ Gateway restarting — Your current task will be interrupted. "
                "Send any message after restart and I'll try to resume where you left off."
            )
        restart_key = None
        if restart_source is not None:
            with suppress(Exception):
                restart_key = _notice_target_key(
                    restart_source.platform.value, restart_source.chat_id, restart_source.thread_id
                )
        notified: set[tuple[str, str, Optional[str]]] = set()
        for session_key in self._snapshot_running_agents():
            target = await self._shutdown_notification_target(session_key)
            if target is None:
                continue
            source, platform_str, chat_id, thread_id = target
            dedup_key = _notice_target_key(platform_str, chat_id, thread_id)
            if dedup_key in notified:
                continue
            try:
                platform = Platform(platform_str)
                adapter = self.adapters.get(platform)
                if not adapter:
                    continue
                if not self._notice_allowed(platform, "active session"):
                    continue
                reply_to_message_id = getattr(source, "message_id", None)
                if reply_to_message_id is None and restart_key == dedup_key:
                    reply_to_message_id = getattr(restart_source, "message_id", None)
                metadata = self._thread_metadata_for_target(
                    platform, chat_id, thread_id, chat_type=getattr(source, "chat_type", None),
                    reply_to_message_id=reply_to_message_id, adapter=adapter,
                )
            except Exception as e:
                logger.debug("Failed to send shutdown notification to %s:%s: %s", platform_str, chat_id, e)
                continue
            if await self._send_shutdown_notice(adapter, chat_id, msg, "active chat", platform_str, metadata=metadata):
                notified.add(dedup_key)
        if self._restart_requested and restart_source is not None:
            logger.debug("Skipping home-channel shutdown notifications for in-chat restart")
            return
        # A quiet drain (routine fleet auto-update) suppresses ONLY the home-channel broadcast; per-session
        # pings above stay. Current-epoch marker only; a failing check fails toward the louder behaviour.
        with _log_suppressed(logging.DEBUG, "drain_notification_suppressed check failed: %s"):
            from gateway.drain_control import drain_notification_suppressed
            if drain_notification_suppressed():
                logger.info(
                    "Home-channel shutdown broadcast suppressed by drain marker (suppress_notification=true)"
                )
                return
        # Snapshot adapters: adapter.send() can hit a fatal path (_handle_fatal) that pops the adapter
        # from self.adapters -> ``RuntimeError: dictionary changed size during iteration``.
        for platform, adapter in list(self.adapters.items()):
            home = self.config.get_home_channel(platform)
            if not home or not home.chat_id:
                continue
            if not self._notice_allowed(platform, "home channel"):
                continue
            dedup_key = _notice_target_key(platform.value, home.chat_id, home.thread_id)
            if dedup_key in notified:
                continue
            try:
                metadata = self._thread_metadata_for_target(platform, home.chat_id, home.thread_id, adapter=adapter)
            except Exception as e:
                logger.debug(
                    "Failed to send shutdown notification to home channel %s:%s: %s", platform.value, home.chat_id, e,
                )
                continue
            # Home channels omit ``metadata=`` when empty (adapter doubles may not accept the kwarg).
            if await self._send_shutdown_notice(
                adapter, str(home.chat_id), msg, "home channel", platform.value,
                **({"metadata": metadata} if metadata else {}),
            ):
                notified.add(dedup_key)

    # Agent finalization / resource cleanup
    @staticmethod
    def _flush_agent_transcript_at_shutdown(agent: Any) -> None:
        """Persist an in-flight transcript before teardown.

        A force-interrupted agent may never reach finalize_turn (the only mid-turn flush), so its
        tool rounds would vanish on resume. Idempotent; gracefully finished agents re-flush nothing.
        """
        with _log_suppressed(logging.DEBUG, "Shutdown transcript flush failed: %s"):
            # Persist any in-flight transcript to the SQLite session store before teardown (#13121). An
            # agent forcibly interrupted by the drain-timeout escalation may never reach
            # ``turn_finalizer.finalize_turn`` (the only place that flushes the turn to state.db) — e.g. it
            # was blocked in a tool call that did not abort within the post-interrupt grace window. Its
            # in-flight tool rounds live only in the in-memory ``_session_messages`` (refreshed per tool
            # round in ``conversation_loop`` but never written to SQLite mid-turn), so the immediate
            # pre-restart turn is silently dropped from ``load_transcript()`` on resume. Flushing here
            # closes that gap; the resume_pending / fresh-tool-tail branches in
            # ``_handle_message_with_agent`` already expect a transcript whose tail may be a pending tool
            # result.
            _flush = getattr(agent, "_flush_messages_to_session_db", None)
            _session_messages = getattr(agent, "_session_messages", None)
            if not (callable(_flush) and isinstance(_session_messages, list) and _session_messages):
                return
            # Strip empty-response retry scaffolding from the tail first (as ``_persist_session``
            # does) so a resumed turn doesn't replay synthetic recovery nudges.
            _strip = getattr(agent, "_drop_trailing_empty_response_scaffolding", None)
            if callable(_strip):
                with suppress(Exception):
                    _strip(_session_messages)
            try:
                _flush(_session_messages)
            except Exception as _flush_err:
                # Transcript could not be persisted (e.g. FTS/SQLite corruption): dump the live history
                # to a JSON recovery snapshot rather than lose it. Non-fatal.
                logger.warning(
                    "Shutdown transcript flush failed (%s); preserving %d in-memory message(s) to recovery snapshot",
                    _flush_err, len(_session_messages),
                )
                from gateway.shutdown_flush import flush_agent_history_to_file
                flush_agent_history_to_file(getattr(agent, "session_id", None), _session_messages)

    async def _finalize_shutdown_agents(self, active_agents: Dict[str, Any]) -> None:
        for agent in active_agents.values():
            self._flush_agent_transcript_at_shutdown(agent)
            # Off-loop + bounded: plugin on_session_finalize hooks can do arbitrary synchronous work
            # (e.g. a full-session trace export) — same hang class as the memory provider below.
            await self._finalize_session_off_loop(
                session_id=getattr(agent, "session_id", None), platform="gateway", reason="shutdown",
            )
            # Off-loop + bounded: a wedged memory provider here used to hang the whole shutdown so
            # SIGTERM never completed.
            await self._cleanup_agent_resources_off_loop(agent, context="shutdown finalize")

    def _should_emit_long_running_notification(
        self, session_key: Optional[str], agent: Any, executor_task: Optional[Any],
    ) -> bool:
        """Emit the heartbeat only while this task still owns the live run (not after ``/new`` rebinds).

        Guards against a stale ``running: delegate_task`` heartbeat outliving the run that started it: stop
        once the executor finishes, the agent is gone, or the session key has been rebound to a different
        live agent (e.g. the user sent ``/new`` and a fresh agent took the slot mid-run, #12029).
        """
        if agent is None or (executor_task is not None and executor_task.done()):
            return False
        if session_key:
            _hb_state = self._peek_session_state(session_key)
            if (_hb_state.turn.agent if _hb_state else None) is not agent:
                return False
        return True

    def _defer_agent_cleanup_until_future_done(self, future: asyncio.Future, agent: Any, *, context: str) -> None:
        """Clean up ``agent`` only after its executor future finishes (it may still use the agent's clients)."""

        async def _cleanup_when_done() -> None:
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError:
                # Loop shutdown can cancel this waiter while the executor still runs. Never turn
                # that cancellation into premature cleanup.
                return
            except Exception as exc:
                logger.debug(
                    "Deferred agent worker%s finished with an error: %s", f" ({context})" if context else "", exc,
                )
            await self._cleanup_agent_resources_off_loop(agent, context=context)

        self._track_deferred_agent_worker(future, agent)
        tasks = getattr(self, "_deferred_agent_cleanup_tasks", None)
        if tasks is None:
            tasks = self._deferred_agent_cleanup_tasks = set()
        self._track_task_in(tasks, asyncio.create_task(_cleanup_when_done()))

    async def _finalize_session_off_loop(self, *, session_id: Any, platform: str, reason: str, **extra: Any) -> None:
        """Run hermes_cli.lifecycle.finalize_session off-loop, bounded; on timeout the worker is left alone."""

        def _call() -> None:
            from hermes_cli.lifecycle import finalize_session
            finalize_session(session_id=session_id, platform=platform, reason=reason, **extra)

        try:
            await asyncio.wait_for(self._run_in_executor_with_context(_call), timeout=self._FINALIZE_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.warning(
                "Session finalize hooks (%s, reason=%s) exceeded %ss; proceeding without blocking the event loop "
                "(the worker thread is left to finish on its own).", session_id, reason, self._FINALIZE_TIMEOUT_S,
            )
        except Exception as finalize_exc:
            logger.debug("Session finalize hooks (%s, reason=%s) failed: %s", session_id, reason, finalize_exc)

    async def _cleanup_agent_resources_off_loop(self, agent: Any, *, context: str = "") -> None:
        """Run _cleanup_agent_resources in a worker thread, bounded; on timeout the worker is left alone."""
        if agent is None:
            return
        if context.startswith("shutdown") or context == "session expiry":
            with suppress(Exception):
                agent._end_session_on_close = False
        ctx_label = f" ({context})" if context else ""
        try:
            await asyncio.wait_for(
                self._run_in_executor_with_context(self._cleanup_agent_resources, agent),
                timeout=self._CLEANUP_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Agent resource cleanup%s exceeded %ss; proceeding without blocking the event loop (the worker "
                "thread is left to finish on its own). (#53175)", ctx_label, self._CLEANUP_TIMEOUT_S,
            )
        except Exception as cleanup_exc:
            logger.warning("Agent resource cleanup%s failed: %s (#53175)", ctx_label, cleanup_exc)

    def _cleanup_agent_resources(self, agent: Any) -> None:
        """Best-effort cleanup for temporary or cached agent instances."""
        if agent is None:
            return
        with suppress(Exception):
            if hasattr(agent, "shutdown_memory_provider"):
                # Drain queued memory writes BEFORE teardown (shutdown_all() gives the worker only ~5s, so a
                # /reset or rotation could drop them). Bounded; a failure never blocks teardown.
                # The memory manager persists per-turn sync and end-of-session extraction on a single
                # serialized background worker. shutdown_memory_provider() -> shutdown_all() only gives that
                # worker a ~5s bounded drain and abandons (cancels) anything still queued past it, so a
                # /reset — or any gateway session rotation that reaches this cleanup path — could silently
                # drop writes the session had already handed off. The next session then loads stale memory
                # (#73297). Give pending work a bounded head start through the manager's own barrier first,
                # mirroring the CLI exit path (cli.py). Best-effort: a flush failure must never block
                # teardown.
                _mm = getattr(agent, "_memory_manager", None)
                if _mm is not None and hasattr(_mm, "flush_pending"):
                    with suppress(Exception):
                        _mm.flush_pending(timeout=10)
                # Pass the real transcript so ``on_session_end`` hooks don't see the empty default.
                # ``_session_messages`` may be absent on ``object.__new__`` test stubs, hence getattr.
                # ``_session_messages`` is set on ``AIAgent`` (run_agent.py:1518) and refreshed at the end
                # of every ``run_conversation`` turn via ``_persist_session``; on an agent built through
                # ``object.__new__`` (test stubs) the attribute may be absent, so ``getattr`` with a
                # ``None`` default keeps the call signature-compatible with the pre-fix behaviour
                # (``shutdown_memory_provider(messages=None)``). See #15165.
                session_messages = getattr(agent, "_session_messages", None)
                if isinstance(session_messages, list):
                    agent.shutdown_memory_provider(session_messages)
                else:
                    agent.shutdown_memory_provider()
        # Close tool resources (sandboxes, browser daemons, background processes, httpx clients).
        with suppress(Exception):
            if hasattr(agent, "close"):
                agent.close()
        # Auxiliary async clients live in a process-global cache created from worker threads; drop
        # entries whose event loop is dead so httpx transports don't accumulate across turns.
        with suppress(Exception):
            from agent.auxiliary_client import cleanup_stale_async_clients
            cleanup_stale_async_clients()

    # Stuck-loop (restart failure) counters
    def _stuck_loop_counts_path(self) -> Path:
        from gateway.run import _hermes_home
        return _hermes_home / self._STUCK_LOOP_FILE

    @staticmethod
    def _read_json_counts(path: Path) -> Optional[dict]:
        """Parsed counter dict, or None when the file is missing/unreadable (no exists() pre-check needed)."""
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _increment_restart_failure_counts(self, active_session_keys: set) -> None:
        """Increment persisted restart-failure counters for active sessions; drop the rest (loop broken)."""
        from utils import atomic_json_write
        path = self._stuck_loop_counts_path()
        counts = self._read_json_counts(path) or {}
        with suppress(Exception):
            atomic_json_write(path, {key: counts.get(key, 0) + 1 for key in active_session_keys}, indent=None)

    def _suspend_stuck_loop_sessions(self) -> int:
        """Suspend sessions active across too many restarts (startup, AFTER suspend_recently_active())."""
        path = self._stuck_loop_counts_path()
        if not path.exists():
            return 0
        counts = self._read_json_counts(path)
        if counts is None:
            return 0
        suspended = 0
        for session_key in [k for k, v in counts.items() if v >= self._STUCK_LOOP_THRESHOLD]:
            with suppress(Exception):
                entry = self.session_store._entries.get(session_key)
                if entry and not entry.suspended:
                    entry.suspended = True
                    suspended += 1
                    logger.warning(
                        "Auto-suspended stuck session %s (active across %d consecutive restarts — likely a stuck loop)",
                        session_key, counts[session_key],
                    )
        if suspended:
            with suppress(Exception):
                self.session_store._save()
        # Clear the file — counters start fresh after suspension
        with suppress(Exception):
            path.unlink(missing_ok=True)
        return suspended

    async def _clear_restart_failure_count(self, session_key: str) -> None:
        """Clear a completed session's restart-failure counter off-loop (atomic_json_write fsyncs)."""
        from utils import atomic_json_write
        path = self._stuck_loop_counts_path()
        if not path.exists():
            return
        # The whole read/mutate/write is guarded (as on main): a corrupt counters file
        # (non-dict JSON) must never raise out of a session-completion path.
        try:
            counts = self._read_json_counts(path) or {}
            if session_key in counts:
                del counts[session_key]
                if counts:
                    await asyncio.to_thread(atomic_json_write, path, counts, indent=None)
                else:
                    path.unlink(missing_ok=True)
        except Exception:
            pass

    # Restart orchestration
    @staticmethod
    def _restart_watcher_env() -> dict:
        """Watcher env minus ``_HERMES_GATEWAY`` (else the CLI's self-restart guard refuses; gateway stays down)."""
        from gateway.control_plane import scrub_gateway_markers_for_restart_watcher
        from tools.environments.local import build_subprocess_env
        watcher_env = build_subprocess_env(scrub_secrets=False, inherit_profile_home=True)
        scrub_gateway_markers_for_restart_watcher(watcher_env)
        return watcher_env

    @staticmethod
    def _spawn_windows_restart_watcher(hermes_cmd: list, current_pid: int, restart_after_s: float) -> None:
        """Spawn the detached Windows watcher (``python -c``), retrying once without job breakaway."""
        import subprocess
        from hermes_cli._subprocess_compat import (
            windows_detach_flags_without_breakaway, windows_detach_popen_kwargs
        )
        watcher_env = GatewayShutdownMixin._restart_watcher_env()
        project_root = Path(__file__).resolve().parent.parent
        # Console python under CREATE_NO_WINDOW: nothing flashes. NOT pythonw.exe — a console-less
        # watcher makes every console-subsystem descendant allocate a visible conhost (#54220/#56747).
        # The watcher runs sys.executable (console python) under the CREATE_NO_WINDOW detach kwargs below:
        # it owns one hidden console, inherited by the `hermes gateway restart` child, so nothing flashes.
        # See #54220, #56747.
        watcher_python = sys.executable
        venv_dir = Path(watcher_env.get("VIRTUAL_ENV") or project_root / "venv")
        site_packages = venv_dir / "Lib" / "site-packages"
        if site_packages.exists():
            watcher_env["VIRTUAL_ENV"] = str(venv_dir)
            pythonpath = [str(project_root), str(site_packages)]
            if watcher_env.get("PYTHONPATH"):
                pythonpath.append(watcher_env["PYTHONPATH"])
            watcher_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath))
        watcher_argv = [
            watcher_python, "-c", _WINDOWS_RESTART_WATCHER,
            str(current_pid), str(restart_after_s), *hermes_cmd, "gateway", "restart",
        ]
        popen_kwargs = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=watcher_env)
        # Break away from the parent CLI's job object or be reaped when the CLI exits; a job without
        # BREAKAWAY_OK rejects CREATE_BREAKAWAY_FROM_JOB (OSError) — retry once without the bit.
        try:
            subprocess.Popen(watcher_argv, **popen_kwargs, **windows_detach_popen_kwargs())
        except OSError:
            try:
                subprocess.Popen(
                    watcher_argv, **popen_kwargs, creationflags=windows_detach_flags_without_breakaway(),
                )
            except OSError as exc:
                # Both spawns failed. Log only the interpreter basename and numeric errno — never
                # argv, env, watcher source, or str(exc) (may carry a full path) — and return.
                winerror = getattr(exc, "winerror", None)
                logger.warning(
                    "Detached restart watcher was not started after the "
                    "no-breakaway retry (%s; %s=%r). The gateway will not "
                    "be respawned by this restart attempt.", os.path.basename(watcher_python),
                    "winerror" if winerror is not None else "errno",
                    winerror if winerror is not None else exc.errno,
                )

    async def _launch_detached_restart_command(self) -> None:
        from gateway.run import _resolve_hermes_bin
        import shutil
        import subprocess
        hermes_cmd = _resolve_hermes_bin()
        if not hermes_cmd:
            logger.error("Could not locate hermes binary for detached /restart")
            return
        if self._detached_restart_helper_started:
            return
        self._detached_restart_helper_started = True
        current_pid = os.getpid()
        restart_after_s = max(float(getattr(self, "_restart_drain_timeout", 0.0) or 0.0) + 5.0, 5.0)
        if sys.platform == "win32":
            GatewayShutdownMixin._spawn_windows_restart_watcher(hermes_cmd, current_pid, restart_after_s)
            return
        cmd = " ".join(shlex.quote(part) for part in hermes_cmd)
        shell_cmd = (
            f"deadline=$(( $(date +%s) + {int(restart_after_s)} )); "
            f"while kill -0 {current_pid} 2>/dev/null && [ $(date +%s) -lt $deadline ]; do sleep 0.2; done; "
            f"{cmd} gateway restart"
        )
        setsid_bin = shutil.which("setsid")
        argv = [setsid_bin, "bash", "-lc", shell_cmd] if setsid_bin else ["bash", "-lc", shell_cmd]
        subprocess.Popen(
            argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=GatewayShutdownMixin._restart_watcher_env(), start_new_session=True,
        )

    def _wedged_agent_count(self) -> int:
        """Running chat agents with no activity for ``agent.gateway_timeout`` (0 when disabled).

        Cron/API work has no activity clock and pending sentinels are brand-new, so neither counts;
        an unreadable activity summary means "not wedged".
        """
        from gateway.run import _AGENT_PENDING_SENTINEL, _float_env
        timeout = _float_env("HERMES_AGENT_TIMEOUT", 1800)
        if timeout <= 0:
            return 0

        def _idle_seconds(agent: Any) -> Optional[float]:
            summary_fn = getattr(agent, "get_activity_summary", None)
            if not callable(summary_fn):
                return None
            try:
                summary = summary_fn()
                return float(summary.get("seconds_since_activity", 0.0)) if isinstance(summary, dict) else None
            except Exception:
                return None

        return sum(
            1
            for agent in list((getattr(self, "_running_agents", None) or {}).values())
            if agent is not None and agent is not _AGENT_PENDING_SENTINEL
            and (idle := _idle_seconds(agent)) is not None and idle >= timeout
        )

    def _awaitable_work_count(self) -> int:
        """Active work minus wedged turns — what the restart wait waits on."""
        return max(0, self._active_work_count() - self._wedged_agent_count())

    async def _await_active_work_before_restart(self) -> bool:
        """Wait for in-flight work before ``stop()`` so the requesting turn isn't force-interrupted.

        Wedged turns are excluded (restart is their remedy). True when drained to zero, False when the
        cap elapsed or only wedged work remains (caller proceeds to ``stop()``).
        """
        active = self._active_work_count()
        if active <= 0:
            return True
        if self._awaitable_work_count() <= 0:
            logger.warning(
                "Restart requested with %d active work unit(s), all wedged "
                "past the inactivity timeout; skipping the after-turn wait "
                "and proceeding to stop()/drain which will interrupt them", active,
            )
            return False
        timeout = float(getattr(self, "_restart_after_turn_timeout", 0.0) or 0.0)
        if timeout <= 0:
            logger.info(
                "Restart requested with %d active work unit(s); "
                "restart_after_turn_timeout=0 — entering stop()/drain immediately", active,
            )
            return False
        logger.info(
            "Restart requested with %d active work unit(s); "
            "deferring stop() until they finish (cap=%.0fs) so in-flight "
            "turns are not amputated (#77184)", active, timeout,
        )
        self._scale_to_zero_status("draining", "restart wait: status mark failed")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        last_status_at = 0.0
        while self._awaitable_work_count() > 0:
            now = loop.time()
            if now >= deadline:
                logger.warning(
                    "Restart after-turn wait timed out after %.0fs with %d "
                    "still active; proceeding to stop()/drain which may "
                    "interrupt remaining work (#77184)", timeout, self._active_work_count(),
                )
                return False
            if (now - last_status_at) >= 30.0:
                logger.info(
                    "Restart deferred: waiting on %d active work unit(s) "
                    "(%d wedged and excluded; %.0fs remaining before force drain)",
                    self._awaitable_work_count(), self._wedged_agent_count(), deadline - now,
                )
                self._scale_to_zero_status("draining", "restart wait: status mark failed")
                last_status_at = now
            await asyncio.sleep(0.1)
        if self._active_work_count() > 0:
            logger.warning(
                "Restart deferred wait: %d wedged work unit(s) remain; "
                "proceeding to stop()/drain which will interrupt them", self._active_work_count(),
            )
            return False
        logger.info("Restart deferred wait complete — active work drained; proceeding to stop()")
        return True

    def request_restart(self, *, detached: bool = False, via_service: bool = False) -> bool:
        if self._restart_task_started:
            return False
        self._restart_requested = True
        self._restart_detached = detached
        self._restart_via_service = via_service
        self._restart_task_started = True
        # Refuse new turns; keep ``_running`` True so the active turn can still deliver its final response.
        self._draining = True

        async def _run_restart() -> None:
            drained = await self._await_active_work_before_restart()
            try:
                from gateway.control_plane import RESTART_HANDOFF_FILENAME, mark_restart_handoff
                from gateway.run import _hermes_home

                mark_restart_handoff(
                    _hermes_home / RESTART_HANDOFF_FILENAME,
                    outbound_flushed=bool(drained),
                )
            except Exception:
                pass
            # Detached helper only AFTER the after-turn wait, or its drain_timeout+5 deadline fires mid-turn.
            if detached:
                with _log_suppressed(logging.ERROR, "Failed to launch detached gateway restart helper: %s"):
                    await self._launch_detached_restart_command()
            await asyncio.sleep(0.05)
            await self.stop(restart=True, detached_restart=detached, service_restart=via_service)

        # NOT in _background_tasks: _stop_impl cancels those, which would skip _shutdown_event.set() / exit 75.
        # _run_restart is a short-lived self-terminating task (calls stop() then returns). Don't add it to
        # _background_tasks — _stop_impl cancels all entries in that set, which would cancel _run_restart
        # while it's awaiting _stop_task, propagating CancelledError into _stop_impl and preventing
        # _shutdown_event.set() / _exit_code = 75. See #12875. We still hold a strong reference in
        # self._restart_task: a bare asyncio.create_task() keeps only a weak reference, so the event loop
        # may garbage-collect a still-pending task mid-flight. The cancel loop in _stop_impl explicitly
        # skips _restart_task for the same reason it skips _stop_task.
        self._restart_task = asyncio.create_task(_run_restart())
        return True

    def _start_systemd_watchdog(self) -> bool:
        """Start sd_notify only after a configured gateway is truly running."""
        if not self._running or self.config.systemd_watchdog_seconds <= 0:
            return False
        if self._systemd_watchdog is not None:
            return True
        from gateway.systemd_notify import SystemdWatchdog
        watchdog = SystemdWatchdog(config_enabled=True)
        if not watchdog.start():
            return False
        self._systemd_watchdog = watchdog
        watchdog.ready("Hermes Gateway running")
        return True

    async def _stop_systemd_watchdog(self) -> None:
        """Stop heartbeats before any potentially long shutdown drain."""
        watchdog = self._systemd_watchdog
        if watchdog is None:
            return
        self._systemd_watchdog = None
        await watchdog.stop()

    # stop() phases. Invoked as ``GatewayRunner._stop_<phase>(self, ctx)`` so shutdown-path tests
    # can drive them from bare doubles that are not GatewayRunner instances.
    @staticmethod
    def _quiet_step(label: str, fn: Callable[[], Any]) -> Any:
        """Run one best-effort teardown step; a failure is debug-logged as ``"<label>: <exc>"``."""
        try:
            return fn()
        except Exception as _e:
            logger.debug("%s: %s", label, _e)
            return None

    @staticmethod
    def _stop_kill_tool_subprocesses(phase: str) -> list:
        """Kill tool subprocesses + terminal envs + browsers; returns cron job IDs marked interrupted.

        Called twice: after a drain timeout (reclaim children before systemd SIGKILLs) and as a final
        catch-all. Best-effort; one failing subsystem cannot block the rest.
        """

        def _step(label: str, fn: Callable[[], Any]) -> Any:
            return GatewayShutdownMixin._quiet_step(f"{label} ({phase}) error", fn)

        def _count_step(fmt: str, fn: Callable[[], int]) -> None:
            n = fn()
            if n:
                logger.info(fmt, phase, n)

        def _kill_processes() -> None:
            from tools.process_registry import process_registry
            _count_step("Shutdown (%s): killed %d tool subprocess(es)", process_registry.kill_all)

        def _mark_cron_interrupted() -> list:
            # kill_all() is global: a cron job mid-dispatch lost its tool subprocess and its agent thread may
            # still emit a plausible response from truncated output — mark it interrupted, never success.
            # Any cron job still dispatched at this instant just had its tool subprocess killed above
            # (kill_all() has no per-job-ID targeting — it's a global sweep). No-op when no cron job is in
            # flight. See #60432.
            from cron.scheduler import mark_running_jobs_interrupted
            _interrupted = mark_running_jobs_interrupted(
                f"Gateway shutdown ({phase}) killed the job's tool subprocess before the run finished."
            )
            if _interrupted:
                logger.warning(
                    "Shutdown (%s): marked %d in-flight cron job(s) interrupted: %s",
                    phase, len(_interrupted), ", ".join(_interrupted),
                )
            return _interrupted

        def _interrupt_delegations() -> None:
            from tools.async_delegation import interrupt_all as _interrupt_async
            _count_step(
                "Shutdown (%s): interrupted %d background delegation(s)",
                lambda: _interrupt_async(reason=f"gateway shutdown ({phase})"),
            )

        _step("process_registry.kill_all", _kill_processes)
        _marked_cron_jobs = _step("mark_running_jobs_interrupted", _mark_cron_interrupted) or []
        _step("async interrupt_all", _interrupt_delegations)
        def _cleanup_environments() -> None:
            from tools.terminal_tool_lifecycle import cleanup_all_environments
            cleanup_all_environments()

        def _cleanup_browsers() -> None:
            from tools.browser_tool_lifecycle import cleanup_all_browsers
            cleanup_all_browsers()

        _step("cleanup_all_environments", _cleanup_environments)
        _step("cleanup_all_browsers", _cleanup_browsers)
        return _marked_cron_jobs

    async def _stop_begin_teardown(self, ctx: "GatewayShutdownMixin._StopContext") -> None:
        """Flag teardown, stop room worker/watchdog, notify sessions."""
        logger.info("Stopping gateway%s...", " for restart" if self._restart_requested else "")
        ctx.started_at = time.monotonic()
        self._running = False
        self._clear_plugin_message_injector()
        self._draining = True
        # getattr-guards: shutdown-path test doubles may lack the room worker / systemd watchdog.
        stop_room_worker = getattr(self, "_stop_hosted_room_worker", None)
        if callable(stop_room_worker):
            try:
                if not await stop_room_worker(timeout=5.0):
                    logger.warning(
                        "Group Chat worker is still settling durable work; the next gateway start will recover it"
                    )
            except Exception:
                logger.warning(
                    "Group Chat worker could not stop cleanly; the next gateway start will recover durable work",
                    exc_info=True,
                )
        stop_watchdog = getattr(self, "_stop_systemd_watchdog", None)
        if callable(stop_watchdog):
            await stop_watchdog()
        await self._cancel_secondary_profile_reconnect_tasks()
        # Notify all chats with active agents BEFORE draining — adapters are still connected here.
        await self._notify_active_sessions_of_shutdown()
        logger.info("Shutdown phase: notify_active_sessions done at +%.2fs", ctx.elapsed())

    async def _stop_drain_active_work(self, timeout: float, ctx: "GatewayShutdownMixin._StopContext") -> None:
        """Pre-mark resume_pending, drain agents/cron/API work into ``ctx``."""
        from gateway.run import GatewayRunner
        # Pre-mark resume_pending BEFORE the drain so a mid-drain SIGKILL still leaves a durable marker.
        _pre_drain_keys = await GatewayRunner._mark_running_sessions_resume_pending(
            self, "pre-drain mark_resume_pending"
        )
        _cron_at_start = self._active_cron_job_count()
        _api_at_start = self._active_api_run_count()
        _deferred_at_start = ctx.deferred_count()
        # Cron floor clamped to the watchdog leash; getattr-guard for bare shutdown-path doubles.
        _cron_drain_cfg = getattr(self, "_cron_drain_timeout", DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT)
        _cron_timeout = resolve_cron_drain_budget(
            timeout, _cron_drain_cfg, watchdog_delay=resolve_shutdown_watchdog_delay(timeout),
            elapsed=ctx.elapsed(),
        )
        if _cron_at_start and _cron_timeout > timeout:
            logger.info(
                "Shutdown drain: %d in-flight cron job(s) — waiting up to "
                "%.0fs for them (cron_drain_timeout=%.0fs, restart_drain_timeout=%.0fs)",
                _cron_at_start, _cron_timeout, _cron_drain_cfg, timeout,
            )
        _drain_started_at = time.monotonic()
        ctx.active_agents, ctx.timed_out = await self._drain_active_agents(timeout, _cron_timeout)
        ctx.drain_elapsed = time.monotonic() - _drain_started_at
        logger.info(
            "Shutdown phase: drain done at +%.2fs (drain took %.2fs, timed_out=%s, active_at_start=%d, "
            "active_now=%d, cron_at_start=%d, cron_now=%d, api_at_start=%d, api_now=%d, "
            "deferred_at_start=%d, deferred_now=%d)", ctx.elapsed(), ctx.drain_elapsed,
            ctx.timed_out, len(ctx.active_agents), self._running_agent_count(), _cron_at_start,
            self._active_cron_job_count(), _api_at_start, self._active_api_run_count(),
            _deferred_at_start, ctx.deferred_count(),
        )
        if ctx.timed_out:
            return
        # Graceful drain: clear the pre-drain resume_pending markers so sessions that finished
        # during the drain window don't carry a stale flag.
        for _sk in _pre_drain_keys:
            if _sk not in self._running_agents:
                try:
                    await self.async_session_store.clear_resume_pending(_sk)
                except Exception as _e:
                    logger.debug("clear_resume_pending after drain failed for %s: %s", _sk, _e)

    async def _stop_interrupt_remaining_work(self, ctx: "GatewayShutdownMixin._StopContext") -> None:
        """Drain timed out: mark resume_pending, interrupt, settle, kill tool subprocesses, notify cron."""
        from gateway.run import GatewayRunner
        logger.warning(
            "Gateway drain timed out after %.1fs with %d active agent(s), "
            "%d in-flight cron job(s), %d api_server run(s), and %d deferred agent worker(s); "
            "interrupting remaining work.", ctx.drain_elapsed, self._running_agent_count(),
            self._active_cron_job_count(), self._active_api_run_count(), ctx.deferred_count(),
        )
        # Mark resume_pending BEFORE interrupting so the next message auto-resumes (stuck sessions
        # still escalate via .restart_failure_counts). CURRENT _running_agents, not the drain snapshot.
        await GatewayRunner._mark_running_sessions_resume_pending(self, "mark_resume_pending")
        reason = GatewayRunner._shutdown_interrupt_reason(self)
        self._interrupt_running_agents(reason)
        interrupt_grace_timeout = GatewayRunner._post_interrupt_grace_timeout(self)
        loop = asyncio.get_running_loop()
        interrupt_deadline = loop.time() + interrupt_grace_timeout
        logger.info("Shutdown phase: allowing %.1fs for interrupted agents to unwind", interrupt_grace_timeout)

        def _work_live() -> bool:
            return bool(self._running_agents or self._active_api_run_count() or ctx.deferred_count())

        # Wait on API-server work too, or an API turn's tool subprocesses are killed before it unwinds.
        while _work_live() and loop.time() < interrupt_deadline:
            self._update_runtime_status("draining")
            await asyncio.sleep(0.1)
        # Work can materialize AFTER the one-shot interrupt (/v1/runs registers on _create_agent return;
        # pending sentinels promote later). Re-signal for a cooperative interrupt, not a bare kill.
        if _work_live():
            self._interrupt_running_agents(reason)
            logger.debug("Re-signaled interrupt for work still live at settle-window exit")
        # Kill tool subprocesses NOW: deferring past adapter/DB teardown risks the systemd cgroup SIGKILL.
        _interrupted_cron_jobs = GatewayRunner._stop_kill_tool_subprocesses("post-interrupt")
        logger.info("Shutdown phase: post-interrupt tool kill done at +%.2fs", ctx.elapsed())
        # Last window with the transport up (the cron worker's own notice arrives after teardown).
        with _log_suppressed(logging.DEBUG, "Cron interrupt notification failed: %s"):
            # The cron worker whose run we just killed will try to deliver its own "interrupted" notice, but
            # it gets there after the adapter teardown below and the message is lost (#82232).
            await self._notify_interrupted_cron_jobs(_interrupted_cron_jobs)
        logger.info("Shutdown phase: cron interrupt notices done at +%.2fs", ctx.elapsed())

    async def _stop_finalize_agents_and_adapters(self, ctx: "GatewayShutdownMixin._StopContext") -> None:
        """Detached restart launch, agent finalization, idle-cache cleanup, adapter teardown."""
        if self._restart_requested and self._restart_detached:
            with _log_suppressed(logging.ERROR, "Failed to launch detached gateway restart: %s"):
                await self._launch_detached_restart_command()
        await self._finalize_shutdown_agents(ctx.active_agents)
        # Idle cached agents too: their MemoryProviders may never have seen on_session_end().
        _cache_lock = getattr(self, "_agent_cache_lock", None)
        _cache = getattr(self, "_agent_cache", None)
        if _cache_lock is not None and _cache is not None:
            with _cache_lock:
                _idle_agents = list(_cache.values())
                _cache.clear()
            for _entry in _idle_agents:
                # Bounded + off-loop: a wedged memory provider here once made SIGTERM hang forever.
                await self._cleanup_agent_resources_off_loop(
                    _entry[0] if isinstance(_entry, tuple) else _entry, context="shutdown idle-cache"
                )
        # Settle completion flush tasks while adapters are alive so every watcher gets a retryable result.
        cancel_completion_batches = getattr(self, "_cancel_process_completion_batch_tasks", None)
        if cancel_completion_batches is not None:
            await cancel_completion_batches()
        for platform, adapter in list(self.adapters.items()):
            await self._bounded_adapter_teardown(adapter, platform)
        # Disconnect secondary-profile adapters (multiplex mode).
        _profile_adapters = getattr(self, "_profile_adapters", {})
        for _prof, _amap in list(_profile_adapters.items()):
            for platform, adapter in list(_amap.items()):
                await self._bounded_adapter_teardown(adapter, platform, profile=_prof)
            _amap.clear()
        _profile_adapters.clear()
        logger.info("Shutdown phase: all adapters disconnected at +%.2fs", ctx.elapsed())

    def _stop_release_runtime_state(self, ctx: "GatewayShutdownMixin._StopContext") -> None:
        """Cancel background tasks, flush pending messages, clear per-session state, final tool kill."""
        from gateway.run import GatewayRunner
        for _task in list(self._background_tasks):
            # _restart_task awaits _stop_task: cancelling it would tunnel into _stop_impl and skip _shutdown_event.set().
            if _task is self._stop_task or _task is self._restart_task:
                continue
            _task.cancel()
        # The restart orchestration task is awaiting _stop_task right now; cancelling it would propagate
        # CancelledError into this _stop_impl and skip _shutdown_event.set() / _exit_code = 75 (#12875). It
        # self-terminates anyway.
        self._background_tasks.clear()
        self.adapters.clear()
        for _session_key in list(self._running_agents):
            self._release_running_agent_state(_session_key)
        # Flush pending messages before clearing: under FTS5 corruption they are the only surviving copy.
        with suppress(Exception):
            from gateway.shutdown_flush import flush_pending_to_file
            flush_pending_to_file(dict(self._pending_messages), reason="shutdown")
        # The overflow FIFO tail lives in SessionState.conversation.queued_events — flush it too.
        with suppress(Exception):
            from gateway.shutdown_flush import flush_overflow_to_file
            flush_overflow_to_file(
                {_k: list(_v) for _k, _v in dict(getattr(self, "_queued_events", None) or {}).items() if _v},
                reason="shutdown",
            )
        # Live SessionState views: clear() resets one field per session (never a wholesale dict swap).
        self._running_agents.clear()
        self._running_agents_ts.clear()
        self._pending_messages.clear()
        self._pending_approvals.clear()
        for _attr in ("_active_session_leases", "_busy_ack_ts"):  # absent on bare shutdown-path doubles
            if hasattr(self, _attr):
                getattr(self, _attr).clear()
        self._shutdown_event.set()
        # Global catch-all subprocess kill (safe to repeat) for the graceful path and late respawns.
        GatewayRunner._stop_kill_tool_subprocesses("final-cleanup")
        logger.info("Shutdown phase: final-cleanup tool kill done at +%.2fs", ctx.elapsed())
        # Reap the auxiliary-client cache: clients bound to dead worker-thread loops leak httpx transports.
        def _reap_aux_clients() -> None:
            # Reap the process-global auxiliary-client cache once at the very end of teardown. Per-turn
            # cleanup runs in _cleanup_agent_resources for each active agent, but clients bound to
            # worker-thread loops that died with their ThreadPoolExecutor (notably cron ticks) only get
            # swept here. Without this, long-running gateways accumulate async httpx transports until they
            # hit EMFILE on macOS's default RLIMIT_NOFILE=256. See #14210.
            from agent.auxiliary_client import shutdown_cached_clients
            shutdown_cached_clients()

        GatewayShutdownMixin._quiet_step("shutdown_cached_clients error", _reap_aux_clients)

    def _stop_quiesce_and_close_session_dbs(self, timeout: float, ctx: "GatewayShutdownMixin._StopContext") -> None:
        """Quiesce the executor, then close SessionDB handles only if no worker is still live."""
        from gateway.run import GatewayRunner, _EXECUTOR_QUIESCE_TIMEOUT
        # Quiesce the thread pool BEFORE closing session DBs: a late executor write after
        # SessionDB.close() checkpointed the WAL reopens the handle and splits the WAL generation
        # (close-time corruption). Clamped to the remaining watchdog leash minus 1s for the close.
        # This used to run *after* the close block below, which left two holes: (a) `_executor_closing` was
        # still False during the close, so any coroutine reaching `_run_in_executor_with_context` minted a
        # brand-new pool and ran more blocking DB work against handles that had just been closed; (b)
        # cancelling `self._background_tasks` above does not stop a `run_in_executor` future that already
        # started — the task dies, the worker thread keeps writing. Either way a write lands after
        # `SessionDB.close()`, which has already checkpointed the WAL and let SQLite unlink the sidecar. The
        # late write silently reopens the handle (#94736) and mints a fresh WAL generation behind that
        # checkpoint, so teardown checkpoints the same file a second time from a connection the shutdown log
        # never accounts for — the close-time page-write damage in #101093 and the split WAL generation in
        # #101064. The wait is bounded and clamped to what is left of the shutdown watchdog leash (minus a
        # second for the close itself), so a stuck worker can never cost us the post-close cleanup window
        # (#82161).
        _exec_quiesce_budget = max(
            0.0, min(_EXECUTOR_QUIESCE_TIMEOUT, resolve_shutdown_watchdog_delay(timeout) - ctx.elapsed() - 1.0),
        )
        _exec_live = GatewayRunner._shutdown_executor(self, drain_timeout=_exec_quiesce_budget)
        if _exec_live:
            # A live worker may be mid-write (the #101093 corruption sequence): skip the close and let
            # SQLite recover from its WAL on next open (at worst a transient "database is locked").
            logger.warning(
                "Shutdown phase: %d executor worker(s) still running after a %.2fs quiesce — skipping the "
                "SessionDB close/checkpoint to avoid racing a live write (#101093); handles are left "
                "open for SQLite to recover on next open", _exec_live, _exec_quiesce_budget,
            )
            return
        logger.info("Shutdown phase: executor quiesced at +%.2fs", ctx.elapsed())
        _step = GatewayShutdownMixin._quiet_step
        # Close SQLite session DBs so --replace's new gateway does not hit 'database is locked'.
        # ``_session_db`` is an AsyncSessionDB facade — unwrap; ``session_store`` holds ``_db``.
        _self_db = getattr(self, "_session_db", None)
        _self_db = getattr(_self_db, "_db", _self_db)
        store = getattr(self, "session_store", None)
        for _db in (_self_db, getattr(store, "_db", None)):
            if _db is not None and hasattr(_db, "close"):
                _step("SessionDB close error", _db.close)
        # Multiplexed session_store caches one SessionDB per profile; sweep the secondary WAL locks too.
        _sweep = getattr(store, "close_all_db_handles", None)
        if _sweep is not None:
            _step("SessionDB handle sweep error", _sweep)
        # Same sweep for the runner's own per-profile session_search handles.
        _step("Runner SessionDB handle sweep error", lambda: GatewayRunner.close_all_session_db_handles(self))

        def _close_shared() -> None:
            # Shared SessionDB instances still held by the process-wide registry (tools, cron, mirror).
            # This is the safety net that guarantees no WAL write lock survives past gateway shutdown
            # (#90837).
            from hermes_state_registry import close_all
            closed = close_all()
            if closed:
                logger.debug("Closed %d shared SessionDB instance(s) at shutdown", closed)

        _step("Shared SessionDB close error", _close_shared)
        logger.info("Shutdown phase: SessionDB close done at +%.2fs", ctx.elapsed())

    def _stop_persist_exit_state(self, ctx: "GatewayShutdownMixin._StopContext") -> None:
        """PID/lock release, clean-shutdown marker, restart markers, terminal runtime status."""
        from gateway.run import _hermes_home, _planned_restart_notification_path, _shutdown_gateway_health_export
        from utils import atomic_json_write
        from gateway.status import remove_pid_file, release_gateway_runtime_lock
        remove_pid_file()
        release_gateway_runtime_lock()
        # Clean-shutdown marker skips suspend_recently_active() next boot; a timed-out drain left
        # half-finished sessions, so no marker — the next startup suspends them.
        if not ctx.timed_out:
            with suppress(Exception):
                (_hermes_home / ".clean_shutdown").touch()
        else:
            logger.info(
                "Skipping .clean_shutdown marker — drain timed out with "
                "interrupted agents; next startup will suspend recently active sessions."
            )
        # Stuck-loop counter: sessions active across 3 consecutive restarts are auto-suspended next boot.
        if ctx.active_agents:
            self._increment_restart_failure_counts(set(ctx.active_agents.keys()))
        if self._restart_requested and self._restart_command_source is None:
            with _log_suppressed(logging.DEBUG, "Failed to write planned restart notification marker: %s"):
                atomic_json_write(
                    _planned_restart_notification_path(),
                    {
                        "requested_at": time.time(),
                        "via_service": bool(self._restart_via_service),
                        "detached": bool(self._restart_detached),
                    },
                    indent=None,
                )
        if self._restart_requested and self._restart_via_service:
            # Exit 75 + ``RestartForceExitStatus=75``: systemd replaces us without a racing helper.
            self._exit_code = GATEWAY_SERVICE_RESTART_EXIT_CODE
            self._exit_reason = self._exit_reason or "Gateway restart requested"
        self._draining = False
        # Terminal gateway_state: "stopped", or "running" on an UNEXPECTED signal (docker restart,
        # OOM) — container_boot.py only auto-starts gateways last seen "running".
        if getattr(self, "_signal_initiated_shutdown", False) and not self._restart_requested:
            logger.info(
                "Gateway stopped by an unexpected signal — persisting "
                "gateway_state=running so container_boot auto-starts on the next boot (issue #42675)"
            )
            self._update_runtime_status("running", self._exit_reason)
        else:
            self._update_runtime_status("stopped", self._exit_reason)
        _shutdown_gateway_health_export(self)
        logger.info("Gateway stopped (total teardown %.2fs)", ctx.elapsed())

    def _shutdown_watchdog_snapshot(self, ctx: "GatewayShutdownMixin._StopContext") -> dict:
        """State dumped by the thread-based shutdown watchdog when teardown hangs."""
        return {
            "restart_requested": bool(self._restart_requested),
            "draining": bool(self._draining),
            "running": bool(self._running),
            "active_agents": self._running_agent_count(),
            "active_cron_jobs": self._active_cron_job_count(),
            "active_api_runs": self._active_api_run_count(),
            "active_deferred_agent_workers": ctx.deferred_count(),
            "restart_drain_timeout": self._restart_drain_timeout,
            "watchdog_delay_s": resolve_shutdown_watchdog_delay(self._restart_drain_timeout),
            "phase_elapsed_s": ctx.elapsed() if ctx.started_at is not None else None,
        }

    async def _stop_impl(self) -> None:
        """Run every ``_stop_*`` phase under the thread-based shutdown watchdog."""
        from gateway.run import GatewayRunner
        # Thread-based watchdog (asyncio timeouts cannot recover a frozen loop): dumps stacks and
        # os._exit past drain+grace so the service manager revives us. Skipped under pytest.
        # Arm a plain OS thread at the start of stop(); if teardown never finishes within drain+grace it
        # dumps faulthandler stacks and os._exit so KeepAlive/systemd can revive. Skip under pytest so
        # stop()-driving unit tests don't get a delayed hard-exit in the worker. See #66892.
        _watchdog_done = threading.Event()
        self._shutdown_watchdog_done = _watchdog_done
        # Shutdown-path doubles may lack the deferred-worker counter.
        ctx = GatewayShutdownMixin._StopContext(
            deferred_count=getattr(self, "_active_deferred_agent_worker_count", lambda: 0)
        )
        if not os.environ.get("PYTEST_CURRENT_TEST"):
            arm_shutdown_watchdog(
                resolve_shutdown_watchdog_delay(self._restart_drain_timeout), done_event=_watchdog_done,
                snapshot_fn=lambda: GatewayRunner._shutdown_watchdog_snapshot(self, ctx), exit_code=1,
            )
        try:
            await GatewayRunner._stop_begin_teardown(self, ctx)
            timeout = self._restart_drain_timeout
            await GatewayRunner._stop_drain_active_work(self, timeout, ctx)
            if ctx.timed_out:
                await GatewayRunner._stop_interrupt_remaining_work(self, ctx)
            await GatewayRunner._stop_finalize_agents_and_adapters(self, ctx)
            GatewayRunner._stop_release_runtime_state(self, ctx)
            GatewayRunner._stop_quiesce_and_close_session_dbs(self, timeout, ctx)
            GatewayRunner._stop_persist_exit_state(self, ctx)
        finally:
            _watchdog_done.set()

    async def stop(
        self, *, restart: bool = False, detached_restart: bool = False, service_restart: bool = False
    ) -> None:
        """Stop the gateway and disconnect all adapters."""
        from gateway.run import GatewayRunner
        # getattr-guard: shutdown-path tests build bare runners via object.__new__ that lack the
        # liveness-guard machinery.
        _stop_guards = getattr(self, "_stop_loop_liveness_guards", None)
        if callable(_stop_guards):
            _stop_guards()
        if restart:
            self._restart_requested = True
            self._restart_detached = detached_restart
            self._restart_via_service = service_restart
        if self._stop_task is not None:
            await self._stop_task
            return
        self._stop_task = asyncio.create_task(GatewayRunner._stop_impl(self))
        await self._stop_task

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()
