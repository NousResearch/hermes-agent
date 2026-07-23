# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------------
# Portions of this file are adapted from BaiLongma
#   Upstream: https://github.com/xiaoyuanda666-ship-it/BaiLongma
#   Original: src/runtime/consciousness-loop.js
#   Copyright (c) 2026 xiaoyuanda666-ship-it — Licensed under MIT
#   License text: see LICENSES/BaiLongma-MIT.txt
# ---------------------------------------------------------------------------
"""Autonomous L2 heartbeat loop (Consciousness Loop).

Wraps a caller-supplied ``run_turn`` coroutine in a self-scheduling loop
so the agent runs *between* user messages: enqueuing due reminders,
pumping pending messages, and — when idle — firing an autonomous L2 tick
that lets the model decide for itself whether to speak, act, or stay
silent.

Design mirrors BaiLongma's ``consciousness-loop.js`` line-for-line:

    * A watchdog wraps every ``run_turn`` call so a stuck LLM turn is
      aborted and the loop keeps spinning.
    * Priority preemption: a higher-priority queue entry can interrupt
      a running turn. Concurrent user messages also preempt each other.
    * Scheduling ladder (high to low):
        1. Messages pending           → 0s
        2. 429 rate-limited           → quota-provided interval
        3. Custom cadence TTL > 0     → caller-provided ms
        4. Awakening ticks remaining  → 10s
        5. Task active                → 30s
        6. Idle                       → configured base tick interval
      Any pending reminder that fires earlier collapses the delay.
    * A cadence configured *during* a tick starts governing the *next*
      tick — its TTL is not spent by the tick that installed it.

Hermes wires this behind ``agent.heartbeat.enabled`` (default: False).
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Optional, Protocol


logger = logging.getLogger(__name__)


# ── Type aliases ────────────────────────────────────────────────────────────

RunTurn = Callable[[Any, str, Optional[Any]], Awaitable[None]]
"""Signature: ``await run_turn(input, label, msg | None)``."""

EmitEvent = Callable[[str, Mapping[str, Any]], None]
"""Signature: ``emit_event(name, payload)``."""

Scheduler = Callable[[], None]
InterruptCallback = Callable[[Any], None]


# ── Priorities (mirror BaiLongma's numeric ladder) ──────────────────────────


@dataclass(frozen=True)
class Priorities:
    """Priority levels; higher wins.

    Callers supply an instance so this module doesn't hardcode the
    numeric ladder; BaiLongma uses ``{ user: 3, background: 1, ... }``.
    """

    user: int = 3
    background: int = 1


# ── Watchdog error ──────────────────────────────────────────────────────────


class WatchdogTimeoutError(Exception):
    """Raised when ``run_turn`` exceeds ``run_turn_watchdog_s``."""


# ── Abort controller shim ───────────────────────────────────────────────────


class AbortLike(Protocol):
    """Minimal shape of an abort controller; whoever owns the current
    execution provides one. Matches JS ``AbortController.abort(reason)``.
    """

    def abort(self, reason: str = ...) -> None: ...  # pragma: no cover - Protocol


# ── Config ──────────────────────────────────────────────────────────────────


@dataclass
class HeartbeatDeps:
    """Dependency-injection bundle mirroring ``createConsciousnessLoop``'s
    keyword arguments in BaiLongma's ``consciousness-loop.js``.

    Every callable is *synchronous* except ``run_turn``. Hermes wires
    each field to an existing service:

    * ``run_turn`` → wraps ``AIAgent.run_conversation`` for one turn.
    * ``get_current_execution`` → returns the currently-running turn's
      priority + start timestamp + label, or ``None`` when idle.
    * ``get_current_abort_controller`` → the ``asyncio.Event`` /
      ``AbortLike`` covering the running turn.
    * ``emit_event`` → forwards to Hermes gateway/CLI event stream.
    * scheduling knobs (``get_base_tick_interval``, etc.) → resolve
      per-user config; testable in isolation.
    """

    run_turn: RunTurn
    run_turn_watchdog_s: float
    get_current_execution: Callable[[], Optional[Mapping[str, Any]]]
    get_current_abort_controller: Callable[[], Optional[AbortLike]]
    clear_current_execution: Callable[[], None]
    emit_event: EmitEvent
    enqueue_due_reminders: Callable[[], None]
    has_messages: Callable[[], bool]
    pop_message: Callable[[], Any]
    has_user_messages: Callable[[], bool]
    get_queue_snapshot: Callable[[], Mapping[str, int]]
    format_tick: Callable[[], Any]
    consume_ticker_tick: Callable[[], None]
    decrement_awakening_tick: Callable[[], None]
    is_startup_self_check_active: Callable[[], bool]
    is_running: Callable[[], bool]
    set_scheduler: Callable[[Scheduler], None]
    set_interrupt_callback: Callable[[InterruptCallback], None]
    is_rate_limited: Callable[[], bool]
    get_tick_interval: Callable[[float], float]
    get_base_tick_interval: Callable[[], float]
    get_custom_interval_ms: Callable[[], Optional[float]]
    get_ticker_status: Callable[[], Optional[Mapping[str, Any]]]
    get_awakening_ticks: Callable[[], int]
    is_task_active: Callable[[], bool]
    get_next_pending_reminder: Callable[[], Optional[Mapping[str, Any]]]
    get_quota_status: Callable[[], Mapping[str, Any]]
    start_consolidation_loop: Callable[[], None]
    ensure_startup_self_check_state: Callable[[], None]
    set_sticky_event: Callable[[str, Mapping[str, Any]], None]
    startup_self_check_version: str
    priorities: Priorities = field(default_factory=Priorities)


# ── The loop ────────────────────────────────────────────────────────────────


class ConsciousnessLoop:
    """Self-scheduling heartbeat loop.

    Public surface mirrors BaiLongma's returned object:
        * ``is_processing()`` → bool
        * ``mark_last_tick_aborted()`` → None
        * ``on_tick()`` coroutine
        * ``schedule_next_tick()`` → None
        * ``trigger_immediate_tick()`` → None
        * ``start(run_immediate_tick=True)`` coroutine
    """

    def __init__(self, deps: HeartbeatDeps) -> None:
        self._d = deps
        self._processing = False
        self._last_tick_aborted = False
        # Handle to the ``call_later`` timer that will fire the next tick;
        # any incoming message clears it and runs on_tick immediately.
        self._current_timer: Optional[asyncio.TimerHandle] = None
        self._loop_started = False
        # Cache the event loop we belong to so timers land on the
        # correct thread (Hermes may have multiple loops).
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ── Public helpers ─────────────────────────────────────────────────────

    def is_processing(self) -> bool:
        return self._processing

    def mark_last_tick_aborted(self) -> None:
        self._last_tick_aborted = True

    # ── Preemption ─────────────────────────────────────────────────────────

    def _should_preempt_for(self, entry: Optional[Mapping[str, Any]]) -> bool:
        """Mirror ``shouldPreemptFor`` in JS.

        Returns ``True`` when the incoming ``entry`` should abort the
        currently-running turn. Rules:

        1. Nothing running → always preempt (start immediately).
        2. Strictly higher priority → preempt.
        3. Two concurrent user messages preempt each other (so a new
           user message can barge through even if the current one is
           stuck in a tool call).
        """
        current_execution = self._d.get_current_execution()
        if not entry or not self._processing or not current_execution:
            return True

        incoming_priority = entry.get("priority") or self._d.priorities.background
        if incoming_priority > current_execution.get("priority", 0):
            return True

        # Allow preemption between concurrent user messages.
        if (
            incoming_priority >= self._d.priorities.user
            and current_execution.get("priority", 0) >= self._d.priorities.user
        ):
            return True

        return False

    # ── Watchdog wrapper ───────────────────────────────────────────────────

    async def _run_turn_with_watchdog(
        self, input_: Any, label: str, msg: Optional[Any]
    ) -> None:
        """Wrap ``run_turn`` with a watchdog: on timeout, abort the
        current controller and raise ``WatchdogTimeoutError`` so
        ``on_tick``'s ``finally`` block runs and the loop resumes.

        The abandoned ``run_turn`` future is left running; the caller's
        abort should stop it, but even if it doesn't, it becomes an
        unreferenced background task and is eventually reaped by
        garbage collection — same as the JS "never-resolves Promise"
        pattern in BaiLongma.
        """
        timeout_s = self._d.run_turn_watchdog_s

        async def _work() -> None:
            await self._d.run_turn(input_, label, msg)

        task = asyncio.ensure_future(_work())
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout_s)
        except asyncio.TimeoutError as exc:
            current_execution = self._d.get_current_execution() or {}
            stuck_label = current_execution.get("label") or label
            started_at = current_execution.get("started_at")
            elapsed_s = (
                round(time.time() - started_at) if started_at is not None else None
            )
            logger.error(
                "[watchdog] run_turn stuck > %ss (label=%s, elapsed=%ss); forcing abort",
                timeout_s,
                stuck_label,
                elapsed_s,
            )
            try:
                controller = self._d.get_current_abort_controller()
                if controller is not None:
                    controller.abort("watchdog timeout")
            except Exception:  # pragma: no cover - defensive
                pass
            # Clear the global execution ref so subsequent messages don't
            # keep aborting the same controller.
            self._d.clear_current_execution()
            try:
                self._d.emit_event(
                    "error",
                    {
                        "label": "watchdog",
                        "error": f"run_turn stuck > {timeout_s}s",
                    },
                )
            except Exception:  # pragma: no cover - defensive
                pass
            raise WatchdogTimeoutError("run_turn watchdog timeout") from exc

    # ── One tick ───────────────────────────────────────────────────────────

    async def on_tick(self) -> None:
        """Run exactly one heartbeat.

        If there are pending messages, pop and dispatch the head of the
        queue. Otherwise fire an autonomous L2 tick and let the model
        decide what (if anything) to do.
        """
        if self._processing:
            return
        self._processing = True
        self._last_tick_aborted = False
        auto_tick = False
        ticker_revision_at_start: Optional[int] = None

        try:
            self._d.enqueue_due_reminders()
            if self._d.has_messages():
                msg = self._d.pop_message()
                queue_name = getattr(msg, "queueName", None) or getattr(
                    msg, "queue_name", None
                )
                lane = "BG" if queue_name == "background" else "L1"
                from_id = getattr(msg, "fromId", None) or getattr(msg, "from_id", "?")
                raw = getattr(msg, "raw", msg)
                await self._run_turn_with_watchdog(raw, f"{lane} message from {from_id}", msg)
            else:
                auto_tick = True
                ticker = self._d.get_ticker_status()
                ticker_revision_at_start = ticker.get("revision") if ticker else None
                tick_prompt = self._d.format_tick()
                await self._run_turn_with_watchdog(tick_prompt, "L2 TICK", None)
        except WatchdogTimeoutError:
            self._last_tick_aborted = True
        except Exception as err:
            # A failed autonomous turn did not consume a meaningful
            # heartbeat. Preserve cadence/awakening state so the next
            # tick can retry or make a different judgment.
            if auto_tick:
                self._last_tick_aborted = True
            logger.exception("[on_tick] run_turn raised: %s", err)
        finally:
            self._processing = False
            # Cadence TTL and awakening state describe autonomous
            # heartbeats, not user/background messages sharing this
            # scheduler slot. A cadence configured *during* this tick
            # starts governing the *next* tick — do not spend one of
            # its requested TTL rounds here.
            current_ticker = self._d.get_ticker_status() or {}
            ticker_was_reconfigured = (
                auto_tick
                and ticker_revision_at_start is not None
                and current_ticker.get("revision") != ticker_revision_at_start
            )
            if auto_tick and not self._last_tick_aborted and not ticker_was_reconfigured:
                self._d.consume_ticker_tick()
            # When interrupted by the user, retry the same awakening
            # moment on the next heartbeat.
            if auto_tick and not self._last_tick_aborted:
                self._d.decrement_awakening_tick()

    # ── Scheduler ──────────────────────────────────────────────────────────

    def schedule_next_tick(self) -> None:
        """Decide when the next tick fires and arm the timer.

        Ladder (high to low):
            1. Messages pending           → 0s
            2. 429 rate-limited           → quota-provided interval
            3. Custom cadence TTL > 0     → caller-provided ms
            4. Awakening ticks remaining  → 10s
            5. Task active                → 30s
            6. Idle                       → configured base tick interval
        Any earlier-firing pending reminder collapses ``interval`` to
        that reminder's due delay.
        """
        if not self._d.is_running():
            return
        if self._current_timer is not None:
            self._current_timer.cancel()
            self._current_timer = None

        self._d.enqueue_due_reminders()

        has_pending = self._d.has_messages()
        has_pending_user = self._d.has_user_messages()
        queue_snapshot = self._d.get_queue_snapshot()
        rate_limited = self._d.is_rate_limited()
        custom_ms = self._d.get_custom_interval_ms()
        task_active = self._d.is_task_active()
        next_reminder = self._d.get_next_pending_reminder()
        base_tick_interval_s = self._d.get_base_tick_interval()

        interval_s: float
        label: str
        if has_pending_user:
            interval_s = 0.0
            label = "immediate (user message pending)"
        elif has_pending:
            interval_s = 0.0
            label = "immediate (background message pending)"
        elif rate_limited:
            interval_s = self._d.get_tick_interval(base_tick_interval_s)
            label = f"rate-limited ({interval_s}s)"
        elif custom_ms is not None:
            ticker = self._d.get_ticker_status() or {}
            interval_s = custom_ms / 1000.0
            ttl = ticker.get("ttl", "?")
            reason = ticker.get("reason")
            reason_suffix = f" · {reason}" if reason else ""
            label = f"L2 custom {interval_s}s ({ttl} tick(s) remaining{reason_suffix})"
        elif self._d.get_awakening_ticks() > 0:
            aw_ticks = self._d.get_awakening_ticks()
            interval_s = 10.0
            label = f"awakening 10s ({aw_ticks} tick(s) remaining)"
        elif task_active:
            interval_s = 30.0
            label = "task mode 30s"
        else:
            interval_s = base_tick_interval_s
            label = f"{interval_s}s"

        if next_reminder:
            due_at_ms = next_reminder.get("due_at_ms")
            if due_at_ms is None:
                # Accept ISO string too for cross-language parity.
                due_at = next_reminder.get("due_at")
                if isinstance(due_at, (int, float)):
                    due_at_ms = float(due_at)
                elif isinstance(due_at, str):
                    try:
                        from datetime import datetime

                        due_at_ms = datetime.fromisoformat(due_at).timestamp() * 1000
                    except ValueError:
                        due_at_ms = None
            if due_at_ms is not None:
                due_in_s = max(0.0, (due_at_ms / 1000.0) - time.time())
                if due_in_s < interval_s:
                    interval_s = due_in_s
                    label = f"reminder fires in {int(due_in_s) + 1}s"

        quota = self._d.get_quota_status()
        logger.info(
            "[quota] %s RPM | %s TPM | ratio %s | queue U:%s B:%s | next tick %s",
            quota.get("rpm_used", quota.get("rpmUsed", "?")),
            quota.get("tpm_used", quota.get("tpmUsed", "?")),
            quota.get("ratio", "?"),
            queue_snapshot.get("user", "?"),
            queue_snapshot.get("background", "?"),
            label,
        )
        self._d.emit_event(
            "quota",
            {
                **quota,
                "next_tick_ms": interval_s * 1000,
                "ticker": self._d.get_ticker_status(),
                "queue": queue_snapshot,
            },
        )

        loop = self._ensure_loop()
        self._current_timer = loop.call_later(
            interval_s, lambda: asyncio.ensure_future(self._timer_body())
        )

    async def _timer_body(self) -> None:
        """Timer callback body — always re-arms even if ``on_tick``
        raised, so a single bad tick can't permanently stall the loop.
        """
        self._current_timer = None
        try:
            await self.on_tick()
        except Exception as err:  # pragma: no cover - defensive
            logger.exception("[schedule_next_tick] on_tick threw: %s", err)
        finally:
            self.schedule_next_tick()

    # ── Immediate tick ─────────────────────────────────────────────────────

    def trigger_immediate_tick(self) -> None:
        """Called when a new message arrives: cancel any pending timer
        and run the next tick immediately.

        If currently processing, do nothing — rely on the abort
        mechanism to finish quickly; the post-finish
        ``schedule_next_tick`` will then use ``interval=0`` to resume.
        """
        if self._processing:
            return
        if not self._d.is_running():
            return
        if self._current_timer is not None:
            self._current_timer.cancel()
            self._current_timer = None

        async def _kick() -> None:
            try:
                await self.on_tick()
            except Exception as err:  # pragma: no cover - defensive
                logger.exception("[trigger_immediate_tick] on_tick threw: %s", err)
            finally:
                self.schedule_next_tick()

        asyncio.ensure_future(_kick())

    # ── Startup ────────────────────────────────────────────────────────────

    async def start(self, *, run_immediate_tick: bool = True) -> None:
        """Boot the loop.

        Registers the scheduler + interrupt callback with the caller so
        the control layer can wake the loop, then optionally fires one
        immediate tick (used by the activation flow to trigger a
        startup self-check on first boot).
        """
        if self._loop_started:
            return
        self._loop_started = True

        self._d.start_consolidation_loop()

        # Let the control layer (stop/start) reach the scheduler.
        self._d.set_scheduler(self.schedule_next_tick)

        def _interrupt_callback(entry: Any) -> None:
            current_abort = self._d.get_current_abort_controller()
            if current_abort is not None and self._should_preempt_for(entry):
                from_id = (
                    entry.get("fromId")
                    or entry.get("from_id")
                    or "?"
                    if isinstance(entry, Mapping)
                    else "?"
                )
                queue_name = (
                    entry.get("queueName")
                    or entry.get("queue_name")
                    or "?"
                    if isinstance(entry, Mapping)
                    else "?"
                )
                priority = (
                    entry.get("priority") if isinstance(entry, Mapping) else None
                )
                logger.info(
                    "[system] Higher-priority message arrived — "
                    "interrupting current processing: %s (%s)",
                    from_id,
                    queue_name,
                )
                self._d.emit_event(
                    "processing_preempted",
                    {
                        "by": from_id,
                        "queue_name": queue_name,
                        "priority": priority,
                        "current": self._d.get_current_execution(),
                    },
                )
                current_abort.abort("higher-priority-message")
            self.trigger_immediate_tick()

        self._d.set_interrupt_callback(_interrupt_callback)

        # Initialize self-check state before the first tick so the
        # first tick can run self-check.
        self._d.ensure_startup_self_check_state()
        if self._d.is_startup_self_check_active():
            logger.info("[system] Startup self-check starting")
            payload = {"version": self._d.startup_self_check_version}
            self._d.set_sticky_event("startup_self_check_started", payload)
            self._d.emit_event("startup_self_check_started", payload)

        # Whether to fire an immediate L2 TICK is up to the caller;
        # initial activation uses it to trigger self-check.
        if run_immediate_tick:
            await self.on_tick()
        self.schedule_next_tick()

    # ── Helpers ────────────────────────────────────────────────────────────

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self._loop = asyncio.get_event_loop()
        return self._loop


__all__ = [
    "AbortLike",
    "ConsciousnessLoop",
    "HeartbeatDeps",
    "Priorities",
    "WatchdogTimeoutError",
]
