"""Watchdog for wedged llama.cpp router children (issue #104050).

A router-mode child can wedge: ``/health`` stays 200 and ``GET /models`` still
reports the model loaded, while every ``/v1/chat/completions`` fails (HTTP 500
``Compute error``) and the child even ignores SIGTERM. The crash-restart loop
only sees process exits, so a live-but-wedged child is invisible — and the
agent burns its whole retry budget on ``server_error``, which is not
fallback-eligible, until the turn dies with ``max_retries_exhausted``.

This module counts consecutive inference failures per model (fed by two small
hooks in the agent turn loop) and, at a threshold, confirms with a live
``touch_generate`` probe before healing. Healing still respects the module
boundary documented in ``supervisor.py`` ("router children are its problem"):
the first step ASKS THE ROUTER to unload the model (it owns child lifecycle
and autoloads a fresh child on the next request); only when the router leaves
a probed-dead child resident is the router process we own bounced (its
terminate path already escalates SIGTERM to SIGKILL, which reaps
SIGTERM-ignoring children). Cooldown-bounded, flag-gated, never raises.

Threshold equals the default API retry budget (3): one fully-failed turn at
default settings is exactly enough to suspect a wedge, and the probe confirms
before anything heals — transient blips cost at most one probe generation.
Only ``server_error`` (HTTP 500-class) counts: a wedged child answers 500, and
counting timeouts would heal slow-but-healthy giants (a 27B @ 262k ctx first
token on Metal can take minutes).
"""

from __future__ import annotations

from contextlib import suppress
import logging
import threading
import time

logger = logging.getLogger(__name__)

FAILURE_THRESHOLD = 3
COOLDOWN_S = 30 * 60  # sweeper cadence is 120s: at most one heal per 15 sweeps
PROBE_TIMEOUT_S = 60  # a healthy child answers the 1-word probe in seconds
_RESIDENT = ("loaded", "ready")


class ChildWatchdog:
    """Per-model consecutive-failure counting + probe/unload/bounce healing.

    Pure logic over a duck-typed supervisor (``models()``,
    ``touch_generate()``, ``unload_model()``, ``stop()``/``start()``,
    ``primary_model``); every external call is guarded so healing never
    raises into the maintenance loop, let alone the agent loop.
    """

    def __init__(self, *, failure_threshold: int = FAILURE_THRESHOLD,
                 cooldown_s: float = COOLDOWN_S, enabled: bool = True,
                 clock=time.monotonic):
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self.enabled = enabled
        self._clock = clock
        self._lock = threading.Lock()
        self._consecutive: dict[str, int] = {}
        self._last_heal: dict[str, float] = {}

    # ── hook surface (agent turn loop) ──────────────────────────

    def note_result(self, model_id: str, *, ok: bool) -> None:
        """Record one inference outcome. Success resets the streak; failures
        accumulate only while the flag is on (flag-off is a full no-op)."""
        if not self.enabled or not model_id:
            return
        with self._lock:
            if ok:
                self._consecutive.pop(model_id, None)
            else:
                self._consecutive[model_id] = self._consecutive.get(model_id, 0) + 1

    def consecutive_failures(self, model_id: str) -> int:
        with self._lock:
            return self._consecutive.get(model_id, 0)

    def due_models(self, now: float | None = None) -> "list[str]":
        """Models at/over threshold whose cooldown has expired."""
        if not self.enabled:
            return []
        now = self._clock() if now is None else now
        with self._lock:
            return [m for m, n in self._consecutive.items()
                    if n >= self.failure_threshold
                    and now - self._last_heal.get(m, float("-inf")) >= self.cooldown_s]

    # ── healing (maintenance loop) ──────────────────────────────

    @staticmethod
    def _resolve(model_id: str, resident: dict) -> "str | None":
        """Map a recorded id to a resident router model: exact first, then the
        unique startswith/contains hit (agent model names are stems/aliases of
        the router's ``<stem>.gguf`` ids — same rule as window growth)."""
        if model_id in resident:
            return model_id
        hits = [m for m in resident
                if m.startswith(model_id) or model_id in m]
        return hits[0] if len(hits) == 1 else None

    def maybe_heal(self, sup, *, now: float | None = None) -> "list[str]":
        """Probe every due model; unload (then bounce) the probed-dead ones.
        Returns healed model ids. Never raises."""
        healed: list[str] = []
        try:
            due = self.due_models(now)
            if not due:
                return healed
            try:
                resident = sup.models()
            except Exception:  # noqa: BLE001 — router unreachable; try next sweep
                logger.debug("child watchdog: /models unreadable, deferring")
                return healed
            for model_id in due:
                try:
                    if self._heal_one(sup, model_id, resident):
                        healed.append(model_id)
                except Exception as exc:  # noqa: BLE001 — per-model isolation
                    logger.warning("child watchdog: healing %s failed: %s", model_id, exc)
                finally:
                    stamp = self._clock() if now is None else now
                    with self._lock:
                        self._last_heal[model_id] = stamp
                        self._consecutive.pop(model_id, None)
        except Exception as exc:  # noqa: BLE001 — never break the sweep
            logger.debug("child watchdog skipped: %s", exc)
        return healed

    def _heal_one(self, sup, model_id: str, resident: dict) -> bool:
        target = self._resolve(model_id, resident)
        if target is None:
            logger.info("child watchdog: %s no longer resident; streak dropped", model_id)
            return False
        try:
            if sup.touch_generate(target, timeout_s=PROBE_TIMEOUT_S):
                logger.info("child watchdog: %s probes healthy; streak reset", target)
                return False
        except Exception as exc:  # noqa: BLE001 — probe error == probe failure
            logger.warning("child watchdog: probe of %s errored: %s", target, exc)
        logger.warning("child watchdog: %s failed its probe after %s consecutive "
                       "server errors; unloading", target, self.consecutive_failures(model_id))
        with suppress(Exception):
            sup.unload_model(target)
        try:
            if resident_status(sup, target) not in (*_RESIDENT, "unloading"):
                logger.warning("child watchdog: %s unloaded; router autoloads a "
                               "fresh child on next request", target)
                return True
        except Exception:  # noqa: BLE001 — status unreadable counts as stuck
            pass
        # The router won't drop a probed-dead child: bounce the router we own.
        # _terminate_tree already escalates SIGTERM to SIGKILL, which reaps
        # children that ignore SIGTERM (the reported symptom).
        logger.warning("child watchdog: %s still resident after unload; "
                       "bouncing the managed router", target)
        sup.stop()
        sup.start()
        if getattr(sup, "primary_model", None):
            with suppress(Exception):
                sup.ensure_model_ready(sup.primary_model)
        return True


def resident_status(sup, model_id: str):
    """Current ``/models`` status value for one model (None when absent)."""
    return sup.models().get(model_id)
