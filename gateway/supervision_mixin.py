"""Supervised long-lived background task support for ``GatewayRunner``.

Extracted from ``gateway/run.py`` (god-file decomposition campaign, Wave 1
mixin lift). This mixin holds the task-level supervision cluster:
``_spawn_supervised`` (with its nested ``_done`` / ``_respawn`` callbacks),
which launches long-lived background watchers with capped-exponential-backoff
restart supervision.

Behavior-neutral: the method is lifted verbatim from ``GatewayRunner``.
``self.*`` calls resolve unchanged via the MRO. The class constants
``_MAX_SUPERVISED_RESTARTS`` and ``_SUPERVISED_HEALTHY_SECS`` remain defined
on ``GatewayRunner`` (MRO resolves them); ``self._background_tasks`` and
``self._running`` are instance state. The module-level ``logger`` is
``logging.getLogger("gateway.run")`` so log records keep the exact name
(``"gateway.run"``), matching the sibling mixins' convention.
"""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger("gateway.run")


class GatewaySupervisionMixin:
    def _spawn_supervised(self, coro_factory, name, *, restart=True, _attempt=0, on_spawn=None):
        """Launch a long-lived background task with task-level supervision.

        Complements upstream's per-iteration inner-loop try/except (which only
        guards a single loop-body) by covering what that CANNOT: an exception
        raised in the watcher's OUTER ``while self._running:`` loop or its
        pre-try setup region, plus task-level death generally. A bare
        ``asyncio.create_task`` drops such an exception on the floor — no log,
        no restart, the watcher silently gone. This retains the handle in
        ``self._background_tasks``, logs any crash, and restarts with capped
        exponential backoff up to ``_MAX_SUPERVISED_RESTARTS`` failures in rapid
        succession (each within ``_SUPERVISED_HEALTHY_SECS`` of its restart).
        The counter resets after any run that stayed healthy for at least
        ``_SUPERVISED_HEALTHY_SECS`` — so a long-lived daemon that crashes
        occasionally over days is never permanently abandoned.

        ``on_spawn`` (optional) is invoked with the freshly-created task on
        every spawn, INCLUDING internal backoff respawns. Callers that also
        track the live handle elsewhere (e.g. ``self._reconnect_watcher_task``
        for ``_ensure_reconnect_watcher_running``) MUST pass it — otherwise the
        supervisor's own respawn creates a new task without updating that
        external handle, so ``_ensure_...`` later sees the stale/done handle
        and spawns a SECOND concurrent watcher (double reconnect attempts).
        """
        if getattr(self, "_background_tasks", None) is None:
            self._background_tasks = set()

        # Monotonic spawn timestamp captured per spawn: the ``_done`` callback
        # uses it to distinguish a rapid crash-loop from a healthy-run-then-crash.
        _started = time.monotonic()

        # Deliberately do NOT pass name= to create_task — some test doubles mock
        # create_task with a signature that rejects the name kwarg.
        task = asyncio.create_task(coro_factory())
        self._background_tasks.add(task)
        if on_spawn is not None:
            # Record the live handle NOW so an external tracker (e.g.
            # _reconnect_watcher_task) always points at the current task, not a
            # dead one left behind by a prior supervised respawn.
            try:
                on_spawn(task)
            except Exception:  # pragma: no cover - defensive; a tracker must never kill the spawn
                logger.debug("on_spawn callback for %s raised", name, exc_info=True)

        def _done(t):
            self._background_tasks.discard(t)
            if t.cancelled():
                return
            exc = t.exception()
            if exc is None:
                # Clean return == deliberate shutdown or a self-disabling watcher
                # (e.g. a gated no-op that returns synchronously). Respawning here
                # would busy-spin such a watcher — so NEVER restart on clean exit.
                return
            logger.error("Supervised task %s died: %r", name, exc, exc_info=exc)
            if restart and self._running:
                ran_for = time.monotonic() - _started
                if ran_for >= self._SUPERVISED_HEALTHY_SECS:
                    # Ran healthily for a while before crashing — this is a
                    # FRESH failure, not part of a rapid crash-loop. Reset the
                    # consecutive counter so a daemon that crashes a handful of
                    # times over days is never permanently abandoned.
                    effective_attempt = 0
                else:
                    effective_attempt = _attempt
                if effective_attempt >= self._MAX_SUPERVISED_RESTARTS:
                    logger.error(
                        "Supervised task %s died %d times in rapid succession "
                        "(each within %ds of restart) — giving up restarts",
                        name,
                        effective_attempt,
                        self._SUPERVISED_HEALTHY_SECS,
                    )
                    return
                backoff = min(60, 2 ** min(effective_attempt, 6))

                async def _respawn():
                    await asyncio.sleep(backoff)
                    if self._running:
                        self._spawn_supervised(
                            coro_factory,
                            name,
                            restart=restart,
                            _attempt=effective_attempt + 1,
                            on_spawn=on_spawn,
                        )

                respawn_task = asyncio.create_task(_respawn())
                self._background_tasks.add(respawn_task)
                respawn_task.add_done_callback(self._background_tasks.discard)

        task.add_done_callback(_done)
        return task
