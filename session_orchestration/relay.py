"""
Deterministic relay — acquire the per-session lock, drive via adapter, release.

Responsibilities
----------------
1. **Lock acquisition** — calls ``registry.acquire_lock(task_id, holder, ttl_seconds)``
   with a TTL of 5× the cron interval (default 300 s) before doing anything
   to the tmux pane.  If the lock is held by a non-expired holder, raises
   ``LockConflictError`` (caller decides whether to retry or escalate).

2. **Readiness check + drive** — delegates entirely to the adapter's ``drive()``
   method, which performs the load-buffer/paste-buffer sequence with its own
   prompt-readiness check.  The relay does NOT re-implement tmux calls.

3. **Lock release** — calls ``registry.release_lock(task_id, holder)`` in a
   ``finally`` block so a crash mid-drive self-heals after the TTL.

4. **Handoff detection + deterministic resume** — if ``adapter.detect()`` returns
   ``PAUSED_HANDOFF`` BEFORE driving, the relay calls ``adapter.resume()``
   (``/clear`` + re-inject) and skips the normal ``drive()`` path.  This is
   the ONLY time ``/clear`` is issued by the relay; it is deterministic (no LLM).

Stale-lock reclaim
------------------
The TTL and stale-reclaim logic live in ``registry.acquire_lock``; the relay
does not implement its own.  ``lock_ts`` stores a float expiry epoch (seconds
since epoch), NOT an ISO-8601 string, to avoid ``datetime.fromisoformat``
timezone-parsing ambiguity in Python 3.11 — T002's design decision; relay
matches it via the shared ``acquire_lock`` API and does not introduce a
second timestamp representation.

Concurrency contract
--------------------
A relay-send and a watcher-style capture issued concurrently to the same
``task_id`` use the same ``acquire_lock`` / ``release_lock`` API.  Because
``acquire_lock`` uses ``BEGIN IMMEDIATE`` (SQLite write lock), only one caller
can hold the lock at a time; the other caller blocks on SQLite's ``busy_timeout``
(default 5 s) and then gets ``False`` returned.  This makes concurrent
interleaving provably impossible for any two callers that correctly call
``acquire_lock`` before touching the pane.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from session_orchestration.adapters.base import AgentAdapter
from session_orchestration.registry import SessionOrchestrationRegistry
from session_orchestration.types import SessionHandle, SessionLifecycle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Default lock TTL in seconds (5× a 60 s cron interval).
_DEFAULT_TTL: float = 300.0

#: Seconds to wait between lock-acquire retries when the caller opts in to retry.
_RETRY_INTERVAL: float = 0.25

#: Maximum seconds ``send_message`` will retry lock acquisition when
#: ``retry_on_conflict=True``.
_RETRY_TIMEOUT: float = 10.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LockConflictError(RuntimeError):
    """Raised when the per-session lock cannot be acquired because a non-expired
    holder already owns it and ``retry_on_conflict=False`` (the default).
    """


# ---------------------------------------------------------------------------
# Relay
# ---------------------------------------------------------------------------


class SessionRelay:
    """Mediates all pane-writes for a managed session.

    Parameters
    ----------
    registry:
        The ``SessionOrchestrationRegistry`` instance.
    adapter:
        The ``AgentAdapter`` (e.g. ``ClaudeCodeAdapter``) for this session.
    ttl_seconds:
        Lock TTL for each relay call.  Default 300 s (5× 60 s cron interval).
    holder_prefix:
        Prefix for lock-holder identifiers.  Combined with the process PID
        to form an opaque string like ``"relay:pid:1234"``.
    """

    def __init__(
        self,
        registry: SessionOrchestrationRegistry,
        adapter: AgentAdapter,
        *,
        ttl_seconds: float = _DEFAULT_TTL,
        holder_prefix: str = "relay",
    ) -> None:
        self._registry = registry
        self._adapter = adapter
        self._ttl = ttl_seconds
        self._holder_prefix = holder_prefix

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_message(
        self,
        task_id: str,
        handle: SessionHandle,
        message: str,
        *,
        retry_on_conflict: bool = False,
        pre_keys: list[str] | None = None,
    ) -> None:
        """Send ``message`` to the session identified by ``task_id``/``handle``.

        Acquires the per-session lock, checks for a handoff (issuing
        ``/clear``+resume if detected), drives the message, and releases
        the lock.

        Parameters
        ----------
        task_id:
            Registry row key.
        handle:
            ``SessionHandle`` for the tmux session.
        message:
            Text to deliver to the agent.
        retry_on_conflict:
            If True, spin-retry lock acquisition for up to
            ``_RETRY_TIMEOUT`` seconds.  If False (default), raise
            ``LockConflictError`` immediately on conflict.

        Raises
        ------
        LockConflictError
            When ``retry_on_conflict=False`` and the lock is already held.
        TimeoutError
            Propagated from ``adapter.drive()`` if the pane prompt does not
            become ready within the adapter's configured timeout.
        """
        holder = self._make_holder()
        acquired = self._acquire_with_optional_retry(
            task_id, holder, retry_on_conflict=retry_on_conflict
        )
        if not acquired:
            raise LockConflictError(
                f"Per-session lock for {task_id!r} is held by another process; "
                f"cannot send message."
            )

        try:
            logger.debug(
                "relay.send_message: lock acquired for task_id=%s holder=%s",
                task_id,
                holder,
            )
            # Check for handoff before driving — deterministic /clear+resume.
            lifecycle = self._adapter.detect(handle)
            if lifecycle == SessionLifecycle.PAUSED_HANDOFF:
                logger.info(
                    "relay.send_message: PAUSED_HANDOFF detected for task_id=%s; "
                    "issuing /clear+resume (no LLM).",
                    task_id,
                )
                self._adapter.resume(handle, message)
            elif pre_keys:
                # Only thread pre_keys when present so adapters/fakes with the
                # 2-arg drive() signature keep working on the common path.
                self._adapter.drive(handle, message, pre_keys=pre_keys)
            else:
                self._adapter.drive(handle, message)
        finally:
            self._registry.release_lock(task_id, holder)
            logger.debug(
                "relay.send_message: lock released for task_id=%s holder=%s",
                task_id,
                holder,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_holder(self) -> str:
        """Return an opaque lock-holder identifier for this process."""
        return f"{self._holder_prefix}:pid:{os.getpid()}:{time.time():.3f}"

    def _acquire_with_optional_retry(
        self,
        task_id: str,
        holder: str,
        *,
        retry_on_conflict: bool,
    ) -> bool:
        """Try to acquire the lock.

        If ``retry_on_conflict`` is True, spin for up to ``_RETRY_TIMEOUT``
        seconds.  Returns True on success, False on failure.
        """
        if not retry_on_conflict:
            return self._registry.acquire_lock(task_id, holder, ttl_seconds=self._ttl)

        deadline = time.monotonic() + _RETRY_TIMEOUT
        while time.monotonic() < deadline:
            if self._registry.acquire_lock(task_id, holder, ttl_seconds=self._ttl):
                return True
            time.sleep(_RETRY_INTERVAL)
        return False
