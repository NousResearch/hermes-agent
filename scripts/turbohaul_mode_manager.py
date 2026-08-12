#!/usr/bin/env python3
"""Turbohaul request-scoped GPU-KV / RAM-KV long-context mode manager.

Implements the core mode-switch contract for the P12 long-context UX decision
(kanban t_fb7af9a5; spec in sibling task t_9377f1fb; ADR 0012 in
docs/adr/0012-p12-256k-memory-policy.md):

- Normal Turbohaul operation runs in GPU-KV mode (KV cache in VRAM, weights
  GPU0-only, fast path).
- An explicit long-context route can enter RAM-KV mode (KV cache in host RAM
  via --no-kv-offload, weights still GPU0-only) for huge-context work, then
  must restore GPU-KV mode afterward.
- Mode state is scoped to the request/route invocation. Normal traffic never
  touches the manager; only the long-context route calls it.

Guarantees:
- Restore is invoked on success, on body error, and on timeout.
- At most one request may hold RAM-KV mode at a time (single-resident-model
  semantics — same constraint as Turbohaul's max_parallel_sidecars=1). A
  second concurrent long-context request is rejected with ModeBusyError.
- Re-entering for the same request id is idempotent (no double physical
  switch).
- The physical switch is an injectable seam (``mode_switcher`` callable) so
  the integration task owns how the live sidecar actually changes flags;
  this module stays dependency-free and unit-testable.
- Every transition is logged as a structured one-line record carrying the
  request id for observability.

Timeout semantics: ``run_long_context`` runs the body in a single-worker
executor and raises ModeTimeoutError when the deadline passes; restore still
runs. The timed-out body thread is allowed to finish in the background rather
than being killed mid-inference (killing a llama-server worker mid-request can
corrupt slot state); the executor is shut down with wait=False so the manager
returns promptly.

Usage (to be wired by the explicit long-context route):

    from scripts.turbohaul_mode_manager import TurbohaulModeManager

    mgr = TurbohaulModeManager(mode_switcher=real_switcher, timeout_s=300.0)

    # Direct API:
    mgr.enter_long_context(request_id)   # -> ModeRecord
    mgr.restore_normal(request_id)       # always restores; no-op if idle
    mgr.effective_mode()                 # 'gpu-kv' | 'ram-kv'
    mgr.active_requests()                # {request_id: mode}

    # Guaranteed restore wrapper:
    result = mgr.run_long_context(request_id, fn)

    # Context-manager form:
    with mgr.long_context(request_id):
        ...
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
from typing import Any, Callable, Optional

# Mode identifiers — the contract these transitions are named by.
MODE_GPU_KV = "gpu-kv"
MODE_RAM_KV = "ram-kv"

# Default timeout for run_long_context bodies (seconds).
DEFAULT_TIMEOUT_S = 300.0

# Structured log prefix for every transition record.
_TRANSITION_PREFIX = "MODE_TRANSITION"


class ModeError(RuntimeError):
    """Base class for mode-manager failures."""


class ModeBusyError(ModeError):
    """Another request already holds RAM-KV long-context mode."""


class ModeTimeoutError(ModeError):
    """The long-context body exceeded its deadline (mode was restored)."""


class ModeSwitchError(ModeError):
    """The physical mode switch failed (state was left consistent)."""


class ModeRecord:
    """Result of entering long-context mode for one request."""

    __slots__ = ("mode", "request_id")

    def __init__(self, mode: str, request_id: str):
        self.mode = mode
        self.request_id = request_id

    def __repr__(self):
        return f"ModeRecord(mode={self.mode!r}, request_id={self.request_id!r})"


class TurbohaulModeManager:
    """Request-scoped GPU-KV <-> RAM-KV mode manager.

    ``mode_switcher`` is a callable ``(mode: str, request_id: str) -> None``
    that performs the physical switch (e.g. restart the sidecar with
    --no-kv-offload for 'ram-kv', without it for 'gpu-kv'). It may raise; the
    manager converts failures to ModeSwitchError and keeps its own state
    consistent (never tracking a request whose switch failed).
    """

    def __init__(
        self,
        mode_switcher: Optional[Callable[[str, str], Any]] = None,
        logger: Optional[logging.Logger] = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        self._mode_switcher = mode_switcher
        self._logger = logger or logging.getLogger("turbohaul.mode")
        self._timeout_s = float(timeout_s)
        self._lock = threading.Lock()
        # request_id -> MODE_RAM_KV for every request currently holding
        # long-context mode. Always a subset of one (single-resident model).
        self._active: dict[str, str] = {}
        # True when the last physical restore attempt failed. Operators can
        # poll this / watch logs to reconcile physical reality.
        self._last_restore_failed = False

    # -- state queries (never mutate) --------------------------------------

    def effective_mode(self) -> str:
        """Return the mode normal traffic currently sees.

        'ram-kv' while any request holds long-context mode, else 'gpu-kv'.
        Normal traffic never calls this unless it needs to observe the mode.
        """
        with self._lock:
            return MODE_RAM_KV if self._active else MODE_GPU_KV

    def is_long_context_active(self) -> bool:
        return self.effective_mode() == MODE_RAM_KV

    def active_requests(self) -> dict[str, str]:
        """Return a copy of the request ids currently holding RAM-KV mode."""
        with self._lock:
            return dict(self._active)

    @property
    def last_restore_failed(self) -> bool:
        return self._last_restore_failed

    # -- internal helpers ---------------------------------------------------

    def _physical_switch(self, mode: str, request_id: str) -> None:
        if self._mode_switcher is None:
            raise ModeSwitchError(
                "no mode_switcher injected; cannot physically switch to "
                f"{mode!r} for request {request_id!r}. The explicit "
                "long-context route must inject the real switcher."
            )
        self._mode_switcher(mode, request_id)

    def _log_transition(self, request_id: str, outcome: str, *, level: int = logging.INFO, **fields: Any) -> None:
        parts = [
            _TRANSITION_PREFIX,
            f"request_id={request_id}",
            f"outcome={outcome}",
        ]
        parts.extend(f"{k}={v}" for k, v in sorted(fields.items()))
        self._logger.log(level, " ".join(parts))

    # -- lifecycle API ------------------------------------------------------

    def enter_long_context(self, request_id: str) -> ModeRecord:
        """Enter RAM-KV long-context mode for ``request_id``.

        Idempotent per request: if the same request id is already holding
        RAM-KV, returns its record without a second physical switch. If a
        different request already holds it, raises ModeBusyError. If the
        physical switch fails, raises ModeSwitchError and leaves no state.
        """
        if not request_id or not isinstance(request_id, str):
            raise ValueError("request_id must be a non-empty string")

        with self._lock:
            if request_id in self._active:
                # Already holding for this request — idempotent re-enter.
                return ModeRecord(MODE_RAM_KV, request_id)
            if self._active:
                holders = ", ".join(sorted(self._active))
                raise ModeBusyError(
                    f"RAM-KV long-context mode already held by request(s): "
                    f"{holders}; single-resident model semantics allow one "
                    f"long-context request at a time (request {request_id!r})"
                )
            # Reserve the single-holder slot UNDER the lock, BEFORE the
            # physical switch. A concurrent request now observes the
            # reservation and raises ModeBusyError instead of racing through
            # the empty-check (TOCTOU fix; mirrors Turbohaul
            # max_parallel_sidecars=1 single-resident semantics).
            self._active[request_id] = MODE_RAM_KV

        # Physical switch outside the lock (it may block / call back).
        try:
            self._physical_switch(MODE_RAM_KV, request_id)
        except ModeSwitchError:
            with self._lock:
                self._active.pop(request_id, None)
            raise
        except Exception as exc:
            self._log_transition(
                request_id, "switch-failed", level=logging.ERROR,
                from_mode=MODE_GPU_KV, to_mode=MODE_RAM_KV, error=str(exc),
            )
            with self._lock:
                self._active.pop(request_id, None)
            raise ModeSwitchError(
                f"failed to switch to RAM-KV for request {request_id!r}: {exc}"
            ) from exc

        self._log_transition(
            request_id, "entered", from_mode=MODE_GPU_KV, to_mode=MODE_RAM_KV,
        )
        return ModeRecord(MODE_RAM_KV, request_id)

    def restore_normal(self, request_id: str) -> None:
        """Restore GPU-KV normal mode for ``request_id``.

        No-op when the request is not holding long-context mode. Restore is
        attempted even if the body failed; a failed physical restore is
        surfaced as ModeSwitchError (with the request removed from the active
        set so future long-context work is not permanently blocked).
        """
        with self._lock:
            if request_id not in self._active:
                return  # nothing to restore

        restore_ok = False
        try:
            self._physical_switch(MODE_GPU_KV, request_id)
            restore_ok = True
        except ModeSwitchError:
            self._last_restore_failed = True
            raise
        except Exception as exc:
            self._last_restore_failed = True
            self._log_transition(
                request_id, "switch-failed", level=logging.ERROR,
                from_mode=MODE_RAM_KV, to_mode=MODE_GPU_KV, error=str(exc),
            )
            raise ModeSwitchError(
                f"failed to restore GPU-KV for request {request_id!r}: {exc}"
            ) from exc
        finally:
            with self._lock:
                self._active.pop(request_id, None)

        self._log_transition(
            request_id, "restored", from_mode=MODE_RAM_KV, to_mode=MODE_GPU_KV,
        )

    def run_long_context(
        self, request_id: str, fn: Callable[[], Any], timeout_s: Optional[float] = None
    ) -> Any:
        """Enter RAM-KV, run ``fn``, and guarantee GPU-KV restore.

        Restore runs on success, on body error, and on timeout (the deadline
        uses the manager's ``timeout_s`` when not passed). Raises
        ModeTimeoutError on timeout and propagates body errors, both after
        restoring. A failed physical restore surfaces as ModeSwitchError even
        if the body also failed (restore failure takes precedence so the
        operator sees the louder, actionable error).
        """
        self.enter_long_context(request_id)
        deadline = self._timeout_s if timeout_s is None else float(timeout_s)
        try:
            if deadline is None or deadline <= 0:
                return fn()
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = pool.submit(fn)
            try:
                return future.result(timeout=deadline)
            except concurrent.futures.TimeoutError:
                # Do NOT wait for the timed-out body; let it finish in the
                # background (killing mid-inference corrupts slot state).
                pool.shutdown(wait=False, cancel_futures=False)
                raise ModeTimeoutError(
                    f"long-context body for request {request_id!r} exceeded "
                    f"{deadline:g}s; RAM-KV mode was restored"
                ) from None
            except BaseException:
                pool.shutdown(wait=False, cancel_futures=False)
                raise
        finally:
            self.restore_normal(request_id)

    def long_context(self, request_id: str):
        """Context-manager form: enter on __enter__, restore on __exit__."""
        return _LongContextScope(self, request_id)


class _LongContextScope:
    """Context manager delegating enter/restore to the manager."""

    def __init__(self, manager: TurbohaulModeManager, request_id: str):
        self._manager = manager
        self._request_id = request_id

    def __enter__(self):
        self._manager.enter_long_context(self._request_id)
        return self

    def __exit__(self, exc_type, exc, tb):
        self._manager.restore_normal(self._request_id)
        return False  # do not suppress body exceptions
