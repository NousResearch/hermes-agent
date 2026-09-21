"""Control Room action routing (CR-201..CR-208).

A narrow dispatcher over the authoritative existing operations. Control Room
never owns a second state machine: every action kind maps to ONE executor that
wraps the existing approved API (approval.respond, process kill, subagent
interrupt/steer, delegation pause, Hermes Peer public API, Kanban plugin write
route). Renderers never guess capability — the router computes it.

Safety properties enforced here:
- ``confirmation: required`` actions return ``confirmation_required`` with a
  preview on the first dispatch; only an explicit ``confirmed=True`` dispatch
  executes (CR-207). Renderers may style the preview but cannot omit it.
- ``expected_revision`` forces a re-fetch of the target via ``verifier``
  before mutation; a mismatch returns ``stale`` (CR-201).
- Cross-profile targets are rejected (``CROSS_PROFILE``).
- Duplicate submit of the same action id is rejected once it has executed
  (bounded memory).
- No direct SQLite writes: Kanban actions go through the established plugin /
  locked write route, never raw SQL from this module.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from .contract import (
    ActionTarget,
    ControlRoomAction,
    ControlRoomActionResult,
    ControlRoomError,
    ErrorCode,
)

# Executor signature: (target, parameters, context) -> ControlRoomActionResult
ExecutorFn = Callable[[ActionTarget, Dict[str, Any], Dict[str, Any]], ControlRoomActionResult]
# Verifier signature: (target, context) -> Optional[str]  (live revision or None if gone)
VerifierFn = Callable[[ActionTarget, Dict[str, Any]], Optional[str]]

DEFAULT_ACTION_TTL_SECONDS = 3600


class ControlRoomActionRouter:
    """Routes ControlRoomAction envelopes to registered executors."""

    def __init__(
        self,
        executors: Optional[Dict[str, ExecutorFn]] = None,
        verifier: Optional[VerifierFn] = None,
        *,
        scope_profile: str = "default",
        allow_cross_profile: bool = False,
        action_ttl_seconds: float = DEFAULT_ACTION_TTL_SECONDS,
    ) -> None:
        self.executors: Dict[str, ExecutorFn] = dict(executors or {})
        self.verifier = verifier
        self.scope_profile = scope_profile
        self.allow_cross_profile = allow_cross_profile
        self._executed: Dict[str, float] = {}
        self.action_ttl_seconds = action_ttl_seconds

    def register(self, kind: str, executor: ExecutorFn) -> None:
        self.executors[kind] = executor

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(
        self,
        action: ControlRoomAction,
        context: Optional[Dict[str, Any]] = None,
        *,
        confirmed: bool = False,
    ) -> ControlRoomActionResult:
        ctx = context or {}

        # 1. Cross-profile guard: a target that names a different profile than
        #    the router scope is rejected unless explicitly allowed.
        target_profile = action.target.profile or self.scope_profile
        if target_profile != self.scope_profile and not self.allow_cross_profile:
            return ControlRoomActionResult(
                status="rejected",
                message=f"target profile {target_profile!r} outside scope {self.scope_profile!r}",
                receipt={"error": ErrorCode.CROSS_PROFILE.value},
            )

        # 2. Unknown action kind.
        if action.target.kind not in self.executors:
            return ControlRoomActionResult(
                status="unavailable",
                message=f"no executor for action kind {action.target.kind!r}",
                receipt={"error": ErrorCode.UNKNOWN_ACTION.value},
            )

        # 3. Two-step confirmation contract (CR-207): required actions need an
        #    explicit confirmed dispatch; the first call returns a preview.
        if action.confirmation == "required" and not confirmed:
            return ControlRoomActionResult(
                status="confirmation_required",
                message=self._preview(action),
                receipt={"action_id": action.id, "target": action.target.model_dump()},
            )

        # 4. Stale-revision check (CR-201): re-fetch the target before mutation.
        if action.expected_revision is not None and self.verifier is not None:
            try:
                live = self.verifier(action.target, ctx)
            except Exception as exc:  # noqa: BLE001
                return ControlRoomActionResult(
                    status="failed",
                    message=f"re-fetch failed: {exc}",
                    receipt={"error": ErrorCode.BACKEND_FAILED.value},
                )
            if live != action.expected_revision:
                return ControlRoomActionResult(
                    status="stale",
                    message="target changed since this row was loaded; refresh required",
                    receipt={"expected": action.expected_revision, "live": live},
                )

        # 5. Duplicate-submit guard: once an action id has executed, reject
        #    identical resubmission within the TTL window.
        now = time.monotonic()
        if action.id in self._executed:
            executed_at = self._executed[action.id]
            if now - executed_at < self.action_ttl_seconds:
                return ControlRoomActionResult(
                    status="rejected",
                    message=f"action {action.id} already executed; duplicate submit rejected",
                    receipt={"error": ErrorCode.UNKNOWN_ACTION.value, "duplicate": True},
                )

        # 6. Execute through the authoritative executor.
        executor = self.executors[action.target.kind]
        try:
            result = executor(action.target, action.parameters, ctx)
        except Exception as exc:  # noqa: BLE001 - executor isolation boundary
            return ControlRoomActionResult(
                status="failed",
                message=f"executor error: {exc}",
                receipt={"error": ErrorCode.BACKEND_FAILED.value},
            )

        if result.status == "completed":
            self._executed[action.id] = now
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _preview(self, action: ControlRoomAction) -> str:
        params = ", ".join(f"{k}={v}" for k, v in (action.parameters or {}).items())
        suffix = f" ({params})" if params else ""
        return f"Confirm {action.target.kind} on {action.target.id}{suffix}?"

    def capability(self, kind: str) -> bool:
        """Server-side capability probe for renderers (never guess)."""
        return kind in self.executors


def error_result(code: ErrorCode, message: str) -> ControlRoomActionResult:
    return ControlRoomActionResult(
        status="failed",
        message=message,
        receipt={"error": code.value},
    )


def from_error(err: ControlRoomError) -> ControlRoomActionResult:
    return ControlRoomActionResult(
        status="failed",
        message=err.message,
        receipt={"error": err.code.value, "details": err.details},
    )
