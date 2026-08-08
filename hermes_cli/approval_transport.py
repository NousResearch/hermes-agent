"""Host-owned contract for plugin-provided human approval transports.

Transports only present an immutable, redacted request and return a correlated
human decision. They do not participate in command detection or authorization
policy. The host validates scope, request binding, and timeout fail-closed.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal

logger = logging.getLogger(__name__)

_MAX_ACTIVE_TRANSPORT_WORKERS = 8
_transport_worker_slots = threading.BoundedSemaphore(_MAX_ACTIVE_TRANSPORT_WORKERS)

ApprovalChoice = Literal["once", "session", "always", "deny"]
ApprovalPresentFn = Callable[
    ["ApprovalRequest"], "ApprovalDecision | Awaitable[ApprovalDecision]"
]


@dataclass(frozen=True)
class ApprovalDecision:
    """A transport response bound to one exact host-created request."""

    request_id: str
    request_digest: str
    choice: str


@dataclass(frozen=True)
class ApprovalRequest:
    """Immutable, display-only approval request passed to a transport plugin."""

    schema_version: int
    request_id: str
    digest: str
    command: str
    description: str
    pattern_key: str
    pattern_keys: tuple[str, ...]
    surface: str
    timeout_seconds: float
    allowed_choices: tuple[ApprovalChoice, ...]

    @classmethod
    def create(
        cls,
        *,
        command: str,
        description: str,
        pattern_key: str,
        pattern_keys: tuple[str, ...],
        session_key: str,
        surface: str,
        allow_session: bool,
        allow_permanent: bool,
        timeout_seconds: float = 300,
    ) -> "ApprovalRequest":
        request_id = uuid.uuid4().hex
        choices: list[ApprovalChoice] = ["once"]
        if allow_session:
            choices.append("session")
        if allow_permanent:
            choices.append("always")
        choices.append("deny")
        canonical = {
            "schema_version": 1,
            "request_id": request_id,
            "command": command,
            "description": description,
            "pattern_key": pattern_key,
            "pattern_keys": list(pattern_keys),
            "session_key": session_key,
            "surface": surface,
            "timeout_seconds": timeout_seconds,
            "allowed_choices": choices,
        }
        digest = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return cls(
            schema_version=1,
            request_id=request_id,
            digest=digest,
            command=command,
            description=description,
            pattern_key=pattern_key,
            pattern_keys=pattern_keys,
            surface=surface,
            timeout_seconds=timeout_seconds,
            allowed_choices=tuple(choices),
        )

    def respond(self, choice: ApprovalChoice | str) -> ApprovalDecision:
        """Build the correlated response a transport should return."""
        return ApprovalDecision(
            request_id=self.request_id,
            request_digest=self.digest,
            choice=choice,
        )


@dataclass(frozen=True)
class ApprovalTransportResult:
    """Normalized host result. Any failure is represented as a denial."""

    choice: ApprovalChoice
    failure: str | None = None


@dataclass(frozen=True)
class RegisteredApprovalTransport:
    """Plugin-owned registration retained by one profile's PluginManager."""

    name: str
    present: ApprovalPresentFn
    plugin_id: str
    profile_home: str


def invoke_approval_transport(
    present: ApprovalPresentFn,
    request: ApprovalRequest,
    *,
    timeout_seconds: float,
    poll_interval: float = 1.0,
    on_poll: Callable[[], None] | None = None,
    is_interrupted: Callable[[], bool] | None = None,
) -> ApprovalTransportResult:
    """Run a sync or async transport on a bounded daemon worker.

    Async callbacks are awaited with ``asyncio.run`` on that worker, never on a
    gateway or TUI event loop. A callback must return before the host timeout;
    late results are discarded and cannot authorize another request.
    """

    if not _transport_worker_slots.acquire(blocking=False):
        logger.warning("Approval transport worker capacity exhausted")
        return ApprovalTransportResult("deny", "busy")

    results: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)

    async def _await_value(value):
        return await value

    def _run() -> None:
        try:
            value = present(request)
            if inspect.isawaitable(value):
                value = asyncio.run(_await_value(value))
            results.put_nowait(("result", value))
        except BaseException as exc:  # fail closed even for unusual callback exits
            try:
                results.put_nowait(("error", exc))
            except queue.Full:
                pass
        finally:
            _transport_worker_slots.release()

    worker = threading.Thread(
        target=_run,
        name=f"approval-transport-{request.request_id[:8]}",
        daemon=True,
    )
    try:
        worker.start()
    except BaseException:
        _transport_worker_slots.release()
        logger.warning("Could not start approval transport worker")
        return ApprovalTransportResult("deny", "error")
    deadline = time.monotonic() + max(float(timeout_seconds), 0.0)
    while True:
        if is_interrupted is not None and is_interrupted():
            logger.info("Approval transport wait interrupted for %s", request.request_id)
            return ApprovalTransportResult("deny", "interrupted")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning("Approval transport timed out for request %s", request.request_id)
            return ApprovalTransportResult("deny", "timeout")
        try:
            kind, value = results.get(
                timeout=min(max(float(poll_interval), 0.001), remaining)
            )
            break
        except queue.Empty:
            if on_poll is not None:
                try:
                    on_poll()
                except Exception:
                    logger.debug("Approval transport poll callback failed", exc_info=True)

    if kind == "error":
        logger.warning("Approval transport failed for request %s", request.request_id)
        return ApprovalTransportResult("deny", "error")
    if not isinstance(value, ApprovalDecision):
        logger.warning("Approval transport returned an invalid decision type")
        return ApprovalTransportResult("deny", "invalid")
    if value.request_id != request.request_id or value.request_digest != request.digest:
        logger.warning("Approval transport returned a stale or mismatched decision")
        return ApprovalTransportResult("deny", "stale")
    if value.choice not in request.allowed_choices:
        logger.warning("Approval transport returned a disallowed choice")
        return ApprovalTransportResult("deny", "invalid")
    return ApprovalTransportResult(value.choice)
