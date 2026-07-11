"""Durable per-session SELF restart requests.

Plain resume requests are one-shot dropbox messages. Deferred restarts are a
separate typed lifecycle: the initiating turn arms its own request, one task
wins a gateway-wide mkdir CAS, and the next boot is the sole terminal consumer.
A later boot never signals; that boot already satisfied the restart intent.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from gateway.resume_requests import dropbox_dir

logger = logging.getLogger(__name__)

DEFERRED_RESTART_KIND = "deferred_restart"
REQUEST_STATES = frozenset(
    {
        "submitted",
        "armed",
        "claimed",
        "coalesce_pending",
        "consumed",
        "rejected",
    }
)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def _state_from_name(path: Path) -> str:
    parts = path.name.split(".")
    if len(parts) < 3 or parts[-1] != "json":
        raise ValueError(f"invalid deferred restart filename: {path.name}")
    state = parts[-2]
    if state not in REQUEST_STATES:
        raise ValueError(f"invalid deferred restart state: {state}")
    return state


@dataclass(frozen=True)
class DeferredRestartRequest:
    request_id: str
    session_key: str
    handoff: str
    boot_id: str
    intent_ts: float
    state: str
    path: Path

    @classmethod
    def load(cls, path: Path) -> "DeferredRestartRequest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("kind") != DEFERRED_RESTART_KIND:
            raise ValueError("not a deferred restart request")
        state = _state_from_name(Path(path))
        return cls(
            request_id=str(payload["request_id"]),
            session_key=str(payload["session_key"]),
            handoff=str(payload.get("handoff") or ""),
            boot_id=str(payload["boot_id"]),
            intent_ts=float(payload["intent_ts"]),
            state=state,
            path=Path(path),
        )

    def payload(self, *, state: str | None = None) -> dict[str, Any]:
        return {
            "kind": DEFERRED_RESTART_KIND,
            "request_id": self.request_id,
            "session_key": self.session_key,
            "handoff": self.handoff,
            "boot_id": self.boot_id,
            "intent_ts": self.intent_ts,
            "state": state or self.state,
        }


def submit_deferred_restart(
    hermes_home: Path,
    *,
    session_key: str,
    handoff: str,
    boot_id: str,
    intent_ts: float | None = None,
    request_id: str | None = None,
) -> DeferredRestartRequest:
    """Atomically publish a typed deferred SELF restart request."""
    if not session_key:
        raise ValueError("session_key is required")
    if not boot_id:
        raise ValueError("boot_id is required")
    rid = str(request_id or uuid.uuid4().hex)
    if not rid or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in rid):
        raise ValueError("request_id contains unsafe characters")
    directory = dropbox_dir(Path(hermes_home))
    path = directory / f"{rid}.submitted.json"
    request = DeferredRestartRequest(
        request_id=rid,
        session_key=str(session_key),
        handoff=str(handoff or ""),
        boot_id=str(boot_id),
        # Persist wall time: monotonic epochs reset across host reboots, while
        # deferred requests intentionally survive both process and host boots.
        intent_ts=float(time.time() if intent_ts is None else intent_ts),
        state="submitted",
        path=path,
    )
    _atomic_write_json(path, request.payload())
    return request


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


class DeferredRestartCoordinator:
    """Own same-boot arm/election while boot reconciliation owns deletion."""

    def __init__(self, hermes_home: Path, *, boot_id: str) -> None:
        self.hermes_home = Path(hermes_home)
        self.boot_id = str(boot_id)
        self.requests_dir = dropbox_dir(self.hermes_home)
        self.leader_dir = self.hermes_home / "gateway" / ".restart-leader.d"
        self._scheduled: dict[str, asyncio.Task] = {}
        self._owned_requests: dict[str, DeferredRestartRequest] = {}
        self._delivery_ready: set[str] = set()

    def scan(self) -> list[DeferredRestartRequest]:
        try:
            paths = sorted(self.requests_dir.glob("*.json"))
        except OSError:
            return []
        requests: list[DeferredRestartRequest] = []
        for path in paths:
            try:
                requests.append(DeferredRestartRequest.load(path))
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
        return requests

    def transition(
        self, request: DeferredRestartRequest, state: str
    ) -> DeferredRestartRequest:
        """CAS by rename, then atomically refresh the redundant payload state."""
        if state not in REQUEST_STATES:
            raise ValueError(f"invalid deferred restart state: {state}")
        if request.state == state:
            return request
        destination = request.path.with_name(f"{request.request_id}.{state}.json")
        os.replace(request.path, destination)
        updated = replace(request, state=state, path=destination)
        if state in {"armed", "claimed", "coalesce_pending"}:
            self._owned_requests[updated.session_key] = updated
        else:
            self._owned_requests.pop(updated.session_key, None)
        try:
            _atomic_write_json(destination, updated.payload())
        except OSError:
            # The filename is the authoritative lifecycle state. Once the CAS
            # rename succeeds, a redundant payload refresh must not orphan the
            # durable request before its in-memory owner can be scheduled.
            logger.warning(
                "Deferred restart payload refresh failed after transition to %s; "
                "filename state remains authoritative for %s",
                state,
                request.request_id,
                exc_info=True,
            )
        return updated

    def reject(self, request: DeferredRestartRequest, reason: str) -> DeferredRestartRequest:
        rejected = self.transition(request, "rejected")
        payload = rejected.payload()
        payload["rejected_reason"] = str(reason)
        _atomic_write_json(rejected.path, payload)
        logger.warning(
            "Deferred restart request %s rejected: %s", request.request_id, reason
        )
        return rejected

    def arm_for_session(
        self,
        session_key: str,
        *,
        consume_breadcrumb: Callable[[str], bool],
    ) -> str:
        """Validate and atomically arm this session's submitted request."""
        matching = [
            request
            for request in self.scan()
            if request.state == "submitted" and request.session_key == session_key
        ]
        if not matching:
            return "absent"
        request = matching[0]
        if request.boot_id != self.boot_id:
            self.reject(request, "stale boot_id")
            return "rejected"
        try:
            breadcrumb_ok = bool(consume_breadcrumb(session_key))
        except Exception:
            breadcrumb_ok = False
        if not breadcrumb_ok:
            self.reject(request, "missing, mismatched, or replayed breadcrumb")
            return "rejected"
        try:
            armed = self.transition(request, "armed")
        except FileNotFoundError:
            return "lost"
        self._owned_requests[session_key] = armed
        return "armed"

    def schedule_armed(
        self,
        session_key: str,
        *,
        delivery_event: asyncio.Event,
        delivery_timeout: float,
        record_replay: Callable[[DeferredRestartRequest], Any],
        mark_self: Callable[[DeferredRestartRequest], Any],
        signal_restart: Callable[[], Any],
        checkpoint: Callable[[str, DeferredRestartRequest], Any] | None = None,
    ) -> asyncio.Task:
        existing = self._scheduled.get(session_key)
        if existing is not None and not existing.done():
            return existing
        request = self._owned_requests.get(session_key)
        if request is None or request.state not in {"armed", "claimed"}:
            request = next(
                (
                    item
                    for item in self.scan()
                    if item.session_key == session_key
                    and item.state in {"armed", "claimed"}
                ),
                None,
            )
        if request is None:
            raise LookupError(f"no armed deferred restart for {session_key}")
        self._owned_requests[session_key] = request
        task = asyncio.create_task(
            self._run_armed(
                request,
                delivery_event=delivery_event,
                delivery_timeout=delivery_timeout,
                record_replay=record_replay,
                mark_self=mark_self,
                signal_restart=signal_restart,
                checkpoint=checkpoint,
            )
        )
        self._scheduled[session_key] = task
        return task

    def _publish_leader_meta(
        self,
        request: DeferredRestartRequest,
        *,
        committed: bool,
        commit_ts: float | None = None,
    ) -> None:
        _atomic_write_json(
            self.leader_dir / "meta.json",
            {
                "request_id": request.request_id,
                "epoch": self.boot_id,
                "pid": os.getpid(),
                "boot_id": self.boot_id,
                "committed": bool(committed),
                "commit_ts": commit_ts,
            },
        )

    def _read_leader_meta(self) -> dict[str, Any] | None:
        try:
            payload = json.loads((self.leader_dir / "meta.json").read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    async def _run_armed(
        self,
        request: DeferredRestartRequest,
        *,
        delivery_event: asyncio.Event,
        delivery_timeout: float,
        record_replay: Callable[[DeferredRestartRequest], Any],
        mark_self: Callable[[DeferredRestartRequest], Any],
        signal_restart: Callable[[], Any],
        checkpoint: Callable[[str, DeferredRestartRequest], Any] | None,
    ) -> str:
        try:
            await asyncio.wait_for(delivery_event.wait(), timeout=max(0.0, delivery_timeout))
        except asyncio.TimeoutError:
            logger.warning(
                "SELF restart delivery barrier timed out for %s; delivery state is UNKNOWN "
                "and the final response may be lost",
                request.session_key,
            )
        self._delivery_ready.add(request.request_id)

        retry_delay = 0.01
        while True:
            try:
                os.mkdir(self.leader_dir)
                winner = True
            except FileExistsError:
                winner = False
            if winner:
                try:
                    return await self._run_as_leader(
                        request,
                        record_replay=record_replay,
                        mark_self=mark_self,
                        signal_restart=signal_restart,
                        checkpoint=checkpoint,
                    )
                except asyncio.CancelledError:
                    task = asyncio.current_task()
                    if task is not None and task.cancelling():
                        raise
                    # A checkpoint/fallible callback may inject cancellation
                    # without cancelling the owning task. Keep durable work
                    # owned in the one-request case instead of requiring an
                    # incidental second initiator to re-elect.
                except Exception:
                    logger.exception(
                        "Deferred SELF restart leader failed before commit; retrying %s",
                        request.request_id,
                    )

                meta = self._read_leader_meta()
                if meta and meta.get("committed") is True:
                    raise RuntimeError(
                        "deferred restart signal failed after actuation commit"
                    )
                retryable = self._owned_requests.get(request.session_key)
                if retryable is None or retryable.state not in {"armed", "claimed"}:
                    retryable = next(
                        (
                            candidate
                            for candidate in self.scan()
                            if candidate.session_key == request.session_key
                            and candidate.state in {"armed", "claimed"}
                        ),
                        None,
                    )
                if retryable is None:
                    raise LookupError(
                        "deferred restart request disappeared during retry: "
                        f"{request.request_id}"
                    )
                request = retryable
                await asyncio.sleep(retry_delay)
                retry_delay = min(1.0, retry_delay * 2.0)
                continue

            meta = self._read_leader_meta()
            leader_boot = meta.get("boot_id") if meta else None
            if (
                isinstance(leader_boot, str)
                and leader_boot
                and leader_boot != self.boot_id
            ):
                # Epoch ownership, never PID liveness, proves this latch stale.
                shutil.rmtree(self.leader_dir, ignore_errors=True)
                if self.leader_dir.exists():
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(1.0, retry_delay * 2.0)
                else:
                    retry_delay = 0.01
                continue
            commit_ts: float | None = None
            if meta and meta.get("committed") is True:
                raw_commit_ts = meta.get("commit_ts")
                if (
                    isinstance(raw_commit_ts, (int, float))
                    and not isinstance(raw_commit_ts, bool)
                    and math.isfinite(float(raw_commit_ts))
                    and float(raw_commit_ts) > 0.0
                ):
                    commit_ts = float(raw_commit_ts)
            if commit_ts is not None:
                if request.intent_ts <= commit_ts:
                    await _maybe_await(record_replay(request))
                    await _maybe_await(mark_self(request))
                    current = DeferredRestartRequest.load(request.path)
                    self.transition(current, "coalesce_pending")
                    return "coalesced"
                return "pending_next_boot"

            # committed:false, missing, or torn metadata is never a coalescence
            # trigger. Stay armed and retry after the winner releases the latch.
            await asyncio.sleep(retry_delay)
            retry_delay = min(1.0, retry_delay * 2.0)

    async def _run_as_leader(
        self,
        request: DeferredRestartRequest,
        *,
        record_replay: Callable[[DeferredRestartRequest], Any],
        mark_self: Callable[[DeferredRestartRequest], Any],
        signal_restart: Callable[[], Any],
        checkpoint: Callable[[str, DeferredRestartRequest], Any] | None,
    ) -> str:
        committed = False
        current = request
        try:
            self._publish_leader_meta(current, committed=False)
            if checkpoint:
                checkpoint("after_meta_publish", current)

            current = self.transition(current, "claimed")
            if checkpoint:
                checkpoint("after_claim", current)

            await _maybe_await(record_replay(current))
            if checkpoint:
                checkpoint("after_replay_mark", current)

            await _maybe_await(mark_self(current))
            if checkpoint:
                checkpoint("after_self_mark", current)

            while True:
                pending_delivery_ids = {
                    candidate.request_id
                    for candidate in [
                        *self._owned_requests.values(),
                        *self.scan(),
                    ]
                    if candidate.boot_id == self.boot_id
                    and candidate.state in {"armed", "claimed"}
                } - self._delivery_ready
                if not pending_delivery_ids:
                    break
                await asyncio.sleep(0.01)
            if checkpoint:
                checkpoint("after_peer_deliveries", current)

            if checkpoint:
                checkpoint("before_commit_publish", current)
            commit_ts = time.time()
            self._publish_leader_meta(current, committed=True, commit_ts=commit_ts)
            committed = True
            signal_restart()
            return "signaled"
        finally:
            if not committed:
                shutil.rmtree(self.leader_dir, ignore_errors=True)


def reconcile_deferred_restarts_at_boot(
    hermes_home: Path,
    *,
    current_boot_id: str,
    boot_started_at: float,
    has_durable_mark: Callable[[DeferredRestartRequest], bool],
    record_replay: Callable[[DeferredRestartRequest], Any],
    mark_in_memory: Callable[[DeferredRestartRequest], Any],
    flush_sessions: Callable[[], Any],
    signal_restart: Callable[[], Any],
    checkpoint: Callable[[str, DeferredRestartRequest], Any] | None = None,
    session_exists: Callable[[DeferredRestartRequest], bool] | None = None,
) -> int:
    """Reconcile survivors before scheduling; never signal in a later boot."""
    del signal_restart  # Cross-boot authority is intentionally absent.
    coordinator = DeferredRestartCoordinator(hermes_home, boot_id=current_boot_id)
    reconciled = 0
    for request in coordinator.scan():
        if request.state in {"consumed", "rejected"}:
            try:
                request.path.unlink()
            except FileNotFoundError:
                pass
            continue
        if boot_started_at < request.intent_ts:
            coordinator.reject(request, "boot start predates intent_ts")
            continue
        if session_exists is not None and not session_exists(request):
            coordinator.reject(request, "session does not exist")
            continue

        replay_result = record_replay(request)
        if inspect.isawaitable(replay_result):
            raise TypeError("boot reconciliation callbacks must be synchronous")

        if not has_durable_mark(request):
            mark_result = mark_in_memory(request)
            if inspect.isawaitable(mark_result):
                raise TypeError("boot reconciliation callbacks must be synchronous")
            if checkpoint:
                checkpoint("after_boot_mark_before_flush", request)
            flush_result = flush_sessions()
            if inspect.isawaitable(flush_result):
                raise TypeError("boot reconciliation callbacks must be synchronous")
            if checkpoint:
                checkpoint("after_boot_flush_before_consume", request)

        consumed = coordinator.transition(request, "consumed")
        if checkpoint:
            checkpoint("after_boot_consume_before_unlink", consumed)
        consumed.path.unlink(missing_ok=True)
        reconciled += 1

    # A leader latch belongs to one boot only. Remove stale/uncommitted state;
    # committed state is no longer needed after every request was reconciled.
    meta = coordinator._read_leader_meta()
    if coordinator.leader_dir.exists() and (
        not meta or meta.get("boot_id") != current_boot_id
    ):
        shutil.rmtree(coordinator.leader_dir, ignore_errors=True)
    return reconciled
