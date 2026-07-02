"""In-memory run manager for the /v1/runs runtime API.

Isolated storage layer — can be swapped for a durable backend
without changing the route contract.
"""

import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from gateway.runtime.models import (
    RuntimeEvent,
    RuntimeStatus,
    RUN_STATUS_QUEUED,
    RUN_STATUS_AWAITING_APPROVAL,
    RUN_STATUS_AWAITING_CLARIFY,
    RUN_STATUS_CANCELLING,
    RUN_STATUS_CANCELLED,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    EVENT_RUN_STARTED,
    EVENT_RUN_STATUS,
    EVENT_APPROVAL_REQUESTED,
    EVENT_APPROVAL_RESOLVED,
    EVENT_CLARIFY_REQUESTED,
    EVENT_CLARIFY_RESOLVED,
    EVENT_DONE,
    EVENT_ERROR,
    TERMINAL_STATUSES,
    TERMINAL_EVENT_TYPES,
    redact_secrets,
)


class RunManager:
    """Manages run lifecycle: creation, events, status, stop/approval/clarify.

    Uses in-memory storage with threading locks.  The public API returns
    plain dicts so route handlers can serialize them directly without
    coupling to internal dataclass shapes.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._runs: Dict[str, RuntimeStatus] = {}
        self._events: Dict[str, List[RuntimeEvent]] = {}
        self._seq_counters: Dict[str, int] = {}

    def create_run(
        self,
        session_id: str,
        *,
        message: Optional[str] = None,
        workspace: Optional[str] = None,
        profile: Optional[str] = None,
        model: Optional[str] = None,
        toolsets: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            if run_id is None:
                run_id = f"run_{uuid.uuid4().hex}"
            now = time.time()

            status = RuntimeStatus(
                run_id=run_id,
                session_id=session_id,
                status=RUN_STATUS_QUEUED,
                controls=["observe", "stop"],
                created_at=now,
                updated_at=now,
            )
            self._runs[run_id] = status
            self._events[run_id] = []
            self._seq_counters[run_id] = 0

            self._append_event(
                run_id=run_id,
                session_id=session_id,
                event_type=EVENT_RUN_STARTED,
                payload={
                    "message": message,
                    "workspace": workspace,
                    "profile": profile,
                    "model": model,
                    "toolsets": toolsets,
                    "metadata": metadata or {},
                },
            )

            return {
                "run_id": run_id,
                "session_id": session_id,
                "status": status.status,
                "events_url": f"/v1/runs/{run_id}/events",
                "status_url": f"/v1/runs/{run_id}",
                "controls": status.controls,
            }

    def get_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            status = self._runs.get(run_id)
            if status is None:
                return None
            return status.to_dict()

    def append_event(
        self,
        run_id: str,
        event_type: str,
        *,
        session_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[RuntimeEvent]:
        with self._lock:
            if run_id not in self._runs:
                return None
            if session_id is None:
                session_id = self._runs[run_id].session_id
            return self._append_event(
                run_id=run_id,
                session_id=session_id,
                event_type=event_type,
                payload=payload,
            )

    def read_events(
        self,
        run_id: str,
        *,
        after_seq: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            if run_id not in self._runs:
                return None
            events = list(self._events.get(run_id, []))

        if after_seq is not None:
            events = [e for e in events if e.seq > after_seq]

        if limit is not None and limit > 0:
            events = events[:limit]

        return {
            "run_id": run_id,
            "events": [e.to_dict() for e in events],
        }

    def stop_run(self, run_id: str) -> Dict[str, Any]:
        with self._lock:
            status = self._runs.get(run_id)
            if status is None:
                return {
                    "error": "not_found",
                    "message": f"Run not found: {run_id}",
                }

            if status.terminal:
                return {
                    "run_id": run_id,
                    "status": status.status,
                    "message": f"Run is already in terminal state: {status.status}",
                    "terminal": True,
                    "controls": status.controls,
                }

            previous_status = status.status

            if previous_status in {RUN_STATUS_QUEUED, RUN_STATUS_CANCELLING}:
                status.status = RUN_STATUS_CANCELLED
                status.terminal = True
                status.controls = []
                status.updated_at = time.time()
                self._append_event(
                    run_id=run_id,
                    session_id=status.session_id,
                    event_type=EVENT_RUN_STATUS,
                    payload={"status": RUN_STATUS_CANCELLED, "previous": previous_status},
                )
                self._append_event(
                    run_id=run_id,
                    session_id=status.session_id,
                    event_type=EVENT_DONE,
                    payload={"status": RUN_STATUS_CANCELLED},
                )
            else:
                status.status = RUN_STATUS_CANCELLING
                status.controls = ["observe"]
                status.updated_at = time.time()
                self._append_event(
                    run_id=run_id,
                    session_id=status.session_id,
                    event_type=EVENT_RUN_STATUS,
                    payload={"status": RUN_STATUS_CANCELLING, "previous": previous_status},
                )
                status.status = RUN_STATUS_CANCELLED
                status.terminal = True
                status.controls = []
                status.updated_at = time.time()
                self._append_event(
                    run_id=run_id,
                    session_id=status.session_id,
                    event_type=EVENT_DONE,
                    payload={"status": RUN_STATUS_CANCELLED},
                )

            return {
                "run_id": run_id,
                "status": status.status,
                "terminal": status.terminal,
                "controls": status.controls,
            }

    def transition_status(self, run_id: str, new_status: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            status = self._runs.get(run_id)
            if status is None:
                return None
            previous = status.status
            status.status = new_status
            status.terminal = new_status in TERMINAL_STATUSES
            status.updated_at = time.time()
            self._append_event(
                run_id=run_id,
                session_id=status.session_id,
                event_type=EVENT_RUN_STATUS,
                payload={"status": new_status, "previous": previous},
            )
            return status.to_dict()

    def complete_run(self, run_id: str, *, result: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._lock:
            status = self._runs.get(run_id)
            if status is None:
                return None
            if status.terminal:
                return status.to_dict()
            status.status = RUN_STATUS_COMPLETED
            status.terminal = True
            status.result = result
            status.controls = []
            status.updated_at = time.time()
            self._append_event(
                run_id=run_id,
                session_id=status.session_id,
                event_type=EVENT_DONE,
                payload={"status": RUN_STATUS_COMPLETED, "result": result},
            )
            return status.to_dict()

    def fail_run(self, run_id: str, *, error: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._lock:
            status = self._runs.get(run_id)
            if status is None:
                return None
            if status.terminal:
                return status.to_dict()
            status.status = RUN_STATUS_FAILED
            status.terminal = True
            status.error = error
            status.controls = []
            status.updated_at = time.time()
            self._append_event(
                run_id=run_id,
                session_id=status.session_id,
                event_type=EVENT_ERROR,
                payload={"status": RUN_STATUS_FAILED, "error": error},
            )
            self._append_event(
                run_id=run_id,
                session_id=status.session_id,
                event_type=EVENT_DONE,
                payload={"status": RUN_STATUS_FAILED},
            )
            return status.to_dict()

    def request_approval(
        self,
        run_id: str,
        approval_id: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            status = self._runs.get(run_id)
            if status is None:
                return None
            if status.terminal:
                return None
            payload = payload or {}
            clean_payload = redact_secrets(dict(payload))
            status.pending_approval_ids.append(approval_id)
            status.status = RUN_STATUS_AWAITING_APPROVAL
            status.updated_at = time.time()
            self._append_event(
                run_id=run_id,
                session_id=status.session_id,
                event_type=EVENT_APPROVAL_REQUESTED,
                payload={
                    "approval_id": approval_id,
                    "payload": clean_payload,
                },
            )
            self._append_event(
                run_id=run_id,
                session_id=status.session_id,
                event_type=EVENT_RUN_STATUS,
                payload={
                    "status": RUN_STATUS_AWAITING_APPROVAL,
                    "previous": RUN_STATUS_QUEUED,
                },
            )
            return {
                "run_id": run_id,
                "approval_id": approval_id,
                "status": "requested",
            }

    def resolve_approval(
        self,
        run_id: str,
        approval_id: str,
        choice: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            status = self._runs.get(run_id)
            if status is None:
                return {
                    "error": "not_found",
                    "message": f"Run not found: {run_id}",
                }
            if approval_id not in status.pending_approval_ids:
                already = False
                events = self._events.get(run_id, [])
                for ev in events:
                    p = ev.payload
                    if isinstance(p, dict) and p.get("approval_id") == approval_id:
                        if ev.type in (EVENT_APPROVAL_RESOLVED,):
                            already = True
                            break
                if already:
                    return {
                        "error": "conflict",
                        "message": f"Approval {approval_id} has already been resolved.",
                    }
                return {
                    "error": "not_found",
                    "message": f"Approval {approval_id} not found for run {run_id}.",
                }
            if status.terminal:
                return {
                    "error": "conflict",
                    "message": f"Run {run_id} is in terminal state and cannot accept approval resolution.",
                }
            payload = payload or {}
            clean_payload = redact_secrets(dict(payload))
            status.pending_approval_ids.remove(approval_id)
            if not status.pending_approval_ids and not status.pending_clarify_ids:
                status.status = RUN_STATUS_QUEUED
            status.controls = list(status.controls)
            status.updated_at = time.time()
            event = self._append_event(
                run_id=run_id,
                session_id=status.session_id,
                event_type=EVENT_APPROVAL_RESOLVED,
                payload={
                    "approval_id": approval_id,
                    "choice": choice,
                    "payload": clean_payload,
                },
            )
            return {
                "run_id": run_id,
                "action_id": approval_id,
                "type": "approval",
                "status": "resolved",
                "event": event.to_dict(redact=True),
            }

    def request_clarify(
        self,
        run_id: str,
        clarify_id: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            status = self._runs.get(run_id)
            if status is None:
                return None
            if status.terminal:
                return None
            payload = payload or {}
            clean_payload = redact_secrets(dict(payload))
            status.pending_clarify_ids.append(clarify_id)
            status.status = RUN_STATUS_AWAITING_CLARIFY
            status.updated_at = time.time()
            self._append_event(
                run_id=run_id,
                session_id=status.session_id,
                event_type=EVENT_CLARIFY_REQUESTED,
                payload={
                    "clarify_id": clarify_id,
                    "payload": clean_payload,
                },
            )
            self._append_event(
                run_id=run_id,
                session_id=status.session_id,
                event_type=EVENT_RUN_STATUS,
                payload={
                    "status": RUN_STATUS_AWAITING_CLARIFY,
                    "previous": RUN_STATUS_QUEUED,
                },
            )
            return {
                "run_id": run_id,
                "clarify_id": clarify_id,
                "status": "requested",
            }

    def resolve_clarify(
        self,
        run_id: str,
        clarify_id: str,
        answer: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            status = self._runs.get(run_id)
            if status is None:
                return {
                    "error": "not_found",
                    "message": f"Run not found: {run_id}",
                }
            if clarify_id not in status.pending_clarify_ids:
                already = False
                events = self._events.get(run_id, [])
                for ev in events:
                    p = ev.payload
                    if isinstance(p, dict) and p.get("clarify_id") == clarify_id:
                        if ev.type in (EVENT_CLARIFY_RESOLVED,):
                            already = True
                            break
                if already:
                    return {
                        "error": "conflict",
                        "message": f"Clarify {clarify_id} has already been resolved.",
                    }
                return {
                    "error": "not_found",
                    "message": f"Clarify {clarify_id} not found for run {run_id}.",
                }
            if status.terminal:
                return {
                    "error": "conflict",
                    "message": f"Run {run_id} is in terminal state and cannot accept clarify resolution.",
                }
            payload = payload or {}
            clean_payload = redact_secrets(dict(payload))
            status.pending_clarify_ids.remove(clarify_id)
            if not status.pending_approval_ids and not status.pending_clarify_ids:
                status.status = RUN_STATUS_QUEUED
            status.controls = list(status.controls)
            status.updated_at = time.time()
            event = self._append_event(
                run_id=run_id,
                session_id=status.session_id,
                event_type=EVENT_CLARIFY_RESOLVED,
                payload={
                    "clarify_id": clarify_id,
                    "answer": answer,
                    "payload": clean_payload,
                },
            )
            return {
                "run_id": run_id,
                "action_id": clarify_id,
                "type": "clarify",
                "status": "resolved",
                "event": event.to_dict(redact=True),
            }

    def _append_event(
        self,
        run_id: str,
        session_id: str,
        event_type: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
    ) -> RuntimeEvent:
        seq = self._seq_counters.get(run_id, 0) + 1
        self._seq_counters[run_id] = seq

        terminal = event_type in TERMINAL_EVENT_TYPES

        event = RuntimeEvent(
            event_id=f"{run_id}:{seq}",
            seq=seq,
            run_id=run_id,
            session_id=session_id,
            type=event_type,
            terminal=terminal,
            payload=payload or {},
        )

        self._events.setdefault(run_id, []).append(event)

        status = self._runs.get(run_id)
        if status is not None:
            status.last_event_id = event.event_id
            status.last_seq = seq
            status.terminal = status.terminal or terminal
            status.updated_at = time.time()

        return event
