"""Durable plugin API for external background-task lifecycle.

Plugin-facing service exposed lazily as ``PluginContext.background_tasks``.
It lets a trusted plugin register an EXTERNAL background task (a run owned by
a separate system — a detached Code Agent run, a remote job, a queued worker),
immediately hand the parent agent an unguessable handle, and later complete,
fail, or request cancellation of that task through the SAME durable
parent-session completion delivery Hermes already uses for
``delegate_task(background=True)``.

Design rules honored here (see the goal brief):

- **Parent/session binding is host-owned.** Registration binds ONLY to the
  currently active host parent (``agent.host_context``), never to platform /
  chat / session values supplied by the plugin. The method signature has no
  session parameters at all, so the active parent cannot be forged.
- **Plugin ownership.** Every handle is bound to the registering plugin's
  identity (captured from ``PluginContext``). A plugin may list, complete,
  fail, or cancel only handles it owns; cross-plugin access is rejected
  without leaking whether another plugin owns a handle.
- **Unguessable, tamper-evident handles.** Each handle carries an opaque
  ``task_id`` plus an HMAC signature computed with a persisted profile-local
  key (``state.db`` ``state_meta``), so handles stay valid across process
  restarts and cannot be forged by a plugin that never sees the key.
- **Durability + idempotency.** Registrations and terminal completion intent
  live in a profile-local ``external_background_tasks`` table in ``state.db``
  (the same profile-local state SQLite file the async-delegation registry
  uses; see ``agent.background_tasks_store``). Registration is idempotent by
  (plugin, active parent, request key, canonical payload hash); conflicting
  replays fail. Terminal events carry a caller ``event_id`` and are
  idempotent. Terminal transitions are atomic and single: a completed task
  cannot fail later or emit a second parent completion.
- **Delivery reuses the existing rail.** The terminal completion is written as
  a normal async-delegation row (``tools.async_delegation``
  ``insert_external_completion_row``) and enqueued onto the shared
  ``process_registry.completion_queue``, so the CLI / gateway / TUI drains,
  the claim / ack / release / retry machinery, and restart recovery all apply
  unchanged. No second completion rail exists.

The service never exposes private ``tools.async_delegation`` internals as its
public API; it returns stable frozen dataclasses with ``to_dict()`` JSON-safe
mappings (see ``agent.background_tasks_types``).
"""

from __future__ import annotations

import hmac
import json
import logging
import secrets
import time
import uuid
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.background_tasks_store import (
    _DB_LOCK,
    canonical_hash,
    load_or_create_hmac_key,
    row_to_dict,
    terminal_hash,
    transaction,
)
from agent.background_tasks_types import (
    MAX_ERROR_CHARS,
    MAX_EVENT_ID_CHARS,
    MAX_EXTERNAL_ID_CHARS,
    MAX_IDEMPOTENCY_KEY_CHARS,
    MAX_LABEL_CHARS,
    MAX_PAYLOAD_BYTES,
    MAX_SUMMARY_CHARS,
    PUBLIC_CONTRACT_VERSION,
    BackgroundTaskError,
    ExternalTaskHandle,
    ExternalTaskResult,
    ExternalTaskState,
    ExternalTaskStatus,
    coerce_handle,
    handle_from_row,
    sign as _sign_handle,
)
from tools.async_delegation import (
    capture_parent_session_routing,
    enqueue_completion_event,
    insert_external_completion_row,
)

logger = logging.getLogger(__name__)

# Re-export the public contract so `from agent.background_tasks import
# ExternalTaskHandle, ...` keeps working.
__all__ = [
    "PUBLIC_CONTRACT_VERSION",
    "BackgroundTaskError",
    "ExternalTaskHandle",
    "ExternalTaskResult",
    "ExternalTaskState",
    "ExternalTaskStatus",
]


def _sign(
    key: bytes, task_id: str, plugin_id: str, parent_session_id: str, created_at: float
) -> str:
    return _sign_handle(key, task_id, plugin_id, parent_session_id, created_at)


_HOST_SERVICE_CAPABILITY = object()


def _create_external_background_tasks_service(
    *, plugin_id: str, parent_agent_resolver: Callable[[], Any]
) -> "_ExternalBackgroundTasksService":
    """Host-private constructor used by :class:`PluginContext` and core tests."""

    return _ExternalBackgroundTasksService(
        plugin_id=plugin_id,
        parent_agent_resolver=parent_agent_resolver,
        _host_capability=_HOST_SERVICE_CAPABILITY,
    )


class _ExternalBackgroundTasksService:
    """Stable public service returned by :attr:`PluginContext.background_tasks`.

    ``plugin_id`` is captured from the ``PluginContext`` manifest (never passed
    by the caller per call); ``parent_agent_resolver`` is the host-owned
    ``agent.host_context.get_active_host_parent``.
    """

    def __init__(
        self,
        plugin_id: str,
        parent_agent_resolver: Callable[[], Any],
        *,
        _host_capability: object | None = None,
    ) -> None:
        if _host_capability is not _HOST_SERVICE_CAPABILITY:
            raise BackgroundTaskError(
                "External background-task services are host-owned; use PluginContext.background_tasks."
            )
        if not isinstance(plugin_id, str) or not plugin_id:
            raise BackgroundTaskError("plugin_id must be a non-empty string.")
        if not callable(parent_agent_resolver):
            raise BackgroundTaskError("parent_agent_resolver must be callable.")
        self._plugin_id = plugin_id
        self._parent_agent_resolver = parent_agent_resolver

    @property
    def plugin_id(self) -> str:
        """Plugin identity captured at construction and immutable via the public API."""

        return self._plugin_id

    # -- registration -------------------------------------------------------

    def register_external(
        self,
        *,
        external_id: str,
        payload: Optional[Mapping[str, Any]] = None,
        idempotency_key: str = "",
        label: str = "",
    ) -> ExternalTaskHandle:
        """Register an external background task bound to the ACTIVE parent.

        The parent/session routing is captured from the currently active host
        parent supplied by Hermes; no session value is accepted from the
        caller. Returns immediately with an unguessable handle the plugin
        persists and later passes to ``complete`` / ``fail`` /
        ``request_cancel``.

        Idempotent by (plugin, active parent, ``idempotency_key`` or
        ``external_id``) + canonical payload hash: a replay returns the same
        handle; the same key with a different payload raises
        :class:`BackgroundTaskError`.
        """
        parent = self._parent_agent_resolver()
        if parent is None:
            raise BackgroundTaskError(
                "No active Hermes parent session is available. register_external "
                "must be called while an agent turn is running so Hermes can "
                "bind the task to the ACTIVE parent session; it cannot accept "
                "session values from the caller."
            )
        self._validate_register_inputs(external_id, payload, idempotency_key, label)
        parent_session_id = str(getattr(parent, "session_id", "") or "")
        if not parent_session_id:
            raise BackgroundTaskError(
                "The active Hermes parent session has no durable session id; "
                "registration is not possible."
            )
        routing = capture_parent_session_routing(parent)
        dedup_key = idempotency_key or external_id
        payload_hash = canonical_hash(dict(payload) if payload is not None else {})
        now = time.time()
        task_id = secrets.token_hex(16)
        with _DB_LOCK, transaction() as conn:
            key = load_or_create_hmac_key(conn)
            signature = _sign(key, task_id, self.plugin_id, parent_session_id, now)
            cur = conn.execute(
                """INSERT OR IGNORE INTO external_background_tasks
                   (task_id, plugin_id, parent_session_id, session_key,
                    origin_ui_session_id, origin_session_id, external_id,
                    idempotency_key, label, payload_hash, payload_json,
                    state, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'registered', ?, ?)""",
                (
                    task_id,
                    self.plugin_id,
                    parent_session_id,
                    routing["session_key"],
                    routing["origin_ui_session_id"],
                    routing["origin_session_id"],
                    external_id,
                    dedup_key,
                    label,
                    payload_hash,
                    json.dumps(dict(payload)) if payload is not None else "{}",
                    now,
                    now,
                ),
            )
            if cur.rowcount == 0:
                row = conn.execute(
                    """SELECT * FROM external_background_tasks
                       WHERE plugin_id=? AND parent_session_id=? AND idempotency_key=?""",
                    (self.plugin_id, parent_session_id, dedup_key),
                ).fetchone()
                if row is None:
                    raise BackgroundTaskError(
                        "Conflicting registration: the idempotency key was "
                        "taken by another task in this parent session."
                    )
                existing = row_to_dict(row)
                if existing["payload_hash"] != payload_hash:
                    raise BackgroundTaskError(
                        "Conflicting replay: the same idempotency key was "
                        "already registered with a different payload."
                    )
                logger.info(
                    "External background task %s re-registered idempotently "
                    "(plugin=%s, external_id=%s)",
                    existing["task_id"][:12],
                    self.plugin_id,
                    external_id,
                )
                return handle_from_row(key, existing)
        logger.info(
            "Registered external background task %s (plugin=%s, external_id=%s)",
            task_id[:12],
            self.plugin_id,
            external_id,
        )
        return ExternalTaskHandle(
            PUBLIC_CONTRACT_VERSION,
            task_id,
            self.plugin_id,
            parent_session_id,
            now,
            signature,
        )

    # -- terminal transitions -----------------------------------------------

    def complete(
        self,
        handle: Any,
        *,
        event_id: str,
        summary: str = "",
        result_payload: Optional[Mapping[str, Any]] = None,
    ) -> ExternalTaskResult:
        """Record a successful terminal completion and deliver it to the parent.

        Idempotent by (event_id, payload): a duplicate event with the same
        payload is harmless; the same event_id with a different payload
        conflicts. Emits at most one parent completion.
        """
        self._validate_terminal_inputs(
            event_id=event_id,
            summary=summary,
            error=None,
            result_payload=result_payload,
        )
        resolved = self._resolve_handle(handle)
        if resolved is None:
            return ExternalTaskResult(unknown_handle=True)
        row, coerced = resolved
        payload_hash = terminal_hash(
            ExternalTaskState.COMPLETED.value, event_id, summary, None, result_payload
        )
        return self._transition_terminal(
            row,
            coerced,
            ExternalTaskState.COMPLETED.value,
            event_id,
            payload_hash,
            summary=summary,
            error=None,
            result_payload=result_payload,
        )

    def fail(
        self,
        handle: Any,
        *,
        event_id: str,
        error: str = "",
    ) -> ExternalTaskResult:
        """Record a terminal failure and deliver it to the parent.

        Same idempotency contract as :meth:`complete`.
        """
        self._validate_terminal_inputs(
            event_id=event_id, summary=None, error=error, result_payload=None
        )
        resolved = self._resolve_handle(handle)
        if resolved is None:
            return ExternalTaskResult(unknown_handle=True)
        row, coerced = resolved
        payload_hash = terminal_hash(
            ExternalTaskState.FAILED.value, event_id, None, error, None
        )
        return self._transition_terminal(
            row,
            coerced,
            ExternalTaskState.FAILED.value,
            event_id,
            payload_hash,
            summary=None,
            error=error,
            result_payload=None,
        )

    def request_cancel(self, handle: Any) -> ExternalTaskResult:
        """Record durable cancellation INTENT for a task.

        Only records the intent; it does NOT claim the external system was
        cancelled. The plugin/consumer still performs the actual external
        cancellation and later reports the real terminal state via
        ``complete`` or ``fail``. Idempotent.
        """
        resolved = self._resolve_handle(handle)
        if resolved is None:
            return ExternalTaskResult(unknown_handle=True)
        row, coerced = resolved
        task_id = row["task_id"]
        if row["state"] in (
            ExternalTaskState.COMPLETED.value,
            ExternalTaskState.FAILED.value,
        ):
            return ExternalTaskResult(
                handle=coerced,
                accepted=False,
                already_terminal=True,
                state=row["state"],
            )
        now = time.time()
        with _DB_LOCK, transaction() as conn:
            cur = conn.execute(
                """UPDATE external_background_tasks
                   SET state='cancel_requested', updated_at=?,
                       cancel_requested_at=COALESCE(cancel_requested_at, ?)
                   WHERE task_id=? AND state='registered'""",
                (now, now, task_id),
            )
            already = cur.rowcount == 0
        if already:
            fresh = self._read_row(task_id)
            if fresh is not None and fresh["state"] in (
                ExternalTaskState.COMPLETED.value,
                ExternalTaskState.FAILED.value,
            ):
                return ExternalTaskResult(
                    handle=coerced,
                    accepted=False,
                    already_terminal=True,
                    state=fresh["state"],
                )
        return ExternalTaskResult(
            handle=coerced,
            accepted=True,
            state=ExternalTaskState.CANCEL_REQUESTED.value,
            cancel_already_requested=already,
        )

    # -- listing ------------------------------------------------------------

    def list_pending(self) -> List[ExternalTaskStatus]:
        """Return this plugin's non-terminal OR undelivered task handles.

        A terminal task stays listed until its completion has been delivered
        to the parent session (claimed and acknowledged by a drain consumer).
        After a process restart this returns the same handles, restored from
        durable profile-local state.
        """
        with _DB_LOCK, transaction() as conn:
            rows = [
                row_to_dict(r)
                for r in conn.execute(
                    "SELECT * FROM external_background_tasks WHERE plugin_id=?",
                    (self.plugin_id,),
                ).fetchall()
            ]
            key = load_or_create_hmac_key(conn)
        statuses: List[ExternalTaskStatus] = []
        for row in rows:
            if row["state"] not in (
                ExternalTaskState.REGISTERED.value,
                ExternalTaskState.CANCEL_REQUESTED.value,
            ):
                self._sync_delivery_state(row)
            if (
                row["state"]
                in (
                    ExternalTaskState.REGISTERED.value,
                    ExternalTaskState.CANCEL_REQUESTED.value,
                )
                or row["delivery_state"] != "delivered"
            ):
                statuses.append(
                    ExternalTaskStatus(
                        handle=handle_from_row(key, row),
                        state=ExternalTaskState(row["state"]),
                        delivery_state=row["delivery_state"],
                        external_id=row["external_id"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                        completed_at=row["completed_at"],
                    )
                )
        return statuses

    # -- internals ----------------------------------------------------------

    def _resolve_handle(
        self, value: Any
    ) -> Optional[Tuple[Dict[str, Any], ExternalTaskHandle]]:
        """Verify handle shape, plugin ownership, signature, and row existence.

        Returns ``None`` for anything that does not positively resolve —
        foreign-plugin handles, tampered signatures, unknown task ids — so
        callers never learn whether another plugin owns a handle.
        """
        handle = coerce_handle(value)
        if handle is None or handle.plugin_id != self.plugin_id:
            return None
        with _DB_LOCK, transaction() as conn:
            key = load_or_create_hmac_key(conn)
            row = conn.execute(
                "SELECT * FROM external_background_tasks WHERE task_id=?",
                (handle.task_id,),
            ).fetchone()
        if not hmac.compare_digest(
            handle.signature,
            _sign(
                key,
                handle.task_id,
                handle.plugin_id,
                handle.parent_session_id,
                handle.created_at,
            ),
        ):
            return None
        if row is None:
            return None
        data = row_to_dict(row)
        if (
            data["plugin_id"] != handle.plugin_id
            or data["parent_session_id"] != handle.parent_session_id
        ):
            return None
        return data, handle

    def _read_row(self, task_id: str) -> Optional[Dict[str, Any]]:
        with _DB_LOCK, transaction() as conn:
            row = conn.execute(
                "SELECT * FROM external_background_tasks WHERE task_id=?",
                (task_id,),
            ).fetchone()
        return row_to_dict(row) if row is not None else None

    def _transition_terminal(
        self,
        row: Dict[str, Any],
        handle: ExternalTaskHandle,
        status: str,
        event_id: str,
        payload_hash: str,
        *,
        summary: Optional[str],
        error: Optional[str],
        result_payload: Optional[Mapping[str, Any]],
    ) -> ExternalTaskResult:
        """Atomically flip the task to terminal and persist its delivery row.

        The task-state transition and the async-delegation delivery-row insert
        share one SQLite transaction, so a crash can never leave a terminal
        task without a durable completion (or vice versa). The completion is
        enqueued only AFTER commit; a failed enqueue is recovered by the
        durable restart-restore rail.
        """
        task_id = row["task_id"]
        now = time.time()
        delegation_id = f"ext_{uuid.uuid4().hex[:12]}"
        evt = self._build_completion_event(
            row, delegation_id, status, summary, error, result_payload, now
        )
        result = {
            "status": status,
            "summary": summary,
            "error": error,
            "result_payload": result_payload,
        }
        with _DB_LOCK, transaction() as conn:
            cur = conn.execute(
                """UPDATE external_background_tasks
                   SET state=?, completed_at=?, updated_at=?,
                       terminal_event_id=?, terminal_payload_hash=?,
                       summary=?, error=?, result_json=?,
                       delivery_delegation_id=?, delivery_state='pending'
                   WHERE task_id=? AND state IN ('registered','cancel_requested')""",
                (
                    status,
                    now,
                    now,
                    event_id,
                    payload_hash,
                    summary,
                    error,
                    json.dumps(result_payload) if result_payload is not None else None,
                    delegation_id,
                    task_id,
                ),
            )
            if cur.rowcount == 1:
                insert_external_completion_row(conn, evt, result)
                committed = True
            else:
                committed = False
        if committed:
            enqueue_completion_event(evt)
            return ExternalTaskResult(handle=handle, accepted=True, state=status)
        # Lost the terminal race (or already terminal) — resolve honestly.
        fresh = self._read_row(task_id)
        if fresh is None:
            return ExternalTaskResult(unknown_handle=True)
        if fresh["terminal_event_id"] == event_id:
            if fresh["terminal_payload_hash"] == payload_hash:
                return ExternalTaskResult(
                    handle=handle,
                    accepted=True,
                    already_terminal=True,
                    state=fresh["state"],
                )
            return ExternalTaskResult(
                handle=handle,
                accepted=False,
                conflict=True,
                state=fresh["state"],
                message="The same event id was already applied with a "
                "different payload.",
            )
        return ExternalTaskResult(
            handle=handle,
            accepted=False,
            already_terminal=True,
            state=fresh["state"],
            message="Task already reached a terminal state.",
        )

    def _build_completion_event(
        self,
        row: Dict[str, Any],
        delegation_id: str,
        status: str,
        summary: Optional[str],
        error: Optional[str],
        result_payload: Optional[Mapping[str, Any]],
        now: float,
    ) -> Dict[str, Any]:
        evt: Dict[str, Any] = {
            "type": "async_delegation",
            "delegation_id": delegation_id,
            "session_key": row["session_key"],
            "origin_ui_session_id": row["origin_ui_session_id"],
            "origin_session_id": row["origin_session_id"],
            "parent_session_id": row["parent_session_id"],
            "goal": row["label"] or f"external task {row['external_id']}",
            "status": status,
            "summary": summary,
            "error": error,
            "api_calls": 0,
            "duration_seconds": round(max(0.0, now - row["created_at"]), 2),
            "dispatched_at": row["created_at"],
            "completed_at": now,
            "external_background_task": True,
        }
        if result_payload is not None:
            evt["result_payload"] = result_payload
        return evt

    def _sync_delivery_state(self, row: Dict[str, Any]) -> None:
        """Refresh a terminal row's delivery state from the shared rail."""
        if row["delivery_state"] == "delivered":
            return
        delegation_id = row.get("delivery_delegation_id") or ""
        if not delegation_id:
            return
        from tools.async_delegation import get_durable_delegation

        info = get_durable_delegation(delegation_id)
        if info is None:
            # The async-delegation row was pruned after delivery (or after a
            # dropped terminal disposition) — treat as delivered.
            new_state = "delivered"
        else:
            new_state = info.get("delivery_state") or "pending"
        if new_state != row["delivery_state"]:
            with _DB_LOCK, transaction() as conn:
                conn.execute(
                    "UPDATE external_background_tasks SET delivery_state=?, "
                    "updated_at=? WHERE task_id=?",
                    (new_state, time.time(), row["task_id"]),
                )
            row["delivery_state"] = new_state

    @staticmethod
    def _validate_register_inputs(
        external_id: str,
        payload: Optional[Mapping[str, Any]],
        idempotency_key: str,
        label: str,
    ) -> None:
        if not isinstance(external_id, str) or not external_id.strip():
            raise BackgroundTaskError("external_id must be a non-empty string.")
        if len(external_id) > MAX_EXTERNAL_ID_CHARS:
            raise BackgroundTaskError(
                f"external_id exceeds {MAX_EXTERNAL_ID_CHARS} characters."
            )
        if not isinstance(idempotency_key, str):
            raise BackgroundTaskError("idempotency_key must be a string.")
        if len(idempotency_key) > MAX_IDEMPOTENCY_KEY_CHARS:
            raise BackgroundTaskError(
                f"idempotency_key exceeds {MAX_IDEMPOTENCY_KEY_CHARS} characters."
            )
        if not isinstance(label, str):
            raise BackgroundTaskError("label must be a string.")
        if len(label) > MAX_LABEL_CHARS:
            raise BackgroundTaskError(f"label exceeds {MAX_LABEL_CHARS} characters.")
        if payload is not None:
            if not isinstance(payload, Mapping):
                raise BackgroundTaskError("payload must be a JSON object (mapping).")
            try:
                raw = json.dumps(dict(payload), sort_keys=True, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise BackgroundTaskError("payload must be JSON-serializable.") from exc
            if len(raw.encode("utf-8")) > MAX_PAYLOAD_BYTES:
                raise BackgroundTaskError(f"payload exceeds {MAX_PAYLOAD_BYTES} bytes.")

    @staticmethod
    def _validate_terminal_inputs(
        *,
        event_id: str,
        summary: Optional[str],
        error: Optional[str],
        result_payload: Optional[Mapping[str, Any]],
    ) -> None:
        if not isinstance(event_id, str) or not event_id.strip():
            raise BackgroundTaskError("event_id must be a non-empty string.")
        if len(event_id) > MAX_EVENT_ID_CHARS:
            raise BackgroundTaskError(
                f"event_id exceeds {MAX_EVENT_ID_CHARS} characters."
            )
        if summary is not None:
            if not isinstance(summary, str):
                raise BackgroundTaskError("summary must be a string.")
            if len(summary) > MAX_SUMMARY_CHARS:
                raise BackgroundTaskError(
                    f"summary exceeds {MAX_SUMMARY_CHARS} characters."
                )
        if error is not None:
            if not isinstance(error, str):
                raise BackgroundTaskError("error must be a string.")
            if len(error) > MAX_ERROR_CHARS:
                raise BackgroundTaskError(
                    f"error exceeds {MAX_ERROR_CHARS} characters."
                )
        if result_payload is not None:
            if not isinstance(result_payload, Mapping):
                raise BackgroundTaskError(
                    "result_payload must be a JSON object (mapping)."
                )
            try:
                raw = json.dumps(dict(result_payload), sort_keys=True, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise BackgroundTaskError(
                    "result_payload must be JSON-serializable."
                ) from exc
            if len(raw.encode("utf-8")) > MAX_PAYLOAD_BYTES:
                raise BackgroundTaskError(
                    f"result_payload exceeds {MAX_PAYLOAD_BYTES} bytes."
                )
