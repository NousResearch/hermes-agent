"""Durable ``/v1/runs`` admission, status, events, and control handlers."""

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

try:
    from aiohttp import web
    from aiohttp.web_request import RequestKey
except ImportError:
    web = None  # type: ignore[assignment]
    RequestKey = None  # type: ignore[assignment,misc]

from gateway.platforms.api_server_room_grants import _json_error, _room_grant_error_response
from gateway.platforms.api_server_run_idempotency import TERMINAL_STATUSES


logger = logging.getLogger("gateway.platforms.api_server")
_ROOM_RETENTION_REQUEST_KEY = (
    RequestKey("hermes.room_run_retention_until", float) if RequestKey is not None
    else "hermes.room_run_retention_until")
# Forwarded subagent lifecycle fields; free-text ones are secret-redacted.
_SUBAGENT_EVENT_KEYS = (
    "goal", "task_count", "task_index", "subagent_id", "child_session_id", "delegation_id", "parent_id",
    "depth", "model", "tool_count", "status", "summary", "duration_seconds", "input_tokens",
    "output_tokens", "reasoning_tokens", "api_calls", "cost_usd", "files_read", "files_written",
    "output_tail")
_SUBAGENT_TEXT_KEYS = ("goal", "summary", "output_tail")
# Terminal usage payload: (wire key, agent attribute), in wire order.
_USAGE_FIELDS = (
    ("input_tokens", "session_prompt_tokens"), ("output_tokens", "session_completion_tokens"),
    ("total_tokens", "session_total_tokens"))
# Tool-progress event -> SSE payload fields (tool_name, preview, kwargs); key order is wire format.
_FIXED_EVENT_FIELDS = {
    "tool.started": lambda tool, preview, kw: {"tool": tool, "preview": preview},
    "tool.completed": lambda tool, preview, kw: {
        "tool": tool, "duration": round(kw.get("duration", 0), 3), "error": kw.get("is_error", False)},
    "reasoning.available": lambda tool, preview, kw: {"text": preview or ""}}


def _remember_room_retention(request: "web.Request", claims: dict[str, Any]) -> None:
    value = float(claims.get("status_expires_at") or claims.get("expires_at") or 0)
    try:
        request[_ROOM_RETENTION_REQUEST_KEY] = value
    except (AttributeError, TypeError):
        setattr(request, "_hermes_room_run_retention_until", value)


def _room_retention_until(request: "web.Request") -> float:
    try:
        value = request.get(_ROOM_RETENTION_REQUEST_KEY, 0)
    except AttributeError:
        value = getattr(request, "_hermes_room_run_retention_until", 0)
    return max(0.0, float(value or 0))


def _run_event(run_id: str, name: str, **fields: Any) -> Dict[str, Any]:
    """Build one SSE event payload (key order is part of the wire format)."""
    return {"event": name, "run_id": run_id, "timestamp": time.time(), **fields}


def _run_not_found(_openai_error, run_id: str) -> "web.Response":
    return _json_error(_openai_error, f"Run not found: {run_id}", code="run_not_found", status=404)


def _uses_room_run_auth(self, request: "web.Request") -> bool:
    return request.path.endswith("/v1/runs") and bool(self._room_grant_token(request))


def _initialize_run_state(self, *, store_factory) -> None:
    """Initialize adapter-owned durable and live ``/v1/runs`` state."""
    self._run_idempotency_store = store_factory()
    self._run_owner_pid = os.getpid()
    try:
        from gateway.status import get_process_start_time
        self._run_owner_started = int(get_process_start_time(self._run_owner_pid) or 0)
    except Exception:
        self._run_owner_started = 0
    # All keyed by run_id: SSE queues (+creation time for the TTL sweep), connected
    # subscribers, live agent/task refs for cooperative stop (the executor thread may
    # outlive the request, hence the separate stopping set), pollable statuses, and
    # approval session keys (approval core resolves by session key, clients by run_id).
    self._run_idempotency_ids: set[str] = set()
    self._run_id_allocation_lock = threading.Lock()
    self._run_stream_subscribers: set[str] = set()
    self._stopping_run_ids: set[str] = set()
    (
        self._run_owners, self._run_streams, self._run_streams_created, self._active_run_agents,
        self._active_run_tasks, self._active_run_workers, self._run_statuses,
        self._run_approval_sessions,
    ) = ({} for _ in range(8))


def _http_routes(self) -> list[tuple[str, str, Any]]:
    return [
        ("POST", "/v1/runs", self._handle_runs), ("GET", "/v1/runs/{run_id}", self._handle_get_run),
        ("GET", "/v1/runs/{run_id}/events", self._handle_run_events),
        ("POST", "/v1/runs/{run_id}/approval", self._handle_run_approval),
        ("POST", "/v1/runs/{run_id}/steer", self._handle_steer_run),
        ("POST", "/v1/runs/{run_id}/stop", self._handle_stop_run)]


def _idempotency_capabilities(self, *, store_type) -> dict[str, Any]:
    return {
        "supported": True,
        "durable": self._run_idempotency_store.durable,
        "retention_seconds": store_type.RETENTION_SECONDS}


def _close_run_state(self) -> None:
    try:
        if getattr(self, "_run_idempotency_store", None) is not None:
            self._run_idempotency_store.close()
    except Exception:
        logger.debug("Failed to close run idempotency store for %s", self.name, exc_info=True)


def _set_run_status(self, run_id: str, status: str, **fields: Any) -> Dict[str, Any]:
    """Update pollable run status without exposing private agent objects."""
    now = time.time()
    current = self._run_statuses.get(run_id, {})
    previous_status = str(current.get("status") or "")
    field_names = set(fields)
    current.update({"object": "hermes.run", "run_id": run_id, "status": status, "updated_at": now})
    current.setdefault("created_at", fields.pop("created_at", now))
    current.update(fields)
    if status != "waiting_for_approval":
        current.pop("approval", None)
    self._run_statuses[run_id] = current
    should_persist = (
        status != previous_status
        or status in TERMINAL_STATUSES
        or bool(field_names & {"output", "error", "usage", "pending_steer", "session_id"}))
    if run_id in self._run_idempotency_ids and should_persist:
        try:
            self._run_idempotency_store.update_status(run_id, current)
        except Exception:
            logger.exception("[api_server] failed to persist idempotent run status %s", run_id)
    return current


def _make_run_event_callback(self, run_id: str, loop: "asyncio.AbstractEventLoop", *, _api_server):
    """Return a callback that pushes structured events to the run SSE queue."""
    redact_sensitive_text = _api_server.redact_sensitive_text

    def _push(event: Dict[str, Any]) -> None:
        self._set_run_status(
            run_id, self._run_statuses.get(run_id, {}).get("status", "running"), last_event=event.get("event"))
        q = self._run_streams.get(run_id)
        if q is not None:
            with suppress(Exception):
                loop.call_soon_threadsafe(q.put_nowait, event)

    def _callback(event_type: str, tool_name: str = None, preview: str = None, args=None, **kwargs):
        # _thinking / subagent.tool / subagent_progress are deliberately dropped (UI noise);
        # lifecycle boundaries must land so clients can observe delegate_task failures.
        fields = _FIXED_EVENT_FIELDS.get(event_type)
        if fields is not None:
            _push(_run_event(run_id, event_type, **fields(tool_name, preview, kwargs)))
        elif event_type in {"subagent.start", "subagent.complete"}:
            event = _run_event(run_id, event_type)
            if preview is not None:
                event["preview"] = redact_sensitive_text(str(preview), force=True)
            for key in _SUBAGENT_EVENT_KEYS:
                value = kwargs.get(key)
                if value is not None:
                    # Free text may carry child tool output: force secret redaction on this public stream.
                    redact = key in _SUBAGENT_TEXT_KEYS and isinstance(value, str)
                    event[key] = redact_sensitive_text(value, force=True) if redact else value
            _push(event)

    return _callback


def _room_permission_for(request: "web.Request") -> str:
    if request.path.endswith("/stop"):
        return "stop"
    if request.path.endswith("/approval"):
        return "approve"
    return "status" if request.method == "GET" else "dispatch"


def _run_idempotency_scope(self, request: "web.Request", *, _api_server) -> str:
    """Opaque auth/profile namespace; never persist bearer credentials."""
    if self._room_grant_token(request):
        claims = self._room_grant_claims(request, permission=_room_permission_for(request))
        _remember_room_retention(request, claims)
        parts = (claims[k] for k in (
            "room_id", "home_install_id", "authority_gateway_id", "authority_epoch",
            "member_id", "target_install_id", "target_profile"))
    else:
        auth_context = _api_server._api_request_auth_context.get()
        credential_owner = getattr(auth_context, "owner_key", None)
        if credential_owner:
            return str(credential_owner)
        parts = (_api_server._api_request_profile.get() or "default",
                 self._expected_api_key() or "unauthenticated-test-listener")
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def _check_run_auth(self, request: "web.Request", *, permission: str, _api_server) -> "web.Response | None":
    if not self._room_grant_token(request):
        return self._check_auth(request)
    try:
        self._room_grant_claims(request, permission=permission)
    except Exception as exc:
        return _room_grant_error_response(exc, _openai_error=_api_server._openai_error)
    return None


def _owner_alive(owner_pid: int, owner_started: int) -> bool:
    """True when the recorded owner pid still exists and is the same process incarnation."""
    try:
        from gateway.status import _pid_exists, get_process_start_time
        return owner_pid > 0 and bool(_pid_exists(owner_pid)) and (
            not owner_started or int(get_process_start_time(owner_pid) or 0) == owner_started)
    except Exception:
        return False


def _durable_run_status(self, request: "web.Request", run_id: str) -> Dict[str, Any] | None:
    """Hydrate a scoped run status and fail stale owners closed."""
    status = self._run_statuses.get(run_id)
    if status is not None:
        if run_id in self._run_idempotency_ids:
            scope = self._run_idempotency_scope(request)
            self._run_idempotency_store.extend_retention(scope, run_id, _room_retention_until(request))
        return status
    scope = self._run_idempotency_scope(request)
    record = self._run_idempotency_store.status_for_run(
        scope, run_id, retention_until=_room_retention_until(request))
    if record is None:
        return None
    status = dict(record["status"])
    if status.get("status") not in TERMINAL_STATUSES and not _owner_alive(
        int(record.get("owner_pid") or 0), int(record.get("owner_started") or 0)):
        status.update(
            status="interrupted", error="The gateway restarted before this run settled.",
            last_event="run.interrupted", updated_at=time.time())
        self._run_idempotency_store.update_status(run_id, status)
    self._run_statuses[run_id] = status
    self._run_idempotency_ids.add(run_id)
    self._run_owners[run_id] = scope
    return status


def _resolve_conversation_history(
    self, body: dict, raw_input: Any, *, _openai_error
) -> "tuple[List[Dict[str, str]], Any, Any, web.Response | None]":
    """Return ``(history, instructions, stored_session_id, error)``; precedence:
    ``conversation_history`` > ``previous_response_id`` chain > all-but-last ``input`` messages."""
    instructions = body.get("instructions")
    previous_response_id = body.get("previous_response_id")
    conversation_history: List[Dict[str, str]] = []
    raw_history = body.get("conversation_history")
    if raw_history:
        if not isinstance(raw_history, list):
            return [], instructions, None, _json_error(
                _openai_error, "'conversation_history' must be an array of message objects", status=400)
        for i, entry in enumerate(raw_history):
            if not isinstance(entry, dict) or {"role", "content"} - set(entry):
                return [], instructions, None, _json_error(
                    _openai_error, f"conversation_history[{i}] must have 'role' and 'content' fields",
                    status=400)
            conversation_history.append({"role": str(entry["role"]), "content": str(entry["content"])})
        if previous_response_id:
            logger.debug("Both conversation_history and previous_response_id provided; using conversation_history")
    stored_session_id = None
    if not conversation_history and previous_response_id:
        stored = self._response_store.get(previous_response_id)
        if stored:
            conversation_history = list(stored.get("conversation_history", []))
            stored_session_id = stored.get("session_id")
            if instructions is None:
                instructions = stored.get("instructions")
    if not conversation_history and isinstance(raw_input, list) and len(raw_input) > 1:
        for msg in raw_input[:-1]:
            if isinstance(msg, dict) and msg.get("role") and msg.get("content"):
                content = msg["content"]
                if isinstance(content, list):  # flatten multi-part content blocks to text
                    content = " ".join(p.get("text", "") for p in content
                                       if isinstance(p, dict) and p.get("type") == "text")
                conversation_history.append({"role": msg["role"], "content": str(content)})
    return conversation_history, instructions, stored_session_id, None


def _accepted_response(
    run_id: str, status: str, gateway_session_key, *, replayed: bool,
    session_id: Optional[str] = None,
) -> "web.Response":
    """202 admission response; replays are flagged via ``Idempotency-Replayed``."""
    headers = {"Idempotency-Replayed": "true"} if replayed else {}
    if gateway_session_key:
        headers["X-Hermes-Session-Key"] = gateway_session_key
    payload = {"run_id": run_id, "status": status, "replayed": replayed}
    if session_id:
        payload["session_id"] = session_id
    return web.json_response(payload, status=202, headers=headers)


def _replay_or_conflict(self, request, outcome, record, gateway_session_key, _openai_error) -> "web.Response":
    """409 for a fingerprint conflict, else a 202 replay of the already-admitted run."""
    if outcome == "conflict":
        return _json_error(
            _openai_error, "Idempotency-Key was already used with a different request payload",
            code="idempotency_key_conflict", status=409)
    original_id = str(record["run_id"])
    status = self._durable_run_status(request, original_id) or record["status"]
    return _accepted_response(
        original_id, status.get("status", "queued"), gateway_session_key, replayed=True,
        session_id=status.get("session_id"),
    )


@dataclass(slots=True)
class _RunLaunch:
    """State for an admitted run's background task; contextvars are captured here
    because the task outlives the request (and its middleware profile scope)."""

    owner: Any
    run_id: str
    queue: "asyncio.Queue[Optional[Dict]]"
    session_id: str
    gateway_session_key: Optional[str]
    declared_selected: bool
    user_message: str
    conversation_history: List[Dict[str, str]]
    agent_kwargs: dict  # ``_create_agent`` keyword arguments (prompt, model overrides, route, room policy)
    request_profile: Any
    request_auth_context: Any
    browser_control_principal: Any
    browser_control_transport_family: Any
    session_db: Any = None
    credential_session_lease_holder: Optional[str] = None
    credential_session_lease_refresh_handle: Any = None

    @property
    def approval_session_key(self) -> str:
        # Isolated per run: session ids are conversation scopes, not authorization namespaces.
        return self.run_id

    def put_event(self, event: Optional[Dict]) -> None:
        """Enqueue only while this run still owns live transport state."""
        if self.owner._run_streams.get(self.run_id) is self.queue:
            self.queue.put_nowait(event)


def _forget_run(self, run_id: str, *tables) -> None:
    """Drop *run_id* from the given run-keyed dicts/sets, then release its owner stamp."""
    for table in tables:
        (table.discard if isinstance(table, set) else lambda k: table.pop(k, None))(run_id)
    self._release_run_owner_if_forgotten(run_id)


def _generated_run_id_exists(self, run_id: str) -> bool:
    """Check every live and durable run namespace before reserving a generated ID."""
    live_namespaces = (
        self._run_owners,
        self._run_streams,
        self._run_streams_created,
        self._active_run_agents,
        self._active_run_tasks,
        self._run_statuses,
        self._run_approval_sessions,
        self._run_stream_subscribers,
        self._stopping_run_ids,
        self._run_idempotency_ids,
    )
    if any(run_id in namespace for namespace in live_namespaces):
        return True
    durable_run = self._run_idempotency_store.has_run_id(run_id)
    if not isinstance(durable_run, bool):
        raise RuntimeError("durable run status store returned an invalid result")
    if durable_run:
        return True
    return False


def _reserve_generated_run_id(self, owner: str, *, attempts: int = 32) -> Optional[str]:
    """Atomically reserve a fresh owner-scoped ID across live and durable run namespaces."""
    with self._run_id_allocation_lock:
        try:
            for _ in range(attempts):
                run_id = f"run_{uuid.uuid4().hex}"
                if not _generated_run_id_exists(self, run_id):
                    self._run_owners[run_id] = owner
                    return run_id
        except Exception:
            logger.warning("Unable to verify generated run-ID namespaces", exc_info=True)
    return None


def _retire_live_run(self, run_id: str) -> None:
    """Retire agent/task/approval control state once the executor-backed task is done."""
    _forget_run(self, run_id, self._active_run_agents, self._active_run_tasks, self._run_approval_sessions,
                self._stopping_run_ids)


def _drop_run_transport(self, run_id: str) -> None:
    _forget_run(self, run_id, self._run_streams, self._run_streams_created)


async def _handle_runs(self, request: "web.Request", *, _api_server) -> "web.Response":
    """POST /v1/runs — start an agent run, return run_id immediately."""
    _openai_error = _api_server._openai_error
    # Long-term memory scope header (see chat_completions for details).
    gateway_session_key, key_err = self._parse_session_key_header(request)
    if key_err is not None:
        return key_err
    try:
        body = await request.json()
    except Exception:
        return _json_error(_openai_error, "Invalid JSON", status=400)
    body, room_error = await self._normalize_room_dispatch(request, body)
    if room_error is not None:
        return room_error
    room_dispatch, room_execution_policy = (
        v if isinstance(v, dict) else None for v in (
            (body.get("hosted_room_dispatch"), body.get("_room_execution_policy"))
            if isinstance(body, dict) else (None, None)))
    idempotency_key = request.headers.get("Idempotency-Key", "").strip()
    if len(idempotency_key) > 255 or any(ord(ch) < 33 or ord(ch) > 126 for ch in idempotency_key):
        return _json_error(
            _openai_error, "Idempotency-Key must be 1-255 visible ASCII characters",
            code="invalid_idempotency_key", status=400)
    idempotency_scope = idempotency_fingerprint = ""
    if idempotency_key:
        idempotency_scope = self._run_idempotency_scope(request)
        idempotency_fingerprint = hashlib.sha256(json.dumps(
            {"body": body, "gateway_session_key": gateway_session_key or ""},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode()).hexdigest()
    raw_input = body.get("input")
    if not raw_input:
        return _json_error(_openai_error, "Missing 'input' field", status=400)
    if isinstance(raw_input, str):
        user_message = raw_input
    else:
        user_message = raw_input[-1].get("content", "") if isinstance(raw_input, list) else ""
    if not user_message:
        return _json_error(_openai_error, "No user message found in input", status=400)
    credential_owner = self._credential_owner()
    if credential_owner is not None and body.get("previous_response_id"):
        return _json_error(
            _openai_error, "previous_response_id is not available to this credential",
            code="credential_operation_forbidden", status=403)
    if credential_owner is not None and not body.get("session_id"):
        return _json_error(
            _openai_error,
            "Credential-authorized runs require an explicit pre-existing session_id",
            code="credential_session_required", status=400)
    conversation_history, instructions, stored_session_id, history_err = (
        _resolve_conversation_history(self, body, raw_input, _openai_error=_openai_error))
    if history_err is not None:
        return history_err
    previous_response_id = body.get("previous_response_id")
    session_id = body.get("session_id") or stored_session_id

    session_error = await self._credential_session_reference_error(session_id)
    if session_error is not None:
        return session_error
    if credential_owner is not None:
        try:
            durability_state = self._run_idempotency_store.durability_state
        except Exception:
            durability_state = "unavailable"
        if durability_state not in {"durable", "memory"}:
            return _json_error(
                _openai_error, "Durable run storage is unavailable; retry shortly",
                code="run_storage_unavailable", status=503)
    route = self._resolve_route(body.get("model"))
    agent_overrides = _api_server._request_agent_overrides(body, virtual_model=self._model_name)
    selection_error = self._request_route_conflict_error(
        session_id=session_id, gateway_session_key=gateway_session_key,
        requested_model=agent_overrides.get("requested_model"),
        requested_provider=agent_overrides.get("requested_provider"), route=route)
    if selection_error:
        return _json_error(_openai_error, selection_error, status=400)
    # A lost-acceptance replay must resolve even while the original run holds the last
    # concurrency slot; this read reserves nothing (the atomic reserve below closes the race).
    if idempotency_key:
        outcome, record = self._run_idempotency_store.lookup(
            idempotency_scope, idempotency_key, idempotency_fingerprint,
            retention_until=_room_retention_until(request))
        if outcome == "conflict" or (outcome == "reused" and record is not None):
            return _replay_or_conflict(self, request, outcome, record, gateway_session_key, _openai_error)
    # Enforce concurrency only for a genuinely new run.
    limited = self._concurrency_limited_response()
    if limited is not None:
        return limited
    run_id = _reserve_generated_run_id(self, self._run_idempotency_scope(request))
    if run_id is None:
        return _json_error(
            _openai_error, "Unable to allocate a run identifier; retry shortly",
            code="run_id_allocation_failed", status=503)

    session_db = None
    implicit_session_reserved = False
    credential_lease_holder = None
    credential_lease_refresh_handle = None

    def _release_allocations(*, delete_implicit: bool = False) -> None:
        if credential_lease_refresh_handle is not None:
            with suppress(Exception):
                credential_lease_refresh_handle.cancel(wait=1.0)
        if credential_lease_holder and session_db is not None:
            with suppress(Exception):
                session_db.release_session_turn_lease(str(session_id), credential_lease_holder)
        if delete_implicit and implicit_session_reserved and session_db is not None:
            with suppress(Exception):
                session_db.delete_session_if_empty(str(session_id))
        _forget_run(self, run_id, self._run_owners)

    # Preserve the legacy static/room contract: without an explicit/chained/header session,
    # the run ID is also the transcript ID. Credential callers never reach this fallback because
    # they must supply an explicit, pre-existing, exact-owned session above.
    _declared_selected = not session_id and bool(gateway_session_key)
    session_id = session_id or self._declared_conversation_session(gateway_session_key)
    load_session_history = bool(session_id)
    if not session_id:
        session_id = run_id

    if credential_owner is not None:
        from hermes_state_errors import SessionTurnLeaseLostError
        try:
            session_db = session_db or await self._ensure_session_db_async()
        except BaseException:
            _release_allocations(delete_implicit=True)
            raise
        if session_db is None:
            _release_allocations(delete_implicit=True)
            return self._session_db_unavailable()
        credential_lease_holder = f"pid={os.getpid()}:api-run={run_id}"
        acquisition_lock = threading.Lock()
        acquisition_state = {
            "finished": False, "acquired": False, "abandoned": False, "release_claimed": False,
        }
        acquisition_token = object()
        with self._session_db_cache_lock:
            self._pending_run_lease_acquisitions.add(acquisition_token)

        def _finish_acquisition_tracking() -> None:
            with self._session_db_cache_lock:
                self._pending_run_lease_acquisitions.discard(acquisition_token)
                close_deferred = self._session_db_close_deferred
            if close_deferred:
                self._close_cached_session_dbs()

        def _release_acquired_lease() -> None:
            try:
                session_db.release_session_turn_lease(str(session_id), credential_lease_holder)
            except Exception:
                logger.warning("Unable to release abandoned credential lease", exc_info=True)
            finally:
                _finish_acquisition_tracking()

        def _acquire_credential_lease() -> bool:
            try:
                acquired_in_worker = session_db.try_acquire_session_turn_lease(
                    str(session_id), credential_lease_holder,
                    expected_credential_owner=credential_owner,
                )
            except BaseException:
                _finish_acquisition_tracking()
                raise
            with acquisition_lock:
                acquisition_state["finished"] = True
                acquisition_state["acquired"] = bool(acquired_in_worker)
                release_in_worker = (
                    acquisition_state["abandoned"]
                    and acquired_in_worker
                    and not acquisition_state["release_claimed"]
                )
                if release_in_worker:
                    acquisition_state["release_claimed"] = True
            if release_in_worker:
                _release_acquired_lease()
            elif not acquired_in_worker:
                _finish_acquisition_tracking()
            return acquired_in_worker

        acquire_task = asyncio.create_task(asyncio.to_thread(_acquire_credential_lease))
        try:
            acquired = await asyncio.shield(acquire_task)
        except SessionTurnLeaseLostError:
            _release_allocations(delete_implicit=True)
            return _json_error(
                _openai_error, f"Session not found: {session_id}",
                code="session_not_found", status=404)
        except asyncio.CancelledError:
            # The worker owns late-acquisition cleanup independently of this repeatedly-cancellable
            # request task and of the asyncio wrapper's lifetime during loop shutdown.
            with acquisition_lock:
                acquisition_state["abandoned"] = True
                release_now = (
                    acquisition_state["finished"]
                    and acquisition_state["acquired"]
                    and not acquisition_state["release_claimed"]
                )
                if release_now:
                    acquisition_state["release_claimed"] = True
            _forget_run(self, run_id, self._run_owners)
            if release_now:
                asyncio.create_task(asyncio.to_thread(_release_acquired_lease))
            raise
        except Exception:
            logger.warning("Unable to acquire credential session lease", exc_info=True)
            _release_allocations(delete_implicit=True)
            return _json_error(
                _openai_error, "Session database unavailable", code="session_db_unavailable", status=503)
        if not acquired:
            _release_allocations(delete_implicit=True)
            return _json_error(
                _openai_error, "Session is busy; retry shortly",
                code="session_busy", status=409)
        with acquisition_lock:
            acquisition_state["release_claimed"] = True
        _finish_acquisition_tracking()
        from agent.periodic_scheduler import schedule

        def _refresh_credential_lease():
            try:
                return None if session_db.refresh_session_turn_lease(
                    str(session_id), credential_lease_holder,
                    expected_credential_owner=credential_owner,
                ) else False
            except Exception:
                logger.warning("Credential session lease refresh failed", exc_info=True)
                return False

        try:
            credential_lease_refresh_handle = schedule(_refresh_credential_lease, 60.0)
        except Exception:
            logger.warning("Unable to start credential session lease refresh", exc_info=True)
            _release_allocations(delete_implicit=True)
            return _json_error(
                _openai_error, "Session database unavailable", code="session_db_unavailable", status=503)

    if not conversation_history and load_session_history and not previous_response_id:
        try:
            conversation_history = await self._conversation_history_for_session(str(session_id))
        except BaseException:
            _release_allocations(delete_implicit=True)
            raise
    q = self._run_streams[run_id] = asyncio.Queue()
    created_at = self._run_streams_created[run_id] = time.time()
    self._run_approval_sessions[run_id] = run_id  # approval session key (see _RunLaunch)
    initial_status = self._set_run_status(
        run_id, "queued", created_at=created_at, session_id=session_id, model=body.get("model", self._model_name))
    launch = _RunLaunch(
        self, run_id, q, str(session_id), gateway_session_key, _declared_selected, user_message,
        conversation_history,
        agent_kwargs=dict(
            ephemeral_system_prompt=instructions, session_id=session_id, gateway_session_key=gateway_session_key,
            route=route, room_dispatch=room_dispatch, room_execution_policy=room_execution_policy,
            credential_owner=credential_owner,
            **{k: agent_overrides.get(k) for k in ("requested_model", "requested_provider", "model_options")}),
        request_profile=_api_server._api_request_profile.get(),
        request_auth_context=_api_server._api_request_auth_context.get(),
        browser_control_principal=_api_server._api_request_browser_control_principal.get(),
        browser_control_transport_family=_api_server._api_request_browser_control_transport_family.get(),
        session_db=session_db,
        credential_session_lease_holder=credential_lease_holder,
        credential_session_lease_refresh_handle=credential_lease_refresh_handle)
    self._activate_admitted_request()
    run_coro = _execute_run(self, launch, _api_server=_api_server)
    try:
        task = asyncio.create_task(run_coro)
    except BaseException:
        run_coro.close()
        _forget_run(
            self, run_id, self._run_streams, self._run_streams_created,
            self._run_approval_sessions, self._run_statuses,
        )
        _release_allocations(delete_implicit=True)
        raise
    self._active_run_tasks[run_id] = task
    with suppress(TypeError):
        self._background_tasks.add(task)  # tracked for shutdown drain
    if hasattr(task, "add_done_callback"):
        task.add_done_callback(self._background_tasks.discard)
    if idempotency_key:
        try:
            outcome, record = self._run_idempotency_store.reserve(
                idempotency_scope, idempotency_key, idempotency_fingerprint, run_id, initial_status,
                owner_pid=self._run_owner_pid, owner_started=self._run_owner_started,
                retention_until=_room_retention_until(request))
        except BaseException:
            task.cancel()
            _forget_run(
                self, run_id, self._active_run_tasks, self._run_streams,
                self._run_streams_created, self._run_approval_sessions, self._run_statuses,
            )
            _release_allocations(delete_implicit=True)
            raise
        if outcome != "created":
            task.cancel()
            _forget_run(
                self, run_id, self._active_run_tasks, self._run_streams,
                self._run_streams_created, self._run_approval_sessions, self._run_statuses,
            )
            _release_allocations(delete_implicit=True)
            return _replay_or_conflict(
                self, request, outcome, record, gateway_session_key, _openai_error
            )
        self._run_idempotency_ids.add(run_id)
    return _accepted_response(
        run_id, "started", gateway_session_key, replayed=False, session_id=str(session_id))


def _run_agent_sync(self, run: _RunLaunch, agent, approval_notify, *, _api_server):
    """Executor-thread body of one run; returns ``(result, usage)``."""
    from gateway.session_context import clear_session_vars, _clear_session_vars_unconditionally
    from gateway.hosted_room_execution_policy import (
        RoomExecutionPolicy, bind_room_execution_policy, reset_room_execution_policy,
        _clear_room_execution_policy_unconditionally)
    # No eager slash-worker pre-warm: slash.exec spawns one on demand (its error path already relies on that
    # respawn to recover from a dead worker). Each worker child runs its own MCP discovery (#61891), so
    # pre-warming one per session forks the full stdio MCP fleet — ~20 OS processes per retained session on
    # a config with a few stdio servers — even for sessions that never run a worker-routed command. Sessions
    # held by a live transport are never reaped, so with the desktop app open for days those fleets
    # accumulate until the OS refuses new process spawns.
    from tools.approval import register_gateway_notify, unregister_gateway_notify
    from tools.approval_context import (
        reset_current_session_key, set_current_session_key,
        _clear_current_session_key_unconditionally)
    session_id = run.session_id
    effective_task_id = session_id or run.run_id
    # (token, reset, fail-safe clear) triples unwound in the finally block; bound only once each step succeeds.
    resets: list[tuple[Any, Callable, Callable]] = []
    with self._profile_scope(run.request_profile):
        auth_token = _api_server._api_request_auth_context.set(run.request_auth_context)
        try:
            try:
                # Contextvars, not process env: concurrent runs must not share identity.
                resets.append((
                    set_current_session_key(run.approval_session_key),
                    reset_current_session_key,
                    _clear_current_session_key_unconditionally,
                ))
                # chat_id carries the raw session id like _run_agent() does; without it
                # tools.async_delegation sees no HERMES_SESSION_CHAT_ID and forces delegations sync.
                session_tokens = self._bind_api_server_session(
                    chat_id=session_id or "", session_key=run.approval_session_key, session_id=session_id or "",
                    browser_control_principal=run.browser_control_principal,
                    browser_control_transport_family=run.browser_control_transport_family)
                if session_tokens:
                    resets.append((
                        session_tokens, clear_session_vars, _clear_session_vars_unconditionally,
                    ))
                if run.agent_kwargs["room_dispatch"] is not None:
                    policy = RoomExecutionPolicy.from_mapping(run.agent_kwargs["room_execution_policy"] or {})
                    resets.append((
                        bind_room_execution_policy(policy), reset_room_execution_policy,
                        _clear_room_execution_policy_unconditionally,
                    ))
                register_gateway_notify(run.approval_session_key, approval_notify)
                # /v1/runs owns its agent lifecycle (no TurnRunner): record process ownership
                # so stop/cancel reaps only the background processes this run created.
                _api_server._publish_turn_process_ownership(agent, effective_task_id)
                r = agent.run_conversation(
                    user_message=run.user_message, conversation_history=run.conversation_history,
                    task_id=effective_task_id)
            finally:
                cleanup_error = None
                cleanup_actions = [
                    lambda: _api_server._clear_turn_process_ownership(agent),
                    lambda: (
                        self._bind_declared_conversation(
                            getattr(agent, "session_id", None) or session_id, run.gateway_session_key)
                        if run.declared_selected else None),
                    lambda: unregister_gateway_notify(run.approval_session_key),
                ]
                for cleanup in cleanup_actions:
                    try:
                        cleanup()
                    except BaseException as exc:
                        if cleanup_error is None:
                            cleanup_error = exc
                for token, reset, fail_safe_clear in reversed(resets):
                    try:
                        reset(token)
                    except BaseException as exc:
                        with suppress(BaseException):
                            fail_safe_clear()
                        if cleanup_error is None:
                            cleanup_error = exc
                if cleanup_error is not None:
                    raise cleanup_error
        finally:
            _api_server._reset_contextvars_fail_safe(
                (_api_server._api_request_auth_context, auth_token))
        return r, {key: getattr(agent, attr, 0) or 0 for key, attr in _USAGE_FIELDS}


def _make_approval_notify(self, run: _RunLaunch, *, _api_server) -> Callable[[Dict[str, Any]], None]:
    """Approval-request bridge: redact, stamp the event envelope, park the run status, enqueue."""
    run_id, q, loop = run.run_id, run.queue, asyncio.get_running_loop()

    def _approval_notify(approval_data: Dict[str, Any]) -> None:
        event = dict(approval_data or {})
        # Clients must never receive the raw flagged command: redact before it hits the stream.
        # Redact credentials from the command before it enters the SSE/API event stream — same egress bug as
        # #48456, second transport: API/desktop clients would otherwise receive the raw command Tirith
        # flagged. Reuse the gateway seam.
        if "command" in event:
            from gateway.run import _redact_approval_command
            event["command"] = _redact_approval_command(event.get("command"))
        event.update(_run_event(run_id, "approval.request", choices=_api_server._approval_event_choices(
            smart_denied=bool(event.get("smart_denied")),
            allow_session=event.get("allow_session") is not False,
            allow_permanent=event.get("allow_permanent") is not False)))
        self._set_run_status(run_id, "waiting_for_approval", last_event="approval.request", approval=event)
        with suppress(Exception):
            loop.call_soon_threadsafe(q.put_nowait, event)

    return _approval_notify


def _release_credential_run_lease(run: _RunLaunch) -> None:
    """Release a credential lease only from the executor worker that owned model/tool work."""
    if run.credential_session_lease_refresh_handle is not None:
        with suppress(Exception):
            run.credential_session_lease_refresh_handle.cancel(wait=1.0)
    if run.credential_session_lease_holder and run.session_db is not None:
        with suppress(Exception):
            run.session_db.release_session_turn_lease(
                run.session_id, run.credential_session_lease_holder
            )


def _run_agent_sync_with_lease(self, run: _RunLaunch, agent, approval_notify, *, _api_server):
    try:
        return _run_agent_sync(self, run, agent, approval_notify, _api_server=_api_server)
    finally:
        _release_credential_run_lease(run)


async def _execute_run(self, run: _RunLaunch, *, _api_server) -> None:
    """Drive one admitted run, publish its terminal event/status, release live state."""
    _redact_api_error_text = _api_server._redact_api_error_text
    run_id, loop = run.run_id, asyncio.get_running_loop()
    executor_owns_lease = False

    def _text_cb(delta: Optional[str]) -> None:
        if delta is None or run_id not in self._run_streams:
            return
        with suppress(Exception):
            loop.call_soon_threadsafe(run.put_event, _run_event(run_id, "message.delta", delta=delta))

    def _finish(status: str, extra: Optional[dict] = None, **fields: Any) -> None:
        """Terminal status, then best-effort ``run.<status>`` event; key order is wire shape."""
        extra = extra or {}
        self._set_run_status(run_id, status, **fields, last_event=f"run.{status}", **extra)
        with suppress(Exception):
            run.put_event(_run_event(run_id, f"run.{status}", **fields, **extra))

    try:
        self._set_run_status(run_id, "running")
        if run_id in self._stopping_run_ids:
            _finish("cancelled")
            return
        with self._profile_scope(run.request_profile):
            agent = self._create_agent(
                stream_delta_callback=_text_cb, tool_progress_callback=self._make_run_event_callback(run_id, loop),
                **run.agent_kwargs)
        if run.credential_session_lease_holder:
            agent._preacquired_session_turn_lease_holder = run.credential_session_lease_holder
        self._active_run_agents[run_id] = agent
        approval_notify = _make_approval_notify(self, run, _api_server=_api_server)
        worker = loop.run_in_executor(
            None,
            lambda: _run_agent_sync_with_lease(
                self, run, agent, approval_notify, _api_server=_api_server
            ),
        )
        executor_owns_lease = True
        self._active_run_workers[run_id] = (worker, agent)

        def _retire_worker(done_worker) -> None:
            if self._active_run_workers.get(run_id, (None, None))[0] is done_worker:
                self._active_run_workers.pop(run_id, None)
                _release_run_owner_if_forgotten(self, run_id)
            if not done_worker.cancelled():
                with suppress(Exception):
                    done_worker.exception()

        worker.add_done_callback(_retire_worker)
        # Cancelling an asyncio Future cannot stop its executor thread. Shield the worker;
        # its finally block remains the sole lease-release owner after submission.
        result, usage = await asyncio.shield(worker)
        if not isinstance(result, dict):
            result = {}
        if run_id in self._stopping_run_ids and result.get("interrupted") is True:
            _finish("cancelled")
        elif result.get("failed"):
            # Non-retryable client errors (401/400) return failed=True rather than raising.
            _finish("failed", error=_redact_api_error_text(result.get("error") or "agent run failed"))
        else:
            # Undelivered steer text rides on the terminal event/status for client replay.
            extra = {"pending_steer": result["pending_steer"]} if result.get("pending_steer") else {}
            candidate_session_id = getattr(agent, "session_id", None)
            effective_session_id = (
                candidate_session_id
                if isinstance(candidate_session_id, str) and candidate_session_id
                else run.session_id
            )
            _finish(
                "completed", extra, output=result.get("final_response", ""), usage=usage,
                session_id=effective_session_id,
            )
    except asyncio.CancelledError:
        _finish("cancelled")
        raise
    except _api_server._ProviderAuthResolutionError as exc:
        # Same controlled provider-auth message the _run_agent() endpoints give.
        logger.warning("Provider authentication failed for run=%s: %s", run_id, exc)
        _finish("failed", error=f"⚠️ Provider authentication failed: {exc}")
    except Exception as exc:
        logger.exception("[api_server] run %s failed", run_id)
        _finish("failed", error=_redact_api_error_text(exc))
    finally:
        # On cancellation (/stop) the executor thread may still block on an approval
        # Event; unregistering releases it. Idempotent on normal completion.
        _unregister_approval_notify(run.approval_session_key)
        with suppress(Exception):
            run.put_event(None)  # sentinel: close the SSE stream
        if not executor_owns_lease:
            _release_credential_run_lease(run)
        _retire_live_run(self, run_id)


def _unregister_approval_notify(approval_session_key: Optional[str]) -> None:
    """Best-effort release of a run's approval waiter (no-op without a key)."""
    with suppress(Exception):
        from tools.approval import unregister_gateway_notify
        if approval_session_key:
            unregister_gateway_notify(approval_session_key)


def _release_run_owner_if_forgotten(self, run_id: str) -> None:
    """Drop the owner stamp only once nothing keyed by *run_id* survives: ownership must
    outlive every surface it protects (retired on different clocks); ownerless = fail-closed."""
    live = (
        self._run_statuses, self._active_run_agents, self._active_run_tasks,
        self._active_run_workers, self._run_streams, self._run_approval_sessions,
    )
    if not any(run_id in table for table in live):
        self._run_owners.pop(run_id, None)


def _request_owns_run(self, request: "web.Request", run_id: str) -> bool:
    scope = self._run_idempotency_scope(request)
    owner = self._run_owners.get(run_id)
    if owner is not None:
        return owner == scope
    # No in-memory owner: only a durable record under the caller's scope admits it.
    # Under multiplex_profiles every profile holds a valid key, so ownerless = allow-all.
    # Run state that exists without an owner stamp is an unanswered authorization question, not a run anyone
    # may control — under gateway.multiplex_profiles every served profile holds a valid key, so admitting it
    # would make the boundary allow-all (#93689).
    return self._run_idempotency_store.owns_run(scope, run_id)


def _load_owned_run(self, request, *, _api_server, permission: Optional[str], active_fallback: bool):
    """Authenticate (*permission* -> room-grant aware; ``None`` -> API key only) and resolve
    ``(run_id, status, agent, task, error)``; *active_fallback* reports a live in-process run
    without pollable status as ``running`` instead of 404."""
    auth_err = self._check_run_auth(request, permission=permission) if permission else self._check_auth(request)
    if auth_err:
        return None, None, None, None, auth_err
    _openai_error = _api_server._openai_error
    run_id = request.match_info["run_id"]
    if not self._request_owns_run(request, run_id):
        return run_id, None, None, None, _run_not_found(_openai_error, run_id)
    agent = self._active_run_agents.get(run_id)
    task = self._active_run_tasks.get(run_id)
    status = self._durable_run_status(request, run_id)
    if status is None and active_fallback and (agent is not None or task is not None):
        status = self._set_run_status(run_id, "running")
    if status is None:
        return run_id, None, agent, task, _run_not_found(_openai_error, run_id)
    return run_id, status, agent, task, None


async def _handle_get_run(self, request: "web.Request", *, _api_server) -> "web.Response":
    """GET /v1/runs/{run_id} — return pollable run status for external UIs."""
    _, status, _, _, err = _load_owned_run(
        self, request, _api_server=_api_server, permission="status", active_fallback=True)
    return err or web.json_response(status)


async def _handle_run_events(self, request: "web.Request", *, _api_server) -> "web.StreamResponse":
    """GET /v1/runs/{run_id}/events — stream structured agent lifecycle events."""
    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err
    run_id = request.match_info["run_id"]
    if not self._request_owns_run(request, run_id):
        return _run_not_found(_api_server._openai_error, run_id)
    # Allow subscribing slightly before the run is registered (race window).
    # Confirm the force-kill actually reaped the process before we clear its PID file / scoped locks.
    # SIGKILL can fail to take (e.g. an uninterruptible-sleep or zombie-reaping parent), and if we blindly
    # clear the metadata and start a fresh instance we end up with two live gateways fighting over the same
    # token — the duplicate-gateway failure in #19471.
    for _ in range(20):
        if run_id in self._run_streams:
            break
        await asyncio.sleep(0.05)
    else:
        return _run_not_found(_api_server._openai_error, run_id)
    q = self._run_streams[run_id]
    self._run_stream_subscribers.add(run_id)
    response = web.StreamResponse(status=200, headers={
        "Content-Type": "text/event-stream", "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    await response.prepare(request)
    try:
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=30.0)
            except asyncio.TimeoutError:
                if not await self._credential_context_is_current(
                    request, _api_server.APIServerOperation.RUN_EVENTS_READ
                ):
                    break
                await response.write(b": keepalive\n\n")
                continue
            if not await self._credential_context_is_current(
                request, _api_server.APIServerOperation.RUN_EVENTS_READ
            ):
                break
            if event is None:  # run finished
                await response.write(b": stream closed\n\n")
                break
            await response.write(_api_server._sse_frame(event))
    except Exception as exc:
        logger.debug("[api_server] SSE stream error for run %s: %s", run_id, exc)
    finally:
        self._run_stream_subscribers.discard(run_id)
        _drop_run_transport(self, run_id)
    return response


def _mark_run_event(self, run_id: str, name: str, **fields: Any) -> None:
    """Record a control-plane event on the run status and (best effort) its SSE stream."""
    self._set_run_status(run_id, "running", last_event=name)
    q = self._run_streams.get(run_id)
    if q is not None:
        with suppress(Exception):
            q.put_nowait(_run_event(run_id, name, **fields))


_APPROVAL_CHOICE_ALIASES = {"approve": "once", "approved": "once", "allow": "once"}


async def _handle_run_approval(self, request: "web.Request", *, _api_server) -> "web.Response":
    """POST /v1/runs/{run_id}/approval — resolve a pending run approval."""
    _openai_error = _api_server._openai_error
    run_id, _, _, _, err = _load_owned_run(
        self, request, _api_server=_api_server, permission="approve", active_fallback=False)
    if err is not None:
        return err
    try:
        body = await request.json()
    except Exception:
        return _json_error(_openai_error, "Invalid JSON", status=400)
    raw_choice = str(body.get("choice", "")).strip().lower()
    choice = _APPROVAL_CHOICE_ALIASES.get(raw_choice, raw_choice)
    room_scoped = bool(self._room_grant_token(request))
    raw_request_id = body.get("request_id")
    request_id = raw_request_id.strip() if isinstance(raw_request_id, str) else ""
    # Room grants may resolve exactly one request and never widen to session/always.
    allowed = {"once", "deny"} if room_scoped else {"once", "session", "always", "deny"}
    resolve_all = any(_api_server._coerce_request_bool(body.get(k), default=False) for k in ("all", "resolve_all"))
    approval_session_key = self._run_approval_sessions.get(run_id)
    for failed, message, code, status in (
        (raw_request_id is not None and (not request_id or len(request_id) > 256),
         "Approval request_id is invalid.", "invalid_approval_request", 400),
        (choice not in allowed,
         "Invalid approval choice; expected one of: " + ", ".join(sorted(allowed)),
         "invalid_approval_choice", 400),
        (room_scoped and resolve_all,
         "Room approvals can resolve only one exact request", "invalid_approval_scope", 400),
        (room_scoped and not request_id,
         "Room approvals require the exact request_id.", "approval_request_required", 400),
        (not approval_session_key,
         f"Run has no active approval session: {run_id}", "approval_not_active", 409)):
        if failed:
            return _json_error(_openai_error, message, code=code, status=status)
    try:
        from tools.approval import resolve_gateway_approval
        resolved = resolve_gateway_approval(
            approval_session_key, choice, resolve_all=resolve_all, request_id=request_id or None)
    except Exception as exc:
        logger.exception("[api_server] approval resolution failed for run %s", run_id)
        return _json_error(_openai_error, str(exc), status=500)
    if resolved <= 0:
        return _json_error(
            _openai_error, f"Run has no pending approval: {run_id}", code="approval_not_pending", status=409)
    request_id_field = {"request_id": request_id} if request_id else {}
    _mark_run_event(self, run_id, "approval.responded", choice=choice, **request_id_field, resolved=resolved)
    return web.json_response({
        "object": "hermes.run.approval_response", "run_id": run_id, "choice": choice, **request_id_field,
        "resolved": resolved})


async def _handle_steer_run(self, request: "web.Request", *, _api_server) -> "web.Response":
    """POST /v1/runs/{run_id}/steer — inject guidance into a running agent."""
    _openai_error = _api_server._openai_error
    run_id, status, agent, _, err = _load_owned_run(
        self, request, _api_server=_api_server, permission=None, active_fallback=False)
    if err is not None:
        return err
    # /stop keeps agent refs during cooperative shutdown, so the status gate (not the
    # agent ref) is what rejects stop-then-steer.
    if status.get("status") != "running" or not hasattr(agent, "steer"):
        return _json_error(
            _openai_error, f"Run is not currently accepting steer input: {run_id}",
            code="run_not_accepting_steer", status=409)
    body, err = await self._read_json_body(request)
    if err:
        return err
    raw_text = body.get("input") or body.get("message") or body.get("text") or ""
    steer_text = _api_server._normalize_chat_content(raw_text).strip()
    if not steer_text:
        return _json_error(
            _openai_error, "Missing non-empty steer text; expected 'input', 'message', or 'text'.",
            code="invalid_steer_input", status=400)
    try:
        accepted = bool(agent.steer(steer_text))
    except Exception as exc:
        logger.exception("[api_server] steer failed for run %s", run_id)
        return _json_error(_openai_error, _api_server._redact_api_error_text(exc), code="steer_failed", status=500)
    if not accepted:
        return _json_error(
            _openai_error, f"Run did not accept steer text: {run_id}", code="steer_not_accepted", status=409)
    _mark_run_event(self, run_id, "run.steered", accepted=True)
    return web.json_response({"object": "hermes.run.steer", "run_id": run_id, "accepted": True})


async def _handle_stop_run(self, request: "web.Request", *, _api_server) -> "web.Response":
    """POST /v1/runs/{run_id}/stop — interrupt a running agent."""
    _openai_error = _api_server._openai_error
    run_id, status, agent, task, err = _load_owned_run(
        self, request, _api_server=_api_server, permission="stop", active_fallback=True)
    if err is not None:
        return err
    if status.get("status") in TERMINAL_STATUSES:
        return web.json_response(status)
    if agent is None and task is None:
        return _json_error(
            _openai_error, f"Run is not active in this gateway process: {run_id}",
            code="run_not_active", status=409)
    self._set_run_status(run_id, "stopping", last_event="run.stopping")
    self._stopping_run_ids.add(run_id)
    if agent is not None:
        with suppress(Exception):
            _api_server.request_hard_interrupt(agent, "Stop requested via API")
        # Reap only this run's background processes (epoch-gated inside, so a concurrent
        # run on the same session_id keeps its own); no-op if the run already finished.
        _api_server._reap_disconnected_agent_processes(agent, source="api_server_run_stop")
    return web.json_response({"run_id": run_id, "status": "stopping"})


async def _sweep_orphaned_runs(self) -> None:
    """Periodically expire transport buffers and terminal status records."""
    while True:
        await asyncio.sleep(60)
        self._sweep_orphaned_runs_once(time.time())


def _sweep_orphaned_runs_once(self, now: Optional[float] = None) -> None:
    """Expire old SSE buffers without treating transport age as run age."""
    if now is None:
        now = time.time()
    for run_id, created_at in list(self._run_streams_created.items()):
        if now - created_at <= self._RUN_STREAM_TTL or run_id in self._run_stream_subscribers:
            continue
        logger.debug("[api_server] sweeping expired run transport %s", run_id)
        task = self._active_run_tasks.get(run_id)
        # Transport TTL bounds buffering; live control state survives until the task returns.
        _drop_run_transport(self, run_id)
        if task is None or task.done():
            _unregister_approval_notify(self._run_approval_sessions.get(run_id))
            _retire_live_run(self, run_id)
    for run_id, status in list(self._run_statuses.items()):
        if (status.get("status") in {"completed", "failed", "cancelled"}
                and now - float(status.get("updated_at", 0) or 0) > self._RUN_STATUS_TTL):
            _forget_run(self, run_id, self._run_statuses, self._run_idempotency_ids)
