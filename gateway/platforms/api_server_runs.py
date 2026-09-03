"""Durable ``/v1/runs`` admission, status, events, and control handlers."""

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from contextlib import suppress
from typing import Any, Dict, List, Optional

try:
    from aiohttp import web
    from aiohttp.web_request import RequestKey
except ImportError:
    web = None  # type: ignore[assignment]
    RequestKey = None  # type: ignore[assignment,misc]


logger = logging.getLogger("gateway.platforms.api_server")
_ROOM_RETENTION_REQUEST_KEY = (
    RequestKey("hermes.room_run_retention_until", float)
    if RequestKey is not None
    else "hermes.room_run_retention_until"
)


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


def _uses_room_run_auth(self, request: "web.Request") -> bool:
    return request.path.endswith("/v1/runs") and bool(
        self._room_grant_token(request)
    )


def _initialize_run_state(self, *, store_factory) -> None:
    """Initialize adapter-owned durable and live ``/v1/runs`` state."""
    self._run_idempotency_store = store_factory()
    self._run_idempotency_ids: set[str] = set()
    self._run_owners: Dict[str, str] = {}
    self._run_owner_pid = os.getpid()
    try:
        from gateway.status import get_process_start_time

        self._run_owner_started = int(
            get_process_start_time(self._run_owner_pid) or 0
        )
    except Exception:
        self._run_owner_started = 0
    # Active run streams: run_id -> asyncio.Queue of SSE event dicts
    self._run_streams: Dict[str, "asyncio.Queue[Optional[Dict]]"] = {}
    # Creation timestamps for orphaned-run TTL sweep
    self._run_streams_created: Dict[str, float] = {}
    # Runs with a connected SSE consumer; their queue is actively draining.
    self._run_stream_subscribers: set[str] = set()
    # Active run agent/task references for stop support
    self._active_run_agents: Dict[str, Any] = {}
    self._active_run_tasks: Dict[str, "asyncio.Task"] = {}
    # Stop is cooperative: the executor thread may outlive the HTTP request.
    self._stopping_run_ids: set[str] = set()
    # Pollable run status for dashboards and external control-plane UIs.
    self._run_statuses: Dict[str, Dict[str, Any]] = {}
    # Active approval session key for each run_id. The approval core resolves
    # requests by session key, while API clients address them by run_id.
    self._run_approval_sessions: Dict[str, str] = {}


def _http_routes(self) -> list[tuple[str, str, Any]]:
    return [
        ("POST", "/v1/runs", self._handle_runs),
        ("GET", "/v1/runs/{run_id}", self._handle_get_run),
        ("GET", "/v1/runs/{run_id}/events", self._handle_run_events),
        ("POST", "/v1/runs/{run_id}/approval", self._handle_run_approval),
        ("POST", "/v1/runs/{run_id}/steer", self._handle_steer_run),
        ("POST", "/v1/runs/{run_id}/stop", self._handle_stop_run),
    ]


def _idempotency_capabilities(self, *, store_type) -> dict[str, Any]:
    return {
        "supported": True,
        "durable": self._run_idempotency_store.durable,
        "retention_seconds": store_type.RETENTION_SECONDS,
    }


def _close_run_state(self) -> None:
    store = getattr(self, "_run_idempotency_store", None)
    if store is None:
        return
    try:
        store.close()
    except Exception:
        logger.debug(
            "Failed to close run idempotency store for %s",
            self.name,
            exc_info=True,
        )


def _set_run_status(
    self,
    run_id: str,
    status: str,
    **fields: Any,
) -> Dict[str, Any]:
    """Update pollable run status without exposing private agent objects."""
    now = time.time()
    current = self._run_statuses.get(run_id, {})
    previous_status = str(current.get("status") or "")
    field_names = set(fields)
    current.update({
        "object": "hermes.run",
        "run_id": run_id,
        "status": status,
        "updated_at": now,
    })
    current.setdefault("created_at", fields.pop("created_at", now))
    current.update(fields)
    if status != "waiting_for_approval":
        current.pop("approval", None)
    self._run_statuses[run_id] = current
    should_persist = (
        status != previous_status
        or status in {"completed", "failed", "cancelled", "interrupted"}
        or bool(
            field_names
            & {"output", "error", "usage", "pending_steer", "session_id"}
        )
    )
    if run_id in self._run_idempotency_ids and should_persist:
        try:
            self._run_idempotency_store.update_status(run_id, current)
        except Exception:
            logger.exception(
                "[api_server] failed to persist idempotent run status %s", run_id
            )
    return current


def _make_run_event_callback(
    self,
    run_id: str,
    loop: "asyncio.AbstractEventLoop",
    *,
    _api_server,
):
    """Return a callback that pushes structured events to the run SSE queue."""
    redact_sensitive_text = _api_server.redact_sensitive_text

    def _push(event: Dict[str, Any]) -> None:
        self._set_run_status(
            run_id,
            self._run_statuses.get(run_id, {}).get("status", "running"),
            last_event=event.get("event"),
        )
        q = self._run_streams.get(run_id)
        if q is None:
            return
        try:
            loop.call_soon_threadsafe(q.put_nowait, event)
        except Exception:
            pass

    def _callback(
        event_type: str,
        tool_name: str = None,
        preview: str = None,
        args=None,
        **kwargs,
    ):
        ts = time.time()
        if event_type == "tool.started":
            _push({
                "event": "tool.started",
                "run_id": run_id,
                "timestamp": ts,
                "tool": tool_name,
                "preview": preview,
            })
        elif event_type == "tool.completed":
            _push({
                "event": "tool.completed",
                "run_id": run_id,
                "timestamp": ts,
                "tool": tool_name,
                "duration": round(kwargs.get("duration", 0), 3),
                "error": kwargs.get("is_error", False),
            })
        elif event_type == "reasoning.available":
            _push({
                "event": "reasoning.available",
                "run_id": run_id,
                "timestamp": ts,
                "text": preview or "",
            })
        elif event_type in {"subagent.start", "subagent.complete"}:
            event = {
                "event": event_type,
                "run_id": run_id,
                "timestamp": ts,
            }
            if preview is not None:
                event["preview"] = redact_sensitive_text(
                    str(preview), force=True
                )
            for key in (
                "goal",
                "task_count",
                "task_index",
                "subagent_id",
                "child_session_id",
                "delegation_id",
                "parent_id",
                "depth",
                "model",
                "tool_count",
                "status",
                "summary",
                "duration_seconds",
                "input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "api_calls",
                "cost_usd",
                "files_read",
                "files_written",
                "output_tail",
            ):
                value = kwargs.get(key)
                if value is None:
                    continue
                # Free-text fields can carry child terminal/tool output —
                # force the same secret redaction the API applies to error
                # text before it leaves the process on a public stream.
                if key in ("goal", "summary", "output_tail") and isinstance(
                    value, str
                ):
                    value = redact_sensitive_text(value, force=True)
                event[key] = value
            _push(event)
        # _thinking, subagent.tool, and subagent_progress are intentionally
        # not forwarded on the /v1/runs stream: they are high-volume UI
        # noise. Lifecycle boundaries (start/complete) still need to land
        # so clients can observe delegate_task timeouts and failures.

    return _callback


def _run_idempotency_scope(
    self,
    request: "web.Request",
    *,
    _api_server,
) -> str:
    """Opaque auth/profile namespace; never persist bearer credentials."""
    _api_request_profile = _api_server._api_request_profile

    room_token = self._room_grant_token(request)
    if room_token:
        claims = self._room_grant_claims(
            request,
            permission=(
                "stop"
                if request.path.endswith("/stop")
                else "approve"
                if request.path.endswith("/approval")
                else "status"
                if request.method == "GET"
                else "dispatch"
            ),
        )
        _remember_room_retention(request, claims)
        identity = (
            f"{claims['room_id']}\0{claims['home_install_id']}\0"
            f"{claims['authority_gateway_id']}\0{claims['authority_epoch']}\0"
            f"{claims['member_id']}\0{claims['target_install_id']}\0"
            f"{claims['target_profile']}"
        )
        return hashlib.sha256(identity.encode()).hexdigest()
    profile = _api_request_profile.get() or "default"
    expected_key = self._expected_api_key()
    identity = expected_key or "unauthenticated-test-listener"
    return hashlib.sha256(f"{profile}\0{identity}".encode()).hexdigest()


def _check_run_auth(
    self,
    request: "web.Request",
    *,
    permission: str,
    _api_server,
) -> "web.Response | None":
    _openai_error = _api_server._openai_error

    if not self._room_grant_token(request):
        return self._check_auth(request)
    try:
        self._room_grant_claims(request, permission=permission)
    except Exception as exc:
        from gateway.platforms.api_server_room_grants import (
            RoomGrantReauthorizationRequired,
        )

        reauthorization = isinstance(exc, RoomGrantReauthorizationRequired)
        return web.json_response(
            _openai_error(
                (
                    "Room authorization needs to be renewed."
                    if reauthorization
                    else "Room authorization is invalid or expired."
                ),
                err_type="gateway_auth_error",
                code=(
                    "room_reauthorization_required"
                    if reauthorization
                    else "invalid_room_grant"
                ),
            ),
            status=403 if reauthorization else 401,
        )
    return None


def _durable_run_status(
    self,
    request: "web.Request",
    run_id: str,
) -> Dict[str, Any] | None:
    """Hydrate a scoped run status and fail stale owners closed."""
    status = self._run_statuses.get(run_id)
    if status is not None:
        if run_id in self._run_idempotency_ids:
            scope = self._run_idempotency_scope(request)
            self._run_idempotency_store.extend_retention(
                scope,
                run_id,
                _room_retention_until(request),
            )
        return status

    scope = self._run_idempotency_scope(request)
    record = self._run_idempotency_store.status_for_run(
        scope,
        run_id,
        retention_until=_room_retention_until(request),
    )
    if record is None:
        return None

    status = dict(record["status"])
    owner_pid = int(record.get("owner_pid") or 0)
    owner_started = int(record.get("owner_started") or 0)
    nonterminal = status.get("status") not in {
        "completed",
        "failed",
        "cancelled",
        "interrupted",
    }
    owner_alive = False
    if owner_pid > 0:
        try:
            from gateway.status import _pid_exists, get_process_start_time

            owner_alive = bool(_pid_exists(owner_pid))
            if owner_alive and owner_started:
                owner_alive = (
                    int(get_process_start_time(owner_pid) or 0) == owner_started
                )
        except Exception:
            owner_alive = False

    if nonterminal and not owner_alive:
        status.update(
            {
                "status": "interrupted",
                "error": "The gateway restarted before this run settled.",
                "last_event": "run.interrupted",
                "updated_at": time.time(),
            }
        )
        self._run_idempotency_store.update_status(run_id, status)

    self._run_statuses[run_id] = status
    self._run_idempotency_ids.add(run_id)
    self._run_owners[run_id] = scope
    return status


async def _handle_runs(
    self,
    request: "web.Request",
    *,
    _api_server,
) -> "web.Response":
    """POST /v1/runs — start an agent run, return run_id immediately."""
    _ProviderAuthResolutionError = _api_server._ProviderAuthResolutionError
    _api_request_browser_control_principal = (
        _api_server._api_request_browser_control_principal
    )
    _api_request_browser_control_transport_family = (
        _api_server._api_request_browser_control_transport_family
    )
    _api_request_profile = _api_server._api_request_profile
    _approval_event_choices = _api_server._approval_event_choices
    _clear_turn_process_ownership = _api_server._clear_turn_process_ownership
    _openai_error = _api_server._openai_error
    _publish_turn_process_ownership = _api_server._publish_turn_process_ownership
    _redact_api_error_text = _api_server._redact_api_error_text
    _request_agent_overrides = _api_server._request_agent_overrides

    # Long-term memory scope header (see chat_completions for details).
    gateway_session_key, key_err = self._parse_session_key_header(request)
    if key_err is not None:
        return key_err

    try:
        body = await request.json()
    except Exception:
        return web.json_response(_openai_error("Invalid JSON"), status=400)

    body, room_error = await self._normalize_room_dispatch(request, body)
    if room_error is not None:
        return room_error
    room_dispatch = (
        body.get("hosted_room_dispatch")
        if isinstance(body, dict)
        and isinstance(body.get("hosted_room_dispatch"), dict)
        else None
    )
    room_execution_policy = (
        body.get("_room_execution_policy")
        if isinstance(body, dict)
        and isinstance(body.get("_room_execution_policy"), dict)
        else None
    )

    idempotency_key = request.headers.get("Idempotency-Key", "").strip()
    if idempotency_key and (
        len(idempotency_key) > 255
        or any(ord(ch) < 33 or ord(ch) > 126 for ch in idempotency_key)
    ):
        return web.json_response(
            _openai_error(
                "Idempotency-Key must be 1-255 visible ASCII characters",
                code="invalid_idempotency_key",
            ),
            status=400,
        )
    idempotency_scope = (
        self._run_idempotency_scope(request) if idempotency_key else ""
    )
    idempotency_fingerprint = (
        hashlib.sha256(
            json.dumps(
                {
                    "body": body,
                    "gateway_session_key": gateway_session_key or "",
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        if idempotency_key
        else ""
    )

    raw_input = body.get("input")
    if not raw_input:
        return web.json_response(_openai_error("Missing 'input' field"), status=400)

    user_message = (
        raw_input
        if isinstance(raw_input, str)
        else (
            raw_input[-1].get("content", "") if isinstance(raw_input, list) else ""
        )
    )
    if not user_message:
        return web.json_response(
            _openai_error("No user message found in input"), status=400
        )

    instructions = body.get("instructions")
    previous_response_id = body.get("previous_response_id")

    # Accept explicit conversation_history from the request body.
    # Precedence: explicit conversation_history > previous_response_id.
    conversation_history: List[Dict[str, str]] = []
    raw_history = body.get("conversation_history")
    if raw_history:
        if not isinstance(raw_history, list):
            return web.json_response(
                _openai_error("'conversation_history' must be an array of message objects"),
                status=400,
            )
        for i, entry in enumerate(raw_history):
            if not isinstance(entry, dict) or "role" not in entry or "content" not in entry:
                return web.json_response(
                    _openai_error(f"conversation_history[{i}] must have 'role' and 'content' fields"),
                    status=400,
                )
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

    # When input is a multi-message array, extract all but the last
    # message as conversation history (the last becomes user_message).
    # Only fires when no explicit history was provided.
    if not conversation_history and isinstance(raw_input, list) and len(raw_input) > 1:
        for msg in raw_input[:-1]:
            if isinstance(msg, dict) and msg.get("role") and msg.get("content"):
                content = msg["content"]
                if isinstance(content, list):
                    # Flatten multi-part content blocks to text
                    content = " ".join(
                        part.get("text", "") for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                conversation_history.append({"role": msg["role"], "content": str(content)})

    session_id = body.get("session_id") or stored_session_id
    route = self._resolve_route(body.get("model"))
    agent_overrides = _request_agent_overrides(body, virtual_model=self._model_name)
    selection_error = (
        agent_overrides.get("request_error")
        or self._request_route_conflict_error(
            session_id=session_id,
            gateway_session_key=gateway_session_key,
            requested_model=agent_overrides.get("requested_model"),
            requested_provider=agent_overrides.get("requested_provider"),
            route=route,
        )
    )
    if selection_error:
        return web.json_response(_openai_error(selection_error), status=400)

    # A lost-acceptance replay must resolve even while the original run
    # consumes the final concurrency slot. This read does not reserve a
    # missing key; the atomic reserve below closes the concurrent-miss race.
    if idempotency_key:
        outcome, record = self._run_idempotency_store.lookup(
            idempotency_scope,
            idempotency_key,
            idempotency_fingerprint,
            retention_until=_room_retention_until(request),
        )
        if outcome == "conflict":
            return web.json_response(
                _openai_error(
                    "Idempotency-Key was already used with a different request payload",
                    code="idempotency_key_conflict",
                ),
                status=409,
            )
        if outcome == "reused" and record is not None:
            original_id = str(record["run_id"])
            status = self._durable_run_status(request, original_id) or record[
                "status"
            ]
            headers = {"Idempotency-Replayed": "true"}
            if gateway_session_key:
                headers["X-Hermes-Session-Key"] = gateway_session_key
            return web.json_response(
                {
                    "run_id": original_id,
                    "status": status.get("status", "queued"),
                    "replayed": True,
                },
                status=202,
                headers=headers,
            )

    # Enforce concurrency only for a genuinely new run.
    limited = self._concurrency_limited_response()
    if limited is not None:
        return limited

    if not conversation_history and session_id and not previous_response_id:
        conversation_history = await self._conversation_history_for_session(
            str(session_id)
        )

    run_id = f"run_{uuid.uuid4().hex}"
    self._run_owners[run_id] = self._run_idempotency_scope(request)
    # Same rule as /v1/responses: an explicit body session_id wins, then
    # the response chain, then the conversation the client declared via
    # ``X-Hermes-Session-Key``.  Falling straight through to ``run_id``
    # made the run id the conversation identity, so a declared channel
    # re-keyed every affinity surface once per run (#96811).
    # Same precedence gate as /v1/responses: an explicit body session_id
    # or a chained session owns its own routing key and must not be
    # rebound to this request's header key.
    _declared_selected = not session_id and bool(gateway_session_key)
    session_id = (
        session_id
        or self._declared_conversation_session(gateway_session_key)
        or run_id
    )
    # Approval queues gate host-side tool execution and must be isolated
    # per API run. Client-provided session IDs and memory session keys are
    # conversation/memory scopes, not authorization namespaces: multiple
    # concurrent runs can intentionally share them, and resolving an
    # approval for one run must not unblock another run's dangerous command.
    approval_session_key = run_id
    ephemeral_system_prompt = instructions
    loop = asyncio.get_running_loop()
    q: "asyncio.Queue[Optional[Dict]]" = asyncio.Queue()
    created_at = time.time()
    self._run_streams[run_id] = q
    self._run_streams_created[run_id] = created_at
    self._run_approval_sessions[run_id] = approval_session_key

    event_cb = self._make_run_event_callback(run_id, loop)

    def _put_event_if_active(event: Optional[Dict]) -> None:
        """Enqueue only while this run still owns live transport state."""
        if self._run_streams.get(run_id) is q:
            q.put_nowait(event)

    # Also wire stream_delta_callback so message.delta events flow through.
    def _text_cb(delta: Optional[str]) -> None:
        if delta is None:
            return
        if run_id not in self._run_streams:
            return
        try:
            loop.call_soon_threadsafe(_put_event_if_active, {
                "event": "message.delta",
                "run_id": run_id,
                "timestamp": time.time(),
                "delta": delta,
            })
        except Exception:
            pass

    initial_status = self._set_run_status(
        run_id,
        "queued",
        created_at=created_at,
        session_id=session_id,
        model=body.get("model", self._model_name),
    )
    if idempotency_key:
        outcome, record = self._run_idempotency_store.reserve(
            idempotency_scope,
            idempotency_key,
            idempotency_fingerprint,
            run_id,
            initial_status,
            owner_pid=self._run_owner_pid,
            owner_started=self._run_owner_started,
            retention_until=_room_retention_until(request),
        )
        if outcome != "created":
            self._run_streams.pop(run_id, None)
            self._run_streams_created.pop(run_id, None)
            self._run_approval_sessions.pop(run_id, None)
            self._run_statuses.pop(run_id, None)
            self._run_owners.pop(run_id, None)
            if outcome == "conflict":
                return web.json_response(
                    _openai_error(
                        "Idempotency-Key was already used with a different request payload",
                        code="idempotency_key_conflict",
                    ), status=409,
                )
            original_id = record["run_id"]
            replay_status = self._durable_run_status(request, original_id) or record[
                "status"
            ]
            headers = {"Idempotency-Replayed": "true"}
            if gateway_session_key:
                headers["X-Hermes-Session-Key"] = gateway_session_key
            return web.json_response(
                {
                    "run_id": original_id,
                    "status": replay_status.get("status", "queued"),
                    "replayed": True,
                },
                status=202,
                headers=headers,
            )
        self._run_idempotency_ids.add(run_id)

    # Background task outlives the HTTP response (and thus the middleware
    # profile scope). Capture now and re-enter inside the task/executor.
    request_profile = _api_request_profile.get()
    request_browser_control_principal = (
        _api_request_browser_control_principal.get()
    )
    request_browser_control_transport_family = (
        _api_request_browser_control_transport_family.get()
    )

    async def _run_and_close():
        try:
            self._set_run_status(run_id, "running")
            if run_id in self._stopping_run_ids:
                _put_event_if_active({
                    "event": "run.cancelled",
                    "run_id": run_id,
                    "timestamp": time.time(),
                })
                self._set_run_status(
                    run_id,
                    "cancelled",
                    last_event="run.cancelled",
                )
                return
            with self._profile_scope(request_profile):
                agent = self._create_agent(
                    ephemeral_system_prompt=ephemeral_system_prompt,
                    session_id=session_id,
                    stream_delta_callback=_text_cb,
                    tool_progress_callback=event_cb,
                    gateway_session_key=gateway_session_key,
                    requested_model=agent_overrides.get("requested_model"),
                    requested_provider=agent_overrides.get("requested_provider"),
                    model_options=agent_overrides.get("model_options"),
                    route=route,
                    room_dispatch=room_dispatch,
                    room_execution_policy=room_execution_policy,
                )
            self._active_run_agents[run_id] = agent

            def _approval_notify(approval_data: Dict[str, Any]) -> None:
                event = dict(approval_data or {})
                # Redact credentials from the command before it enters the
                # SSE/API event stream — same egress bug as #48456, second
                # transport: API/desktop clients would otherwise receive the
                # raw command Tirith flagged. Reuse the gateway seam.
                if "command" in event:
                    from gateway.run import _redact_approval_command

                    event["command"] = _redact_approval_command(event.get("command"))
                event.update({
                    "event": "approval.request",
                    "run_id": run_id,
                    "timestamp": time.time(),
                    "choices": _approval_event_choices(
                        smart_denied=bool(event.get("smart_denied")),
                        allow_session=event.get("allow_session") is not False,
                        allow_permanent=event.get("allow_permanent") is not False,
                    ),
                })
                self._set_run_status(
                    run_id,
                    "waiting_for_approval",
                    last_event="approval.request",
                    approval=event,
                )
                try:
                    loop.call_soon_threadsafe(q.put_nowait, event)
                except Exception:
                    pass

            def _run_sync():
                from gateway.session_context import clear_session_vars
                from tools.approval import (
                    register_gateway_notify,
                    reset_current_session_key,
                    set_current_session_key,
                    unregister_gateway_notify,
                )

                effective_task_id = session_id or run_id
                approval_token = None
                session_tokens = []
                room_policy_token = None
                with self._profile_scope(request_profile):
                    try:
                        # Bind approval/session identity for this API run via
                        # contextvars so concurrent runs do not share process
                        # environment state.
                        approval_token = set_current_session_key(approval_session_key)
                        session_tokens = self._bind_api_server_session(
                            # chat_id carries the raw session id (the
                            # X-Hermes-Session-Id equivalent) exactly like
                            # the other agent-entry routes bind it via
                            # _run_agent(). Without it,
                            # tools.async_delegation reads an empty
                            # HERMES_SESSION_CHAT_ID on /v1/runs and
                            # background delegations stay forced-sync
                            # (no wake target).
                            chat_id=session_id or "",
                            session_key=approval_session_key,
                            session_id=session_id or "",
                            browser_control_principal=(
                                request_browser_control_principal
                            ),
                            browser_control_transport_family=(
                                request_browser_control_transport_family
                            ),
                        )
                        if room_dispatch is not None:
                            from gateway.hosted_room_execution_policy import (
                                RoomExecutionPolicy,
                                bind_room_execution_policy,
                            )

                            policy = RoomExecutionPolicy.from_mapping(
                                room_execution_policy or {}
                            )
                            room_policy_token = bind_room_execution_policy(policy)
                        register_gateway_notify(approval_session_key, _approval_notify)
                        # /v1/runs runs its own agent lifecycle (no
                        # TurnRunner, no _run_agent) — record turn process
                        # ownership so stop/cancel can reap only the
                        # background processes this run created (#76115).
                        _publish_turn_process_ownership(agent, effective_task_id)
                        r = agent.run_conversation(
                            user_message=user_message,
                            conversation_history=conversation_history,
                            task_id=effective_task_id,
                        )
                    finally:
                        # Worker finished (interrupted or complete) —
                        # clear turn ownership immediately so a later
                        # stop/cancel can't reap background work this
                        # run deliberately left running (same race-window
                        # guard as gateway/run.py and _run_agent above).
                        _clear_turn_process_ownership(agent)
                        # /v1/runs owns its agent lifecycle, so it records
                        # the declared conversation itself rather than
                        # through _run_agent's bind_declared_conversation
                        # -- carrying the same precedence gate, which an
                        # explicit body session_id turns off.
                        if _declared_selected:
                            self._bind_declared_conversation(
                                getattr(agent, "session_id", None) or session_id,
                                gateway_session_key,
                            )
                        try:
                            unregister_gateway_notify(approval_session_key)
                        finally:
                            if approval_token is not None:
                                try:
                                    reset_current_session_key(approval_token)
                                except Exception:
                                    pass
                            if session_tokens:
                                try:
                                    clear_session_vars(session_tokens)
                                except Exception:
                                    pass
                            if room_policy_token is not None:
                                try:
                                    from gateway.hosted_room_execution_policy import (
                                        reset_room_execution_policy,
                                    )

                                    reset_room_execution_policy(room_policy_token)
                                except Exception:
                                    pass
                    u = {
                        "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
                        "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
                        "total_tokens": getattr(agent, "session_total_tokens", 0) or 0,
                    }
                    return r, u

            result, usage = await asyncio.get_running_loop().run_in_executor(None, _run_sync)
            if (
                run_id in self._stopping_run_ids
                and isinstance(result, dict)
                and result.get("interrupted") is True
            ):
                _put_event_if_active({
                    "event": "run.cancelled",
                    "run_id": run_id,
                    "timestamp": time.time(),
                })
                self._set_run_status(
                    run_id,
                    "cancelled",
                    last_event="run.cancelled",
                )
            # Check for structured failure (non-retryable client errors like
            # 401/400 return failed=True instead of raising, so the except
            # block below never fires — issue #15561).
            elif isinstance(result, dict) and result.get("failed"):
                error_msg = _redact_api_error_text(result.get("error") or "agent run failed")
                _put_event_if_active({
                    "event": "run.failed",
                    "run_id": run_id,
                    "timestamp": time.time(),
                    "error": error_msg,
                })
                self._set_run_status(
                    run_id,
                    "failed",
                    error=error_msg,
                    last_event="run.failed",
                )
            else:
                final_response = result.get("final_response", "") if isinstance(result, dict) else ""
                # Undelivered steer text (accepted after the final response;
                # see turn_finalizer) rides on the terminal event/status so
                # the client can replay it as the next user turn.
                pending_steer = result.get("pending_steer") if isinstance(result, dict) else None
                completed_event = {
                    "event": "run.completed",
                    "run_id": run_id,
                    "timestamp": time.time(),
                    "output": final_response,
                    "usage": usage,
                }
                if pending_steer:
                    completed_event["pending_steer"] = pending_steer
                _put_event_if_active(completed_event)
                self._set_run_status(
                    run_id,
                    "completed",
                    output=final_response,
                    usage=usage,
                    last_event="run.completed",
                    **({"pending_steer": pending_steer} if pending_steer else {}),
                )
        except asyncio.CancelledError:
            self._set_run_status(
                run_id,
                "cancelled",
                last_event="run.cancelled",
            )
            try:
                _put_event_if_active({
                    "event": "run.cancelled",
                    "run_id": run_id,
                    "timestamp": time.time(),
                })
            except Exception:
                pass
            raise
        except _ProviderAuthResolutionError as exc:
            # /v1/runs builds its own agent via _create_agent() and does
            # not route through _run_agent() (see that method's own
            # _ProviderAuthResolutionError branch), so it needs its own
            # handling to surface the same distinguished, controlled
            # message the other endpoints give a provider auth/credential
            # failure, instead of falling through to the generic
            # except-Exception branch below.
            logger.warning("Provider authentication failed for run=%s: %s", run_id, exc)
            error_msg = f"⚠️ Provider authentication failed: {exc}"
            self._set_run_status(
                run_id,
                "failed",
                error=error_msg,
                last_event="run.failed",
            )
            try:
                _put_event_if_active({
                    "event": "run.failed",
                    "run_id": run_id,
                    "timestamp": time.time(),
                    "error": error_msg,
                })
            except Exception:
                pass
        except Exception as exc:
            logger.exception("[api_server] run %s failed", run_id)
            self._set_run_status(
                run_id,
                "failed",
                error=_redact_api_error_text(exc),
                last_event="run.failed",
            )
            try:
                _put_event_if_active({
                    "event": "run.failed",
                    "run_id": run_id,
                    "timestamp": time.time(),
                    "error": _redact_api_error_text(exc),
                })
            except Exception:
                pass
        finally:
            # If the asyncio wrapper is cancelled (for example via
            # /stop), the executor thread can still be blocked waiting
            # on an approval Event. Unregistering here releases those
            # waits immediately; the in-thread unregister is harmlessly
            # idempotent on normal completion.
            try:
                from tools.approval import unregister_gateway_notify

                unregister_gateway_notify(approval_session_key)
            except Exception:
                pass
            # Sentinel: signal SSE stream to close
            try:
                _put_event_if_active(None)
            except Exception:
                pass
            self._active_run_agents.pop(run_id, None)
            self._active_run_tasks.pop(run_id, None)
            self._run_approval_sessions.pop(run_id, None)
            self._stopping_run_ids.discard(run_id)
            self._release_run_owner_if_forgotten(run_id)

    self._activate_admitted_request()
    task = asyncio.create_task(_run_and_close())
    self._active_run_tasks[run_id] = task
    try:
        self._background_tasks.add(task)
    except TypeError:
        pass
    if hasattr(task, "add_done_callback"):
        task.add_done_callback(self._background_tasks.discard)

    response_headers = (
        {"X-Hermes-Session-Key": gateway_session_key} if gateway_session_key else {}
    )
    return web.json_response(
        {"run_id": run_id, "status": "started", "replayed": False},
        status=202,
        headers=response_headers,
    )


def _release_run_owner_if_forgotten(self, run_id: str) -> None:
    """Drop the owner stamp only once nothing keyed by *run_id* survives.

    Ownership must outlive every surface it protects (statuses, live
    agent/task refs, SSE transport, approval sessions), which are retired
    on different clocks. Releasing earlier would leave a stateful run
    without an owner, which ``_request_owns_run`` treats as fail-closed.
    """
    if (
        run_id in self._run_statuses
        or run_id in self._active_run_agents
        or run_id in self._active_run_tasks
        or run_id in self._run_streams
        or run_id in self._run_approval_sessions
    ):
        return
    self._run_owners.pop(run_id, None)


def _request_owns_run(self, request: "web.Request", run_id: str) -> bool:
    scope = self._run_idempotency_scope(request)
    owner = self._run_owners.get(run_id)
    if owner is not None:
        return owner == scope
    # No in-memory owner: only a durable record under the caller's own scope
    # admits it. Run state that exists without an owner stamp is an
    # unanswered authorization question, not a run anyone may control —
    # under gateway.multiplex_profiles every served profile holds a valid
    # key, so admitting it would make the boundary allow-all (#93689).
    return self._run_idempotency_store.owns_run(scope, run_id)


async def _handle_get_run(
    self,
    request: "web.Request",
    *,
    _api_server,
) -> "web.Response":
    """GET /v1/runs/{run_id} — return pollable run status for external UIs."""
    _openai_error = _api_server._openai_error

    auth_err = self._check_run_auth(request, permission="status")
    if auth_err:
        return auth_err

    run_id = request.match_info["run_id"]
    if not self._request_owns_run(request, run_id):
        return web.json_response(
            _openai_error(f"Run not found: {run_id}", code="run_not_found"),
            status=404,
        )
    agent = self._active_run_agents.get(run_id)
    task = self._active_run_tasks.get(run_id)
    status = self._durable_run_status(request, run_id)
    if status is None and (agent is not None or task is not None):
        # Compatibility for in-process integrations that registered the
        # active run object before pollable status existed.
        status = self._set_run_status(run_id, "running")
    if status is None:
        return web.json_response(
            _openai_error(f"Run not found: {run_id}", code="run_not_found"),
            status=404,
        )
    return web.json_response(status)


async def _handle_run_events(
    self,
    request: "web.Request",
    *,
    _api_server,
) -> "web.StreamResponse":
    """GET /v1/runs/{run_id}/events — stream structured agent lifecycle events."""
    _openai_error = _api_server._openai_error
    _sse_frame = _api_server._sse_frame

    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err

    run_id = request.match_info["run_id"]
    if not self._request_owns_run(request, run_id):
        return web.json_response(
            _openai_error(f"Run not found: {run_id}", code="run_not_found"),
            status=404,
        )

    # Allow subscribing slightly before the run is registered (race condition window)
    for _ in range(20):
        if run_id in self._run_streams:
            break
        await asyncio.sleep(0.05)
    else:
        return web.json_response(_openai_error(f"Run not found: {run_id}", code="run_not_found"), status=404)

    q = self._run_streams[run_id]
    self._run_stream_subscribers.add(run_id)

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)

    try:
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=30.0)
            except asyncio.TimeoutError:
                await response.write(b": keepalive\n\n")
                continue
            if event is None:
                # Run finished — send final SSE comment and close
                await response.write(b": stream closed\n\n")
                break
            payload = _sse_frame(event)
            await response.write(payload)
    except Exception as exc:
        logger.debug("[api_server] SSE stream error for run %s: %s", run_id, exc)
    finally:
        self._run_stream_subscribers.discard(run_id)
        self._run_streams.pop(run_id, None)
        self._run_streams_created.pop(run_id, None)
        self._release_run_owner_if_forgotten(run_id)

    return response


async def _handle_run_approval(
    self,
    request: "web.Request",
    *,
    _api_server,
) -> "web.Response":
    """POST /v1/runs/{run_id}/approval — resolve a pending run approval."""
    _coerce_request_bool = _api_server._coerce_request_bool
    _openai_error = _api_server._openai_error

    auth_err = self._check_run_auth(request, permission="approve")
    if auth_err:
        return auth_err

    run_id = request.match_info["run_id"]
    if not self._request_owns_run(request, run_id):
        return web.json_response(
            _openai_error(f"Run not found: {run_id}", code="run_not_found"),
            status=404,
        )
    status = self._durable_run_status(request, run_id)
    if status is None:
        return web.json_response(
            _openai_error(f"Run not found: {run_id}", code="run_not_found"),
            status=404,
        )

    try:
        body = await request.json()
    except Exception:
        return web.json_response(_openai_error("Invalid JSON"), status=400)

    raw_choice = str(body.get("choice", "")).strip().lower()
    aliases = {"approve": "once", "approved": "once", "allow": "once"}
    choice = aliases.get(raw_choice, raw_choice)
    room_scoped = bool(self._room_grant_token(request))
    raw_request_id = body.get("request_id")
    request_id = raw_request_id.strip() if isinstance(raw_request_id, str) else ""
    if raw_request_id is not None and (not request_id or len(request_id) > 256):
        return web.json_response(
            _openai_error(
                "Approval request_id is invalid.",
                code="invalid_approval_request",
            ),
            status=400,
        )
    allowed = {"once", "deny"} if room_scoped else {
        "once",
        "session",
        "always",
        "deny",
    }
    if choice not in allowed:
        return web.json_response(
            _openai_error(
                "Invalid approval choice; expected one of: "
                + ", ".join(sorted(allowed)),
                code="invalid_approval_choice",
            ),
            status=400,
        )

    resolve_all = (
        _coerce_request_bool(body.get("all"), default=False)
        or _coerce_request_bool(body.get("resolve_all"), default=False)
    )
    if room_scoped and resolve_all:
        return web.json_response(
            _openai_error(
                "Room approvals can resolve only one exact request",
                code="invalid_approval_scope",
            ),
            status=400,
        )
    if room_scoped and not request_id:
        return web.json_response(
            _openai_error(
                "Room approvals require the exact request_id.",
                code="approval_request_required",
            ),
            status=400,
        )

    approval_session_key = self._run_approval_sessions.get(run_id)
    if not approval_session_key:
        return web.json_response(
            _openai_error(
                f"Run has no active approval session: {run_id}",
                code="approval_not_active",
            ),
            status=409,
        )
    try:
        from tools.approval import resolve_gateway_approval

        resolved = resolve_gateway_approval(
            approval_session_key,
            choice,
            resolve_all=resolve_all,
            request_id=request_id or None,
        )
    except Exception as exc:
        logger.exception("[api_server] approval resolution failed for run %s", run_id)
        return web.json_response(_openai_error(str(exc)), status=500)

    if resolved <= 0:
        return web.json_response(
            _openai_error(
                f"Run has no pending approval: {run_id}",
                code="approval_not_pending",
            ),
            status=409,
        )

    self._set_run_status(run_id, "running", last_event="approval.responded")
    q = self._run_streams.get(run_id)
    if q is not None:
        try:
            q.put_nowait({
                "event": "approval.responded",
                "run_id": run_id,
                "timestamp": time.time(),
                "choice": choice,
                **({"request_id": request_id} if request_id else {}),
                "resolved": resolved,
            })
        except Exception:
            pass

    return web.json_response({
        "object": "hermes.run.approval_response",
        "run_id": run_id,
        "choice": choice,
        **({"request_id": request_id} if request_id else {}),
        "resolved": resolved,
    })


async def _handle_steer_run(
    self,
    request: "web.Request",
    *,
    _api_server,
) -> "web.Response":
    """POST /v1/runs/{run_id}/steer — inject guidance into a running agent."""
    _normalize_chat_content = _api_server._normalize_chat_content
    _openai_error = _api_server._openai_error
    _redact_api_error_text = _api_server._redact_api_error_text

    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err

    run_id = request.match_info["run_id"]
    if not self._request_owns_run(request, run_id):
        return web.json_response(
            _openai_error(f"Run not found: {run_id}", code="run_not_found"),
            status=404,
        )
    status = self._durable_run_status(request, run_id)
    if status is None:
        return web.json_response(
            _openai_error(f"Run not found: {run_id}", code="run_not_found"),
            status=404,
        )
    # Only genuinely running runs are steerable. /stop retains agent/task
    # refs during cooperative shutdown, so the status gate (not the mere
    # presence of an agent ref) is what rejects stop-then-steer.
    agent = self._active_run_agents.get(run_id)
    if status.get("status") != "running" or not hasattr(agent, "steer"):
        return web.json_response(
            _openai_error(
                f"Run is not currently accepting steer input: {run_id}",
                code="run_not_accepting_steer",
            ),
            status=409,
        )

    body, err = await self._read_json_body(request)
    if err:
        return err
    raw_text = body.get("input") or body.get("message") or body.get("text") or ""
    steer_text = _normalize_chat_content(raw_text).strip()
    if not steer_text:
        return web.json_response(
            _openai_error(
                "Missing non-empty steer text; expected 'input', 'message', or 'text'.",
                code="invalid_steer_input",
            ),
            status=400,
        )

    try:
        accepted = bool(agent.steer(steer_text))
    except Exception as exc:
        logger.exception("[api_server] steer failed for run %s", run_id)
        return web.json_response(_openai_error(_redact_api_error_text(exc), code="steer_failed"), status=500)
    if not accepted:
        return web.json_response(
            _openai_error(f"Run did not accept steer text: {run_id}", code="steer_not_accepted"),
            status=409,
        )

    self._set_run_status(run_id, "running", last_event="run.steered")
    q = self._run_streams.get(run_id)
    if q is not None:
        with suppress(Exception):
            q.put_nowait({
                "event": "run.steered",
                "run_id": run_id,
                "timestamp": time.time(),
                "accepted": True,
            })
    return web.json_response({"object": "hermes.run.steer", "run_id": run_id, "accepted": True})


async def _handle_stop_run(
    self,
    request: "web.Request",
    *,
    _api_server,
) -> "web.Response":
    """POST /v1/runs/{run_id}/stop — interrupt a running agent."""
    _openai_error = _api_server._openai_error
    _reap_disconnected_agent_processes = (
        _api_server._reap_disconnected_agent_processes
    )
    request_hard_interrupt = _api_server.request_hard_interrupt

    auth_err = self._check_run_auth(request, permission="stop")
    if auth_err:
        return auth_err

    run_id = request.match_info["run_id"]
    if not self._request_owns_run(request, run_id):
        return web.json_response(
            _openai_error(f"Run not found: {run_id}", code="run_not_found"),
            status=404,
        )
    agent = self._active_run_agents.get(run_id)
    task = self._active_run_tasks.get(run_id)
    status = self._durable_run_status(request, run_id)
    if status is None and (agent is not None or task is not None):
        # Compatibility for in-process integrations that registered the
        # active run object before pollable status existed.
        status = self._set_run_status(run_id, "running")
    if status is None:
        return web.json_response(
            _openai_error(f"Run not found: {run_id}", code="run_not_found"),
            status=404,
        )
    if status.get("status") in {
        "completed",
        "failed",
        "cancelled",
        "interrupted",
    }:
        return web.json_response(status)

    if agent is None and task is None:
        return web.json_response(
            _openai_error(
                f"Run is not active in this gateway process: {run_id}",
                code="run_not_active",
            ),
            status=409,
        )

    self._set_run_status(run_id, "stopping", last_event="run.stopping")
    self._stopping_run_ids.add(run_id)

    if agent is not None:
        try:
            request_hard_interrupt(agent, "Stop requested via API")
        except Exception:
            pass
        # The stopped run is abandoned — reap only the background
        # processes it created (#76115). Epoch-gated inside, so a
        # concurrent run sharing the same session_id keeps its own
        # processes; no-op if the run already finished and cleared
        # its ownership markers.
        _reap_disconnected_agent_processes(
            agent, source="api_server_run_stop"
        )

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
    stale = [
        run_id
        for run_id, created_at in list(self._run_streams_created.items())
        if now - created_at > self._RUN_STREAM_TTL
        and run_id not in self._run_stream_subscribers
    ]
    for run_id in stale:
        logger.debug("[api_server] sweeping expired run transport %s", run_id)
        task = self._active_run_tasks.get(run_id)
        task_done = task is None or task.done()
        if task_done:
            try:
                from tools.approval import unregister_gateway_notify

                approval_session_key = self._run_approval_sessions.get(run_id)
                if approval_session_key:
                    unregister_gateway_notify(approval_session_key)
            except Exception:
                pass
        # The transport TTL always bounds buffering. Live control state is
        # independent and survives until the executor-backed task returns.
        self._run_streams.pop(run_id, None)
        self._run_streams_created.pop(run_id, None)
        if task_done:
            self._active_run_agents.pop(run_id, None)
            self._active_run_tasks.pop(run_id, None)
            self._run_approval_sessions.pop(run_id, None)
            self._stopping_run_ids.discard(run_id)
        self._release_run_owner_if_forgotten(run_id)

    stale_statuses = [
        run_id
        for run_id, status in list(self._run_statuses.items())
        if status.get("status") in {"completed", "failed", "cancelled"}
        and now - float(status.get("updated_at", 0) or 0) > self._RUN_STATUS_TTL
    ]
    for run_id in stale_statuses:
        self._run_statuses.pop(run_id, None)
        self._run_idempotency_ids.discard(run_id)
        self._release_run_owner_if_forgotten(run_id)
