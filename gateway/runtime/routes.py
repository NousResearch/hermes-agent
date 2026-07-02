"""HTTP route handlers for the /v1/runs runtime API.

Provides an aiohttp-compatible route module that delegates to
``RunManager``.  Call ``register_runtime_routes(app)`` to register
the six /v1/runs endpoints on an existing ``web.Application``.

The module is self-contained and can be tested standalone or mounted
into the live API server (``gateway.platforms.api_server``).

When a ``RuntimeControlBridge`` is supplied via the *control_bridge*
parameter, approval / clarify / stop handlers delegate through the
bridge so live AIAgent execution can be interrupted or unblocked.
Without a bridge the handlers fall back to standalone ``RunManager``
behaviour (Phase 11B).

When a ``RuntimeExecutor`` is supplied via the *executor* parameter,
POST /v1/runs with ``execute: true`` in the body will hand the
created run to the executor for background processing.  Without an
executor, the route remains control-plane-only (backward compatible).
"""

import asyncio
import json
from typing import Any, Callable, Dict, Optional

from aiohttp import web

from gateway.runtime.run_manager import RunManager


def _standard_error(
    message: str,
    err_type: str = "invalid_request_error",
    code: str = None,
) -> Dict[str, Any]:
    return {
        "error": {
            "message": str(message),
            "type": err_type,
            "code": code,
        }
    }


def register_runtime_routes(
    app: web.Application,
    *,
    run_manager: Optional[RunManager] = None,
    error_formatter: Optional[Callable[..., Dict[str, Any]]] = None,
    control_bridge: Optional[Any] = None,
    executor: Optional[Any] = None,
    register_create: bool = True,
    register_status: bool = True,
    register_events: bool = True,
) -> RunManager:
    """Register /v1/runs runtime handlers on *app* and return the RunManager.

    Parameters
    ----------
    app:
        The aiohttp ``web.Application`` to register routes on.
    run_manager:
        An existing ``RunManager`` instance.  If ``None`` a fresh one
        is created and stored on ``app["runtime_run_manager"]``.
    error_formatter:
        Optional ``(message, err_type, code) -> dict`` callback for
        error response bodies.  Defaults to a simple OpenAPI-style
        envelope.
    control_bridge:
        Optional ``RuntimeControlBridge`` instance.  When provided,
        approval / clarify / stop handlers delegate through the bridge
        so live AIAgent execution can be interrupted or unblocked.
        Without it, handlers use standalone ``RunManager`` only.
    executor:
        Optional ``RuntimeExecutor`` instance.  When provided and the
        request body includes ``execute: true``, the created run is
        handed to the executor for background processing.  Without an
        executor the route remains control-plane-only.
    register_create:
        When ``False``, the ``POST /v1/runs`` create endpoint is
        skipped.  Callers that want to own run creation (e.g. the
        API server's legacy agent-run handler) set this to ``False``
        and register their own ``POST /v1/runs`` handler that
        coordinates agent spawn + RunManager entry + bridge binding.
    register_status:
        When ``False``, the ``GET /v1/runs/{run_id}`` status endpoint
        is skipped (caller provides their own).
    register_events:
        When ``False``, the ``GET /v1/runs/{run_id}/events`` endpoint
        is skipped (caller provides their own).

    Returns
    -------
    RunManager
        The manager used by the registered handlers.

    Endpoints registered
    --------------------
    POST   /v1/runs           (skipped when *register_create* is False)
    GET    /v1/runs/{run_id}  (skipped when *register_status* is False)
    GET    /v1/runs/{run_id}/events  (skipped when *register_events* is False)
    POST   /v1/runs/{run_id}/stop
    POST   /v1/runs/{run_id}/approval
    POST   /v1/runs/{run_id}/clarify
    """
    if run_manager is None:
        run_manager = RunManager()

    if error_formatter is None:
        error_formatter = _standard_error

    app["runtime_run_manager"] = run_manager
    if control_bridge is not None:
        app["runtime_control_bridge"] = control_bridge
    if executor is not None:
        app["runtime_executor"] = executor

    async def _handle_create_run(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response(
                error_formatter("Invalid JSON"), status=400
            )

        session_id = body.get("session_id")
        message = body.get("message") or body.get("input")

        if not session_id and not message:
            session_id = "default"

        workspace = body.get("workspace")
        profile = body.get("profile")
        model = body.get("model")
        toolsets = body.get("toolsets")
        metadata = body.get("metadata")

        if isinstance(message, list):
            msg_texts = []
            for part in message:
                if isinstance(part, dict):
                    part_type = str(part.get("type") or "").strip().lower()
                    if part_type in {"text", "input_text", "output_text"}:
                        msg_texts.append(part.get("text", ""))
                    else:
                        msg_texts.append(part.get("content", ""))
                elif isinstance(part, str):
                    msg_texts.append(part)
            message = " ".join(s for s in msg_texts if s) or None

        result = run_manager.create_run(
            session_id=session_id or "default",
            message=message,
            workspace=workspace,
            profile=profile,
            model=model,
            toolsets=toolsets,
            metadata=metadata,
        )

        run_id = result["run_id"]
        execute_flag = body.get("execute", False)
        executor_instance = request.app.get("runtime_executor")

        if execute_flag and executor_instance is not None:
            asyncio.create_task(executor_instance.execute_run(run_id))

        return web.json_response(
            {
                "object": "hermes.run",
                "run_id": run_id,
                "session_id": result["session_id"],
                "status": result["status"],
                "events_url": result["events_url"],
                "status_url": result["status_url"],
                "controls": result["controls"],
            },
            status=202,
        )

    async def _handle_get_run(request: web.Request) -> web.Response:
        run_id = request.match_info["run_id"]
        status = run_manager.get_status(run_id)
        if status is None:
            return web.json_response(
                error_formatter(
                    f"Run not found: {run_id}",
                    code="run_not_found",
                ),
                status=404,
            )
        return web.json_response(
            {
                "object": "hermes.run",
                **status,
            }
        )

    async def _handle_run_events(request: web.Request) -> web.Response:
        run_id = request.match_info["run_id"]

        after_seq_raw = request.query.get("after_seq")
        limit_raw = request.query.get("limit")

        after_seq = None
        if after_seq_raw is not None:
            try:
                after_seq = int(after_seq_raw)
            except (TypeError, ValueError):
                return web.json_response(
                    error_formatter(
                        "after_seq must be an integer",
                        code="invalid_query_parameter",
                    ),
                    status=400,
                )

        limit = None
        if limit_raw is not None:
            try:
                limit = int(limit_raw)
            except (TypeError, ValueError):
                return web.json_response(
                    error_formatter(
                        "limit must be an integer",
                        code="invalid_query_parameter",
                    ),
                    status=400,
                )

        accept_sse = "text/event-stream" in request.headers.get(
            "Accept", ""
        )

        result = run_manager.read_events(
            run_id, after_seq=after_seq, limit=limit
        )

        if result is None:
            return web.json_response(
                error_formatter(
                    f"Run not found: {run_id}",
                    code="run_not_found",
                ),
                status=404,
            )

        if accept_sse:
            response = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                },
            )
            await response.prepare(request)
            for event in result["events"]:
                payload = f"data: {json.dumps(event)}\n\n"
                await response.write(payload.encode())
            await response.write(b": stream closed\n\n")
            return response

        return web.json_response(
            {
                "object": "hermes.run.events",
                **result,
            }
        )

    async def _handle_stop_run(request: web.Request) -> web.Response:
        run_id = request.match_info["run_id"]
        bridge = request.app.get("runtime_control_bridge")

        if bridge is not None:
            result = bridge.stop_run(run_id)
        else:
            result = run_manager.stop_run(run_id)

        if result.get("error") == "not_found":
            return web.json_response(
                error_formatter(
                    f"Run not found: {run_id}",
                    code="run_not_found",
                ),
                status=404,
            )
        return web.json_response(
            {
                "object": "hermes.run",
                **result,
            }
        )

    async def _handle_approval(request: web.Request) -> web.Response:
        run_id = request.match_info["run_id"]
        bridge = request.app.get("runtime_control_bridge")

        try:
            body = await request.json()
        except Exception:
            return web.json_response(
                error_formatter("Invalid JSON"), status=400
            )

        body_run_id = str(body.get("run_id", "")).strip()
        if body_run_id and body_run_id != run_id:
            return web.json_response(
                error_formatter(
                    "run_id mismatch: URL path run_id must match or be omitted from body",
                    code="invalid_request_error",
                ),
                status=400,
            )

        approval_id = str(body.get("approval_id", "")).strip()
        choice = str(body.get("choice", "")).strip()
        if not choice:
            return web.json_response(
                error_formatter("Missing 'choice' field", code="missing_field"),
                status=400,
            )

        if bridge is not None:
            result = bridge.resolve_approval(
                run_id, approval_id, choice,
                payload=body.get("payload"),
            )
        else:
            result = run_manager.resolve_approval(
                run_id, approval_id, choice,
            )

        if result.get("error") == "not_found":
            msg = str(result.get("message", ""))
            code = "run_not_found"
            if "Approval" in msg:
                code = "action_not_found"
            return web.json_response(
                error_formatter(msg, code=code),
                status=404,
            )

        if result.get("error") == "conflict":
            return web.json_response(
                error_formatter(
                    str(result.get("message", "")),
                    code="conflict",
                ),
                status=409,
            )

        return web.json_response(
            {
                "object": "hermes.run.approval_response",
                **result,
            }
        )

    async def _handle_clarify(request: web.Request) -> web.Response:
        run_id = request.match_info["run_id"]
        bridge = request.app.get("runtime_control_bridge")

        try:
            body = await request.json()
        except Exception:
            return web.json_response(
                error_formatter("Invalid JSON"), status=400
            )

        body_run_id = str(body.get("run_id", "")).strip()
        if body_run_id and body_run_id != run_id:
            return web.json_response(
                error_formatter(
                    "run_id mismatch: URL path run_id must match or be omitted from body",
                    code="invalid_request_error",
                ),
                status=400,
            )

        clarify_id = str(body.get("clarify_id", "")).strip()
        response_text = body.get("response") or body.get("text") or body.get("answer") or ""
        if not response_text:
            return web.json_response(
                error_formatter(
                    "Missing 'response' field", code="missing_field"
                ),
                status=400,
            )

        if bridge is not None:
            result = bridge.resolve_clarify(
                run_id, clarify_id, str(response_text),
                payload=body.get("payload"),
            )
        else:
            result = run_manager.resolve_clarify(
                run_id, clarify_id, str(response_text),
            )

        if result.get("error") == "not_found":
            msg = str(result.get("message", ""))
            code = "run_not_found"
            if "Clarify" in msg:
                code = "action_not_found"
            return web.json_response(
                error_formatter(msg, code=code),
                status=404,
            )

        if result.get("error") == "conflict":
            return web.json_response(
                error_formatter(
                    str(result.get("message", "")),
                    code="conflict",
                ),
                status=409,
            )

        return web.json_response(
            {
                "object": "hermes.run.clarify_response",
                **result,
            }
        )

    if register_create:
        app.router.add_post("/v1/runs", _handle_create_run)
    if register_status:
        app.router.add_get("/v1/runs/{run_id}", _handle_get_run)
    if register_events:
        app.router.add_get("/v1/runs/{run_id}/events", _handle_run_events)
    app.router.add_post("/v1/runs/{run_id}/stop", _handle_stop_run)
    app.router.add_post("/v1/runs/{run_id}/approval", _handle_approval)
    app.router.add_post("/v1/runs/{run_id}/clarify", _handle_clarify)

    return run_manager
