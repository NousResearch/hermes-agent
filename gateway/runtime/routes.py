"""HTTP route handlers for the /v1/runs runtime API.

Provides an aiohttp-compatible route module that delegates to
``RunManager``.  Call ``register_runtime_routes(app)`` to register
the six /v1/runs endpoints on an existing ``web.Application``.

The module is self-contained and can be tested standalone or mounted
into the live API server (``gateway.platforms.api_server``).
"""

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

    Returns
    -------
    RunManager
        The manager used by the registered handlers.

    Endpoints registered
    --------------------
    POST   /v1/runs
    GET    /v1/runs/{run_id}
    GET    /v1/runs/{run_id}/events
    POST   /v1/runs/{run_id}/stop
    POST   /v1/runs/{run_id}/approval
    POST   /v1/runs/{run_id}/clarify
    """
    if run_manager is None:
        run_manager = RunManager()

    if error_formatter is None:
        error_formatter = _standard_error

    app["runtime_run_manager"] = run_manager

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
                    msg_texts.append(part.get("content", ""))
                elif isinstance(part, str):
                    msg_texts.append(part)
            message = " ".join(msg_texts) if msg_texts else None

        result = run_manager.create_run(
            session_id=session_id or "default",
            message=message,
            workspace=workspace,
            profile=profile,
            model=model,
            toolsets=toolsets,
            metadata=metadata,
        )

        return web.json_response(
            {
                "object": "hermes.run",
                "run_id": result["run_id"],
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

        try:
            body = await request.json()
        except Exception:
            return web.json_response(
                error_formatter("Invalid JSON"), status=400
            )

        choice = str(body.get("choice", "")).strip()
        if not choice:
            return web.json_response(
                error_formatter("Missing 'choice' field", code="missing_field"),
                status=400,
            )

        result = run_manager.resolve_approval(run_id, choice)

        if result.get("error") == "not_found":
            return web.json_response(
                error_formatter(
                    f"Run not found: {run_id}",
                    code="run_not_found",
                ),
                status=404,
            )

        if result.get("error") == "not_supported":
            return web.json_response(
                {
                    "object": "hermes.run.approval_response",
                    "run_id": run_id,
                    "status": "not_supported",
                    "message": result.get("message", ""),
                },
                status=501,
            )

        return web.json_response(
            {
                "object": "hermes.run.approval_response",
                **result,
            }
        )

    async def _handle_clarify(request: web.Request) -> web.Response:
        run_id = request.match_info["run_id"]

        try:
            body = await request.json()
        except Exception:
            return web.json_response(
                error_formatter("Invalid JSON"), status=400
            )

        response_text = body.get("response") or body.get("text") or ""
        if not response_text:
            return web.json_response(
                error_formatter(
                    "Missing 'response' field", code="missing_field"
                ),
                status=400,
            )

        result = run_manager.resolve_clarify(run_id, str(response_text))

        if result.get("error") == "not_found":
            return web.json_response(
                error_formatter(
                    f"Run not found: {run_id}",
                    code="run_not_found",
                ),
                status=404,
            )

        if result.get("error") == "not_supported":
            return web.json_response(
                {
                    "object": "hermes.run.clarify_response",
                    "run_id": run_id,
                    "status": "not_supported",
                    "message": result.get("message", ""),
                },
                status=501,
            )

        return web.json_response(
            {
                "object": "hermes.run.clarify_response",
                **result,
            }
        )

    app.router.add_post("/v1/runs", _handle_create_run)
    app.router.add_get("/v1/runs/{run_id}", _handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", _handle_run_events)
    app.router.add_post("/v1/runs/{run_id}/stop", _handle_stop_run)
    app.router.add_post("/v1/runs/{run_id}/approval", _handle_approval)
    app.router.add_post("/v1/runs/{run_id}/clarify", _handle_clarify)

    return run_manager
