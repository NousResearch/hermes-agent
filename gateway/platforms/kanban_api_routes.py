"""Gateway /v1/kanban/* routes — aiohttp adapter over hermes_cli.kanban_http."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

from aiohttp import web

from hermes_cli import kanban_http as kh
from gateway.read_model_cache import read_model_etag, read_model_metric_headers, request_fingerprint

if TYPE_CHECKING:
    from gateway.platforms.api_server import APIServerAdapter

logger = logging.getLogger(__name__)


def _json(data: Any, *, status: int = 200) -> web.Response:
    return web.json_response(data, status=status)


def _error(exc: Exception) -> web.Response:
    if isinstance(exc, kh.KanbanAPIError):
        return _json({"detail": exc.detail}, status=exc.status_code)
    if isinstance(exc, ValueError):
        return _json({"detail": str(exc)}, status=400)
    logger.exception("kanban gateway error")
    return _json({"detail": str(exc)}, status=500)


def _body(request: web.Request) -> dict:
    if request.body_exists:
        try:
            return asyncio.get_event_loop().run_until_complete(request.json())  # noqa: not used sync
        except Exception:
            pass
    return {}


def _board_fingerprint(q: Any) -> str:
    import hashlib
    import time

    board = kh.resolve_board(q.get("board"))
    try:
        conn = kh._conn(board=board)
        try:
            latest_event = conn.execute("SELECT COALESCE(MAX(id), 0) AS value FROM task_events").fetchone()["value"]
            task_count = conn.execute("SELECT COUNT(*) AS value FROM tasks").fetchone()["value"]
        finally:
            conn.close()
    except Exception:
        latest_event = int(time.time())
        task_count = 0
    parts = [
        "kanban-board",
        str(board),
        str(q.get("tenant") or ""),
        str(q.get("include_archived") or ""),
        str(q.get("workflow_template_id") or ""),
        str(q.get("current_step_key") or ""),
        str(latest_event),
        str(task_count),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


async def _read_json(request: web.Request) -> dict:
    if not request.body_exists:
        return {}
    try:
        data = await request.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def register_kanban_routes(app: web.Application, adapter: "APIServerAdapter") -> None:
    check: Callable[[web.Request], Optional[web.Response]] = adapter._check_auth

    async def _guard(request: web.Request) -> Optional[web.Response]:
        return check(request)

    async def handle_board(request: web.Request) -> web.Response:
        if err := await _guard(request):
            return err
        try:
            q = request.rel_url.query
            fingerprint = _board_fingerprint(q)
            key = (
                f"kanban:board:{q.get('board') or ''}:{q.get('tenant') or ''}:"
                f"{q.get('include_archived') or ''}:{q.get('workflow_template_id') or ''}:"
                f"{q.get('current_step_key') or ''}"
            )
            cached = await adapter._read_model_cache.get_or_compute(
                key=key,
                fingerprint=fingerprint,
                compute=lambda: kh.http_get_board(
                    q.get("tenant"),
                    q.get("include_archived", "false").lower() in {"1", "true", "yes"},
                    q.get("board"),
                    q.get("workflow_template_id"),
                    q.get("current_step_key"),
                ),
            )
            if request_fingerprint(request) == cached.fingerprint:
                headers = {"ETag": read_model_etag(cached.fingerprint)}
                headers.update(read_model_metric_headers(cached, payload_size_bytes=0))
                logger.info(
                    "oryn read-model kanban.board status=304 cache=%s total_ms=%.2f compute_ms=%.2f bytes=0",
                    cached.cache_status,
                    cached.total_ms,
                    cached.compute_ms,
                )
                return web.Response(status=304, headers=headers)
            data = dict(cached.payload)
            data.setdefault("fingerprint", cached.fingerprint)
            payload_size = len(json.dumps(data, ensure_ascii=False, default=str).encode("utf-8"))
            response = _json(data)
            response.headers["ETag"] = read_model_etag(cached.fingerprint)
            response.headers.update(read_model_metric_headers(cached, payload_size_bytes=payload_size))
            logger.info(
                "oryn read-model kanban.board status=200 cache=%s total_ms=%.2f compute_ms=%.2f cache_read_ms=%.2f store_ms=%.2f bytes=%s",
                cached.cache_status,
                cached.total_ms,
                cached.compute_ms,
                cached.cache_read_ms,
                cached.store_ms,
                payload_size,
            )
            return response
        except Exception as exc:
            return _error(exc)

    async def handle_task_get(request: web.Request) -> web.Response:
        if err := await _guard(request):
            return err
        try:
            q = request.rel_url.query
            data = kh.http_get_task(
                request.match_info["task_id"],
                q.get("board"),
                q.get("run_state_type"),
                q.get("run_state_name"),
            )
            return _json(data)
        except Exception as exc:
            return _error(exc)

    async def handle_task_create(request: web.Request) -> web.Response:
        if err := await _guard(request):
            return err
        try:
            payload = await _read_json(request)
            payload.setdefault("created_by", "oryn-workspace")
            data = kh.http_create_task(kh.CreateTaskRequest(**payload), request.rel_url.query.get("board"))
            return _json(data)
        except Exception as exc:
            return _error(exc)

    async def handle_task_patch(request: web.Request) -> web.Response:
        if err := await _guard(request):
            return err
        try:
            payload = await _read_json(request)
            data = kh.http_update_task(
                request.match_info["task_id"],
                kh.UpdateTaskRequest(**payload),
                request.rel_url.query.get("board"),
            )
            return _json(data)
        except Exception as exc:
            return _error(exc)

    async def handle_task_delete(request: web.Request) -> web.Response:
        if err := await _guard(request):
            return err
        try:
            data = kh.http_delete_task(request.match_info["task_id"], request.rel_url.query.get("board"))
            return _json(data)
        except Exception as exc:
            return _error(exc)

    async def handle_bulk(request: web.Request) -> web.Response:
        if err := await _guard(request):
            return err
        try:
            payload = await _read_json(request)
            data = kh.http_bulk_update(kh.BulkTaskRequest(**payload), request.rel_url.query.get("board"))
            return _json(data)
        except Exception as exc:
            return _error(exc)

    async def handle_comment(request: web.Request) -> web.Response:
        if err := await _guard(request):
            return err
        try:
            payload = await _read_json(request)
            data = kh.http_add_comment(
                request.match_info["task_id"],
                kh.CommentRequest(**payload),
                request.rel_url.query.get("board"),
            )
            return _json(data)
        except Exception as exc:
            return _error(exc)

    async def handle_link_post(request: web.Request) -> web.Response:
        if err := await _guard(request):
            return err
        try:
            payload = await _read_json(request)
            data = kh.http_add_link(kh.LinkRequest(**payload), request.rel_url.query.get("board"))
            return _json(data)
        except Exception as exc:
            return _error(exc)

    async def handle_link_delete(request: web.Request) -> web.Response:
        if err := await _guard(request):
            return err
        try:
            q = request.rel_url.query
            data = kh.http_delete_link(q.get("parent_id", ""), q.get("child_id", ""), q.get("board"))
            return _json(data)
        except Exception as exc:
            return _error(exc)

    async def handle_config(request: web.Request) -> web.Response:
        if err := await _guard(request):
            return err
        return _json(kh.http_get_config())

    async def handle_boards_list(request: web.Request) -> web.Response:
        if err := await _guard(request):
            return err
        q = request.rel_url.query
        return _json(kh.http_list_boards(q.get("include_archived", "false").lower() in {"1", "true", "yes"}))

    async def handle_dispatch(request: web.Request) -> web.Response:
        if err := await _guard(request):
            return err
        try:
            q = request.rel_url.query
            dry = q.get("dry_run", "false").lower() in {"1", "true", "yes"}
            max_n = int(q.get("max", "8"))
            data = kh.http_dispatch(dry, max_n, q.get("board"))
            return _json(data)
        except Exception as exc:
            return _error(exc)

    async def handle_diagnostics(request: web.Request) -> web.Response:
        if err := await _guard(request):
            return err
        try:
            q = request.rel_url.query
            return _json(kh.http_list_diagnostics(q.get("board"), q.get("severity")))
        except Exception as exc:
            return _error(exc)

    async def handle_events(request: web.Request) -> web.StreamResponse:
        if err := await _guard(request):
            return err
        q = request.rel_url.query
        try:
            cursor = int(q.get("since", "0"))
        except ValueError:
            cursor = 0
        board = q.get("board")
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
                if await request.transport is None:
                    break
                new_cursor, events = await asyncio.to_thread(kh.fetch_events, cursor, board)
                if events:
                    cursor = new_cursor
                    payload = json.dumps({"events": events, "cursor": cursor})
                    await response.write(f"data: {payload}\n\n".encode())
                await asyncio.sleep(kh.EVENT_POLL_SECONDS)
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        except Exception as exc:
            logger.debug("kanban SSE closed: %s", exc)
        return response

    # passthrough helpers for remaining endpoints
    async def _simple(fn, request: web.Request, **kwargs) -> web.Response:
        if err := await _guard(request):
            return err
        try:
            return _json(fn(**kwargs))
        except Exception as exc:
            return _error(exc)

    app.router.add_get("/v1/kanban/board", handle_board)
    app.router.add_get("/v1/kanban/config", handle_config)
    app.router.add_get("/v1/kanban/boards", handle_boards_list)
    app.router.add_get("/v1/kanban/tasks/{task_id}", handle_task_get)
    app.router.add_post("/v1/kanban/tasks", handle_task_create)
    app.router.add_patch("/v1/kanban/tasks/{task_id}", handle_task_patch)
    app.router.add_delete("/v1/kanban/tasks/{task_id}", handle_task_delete)
    app.router.add_post("/v1/kanban/tasks/bulk", handle_bulk)
    app.router.add_post("/v1/kanban/tasks/{task_id}/comments", handle_comment)
    app.router.add_post("/v1/kanban/links", handle_link_post)
    app.router.add_delete("/v1/kanban/links", handle_link_delete)
    app.router.add_get("/v1/kanban/diagnostics", handle_diagnostics)
    app.router.add_post("/v1/kanban/dispatch", handle_dispatch)
    app.router.add_get("/v1/kanban/events", handle_events)

    async def handle_stats(request: web.Request) -> web.Response:
        return await _simple(kh.http_get_stats, request, board=request.rel_url.query.get("board"))

    async def handle_assignees(request: web.Request) -> web.Response:
        return await _simple(kh.http_get_assignees, request, board=request.rel_url.query.get("board"))

    async def handle_workers(request: web.Request) -> web.Response:
        return await _simple(kh.http_list_active_workers, request, board=request.rel_url.query.get("board"))

    async def handle_orchestration_get(request: web.Request) -> web.Response:
        return await _simple(lambda: kh.http_get_orchestration_settings(), request)

    app.router.add_get("/v1/kanban/stats", handle_stats)
    app.router.add_get("/v1/kanban/assignees", handle_assignees)
    app.router.add_get("/v1/kanban/workers/active", handle_workers)
    app.router.add_get("/v1/kanban/orchestration", handle_orchestration_get)



    async def handle_task_log(request: web.Request) -> web.Response:
        if err := await _guard(request):
            return err
        try:
            return _json(kh.http_get_task_log(request.match_info["task_id"], request.rel_url.query.get("board")))
        except Exception as exc:
            return _error(exc)

    async def handle_run_get(request: web.Request) -> web.Response:
        if err := await _guard(request):
            return err
        try:
            return _json(kh.http_get_run_endpoint(int(request.match_info["run_id"]), request.rel_url.query.get("board")))
        except Exception as exc:
            return _error(exc)

    async def handle_run_inspect(request: web.Request) -> web.Response:
        if err := await _guard(request):
            return err
        try:
            return _json(kh.http_inspect_run_endpoint(int(request.match_info["run_id"]), request.rel_url.query.get("board")))
        except Exception as exc:
            return _error(exc)

    async def handle_reclaim(request: web.Request) -> web.Response:
        if err := await _guard(request):
            return err
        try:
            payload = await _read_json(request)
            return _json(kh.http_reclaim_task_endpoint(request.match_info["task_id"], kh.ReclaimRequest(**payload), request.rel_url.query.get("board")))
        except Exception as exc:
            return _error(exc)

    async def handle_home_channels(request: web.Request) -> web.Response:
        if err := await _guard(request):
            return err
        q = request.rel_url.query
        return _json(kh.http_get_home_channels(q.get("task_id"), q.get("board")))

    async def handle_board_create(request: web.Request) -> web.Response:
        if err := await _guard(request):
            return err
        try:
            payload = await _read_json(request)
            return _json(kh.http_create_board_endpoint(kh.CreateBoardRequest(**payload)))
        except Exception as exc:
            return _error(exc)

    async def handle_board_switch(request: web.Request) -> web.Response:
        if err := await _guard(request):
            return err
        try:
            return _json(kh.http_switch_board(request.match_info["slug"]))
        except Exception as exc:
            return _error(exc)

    async def handle_orchestration_put(request: web.Request) -> web.Response:
        if err := await _guard(request):
            return err
        try:
            payload = await _read_json(request)
            return _json(kh.http_set_orchestration_settings(kh.OrchestrationSettingsRequest(**payload)))
        except Exception as exc:
            return _error(exc)

    app.router.add_get("/v1/kanban/tasks/{task_id}/log", handle_task_log)
    app.router.add_get("/v1/kanban/runs/{run_id}", handle_run_get)
    app.router.add_get("/v1/kanban/runs/{run_id}/inspect", handle_run_inspect)
    app.router.add_post("/v1/kanban/tasks/{task_id}/reclaim", handle_reclaim)
    app.router.add_get("/v1/kanban/home-channels", handle_home_channels)
    app.router.add_post("/v1/kanban/boards", handle_board_create)
    app.router.add_post("/v1/kanban/boards/{slug}/switch", handle_board_switch)
    app.router.add_put("/v1/kanban/orchestration", handle_orchestration_put)

    logger.info("Registered /v1/kanban/* gateway routes")
