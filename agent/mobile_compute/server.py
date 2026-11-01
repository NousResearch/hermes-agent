"""Minimal HTTP server for Mobile Compute Node (iPhone 16e).

This is the Windows test/development server. The production iPhone version
will be implemented in Swift using Vapor/NIO or a lightweight HTTP framework.

Endpoints:
    GET  /health          -> {"status": "ok", "device": "iPhone 16e", "version": "..."}
    GET  /capabilities    -> {"compute_types": [...], "memory_mb": ..., "os_version": "..."}
    POST /compute         -> Submit compute task (enqueues, returns task_id)
    GET  /tasks           -> List all tasks
    GET  /tasks/{task_id} -> Get task status/result

No LLM, no Core ML. Pure transport + task queue scaffolding.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aiohttp import web

from agent.mobile_compute.config import MobileComputeServerConfig, TLSConfig

logger = logging.getLogger("agent.mobile_compute.server")


@dataclass
class ComputeTask:
    """A single compute task in the queue."""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    status: str = "queued"  # queued | running | completed | failed
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class MobileComputeServer:
    """aiohttp-based server for Mobile Compute (Windows dev / test only)."""

    def __init__(self, config: MobileComputeServerConfig) -> None:
        self._config = config
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._tasks: Dict[str, ComputeTask] = {}
        self._queue: asyncio.Queue[ComputeTask] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._started = False

    def _build_app(self) -> web.Application:
        """Construct the aiohttp application with routes."""
        app = web.Application()
        app.router.add_get("/health", self._handle_health)
        app.router.add_get("/capabilities", self._handle_capabilities)
        app.router.add_post("/compute", self._handle_compute)
        app.router.add_get("/tasks", self._handle_list_tasks)
        app.router.add_get("/tasks/{task_id}", self._handle_get_task)
        app.on_startup.append(self._on_startup)
        app.on_cleanup.append(self._on_cleanup)
        return app

    async def _on_startup(self, app: web.Application) -> None:
        """Start the background worker."""
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("Mobile Compute worker started")

    async def _on_cleanup(self, app: web.Application) -> None:
        """Stop the background worker."""
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Mobile Compute worker stopped")

    async def _worker_loop(self) -> None:
        """Background worker that processes queued tasks."""
        while True:
            try:
                task = await self._queue.get()
                await self._execute_task(task)
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception("Worker loop error: %s", exc)
                await asyncio.sleep(1)

    async def _execute_task(self, task: ComputeTask) -> None:
        """Execute a single task. This is where real computation would happen.
        
        For phase 0 (scaffold), this just echoes the payload back.
        """
        task.status = "running"
        task.started_at = time.time()
        logger.info("Executing task %s (%s)", task.task_id, task.task_type)

        try:
            # Phase 0: echo / no-op. Replace with actual computation later.
            if task.task_type == "echo":
                task.result = {"echoed": task.payload}
            elif task.task_type == "info":
                task.result = {
                    "device": "iPhone 16e (simulated)",
                    "capabilities": await self._get_capabilities(),
                    "queue_size": self._queue.qsize(),
                }
            else:
                task.result = {"received": task.payload, "note": "scaffold only"}

            task.status = "completed"
        except Exception as exc:
            task.status = "failed"
            task.error = str(exc)
            logger.exception("Task %s failed: %s", task.task_id, exc)
        finally:
            task.completed_at = time.time()

    async def _get_capabilities(self) -> Dict[str, Any]:
        return {
            "compute_types": ["echo", "info"],
            "memory_mb": 8192,
            "os_version": "iOS 18.0 (simulated)",
            "chip": "A18 (simulated)",
            "note": "Phase 0 scaffold — no real compute",
        }

    # --- HTTP Handlers ---

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({
            "status": "ok",
            "device": "iPhone 16e (simulated)",
            "version": "0.1.0-scaffold",
            "uptime_seconds": time.time() - self._start_time if hasattr(self, "_start_time") else 0,
        })

    async def _handle_capabilities(self, request: web.Request) -> web.Response:
        caps = await self._get_capabilities()
        return web.json_response(caps)

    async def _handle_compute(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        task_type = body.get("task_type")
        payload = body.get("payload", {})
        task_id = body.get("task_id") or str(uuid.uuid4())[:8]

        if not task_type:
            return web.json_response({"error": "task_type required"}, status=400)

        task = ComputeTask(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
        )
        self._tasks[task_id] = task
        await self._queue.put(task)

        return web.json_response({
            "success": True,
            "task_id": task_id,
            "status": "queued",
        })

    async def _handle_list_tasks(self, request: web.Request) -> web.Response:
        tasks = [
            {
                "task_id": t.task_id,
                "task_type": t.task_type,
                "status": t.status,
                "created_at": t.created_at,
                "started_at": t.started_at,
                "completed_at": t.completed_at,
                "error": t.error,
            }
            for t in self._tasks.values()
        ]
        return web.json_response({"tasks": tasks})

    async def _handle_get_task(self, request: web.Request) -> web.Response:
        task_id = request.match_info["task_id"]
        task = self._tasks.get(task_id)
        if not task:
            return web.json_response({"error": "Task not found"}, status=404)

        return web.json_response({
            "task_id": task.task_id,
            "task_type": task.task_type,
            "status": task.status,
            "payload": task.payload,
            "result": task.result,
            "error": task.error,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
        })

    # --- Lifecycle ---

    async def start(self) -> None:
        """Start the server."""
        if self._started:
            return

        self._app = self._build_app()
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        ssl_context = None
        if self._config.tls.enabled:
            import ssl
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(
                self._config.tls.cert_path,
                self._config.tls.key_path,
            )
            if self._config.tls.ca_path:
                ssl_context.load_verify_locations(self._config.tls.ca_path)
            if self._config.tls.require_client_cert:
                ssl_context.verify_mode = ssl.CERT_REQUIRED

        self._site = web.TCPSite(
            self._runner,
            self._config.host,
            self._config.port,
            ssl_context=ssl_context,
        )
        await self._site.start()
        self._started = True
        self._start_time = time.time()
        logger.info("Mobile Compute server listening on %s:%d",
                    self._config.host, self._config.port)

    async def stop(self) -> None:
        """Stop the server."""
        if not self._started:
            return

        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        self._started = False
        logger.info("Mobile Compute server stopped")

    async def __aenter__(self) -> "MobileComputeServer":
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()


async def run_server_forever(config: MobileComputeServerConfig) -> None:
    """Run server until interrupted. For CLI entry point."""
    import signal

    server = MobileComputeServer(config)
    await server.start()

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass  # Windows

    await stop_event.wait()
    await server.stop()


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    config = MobileComputeServerConfig(enabled=True)
    asyncio.run(run_server_forever(config))