"""
AG-UI Protocol adapter for Hermes.

Exposes an AG-UI-compatible HTTP endpoint so any AG-UI frontend
(CopilotKit, custom React UIs, mobile apps) can connect to Hermes
without a custom per-frontend integration.

Endpoint:
  POST /ag-ui/runs          — start a run, stream AG-UI events via SSE
  GET  /ag-ui/health        — liveness check
  GET  /ag-ui/capabilities  — machine-readable capability descriptor

Wire format: Server-Sent Events (text/event-stream), one JSON object
per line, prefixed with "data: ", matching the AG-UI protocol spec.

Design:
  - Bridges AGUIRunAgentInput → Hermes session/run lifecycle
  - Translates Hermes internal events to AG-UI EventType values
  - Reuses the existing api_server asyncio.Queue SSE infrastructure
  - Zero changes to api_server.py or any core agent code
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncIterator, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.ag_ui_protocol import (
    AGUIRunAgentInput,
    AGUIRunStartedEvent,
    AGUIRunFinishedEvent,
    AGUIRunErrorEvent,
    AGUIStepStartedEvent,
    AGUIStepFinishedEvent,
    AGUITextMessageStartEvent,
    AGUITextMessageContentEvent,
    AGUITextMessageEndEvent,
    AGUIToolCallStartEvent,
    AGUIToolCallArgsEvent,
    AGUIToolCallEndEvent,
    AGUIToolCallResultEvent,
    AGUIStateSnapshotEvent,
    AGUICustomEvent,
)

_DEFAULT_PORT = 8643
_DEFAULT_HOST = "0.0.0.0"
_KEEPALIVE_INTERVAL = 15  # seconds


def check_ag_ui_requirements() -> bool:
    """Check runtime dependencies for the AG-UI adapter."""
    if not AIOHTTP_AVAILABLE:
        logger.warning("AG-UI: aiohttp is not installed — adapter unavailable")
        return False
    return True


class AGUIServerAdapter(BasePlatformAdapter):
    """
    AG-UI protocol server adapter.

    Runs an aiohttp HTTP server that accepts AG-UI RunAgentInput,
    drives a Hermes agent session, and streams AG-UI events back
    over Server-Sent Events.
    """

    def __init__(self, config: PlatformConfig) -> None:
        super().__init__(config, Platform.AG_UI)
        self._host: str = config.extra.get("host", _DEFAULT_HOST)
        self._port: int = int(config.extra.get("port", _DEFAULT_PORT))
        self._api_key: Optional[str] = config.token or None
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        # run_id -> asyncio.Queue of serialised SSE lines (str) or None sentinel
        self._run_queues: Dict[str, asyncio.Queue] = {}

    # ------------------------------------------------------------------
    # BasePlatformAdapter interface
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        if not check_ag_ui_requirements():
            return False
        try:
            self._app = web.Application()
            self._app.router.add_post("/ag-ui/runs", self._handle_run)
            self._app.router.add_get("/ag-ui/health", self._handle_health)
            self._app.router.add_get("/ag-ui/capabilities", self._handle_capabilities)
            self._runner = web.AppRunner(self._app)
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, self._host, self._port)
            await self._site.start()
            logger.info("AG-UI server listening on %s:%d", self._host, self._port)
            return True
        except Exception as exc:
            logger.error("AG-UI: failed to start server: %s", exc)
            return False

    async def disconnect(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
            self._site = None
        self._run_queues.clear()
        logger.info("AG-UI server stopped")

    async def send(self, chat_id: str, text: str, **kwargs: Any) -> SendResult:
        # AG-UI is a stateless run protocol — outbound messages are
        # streamed as SSE events during the run, not pushed post-hoc.
        return SendResult(success=False, error="AG-UI uses SSE streaming; use run endpoint")

    async def send_typing(self, chat_id: str) -> None:
        pass  # AG-UI has no typing indicator concept

    async def send_image(self, chat_id: str, image_url: str, caption: str = "") -> SendResult:
        return SendResult(success=False, error="AG-UI: use run SSE stream for media")

    async def get_chat_info(self, chat_id: str) -> dict:
        return {"name": f"ag-ui:{chat_id}", "type": "api", "chat_id": chat_id}

    # ------------------------------------------------------------------
    # Auth helper
    # ------------------------------------------------------------------

    def _check_auth(self, request: "web.Request") -> bool:
        if not self._api_key:
            return True
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:] == self._api_key
        return request.headers.get("X-API-Key", "") == self._api_key

    # ------------------------------------------------------------------
    # SSE helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sse_line(event_obj: Any) -> str:
        """Serialise an AG-UI event to a single SSE data line."""
        return "data: " + event_obj.model_dump_json(
            by_alias=True, exclude_none=True
        ) + "\n\n"

    @staticmethod
    def _sse_comment() -> str:
        return ": ping\n\n"

    # ------------------------------------------------------------------
    # Route handlers
    # ------------------------------------------------------------------

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        return web.json_response({"status": "ok", "platform": "ag-ui"})

    async def _handle_capabilities(self, request: "web.Request") -> "web.Response":
        return web.json_response({
            "streaming": True,
            "tools": True,
            "approval": True,
            "state_snapshot": True,
            "protocol": "ag-ui",
            "hermes_version": "1.0",
        })

    async def _handle_run(self, request: "web.Request") -> "web.StreamResponse":
        if not self._check_auth(request):
            return web.Response(status=401, text="Unauthorized")

        try:
            body = await request.json()
            run_input = AGUIRunAgentInput.model_validate(body)
        except Exception as exc:
            return web.Response(status=400, text=f"Invalid request: {exc}")

        run_id = run_input.run_id or f"agui_{uuid.uuid4().hex}"
        thread_id = run_input.thread_id or uuid.uuid4().hex

        queue: asyncio.Queue = asyncio.Queue()
        self._run_queues[run_id] = queue

        # Start agent in background
        asyncio.ensure_future(
            self._drive_agent(run_input, run_id, thread_id, queue)
        )

        # Stream SSE response
        response = web.StreamResponse(headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        })
        await response.prepare(request)

        try:
            async for line in self._consume_queue(queue):
                await response.write(line.encode())
        finally:
            self._run_queues.pop(run_id, None)

        return response

    # ------------------------------------------------------------------
    # Queue consumer with keepalive
    # ------------------------------------------------------------------

    async def _consume_queue(self, queue: asyncio.Queue) -> AsyncIterator[str]:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=_KEEPALIVE_INTERVAL)
            except asyncio.TimeoutError:
                yield self._sse_comment()
                continue
            if item is None:  # sentinel — run finished
                break
            yield item

    # ------------------------------------------------------------------
    # Agent driver — translates Hermes events → AG-UI SSE lines
    # ------------------------------------------------------------------

    async def _drive_agent(
        self,
        run_input: AGUIRunAgentInput,
        run_id: str,
        thread_id: str,
        queue: asyncio.Queue,
    ) -> None:
        """
        Drive a Hermes agent session for one AG-UI run.

        Emits AG-UI lifecycle events into *queue* as serialised SSE lines.
        A None sentinel is placed on the queue when the run ends.
        """

        def enqueue(event_obj: Any) -> None:
            queue.put_nowait(self._sse_line(event_obj))

        # RUN_STARTED
        enqueue(AGUIRunStartedEvent(thread_id=thread_id, run_id=run_id))
        enqueue(AGUIStepStartedEvent(step_name="agent"))

        try:
            # Build the user message from AG-UI input
            user_text = ""
            for msg in run_input.messages:
                if msg.role == "user" and msg.content:
                    user_text = msg.content
                    break

            if not user_text:
                enqueue(AGUIRunErrorEvent(
                    message="No user message found in run input",
                    code="NO_USER_MESSAGE",
                ))
                return

            # Construct a synthetic Hermes MessageEvent and dispatch
            from gateway.session import SessionSource
            from gateway.message_types import MessageEvent, MessageType

            source = SessionSource(
                platform=Platform.AG_UI,
                chat_id=thread_id,
                user_id=thread_id,
                username="ag-ui-user",
            )

            # Tracking state for multi-event message assembly
            message_id = uuid.uuid4().hex
            enqueue(AGUITextMessageStartEvent(message_id=message_id, role="assistant"))

            active_tool_calls: Dict[str, str] = {}  # tool_call_id -> message_id

            def on_delta(delta: str) -> None:
                enqueue(AGUITextMessageContentEvent(
                    message_id=message_id, delta=delta
                ))

            def on_tool_start(tool_call_id: str, function_name: str, args: str) -> None:
                active_tool_calls[tool_call_id] = message_id
                enqueue(AGUIToolCallStartEvent(
                    tool_call_id=tool_call_id,
                    tool_call_name=function_name,
                    parent_message_id=message_id,
                ))
                if args:
                    enqueue(AGUIToolCallArgsEvent(
                        tool_call_id=tool_call_id, delta=args
                    ))

            def on_tool_complete(
                tool_call_id: str,
                function_name: str,
                args: str,
                result: str,
            ) -> None:
                enqueue(AGUIToolCallEndEvent(tool_call_id=tool_call_id))
                result_msg_id = uuid.uuid4().hex
                enqueue(AGUIToolCallResultEvent(
                    message_id=result_msg_id,
                    tool_call_id=tool_call_id,
                    content=str(result)[:4096],
                ))
                active_tool_calls.pop(tool_call_id, None)

            def on_approval_needed(state: Dict[str, Any]) -> None:
                enqueue(AGUIStateSnapshotEvent(snapshot={
                    "status": "waiting_for_approval",
                    **state,
                }))

            # Dispatch to Hermes gateway
            event = MessageEvent(
                source=source,
                message_type=MessageType.TEXT,
                text=user_text,
                metadata={
                    "ag_ui_run_id": run_id,
                    "ag_ui_thread_id": thread_id,
                    "stream_delta_callback": on_delta,
                    "tool_start_callback": on_tool_start,
                    "tool_complete_callback": on_tool_complete,
                    "approval_callback": on_approval_needed,
                },
            )
            await self.handle_message(event)

            enqueue(AGUITextMessageEndEvent(message_id=message_id))
            enqueue(AGUIStepFinishedEvent(step_name="agent"))
            enqueue(AGUIRunFinishedEvent(thread_id=thread_id, run_id=run_id))

        except Exception as exc:
            logger.exception("AG-UI: agent run %s failed", run_id)
            enqueue(AGUIRunErrorEvent(
                message=str(exc),
                code="AGENT_ERROR",
            ))
        finally:
            queue.put_nowait(None)  # sentinel
