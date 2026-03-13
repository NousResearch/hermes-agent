"""
HTTP API platform adapter.

Runs a FastAPI HTTP server inside the gateway process, exposing an
OpenAI-compatible API for chat completions with SSE streaming support.
Designed for mobile app clients (e.g. DigiFrensiOS) and programmatic access.

Requires:
- fastapi, uvicorn, sse-starlette (pip install 'hermes-agent[http]')
- HTTP_AUTH_TOKEN env var (optional, enables Bearer auth)
- HTTP_PORT env var (default: 8720)
- HTTP_HOST env var (default: 0.0.0.0)

Usage:
    # Enable via environment
    export HTTP_AUTH_TOKEN=mysecret
    export HTTP_PORT=8720
    hermes gateway

    # Or via config.yaml
    gateway:
      http:
        enabled: true
        token: mysecret
        port: 8720
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Request, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    from sse_starlette.sse import EventSourceResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None  # type: ignore[assignment,misc]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource

logger = logging.getLogger(__name__)


def check_http_requirements() -> bool:
    """Check if HTTP adapter dependencies are available."""
    if not FASTAPI_AVAILABLE:
        logger.warning("HTTP adapter requires fastapi, uvicorn, sse-starlette. "
                       "Install with: pip install 'hermes-agent[http]'")
        return False
    return True


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatCompletionRequest(BaseModel):
        messages: list[ChatMessage]
        model: str = "hermes-agent"
        stream: bool = False
        temperature: float = 0.7
        max_tokens: int = 2048

    class NodeRegisterRequest(BaseModel):
        node_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        capabilities: list[str] = []
        device_info: dict = {}

    class NodeResponsePayload(BaseModel):
        result: Any = None
        error: Optional[str] = None


# ---------------------------------------------------------------------------
# HTTP Platform Adapter
# ---------------------------------------------------------------------------

class HTTPAdapter(BasePlatformAdapter):
    """
    HTTP API adapter.

    Runs a FastAPI server that accepts OpenAI-compatible chat completion
    requests and returns responses from the Hermes agent. Supports both
    synchronous and SSE streaming modes.

    Message flow:
    1. Client POSTs to /v1/chat/completions
    2. Adapter creates a MessageEvent and calls handle_message()
    3. GatewayRunner processes the message through the agent
    4. Framework calls adapter.send() with the response
    5. send() resolves the pending Future, returning the response to the client
    """

    MAX_MESSAGE_LENGTH = 100_000  # No practical limit for HTTP

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.HTTP)

        extra = config.extra or {}
        self._port = int(extra.get("port") or os.getenv("HTTP_PORT", "8720"))
        self._host = extra.get("host") or os.getenv("HTTP_HOST", "0.0.0.0")
        self._auth_token = config.token or os.getenv("HTTP_AUTH_TOKEN", "")

        # Pending responses: request_id -> asyncio.Future[str]
        self._pending_responses: Dict[str, asyncio.Future] = {}

        # Pending streaming queues: request_id -> asyncio.Queue
        self._streaming_queues: Dict[str, asyncio.Queue] = {}

        # Node capability state
        self._registered_nodes: Dict[str, Dict] = {}
        self._pending_node_requests: Dict[str, asyncio.Event] = {}
        self._node_responses: Dict[str, Dict] = {}

        # Server state
        self._app: Optional[Any] = None
        self._server: Optional[Any] = None
        self._serve_task: Optional[asyncio.Task] = None
        self._start_time = time.time()

        if FASTAPI_AVAILABLE:
            self._app = FastAPI(
                title="Hermes Agent HTTP Gateway",
                version="1.0.0",
                docs_url=None,   # Disable Swagger UI in production
                redoc_url=None,
            )
            self._app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["*"],
                allow_headers=["*"],
            )
            self._setup_routes()

    # ------------------------------------------------------------------
    # Auth dependency
    # ------------------------------------------------------------------

    def _get_auth_dependency(self):
        """Create a FastAPI dependency for Bearer token validation."""
        auth_token = self._auth_token

        async def verify_token(request: Request):
            if not auth_token:
                return  # No auth configured
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth[7:] != auth_token:
                raise HTTPException(status_code=401, detail="Invalid or missing auth token")

        return Depends(verify_token)

    # ------------------------------------------------------------------
    # Route setup
    # ------------------------------------------------------------------

    def _setup_routes(self):
        """Register all API routes on the FastAPI app."""
        app = self._app
        auth = self._get_auth_dependency()

        @app.get("/v1/status", dependencies=[auth])
        async def get_status():
            return self._build_status()

        @app.get("/v1/models", dependencies=[auth])
        async def list_models():
            return {
                "object": "list",
                "data": [{
                    "id": "hermes-agent",
                    "object": "model",
                    "created": int(self._start_time),
                    "owned_by": "self-hosted",
                }],
            }

        @app.get("/v1/skills", dependencies=[auth])
        async def list_skills():
            return self._get_skills()

        @app.get("/v1/tools", dependencies=[auth])
        async def list_tools():
            return self._get_tools()

        @app.post("/v1/chat/completions", dependencies=[auth])
        async def chat_completions(req: ChatCompletionRequest):
            if not self._message_handler:
                raise HTTPException(status_code=503, detail="Agent not ready")

            if req.stream:
                return EventSourceResponse(
                    self._handle_streaming_request(req),
                    media_type="text/event-stream",
                )
            else:
                return await self._handle_request(req)

        @app.post("/v1/node/register", dependencies=[auth])
        async def register_node(req: NodeRegisterRequest):
            self._registered_nodes[req.node_id] = {
                "capabilities": req.capabilities,
                "device_info": req.device_info,
                "registered_at": time.time(),
            }
            return {"status": "registered", "node_id": req.node_id}

        @app.post("/v1/node/respond/{request_id}", dependencies=[auth])
        async def node_respond(request_id: str, req: NodeResponsePayload):
            if request_id not in self._pending_node_requests:
                raise HTTPException(status_code=404, detail="No pending request")
            self._node_responses[request_id] = {
                "result": req.result,
                "error": req.error,
            }
            self._pending_node_requests[request_id].set()
            return {"status": "received"}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Start the HTTP server in a background task."""
        if not FASTAPI_AVAILABLE:
            logger.warning("[%s] fastapi/uvicorn not installed", self.name)
            return False

        try:
            config = uvicorn.Config(
                self._app,
                host=self._host,
                port=self._port,
                log_level="warning",
                access_log=False,
            )
            self._server = uvicorn.Server(config)
            self._serve_task = asyncio.create_task(self._server.serve())
            self._running = True
            logger.info("[%s] HTTP API listening on http://%s:%d",
                        self.name, self._host, self._port)
            return True
        except Exception as e:
            logger.error("[%s] Failed to start HTTP server: %s", self.name, e)
            return False

    async def disconnect(self) -> None:
        """Stop the HTTP server."""
        self._running = False
        if self._server:
            self._server.should_exit = True
        if self._serve_task:
            self._serve_task.cancel()
            try:
                await self._serve_task
            except asyncio.CancelledError:
                pass
        # Resolve any pending requests with an error
        for req_id, future in list(self._pending_responses.items()):
            if not future.done():
                future.set_exception(
                    Exception("Gateway shutting down")
                )
        self._pending_responses.clear()
        logger.info("[%s] HTTP server stopped", self.name)

    # ------------------------------------------------------------------
    # Required: send() — called by framework to deliver agent response
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """
        Deliver agent response back to the HTTP client.

        For non-streaming: resolves the pending Future so the HTTP handler
        can return the response.

        For streaming: pushes content chunks to the streaming queue.
        """
        request_id = chat_id  # We use request_id as chat_id

        # Streaming mode: push to queue
        if request_id in self._streaming_queues:
            queue = self._streaming_queues[request_id]
            # Split into word-level chunks for streaming effect
            words = content.split(" ")
            for i, word in enumerate(words):
                chunk = word if i == len(words) - 1 else word + " "
                await queue.put(("token", chunk))
            await queue.put(("done", None))
            return SendResult(success=True)

        # Non-streaming mode: resolve future
        if request_id in self._pending_responses:
            future = self._pending_responses[request_id]
            if not future.done():
                future.set_result(content)
            return SendResult(success=True)

        logger.warning("[%s] No pending request for chat_id=%s", self.name, request_id)
        return SendResult(success=False, error="No pending request for this ID")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Send typing indicator — for streaming, emit a heartbeat."""
        if chat_id in self._streaming_queues:
            # SSE comment line as keepalive
            await self._streaming_queues[chat_id].put(("comment", "typing"))

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return info about the HTTP chat session."""
        return {
            "name": f"HTTP Client {chat_id[:8]}",
            "type": "dm",
            "chat_id": chat_id,
        }

    # ------------------------------------------------------------------
    # Non-streaming request handler
    # ------------------------------------------------------------------

    async def _handle_request(self, req: ChatCompletionRequest) -> dict:
        """Handle a non-streaming chat completion request."""
        request_id = str(uuid.uuid4())
        user_message = req.messages[-1].content if req.messages else ""

        # Create a Future to wait for the agent's response
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending_responses[request_id] = future

        try:
            # Build MessageEvent and dispatch to gateway
            event = self._build_message_event(request_id, user_message, req.messages)
            await self.handle_message(event)

            # Wait for send() to resolve the future (with timeout)
            content = await asyncio.wait_for(future, timeout=120.0)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Agent response timed out")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Agent error: {e}")
        finally:
            self._pending_responses.pop(request_id, None)

        return {
            "id": f"chatcmpl-{request_id[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "hermes-agent",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    # ------------------------------------------------------------------
    # Streaming request handler
    # ------------------------------------------------------------------

    async def _handle_streaming_request(self, req: ChatCompletionRequest):
        """Handle a streaming chat completion request via SSE."""
        request_id = str(uuid.uuid4())
        user_message = req.messages[-1].content if req.messages else ""
        completion_id = f"chatcmpl-{request_id[:8]}"

        # Create a queue for streaming chunks
        queue: asyncio.Queue = asyncio.Queue()
        self._streaming_queues[request_id] = queue

        try:
            # Build MessageEvent and dispatch (runs in background)
            event = self._build_message_event(request_id, user_message, req.messages)
            asyncio.create_task(self._dispatch_and_wait(event, request_id))

            # Yield SSE events from the queue
            while True:
                try:
                    event_type, data = await asyncio.wait_for(
                        queue.get(), timeout=120.0
                    )
                except asyncio.TimeoutError:
                    break

                if event_type == "done":
                    break
                elif event_type == "comment":
                    yield {"comment": data}
                elif event_type == "tool_use":
                    yield {"event": "tool_use", "data": json.dumps(data)}
                elif event_type == "node_invoke":
                    yield {"event": "node_invoke", "data": json.dumps(data)}
                elif event_type == "token":
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "hermes-agent",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": data},
                            "finish_reason": None,
                        }],
                    }
                    yield {"data": json.dumps(chunk)}

        finally:
            self._streaming_queues.pop(request_id, None)

        # Send done signal
        yield {"data": "[DONE]"}

    async def _dispatch_and_wait(self, event: MessageEvent, request_id: str):
        """Dispatch message to gateway handler in background."""
        try:
            await self.handle_message(event)
        except Exception as e:
            logger.error("[%s] Error processing message %s: %s",
                         self.name, request_id, e)
            # Push error to streaming queue if still active
            if request_id in self._streaming_queues:
                queue = self._streaming_queues[request_id]
                await queue.put(("token", f"\n\n[Error: {e}]"))
                await queue.put(("done", None))

    # ------------------------------------------------------------------
    # Message event construction
    # ------------------------------------------------------------------

    def _build_message_event(
        self,
        request_id: str,
        user_message: str,
        messages: list,
    ) -> MessageEvent:
        """Build a MessageEvent from the HTTP request."""
        # Use request_id as chat_id so send() can route the response back
        source = self.build_source(
            chat_id=request_id,
            chat_name="HTTP API",
            chat_type="dm",
            user_id=request_id,
            user_name="HTTP Client",
        )

        return MessageEvent(
            text=user_message,
            message_type=MessageType.TEXT,
            source=source,
            message_id=request_id,
            timestamp=datetime.now(),
        )

    # ------------------------------------------------------------------
    # Status / metadata endpoints
    # ------------------------------------------------------------------

    def _build_status(self) -> dict:
        """Build status response for /v1/status."""
        return {
            "is_online": self._running and self._message_handler is not None,
            "agent_name": "Hermes Agent",
            "uptime": time.time() - self._start_time,
            "memory_count": 0,   # Populated by subclass or hook if available
            "skill_count": 0,
            "active_tool_count": 0,
        }

    def _get_skills(self) -> list:
        """Return agent skills for /v1/skills."""
        # Skills are stored as SKILL.md files in ~/.hermes/skills/
        skills = []
        from pathlib import Path
        skills_dir = Path(os.path.expanduser("~/.hermes/skills"))
        if skills_dir.exists():
            for skill_file in skills_dir.glob("*.md"):
                name = skill_file.stem
                # Read first line as description
                try:
                    first_line = skill_file.read_text().split("\n")[0].strip("# ")
                except Exception:
                    first_line = ""
                skills.append({
                    "id": name,
                    "name": name.replace("-", " ").replace("_", " ").title(),
                    "description": first_line,
                    "last_used": None,
                })
        return skills

    def _get_tools(self) -> list:
        """Return available tools for /v1/tools."""
        # Read from toolsets configuration
        tools = []
        try:
            from toolsets import get_toolset
            toolset = get_toolset("hermes-http")
            if toolset:
                for tool_name in toolset.get("tools", []):
                    tools.append({
                        "id": tool_name,
                        "name": tool_name,
                        "description": "",
                        "provider": "built-in",
                    })
        except ImportError:
            pass
        return tools

    # ------------------------------------------------------------------
    # Node capability support
    # ------------------------------------------------------------------

    async def invoke_node_capability(
        self,
        method: str,
        params: Optional[Dict] = None,
        timeout: float = 10.0,
    ) -> Dict:
        """
        Request data from a connected iOS device mid-conversation.

        Emits a node_invoke SSE event and waits for the client to POST
        the result back to /v1/node/respond/{request_id}.
        """
        if not self._registered_nodes:
            return {"error": "No nodes registered"}

        request_id = str(uuid.uuid4())
        event = asyncio.Event()
        self._pending_node_requests[request_id] = event

        # Find the active streaming queue to emit the invoke event
        for req_id, queue in self._streaming_queues.items():
            await queue.put(("node_invoke", {
                "requestId": request_id,
                "method": method,
                "params": params or {},
            }))
            break  # Send to the first active stream

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return self._node_responses.pop(request_id, {"error": "No response"})
        except asyncio.TimeoutError:
            return {"error": f"Node invoke timed out after {timeout}s"}
        finally:
            self._pending_node_requests.pop(request_id, None)
