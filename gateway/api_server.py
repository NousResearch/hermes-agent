"""
FastAPI application for Hermes Agent REST/WebSocket API.

Provides:
  - POST /v1/chat          — synchronous chat (send message, get full response)
  - WS   /v1/chat/stream   — streaming chat via WebSocket
  - POST /v1/chat/interrupt — interrupt a running agent
  - GET  /v1/sessions       — list active sessions
  - GET  /v1/sessions/{id}  — get session transcript
  - GET  /v1/health         — health check
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from gateway.platforms.api import APIPlatformAdapter

logger = logging.getLogger(__name__)

RESPONSE_TIMEOUT = int(os.getenv("API_RESPONSE_TIMEOUT", "300"))


# ── Request / Response models ────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=100_000)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    media: List[Dict[str, Any]] = []


# ── App factory ──────────────────────────────────────────────────────────


def create_app(adapter: APIPlatformAdapter) -> FastAPI:
    """Build and return the FastAPI application wired to *adapter*."""
    app = FastAPI(title="Hermes Agent API", version="1.0.0")

    # ── Auth dependency ──────────────────────────────────────────────

    async def verify_api_key(authorization: str = Header(...)) -> None:
        expected = os.getenv("API_KEY", "").strip()
        if not expected:
            raise HTTPException(500, "API_KEY not configured on server")
        token = authorization.replace("Bearer ", "", 1).strip()
        if not secrets.compare_digest(token, expected):
            raise HTTPException(401, "Invalid API key")

    # ── Health ───────────────────────────────────────────────────────

    @app.get("/v1/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    # ── Synchronous chat ─────────────────────────────────────────────

    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest, _: None = Depends(verify_api_key)) -> ChatResponse:
        session_id = req.session_id or str(uuid4())
        queue = adapter.register_queue(session_id)

        try:
            await adapter.handle_request(session_id, req.message)

            text_parts: list[str] = []
            media: list[dict] = []

            while True:
                msg = await asyncio.wait_for(queue.get(), timeout=RESPONSE_TIMEOUT)
                if msg["type"] == "done":
                    break
                elif msg["type"] == "message":
                    text_parts.append(msg["content"])
                else:
                    media.append(msg)
        except asyncio.TimeoutError:
            logger.warning("API chat timeout for session %s", session_id)
        finally:
            adapter.unregister_queue(session_id)

        return ChatResponse(
            response="\n".join(text_parts),
            session_id=session_id,
            media=media,
        )

    # ── WebSocket streaming ──────────────────────────────────────────

    @app.websocket("/v1/chat/stream")
    async def chat_stream(ws: WebSocket) -> None:
        await ws.accept()

        # First-message auth (browsers cannot send custom headers on WS)
        try:
            auth_msg = await asyncio.wait_for(ws.receive_json(), timeout=10)
        except (asyncio.TimeoutError, WebSocketDisconnect):
            await ws.close(code=4001, reason="Auth timeout")
            return

        if auth_msg.get("type") != "auth":
            await ws.close(code=4001, reason="First message must be {\"type\": \"auth\", \"token\": \"...\"}")
            return

        expected = os.getenv("API_KEY", "").strip()
        if not expected or not secrets.compare_digest(auth_msg.get("token", ""), expected):
            await ws.close(code=4001, reason="Invalid API key")
            return

        await ws.send_json({"type": "auth_ok"})
        session_id = auth_msg.get("session_id") or str(uuid4())

        try:
            while True:
                data = await ws.receive_json()
                message = data.get("message", "")
                if not message:
                    await ws.send_json({"type": "error", "content": "Empty message"})
                    continue

                queue = adapter.register_queue(session_id)
                try:
                    await adapter.handle_request(session_id, message)

                    while True:
                        msg = await asyncio.wait_for(queue.get(), timeout=RESPONSE_TIMEOUT)
                        if msg["type"] == "done":
                            await ws.send_json({"type": "done", "session_id": session_id})
                            break
                        await ws.send_json(msg)
                except asyncio.TimeoutError:
                    await ws.send_json({"type": "error", "content": "Response timeout"})
                finally:
                    adapter.unregister_queue(session_id)
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected (session %s)", session_id)

    # ── Interrupt ────────────────────────────────────────────────────

    @app.post("/v1/chat/interrupt")
    async def interrupt(
        session_id: str, _: None = Depends(verify_api_key)
    ) -> Dict[str, Any]:
        session_key = adapter._build_session_key(session_id)
        if session_key in adapter._active_sessions:
            adapter._active_sessions[session_key].set()
            return {"interrupted": True, "session_id": session_id}
        return {
            "interrupted": False,
            "session_id": session_id,
            "reason": "no active session",
        }

    # ── Sessions ─────────────────────────────────────────────────────

    @app.get("/v1/sessions")
    async def list_sessions(_: None = Depends(verify_api_key)) -> Dict[str, Any]:
        active = list(adapter._active_sessions.keys())
        return {"sessions": active}

    @app.get("/v1/sessions/{session_id}")
    async def get_session(
        session_id: str, _: None = Depends(verify_api_key)
    ) -> Dict[str, Any]:
        # Transcript access requires GatewayRunner's session_store.
        # The adapter receives a reference when the gateway sets it up.
        session_key = adapter._build_session_key(session_id)
        if hasattr(adapter, "_session_store") and adapter._session_store:
            transcript = adapter._session_store.load_transcript(session_key)
            return {"session_id": session_id, "messages": transcript or []}
        return {"session_id": session_id, "messages": [], "note": "session store not available"}

    return app
