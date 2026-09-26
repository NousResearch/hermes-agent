"""MoonPie native-client API routes.

REST endpoints for device registration, conversation sync, job queries,
approval responses, and a WebSocket gateway for real-time push.

All routes are prefixed with ``/api/moonpie`` by the mounting code in
``web_server.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from hermes_cli.web_routers._common import http_failure, require, _config_profile_scope

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter(prefix="/api/moonpie")

# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------

_agent_instance: Optional[Any] = None
_agent_lock = asyncio.Lock()
_device_histories: Dict[str, List[Dict[str, Any]]] = {}


async def _get_agent() -> Optional[Any]:
    """Lazily initialize a shared AIAgent for MoonPie clients.

    Runs inside ``_config_profile_scope`` so the agent resolves credentials,
    model, and memory from the active Hermes profile.
    """
    global _agent_instance
    if _agent_instance is not None:
        return _agent_instance
    async with _agent_lock:
        if _agent_instance is not None:
            return _agent_instance
        try:
            from run_agent import AIAgent

            def _init():
                with _config_profile_scope(None):
                    return AIAgent(
                        platform="moonpie",
                        quiet_mode=True,
                    )

            _agent_instance = await asyncio.to_thread(_init)
            _log.info("MoonPie agent initialized")
        except Exception as exc:
            _log.warning("MoonPie agent init failed: %s", exc, exc_info=True)
            _agent_instance = None
    return _agent_instance

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class DeviceRegisterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    model: str = Field(default="", max_length=128)
    os_version: str = Field(default="", max_length=64)
    public_key: str = Field(..., min_length=1)


class DeviceRegisterResponse(BaseModel):
    device_id: str
    pairing_code: str
    expires_in: int = 300


class DeviceVerifyRequest(BaseModel):
    device_id: str
    pairing_code: str


class DeviceVerifyResponse(BaseModel):
    device_token: str
    token_type: str = "Bearer"
    expires_in: int = 7776000  # 90 days


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


class ConversationDetail(BaseModel):
    id: str
    title: str
    messages: List[Dict[str, Any]] = []
    created_at: str
    updated_at: str


class JobSummary(BaseModel):
    id: str
    title: str
    status: str
    progress: Optional[float] = None
    message: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


class JobDetail(JobSummary):
    diff: Optional[str] = None
    result_url: Optional[str] = None
    error: Optional[str] = None


class ApprovalSummary(BaseModel):
    id: str
    job_id: str
    title: str
    description: str
    actions: List[str]
    timeout_at: str


class ApprovalRespondRequest(BaseModel):
    action: str  # "approve", "reject", "view_diff"


# ---------------------------------------------------------------------------
# In-memory stores (replace with SessionDB / persistent storage)
# ---------------------------------------------------------------------------

_pending_pairings: Dict[str, Dict[str, Any]] = {}
_registered_devices: Dict[str, Dict[str, Any]] = {}
_device_tokens: Dict[str, str] = {}  # token -> device_id

# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _bearer_token_from_header(authorization: str = "") -> str:
    if authorization.lower().startswith("bearer "):
        return authorization[7:]
    return ""


def _authenticate_device(authorization: str = "") -> str:
    token = _bearer_token_from_header(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    device_id = _device_tokens.get(token)
    if not device_id:
        raise HTTPException(status_code=401, detail="Invalid or expired device token")
    return device_id


# ---------------------------------------------------------------------------
# Device registration
# ---------------------------------------------------------------------------

@router.post("/devices/register", response_model=DeviceRegisterResponse)
async def device_register(req: DeviceRegisterRequest):
    """Register a new device and return a pairing code.

    The user must confirm the pairing code through an already-authenticated
    session (dashboard, CLI, or messaging) before the device receives a JWT.
    """
    device_id = f"moonpie-{uuid.uuid4().hex[:12]}"
    pairing_code = uuid.uuid4().hex[:6].upper()

    _pending_pairings[device_id] = {
        "device_id": device_id,
        "name": req.name,
        "model": req.model,
        "os_version": req.os_version,
        "public_key": req.public_key,
        "pairing_code": pairing_code,
        "created_at": time.time(),
        "confirmed": False,
    }

    _log.info("MoonPie device registered: %s (%s)", device_id, req.name)
    return DeviceRegisterResponse(
        device_id=device_id,
        pairing_code=pairing_code,
        expires_in=300,
    )


@router.post("/devices/verify", response_model=DeviceVerifyResponse)
async def device_verify(req: DeviceVerifyRequest):
    """Exchange a confirmed pairing code for a long-lived device JWT."""
    pending = _pending_pairings.get(req.device_id)
    if not pending:
        raise HTTPException(status_code=404, detail="Device not found or pairing expired")

    if pending["pairing_code"] != req.pairing_code:
        raise HTTPException(status_code=400, detail="Invalid pairing code")

    if not pending.get("confirmed"):
        raise HTTPException(status_code=403, detail="Pairing not yet confirmed by user")

    # Generate device JWT (placeholder — replace with real JWT signing)
    token = f"mpdt-{uuid.uuid4().hex}"
    _device_tokens[token] = req.device_id
    _registered_devices[req.device_id] = {
        **_pending_pairings[req.device_id],
        "token": token,
        "confirmed_at": time.time(),
    }
    del _pending_pairings[req.device_id]

    _log.info("MoonPie device verified: %s", req.device_id)
    return DeviceVerifyResponse(device_token=token)


@router.post("/devices/{device_id}/confirm")
async def device_confirm(device_id: str):
    """Confirm a pending device pairing (called by the dashboard / CLI)."""
    pending = _pending_pairings.get(device_id)
    if not pending:
        raise HTTPException(status_code=404, detail="Device not found or pairing expired")

    pending["confirmed"] = True
    _log.info("MoonPie device pairing confirmed: %s", device_id)
    return {"ok": True, "device_id": device_id}


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

@router.get("/conversations", response_model=List[ConversationSummary])
async def list_conversations(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    authorization: str = "",
):
    device_id = _authenticate_device(authorization)
    _log.debug("list_conversations for %s", device_id)
    # TODO: Query SessionDB for conversations belonging to this device/user
    return []


@router.post("/conversations", response_model=ConversationDetail)
async def create_conversation(authorization: str = ""):
    device_id = _authenticate_device(authorization)
    conv_id = f"conv-{uuid.uuid4().hex[:12]}"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _log.info("MoonPie conversation created: %s by %s", conv_id, device_id)
    return ConversationDetail(
        id=conv_id,
        title="New Conversation",
        messages=[],
        created_at=now,
        updated_at=now,
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: str, authorization: str = ""):
    device_id = _authenticate_device(authorization)
    _log.debug("get_conversation %s for %s", conversation_id, device_id)
    # TODO: Query SessionDB
    raise HTTPException(status_code=404, detail="Conversation not found")


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

@router.get("/jobs", response_model=List[JobSummary])
async def list_jobs(
    status: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    authorization: str = "",
):
    device_id = _authenticate_device(authorization)
    _log.debug("list_jobs for %s", device_id)
    # TODO: Query kanban / session state for jobs
    return []


@router.get("/jobs/{job_id}", response_model=JobDetail)
async def get_job(job_id: str, authorization: str = ""):
    device_id = _authenticate_device(authorization)
    _log.debug("get_job %s for %s", job_id, device_id)
    # TODO: Query job state
    raise HTTPException(status_code=404, detail="Job not found")


@router.get("/jobs/{job_id}/diff")
async def get_job_diff(job_id: str, authorization: str = ""):
    device_id = _authenticate_device(authorization)
    _log.debug("get_job_diff %s for %s", job_id, device_id)
    # TODO: Return job diff
    raise HTTPException(status_code=404, detail="Job diff not found")


# ---------------------------------------------------------------------------
# Approvals
# ---------------------------------------------------------------------------

@router.get("/approvals", response_model=List[ApprovalSummary])
async def list_approvals(authorization: str = ""):
    device_id = _authenticate_device(authorization)
    _log.debug("list_approvals for %s", device_id)
    # TODO: Query pending approvals from approval queue
    return []


@router.post("/approvals/{approval_id}/respond")
async def respond_approval(approval_id: str, req: ApprovalRespondRequest, authorization: str = ""):
    device_id = _authenticate_device(authorization)
    _log.info("MoonPie approval response: %s action=%s from %s", approval_id, req.action, device_id)
    # TODO: Route to the running agent / kanban system
    return {"ok": True, "approval_id": approval_id, "action": req.action}


# ---------------------------------------------------------------------------
# WebSocket gateway for native clients
# ---------------------------------------------------------------------------

class _MoonPieConnection:
    """One WebSocket connection from a MoonPie client."""

    def __init__(self, websocket: WebSocket, device_id: str):
        self.websocket = websocket
        self.device_id = device_id
        self.connected_at = time.time()

    async def send_json(self, data: dict):
        try:
            await self.websocket.send_text(json.dumps(data))
        except Exception:
            _log.warning("send_json failed to %s", self.device_id, exc_info=True)


_moonpie_connections: Dict[str, _MoonPieConnection] = {}


@router.websocket("/ws")
async def moonpie_websocket(websocket: WebSocket):
    """Bidirectional WebSocket for MoonPie native clients.

    Auth is via a ``device_token`` query parameter or the first JSON-RPC
    ``auth.login`` message.
    """
    await websocket.accept()

    device_id: Optional[str] = None
    token = websocket.query_params.get("device_token", "")

    if token:
        device_id = _device_tokens.get(token)

    # If no query token, wait for auth.login message
    if not device_id:
        try:
            msg = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
            data = json.loads(msg)
            if data.get("method") == "auth.login":
                token = data.get("params", {}).get("device_token", "")
                device_id = _device_tokens.get(token)
        except asyncio.TimeoutError:
            await websocket.close(code=4001, reason="Authentication timeout")
            return
        except Exception:
            await websocket.close(code=4001, reason="Invalid authentication")
            return

    if not device_id:
        await websocket.close(code=4001, reason="Invalid device token")
        return

    conn = _MoonPieConnection(websocket, device_id)
    _moonpie_connections[device_id] = conn
    _log.info("MoonPie WebSocket connected: %s", device_id)

    try:
        await _moonpie_loop(conn)
    except WebSocketDisconnect:
        _log.info("MoonPie WebSocket disconnected: %s", device_id)
    finally:
        _moonpie_connections.pop(device_id, None)


async def _moonpie_loop(conn: _MoonPieConnection):
    """Read JSON-RPC requests from the client and dispatch them."""
    while True:
        try:
            text = await conn.websocket.receive_text()
        except WebSocketDisconnect:
            break

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            await conn.send_json({"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}})
            continue

        method = data.get("method")
        req_id = data.get("id")
        params = data.get("params", {})

        if method == "conversation.message":
            payload = params.get("payload", {})
            content = payload.get("text", params.get("content", ""))
            conversation_id = params.get("conversation_id", "")
            agent = await _get_agent()

            if agent is None:
                await conn.send_json({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32000, "message": "Agent not available"},
                })
                continue

            loop = asyncio.get_running_loop()
            accumulated = []

            def stream_callback(delta: str):
                accumulated.append(delta)
                asyncio.run_coroutine_threadsafe(
                    conn.send_json({
                        "jsonrpc": "2.0",
                        "method": "conversation.delta",
                        "params": {
                            "content": delta,
                            "conversation_id": conversation_id,
                        },
                    }),
                    loop,
                )

            def _run_chat():
                with _config_profile_scope(None):
                    return agent.chat(content, stream_callback=stream_callback)

            try:
                final_response = await asyncio.to_thread(_run_chat)

                await conn.send_json({
                    "jsonrpc": "2.0",
                    "method": "conversation.complete",
                    "params": {"conversation_id": conversation_id},
                })

                await conn.send_json({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"status": "complete", "content": final_response},
                })

                # Stream TTS audio as binary frames after text completes
                await _send_tts_audio(conn, final_response)

            except Exception as exc:
                _log.error("MoonPie agent turn failed: %s", exc, exc_info=True)
                await conn.send_json({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32000, "message": str(exc)},
                })

        elif method == "conversation.start":
            conv_id = f"conv-{uuid.uuid4().hex[:12]}"
            await conn.send_json({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"conversation_id": conv_id},
            })

        elif method == "approval.respond":
            approval_id = params.get("approval_id", "")
            action = params.get("action", "")
            _log.info("MoonPie approval response: %s action=%s from %s", approval_id, action, conn.device_id)
            # TODO: Route to the active kanban approval queue
            await conn.send_json({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"status": "received", "approval_id": approval_id},
            })

        elif method == "device.capabilities":
            await conn.send_json({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "workspaces": [],
                    "bridge_available": False,
                },
            })

        elif method == "ping":
            await conn.send_json({"jsonrpc": "2.0", "id": req_id, "result": "pong"})

        else:
            await conn.send_json({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            })


async def _send_tts_audio(conn: _MoonPieConnection, text: str):
    """Synthesize speech for ``text`` and send it as a binary WebSocket frame."""
    if not text:
        return
    try:
        def _synthesize():
            with _config_profile_scope(None):
                from tools.tts_tool import text_to_speech_tool
                return text_to_speech_tool(text)

        result_json = await asyncio.to_thread(_synthesize)
        result = json.loads(result_json) if isinstance(result_json, str) else result_json
        if not result.get("success"):
            _log.warning("MoonPie TTS synthesis failed: %s", result.get("error"))
            return

        file_path = result.get("file_path")
        if not file_path or not os.path.isfile(file_path):
            _log.warning("MoonPie TTS audio file missing: %s", file_path)
            return

        def _read_and_unlink() -> bytes:
            try:
                with open(file_path, "rb") as fh:
                    return fh.read()
            finally:
                try:
                    os.unlink(file_path)
                except OSError:
                    pass

        audio_bytes = await asyncio.to_thread(_read_and_unlink)
        await conn.websocket.send_bytes(audio_bytes)
        _log.debug("MoonPie sent TTS audio: %d bytes to %s", len(audio_bytes), conn.device_id)
    except Exception:
        _log.warning("MoonPie TTS audio send failed", exc_info=True)


# ---------------------------------------------------------------------------
# Broadcast helpers (called by the gateway / agent loop)
# ---------------------------------------------------------------------------

async def broadcast_to_device(device_id: str, payload: dict):
    """Push a JSON-RPC notification to a connected MoonPie device."""
    conn = _moonpie_connections.get(device_id)
    if conn:
        await conn.send_json({"jsonrpc": "2.0", "method": payload["method"], "params": payload.get("params", {})})


async def broadcast_to_all(payload: dict):
    """Push a JSON-RPC notification to every connected MoonPie device."""
    for conn in list(_moonpie_connections.values()):
        await conn.send_json({"jsonrpc": "2.0", "method": payload["method"], "params": payload.get("params", {})})
