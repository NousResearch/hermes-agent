"""
FastAPI application for Hermes Agent REST/WebSocket API.

Provides:
  - POST /v1/chat          — synchronous chat (send message, get full response)
  - WS   /v1/chat/stream   — streaming chat via WebSocket
  - POST /v1/chat/interrupt — interrupt a running agent
  - GET  /v1/sessions       — list active sessions
  - GET  /v1/sessions/{id}  — get session transcript
  - GET  /v1/media/{token}/{filename} — download media files (audio, images, etc.)
  - GET  /v1/health         — health check
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import uuid4

import tempfile

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from gateway.platforms.base import MessageType
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from gateway.platforms.api import APIPlatformAdapter

logger = logging.getLogger(__name__)

RESPONSE_TIMEOUT = int(os.getenv("API_RESPONSE_TIMEOUT", "300"))


# ── Simple in-memory rate limiter (no external dependency) ────────────

class _RateLimiter:
    """Token bucket rate limiter keyed by endpoint name."""

    def __init__(self):
        self._buckets: Dict[str, list] = {}  # key -> [tokens, last_refill_time]
        self._limits = {
            "chat": (10, 60),       # 10 requests per 60 seconds
            "voice": (10, 60),
            "upload": (20, 60),
        }

    def check(self, endpoint: str, client_id: str = "") -> bool:
        """Return True if request is allowed, False if rate limited."""
        import time as _time
        if endpoint not in self._limits:
            return True
        max_tokens, window = self._limits[endpoint]
        now = _time.monotonic()
        # Per-client bucket when client_id provided, else global
        bucket_key = f"{endpoint}:{client_id}" if client_id else endpoint

        if bucket_key not in self._buckets:
            self._buckets[bucket_key] = [max_tokens - 1, now]
            return True

        bucket = self._buckets[bucket_key]
        elapsed = now - bucket[1]
        # Refill tokens based on elapsed time
        bucket[0] = min(max_tokens, bucket[0] + elapsed * (max_tokens / window))
        bucket[1] = now

        if bucket[0] >= 1:
            bucket[0] -= 1
            return True
        return False


_rate_limiter = _RateLimiter()


# Media file token signing — prevents path traversal by signing the full path
_MEDIA_SECRET = secrets.token_hex(32)  # Separate from API_KEY, regenerated per process


import re as _re

def _safe_suffix(filename: str, default: str = "") -> str:
    """Extract and sanitize file extension from user-supplied filename."""
    if not filename:
        return default
    suffix = Path(filename).suffix.lower()
    # Only allow alphanumeric extensions
    if _re.match(r'^\.[a-z0-9]{1,10}$', suffix):
        return suffix
    return default


def _sign_media_path(file_path: str) -> str:
    """Create an HMAC token for a media file path."""
    return hmac.new(_MEDIA_SECRET.encode(), file_path.encode(), hashlib.sha256).hexdigest()


def _make_media_url(file_path: str, host: str = "") -> str:
    """Build a /v1/media/{token}/{filename} URL for a local file."""
    token = _sign_media_path(file_path)
    filename = Path(file_path).name
    base = host.rstrip("/") if host else ""
    return f"{base}/v1/media/{token}/{filename}"


# ── Request / Response models ────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=100_000)
    session_id: Optional[str] = Field(None, max_length=64, pattern=r'^[a-zA-Z0-9_\-]+$')


class ChatResponse(BaseModel):
    response: str
    session_id: str
    media: List[Dict[str, Any]] = []


# ── App factory ──────────────────────────────────────────────────────────


def create_app(adapter: APIPlatformAdapter) -> FastAPI:
    """Build and return the FastAPI application wired to *adapter*."""
    app = FastAPI(title="Hermes Agent API", version="1.0.0", docs_url=None, redoc_url=None, openapi_url=None)

    # CORS — allow same-origin by default, configurable via env
    cors_origins = os.getenv("API_CORS_ORIGINS", "").strip()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins.split(",") if cors_origins else [],
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    # ── Web UI ────────────────────────────────────────────────────────

    _static_dir = Path(__file__).parent / "static"

    @app.get("/", response_class=HTMLResponse)
    async def web_ui():
        index = _static_dir / "index.html"
        if index.is_file():
            return HTMLResponse(index.read_text())
        return HTMLResponse("<h1>Hermes Agent API</h1><p>Web UI not found.</p>")

    @app.get("/manifest.json")
    async def manifest():
        f = _static_dir / "manifest.json"
        if f.is_file():
            return FileResponse(f, media_type="application/manifest+json")
        raise HTTPException(404)

    @app.get("/icons/{filename}")
    async def icons(filename: str):
        f = _static_dir / "icons" / filename
        if f.is_file() and f.suffix in (".png", ".svg", ".ico"):
            return FileResponse(f)
        raise HTTPException(404)

    @app.get("/apple-touch-icon.png")
    @app.get("/apple-touch-icon-precomposed.png")
    async def apple_touch_icon():
        f = _static_dir / "icons" / "apple-touch-icon.png"
        if f.is_file():
            return FileResponse(f)
        raise HTTPException(404)

    @app.get("/favicon.ico")
    async def favicon_ico():
        f = _static_dir / "icons" / "favicon.png"
        if f.is_file():
            return FileResponse(f, media_type="image/png")
        raise HTTPException(404)

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

    # ── Shared response collector ───────────────────────────────────

    async def _collect_response(sid: str, message: str, message_type: MessageType = MessageType.TEXT) -> ChatResponse:
        """Register queue, send message, collect response, unregister."""
        try:
            queue = adapter.register_queue(sid)
        except ValueError:
            raise HTTPException(409, "Session is busy. Wait for the previous request to complete.")
        try:
            await adapter.handle_request(sid, message, message_type=message_type)
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
            logger.warning("API chat timeout for session %s", sid)
        finally:
            adapter.unregister_queue(sid)
        return ChatResponse(response="\n".join(text_parts), session_id=sid, media=media)

    # ── Synchronous chat ─────────────────────────────────────────────

    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest, _: None = Depends(verify_api_key)) -> ChatResponse:
        if not _rate_limiter.check("chat"):
            raise HTTPException(429, "Rate limited. Please slow down.")
        sid = req.session_id or str(uuid4())
        return await _collect_response(sid, req.message)

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
        raw_sid = auth_msg.get("session_id", "")
        if not raw_sid or len(raw_sid) > 64 or not _re.match(r'^[a-zA-Z0-9_\-]+$', raw_sid):
            session_id = str(uuid4())
        else:
            session_id = raw_sid

        try:
            while True:
                data = await ws.receive_json()
                message = data.get("message", "")
                if not message:
                    await ws.send_json({"type": "error", "content": "Empty message"})
                    continue
                if len(message) > 100_000:
                    await ws.send_json({"type": "error", "content": "Message too long (max 100K chars)"})
                    continue
                if not _rate_limiter.check("chat"):
                    await ws.send_json({"type": "error", "content": "Rate limited. Please slow down."})
                    continue

                try:
                    queue = adapter.register_queue(session_id)
                except ValueError:
                    await ws.send_json({"type": "error", "content": "Session is busy. Wait for the previous response."})
                    continue
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

    # ── File upload ────────────────────────────────────────────────────

    _UPLOAD_DIR = Path(
        os.getenv("HERMES_HOME", Path.home() / ".hermes")
    ) / "api_uploads"
    _MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB
    _CHUNK_SIZE = 64 * 1024  # 64 KB read chunks

    async def _read_upload_safe(self_file: UploadFile, max_bytes: int) -> bytes:
        """Read upload in chunks, rejecting if over max_bytes."""
        chunks = []
        total = 0
        while True:
            chunk = await self_file.read(_CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(413, f"File too large (max {max_bytes // 1024 // 1024} MB)")
            chunks.append(chunk)
        return b"".join(chunks)

    @app.post("/v1/upload")
    async def upload_file(
        file: UploadFile = File(...),
        _: None = Depends(verify_api_key),
    ) -> Dict[str, Any]:
        """Upload a file and return a download URL.

        Accepted for any file type. Max 25 MB.
        Use the returned ``url`` in chat messages or for voice input.
        """
        if not _rate_limiter.check("upload"):
            raise HTTPException(429, "Rate limited. Please slow down.")
        data = await _read_upload_safe(file, _MAX_UPLOAD_BYTES)

        _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        suffix = _safe_suffix(file.filename)
        dest = _UPLOAD_DIR / f"upload_{uuid4().hex[:12]}{suffix}"
        dest.write_bytes(data)

        url = adapter._register_media(str(dest))
        # Sanitize filename for safe display (strip path components)
        safe_name = Path(file.filename).name if file.filename else "upload"
        return {
            "url": url,
            "filename": safe_name,
            "size": len(data),
            "content_type": file.content_type,
        }

    # ── Voice message (upload + transcribe + chat) ────────────────────

    @app.post("/v1/chat/voice")
    async def chat_voice(
        file: UploadFile = File(...),
        session_id: Optional[str] = Form(None),
        _: None = Depends(verify_api_key),
    ) -> ChatResponse:
        """Upload a voice recording, transcribe it, and send to the agent.

        Works like POST /v1/chat but accepts an audio file instead of text.
        The audio is transcribed via the configured STT provider, then the
        transcript is sent to the agent as a voice message.
        """
        if not _rate_limiter.check("voice"):
            raise HTTPException(429, "Rate limited. Please slow down.")
        data = await _read_upload_safe(file, _MAX_UPLOAD_BYTES)

        # Save to temp file for transcription
        suffix = _safe_suffix(file.filename, ".webm")
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, prefix="api_voice_", delete=False)
        tmp.write(data)
        tmp.close()

        try:
            from tools.transcription_tools import transcribe_audio, get_stt_model_from_config
            stt_model = get_stt_model_from_config()
            result = await asyncio.to_thread(transcribe_audio, tmp.name, model=stt_model)

            if not result.get("success"):
                logger.warning("Voice transcription failed: %s", result.get("error", "unknown"))
                raise HTTPException(422, "Transcription failed. Check server logs for details.")

            transcript = result.get("transcript", "").strip()
            if not transcript:
                raise HTTPException(422, "Could not transcribe audio (empty result)")

            logger.info("API voice input: %s (lang=%s, prob=%.2f)",
                        transcript[:80],
                        result.get("language", "?"),
                        result.get("language_probability", 0.0))
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

        # Send transcript to agent as voice message
        sid = session_id or str(uuid4())
        return await _collect_response(sid, transcript, message_type=MessageType.VOICE)

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

    # ── Media download ────────────────────────────────────────────────

    @app.get("/v1/media/{token}/{filename}")
    async def download_media(token: str, filename: str):
        """Serve a media file if the token is valid.

        The token is an HMAC of the full file path, preventing path traversal.
        No API key header needed — the token itself authenticates the request.
        """
        # Look up the file in the adapter's media registry
        file_path = adapter._media_files.get(f"{token}/{filename}")
        if not file_path:
            raise HTTPException(404, "Media not found or expired")

        path = Path(file_path)
        if not path.is_file():
            raise HTTPException(410, "Media file no longer available")

        # Verify token matches the registered path
        expected_token = _sign_media_path(file_path)
        if not secrets.compare_digest(token, expected_token):
            raise HTTPException(403, "Invalid media token")

        return FileResponse(
            path=file_path,
            filename=filename,
            media_type=_guess_media_type(filename),
        )

    return app


def _guess_media_type(filename: str) -> str:
    """Return a MIME type based on file extension."""
    ext = Path(filename).suffix.lower()
    return {
        ".ogg": "audio/ogg",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".opus": "audio/opus",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".pdf": "application/pdf",
    }.get(ext, "application/octet-stream")
