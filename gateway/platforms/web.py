"""
Web platform adapter.

Provides a browser-based chat interface via HTTP + WebSocket.
Serves a single-page chat UI with markdown rendering, code highlighting,
voice messages, and mobile responsive design.

No external dependencies beyond aiohttp (already in messaging extra).
"""

import asyncio
import base64
import json
import logging
import os
import secrets
import shutil
import socket
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)


def check_web_requirements() -> bool:
    """Check if aiohttp is available."""
    return AIOHTTP_AVAILABLE


class WebAdapter(BasePlatformAdapter):
    """
    Web-based chat adapter.

    Runs a local HTTP server serving a chat UI. Clients connect via
    WebSocket for real-time bidirectional messaging.
    """

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WEB)
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

        # Config
        self._host: str = config.extra.get("host", "0.0.0.0")
        self._port: int = config.extra.get("port", 8765)
        self._token: str = config.extra.get("token", "") or secrets.token_hex(16)

        # Connected WebSocket clients: session_id -> ws
        self._clients: Dict[str, web.WebSocketResponse] = {}

        # Media directory for uploaded/generated files
        self._media_dir = Path.home() / ".hermes" / "web_media"

        # Cleanup task handle
        self._cleanup_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """Start the HTTP server and begin accepting connections."""
        if not AIOHTTP_AVAILABLE:
            return False

        self._media_dir.mkdir(parents=True, exist_ok=True)

        self._app = web.Application(client_max_size=50 * 1024 * 1024)  # 50MB upload limit
        self._app.router.add_get("/", self._handle_index)
        self._app.router.add_get("/ws", self._handle_websocket)
        self._app.router.add_post("/upload", self._handle_upload)
        self._app.router.add_static("/media", str(self._media_dir), show_index=False)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        try:
            self._site = web.TCPSite(self._runner, self._host, self._port)
            await self._site.start()
        except OSError as e:
            logger.error("Failed to start web server on %s:%s — %s", self._host, self._port, e)
            await self._runner.cleanup()
            return False

        self._running = True
        self._cleanup_task = asyncio.ensure_future(self._media_cleanup_loop())

        all_ips = self._get_local_ips()
        primary_ip = self._get_local_ip()
        print(f"[{self.name}] Web UI: http://{primary_ip}:{self._port}")
        for ip in all_ips:
            if ip != primary_ip:
                print(f"[{self.name}]   also: http://{ip}:{self._port}")
        print(f"[{self.name}] Access token: {self._token}")

        return True

    async def disconnect(self) -> None:
        """Stop the server and close all connections."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None

        for ws in list(self._clients.values()):
            try:
                await ws.close()
            except Exception:
                pass
        self._clients.clear()

        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()

        self._running = False
        self._app = None
        self._runner = None
        self._site = None
        print(f"[{self.name}] Disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text message to all connected clients."""
        msg_id = str(uuid.uuid4())[:8]
        payload = {
            "type": "message",
            "id": msg_id,
            "content": content,
            "timestamp": time.time(),
        }
        await self._broadcast(payload)
        return SendResult(success=True, message_id=msg_id)

    async def edit_message(
        self, chat_id: str, message_id: str, content: str
    ) -> SendResult:
        """Edit a previously sent message (used for streaming updates)."""
        payload = {
            "type": "edit",
            "id": message_id,
            "content": content,
            "timestamp": time.time(),
        }
        await self._broadcast(payload)
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Send typing indicator to all clients."""
        await self._broadcast({"type": "typing"})

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> SendResult:
        """Send an image to all connected clients."""
        msg_id = str(uuid.uuid4())[:8]
        payload = {
            "type": "image",
            "id": msg_id,
            "url": image_url,
            "caption": caption or "",
            "timestamp": time.time(),
        }
        await self._broadcast(payload)
        return SendResult(success=True, message_id=msg_id)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a voice message by copying audio to media dir and broadcasting URL."""
        filename = f"voice_{uuid.uuid4().hex[:8]}{Path(audio_path).suffix}"
        dest = self._media_dir / filename
        try:
            shutil.copy2(audio_path, dest)
        except Exception as e:
            return SendResult(success=False, error=f"Failed to copy audio: {e}")

        msg_id = str(uuid.uuid4())[:8]
        payload = {
            "type": "voice",
            "id": msg_id,
            "url": f"/media/{filename}",
            "caption": caption or "",
            "timestamp": time.time(),
        }
        await self._broadcast(payload)
        return SendResult(success=True, message_id=msg_id)

    async def play_tts(
        self,
        chat_id: str,
        audio_path: str,
        **kwargs,
    ) -> SendResult:
        """Play TTS audio invisibly — no bubble in chat, just audio playback."""
        filename = f"tts_{uuid.uuid4().hex[:8]}{Path(audio_path).suffix}"
        dest = self._media_dir / filename
        try:
            shutil.copy2(audio_path, dest)
        except Exception as e:
            return SendResult(success=False, error=f"Failed to copy audio: {e}")

        payload = {
            "type": "play_audio",
            "url": f"/media/{filename}",
        }
        await self._broadcast(payload)
        return SendResult(success=True)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> SendResult:
        """Send a local image file by copying to media dir."""
        filename = f"img_{uuid.uuid4().hex[:8]}{Path(image_path).suffix}"
        dest = self._media_dir / filename
        try:
            shutil.copy2(image_path, dest)
        except Exception as e:
            return SendResult(success=False, error=f"Failed to copy image: {e}")
        return await self.send_image(chat_id, f"/media/{filename}", caption, reply_to)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a document file by copying to media dir."""
        orig_name = file_name or Path(file_path).name
        safe_name = f"{uuid.uuid4().hex[:8]}_{orig_name}"
        dest = self._media_dir / safe_name
        try:
            shutil.copy2(file_path, dest)
        except Exception as e:
            return SendResult(success=False, error=f"Failed to copy file: {e}")

        msg_id = str(uuid.uuid4())[:8]
        payload = {
            "type": "document",
            "id": msg_id,
            "url": f"/media/{safe_name}",
            "filename": orig_name,
            "caption": caption or "",
            "timestamp": time.time(),
        }
        await self._broadcast(payload)
        return SendResult(success=True, message_id=msg_id)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic chat info for the web session."""
        return {"name": "Web Chat", "type": "dm"}

    # ---- HTTP Handlers ----

    async def _handle_index(self, request: web.Request) -> web.Response:
        """Serve the chat UI HTML page."""
        html = _build_chat_html()
        return web.Response(text=html, content_type="text/html")

    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connections for real-time chat."""
        ws = web.WebSocketResponse(max_msg_size=50 * 1024 * 1024)
        await ws.prepare(request)

        session_id = uuid.uuid4().hex[:12]
        authenticated = False

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue

                    msg_type = data.get("type", "")

                    # Auth handshake
                    if msg_type == "auth":
                        if data.get("token") == self._token:
                            authenticated = True
                            self._clients[session_id] = ws
                            await ws.send_str(json.dumps({
                                "type": "auth_ok",
                                "session_id": session_id,
                            }))
                        else:
                            await ws.send_str(json.dumps({
                                "type": "auth_fail",
                                "error": "Invalid token",
                            }))
                        continue

                    if not authenticated:
                        await ws.send_str(json.dumps({"type": "auth_required"}))
                        continue

                    # Chat message
                    if msg_type == "message":
                        text = data.get("text", "").strip()
                        if text:
                            await self._process_user_message(session_id, text)

                    # Voice message (base64 audio)
                    elif msg_type == "voice":
                        await self._process_voice_message(session_id, data)

                elif msg.type in (web.WSMsgType.ERROR, web.WSMsgType.CLOSE):
                    break
        except Exception as e:
            logger.debug("WebSocket session %s error: %s", session_id, e)
        finally:
            self._clients.pop(session_id, None)

        return ws

    async def _handle_upload(self, request: web.Request) -> web.Response:
        """Handle file uploads (images, voice recordings)."""
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if token != self._token:
            return web.json_response({"error": "Unauthorized"}, status=401)

        reader = await request.multipart()
        field = await reader.next()
        if not field:
            return web.json_response({"error": "No file"}, status=400)

        orig_name = field.filename or "file"
        filename = f"upload_{uuid.uuid4().hex[:8]}_{orig_name}"
        dest = self._media_dir / filename

        with open(dest, "wb") as f:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                f.write(chunk)

        return web.json_response({"url": f"/media/{filename}", "filename": filename})

    # ---- Message Processing ----

    async def _process_user_message(self, session_id: str, text: str) -> None:
        """Build MessageEvent from user text and feed to handler."""
        msg_type = MessageType.COMMAND if text.startswith("/") else MessageType.TEXT

        source = self.build_source(
            chat_id="web",
            chat_name="Web Chat",
            chat_type="dm",
            user_id=session_id,
            user_name="Web User",
        )

        event = MessageEvent(
            text=text,
            message_type=msg_type,
            source=source,
            message_id=uuid.uuid4().hex[:8],
        )

        if self._message_handler:
            await self.handle_message(event)

    async def _process_voice_message(self, session_id: str, data: dict) -> None:
        """Decode base64 voice audio, transcribe via STT, and process as message."""
        import tempfile

        audio_b64 = data.get("audio", "")
        if not audio_b64:
            return

        audio_bytes = base64.b64decode(audio_b64)
        fmt = data.get("format", "webm")
        tmp_path = os.path.join(
            tempfile.gettempdir(),
            f"web_voice_{uuid.uuid4().hex[:8]}.{fmt}",
        )

        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)

        try:
            from tools.transcription_tools import transcribe_audio
            result = await asyncio.to_thread(transcribe_audio, tmp_path)

            if not result.get("success"):
                await self._send_to_session(session_id, {
                    "type": "error",
                    "error": f"Transcription failed: {result.get('error', 'Unknown')}",
                })
                return

            transcript = result.get("transcript", "").strip()
            if not transcript:
                return

            # Show transcript to user
            await self._send_to_session(session_id, {
                "type": "transcript",
                "text": transcript,
            })

            # Process as voice message
            source = self.build_source(
                chat_id="web",
                chat_name="Web Chat",
                chat_type="dm",
                user_id=session_id,
                user_name="Web User",
            )
            event = MessageEvent(
                text=transcript,
                message_type=MessageType.VOICE,
                source=source,
                message_id=uuid.uuid4().hex[:8],
                media_urls=[tmp_path],
                media_types=[f"audio/{fmt}"],
            )
            if self._message_handler:
                await self.handle_message(event)
        except Exception as e:
            logger.warning("Voice processing failed: %s", e, exc_info=True)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ---- Internal Utilities ----

    async def _broadcast(self, payload: dict) -> None:
        """Send JSON payload to all connected WebSocket clients."""
        data = json.dumps(payload)
        dead: List[str] = []
        for sid, ws in self._clients.items():
            try:
                await ws.send_str(data)
            except Exception:
                dead.append(sid)
        for sid in dead:
            self._clients.pop(sid, None)

    async def _send_to_session(self, session_id: str, payload: dict) -> None:
        """Send a message to a specific client session."""
        ws = self._clients.get(session_id)
        if ws:
            try:
                await ws.send_str(json.dumps(payload))
            except Exception:
                self._clients.pop(session_id, None)

    async def _media_cleanup_loop(self) -> None:
        """Periodically delete old media files (older than 24h)."""
        try:
            while self._running:
                await asyncio.sleep(3600)
                cutoff = time.time() - 86400
                removed = 0
                for f in self._media_dir.iterdir():
                    if f.is_file() and f.stat().st_mtime < cutoff:
                        try:
                            f.unlink()
                            removed += 1
                        except OSError:
                            pass
                if removed:
                    logger.debug("Web media cleanup: removed %d old file(s)", removed)
        except asyncio.CancelledError:
            pass

    @staticmethod
    def _get_local_ips() -> List[str]:
        """Get all non-loopback IPv4 addresses on this machine."""
        ips = []
        try:
            import netifaces
            for iface in netifaces.interfaces():
                addrs = netifaces.ifaddresses(iface).get(netifaces.AF_INET, [])
                for addr in addrs:
                    ip = addr.get("addr", "")
                    if ip and not ip.startswith("127."):
                        ips.append(ip)
        except ImportError:
            # Fallback: parse ifconfig output
            import subprocess
            try:
                out = subprocess.check_output(["ifconfig"], text=True, timeout=5)
                for line in out.splitlines():
                    line = line.strip()
                    if line.startswith("inet ") and "127.0.0.1" not in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            ips.append(parts[1])
            except Exception:
                pass
        if not ips:
            # Last resort: UDP trick (may return VPN IP)
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                ips.append(s.getsockname()[0])
                s.close()
            except Exception:
                ips.append("127.0.0.1")
        return ips

    @staticmethod
    def _get_local_ip() -> str:
        """Get the most likely LAN IP address."""
        ips = WebAdapter._get_local_ips()
        # Prefer 192.168.x.x or 10.x.x.x over VPN ranges like 172.16.x.x
        for ip in ips:
            if ip.startswith("192.168.") or ip.startswith("10."):
                return ip
        return ips[0] if ips else "127.0.0.1"


# ---------------------------------------------------------------------------
# Chat UI HTML
# ---------------------------------------------------------------------------

def _build_chat_html() -> str:
    """Build the complete single-page chat UI as an HTML string."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>Hermes</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
:root {
    --bg: #08090d;
    --bg-secondary: rgba(14,16,24,0.85);
    --bg-input: rgba(20,24,36,0.9);
    --glass: rgba(16,20,32,0.6);
    --glass-border: rgba(100,120,200,0.12);
    --border: rgba(80,100,160,0.15);
    --text: #e2e8f0;
    --text-muted: #64748b;
    --accent: #6c5ce7;
    --accent-glow: rgba(108,92,231,0.3);
    --accent-hover: #a29bfe;
    --user-bg: linear-gradient(135deg, #6c5ce7 0%, #4834d4 100%);
    --bot-bg: rgba(20,24,40,0.7);
    --error: #ff6b6b;
    --success: #51cf66;
    --radius: 16px;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100dvh;
    overflow: hidden;
    /* Subtle grid background */
    background-image:
        radial-gradient(ellipse at 50% 0%, rgba(108,92,231,0.08) 0%, transparent 60%),
        linear-gradient(rgba(30,35,60,0.3) 1px, transparent 1px),
        linear-gradient(90deg, rgba(30,35,60,0.3) 1px, transparent 1px);
    background-size: 100% 100%, 40px 40px, 40px 40px;
}

/* Auth Screen */
#auth-screen {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100dvh;
    gap: 20px;
}
#auth-screen h1 {
    font-size: 32px;
    font-weight: 700;
    background: linear-gradient(135deg, #e2e8f0, #a29bfe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
#auth-screen p { color: var(--text-muted); font-size: 14px; }
#token-input {
    background: var(--bg-input);
    border: 1px solid var(--glass-border);
    color: var(--text);
    padding: 14px 18px;
    border-radius: 12px;
    font-size: 16px;
    width: 320px;
    max-width: 80vw;
    text-align: center;
    outline: none;
    backdrop-filter: blur(12px);
    transition: border-color 0.2s, box-shadow 0.2s;
}
#token-input:focus {
    border-color: var(--accent);
    box-shadow: 0 0 20px var(--accent-glow);
}
#auth-btn {
    background: var(--accent);
    color: #fff;
    border: none;
    padding: 12px 36px;
    border-radius: 12px;
    font-size: 15px;
    cursor: pointer;
    font-weight: 600;
    letter-spacing: 0.3px;
    transition: all 0.2s;
}
#auth-btn:hover {
    background: var(--accent-hover);
    box-shadow: 0 4px 24px var(--accent-glow);
    transform: translateY(-1px);
}
#auth-error { color: var(--error); font-size: 13px; display: none; }

/* Chat container — centered on desktop */
#chat-screen {
    display: none;
    flex-direction: column;
    height: 100dvh;
    max-width: 760px;
    margin: 0 auto;
    position: relative;
}
@media (min-width: 800px) {
    #chat-screen {
        border-left: 1px solid var(--glass-border);
        border-right: 1px solid var(--glass-border);
        background: rgba(8,9,13,0.5);
        backdrop-filter: blur(8px);
    }
}

/* Status bar */
#status-bar {
    background: var(--bg-secondary);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--glass-border);
    padding: 12px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
}
#status-bar .title {
    font-weight: 700;
    font-size: 16px;
    letter-spacing: -0.3px;
}
#status-bar .conn-label {
    font-size: 12px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 500;
}
#status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--success);
    display: inline-block;
    margin-right: 8px;
    box-shadow: 0 0 8px rgba(81,207,102,0.5);
}
#status-dot.disconnected {
    background: var(--error);
    box-shadow: 0 0 8px rgba(255,107,107,0.5);
}

/* Messages */
#messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    scroll-behavior: smooth;
}
#messages::-webkit-scrollbar { width: 4px; }
#messages::-webkit-scrollbar-track { background: transparent; }
#messages::-webkit-scrollbar-thumb { background: rgba(100,120,200,0.2); border-radius: 4px; }

.msg {
    max-width: 80%;
    padding: 12px 16px;
    border-radius: var(--radius);
    font-size: 14px;
    line-height: 1.6;
    word-wrap: break-word;
    overflow-wrap: break-word;
    animation: msgIn 0.25s ease-out;
}
@keyframes msgIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
.msg.user {
    align-self: flex-end;
    background: var(--user-bg);
    border-bottom-right-radius: 4px;
    box-shadow: 0 2px 16px rgba(108,92,231,0.2);
}
.msg.bot {
    align-self: flex-start;
    background: var(--bot-bg);
    border: 1px solid var(--glass-border);
    border-bottom-left-radius: 4px;
    backdrop-filter: blur(8px);
}
.msg.bot pre {
    background: rgba(8,9,13,0.8);
    border: 1px solid var(--glass-border);
    border-radius: 8px;
    padding: 12px;
    overflow-x: auto;
    margin: 8px 0;
}
.msg.bot code {
    font-family: "JetBrains Mono", "Fira Code", "SF Mono", monospace;
    font-size: 13px;
}
.msg.bot p code {
    background: rgba(108,92,231,0.15);
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 13px;
}
.msg.bot img {
    max-width: 100%;
    border-radius: 10px;
    margin: 8px 0;
    cursor: pointer;
}

/* Voice message bubble */
.voice-bubble {
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 220px;
    padding: 4px 0;
}
.voice-play-btn {
    width: 36px; height: 36px;
    border-radius: 50%;
    background: var(--accent);
    border: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    transition: all 0.2s;
    box-shadow: 0 2px 12px var(--accent-glow);
}
.voice-play-btn:hover { background: var(--accent-hover); }
.voice-play-btn svg { fill: #fff; width: 16px; height: 16px; }
.voice-waveform {
    flex: 1;
    display: flex;
    align-items: center;
    gap: 2px;
    height: 28px;
    cursor: pointer;
    position: relative;
}
.voice-waveform .bar {
    width: 3px;
    border-radius: 2px;
    background: var(--accent);
    opacity: 0.3;
    transition: opacity 0.15s;
}
.voice-waveform .bar.played { opacity: 1.0; }
.voice-time {
    font-size: 12px;
    color: var(--text-muted);
    min-width: 38px;
    text-align: right;
    flex-shrink: 0;
}

.msg .timestamp {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 4px;
    text-align: right;
}
.msg.system {
    align-self: center;
    background: transparent;
    color: var(--text-muted);
    font-size: 13px;
    font-style: italic;
    padding: 4px 8px;
}
.msg.transcript {
    align-self: flex-end;
    background: transparent;
    border: 1px dashed var(--border);
    color: var(--text-muted);
    font-size: 13px;
    font-style: italic;
}

/* Typing indicator */
.typing-indicator {
    display: none;
    align-self: flex-start;
    padding: 10px 16px;
    background: var(--bot-bg);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    border-bottom-left-radius: 4px;
    backdrop-filter: blur(8px);
}
.typing-indicator span {
    display: inline-block;
    width: 7px; height: 7px;
    background: var(--accent);
    border-radius: 50%;
    margin: 0 2px;
    animation: typing 1.4s infinite;
}
.typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
@keyframes typing {
    0%, 60%, 100% { opacity: 0.3; transform: translateY(0); }
    30% { opacity: 1; transform: translateY(-5px); }
}

/* Input bar */
#input-bar {
    background: var(--bg-secondary);
    backdrop-filter: blur(16px);
    border-top: 1px solid var(--glass-border);
    padding: 12px 16px;
    display: flex;
    gap: 10px;
    align-items: flex-end;
    flex-shrink: 0;
}
#input {
    flex: 1;
    background: var(--bg-input);
    border: 1px solid var(--glass-border);
    color: var(--text);
    padding: 11px 16px;
    border-radius: 22px;
    font-size: 15px;
    font-family: inherit;
    resize: none;
    max-height: 120px;
    min-height: 44px;
    line-height: 1.4;
    outline: none;
    transition: border-color 0.2s, box-shadow 0.2s;
}
#input:focus {
    border-color: var(--accent);
    box-shadow: 0 0 16px var(--accent-glow);
}
#input::placeholder { color: var(--text-muted); }
.input-btn {
    width: 44px; height: 44px;
    border-radius: 50%;
    border: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    transition: all 0.2s;
}
#send-btn {
    background: var(--accent);
    color: #fff;
    box-shadow: 0 2px 12px var(--accent-glow);
}
#send-btn:hover {
    background: var(--accent-hover);
    box-shadow: 0 4px 20px var(--accent-glow);
    transform: translateY(-1px);
}
#send-btn:disabled { opacity: 0.3; cursor: default; box-shadow: none; transform: none; }
#voice-btn {
    background: var(--bg-input);
    border: 1px solid var(--glass-border);
    color: var(--text-muted);
}
#voice-btn:hover {
    border-color: var(--accent);
    color: var(--accent);
    box-shadow: 0 0 12px var(--accent-glow);
}
#voice-btn.recording {
    background: var(--error);
    border-color: var(--error);
    color: #fff;
    animation: pulse 1.5s infinite;
    box-shadow: 0 0 16px rgba(255,107,107,0.4);
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}
.input-btn svg { width: 20px; height: 20px; fill: currentColor; }

/* Download link */
.file-download {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 12px;
    background: rgba(108,92,231,0.1);
    border: 1px solid var(--glass-border);
    border-radius: 8px;
    color: var(--accent);
    text-decoration: none;
    font-size: 13px;
    margin: 4px 0;
}
.file-download:hover { background: rgba(88,166,255,0.2); }

/* Scrollbar */
#messages::-webkit-scrollbar { width: 6px; }
#messages::-webkit-scrollbar-track { background: transparent; }
#messages::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>

<!-- Auth Screen -->
<div id="auth-screen">
    <h1>Hermes</h1>
    <p>Enter access token to connect</p>
    <input type="password" id="token-input" placeholder="Access token" autocomplete="off">
    <button id="auth-btn" onclick="doAuth()">Connect</button>
    <div id="auth-error">Invalid token. Try again.</div>
</div>

<!-- Chat Screen -->
<div id="chat-screen">
    <header id="status-bar">
        <div><span id="status-dot"></span><span class="title">Hermes</span></div>
        <span id="status-text" style="font-size:12px;color:var(--text-muted)">Connected</span>
    </header>
    <div id="messages">
        <div class="typing-indicator" id="typing"><span></span><span></span><span></span></div>
    </div>
    <div id="input-bar">
        <button class="input-btn" id="voice-btn" onclick="toggleVoice()" title="Voice message">
            <svg viewBox="0 0 24 24"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm-1-9c0-.55.45-1 1-1s1 .45 1 1v6c0 .55-.45 1-1 1s-1-.45-1-1V5zm6 6c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/></svg>
        </button>
        <textarea id="input" rows="1" placeholder="Type a message..." onkeydown="handleKey(event)" oninput="autoGrow(this)"></textarea>
        <button class="input-btn" id="send-btn" onclick="sendMessage()">
            <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
        </button>
    </div>
</div>

<script>
// --- State ---
let ws = null;
let sessionId = null;
let authToken = '';
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let typingTimeout = null;
let autoScroll = true;

// --- Markdown setup ---
marked.setOptions({
    highlight: function(code, lang) {
        if (lang && hljs.getLanguage(lang)) {
            return hljs.highlight(code, {language: lang}).value;
        }
        return hljs.highlightAuto(code).value;
    },
    breaks: true,
    gfm: true,
});

// --- Auth ---
document.getElementById('token-input').addEventListener('keydown', function(e) {
    if (e.key === 'Enter') doAuth();
});

function doAuth() {
    authToken = document.getElementById('token-input').value.trim();
    if (!authToken) return;
    document.getElementById('auth-error').style.display = 'none';
    connectWS();
}

// --- WebSocket ---
function connectWS() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${location.host}/ws`);

    ws.onopen = () => {
        ws.send(JSON.stringify({type: 'auth', token: authToken}));
    };

    ws.onmessage = (e) => {
        const data = JSON.parse(e.data);
        handleServerMessage(data);
    };

    ws.onclose = () => {
        setStatus(false);
        // Auto-reconnect after 3s
        setTimeout(() => {
            if (authToken && sessionId) connectWS();
        }, 3000);
    };

    ws.onerror = () => {};
}

function handleServerMessage(data) {
    switch (data.type) {
        case 'auth_ok':
            sessionId = data.session_id;
            document.getElementById('auth-screen').style.display = 'none';
            document.getElementById('chat-screen').style.display = 'flex';
            document.getElementById('input').focus();
            setStatus(true);
            // Check voice support
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                document.getElementById('voice-btn').style.display = 'none';
            }
            break;

        case 'auth_fail':
            document.getElementById('auth-error').style.display = 'block';
            break;

        case 'auth_required':
            break;

        case 'message':
            hideTyping();
            addBotMessage(data.id, data.content, data.timestamp);
            break;

        case 'edit':
            hideTyping();
            editBotMessage(data.id, data.content);
            break;

        case 'typing':
            showTyping();
            break;

        case 'image':
            hideTyping();
            addImageMessage(data.id, data.url, data.caption, data.timestamp);
            break;

        case 'voice':
            hideTyping();
            addVoiceMessage(data.id, data.url, data.caption, data.timestamp);
            break;

        case 'document':
            hideTyping();
            addDocumentMessage(data.id, data.url, data.filename, data.caption, data.timestamp);
            break;

        case 'transcript':
            addTranscriptMessage(data.text);
            break;

        case 'play_audio':
            // Invisible TTS playback — no UI element, just play audio
            { const a = new Audio(data.url); a.play().catch(() => {}); }
            break;

        case 'error':
            addSystemMessage(data.error);
            break;
    }
}

// --- Send ---
function sendMessage() {
    const input = document.getElementById('input');
    const text = input.value.trim();
    if (!text || !ws) return;

    addUserMessage(text);
    ws.send(JSON.stringify({type: 'message', text: text}));
    input.value = '';
    input.style.height = 'auto';
}

function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

function autoGrow(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

// --- Voice Recording ---
async function toggleVoice() {
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({audio: true});
        audioChunks = [];
        mediaRecorder = new MediaRecorder(stream, {mimeType: 'audio/webm;codecs=opus'});
        mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
        mediaRecorder.onstop = async () => {
            stream.getTracks().forEach(t => t.stop());
            if (audioChunks.length === 0) return;
            const blob = new Blob(audioChunks, {type: 'audio/webm'});
            const reader = new FileReader();
            reader.onloadend = () => {
                const b64 = reader.result.split(',')[1];
                ws.send(JSON.stringify({type: 'voice', audio: b64, format: 'webm'}));
            };
            reader.readAsDataURL(blob);
        };
        mediaRecorder.start();
        isRecording = true;
        document.getElementById('voice-btn').classList.add('recording');
    } catch (err) {
        addSystemMessage('Microphone access denied.');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    isRecording = false;
    document.getElementById('voice-btn').classList.remove('recording');
}

// --- Messages UI ---
const messagesEl = document.getElementById('messages');
const typingEl = document.getElementById('typing');

function formatTime(ts) {
    if (!ts) return '';
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
}

function addUserMessage(text) {
    const div = document.createElement('div');
    div.className = 'msg user';
    div.textContent = text;
    messagesEl.insertBefore(div, typingEl);
    scrollToBottom();
}

function addBotMessage(id, content, ts) {
    const div = document.createElement('div');
    div.className = 'msg bot';
    div.id = 'msg-' + id;
    div.innerHTML = renderMarkdown(content);
    if (ts) {
        const time = document.createElement('div');
        time.className = 'timestamp';
        time.textContent = formatTime(ts);
        div.appendChild(time);
    }
    messagesEl.insertBefore(div, typingEl);
    highlightCode(div);
    scrollToBottom();
}

function editBotMessage(id, content) {
    const div = document.getElementById('msg-' + id);
    if (div) {
        // Preserve timestamp
        const ts = div.querySelector('.timestamp');
        div.innerHTML = renderMarkdown(content);
        if (ts) div.appendChild(ts);
        highlightCode(div);
        scrollToBottom();
    } else {
        addBotMessage(id, content, null);
    }
}

function addImageMessage(id, url, caption, ts) {
    const div = document.createElement('div');
    div.className = 'msg bot';
    div.id = 'msg-' + id;
    if (caption) div.innerHTML = renderMarkdown(caption);
    const img = document.createElement('img');
    img.src = url;
    img.alt = caption || 'Image';
    img.onclick = () => window.open(url, '_blank');
    div.appendChild(img);
    if (ts) {
        const time = document.createElement('div');
        time.className = 'timestamp';
        time.textContent = formatTime(ts);
        div.appendChild(time);
    }
    messagesEl.insertBefore(div, typingEl);
    scrollToBottom();
}

function addVoiceMessage(id, url, caption, ts) {
    const div = document.createElement('div');
    div.className = 'msg bot';
    div.id = 'msg-' + id;
    if (caption) {
        const p = document.createElement('p');
        p.textContent = caption;
        div.appendChild(p);
    }

    const audio = new Audio(url);
    audio.preload = 'metadata';

    // Build voice bubble
    const bubble = document.createElement('div');
    bubble.className = 'voice-bubble';

    // Play/pause button
    const btn = document.createElement('button');
    btn.className = 'voice-play-btn';
    const playSvg = '<svg viewBox="0 0 24 24"><polygon points="6,3 20,12 6,21"/></svg>';
    const pauseSvg = '<svg viewBox="0 0 24 24"><rect x="5" y="3" width="4" height="18"/><rect x="15" y="3" width="4" height="18"/></svg>';
    btn.innerHTML = playSvg;
    bubble.appendChild(btn);

    // Waveform bars
    const waveform = document.createElement('div');
    waveform.className = 'voice-waveform';
    const barCount = 35;
    const bars = [];
    for (let i = 0; i < barCount; i++) {
        const bar = document.createElement('div');
        bar.className = 'bar';
        const h = 6 + Math.random() * 22;
        bar.style.height = h + 'px';
        waveform.appendChild(bar);
        bars.push(bar);
    }
    bubble.appendChild(waveform);

    // Duration display
    const timeEl = document.createElement('div');
    timeEl.className = 'voice-time';
    timeEl.textContent = '0:00';
    bubble.appendChild(timeEl);

    function fmtDur(s) {
        const m = Math.floor(s / 60);
        const sec = Math.floor(s % 60);
        return m + ':' + (sec < 10 ? '0' : '') + sec;
    }

    audio.addEventListener('loadedmetadata', () => {
        if (isFinite(audio.duration)) timeEl.textContent = fmtDur(audio.duration);
    });

    let playing = false;
    function updateProgress() {
        if (!isFinite(audio.duration)) return;
        const pct = audio.currentTime / audio.duration;
        const playedIdx = Math.floor(pct * barCount);
        bars.forEach((b, i) => b.classList.toggle('played', i <= playedIdx));
        timeEl.textContent = fmtDur(audio.currentTime);
        if (playing) requestAnimationFrame(updateProgress);
    }

    btn.onclick = () => {
        if (playing) { audio.pause(); }
        else { audio.play(); }
    };
    audio.onplay = () => { playing = true; btn.innerHTML = pauseSvg; updateProgress(); };
    audio.onpause = () => { playing = false; btn.innerHTML = playSvg; };
    audio.onended = () => {
        playing = false;
        btn.innerHTML = playSvg;
        bars.forEach(b => b.classList.remove('played'));
        if (isFinite(audio.duration)) timeEl.textContent = fmtDur(audio.duration);
    };

    // Click on waveform to seek
    waveform.onclick = (e) => {
        if (!isFinite(audio.duration)) return;
        const rect = waveform.getBoundingClientRect();
        const pct = (e.clientX - rect.left) / rect.width;
        audio.currentTime = pct * audio.duration;
        if (!playing) audio.play();
    };

    div.appendChild(bubble);

    if (ts) {
        const time = document.createElement('div');
        time.className = 'timestamp';
        time.textContent = formatTime(ts);
        div.appendChild(time);
    }
    messagesEl.insertBefore(div, typingEl);
    scrollToBottom();

    // Autoplay
    audio.play().catch(() => {});
}

function addDocumentMessage(id, url, filename, caption, ts) {
    const div = document.createElement('div');
    div.className = 'msg bot';
    div.id = 'msg-' + id;
    if (caption) div.innerHTML = renderMarkdown(caption);
    const a = document.createElement('a');
    a.className = 'file-download';
    a.href = url;
    a.download = filename;
    a.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zm4 18H6V4h7v5h5v11z"/></svg>' + filename;
    div.appendChild(a);
    if (ts) {
        const time = document.createElement('div');
        time.className = 'timestamp';
        time.textContent = formatTime(ts);
        div.appendChild(time);
    }
    messagesEl.insertBefore(div, typingEl);
    scrollToBottom();
}

function addSystemMessage(text) {
    const div = document.createElement('div');
    div.className = 'msg system';
    div.textContent = text;
    messagesEl.insertBefore(div, typingEl);
    scrollToBottom();
}

function addTranscriptMessage(text) {
    // Show transcribed voice as a normal user message with mic icon
    const div = document.createElement('div');
    div.className = 'msg user';
    div.textContent = text;
    const ts = document.createElement('div');
    ts.className = 'timestamp';
    ts.textContent = formatTime(Date.now() / 1000);
    div.appendChild(ts);
    messagesEl.insertBefore(div, typingEl);
    scrollToBottom();
}

function renderMarkdown(text) {
    try {
        return marked.parse(text);
    } catch (e) {
        return text.replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }
}

function highlightCode(el) {
    el.querySelectorAll('pre code').forEach(block => {
        hljs.highlightElement(block);
    });
}

// --- Typing ---
function showTyping() {
    typingEl.style.display = 'block';
    scrollToBottom();
    clearTimeout(typingTimeout);
    typingTimeout = setTimeout(hideTyping, 10000);
}

function hideTyping() {
    typingEl.style.display = 'none';
    clearTimeout(typingTimeout);
}

// --- Scroll ---
messagesEl.addEventListener('scroll', () => {
    const atBottom = messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 60;
    autoScroll = atBottom;
});

function scrollToBottom() {
    if (autoScroll) {
        requestAnimationFrame(() => {
            messagesEl.scrollTop = messagesEl.scrollHeight;
        });
    }
}

// --- Status ---
function setStatus(connected) {
    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');
    if (connected) {
        dot.className = '';
        dot.id = 'status-dot';
        text.textContent = 'Connected';
    } else {
        dot.className = 'disconnected';
        dot.id = 'status-dot';
        text.textContent = 'Reconnecting...';
    }
}
</script>
</body>
</html>'''
