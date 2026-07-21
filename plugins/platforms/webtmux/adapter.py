"""WebTmux platform adapter for Hermes Gateway.

Runs an aiohttp HTTP/WebSocket server that exposes tmux sessions over the web.
Each session is a 2x2 tmux pane grid:
  pane 0: hermes chat CLI
  pane 1: nvim
  pane 2: terminal
  pane 3: terminal

Web UI has session tabs. Messages typed into the chat box are routed to
the Hermes gateway as MessageEvents. Hermes replies are rendered into
the chat history in the web UI.

MVP features:
  - HTTP server for web UI (HTML/JS)
  - WebSocket for real-time pane snapshots + chat input/output
  - tmux session creation via subprocess (reliable 2x2 grid)
  - Keyboard input forwarded to tmux panes (literal + special keys)
  - Chat messages routed to Hermes gateway and replies shown in UI
  - Streaming edit_message support
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

try:
    from aiohttp import web, WSMsgType
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore
    WSMsgType = None

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8650
DEFAULT_FRAME_INTERVAL = 0.15  # seconds between pane capture broadcasts
PANES_PER_SESSION = 4

# Mapping from JS key names to tmux send-keys arguments.
# Keys listed here are sent WITHOUT the -l (literal) flag.
SPECIAL_KEY_MAP = {
    "Enter": "Enter",
    "Backspace": "BSpace",
    "Tab": "Tab",
    "Escape": "Escape",
    "ArrowUp": "Up",
    "ArrowDown": "Down",
    "ArrowLeft": "Left",
    "ArrowRight": "Right",
    "Home": "Home",
    "End": "End",
    "PageUp": "PageUp",
    "PageDown": "PageDown",
    "Delete": "DC",
    "Insert": "IC",
    "F1": "F1",
    "F2": "F2",
    "F3": "F3",
    "F4": "F4",
    "F5": "F5",
    "F6": "F6",
    "F7": "F7",
    "F8": "F8",
    "F9": "F9",
    "F10": "F10",
    "F11": "F11",
    "F12": "F12",
}

# Ctrl-key combos: "C-a" = Ctrl+A, etc.
CTRL_KEY_MAP = {
    "a": "C-a", "b": "C-b", "c": "C-c", "d": "C-d",
    "e": "C-e", "f": "C-f", "g": "C-g", "k": "C-k",
    "l": "C-l", "n": "C-n", "p": "C-p", "r": "C-r",
    "u": "C-u", "w": "C-w", "z": "C-z",
}


class TmuxSessionManager:
    """Manages tmux sessions: create, destroy, capture, send keys."""

    def __init__(self):
        self._sessions: dict[str, dict] = {}  # sid -> session info
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Helpers: subprocess commands
    # ------------------------------------------------------------------

    @staticmethod
    def _tmux_cmd(args: list[str], timeout: int = 10) -> str:
        cmd = ["tmux"] + args
        try:
            out = subprocess.check_output(
                cmd, text=True, stderr=subprocess.DEVNULL, timeout=timeout,
            )
            return out
        except subprocess.CalledProcessError as e:
            logger.error("[webtmux] tmux command failed: %s -> %s", cmd, e)
            return ""
        except FileNotFoundError:
            logger.error("[webtmux] tmux binary not found in PATH")
            return ""

    def _session_exists(self, name: str) -> bool:
        try:
            subprocess.check_output(
                ["tmux", "has-session", "-t", name],
                stderr=subprocess.DEVNULL, timeout=5,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_pane_ids(self, session_name: str) -> list[str]:
        """Return list of %pane_id values for a session."""
        raw = self._tmux_cmd([
            "list-panes", "-t", session_name,
            "-F", "#{pane_id}",
        ]).strip().splitlines()
        return [p.strip() for p in raw if p.strip()]

    def _get_active_pane_id(self, session_name: str) -> str | None:
        """Return the %pane_id of the currently active pane in a session."""
        raw = self._tmux_cmd([
            "list-panes", "-t", session_name,
            "-F", "#{pane_active}:#{pane_id}",
        ]).strip().splitlines()
        for line in raw:
            line = line.strip()
            if line.startswith("1:"):
                return line.split(":", 1)[1]
        return None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def create_session(self, sid: str, cwd: str | None = None) -> dict:
        """Create a new tmux session with a reliable 2x2 pane grid.

        Uses %pane_id targeting which is base-index-independent (works
        regardless of ``base-index`` or ``pane-base-index`` settings).

        Layout:
            +-------------+-------------+
            | pane 0      | pane 1      |
            | (hermes)    | (nvim)      |
            +-------------+-------------+
            | pane 2      | pane 3      |
            | (terminal)  | (terminal)  |
            +-------------+-------------+

        Returns {"session_name": str, "pane_ids": list[str], "cwd": str}.
        """
        async with self._lock:
            if sid in self._sessions:
                return self._sessions[sid]

            session_name = f"webtmux-{sid}"
            work_dir = cwd or os.getcwd()

            # 1. Create session (gives us the initial pane — top-left)
            self._tmux_cmd([
                "new-session", "-d",
                "-s", session_name,
                "-c", work_dir,
                "-x", "120", "-y", "40",
            ])

            # Capture the initial pane's %id (base-index-independent)
            initial_ids = self._get_pane_ids(session_name)
            if not initial_ids:
                logger.error("[webtmux] Failed to create session %s", session_name)
                return {"session_name": session_name, "pane_ids": [], "cwd": work_dir}
            initial_pane = initial_ids[0]

            # 2. Split initial pane horizontally → creates top-right pane
            self._tmux_cmd([
                "split-window", "-h",
                "-t", initial_pane,
                "-c", work_dir,
            ])
            # The new pane is now active; capture its ID
            right_pane = self._get_active_pane_id(session_name) or ""

            # 3. Split initial pane vertically → creates bottom-left pane
            self._tmux_cmd([
                "split-window", "-v",
                "-t", initial_pane,
                "-c", work_dir,
            ])
            bottom_left = self._get_active_pane_id(session_name) or ""

            # 4. Split right pane vertically → creates bottom-right pane
            self._tmux_cmd([
                "split-window", "-v",
                "-t", right_pane,
                "-c", work_dir,
            ])
            bottom_right = self._get_active_pane_id(session_name) or ""

            # Build explicit pane_ids list (no reliance on list-panes ordering)
            pane_ids = [initial_pane, right_pane, bottom_left, bottom_right]
            pane_ids = [p for p in pane_ids if p]  # filter out empty strings

            if len(pane_ids) < 4:
                logger.error(
                    "[webtmux] Expected 4 panes, got %d for session %s",
                    len(pane_ids), session_name,
                )

            info = {
                "session_name": session_name,
                "pane_ids": pane_ids,
                "cwd": work_dir,
            }
            self._sessions[sid] = info
            logger.info(
                "[webtmux] Created session %s with panes %s",
                session_name, pane_ids,
            )

            # Start hermes chat in pane 0 (literal text + separate Enter)
            if len(pane_ids) >= 1:
                self._tmux_cmd([
                    "send-keys", "-l", "-t", pane_ids[0], "hermes chat",
                ])
                self._tmux_cmd([
                    "send-keys", "-t", pane_ids[0], "Enter",
                ])

            # Pane 1 is a plain terminal (nvim removed — too heavy for now)

            return info

    async def destroy_session(self, sid: str) -> None:
        async with self._lock:
            info = self._sessions.pop(sid, None)
            if info is None:
                return
            session_name = info["session_name"]
            self._tmux_cmd(["kill-session", "-t", session_name])
            logger.info("[webtmux] Destroyed session %s", session_name)

    async def list_sessions(self) -> list[dict]:
        async with self._lock:
            return [
                {"id": sid, "name": info["session_name"], "panes": len(info["pane_ids"])}
                for sid, info in self._sessions.items()
            ]

    # ------------------------------------------------------------------
    # Pane controls
    # ------------------------------------------------------------------

    async def capture_pane(self, sid: str, pane_index: int) -> str:
        """Capture the text content of a pane."""
        info = self._sessions.get(sid)
        if info is None:
            return ""
        pane_ids = info["pane_ids"]
        if pane_index >= len(pane_ids):
            return ""
        return self._tmux_cmd(["capture-pane", "-p", "-e", "-t", pane_ids[pane_index]])

    async def send_keys(self, sid: str, pane_index: int, key: str, *, ctrl: bool = False) -> bool:
        """Send a keypress to a tmux pane.

        - For literal characters: uses ``tmux send-keys -l <char>``
        - For special keys (Enter, arrows, etc.): uses ``tmux send-keys <TmuxName>``
        - For Ctrl combos: uses ``tmux send-keys C-<key>``
        """
        info = self._sessions.get(sid)
        if info is None:
            return False
        pane_ids = info["pane_ids"]
        if pane_index >= len(pane_ids):
            return False
        pane_id = pane_ids[pane_index]

        # Ctrl+key combos
        if ctrl and key.lower() in CTRL_KEY_MAP:
            tmux_key = CTRL_KEY_MAP[key.lower()]
            self._tmux_cmd(["send-keys", "-t", pane_id, tmux_key])
            return True

        # Named special keys
        tmux_name = SPECIAL_KEY_MAP.get(key)
        if tmux_name:
            self._tmux_cmd(["send-keys", "-t", pane_id, tmux_name])
            return True

        # Literal character (single printable char)
        if len(key) == 1:
            self._tmux_cmd(["send-keys", "-l", "-t", pane_id, key])
            return True

        # Unrecognized key — ignore
        logger.debug("[webtmux] Ignoring unrecognized key: %r", key)
        return False

    async def send_text(self, sid: str, pane_index: int, text: str) -> bool:
        """Send a literal string to a tmux pane in one call (batch input)."""
        info = self._sessions.get(sid)
        if info is None:
            return False
        pane_ids = info["pane_ids"]
        if pane_index >= len(pane_ids):
            return False
        if text:
            self._tmux_cmd(["send-keys", "-l", "-t", pane_ids[pane_index], text])
        return True

    async def resize_pane(self, sid: str, pane_index: int, width: int, height: int) -> None:
        info = self._sessions.get(sid)
        if info is None:
            return
        pane_ids = info["pane_ids"]
        if pane_index >= len(pane_ids):
            return
        pane_id = pane_ids[pane_index]
        self._tmux_cmd(["resize-pane", "-x", str(width), "-y", str(height), "-t", pane_id])

    async def zoom_pane(self, sid: str, pane_index: int) -> None:
        """Toggle tmux native zoom on a pane (resize-pane -Z)."""
        info = self._sessions.get(sid)
        if info is None:
            return
        pane_ids = info["pane_ids"]
        if pane_index >= len(pane_ids):
            return
        self._tmux_cmd(["resize-pane", "-Z", "-t", pane_ids[pane_index]])

    async def equalize_layout(self, sid: str) -> None:
        """Reset tmux layout to tiled (equal pane sizes)."""
        info = self._sessions.get(sid)
        if info is None:
            return
        self._tmux_cmd(["select-layout", "-t", info["session_name"], "tiled"])


def check_webtmux_requirements() -> bool:
    """Check if tmux and aiohttp are available."""
    if not AIOHTTP_AVAILABLE:
        return False
    if not shutil.which("tmux"):
        return False
    try:
        subprocess.check_output(["tmux", "-V"], stderr=subprocess.DEVNULL, timeout=3)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


class WebTmuxAdapter(BasePlatformAdapter):
    """
    Gateway platform adapter that serves tmux sessions over HTTP/WebSocket.
    """

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("webtmux"))
        self._host: str = str(
            os.environ.get("WEBTMUX_HOST") or config.extra.get("host", DEFAULT_HOST)
        )
        self._port: int = int(
            os.environ.get("WEBTMUX_PORT") or config.extra.get("port", DEFAULT_PORT)
        )
        self._frame_interval: float = float(
            config.extra.get("frame_interval", DEFAULT_FRAME_INTERVAL)
        )
        self._access_code: str = str(
            os.environ.get("WEBTMUX_ACCESS_CODE") or config.extra.get("access_code", "")
        )
        self._max_sessions: int = int(config.extra.get("max_sessions", 10))

        self._runner: Any = None
        self._tmux = TmuxSessionManager()

        # Active WebSocket connections: ws -> {"sid": str, "tasks": list}
        self._clients: dict[Any, dict] = {}

        # Session -> list of connected ws objects
        self._session_ws: dict[str, list[Any]] = {}

        # Track sent message IDs for edit_message support (monotonic counter)
        self._msg_counter: int = 0

    # ------------------------------------------------------------------
    # Gateway interface: required methods
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        if not check_webtmux_requirements():
            logger.error("[webtmux] tmux or aiohttp not available")
            return False

        app = web.Application()
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/static/{path:.*}", self._handle_static)
        app.router.add_get("/api/sessions", self._handle_list_sessions)
        app.router.add_post("/api/sessions", self._handle_create_session)
        app.router.add_delete("/api/sessions/{sid}", self._handle_destroy_session)
        app.router.add_get("/ws", self._handle_ws)

        # Port availability check
        import socket as _socket
        try:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
                _s.settimeout(1)
                _s.connect(("127.0.0.1", self._port))
            logger.error("[webtmux] Port %d already in use", self._port)
            return False
        except (ConnectionRefusedError, OSError):
            pass  # Port is free

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        self._mark_connected()

        code_hint = f"?code={self._access_code}" if self._access_code else ""
        logger.info(
            "[webtmux] Web UI at http://%s:%d/%s",
            self._host, self._port, code_hint,
        )
        return True

    async def disconnect(self) -> None:
        for ws, info in list(self._clients.items()):
            for t in info.get("tasks", []):
                t.cancel()
            try:
                await ws.close()
            except Exception:
                pass
        self._clients.clear()
        self._session_ws.clear()

        if self._runner:
            await self._runner.cleanup()
            self._runner = None

        # Destroy all tmux sessions
        for sid in list(self._tmux._sessions.keys()):
            await self._tmux.destroy_session(sid)

        self._mark_disconnected()
        logger.info("[webtmux] Disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        metadata: dict | None = None,
    ) -> SendResult:
        """Deliver an agent reply to the web UI.

        chat_id format: ``webtmux:<sid>:<pane_index>``
        """
        parts = chat_id.split(":", 2)
        if len(parts) < 2:
            return SendResult(success=False, error="Invalid chat_id format")
        sid = parts[1]

        self._msg_counter += 1
        msg_id = f"wt-{self._msg_counter}"

        payload = {
            "type": "chat_reply",
            "sid": sid,
            "message_id": msg_id,
            "content": content,
            "final": True,
        }
        await self._broadcast_to_session(sid, payload)
        return SendResult(success=True, message_id=msg_id)

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
    ) -> SendResult:
        """Edit a previously sent message (streaming support).

        The web UI replaces the content of the message with the given ID.
        """
        parts = chat_id.split(":", 2)
        if len(parts) < 2:
            return SendResult(success=False, error="Invalid chat_id format")
        sid = parts[1]

        payload = {
            "type": "chat_edit",
            "sid": sid,
            "message_id": message_id,
            "content": content,
            "final": finalize,
        }
        await self._broadcast_to_session(sid, payload)
        return SendResult(success=True, message_id=message_id)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return chat info for a webtmux session."""
        parts = chat_id.split(":", 2)
        sid = parts[1] if len(parts) >= 2 else chat_id
        return {
            "name": f"webtmux-session-{sid}",
            "type": "dm",
            "platform": "webtmux",
        }

    async def send_typing(self, chat_id: str, metadata: Any = None) -> None:
        """Show typing indicator in the web UI."""
        parts = chat_id.split(":", 2)
        sid = parts[1] if len(parts) >= 2 else ""
        await self._broadcast_to_session(sid, {"type": "typing", "sid": sid})

    # ------------------------------------------------------------------
    # Broadcast helpers
    # ------------------------------------------------------------------

    async def _broadcast_to_session(self, sid: str, payload: dict) -> None:
        msg = json.dumps(payload)
        dead: list = []
        for ws, info in self._clients.items():
            if info.get("sid") == sid:
                try:
                    await ws.send_str(msg)
                except Exception:
                    dead.append(ws)
        for ws in dead:
            self._unregister_ws(ws)

    def _register_ws(self, ws: Any, sid: str) -> None:
        self._clients[ws] = {"sid": sid, "tasks": []}
        if sid not in self._session_ws:
            self._session_ws[sid] = []
        self._session_ws[sid].append(ws)

    def _unregister_ws(self, ws: Any) -> None:
        info = self._clients.pop(ws, None)
        if info:
            sid = info.get("sid")
            if sid and sid in self._session_ws:
                try:
                    self._session_ws[sid].remove(ws)
                except ValueError:
                    pass
                if not self._session_ws[sid]:
                    self._session_ws.pop(sid, None)
            for t in info.get("tasks", []):
                t.cancel()

    # ------------------------------------------------------------------
    # Background tasks: pane capture streaming
    # ------------------------------------------------------------------

    async def _stream_frames(self, ws: Any, sid: str) -> None:
        """Periodically capture tmux panes and send snapshots to the client."""
        try:
            while not ws.closed:
                snapshots = []
                for i in range(PANES_PER_SESSION):
                    text = await self._tmux.capture_pane(sid, i)
                    snapshots.append(text)
                payload = {
                    "type": "pane_snapshot",
                    "sid": sid,
                    "panes": snapshots,
                }
                try:
                    await ws.send_str(json.dumps(payload))
                except Exception:
                    break
                await asyncio.sleep(self._frame_interval)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug("[webtmux] Frame stream error for %s: %s", sid, e)

    # ------------------------------------------------------------------
    # HTTP handlers
    # ------------------------------------------------------------------

    def _check_access(self, request: Any) -> bool:
        """Validate access code if one is configured."""
        if not self._access_code:
            return True
        code = request.query.get("code", "") or request.headers.get("X-Access-Code", "")
        return code == self._access_code

    async def _handle_index(self, request: Any) -> Any:
        if not self._check_access(request):
            raise web.HTTPUnauthorized(text="Invalid or missing access code")
        return web.FileResponse(self._static_path("index.html"))

    async def _handle_static(self, request: Any) -> Any:
        path = request.match_info["path"]
        file_path = self._static_path(path)
        if not file_path.exists():
            raise web.HTTPNotFound()
        return web.FileResponse(file_path)

    def _static_path(self, rel: str) -> Any:
        from pathlib import Path
        return Path(__file__).parent / "static" / rel

    async def _handle_list_sessions(self, request: Any) -> Any:
        if not self._check_access(request):
            raise web.HTTPUnauthorized()
        sessions = await self._tmux.list_sessions()
        return web.json_response({"sessions": sessions})

    async def _handle_create_session(self, request: Any) -> Any:
        if not self._check_access(request):
            raise web.HTTPUnauthorized()

        # Enforce max_sessions limit
        current = await self._tmux.list_sessions()
        if len(current) >= self._max_sessions:
            raise web.HTTPBadRequest(
                text=f"Max sessions ({self._max_sessions}) reached",
            )

        data = await request.json() if request.can_read_body else {}
        sid = data.get("id") or f"sess_{int(time.time() * 1000)}"
        cwd = data.get("cwd")
        info = await self._tmux.create_session(sid, cwd=cwd)
        return web.json_response({
            "id": sid,
            "name": info["session_name"],
            "panes": info["pane_ids"],
        })

    async def _handle_destroy_session(self, request: Any) -> Any:
        if not self._check_access(request):
            raise web.HTTPUnauthorized()
        sid = request.match_info["sid"]

        # Close all websockets for this session
        for ws in list(self._session_ws.get(sid, [])):
            try:
                await ws.close(code=4002, message=b"Session destroyed")
            except Exception:
                pass
            self._unregister_ws(ws)

        await self._tmux.destroy_session(sid)
        return web.json_response({"ok": True})

    # ------------------------------------------------------------------
    # WebSocket handler
    # ------------------------------------------------------------------

    async def _handle_ws(self, request: Any) -> Any:
        if not self._check_access(request):
            raise web.HTTPUnauthorized()
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        sid = request.query.get("sid", "")
        if not sid:
            await ws.close(code=4000, message=b"Missing sid parameter")
            return ws

        if sid not in self._tmux._sessions:
            await ws.close(code=4001, message=b"Unknown session")
            return ws

        self._register_ws(ws, sid)
        logger.info("[webtmux] WS connected to session %s", sid)

        # Start frame streaming task
        task = asyncio.create_task(self._stream_frames(ws, sid))
        self._clients[ws]["tasks"].append(task)

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    await self._handle_ws_message(ws, sid, msg.data)
                elif msg.type == WSMsgType.ERROR:
                    logger.debug("[webtmux] WS error: %s", ws.exception())
        finally:
            self._unregister_ws(ws)
            logger.info("[webtmux] WS disconnected from session %s", sid)
        return ws

    async def _handle_ws_message(self, ws: Any, sid: str, data: str) -> None:
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return
        msg_type = payload.get("type")

        if msg_type == "chat":
            # User sent a chat message to Hermes
            text = payload.get("text", "").strip()
            if not text:
                return
            source = self.build_source(
                chat_id=f"webtmux:{sid}:0",
                chat_name=f"webtmux-session-{sid}",
                chat_type="dm",
                user_id="webtmux-user",
                user_name="WebTmux User",
            )
            event = MessageEvent(
                text=text,
                message_type=MessageType.TEXT,
                source=source,
                raw_message=payload,
            )
            await self.handle_message(event)

        elif msg_type == "key":
            # Forward keypress to a tmux pane
            pane_index = int(payload.get("pane", 0))
            key = payload.get("key", "")
            ctrl = payload.get("ctrl", False)
            if key:
                await self._tmux.send_keys(sid, pane_index, key, ctrl=ctrl)

        elif msg_type == "keys":
            # Batch: send literal text string in one tmux call
            pane_index = int(payload.get("pane", 0))
            text = payload.get("text", "")
            if text:
                await self._tmux.send_text(sid, pane_index, text)

        elif msg_type == "resize":
            pane_index = int(payload.get("pane", 0))
            width = int(payload.get("width", 120))
            height = int(payload.get("height", 40))
            await self._tmux.resize_pane(sid, pane_index, width, height)

        elif msg_type == "zoom":
            pane_index = int(payload.get("pane", 0))
            await self._tmux.zoom_pane(sid, pane_index)

        elif msg_type == "equalize":
            await self._tmux.equalize_layout(sid)


# ------------------------------------------------------------------
# Standalone sender stub (cron compatibility)
# ------------------------------------------------------------------

async def standalone_send_webtmux(
    pconfig, chat_id, message, *, thread_id=None, media_files=None, force_document=False
):
    return {"error": "WebTmux does not support standalone sends"}
