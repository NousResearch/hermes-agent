"""Async mpv IPC client.

Spawns mpv as a subprocess and communicates via JSON IPC over a Unix domain
socket.  Provides play/pause/skip/volume/seek controls and emits callbacks
on metadata changes and track-end events.
"""

import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def _default_socket_path(label: str = "main") -> str:
    """Return a per-user socket path that won't collide."""
    return os.path.join(tempfile.gettempdir(), f"hermes-radio-{label}-{os.getuid()}.sock")


class MpvClient:
    """Async controller for a headless mpv instance via JSON IPC."""

    def __init__(self, socket_path: Optional[str] = None, label: str = "main"):
        self._socket_path = socket_path or _default_socket_path(label)
        self._label = label
        self._process: Optional[subprocess.Popen] = None
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._request_id = 0
        self._pending: Dict[int, asyncio.Future] = {}
        self._event_callbacks: Dict[str, List[Callable]] = {}
        self._listen_task: Optional[asyncio.Task] = None
        self._connected = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Spawn mpv and connect to its IPC socket."""
        mpv_bin = shutil.which("mpv")
        if not mpv_bin:
            raise RuntimeError("mpv not found in PATH. Install: brew install mpv")

        # Clean up stale socket
        if os.path.exists(self._socket_path):
            os.unlink(self._socket_path)

        self._process = subprocess.Popen(
            [
                mpv_bin,
                "--idle",
                "--no-video",
                "--no-terminal",
                "--really-quiet",
                f"--input-ipc-server={self._socket_path}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("[%s] mpv started (pid %d, socket %s)", self._label, self._process.pid, self._socket_path)

        # Wait for socket to appear
        for _ in range(50):
            if os.path.exists(self._socket_path):
                break
            await asyncio.sleep(0.1)
        else:
            raise RuntimeError(f"mpv IPC socket did not appear at {self._socket_path}")

        await self._connect()

    async def _connect(self) -> None:
        """Open the Unix socket connection and start the listener."""
        self._reader, self._writer = await asyncio.open_unix_connection(self._socket_path)
        self._connected = True
        self._listen_task = asyncio.create_task(self._listen_loop())
        logger.info("[%s] connected to mpv IPC", self._label)

    async def stop(self) -> None:
        """Shut down mpv and clean up."""
        self._connected = False

        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass

        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None
            self._reader = None

        if self._process and self._process.poll() is None:
            self._process.send_signal(signal.SIGTERM)
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)
            logger.info("[%s] mpv stopped", self._label)

        if os.path.exists(self._socket_path):
            try:
                os.unlink(self._socket_path)
            except OSError:
                pass

        # Reject pending futures
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(ConnectionError("mpv stopped"))
        self._pending.clear()

    @property
    def running(self) -> bool:
        return self._connected and self._process is not None and self._process.poll() is None

    # ------------------------------------------------------------------
    # Event system
    # ------------------------------------------------------------------

    def on(self, event_name: str, callback: Callable) -> None:
        """Register a callback for an mpv event (e.g. 'end-file', 'metadata-update')."""
        self._event_callbacks.setdefault(event_name, []).append(callback)

    def _fire_event(self, event_name: str, data: dict) -> None:
        for cb in self._event_callbacks.get(event_name, []):
            try:
                result = cb(data)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception:
                logger.exception("[%s] event callback error for %s", self._label, event_name)

    # ------------------------------------------------------------------
    # IPC communication
    # ------------------------------------------------------------------

    async def _send(self, command: list, *, request_id: Optional[int] = None) -> int:
        """Send a JSON command to mpv.  Returns the request_id used."""
        if not self._writer or not self._connected:
            raise ConnectionError("Not connected to mpv")

        if request_id is None:
            self._request_id += 1
            request_id = self._request_id

        msg = json.dumps({"command": command, "request_id": request_id}) + "\n"
        self._writer.write(msg.encode())
        await self._writer.drain()
        return request_id

    async def command(self, *args) -> Any:
        """Send a command and wait for the response."""
        self._request_id += 1
        rid = self._request_id
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[rid] = fut
        await self._send(list(args), request_id=rid)
        try:
            return await asyncio.wait_for(fut, timeout=5.0)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            raise TimeoutError(f"mpv command timed out: {args}")

    async def _listen_loop(self) -> None:
        """Read and dispatch messages from mpv."""
        try:
            while self._connected and self._reader:
                line = await self._reader.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line.decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

                # Response to a command
                if "request_id" in msg and msg["request_id"] in self._pending:
                    rid = msg["request_id"]
                    fut = self._pending.pop(rid)
                    if not fut.done():
                        if msg.get("error") == "success":
                            fut.set_result(msg.get("data"))
                        else:
                            fut.set_exception(RuntimeError(f"mpv error: {msg.get('error')}"))

                # Event
                if "event" in msg:
                    self._fire_event(msg["event"], msg)

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("[%s] mpv listen loop error", self._label)
        finally:
            self._connected = False

    # ------------------------------------------------------------------
    # Playback controls
    # ------------------------------------------------------------------

    async def loadfile(self, path_or_url: str, mode: str = "replace") -> Any:
        """Load a file or URL.  mode: 'replace' | 'append' | 'append-play'."""
        return await self.command("loadfile", path_or_url, mode)

    async def play(self) -> None:
        """Resume playback."""
        await self.command("set_property", "pause", False)

    async def pause(self) -> None:
        """Pause playback."""
        await self.command("set_property", "pause", True)

    async def toggle_pause(self) -> None:
        paused = await self.get_property("pause")
        await self.command("set_property", "pause", not paused)

    async def stop(self) -> None:
        """Stop playback (clear playlist)."""
        await self.command("stop")

    async def playlist_next(self) -> None:
        await self.command("playlist-next")

    async def playlist_prev(self) -> None:
        await self.command("playlist-prev")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    async def get_property(self, name: str) -> Any:
        """Get an mpv property value."""
        return await self.command("get_property", name)

    async def set_property(self, name: str, value: Any) -> None:
        """Set an mpv property."""
        await self.command("set_property", name, value)

    async def observe_property(self, prop_id: int, name: str) -> None:
        """Subscribe to property change events."""
        await self.command("observe_property", prop_id, name)

    async def get_volume(self) -> float:
        return await self.get_property("volume")

    async def set_volume(self, level: float) -> None:
        """Set volume (0-100)."""
        await self.set_property("volume", max(0, min(100, level)))

    async def get_position(self) -> Optional[float]:
        """Get current playback position in seconds, or None."""
        try:
            return await self.get_property("time-pos")
        except Exception:
            return None

    async def get_duration(self) -> Optional[float]:
        """Get current track duration in seconds, or None."""
        try:
            return await self.get_property("duration")
        except Exception:
            return None

    async def get_metadata(self) -> Dict[str, str]:
        """Get current track metadata dict."""
        try:
            meta = await self.get_property("metadata")
            return meta if isinstance(meta, dict) else {}
        except Exception:
            return {}

    async def get_media_title(self) -> str:
        """Get the current media-title (ICY or filename)."""
        try:
            title = await self.get_property("media-title")
            return str(title) if title else ""
        except Exception:
            return ""

    async def is_paused(self) -> bool:
        try:
            return bool(await self.get_property("pause"))
        except Exception:
            return False

    async def is_idle(self) -> bool:
        try:
            return bool(await self.get_property("idle-active"))
        except Exception:
            return True

    # ------------------------------------------------------------------
    # Volume ramp (for mic break ducking)
    # ------------------------------------------------------------------

    async def ramp_volume(self, target: float, duration_ms: int = 500, steps: int = 10) -> None:
        """Smoothly ramp volume from current level to target over duration_ms."""
        try:
            current = await self.get_volume()
        except Exception:
            current = 100.0
        delta = target - current
        step_delay = duration_ms / 1000.0 / steps
        for i in range(1, steps + 1):
            vol = current + (delta * i / steps)
            try:
                await self.set_volume(vol)
            except Exception:
                break
            if i < steps:
                await asyncio.sleep(step_delay)

    # ------------------------------------------------------------------
    # Status snapshot
    # ------------------------------------------------------------------

    async def status(self) -> Dict[str, Any]:
        """Return a snapshot of the current playback state."""
        if not self.running:
            return {"playing": False, "idle": True}

        idle = await self.is_idle()
        if idle:
            return {"playing": False, "idle": True}

        paused = await self.is_paused()
        title = await self.get_media_title()
        pos = await self.get_position()
        dur = await self.get_duration()
        vol = await self.get_volume()
        meta = await self.get_metadata()

        return {
            "playing": not paused,
            "idle": False,
            "paused": paused,
            "title": title,
            "position": pos,
            "duration": dur,
            "volume": vol,
            "metadata": meta,
        }
