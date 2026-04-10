"""
Weixin platform adapter.

This adapter uses a lightweight Node.js bridge that implements:
- QR login against the iLink Weixin bot endpoints
- Long-poll getUpdates monitoring
- Text message sending via the same bot token
"""

import asyncio
import logging
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

_IS_WINDOWS = platform.system() == "Windows"

from hermes_constants import get_hermes_dir

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult

logger = logging.getLogger(__name__)


def _kill_port_process(port: int) -> None:
    """Kill any process listening on the given TCP port."""
    try:
        if _IS_WINDOWS:
            result = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 5 and parts[3] == "LISTENING":
                    local_addr = parts[1]
                    if local_addr.endswith(f":{port}"):
                        try:
                            subprocess.run(
                                ["taskkill", "/PID", parts[4], "/F"],
                                capture_output=True,
                                timeout=5,
                            )
                        except subprocess.SubprocessError:
                            pass
        else:
            result = subprocess.run(
                ["fuser", f"{port}/tcp"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                subprocess.run(
                    ["fuser", "-k", f"{port}/tcp"],
                    capture_output=True,
                    timeout=5,
                )
    except Exception:
        pass


def check_weixin_requirements() -> bool:
    """Check if the local environment can run the Weixin bridge."""
    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


class WeixinAdapter(BasePlatformAdapter):
    """Weixin adapter backed by the local Node bridge."""

    MAX_MESSAGE_LENGTH = 4000
    _DEFAULT_BRIDGE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "weixin-bridge"

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WEIXIN)
        self._bridge_process: Optional[subprocess.Popen] = None
        self._bridge_port: int = int(config.extra.get("bridge_port", 3010))
        self._bridge_script: str = str(Path(
            config.extra.get("bridge_script", self._DEFAULT_BRIDGE_DIR / "bridge.js")
        ).expanduser())
        self._session_path: Path = Path(
            config.extra.get(
                "session_path",
                get_hermes_dir("weixin/session", "platforms/weixin/session"),
            )
        ).expanduser()
        self._bridge_log: Optional[Path] = None
        self._bridge_log_fh = None
        self._poll_task: Optional[asyncio.Task] = None
        self._http_session: Optional["aiohttp.ClientSession"] = None

    async def connect(self) -> bool:
        """Start the Weixin bridge and begin polling messages."""
        if not check_weixin_requirements():
            logger.warning("[%s] Node.js not found. Weixin requires Node.js.", self.name)
            return False

        bridge_path = Path(self._bridge_script)
        if not bridge_path.exists():
            logger.warning("[%s] Bridge script not found: %s", self.name, bridge_path)
            return False

        bridge_dir = bridge_path.parent
        if not (bridge_dir / "node_modules").exists():
            logger.info("[%s] Installing Weixin bridge dependencies...", self.name)
            try:
                install_result = subprocess.run(
                    ["npm", "install", "--silent"],
                    cwd=str(bridge_dir),
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if install_result.returncode != 0:
                    logger.error("[%s] npm install failed: %s", self.name, install_result.stderr)
                    return False
            except Exception as exc:
                logger.error("[%s] Failed to install bridge dependencies: %s", self.name, exc)
                return False

        self._session_path.mkdir(parents=True, exist_ok=True)
        self._bridge_log = self._session_path.parent / "bridge.log"

        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://127.0.0.1:{self._bridge_port}/health",
                    timeout=aiohttp.ClientTimeout(total=2),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        status = str(data.get("status", "unknown"))
                        if status == "connected":
                            logger.info("[%s] Using existing bridge (status=%s)", self.name, status)
                            self._bridge_process = None
                            self._http_session = aiohttp.ClientSession()
                            self._poll_task = asyncio.create_task(self._poll_messages())
                            self._mark_connected()
                            return True
                        logger.info(
                            "[%s] Bridge found but not connected (status=%s), restarting",
                            self.name,
                            status,
                        )
        except Exception:
            pass

        _kill_port_process(self._bridge_port)
        await asyncio.sleep(1)

        try:
            bridge_log_fh = open(self._bridge_log, "a")
            self._bridge_log_fh = bridge_log_fh
            self._bridge_process = subprocess.Popen(
                [
                    "node",
                    str(bridge_path),
                    "--port",
                    str(self._bridge_port),
                    "--session",
                    str(self._session_path),
                ],
                stdout=bridge_log_fh,
                stderr=bridge_log_fh,
                preexec_fn=None if _IS_WINDOWS else os.setsid,
                env=os.environ.copy(),
            )

            http_ready = False
            data: Dict[str, Any] = {}
            for _ in range(15):
                await asyncio.sleep(1)
                if self._bridge_process.poll() is not None:
                    logger.error(
                        "[%s] Bridge process exited with code %s. Check %s",
                        self.name,
                        self._bridge_process.returncode,
                        self._bridge_log,
                    )
                    await self._stop_managed_bridge()
                    return False
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            f"http://127.0.0.1:{self._bridge_port}/health",
                            timeout=aiohttp.ClientTimeout(total=2),
                        ) as resp:
                            if resp.status == 200:
                                http_ready = True
                                data = await resp.json()
                                if data.get("status") == "connected":
                                    logger.info("[%s] Bridge ready (status=connected)", self.name)
                                    break
                except Exception:
                    continue

            if not http_ready:
                logger.error("[%s] Bridge HTTP server did not start. Check %s", self.name, self._bridge_log)
                await self._stop_managed_bridge()
                return False

            if data.get("status") != "connected":
                logger.info("[%s] Bridge HTTP ready, waiting for Weixin connection...", self.name)
                for _ in range(15):
                    await asyncio.sleep(1)
                    if self._bridge_process.poll() is not None:
                        logger.error(
                            "[%s] Bridge process exited during connection with code %s. Check %s",
                            self.name,
                            self._bridge_process.returncode,
                            self._bridge_log,
                        )
                        await self._stop_managed_bridge()
                        return False
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(
                                f"http://127.0.0.1:{self._bridge_port}/health",
                                timeout=aiohttp.ClientTimeout(total=2),
                            ) as resp:
                                if resp.status == 200:
                                    data = await resp.json()
                                    if data.get("status") == "connected":
                                        logger.info("[%s] Bridge ready (status=connected)", self.name)
                                        break
                    except Exception:
                        continue
                else:
                    last_status = str(data.get("status", "starting"))
                    logger.warning(
                        "[%s] Bridge did not reach connected state within 30s (status=%s). "
                        "Finish pairing with `hermes weixin` first. Check %s",
                        self.name,
                        last_status,
                        self._bridge_log,
                    )
                    await self._stop_managed_bridge()
                    return False

            self._http_session = aiohttp.ClientSession()
            self._poll_task = asyncio.create_task(self._poll_messages())
            self._mark_connected()
            logger.info(
                "[%s] Bridge started on port %s (status=%s, log=%s)",
                self.name,
                self._bridge_port,
                data.get("status", "connected"),
                self._bridge_log,
            )
            return True
        except Exception as exc:
            logger.error("[%s] Failed to start bridge: %s", self.name, exc, exc_info=True)
            if self._bridge_process is not None:
                await self._stop_managed_bridge()
            else:
                self._close_bridge_log()
            return False

    def _close_bridge_log(self) -> None:
        if self._bridge_log_fh:
            try:
                self._bridge_log_fh.close()
            except Exception:
                pass
            self._bridge_log_fh = None

    async def _stop_managed_bridge(self) -> None:
        proc = self._bridge_process
        self._bridge_process = None
        if proc is None:
            self._close_bridge_log()
            return

        if proc.poll() is None:
            try:
                import signal

                try:
                    if _IS_WINDOWS:
                        proc.terminate()
                    else:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    proc.terminate()
                await asyncio.sleep(1)
                if proc.poll() is None:
                    try:
                        if _IS_WINDOWS:
                            proc.kill()
                        else:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        proc.kill()
            except Exception as exc:
                logger.warning("[%s] Error stopping bridge: %s", self.name, exc)

        self._close_bridge_log()

    async def _check_managed_bridge_exit(self) -> Optional[str]:
        if self._bridge_process is None:
            return None

        returncode = self._bridge_process.poll()
        if returncode is None:
            return None

        message = f"Weixin bridge process exited unexpectedly (code {returncode})."
        if not self.has_fatal_error:
            logger.error("[%s] %s", self.name, message)
            self._set_fatal_error("weixin_bridge_exited", message, retryable=True)
            self._close_bridge_log()
            await self._notify_fatal_error()
        return self.fatal_error_message or message

    async def disconnect(self) -> None:
        """Stop the bridge and clean up resources."""
        await self._stop_managed_bridge()

        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except (asyncio.CancelledError, Exception):
                pass
        self._poll_task = None

        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
        self._http_session = None

        self._mark_disconnected()
        self._bridge_process = None
        self._close_bridge_log()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text message through the local bridge."""
        if not self._running or not self._http_session:
            return SendResult(success=False, error="Not connected")
        bridge_exit = await self._check_managed_bridge_exit()
        if bridge_exit:
            return SendResult(success=False, error=bridge_exit)

        try:
            import aiohttp

            payload: Dict[str, Any] = {"chatId": chat_id, "message": content}
            if reply_to:
                payload["replyTo"] = reply_to
            async with self._http_session.post(
                f"http://127.0.0.1:{self._bridge_port}/send",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return SendResult(
                        success=True,
                        message_id=data.get("messageId"),
                        raw_response=data,
                    )
                return SendResult(success=False, error=await resp.text())
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Best-effort typing indicator through the bridge."""
        if not self._running or not self._http_session:
            return
        if await self._check_managed_bridge_exit():
            return

        try:
            import aiohttp

            await self._http_session.post(
                f"http://127.0.0.1:{self._bridge_port}/typing",
                json={"chatId": chat_id},
                timeout=aiohttp.ClientTimeout(total=5),
            )
        except Exception:
            pass

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic information about a Weixin conversation."""
        if not self._running or not self._http_session:
            return {"name": chat_id, "type": "dm"}
        if await self._check_managed_bridge_exit():
            return {"name": chat_id, "type": "dm"}

        try:
            import aiohttp

            async with self._http_session.get(
                f"http://127.0.0.1:{self._bridge_port}/chat/{chat_id}",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        "name": data.get("name", chat_id),
                        "type": "group" if data.get("isGroup") else "dm",
                    }
        except Exception as exc:
            logger.debug("Could not get Weixin chat info for %s: %s", chat_id, exc)

        return {"name": chat_id, "type": "dm"}

    async def _poll_messages(self) -> None:
        """Poll the bridge for normalized inbound messages."""
        import aiohttp

        while self._running:
            if not self._http_session:
                break
            bridge_exit = await self._check_managed_bridge_exit()
            if bridge_exit:
                logger.error("[%s] %s", self.name, bridge_exit)
                break
            try:
                async with self._http_session.get(
                    f"http://127.0.0.1:{self._bridge_port}/messages",
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 200:
                        messages = await resp.json()
                        for msg_data in messages:
                            event = await self._build_message_event(msg_data)
                            if event:
                                await self.handle_message(event)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                bridge_exit = await self._check_managed_bridge_exit()
                if bridge_exit:
                    logger.error("[%s] %s", self.name, bridge_exit)
                    break
                logger.debug("[%s] Poll error: %s", self.name, exc)
                await asyncio.sleep(5)

            await asyncio.sleep(1)

    async def _build_message_event(self, data: Dict[str, Any]) -> Optional[MessageEvent]:
        """Build a normalized MessageEvent from bridge JSON."""
        try:
            msg_type = MessageType.TEXT
            if data.get("hasMedia"):
                media_type = str(data.get("mediaType", ""))
                if media_type == "image":
                    msg_type = MessageType.PHOTO
                elif media_type == "video":
                    msg_type = MessageType.VIDEO
                elif media_type == "voice":
                    msg_type = MessageType.VOICE
                else:
                    msg_type = MessageType.DOCUMENT

            source = self.build_source(
                chat_id=data.get("chatId", ""),
                chat_name=data.get("chatName"),
                chat_type="dm",
                user_id=data.get("senderId"),
                user_name=data.get("senderName"),
            )

            return MessageEvent(
                text=data.get("body", ""),
                message_type=msg_type,
                source=source,
                raw_message=data,
                message_id=data.get("messageId"),
                media_urls=data.get("mediaUrls", []),
                media_types=data.get("mediaTypes", []),
            )
        except Exception as exc:
            logger.debug("[%s] Error building event: %s", self.name, exc)
            return None
