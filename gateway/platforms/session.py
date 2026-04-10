"""Session Protocol platform adapter.

Session is a decentralized, end-to-end encrypted messenger built on the Oxen
network.  Each user is identified by a Session ID (a hex string starting with
'05' for DMs or '03' for groups).

This adapter is a HYBRID:
- Process management: spawns a Node.js bridge as a child process, routes its
  stdout/stderr to a log file, kills it on disconnect (SIGTERM -> SIGKILL),
  and calls _set_fatal_error on unexpected exits.  (WhatsApp pattern)
- HTTP/SSE transport: uses httpx.AsyncClient for all requests, receives
  inbound messages via a persistent SSE stream with exponential-backoff
  reconnection, and runs a health-monitor asyncio task.  (Signal pattern)

Bridge: scripts/session-bridge/session-bridge.mjs
  Started as a child process via Node.js.
  Exposes a small HTTP API on 127.0.0.1:<bridge_port>:
    GET  /health               -> {"status": "ready"|"starting"|...}
    GET  /session-id           -> {"sessionId": "<hex>"}
    GET  /events               -> SSE stream of inbound message events
    GET  /conversations        -> list of conversation objects
    GET  /conversations/stream -> SSE stream of real-time conversation updates
    GET  /messages/:convId     -> message history for a conversation {?limit=N}
    POST /send                 -> send a message {to, body, attachments?, quote?, expireTimer?}
    POST /send-typing          -> send/stop typing indicator {to, isTyping}
    POST /accept-contact       -> accept an incoming contact request {sessionId}
    POST /download-attachment  -> download an attachment to disk {attachment, destDir?}
    POST /react                -> send a reaction {conversationId, messageDbId, emoji}
    POST /create-group         -> create a group {name, members[]}
    POST /add-group-members    -> add members to a group {groupId, sessionIds[], withHistory?}
    POST /remove-group-members -> remove members from a group {groupId, sessionIds[], alsoRemoveMessages?}
    POST /promote-group-members -> promote members to admin {groupId, memberIds[]}
    POST /leave-group          -> leave a group {groupId}
    POST /block-contact        -> block a contact {sessionId}
    POST /unblock-contact      -> unblock a contact {sessionId}
    POST /set-display-name     -> set bot display name {name} (also applied automatically on connect)
    POST /set-display-image    -> set bot avatar {imagePath} (reads local file → Buffer)

Required env vars / config.extra keys:
    mnemonic        (required) 13-word mnemonic seed for the Session account
    bridge_port     HTTP port for the bridge (default 8095)
    bot_name        Display name used for @mention detection (default "Hermes")
    data_path       Where session data / keys are persisted
    startup_timeout Seconds to wait for the bridge to become ready (default 15)
    log_level       Bridge log verbosity (default "warn")
"""

import asyncio
import json
import logging
import mimetypes
import os
import random
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from gateway.config import Platform, PlatformConfig
from hermes_cli.config import get_hermes_home
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_image_from_url,
    cache_image_from_bytes,
    cache_audio_from_bytes,
    cache_document_from_bytes,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_MESSAGE_LENGTH = 2000  # Session protocol message size limit

SSE_RETRY_DELAY_INITIAL = 2.0   # seconds
SSE_RETRY_DELAY_MAX = 60.0      # seconds
SSE_STALE_TIMEOUT = 60.0        # seconds — reconnect if no data received (2x bridge ping interval)
HEALTH_CHECK_INTERVAL = 30.0    # seconds between health pings


# ---------------------------------------------------------------------------
# Module-level requirement check
# ---------------------------------------------------------------------------

def check_session_requirements() -> bool:
    """Return True if the Session bridge can be launched.

    Checks:
    - SESSION_BOT_ID env var is set (written by setup; mnemonic is optional)
    - node >= 24.12.0 is available in PATH
    - The bridge script exists on disk
    """
    import shutil

    bot_id = os.getenv("SESSION_BOT_ID")
    if not bot_id:
        logger.debug("Session: SESSION_BOT_ID not set")
        return False

    node = shutil.which("node")
    if not node:
        logger.warning("Session: Node.js not found in PATH")
        return False

    try:
        result = subprocess.run(
            [node, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        version = result.stdout.strip().lstrip("v")
        major = int(version.split(".")[0])
        if major < 24:
            logger.warning(
                "Session: Node.js %s too old, need >= 24.12.0", version
            )
            return False
    except Exception as e:
        logger.debug("Session: could not verify Node.js version: %s", e)

    bridge_script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "session-bridge"
        / "session-bridge.mjs"
    )
    if not bridge_script.exists():
        logger.warning(
            "Session: bridge script not found at %s", bridge_script
        )
        return False

    return True


def session_doctor_checks(check_ok, check_fail, check_warn, issues: list, should_fix: bool = False) -> int:
    """Detailed Session health checks for `hermes doctor`.

    Accepts the doctor's reporter callbacks so this module stays independent
    of hermes_cli internals. Only runs when SESSION_BOT_ID is configured.

    Returns the number of issues fixed (for --fix mode).
    """
    fixed_count = 0
    session_bot_id = os.getenv("SESSION_BOT_ID")
    if not session_bot_id:
        return fixed_count

    # Validate SESSION_BOT_ID format (66-char hex starting with 05)
    if len(session_bot_id) == 66 and session_bot_id.startswith("05"):
        check_ok("Session SESSION_BOT_ID", f"({session_bot_id[:8]}...)")
    else:
        check_fail("Session SESSION_BOT_ID format invalid", "(expected 66-char hex starting with 05)")
        issues.append("Session SESSION_BOT_ID appears malformed — re-run: hermes setup")

    # Node.js version >= 24.12.0 (required by session-desktop-library)
    import shutil as _shutil
    node = _shutil.which("node")
    if node:
        try:
            ver_result = subprocess.run(
                [node, "--version"], capture_output=True, text=True, timeout=5
            )
            ver_str = ver_result.stdout.strip().lstrip("v")
            parts = [int(x) for x in ver_str.split(".")[:3]]
            if parts >= [24, 12, 0]:
                check_ok("Session Node.js version", f"(v{ver_str} >= 24.12.0)")
            else:
                check_fail("Session Node.js version too old", f"(v{ver_str} — Session requires >= 24.12.0)")
                issues.append("Session upgrade Node.js to >= 24.12.0 for Session gateway")
        except Exception:
            check_warn("Session Node.js version check failed")
    else:
        check_fail("Session Node.js not found", "(required for Session gateway)")
        issues.append("Session install Node.js >= 24.12.0 for Session gateway")

    # Bridge script exists
    bridge_script = (
        Path(__file__).resolve().parents[2]
        / "scripts" / "session-bridge" / "session-bridge.mjs"
    )
    if bridge_script.exists():
        check_ok("Session bridge script")
    else:
        check_fail("Session bridge script not found", f"(expected at {bridge_script})")
        issues.append("Session bridge script missing — check your installation")

    # Bridge deps installed
    if (bridge_script.parent / "node_modules").exists():
        # Check if any deps need updating
        import shutil as _shutil
        npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
        npm = _shutil.which(npm_cmd)
        if not npm:
            check_warn("npm not found", "(cannot check Session bridge dep updates)")
        else:
            try:
                outdated = subprocess.run(
                    [npm, "outdated", "--json"],
                    cwd=str(bridge_script.parent),
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if outdated.stdout.strip() and outdated.stdout.strip() != "{}":
                    import json as _json
                    pkgs = list(_json.loads(outdated.stdout).keys())
                    if should_fix:
                        check_ok("Session bridge deps updating...")
                        install_result = subprocess.run(
                            [npm, "update"],
                            cwd=str(bridge_script.parent),
                            capture_output=True,
                            text=True,
                            timeout=120,
                        )
                        if install_result.returncode == 0:
                            check_ok("Session bridge deps updated", f"({', '.join(pkgs)})")
                            fixed_count += 1
                            issues.append("Run 'hermes gateway restart'")
                        else:
                            check_fail("Session bridge deps update failed", f"({install_result.stderr.strip()})")
                    else:
                        check_warn("Session bridge deps outdated", f"({', '.join(pkgs)})")
                        issues.append("Session bridge deps outdated. Run 'hermes doctor --fix'")
                else:
                    check_ok("Session bridge deps up to date")
            except Exception:
                check_warn("Session bridge dep update check failed")
    else:
        check_fail("Session bridge deps not installed", "Run 'hermes setup'")
        issues.append("Session bridge dependencies missing. Run 'hermes setup'")

    # Data directory exists
    session_data = os.getenv("SESSION_DATA_PATH") or str(get_hermes_home() / "session-data")
    if Path(session_data).exists():
        check_ok("Session data directory", f"({session_data})")
    else:
        check_warn("Session data directory not found", f"({session_data}). Run 'hermes setup'")

    return fixed_count


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class SessionAdapter(BasePlatformAdapter):
    """Session Protocol adapter.

    Spawns session-bridge.mjs as a managed child process and communicates
    with it over a local HTTP/SSE API.
    """

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.SESSION)

        extra = config.extra or {}
        self.bridge_port: int = int(extra.get("bridge_port", 8095))
        self.bridge_url: str = f"http://127.0.0.1:{self.bridge_port}"
        self.bot_name: str = extra.get("bot_name", "Hermes")
        self.startup_timeout: int = int(extra.get("startup_timeout", 15))

        # Process management
        self._bridge_process: Optional[subprocess.Popen] = None
        self._bridge_log: Optional[Path] = None
        self._bridge_log_fh = None

        # HTTP / SSE
        self._http_client: Optional[httpx.AsyncClient] = None
        self._sse_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None

        # Resolved after connect()
        self._bot_session_id: Optional[str] = None

        logger.info(
            "Session adapter initialized: port=%d bot=%s",
            self.bridge_port,
            self.bot_name,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Spawn the bridge, wait for it to be ready, then start SSE listener."""
        
        # Prevent multiple gateways from using the same Session account.
        # We use SESSION_BOT_ID (the public Session ID) as the lock identity
        # because the mnemonic might not be present in all configurations.
        bot_id = (self.config.extra.get("bot_id") or 
                 os.getenv("SESSION_BOT_ID") or "unknown")
        if not self._acquire_platform_lock(
            scope="session-bot-id",
            identity=bot_id,
            resource_desc="Session account (bot ID)"
        ):
            return False

        # 1. Spawn the bridge process
        try:
            self._bridge_process = self._spawn_bridge()
        except Exception as e:
            logger.error("Session: failed to spawn bridge: %s", e, exc_info=True)
            self._close_bridge_log()
            return False

        # 2. Create HTTP client
        self._http_client = httpx.AsyncClient(timeout=30.0)

        # 3. Poll /health until {"status": "ready"} or timeout
        logger.info(
            "Session: waiting up to %ds for bridge to become ready...",
            self.startup_timeout,
        )
        ready = False
        for _ in range(self.startup_timeout):
            await asyncio.sleep(1)

            # Check if the bridge process died before becoming ready
            if self._bridge_process.poll() is not None:
                logger.error(
                    "Session: bridge process exited during startup (code %d). "
                    "Check log: %s",
                    self._bridge_process.returncode,
                    self._bridge_log,
                )
                await self._http_client.aclose()
                self._http_client = None
                self._close_bridge_log()
                return False

            try:
                resp = await self._http_client.get(
                    f"{self.bridge_url}/health", timeout=2.0
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "ready":
                        ready = True
                        break
            except Exception:
                pass  # Bridge HTTP server not up yet — keep polling

        if not ready:
            logger.error(
                "Session: bridge did not become ready in %ds. Check log: %s",
                self.startup_timeout,
                self._bridge_log,
            )
            await self._http_client.aclose()
            self._http_client = None
            self._close_bridge_log()
            return False

        # 4. Resolve the bot's own Session ID
        try:
            resp = await self._http_client.get(
                f"{self.bridge_url}/session-id", timeout=5.0
            )
            if resp.status_code == 200:
                self._bot_session_id = resp.json().get("sessionId")
                logger.info(
                    "Session: bot session ID = %s",
                    (self._bot_session_id or "")[:16] + "...",
                )
        except Exception as e:
            logger.warning("Session: could not fetch session-id: %s", e)

        # 4b. Set display name if configured
        if self.bot_name and self.bot_name != "Anonymous":
            try:
                await self._http_client.post(
                    f"{self.bridge_url}/set-display-name",
                    json={"name": self.bot_name},
                    timeout=10.0,
                )
                logger.info("Session: set display name to '%s'", self.bot_name)
            except Exception as e:
                logger.warning("Session: could not set display name: %s", e)

        # 5. Start SSE listener background task
        self._sse_task = asyncio.create_task(self._sse_listener())

        # 6. Start health monitor background task
        self._health_task = asyncio.create_task(self._health_monitor())

        # 7. Mark connected
        self._mark_connected()
        logger.info("Session: connected on port %d", self.bridge_port)
        return True

    async def disconnect(self) -> None:
        """Stop background tasks, kill the bridge process, clean up."""
        self._running = False

        # 1. Cancel background tasks
        for task_attr in ("_sse_task", "_health_task"):
            task = getattr(self, task_attr, None)
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                setattr(self, task_attr, None)

        # 2. Kill bridge process group (SIGTERM -> wait 1s -> SIGKILL)
        if self._bridge_process:
            try:
                pid = self._bridge_process.pid
                if sys.platform == "win32":
                    # On Windows, shell=True means node is a child of cmd.exe.
                    # taskkill /T kills the whole process tree so node doesn't
                    # survive as an orphan.
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True,
                    )
                else:
                    # Unix: kill the entire process group
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGTERM)
                    except (ProcessLookupError, PermissionError):
                        self._bridge_process.terminate()
                    await asyncio.sleep(1)
                    if self._bridge_process.poll() is None:
                        try:
                            os.killpg(os.getpgid(pid), signal.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            self._bridge_process.kill()
            except Exception as e:
                logger.warning("Session: error stopping bridge: %s", e)
            self._bridge_process = None

        # 3. Close bridge log file
        self._close_bridge_log()

        # 4. Close HTTP client
        if self._http_client:
            try:
                await self._http_client.aclose()
            except Exception:
                pass
            self._http_client = None

        # Release the scoped lock so another gateway instance can use this mnemonic
        self._release_platform_lock()

        self._mark_disconnected()
        logger.info("Session: disconnected")

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a plain-text message."""
        if not self._running:
            return SendResult(success=False, error="Not connected")

        # Guard against a dead bridge
        if self._bridge_process and self._bridge_process.poll() is not None:
            msg = (
                f"Session bridge process exited unexpectedly "
                f"(code {self._bridge_process.returncode})."
            )
            if not self.has_fatal_error:
                logger.error("Session: %s", msg)
                self._set_fatal_error("session_bridge_exited", msg, retryable=True)
                self._close_bridge_log()
                asyncio.create_task(self._notify_fatal_error())
            return SendResult(success=False, error=self.fatal_error_message or msg)

        try:
            payload: Dict[str, Any] = {"to": chat_id, "body": content}
            if reply_to:
                # Full quote object required by Session library
                quote_data = metadata.get("quote_data", {}) if metadata else {}
                payload["quote"] = {
                    "id": reply_to,
                    "author": quote_data.get("author")
                    or quote_data.get("source")
                    or chat_id,
                    "text": quote_data.get("text") or quote_data.get("body") or "",
                }

            resp = await self._http_client.post(
                f"{self.bridge_url}/send",
                json=payload,
                timeout=30.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                return SendResult(
                    success=True,
                    message_id=data.get("id") or data.get("messageId"),
                    raw_response=data,
                )
            else:
                error_text = resp.text[:200] if resp.text else "No error text"
                logger.warning(
                    "Session: /send failed with status %d: %s",
                    resp.status_code,
                    error_text,
                )
                return SendResult(
                    success=False,
                    error=f"Bridge /send returned {resp.status_code}: {error_text}",
                )
        except Exception as e:
            logger.error("Session: exception while sending: %s", e)
            return SendResult(success=False, error=str(e))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Send a typing indicator."""
        if not self._http_client:
            return
        try:
            await self._http_client.post(
                f"{self.bridge_url}/send-typing",
                json={"to": chat_id, "isTyping": True},
                timeout=5.0,
            )
        except Exception as e:
            logger.debug("Session: send_typing failed: %s", e)

    async def stop_typing(self, chat_id: str) -> None:
        """Stop the typing indicator."""
        if not self._http_client:
            return
        try:
            await self._http_client.post(
                f"{self.bridge_url}/send-typing",
                json={"to": chat_id, "isTyping": False},
                timeout=5.0,
            )
        except Exception as e:
            logger.debug("Session: stop_typing failed: %s", e)

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send an image from a URL by downloading it to the local cache first."""
        if not self._running:
            return SendResult(success=False, error="Not connected")
        try:
            local_path = await cache_image_from_url(image_url)
        except Exception as e:
            logger.warning("Session: failed to download image %s: %s", image_url, e)
            return SendResult(success=False, error=str(e))

        return await self.send_image_file(
            chat_id=chat_id,
            image_path=local_path,
            caption=caption,
            **kwargs,
        )

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a local image file as an attachment."""
        if not self._running:
            return SendResult(success=False, error="Not connected")

        path = Path(image_path)
        content_type, _ = mimetypes.guess_type(str(path))
        content_type = content_type or "image/jpeg"

        return await self._send_with_attachment(
            chat_id=chat_id,
            caption=caption or "",
            attachment={
                "path": str(path),
                "contentType": content_type,
                "fileName": path.name,
            },
        )

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        filename: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a document/file attachment."""
        if not self._running:
            return SendResult(success=False, error="Not connected")

        path = Path(file_path)
        effective_name = filename or path.name

        return await self._send_with_attachment(
            chat_id=chat_id,
            caption=caption or "",
            attachment={
                "path": str(path),
                "contentType": "application/octet-stream",
                "fileName": effective_name,
            },
        )

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send an audio file as a voice message."""
        if not self._running:
            return SendResult(success=False, error="Not connected")

        path = Path(audio_path)
        return await self._send_with_attachment(
            chat_id=chat_id,
            caption=caption or "",
            attachment={
                "path": str(path),
                "contentType": "audio/ogg",
                "fileName": path.name,
            },
        )

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a video file as a Session attachment."""
        if not self._running:
            return SendResult(success=False, error="Not connected")

        path = Path(video_path)
        content_type = mimetypes.guess_type(str(path))[0] or "video/mp4"
        return await self._send_with_attachment(
            chat_id=chat_id,
            caption=caption or "",
            attachment={
                "path": str(path),
                "contentType": content_type,
                "fileName": path.name,
            },
        )

    async def _send_reaction(
        self,
        conversation_id: str,
        message_db_id: str,
        emoji: str,
    ) -> bool:
        """Send an emoji reaction to a message via the bridge /react endpoint."""
        if not self._http_client:
            return False
        try:
            resp = await self._http_client.post(
                f"{self.bridge_url}/react",
                json={
                    "conversationId": conversation_id,
                    "messageDbId": message_db_id,
                    "emoji": emoji,
                },
                timeout=10.0,
            )
            return resp.status_code == 200
        except Exception as e:
            logger.debug("Session: _send_reaction failed: %s", e)
            return False

    async def _send_with_attachment(
        self,
        chat_id: str,
        caption: str,
        attachment: Dict[str, Any],
    ) -> SendResult:
        """POST /send with an attachment descriptor."""
        if not self._running:
            return SendResult(success=False, error="Not connected")
        try:
            payload: Dict[str, Any] = {
                "to": chat_id,
                "body": caption,
                "attachments": [attachment],
            }
            resp = await self._http_client.post(
                f"{self.bridge_url}/send",
                json=payload,
                timeout=120.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                return SendResult(
                    success=True,
                    message_id=data.get("id") or data.get("messageId"),
                    raw_response=data,
                )
            else:
                return SendResult(
                    success=False,
                    error=f"Bridge /send returned {resp.status_code}: {resp.text[:200]}",
                )
        except Exception as e:
            return SendResult(success=False, error=str(e))

    # ------------------------------------------------------------------
    # Chat Info
    # ------------------------------------------------------------------

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about a conversation."""
        chat_type = "group" if chat_id.startswith("03") else "dm"
        if not self._http_client:
            return {"name": chat_id, "type": chat_type}
        try:
            resp = await self._http_client.get(
                f"{self.bridge_url}/conversations", timeout=10.0
            )
            if resp.status_code == 200:
                for convo in resp.json():
                    if convo.get("id") == chat_id:
                        name = convo.get("displayName") or chat_id
                        return {"name": name, "type": chat_type}
        except Exception as e:
            logger.debug("Session: get_chat_info failed: %s", e)
        return {"name": chat_id, "type": chat_type}

    # ------------------------------------------------------------------
    # SSE Listener (inbound messages)
    # ------------------------------------------------------------------

    async def _sse_listener(self) -> None:
        """Persistent SSE connection to /events with exponential-backoff reconnection."""
        url = f"{self.bridge_url}/events"
        backoff = SSE_RETRY_DELAY_INITIAL

        while self._running:
            # Check bridge is still alive before attempting connection
            if self._bridge_process and self._bridge_process.poll() is not None:
                await self._handle_bridge_exit()
                break

            try:
                logger.debug("Session SSE: connecting to %s", url)
                async with self._http_client.stream(
                    "GET",
                    url,
                    headers={"Accept": "text/event-stream"},
                    timeout=None,
                ) as response:
                    backoff = SSE_RETRY_DELAY_INITIAL  # reset on successful connection
                    logger.info("Session SSE: connected")

                    buffer = ""
                    aiter = response.aiter_text().__aiter__()
                    while self._running:
                        try:
                            chunk = await asyncio.wait_for(
                                aiter.__anext__(), timeout=SSE_STALE_TIMEOUT
                            )
                        except asyncio.TimeoutError:
                            logger.warning(
                                "Session SSE: no data in %.0fs, reconnecting",
                                SSE_STALE_TIMEOUT,
                            )
                            break
                        except StopAsyncIteration:
                            break

                        # Check for bridge exit mid-stream
                        if (
                            self._bridge_process
                            and self._bridge_process.poll() is not None
                        ):
                            await self._handle_bridge_exit()
                            return

                        buffer += chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.rstrip("\r")

                            # Skip blank lines and SSE comments
                            if not line or line.startswith(":"):
                                continue

                            if line.startswith("data:"):
                                data_str = line[5:].strip()
                                if not data_str:
                                    continue
                                try:
                                    data = json.loads(data_str)
                                    await self._dispatch_sse_event(data)
                                except json.JSONDecodeError:
                                    logger.debug(
                                        "Session SSE: invalid JSON: %s",
                                        data_str[:120],
                                    )
                                except Exception:
                                    logger.exception(
                                        "Session SSE: error handling event"
                                    )

            except asyncio.CancelledError:
                break
            except httpx.HTTPError as e:
                if self._running:
                    logger.warning(
                        "Session SSE: HTTP error: %s (reconnecting in %.0fs)",
                        e,
                        backoff,
                    )
            except Exception as e:
                if self._running:
                    logger.warning(
                        "Session SSE: error: %s (reconnecting in %.0fs)",
                        e,
                        backoff,
                    )

            if self._running:
                # Check bridge before sleeping so we report the exit quickly
                if (
                    self._bridge_process
                    and self._bridge_process.poll() is not None
                ):
                    await self._handle_bridge_exit()
                    break
                jitter = backoff * 0.2 * random.random()
                await asyncio.sleep(backoff + jitter)
                backoff = min(backoff * 2, SSE_RETRY_DELAY_MAX)

    async def _dispatch_sse_event(self, data: dict) -> None:
        """Route an SSE event by its 'type' field."""
        event_type = data.get("type", "")

        if event_type == "ready":
            logger.info("Session SSE: bridge emitted 'ready'")
            session_id = data.get("sessionId")
            if session_id:
                self._bot_session_id = session_id

        elif event_type == "message":
            await self._handle_message_event(data)

        else:
            logger.debug("Session SSE: unhandled event type '%s'", event_type)

    # ------------------------------------------------------------------
    # Message event processing
    # ------------------------------------------------------------------

    async def _handle_message_event(self, msg_data: dict) -> None:
        """Parse a message event from the bridge and dispatch it."""
        # The bridge sends SSE events wrapped as {type: 'message', data: {...}}
        if isinstance(msg_data, dict) and msg_data.get("type") == "message" and "data" in msg_data:
            msg_data = msg_data["data"]

        # Skip outgoing messages (echo filtering)
        if msg_data.get("isOutgoing"):
            return

        conversation_id = msg_data.get("conversationId", "")
        is_group = conversation_id.startswith("03")

        # Group messages: only process when the bot is @mentioned
        if is_group:
            if not self._should_process_group_message(msg_data):
                return

        # Accept incoming contact requests so the user can then send messages
        if msg_data.get("isIncomingRequest"):
            source_id = msg_data.get("source", "")
            if source_id:
                asyncio.create_task(self._accept_contact_if_allowed(source_id))

        # --- Build MessageType from attachments ---
        text = msg_data.get("body") or ""
        attachments = msg_data.get("attachments") or []

        if attachments:
            first = attachments[0]
            ct = (first.get("contentType") or "").lower()
            if ct.startswith("image/"):
                msg_type = MessageType.PHOTO
            elif ct.startswith("audio/"):
                msg_type = MessageType.VOICE
            elif ct.startswith("video/"):
                msg_type = MessageType.VIDEO
            else:
                msg_type = MessageType.DOCUMENT
        else:
            msg_type = MessageType.TEXT

        # --- Download attachments to cache (awaited before dispatch) ---
        media_urls: List[str] = []
        media_types: List[str] = []
        for att in attachments:
            try:
                cached_path, ct = await self._fetch_attachment_via_bridge(att, conversation_id)
                if cached_path:
                    media_urls.append(cached_path)
                    media_types.append(ct)
            except Exception:
                logger.exception("Session: failed to cache attachment")

        # --- Build source ---
        source = self.build_source(
            chat_id=conversation_id,
            chat_name=msg_data.get("senderDisplayName"),
            chat_type="group" if is_group else "dm",
            user_id=msg_data.get("source"),
            user_name=msg_data.get("senderDisplayName"),
        )

        # --- Reply/quote context ---
        quote_text = None
        if msg_data.get("quote"):
            quote_text = msg_data["quote"].get("text")

        # --- Build and dispatch event ---
        event = MessageEvent(
            text=text,
            message_type=msg_type,
            source=source,
            message_id=msg_data.get("id"),
            reply_to_text=quote_text,
            media_urls=media_urls,
            media_types=media_types,
        )

        await self.handle_message(event)

    def _should_process_group_message(self, msg_data: dict) -> bool:
        """Return True if the bot should respond to this group message.

        A group message is processed only when the bot is @mentioned by its
        Session ID or by display name. The client always substitutes the
        Session ID into the message body when the user picks from the mention
        list, so the ID check is the primary path.
        """
        body = (msg_data.get("body") or "").strip()
        if not body:
            return False

        # Primary: @<sessionId> check. Use the runtime-fetched ID when
        # available, fall back to the configured SESSION_BOT_ID so this works
        # even if the /session-id fetch failed or hasn't completed yet.
        bot_id = self._bot_session_id or self.config.extra.get("bot_id") or os.getenv("SESSION_BOT_ID")
        if bot_id and f"@{bot_id}" in body:
            return True

        # Fallback: case-insensitive @BotName check (typed manually, not via
        # the mention picker).
        if re.search(rf"@{re.escape(self.bot_name)}\b", body, re.IGNORECASE):
            return True

        return False

    async def _fetch_attachment_via_bridge(
        self,
        att: dict,
        conversation_id: str = "",
    ) -> tuple:
        """Ask the bridge to download an attachment and route it into the Hermes cache.

        The Session library's attachment objects are opaque (not plain URLs) and
        must be passed back to the bridge's POST /download-attachment endpoint,
        which calls client.downloadAttachment() and returns the local file path.

        Returns (cached_path, content_type) on success, (None, "") on failure.
        """
        ct = (att.get("contentType") or "application/octet-stream").lower()

        # Ask bridge to write the file to a per-conversation attachments subdir
        subdir = conversation_id if conversation_id else "unknown"
        dest_dir = str(get_hermes_home() / "session-data" / "attachments" / subdir)
        try:
            resp = await self._http_client.post(
                f"{self.bridge_url}/download-attachment",
                json={"attachment": att, "destDir": dest_dir},
                timeout=120.0,
            )
            if resp.status_code != 200:
                logger.warning(
                    "Session: /download-attachment returned %d: %s",
                    resp.status_code,
                    resp.text[:200],
                )
                return None, ""
            bridge_path = resp.json().get("path", "")
            if not bridge_path:
                logger.warning("Session: /download-attachment returned no path")
                return None, ""
        except Exception as e:
            logger.warning("Session: /download-attachment request failed: %s", e)
            return None, ""

        # Read the file the bridge wrote and copy it into the Hermes cache
        try:
            data = Path(bridge_path).read_bytes()
        except Exception as e:
            logger.warning("Session: could not read bridge attachment at %s: %s", bridge_path, e)
            return None, ""

        if ct.startswith("image/"):
            ext = "." + ct.split("/")[-1].split(";")[0].strip()
            cached_path = cache_image_from_bytes(data, ext)
        elif ct.startswith("audio/"):
            ext = "." + ct.split("/")[-1].split(";")[0].strip()
            cached_path = cache_audio_from_bytes(data, ext)
        else:
            filename = att.get("fileName") or Path(bridge_path).name or "attachment"
            cached_path = cache_document_from_bytes(data, filename)

        logger.info(
            "Session: cached attachment (%s, %d bytes) -> %s",
            ct,
            len(data),
            cached_path,
        )
        return cached_path, ct

    async def _accept_contact_if_allowed(self, session_id: str) -> None:
        """Tell the bridge to accept an incoming contact request."""
        try:
            await self._http_client.post(
                f"{self.bridge_url}/accept-contact",
                json={"sessionId": session_id},
                timeout=10.0,
            )
            logger.debug(
                "Session: accepted contact request from %s...", session_id[:16]
            )
        except Exception as e:
            logger.debug(
                "Session: could not accept contact request from %s: %s",
                session_id[:16],
                e,
            )

    # ------------------------------------------------------------------
    # Health Monitor
    # ------------------------------------------------------------------

    async def _health_monitor(self) -> None:
        """Periodically check that the bridge process and HTTP server are alive."""
        while self._running:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            if not self._running:
                break

            # Check bridge process
            if self._bridge_process and self._bridge_process.poll() is not None:
                await self._handle_bridge_exit()
                break

            # Ping /health endpoint
            try:
                resp = await self._http_client.get(
                    f"{self.bridge_url}/health", timeout=5.0
                )
                if resp.status_code != 200:
                    logger.warning(
                        "Session: /health returned %d", resp.status_code
                    )
            except Exception as e:
                logger.warning("Session: /health unreachable: %s", e)

    async def _handle_bridge_exit(self) -> None:
        """Called when the bridge process exits unexpectedly."""
        returncode = (
            self._bridge_process.returncode
            if self._bridge_process
            else "?"
        )
        msg = (
            f"Session bridge process exited unexpectedly (code {returncode}). "
            f"Check log: {self._bridge_log}"
        )
        if not self.has_fatal_error:
            logger.error("Session: %s", msg)
            self._set_fatal_error(
                "session_bridge_exited", msg, retryable=True
            )
            self._close_bridge_log()
            await self._notify_fatal_error()

    # ------------------------------------------------------------------
    # Bridge process management
    # ------------------------------------------------------------------

    def _spawn_bridge(self) -> subprocess.Popen:
        """Launch session-bridge.mjs as a child process.

        stdout/stderr are redirected to a log file (not PIPE — avoids
        deadlock when the OS pipe buffer fills up).
        """
        bridge_script = (
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "session-bridge"
            / "session-bridge.mjs"
        )

        # Session data goes in its own directory, but logs go to central logs/
        data_path = Path(
            self.config.extra.get(
                "data_path", str(get_hermes_home() / "session-data")
            )
        )
        data_path.mkdir(parents=True, exist_ok=True)

        # Bridge logs go to central logs directory for consistency
        logs_dir = get_hermes_home() / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        self._bridge_log = logs_dir / "session-bridge.log"
        bridge_log_fh = open(self._bridge_log, "a")
        self._bridge_log_fh = bridge_log_fh

        env = {
            **os.environ,
            "SESSION_DATA_PATH": str(data_path),
            "SESSION_BRIDGE_PORT": str(self.bridge_port),
            "SESSION_BOT_NAME": self.bot_name,
            "SESSION_LOG_LEVEL": self.config.extra.get("log_level", "warn"),
        }

        logger.info(
            "Session: spawning bridge: node %s (port %d, log %s)",
            bridge_script,
            self.bridge_port,
            self._bridge_log,
        )

        if sys.platform == "win32":
            process = subprocess.Popen(
                ["node", str(bridge_script)],
                env=env,
                stdout=bridge_log_fh,
                stderr=bridge_log_fh,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                shell=True,
            )
        else:
            process = subprocess.Popen(
                ["node", str(bridge_script)],
                env=env,
                stdout=bridge_log_fh,
                stderr=bridge_log_fh,
                preexec_fn=os.setsid,
            )
        self._bridge_process = process
        return process

    def _close_bridge_log(self) -> None:
        """Close the bridge log file handle if open."""
        if self._bridge_log_fh:
            try:
                self._bridge_log_fh.close()
            except Exception:
                pass
            self._bridge_log_fh = None
