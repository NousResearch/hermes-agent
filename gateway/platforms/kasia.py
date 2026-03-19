"""
Kasia platform adapter.

This adapter launches a local Node bridge that:
- derives a dedicated Hermes Kasia identity from a configured seed phrase
- polls a configured Kasia indexer for inbound handshakes and messages
- submits outbound transactions through a configured Kaspa wRPC node
"""

import asyncio
import json
import logging
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote

_IS_WINDOWS = platform.system() == "Windows"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from hermes_cli.config import get_hermes_home

logger = logging.getLogger(__name__)


def _is_local_port_in_use(port: int) -> bool:
    """Return True when a local TCP listener already owns the given port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            return sock.connect_ex(("127.0.0.1", int(port))) == 0
    except Exception:
        return False


def check_kasia_requirements(config: Optional[PlatformConfig] = None) -> bool:
    """Check whether Kasia can be launched in the current environment."""
    extra = config.extra if config else {}
    seed_phrase = extra.get("seed_phrase") or os.getenv("KASIA_SEED_PHRASE", "")
    indexer_url = extra.get("indexer_url") or os.getenv("KASIA_INDEXER_URL", "")
    node_url = extra.get("node_wborsh_url") or os.getenv("KASIA_NODE_WBORSH_URL", "")
    if not all([seed_phrase, indexer_url, node_url]):
        return False

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


class KasiaAdapter(BasePlatformAdapter):
    """Bridge-backed Kasia adapter."""

    _DEFAULT_BRIDGE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "kasia-bridge"

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.KASIA)
        self._bridge_port = int(config.extra.get("bridge_port", 3010))
        self._bridge_script = Path(
            config.extra.get("bridge_script", self._DEFAULT_BRIDGE_DIR / "bridge.js")
        )
        self._state_dir = Path(
            config.extra.get("state_dir", get_hermes_home() / "kasia")
        )
        self._seed_phrase = config.extra.get("seed_phrase", "")
        self._indexer_url = config.extra.get("indexer_url", "")
        self._node_url = config.extra.get("node_wborsh_url", "")
        self._network = config.extra.get("network", "mainnet") or "mainnet"
        self._bridge_process: Optional[subprocess.Popen] = None
        self._bridge_log: Optional[Path] = None
        self._bridge_log_fh = None
        self._poll_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """Launch the Kasia bridge process and wait for its health endpoint."""
        if not check_kasia_requirements(self.config):
            logger.warning(
                "[%s] Missing Node.js or Kasia config (seed phrase/indexer/node URL).",
                self.name,
            )
            return False

        if not self._bridge_script.exists():
            logger.warning(
                "[%s] Bridge script not found: %s",
                self.name,
                self._bridge_script,
            )
            return False

        bridge_dir = self._bridge_script.parent
        if not (bridge_dir / "node_modules").exists():
            print(f"[{self.name}] Installing Kasia bridge dependencies...")
            try:
                result = subprocess.run(
                    ["npm", "install", "--silent"],
                    cwd=str(bridge_dir),
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            except Exception as error:
                logger.error("[%s] npm install failed: %s", self.name, error)
                return False
            if result.returncode != 0:
                logger.error(
                    "[%s] npm install failed: %s",
                    self.name,
                    result.stderr.strip() or result.stdout.strip(),
                )
                return False

        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            if _is_local_port_in_use(self._bridge_port):
                logger.error(
                    "[%s] Refusing to start Kasia bridge on port %s because it is already in use.",
                    self.name,
                    self._bridge_port,
                )
                return False

            self._bridge_log = self._state_dir / "bridge.log"
            self._bridge_log_fh = open(self._bridge_log, "a", encoding="utf-8")

            env = os.environ.copy()
            env["KASIA_SEED_PHRASE"] = self._seed_phrase
            env["KASIA_INDEXER_URL"] = self._indexer_url
            env["KASIA_NODE_WBORSH_URL"] = self._node_url
            env["KASIA_NETWORK"] = self._network

            self._bridge_process = subprocess.Popen(
                [
                    "node",
                    str(self._bridge_script),
                    "--port",
                    str(self._bridge_port),
                    "--state-dir",
                    str(self._state_dir),
                ],
                stdout=self._bridge_log_fh,
                stderr=self._bridge_log_fh,
                preexec_fn=None if _IS_WINDOWS else os.setsid,
                env=env,
            )

            for _attempt in range(20):
                await asyncio.sleep(1)
                if self._bridge_process.poll() is not None:
                    logger.error(
                        "[%s] Bridge exited with code %s. Check %s",
                        self.name,
                        self._bridge_process.returncode,
                        self._bridge_log,
                    )
                    self._close_bridge_log()
                    return False

                try:
                    health = await self._request_json("GET", "/health", total=5)
                    if health.get("status") in {"connected", "starting"}:
                        self._mark_connected()
                        self._poll_task = asyncio.create_task(self._poll_messages())
                        print(f"[{self.name}] Bridge started on port {self._bridge_port}")
                        return True
                except Exception:
                    continue

            logger.error("[%s] Bridge did not become healthy in time", self.name)
            self._close_bridge_log()
            return False
        except Exception as error:
            logger.error("[%s] Failed to start bridge: %s", self.name, error, exc_info=True)
            self._close_bridge_log()
            return False

    async def disconnect(self) -> None:
        """Stop the Kasia bridge and polling task."""
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        if self._bridge_process:
            try:
                import signal

                try:
                    if _IS_WINDOWS:
                        self._bridge_process.terminate()
                    else:
                        os.killpg(os.getpgid(self._bridge_process.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    self._bridge_process.terminate()
                await asyncio.sleep(1)
                if self._bridge_process.poll() is None:
                    try:
                        if _IS_WINDOWS:
                            self._bridge_process.kill()
                        else:
                            os.killpg(os.getpgid(self._bridge_process.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        self._bridge_process.kill()
            except Exception as error:
                logger.warning("[%s] Error stopping bridge: %s", self.name, error)

        self._bridge_process = None
        self._close_bridge_log()
        self._mark_disconnected()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a Kasia text message through the bridge."""
        if not self._running:
            return SendResult(success=False, error="Not connected")
        try:
            data = await self._request_json(
                "POST",
                "/send",
                payload={"chatId": chat_id, "message": str(content or "").strip()},
                total=30,
            )
            return SendResult(
                success=True,
                message_id=data.get("txId") or data.get("messageId"),
                raw_response=data,
            )
        except Exception as error:
            return SendResult(success=False, error=str(error))

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Fetch chat information from the bridge state."""
        if not self._running:
            return {"name": chat_id, "type": "dm"}
        try:
            data = await self._request_json(
                "GET",
                f"/chat/{quote(str(chat_id), safe='')}",
                total=10,
            )
            return {
                "name": data.get("name", chat_id),
                "type": data.get("type", "dm"),
                "chat_id": data.get("chat_id", chat_id),
            }
        except Exception as error:
            logger.debug("[%s] Could not fetch chat info for %s: %s", self.name, chat_id, error)
            return {"name": chat_id, "type": "dm"}

    async def _poll_messages(self) -> None:
        """Poll the bridge for queued Kasia events."""
        while self._running:
            try:
                messages = await self._request_json("GET", "/messages", total=30)
                for message in messages:
                    await self._handle_bridge_event(message)
            except asyncio.CancelledError:
                break
            except Exception as error:
                logger.warning("[%s] Poll error: %s", self.name, error)
                await asyncio.sleep(5)

            await asyncio.sleep(1)

    async def _handle_bridge_event(self, data: Dict[str, Any]) -> None:
        event_type = data.get("eventType")
        if event_type == "handshake_request":
            chat_id = data.get("chatId") or data.get("senderId")
            if self._is_address_authorized(chat_id):
                try:
                    await self._request_json(
                        "POST",
                        "/handshakes/respond",
                        payload={"chatId": chat_id},
                        total=30,
                    )
                    logger.info("[%s] Responded to Kasia handshake from %s", self.name, chat_id)
                except Exception as error:
                    logger.warning(
                        "[%s] Failed to respond to Kasia handshake from %s: %s",
                        self.name,
                        chat_id,
                        error,
                    )
            else:
                logger.info(
                    "[%s] Ignoring unauthorized Kasia handshake from %s",
                    self.name,
                    chat_id,
                )
            return

        if event_type != "message":
            logger.debug("[%s] Ignoring unsupported bridge event: %s", self.name, event_type)
            return

        event = self._build_message_event(data)
        if event:
            await self.handle_message(event)

    def _build_message_event(self, data: Dict[str, Any]) -> Optional[MessageEvent]:
        """Normalize a bridge payload into a Hermes MessageEvent."""
        body = data.get("body")
        if not isinstance(body, str) or not body.strip():
            return None

        source = self.build_source(
            chat_id=data.get("chatId", ""),
            chat_name=data.get("senderName"),
            chat_type="dm",
            user_id=data.get("senderId"),
            user_name=data.get("senderName"),
        )
        timestamp_ms = data.get("timestampMs")
        timestamp = (
            datetime.fromtimestamp(float(timestamp_ms) / 1000.0)
            if timestamp_ms
            else datetime.now()
        )

        return MessageEvent(
            text=body,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=data,
            message_id=data.get("messageId"),
            timestamp=timestamp,
        )

    def _is_address_authorized(self, address: Optional[str]) -> bool:
        """Apply Kasia's address allowlist / allow-all rules for handshake responses."""
        normalized = str(address or "").strip().lower()
        if not normalized:
            return False

        if os.getenv("KASIA_ALLOW_ALL_USERS", "").lower() in ("true", "1", "yes"):
            return True

        platform_allowlist = os.getenv("KASIA_ALLOWED_USERS", "").strip()
        global_allowlist = os.getenv("GATEWAY_ALLOWED_USERS", "").strip()
        if not platform_allowlist and not global_allowlist:
            return os.getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in ("true", "1", "yes")

        allowed_ids = {
            item.strip().lower()
            for value in (platform_allowlist, global_allowlist)
            for item in value.split(",")
            if item.strip()
        }
        check_ids = {normalized}
        if normalized.startswith(("kaspa:", "kaspatest:", "kaspasim:")):
            check_ids.add(normalized.split(":", 1)[1])
        return bool(check_ids & allowed_ids)

    async def _request_json(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        total: int = 10,
    ) -> Any:
        try:
            import aiohttp
        except ImportError as error:
            raise RuntimeError("aiohttp not installed. Run: pip install aiohttp") from error

        url = f"http://127.0.0.1:{self._bridge_port}{path}"
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method,
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=total),
            ) as response:
                body = await response.text()
                if response.status != 200:
                    raise RuntimeError(
                        f"Kasia bridge error ({response.status}) on {path}: {body}"
                    )
                if not body.strip():
                    return {}
                return json.loads(body)

    def _close_bridge_log(self) -> None:
        if self._bridge_log_fh:
            try:
                self._bridge_log_fh.close()
            except Exception:
                pass
            self._bridge_log_fh = None
