"""
Kasia platform adapter.

This adapter launches a local Node bridge that:
- derives a dedicated Hermes Kasia identity from a configured seed phrase
- polls a configured Kasia indexer for inbound handshakes and messages
- submits outbound transactions through a configured Kaspa wRPC node
"""

import asyncio
import inspect
import json
import logging
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional
from urllib.parse import quote

_IS_WINDOWS = platform.system() == "Windows"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gateway.config import Platform, PlatformConfig
from gateway.kasia_config import (
    DEFAULT_KASIA_BRIDGE_PORT,
    DEFAULT_KASIA_SEND_WAIT_MS,
    is_kasia_address_authorized,
    load_kasia_settings,
)
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource
from hermes_cli.config import get_hermes_home

logger = logging.getLogger(__name__)

_AuthorizationHandler = Callable[[SessionSource], bool | Awaitable[bool]]


def _is_local_port_in_use(port: int) -> bool:
    """Return True when a local TCP listener already owns the given port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            return sock.connect_ex(("127.0.0.1", int(port))) == 0
    except OSError:
        return False


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


def check_kasia_requirements(config: Optional[PlatformConfig] = None) -> bool:
    """Check whether Kasia can be launched in the current environment."""
    kasia_settings = load_kasia_settings(extra=config.extra if config else None)
    if not kasia_settings.has_required_connection_fields:
        return False

    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


class KasiaAdapter(BasePlatformAdapter):
    """Bridge-backed Kasia adapter."""

    _DEFAULT_BRIDGE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "kasia-bridge"
    _DEFAULT_SEND_WAIT_MS = DEFAULT_KASIA_SEND_WAIT_MS
    _KASIA_ADDRESS_PREFIXES = ("kaspa:", "kaspatest:", "kaspasim:")
    _KNS_TARGET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]*\.kas$")
    _BARE_ADDRESS_RE = re.compile(r"^[qp][a-z0-9]{5,}$")

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.KASIA)
        self._kasia_settings = load_kasia_settings(extra=config.extra)
        self._bridge_port = self._kasia_settings.bridge_port or DEFAULT_KASIA_BRIDGE_PORT
        self._bridge_script = Path(
            config.extra.get("bridge_script", self._DEFAULT_BRIDGE_DIR / "bridge.js")
        )
        self._state_dir = Path(
            config.extra.get("state_dir", get_hermes_home() / "kasia")
        )
        self._send_wait_ms = self._kasia_settings.send_wait_ms or self._DEFAULT_SEND_WAIT_MS
        self._bridge_process: Optional[subprocess.Popen] = None
        self._bridge_log: Optional[Path] = None
        self._bridge_log_fh = None
        self._poll_task: Optional[asyncio.Task] = None
        self._pending_handshake_recovery_task: Optional[asyncio.Task] = None
        self._configured_handshake_bootstrap_task: Optional[asyncio.Task] = None
        self._authorization_handler: Optional[_AuthorizationHandler] = None

    @staticmethod
    def _format_sompi_balance(value: Any) -> Optional[str]:
        """Render a sompi integer value as a KAS string with 8 decimals."""
        try:
            sompi = int(str(value).strip())
        except (TypeError, ValueError):
            return None
        sign = "-" if sompi < 0 else ""
        whole, fractional = divmod(abs(sompi), 100_000_000)
        return f"{sign}{whole}.{fractional:08d}"

    def _unauthorized_dm_behavior(self) -> str:
        """Return how first-contact unauthorized DMs should be handled."""
        return str(
            self.config.extra.get("unauthorized_dm_behavior", "pair")
        ).strip().lower() or "pair"

    def _default_address_prefix(self) -> str:
        """Return the Kaspa address prefix for the configured network."""
        normalized_network = str(self._kasia_settings.network or "").strip().lower()
        if normalized_network.startswith("mainnet"):
            return "kaspa:"
        if normalized_network.startswith("test") or normalized_network.startswith("tn"):
            return "kaspatest:"
        return "kaspasim:"

    def _normalize_bootstrap_target(
        self,
        target: Optional[str],
        *,
        trust_bare: bool,
    ) -> Optional[str]:
        """Normalize a configured Kasia DM target for handshake bootstrap."""
        normalized_target = str(target or "").strip()
        if not normalized_target:
            return None
        lowered_target = normalized_target.lower()
        if lowered_target.startswith("broadcast:") or lowered_target.startswith("#"):
            return None
        if lowered_target.startswith(self._KASIA_ADDRESS_PREFIXES):
            return lowered_target
        if self._KNS_TARGET_RE.match(lowered_target):
            return lowered_target
        if trust_bare or self._BARE_ADDRESS_RE.match(lowered_target):
            return f"{self._default_address_prefix()}{lowered_target}"
        return None

    def _configured_handshake_bootstrap_targets(self) -> List[tuple[str, bool]]:
        """Return explicit DM targets that should receive a startup handshake."""
        configured_targets: List[tuple[str, bool]] = []
        seen_targets: set[str] = set()

        def _add_target(raw_target: Optional[str], *, trust_bare: bool, allow_without_auth: bool) -> None:
            normalized_target = self._normalize_bootstrap_target(
                raw_target,
                trust_bare=trust_bare,
            )
            if not normalized_target or normalized_target in seen_targets:
                return
            seen_targets.add(normalized_target)
            configured_targets.append((normalized_target, allow_without_auth))

        home_channel = getattr(self.config, "home_channel", None)
        if home_channel and getattr(home_channel, "chat_id", None):
            _add_target(
                home_channel.chat_id,
                trust_bare=True,
                allow_without_auth=True,
            )

        allowed_users = str(os.getenv("KASIA_ALLOWED_USERS", "") or "").split(",")
        for raw_target in allowed_users:
            _add_target(
                raw_target,
                trust_bare=True,
                allow_without_auth=False,
            )

        global_allowed_users = str(os.getenv("GATEWAY_ALLOWED_USERS", "") or "").split(",")
        for raw_target in global_allowed_users:
            _add_target(
                raw_target,
                trust_bare=False,
                allow_without_auth=False,
            )

        try:
            from gateway.pairing import PairingStore

            for approved_contact in PairingStore().list_approved(Platform.KASIA.value):
                _add_target(
                    approved_contact.get("canonical_address") or approved_contact.get("user_id"),
                    trust_bare=True,
                    allow_without_auth=False,
                )
        except Exception:
            pass

        return configured_targets

    async def _bootstrap_configured_handshakes(
        self,
        health: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initiate configured Kasia DM conversations after the bridge comes online."""
        funding_state = str((health or {}).get("walletFundingState") or "").strip().lower()
        if funding_state in {"low", "unfunded"}:
            return

        configured_targets = self._configured_handshake_bootstrap_targets()
        if not configured_targets:
            return

        for chat_id, allow_without_auth in configured_targets:
            if not allow_without_auth and not await self._is_address_authorized_async(
                chat_id,
                user_name=chat_id,
            ):
                continue
            try:
                result = await self._request_json(
                    "POST",
                    "/handshakes/initiate",
                    payload={"chatId": chat_id, "retry": False},
                    total=30,
                )
                logger.info(
                    "[%s] Bootstrapped Kasia handshake for %s (%s)",
                    self.name,
                    chat_id,
                    result.get("status") or "sent",
                )
            except Exception as error:
                if self._looks_like_insufficient_funds_error(error):
                    await self._log_wallet_funding_warning(
                        context=f"bootstrapping configured Kasia handshake for {chat_id}",
                        error=error,
                    )
                logger.warning(
                    "[%s] Failed to bootstrap Kasia handshake for %s: %s",
                    self.name,
                    chat_id,
                    error,
                )

    def _start_configured_handshake_bootstrap(
        self,
        health: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Kick off handshake bootstrap for configured Kasia DM targets."""

        async def _run_bootstrap() -> None:
            try:
                await self._bootstrap_configured_handshakes(health)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.warning("[%s] Configured Kasia handshake bootstrap failed: %s", self.name, error)
            finally:
                if self._configured_handshake_bootstrap_task is asyncio.current_task():
                    self._configured_handshake_bootstrap_task = None

        self._configured_handshake_bootstrap_task = asyncio.create_task(_run_bootstrap())

    def _looks_like_insufficient_funds_error(self, error: Exception) -> bool:
        """Return True when a Kasia bridge error points to wallet funding issues."""
        text = str(error or "").lower()
        return (
            "insufficient funds" in text
            or "not have enough mature balance" in text
            or "no transaction was produced" in text
        )

    def _build_wallet_funding_warning(self, health: Optional[Dict[str, Any]]) -> Optional[str]:
        """Build a human-friendly warning from Kasia health funding fields."""
        data = health or {}
        funding_state = str(data.get("walletFundingState", "")).strip().lower()
        if funding_state not in {"low", "unfunded"}:
            return None

        wallet_address = str(data.get("walletAddress") or "").strip() or "<unknown>"
        on_chain = self._format_sompi_balance(data.get("walletBalanceSompi"))
        spendable = self._format_sompi_balance(data.get("availableMatureBalanceSompi"))
        recommended = self._format_sompi_balance(data.get("recommendedMinBalanceSompi"))
        balance_parts = []
        if on_chain is not None:
            balance_parts.append(f"{on_chain} KAS on-chain")
        if spendable is not None:
            balance_parts.append(f"{spendable} KAS spendable")
        if recommended is not None:
            balance_parts.append(f"recommended >= {recommended} KAS")
        balance_text = ", ".join(balance_parts) if balance_parts else "balance unavailable"

        if funding_state == "unfunded":
            return (
                f"Kasia wallet {wallet_address} has no usable funds ({balance_text}). "
                "Pairing replies and outbound DMs will fail until the wallet is topped up."
            )
        return (
            f"Kasia wallet {wallet_address} is low on funds ({balance_text}). "
            "Pairing replies and outbound DMs may fail until the wallet is topped up."
        )

    def set_authorization_handler(
        self,
        handler: _AuthorizationHandler,
    ) -> None:
        """Set a central authorization callback owned by the gateway runner."""
        self._authorization_handler = handler

    async def _log_wallet_funding_warning(
        self,
        *,
        context: str,
        error: Optional[Exception] = None,
        health: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a loud Kasia wallet funding warning when balances are too low."""
        if health is None:
            try:
                health = await self._request_json("GET", "/health", total=5)
            except Exception:
                health = None

        warning = self._build_wallet_funding_warning(health)
        if not warning and error is None:
            return

        if warning:
            if error is None:
                logger.warning("[%s] %s", self.name, warning)
            else:
                logger.error(
                    "[%s] %s Context: %s. Original error: %s",
                    self.name,
                    warning,
                    context,
                    error,
                )
            return

        if error is not None:
            logger.error(
                "[%s] Kasia wallet funding appears insufficient during %s: %s",
                self.name,
                context,
                error,
            )

    def _build_handshake_event(self, data: Dict[str, Any]) -> MessageEvent:
        """Normalize a handshake request into a DM event for the gateway."""
        timestamp_ms = data.get("timestampMs")
        timestamp = (
            datetime.fromtimestamp(float(timestamp_ms) / 1000.0)
            if timestamp_ms
            else datetime.now()
        )
        chat_id = data.get("chatId") or data.get("senderId") or ""
        sender_name = data.get("senderName") or chat_id
        return MessageEvent(
            text="",
            message_type=MessageType.TEXT,
            source=self.build_source(
                chat_id=chat_id,
                chat_name=sender_name,
                chat_type="dm",
                user_id=data.get("senderId") or chat_id,
                user_name=sender_name,
            ),
            raw_message=data,
            message_id=data.get("messageId"),
            timestamp=timestamp,
        )

    def _load_pending_handshake_events(self) -> List[Dict[str, Any]]:
        """Load persisted Kasia handshake requests that still need a response."""
        state_path = self._state_dir / "state.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except Exception as error:
            logger.warning("[%s] Failed to read Kasia state from %s: %s", self.name, state_path, error)
            return []

        conversations = state.get("conversations")
        if not isinstance(conversations, dict):
            return []

        recovered: List[tuple[int, Dict[str, Any]]] = []
        for peer_address, conversation in conversations.items():
            if not isinstance(conversation, dict):
                continue

            pending_handshake = conversation.get("pending_handshake")
            if not isinstance(pending_handshake, dict):
                continue

            chat_id = str(conversation.get("peer_address") or peer_address or "").strip()
            if not chat_id:
                continue

            try:
                block_time = int(pending_handshake.get("block_time") or 0)
            except (TypeError, ValueError):
                block_time = 0

            sender_name = str(
                conversation.get("display_name")
                or conversation.get("nickname")
                or conversation.get("kns_name")
                or chat_id
            ).strip() or chat_id

            recovered.append(
                (
                    block_time,
                    {
                        "eventType": "handshake_request",
                        "messageId": pending_handshake.get("tx_id"),
                        "chatId": chat_id,
                        "senderId": chat_id,
                        "senderName": sender_name,
                        "body": "Handshake request",
                        "timestampMs": block_time or None,
                        "raw": {
                            "recovered": True,
                            "pendingHandshake": pending_handshake,
                        },
                    },
                )
            )

        recovered.sort(key=lambda item: item[0])
        return [event for _, event in recovered]

    async def _recover_pending_handshakes(self, health: Optional[Dict[str, Any]] = None) -> None:
        """Retry persisted handshake requests after the bridge comes back up."""
        funding_state = str((health or {}).get("walletFundingState") or "").strip().lower()
        if funding_state in {"low", "unfunded"}:
            return

        pending_events = self._load_pending_handshake_events()
        if not pending_events:
            return

        logger.info(
            "[%s] Replaying %s pending Kasia handshake(s) from %s",
            self.name,
            len(pending_events),
            self._state_dir / "state.json",
        )
        for event in pending_events:
            try:
                await self._handle_bridge_event(event)
            except Exception as error:
                logger.warning(
                    "[%s] Failed to replay pending Kasia handshake for %s: %s",
                    self.name,
                    event.get("chatId"),
                    error,
                )

    def _start_pending_handshake_recovery(self, health: Optional[Dict[str, Any]] = None) -> None:
        """Replay pending handshakes after connect() returns to the gateway runner."""

        async def _run_recovery() -> None:
            try:
                await self._recover_pending_handshakes(health)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.warning("[%s] Pending Kasia handshake recovery failed: %s", self.name, error)
            finally:
                if self._pending_handshake_recovery_task is asyncio.current_task():
                    self._pending_handshake_recovery_task = None

        self._pending_handshake_recovery_task = asyncio.create_task(_run_recovery())

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
            except (OSError, subprocess.SubprocessError) as error:
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

            # Kill any orphaned bridge from a previous gateway run.
            _kill_port_process(self._bridge_port)
            await asyncio.sleep(1)

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
            env.update(self._kasia_settings.bridge_env())

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
                        await self._log_wallet_funding_warning(
                            context="gateway startup",
                            health=health,
                        )
                        self._mark_connected()
                        self._poll_task = asyncio.create_task(self._poll_messages())
                        self._start_pending_handshake_recovery(health)
                        self._start_configured_handshake_bootstrap(health)
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
        bootstrap_task = self._configured_handshake_bootstrap_task
        self._configured_handshake_bootstrap_task = None
        if bootstrap_task:
            cancel = getattr(bootstrap_task, "cancel", None)
            if callable(cancel):
                cancel()
            if isinstance(bootstrap_task, asyncio.Task):
                try:
                    await bootstrap_task
                except asyncio.CancelledError:
                    pass

        recovery_task = self._pending_handshake_recovery_task
        self._pending_handshake_recovery_task = None
        if recovery_task:
            cancel = getattr(recovery_task, "cancel", None)
            if callable(cancel):
                cancel()
            if isinstance(recovery_task, asyncio.Task):
                try:
                    await recovery_task
                except asyncio.CancelledError:
                    pass

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

        _kill_port_process(self._bridge_port)
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
            endpoint = "/send"
            payload: Dict[str, Any] = {
                "chatId": chat_id,
                "message": str(content or "").strip(),
                "waitMs": self._send_wait_ms,
            }
            if str(chat_id).startswith("broadcast:"):
                endpoint = "/broadcasts/send"
                payload = {
                    "channelName": str(chat_id).split(":", 1)[1],
                    "message": str(content or "").strip(),
                    "waitMs": self._send_wait_ms,
                }
            data = await self._request_json(
                "POST",
                endpoint,
                payload=payload,
                total=max(10, int(self._send_wait_ms / 1000) + 10),
            )
            if data.get("status") in {"failed", "rejected"}:
                return SendResult(
                    success=False,
                    error=data.get("error") or "Kasia send failed",
                    raw_response=data,
                )
            if data.get("status") in {"submitted", "waiting_for_indexer"}:
                logger.info(
                    "[%s] Kasia send for %s is %s: %s",
                    self.name,
                    chat_id,
                    data.get("status"),
                    data.get("statusMessage") or "Waiting for public indexer visibility.",
                )
            return SendResult(
                success=True,
                message_id=data.get("jobId") or data.get("txId") or data.get("messageId"),
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

    async def initiate_handshake(
        self,
        chat_id: str,
        display_name: Optional[str] = None,
        retry: bool = False,
    ) -> Dict[str, Any]:
        """Explicitly start a Kasia conversation with a new peer."""
        if not self._running:
            raise RuntimeError("Kasia bridge is not connected")
        if not await self._is_address_authorized_async(chat_id, user_name=display_name):
            raise RuntimeError(f"Kasia initiation is not authorized for {chat_id}")
        return await self._request_json(
            "POST",
            "/handshakes/initiate",
            payload={
                "chatId": chat_id,
                "displayName": display_name,
                "retry": retry,
            },
            total=30,
        )

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
            sender_name = data.get("senderName")
            if await self._is_address_authorized_async(chat_id, user_name=sender_name):
                try:
                    await self._request_json(
                        "POST",
                        "/handshakes/respond",
                        payload={"chatId": chat_id},
                        total=30,
                    )
                    logger.info("[%s] Responded to Kasia handshake from %s", self.name, chat_id)
                except Exception as error:
                    if self._looks_like_insufficient_funds_error(error):
                        await self._log_wallet_funding_warning(
                            context=f"responding to authorized Kasia handshake from {chat_id}",
                            error=error,
                        )
                    logger.warning(
                        "[%s] Failed to respond to Kasia handshake from %s: %s",
                        self.name,
                        chat_id,
                        error,
                    )
            elif self._unauthorized_dm_behavior() == "pair":
                logger.info(
                    "[%s] Queued unauthorized Kasia handshake from %s for approval",
                    self.name,
                    chat_id,
                )
                await self.handle_message(self._build_handshake_event(data))
            else:
                logger.info(
                    "[%s] Ignoring unauthorized Kasia handshake from %s",
                    self.name,
                    chat_id,
                )
            return

        if event_type == "broadcast":
            event = self._build_message_event(data)
            if event:
                await self.handle_message(event)
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
            chat_name=(
                f"#{data.get('channelName')}"
                if data.get("eventType") == "broadcast" and data.get("channelName")
                else data.get("senderName")
            ),
            chat_type="channel" if data.get("eventType") == "broadcast" else "dm",
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

    def _authorization_source(
        self,
        address: Optional[str],
        *,
        user_name: Optional[str] = None,
    ) -> Optional[SessionSource]:
        canonical_address = str(address or "").strip()
        if not canonical_address:
            return None
        resolved_name = str(user_name or "").strip() or canonical_address
        return SessionSource(
            platform=Platform.KASIA,
            chat_id=canonical_address,
            chat_name=resolved_name,
            chat_type="dm",
            user_id=canonical_address,
            user_name=resolved_name,
        )

    def _is_address_authorized(
        self,
        address: Optional[str],
        *,
        user_name: Optional[str] = None,
    ) -> bool:
        """Apply the gateway's central authorization rules for Kasia peers."""
        if self._authorization_handler:
            source = self._authorization_source(address, user_name=user_name)
            if source is not None:
                try:
                    result = self._authorization_handler(source)
                    if inspect.isawaitable(result):
                        close = getattr(result, "close", None)
                        if callable(close):
                            close()
                        logger.warning(
                            "[%s] Async Kasia auth callback was used in sync mode for %s",
                            self.name,
                            address,
                        )
                    else:
                        return bool(result)
                except Exception as error:
                    logger.warning("[%s] Kasia auth callback failed for %s: %s", self.name, address, error)
        return is_kasia_address_authorized(address, display_name=user_name)

    async def _is_address_authorized_async(
        self,
        address: Optional[str],
        *,
        user_name: Optional[str] = None,
    ) -> bool:
        """Apply Kasia auth rules without blocking the asyncio loop."""
        if self._authorization_handler:
            source = self._authorization_source(address, user_name=user_name)
            if source is not None:
                try:
                    result = self._authorization_handler(source)
                    if inspect.isawaitable(result):
                        return bool(await result)
                    return bool(result)
                except Exception as error:
                    logger.warning("[%s] Kasia auth callback failed for %s: %s", self.name, address, error)
        return self._is_address_authorized(address, user_name=user_name)

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
            except OSError:
                pass
            self._bridge_log_fh = None
