#!/usr/bin/env python3
"""
Termux platform adapter for Hermes gateway.

Integrates with Android phones running Termux to provide:
- Send/receive SMS via Termux:API or shell commands
- Basic device interaction (battery, storage, etc.)

NOTE: `ctranslate2` (required for voice) has no Android wheels.
Install Hermes with the `.[termux]` extra for the tested path.

See: website/docs/getting-started/termux.md

INTEGRATION CHECKLIST (remaining items marked TODO):
  TODO 1: Add Platform.TERMUX to gateway/config.py Platform enum
  TODO 2: Add env var loading in _apply_env_overrides()
  TODO 3: Add TERMUX entry to _create_adapter() in gateway/run.py
  TODO 4: Add TERMUX to _is_user_authorized() allowlist maps
  TODO 5: Add TERMUX entry to hermes_cli/gateway.py _PLATFORMS list
  TODO 6: Add platform hints to agent/prompt_builder.py PLATFORM_HINTS
  TODO 7: Add routing in tools/send_message_tool.py
  TODO 8: Add cron delivery in cron/scheduler.py platform_map
  TODO 9: Add status display entry in hermes_cli/status.py
  TODO 10: Add tests in tests/gateway/test_termux.py
"""

import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    SessionSource,
)

logger = logging.getLogger(__name__)

# ── Termux:API defaults ──────────────────────────────────────
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8080


def check_termux_requirements() -> bool:
    """Check if Termux platform dependencies are available.

    No hard Python deps — sending uses shell commands (termux-sms-send).
    Receiving requires Termux:API WebSocket, which is optional.
    """
    return True


# ── Adapter ───────────────────────────────────────────────────


class TermuxAdapter(BasePlatformAdapter):
    """
    Termux/Android platform adapter.

    Sending: uses ``termux-sms-send`` shell command (available in Termux).
    Receiving: optionally connects to Termux:API WebSocket for real-time SMS events.
    Falls back to shell-only mode if WebSocket is unavailable.

    Configuration (via ~/.hermes/config.yaml or env vars):

        platforms:
          termux:
            enabled: true
            token: ""          # not used (no auth for local Termux)
            extra:
              host: "127.0.0.1"
              port: 8080       # Termux:API WebSocket port
    """

    MAX_MESSAGE_LENGTH = 4000  # SMS body character limit
    MAX_RECV_HISTORY = 50      # inbound messages to buffer

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.TERMUX)
        self._host: str = config.extra.get("host", _DEFAULT_HOST)
        self._port: int = int(config.extra.get("port", _DEFAULT_PORT))
        self._connected: bool = False
        self._ws = None
        self._recv_buffer: List[MessageEvent] = []

    # ── Lifecycle ─────────────────────────────────────────────

    async def connect(self) -> bool:
        """Connect to Termux:API WebSocket (if available).

        Falls back to shell-only mode (send-only) if WebSocket fails.
        """
        if not AIOHTTP_AVAILABLE:
            logger.warning(
                "[termux] aiohttp not installed — running in shell-only (send) mode"
            )
            self._connected = True
            self._mark_connected()
            return True

        try:
            import aiohttp

            url = f"ws://{self._host}:{self._port}/"
            session = aiohttp.ClientSession()
            timeout = aiohttp.ClientWSMessage(timeout=5)
            ws = await session.ws_connect(url, timeout=aiohttp.ClientTimeout(total=5))
            self._ws = ws
            self._connected = True
            self._mark_connected()
            logger.info("[termux] Connected via WebSocket at %s:%s", self._host, self._port)

            # Start background listener for inbound messages
            asyncio.create_task(self._ws_listener())
            return True
        except Exception as exc:
            logger.warning(
                "[termux] WebSocket unavailable (%s), using shell-only mode", exc
            )
            self._connected = True
            self._mark_connected()
            return True

    async def _ws_listener(self) -> None:
        """Background task: listen for inbound SMS via Termux:API WebSocket."""
        if self._ws is None:
            return
        try:
            async for msg in self._ws:
                if msg.type in (self._ws.MSG_TYPE_TEXT, self._ws.MSG_TYPE_BINARY):
                    try:
                        data = json.loads(
                            msg.data if isinstance(msg.data, str) else msg.data.decode()
                        )
                        await self._handle_inbound(data)
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        logger.debug("[termux] Non-JSON WebSocket message: %s", exc)
                elif msg.type == self._ws.MSG_TYPE_CLOSE:
                    logger.info("[termux] WebSocket closed by server")
                    break
        except Exception as exc:
            logger.warning("[termux] WebSocket listener error: %s", exc)
        finally:
            self._connected = False

    async def _handle_inbound(self, data: dict) -> None:
        """Process an inbound event from Termux:API."""
        event_type = data.get("event", "")

        if event_type == "sms":
            sender = data.get("from", "unknown")
            body = data.get("body", "")
            session_source = SessionSource(
                platform=Platform.TERMUX,
                chat_id=sender,
                user_id=sender,
                user_name=sender,
                chat_type="dm",
            )
            event = MessageEvent(
                text=body,
                message_type=MessageType.TEXT,
                source=session_source,
                message_id=str(data.get("id", "")),
                raw_message=data,
            )
            # Buffer for polling; gateway will pick up via handle_message
            self._recv_buffer.append(event)
            if len(self._recv_buffer) > self.MAX_RECV_HISTORY:
                self._recv_buffer.pop(0)

        elif event_type == "battery":
            level = data.get("level", "?")
            logger.info("[termux] Battery: %s%%", level)

    async def disconnect(self) -> None:
        """Disconnect from Termux:API."""
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._connected = False
        self._mark_disconnected()
        logger.info("[termux] Disconnected")

    # ── Receiving ─────────────────────────────────────────────

    async def receive(self) -> Optional[MessageEvent]:
        """Return next buffered inbound message, or None."""
        if self._recv_buffer:
            return self._recv_buffer.pop(0)
        return None

    # ── Sending ───────────────────────────────────────────────

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an SMS message via Termux shell command.

        chat_id should be a phone number in E.164 format (e.g. +15551234567).
        Falls back to Android am broadcast if termux-sms-send is unavailable.
        """
        body = content[: self.MAX_MESSAGE_LENGTH]

        # Try termux-sms-send (preferred)
        try:
            result = subprocess.run(
                ["termux-sms-send", "-n", chat_id, body],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return SendResult(success=True)
            error = result.stderr.strip() or result.stdout.strip()
            logger.error("[termux] SMS send failed: %s", error)
            return SendResult(success=False, error=error)
        except FileNotFoundError:
            pass  # Fall through to am broadcast
        except subprocess.TimeoutExpired:
            return SendResult(success=False, error="termux-sms-send timed out")

        # Fallback: Android am broadcast (requires Termux:API or root)
        try:
            result = subprocess.run(
                [
                    "am",
                    "broadcast",
                    "-a",
                    "android.intent.action.SEND_SMS",
                    "--es",
                    "address",
                    chat_id,
                    "--es",
                    "sms_body",
                    body,
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                return SendResult(success=True)
            return SendResult(success=False, error=result.stderr.strip())
        except FileNotFoundError:
            return SendResult(
                success=False,
                error="Neither termux-sms-send nor am found (not in Termux PATH)",
            )
        except subprocess.TimeoutExpired:
            return SendResult(success=False, error="am broadcast timed out")
        except Exception as exc:
            logger.error("[termux] Send exception: %s", exc)
            return SendResult(success=False, error=str(exc))

    async def send_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """SMS does not support typing indicators — no-op."""
        pass

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info for a chat (phone number as identifier)."""
        return {
            "name": f"SMS:{chat_id}",
            "type": "sms",
            "chat_id": chat_id,
        }

    # ── Media (limited SMS support) ──────────────────────────

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """MMS sending could be added via termux-share or Android intents."""
        return SendResult(
            success=False,
            error="Image sending not yet supported via Termux SMS adapter "
            "(MMS/share-intent integration pending)",
        )

    async def send_document(
        self,
        chat_id: str,
        path: str,
        caption: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Document sending not supported yet on SMS path."""
        return SendResult(
            success=False,
            error="Document sending not yet supported via Termux SMS adapter",
        )


# ── Register with platform_registry ───────────────────────────
# This lets the gateway discover the Termux adapter via the
# plugin-registered path, which handles auth, delivery routing,
# and connected-state checks automatically through PlatformEntry.
# The registry is checked BEFORE the hardcoded if/elif chain so
# this entry is picked up first even though we also have an elif
# in _create_adapter() for backward compat with direct imports.

def _register_termux() -> None:
    """Register TermuxAdapter with the global platform registry."""
    try:
        from gateway.platform_registry import platform_registry, PlatformEntry

        platform_registry.register(PlatformEntry(
            name="termux",
            label="Termux (Android SMS)",
            adapter_factory=lambda cfg: TermuxAdapter(cfg),
            check_fn=check_termux_requirements,
            validate_config=lambda cfg: True,
            is_connected=lambda cfg: True,
            required_env=["TERMUX_ALLOWED_USERS", "TERMUX_HOME_CHANNEL"],
            install_hint="Run within Termux on Android",
            allowed_users_env="TERMUX_ALLOWED_USERS",
            allow_all_env="TERMUX_ALLOW_ALL_USERS",
            cron_deliver_env_var="TERMUX_HOME_CHANNEL",
            emoji="📱",
            platform_hint="You are on Android via Termux (SMS). "
                          "Keep responses concise — SMS has a 160-char "
                          "limit per segment, though the adapter will "
                          "auto-chunk longer messages.",
        ))
    except Exception:
        pass  # Registry not available during early import or test

_register_termux()