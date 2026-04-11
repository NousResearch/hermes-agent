"""
IRC Platform Adapter for Hermes Agent.

A plugin-based gateway adapter that connects to an IRC server and relays
messages to/from the Hermes agent.  Zero external dependencies — uses
Python's stdlib asyncio for the IRC protocol.

Configuration in config.yaml::

    gateway:
      platforms:
        irc:
          enabled: true
          extra:
            server: irc.libera.chat
            port: 6697
            nickname: hermes-bot
            channel: "#hermes"
            use_tls: true
            server_password: ""       # optional server password
            nickserv_password: ""     # optional NickServ identification
            allowed_users: []         # empty = allow all, or list of nicks
            max_message_length: 450   # IRC line limit (safe default)

Or via environment variables (overrides config.yaml):
    IRC_SERVER, IRC_PORT, IRC_NICKNAME, IRC_CHANNEL, IRC_USE_TLS,
    IRC_SERVER_PASSWORD, IRC_NICKSERV_PASSWORD
"""

import asyncio
import logging
import os
import re
import ssl
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import: BasePlatformAdapter and friends live in the main repo.
# We import at function/class level to avoid import errors when the plugin
# is discovered but the gateway hasn't been fully initialised yet.
# ---------------------------------------------------------------------------

from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
    MessageEvent,
    MessageType,
)
from gateway.session import SessionSource
from gateway.config import PlatformConfig, Platform


def _ensure_imports():
    """No-op — kept for backward compatibility with any call sites."""
    pass


# ---------------------------------------------------------------------------
# IRC protocol helpers
# ---------------------------------------------------------------------------

def _parse_irc_message(raw: str) -> dict:
    """Parse a raw IRC protocol line into components.

    Returns dict with keys: prefix, command, params.
    """
    prefix = ""
    trailing = ""

    if raw.startswith(":"):
        prefix, raw = raw[1:].split(" ", 1)

    if " :" in raw:
        raw, trailing = raw.split(" :", 1)

    parts = raw.split()
    command = parts[0] if parts else ""
    params = parts[1:] if len(parts) > 1 else []
    if trailing:
        params.append(trailing)

    return {"prefix": prefix, "command": command, "params": params}


def _extract_nick(prefix: str) -> str:
    """Extract nickname from IRC prefix (nick!user@host)."""
    return prefix.split("!")[0] if "!" in prefix else prefix


# ---------------------------------------------------------------------------
# IRC Adapter
# ---------------------------------------------------------------------------

class IRCAdapter(BasePlatformAdapter):
    """Async IRC adapter implementing the BasePlatformAdapter interface.

    This class is instantiated by the adapter_factory passed to
    register_platform().
    """

    def __init__(self, config, **kwargs):
        platform = Platform("irc")
        super().__init__(config=config, platform=platform)

        extra = getattr(config, "extra", {}) or {}

        # Connection settings (env vars override config.yaml)
        self.server = os.getenv("IRC_SERVER") or extra.get("server", "")
        self.port = int(os.getenv("IRC_PORT") or extra.get("port", 6697))
        self.nickname = os.getenv("IRC_NICKNAME") or extra.get("nickname", "hermes-bot")
        self.channel = os.getenv("IRC_CHANNEL") or extra.get("channel", "")
        self.use_tls = (
            os.getenv("IRC_USE_TLS", "").lower() in ("1", "true", "yes")
            if os.getenv("IRC_USE_TLS")
            else extra.get("use_tls", True)
        )
        self.server_password = os.getenv("IRC_SERVER_PASSWORD") or extra.get("server_password", "")
        self.nickserv_password = os.getenv("IRC_NICKSERV_PASSWORD") or extra.get("nickserv_password", "")

        # Auth
        self.allowed_users: list = extra.get("allowed_users", [])

        # IRC limits
        self.max_message_length = int(extra.get("max_message_length", 450))

        # Runtime state
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._current_nick = self.nickname
        self._registered = False  # IRC registration complete
        self._registration_event = asyncio.Event()

    @property
    def name(self) -> str:
        return "IRC"

    # ── Connection lifecycle ──────────────────────────────────────────────

    async def connect(self) -> bool:
        """Connect to the IRC server, register, and join the channel."""
        if not self.server or not self.channel:
            logger.error("IRC: server and channel must be configured")
            self._set_fatal_error(
                "config_missing",
                "IRC_SERVER and IRC_CHANNEL must be set",
                retryable=False,
            )
            return False

        try:
            ssl_ctx = None
            if self.use_tls:
                ssl_ctx = ssl.create_default_context()

            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.server, self.port, ssl=ssl_ctx),
                timeout=30.0,
            )
        except Exception as e:
            logger.error("IRC: failed to connect to %s:%s — %s", self.server, self.port, e)
            self._set_fatal_error("connect_failed", str(e), retryable=True)
            return False

        # IRC registration sequence
        if self.server_password:
            await self._send_raw(f"PASS {self.server_password}")
        await self._send_raw(f"NICK {self.nickname}")
        await self._send_raw(f"USER {self.nickname} 0 * :Hermes Agent")

        # Start receive loop
        self._recv_task = asyncio.create_task(self._receive_loop())

        # Wait for registration (001 RPL_WELCOME) with timeout
        try:
            await asyncio.wait_for(self._registration_event.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.error("IRC: registration timed out")
            await self.disconnect()
            self._set_fatal_error("registration_timeout", "IRC server did not send RPL_WELCOME", retryable=True)
            return False

        # NickServ identification
        if self.nickserv_password:
            await self._send_raw(f"PRIVMSG NickServ :IDENTIFY {self.nickserv_password}")
            await asyncio.sleep(2)  # Give NickServ time to process

        # Join channel
        await self._send_raw(f"JOIN {self.channel}")

        self._mark_connected()
        logger.info("IRC: connected to %s:%s as %s, joined %s", self.server, self.port, self._current_nick, self.channel)
        return True

    async def disconnect(self) -> None:
        """Quit and close the connection."""
        self._mark_disconnected()
        if self._writer and not self._writer.is_closing():
            try:
                await self._send_raw("QUIT :Hermes Agent shutting down")
                await asyncio.sleep(0.5)
            except Exception:
                pass
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass

        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass

        self._reader = None
        self._writer = None
        self._registered = False
        self._registration_event.clear()

    # ── Sending ───────────────────────────────────────────────────────────

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if not self._writer or self._writer.is_closing():
            return SendResult(success=False, error="Not connected")

        target = chat_id  # channel name or nick for DMs
        lines = self._split_message(content, target)

        for line in lines:
            try:
                await self._send_raw(f"PRIVMSG {target} :{line}")
                # Basic rate limiting to avoid excess flood
                await asyncio.sleep(0.3)
            except Exception as e:
                return SendResult(success=False, error=str(e))

        return SendResult(success=True, message_id=str(int(time.time() * 1000)))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """IRC has no typing indicator — no-op."""
        pass

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        is_channel = chat_id.startswith("#") or chat_id.startswith("&")
        return {
            "name": chat_id,
            "type": "group" if is_channel else "dm",
        }

    # ── Message splitting ─────────────────────────────────────────────────

    def _split_message(self, content: str, target: str) -> List[str]:
        """Split a long message into IRC-safe chunks.

        IRC has a ~512 byte line limit.  After accounting for protocol
        overhead (``PRIVMSG <target> :``), we split content into chunks.
        """
        # Strip markdown formatting that doesn't render in IRC
        content = self._strip_markdown(content)

        overhead = len(f"PRIVMSG {target} :".encode("utf-8")) + 2  # +2 for \r\n
        max_bytes = 510 - overhead
        max_chars = min(self.max_message_length, max_bytes)

        lines: List[str] = []
        for paragraph in content.split("\n"):
            if not paragraph.strip():
                continue
            while len(paragraph) > max_chars:
                # Find a space to break at
                split_at = paragraph.rfind(" ", 0, max_chars)
                if split_at < max_chars // 3:
                    split_at = max_chars
                lines.append(paragraph[:split_at])
                paragraph = paragraph[split_at:].lstrip()
            if paragraph.strip():
                lines.append(paragraph)

        return lines if lines else [""]

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Convert basic markdown to plain text for IRC."""
        # Bold: **text** or __text__ → text
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"__(.+?)__", r"\1", text)
        # Italic: *text* or _text_ → text
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)
        # Inline code: `text` → text
        text = re.sub(r"`(.+?)`", r"\1", text)
        # Code blocks: ```...``` → content
        text = re.sub(r"```\w*\n?", "", text)
        # Images: ![alt](url) → url  (must come BEFORE links)
        text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"\2", text)
        # Links: [text](url) → text (url)
        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
        return text

    # ── Raw IRC I/O ──────────────────────────────────────────────────────

    async def _send_raw(self, line: str) -> None:
        """Send a raw IRC protocol line."""
        if not self._writer or self._writer.is_closing():
            return
        encoded = (line + "\r\n").encode("utf-8")
        self._writer.write(encoded)
        await self._writer.drain()

    async def _receive_loop(self) -> None:
        """Main receive loop — reads lines and dispatches them."""
        buffer = b""
        try:
            while self._reader and not self._reader.at_eof():
                data = await self._reader.read(4096)
                if not data:
                    break
                buffer += data
                while b"\r\n" in buffer:
                    line, buffer = buffer.split(b"\r\n", 1)
                    try:
                        decoded = line.decode("utf-8", errors="replace")
                        await self._handle_line(decoded)
                    except Exception as e:
                        logger.warning("IRC: error handling line: %s", e)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("IRC: receive loop error: %s", e)
        finally:
            if self.is_connected:
                logger.warning("IRC: connection lost, marking disconnected")
                self._set_fatal_error("connection_lost", "IRC connection closed unexpectedly", retryable=True)
                await self._notify_fatal_error()

    async def _handle_line(self, raw: str) -> None:
        """Dispatch a single IRC protocol line."""
        msg = _parse_irc_message(raw)
        command = msg["command"]
        params = msg["params"]

        # PING/PONG keepalive
        if command == "PING":
            payload = params[0] if params else ""
            await self._send_raw(f"PONG :{payload}")
            return

        # RPL_WELCOME (001) — registration complete
        if command == "001":
            self._registered = True
            self._registration_event.set()
            if params:
                # Server may confirm our nick in the first param
                self._current_nick = params[0]
            return

        # ERR_NICKNAMEINUSE (433) — nick collision during registration
        if command == "433":
            self._current_nick = self.nickname + "_"
            await self._send_raw(f"NICK {self._current_nick}")
            return

        # PRIVMSG — incoming message (channel or DM)
        if command == "PRIVMSG" and len(params) >= 2:
            sender_nick = _extract_nick(msg["prefix"])
            target = params[0]
            text = params[1]

            # Ignore our own messages
            if sender_nick.lower() == self._current_nick.lower():
                return

            # CTCP ACTION (/me) — convert to text
            if text.startswith("\x01ACTION ") and text.endswith("\x01"):
                text = f"* {sender_nick} {text[8:-1]}"

            # Ignore other CTCP
            if text.startswith("\x01"):
                return

            # Determine if this is a channel message or DM
            is_channel = target.startswith("#") or target.startswith("&")
            chat_id = target if is_channel else sender_nick
            chat_type = "group" if is_channel else "dm"

            # In channels, only respond if addressed (nick: or nick,)
            if is_channel:
                addressed = False
                for prefix in (f"{self._current_nick}:", f"{self._current_nick},",
                               f"{self._current_nick} "):
                    if text.lower().startswith(prefix.lower()):
                        text = text[len(prefix):].strip()
                        addressed = True
                        break
                if not addressed:
                    return  # Ignore unaddressed channel messages

            # Auth check
            if self.allowed_users and sender_nick not in self.allowed_users:
                logger.debug("IRC: ignoring message from unauthorized user %s", sender_nick)
                return

            await self._dispatch_message(
                text=text,
                chat_id=chat_id,
                chat_type=chat_type,
                user_id=sender_nick,
                user_name=sender_nick,
            )

        # NICK — track our own nick changes
        if command == "NICK" and _extract_nick(msg["prefix"]).lower() == self._current_nick.lower():
            if params:
                self._current_nick = params[0]

    async def _dispatch_message(
        self,
        text: str,
        chat_id: str,
        chat_type: str,
        user_id: str,
        user_name: str,
    ) -> None:
        """Build a MessageEvent and hand it to the base class handler."""
        if not self._message_handler:
            return

        source = self.build_source(
            chat_id=chat_id,
            chat_name=chat_id,
            chat_type=chat_type,
            user_id=user_id,
            user_name=user_name,
        )

        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            message_id=str(int(time.time() * 1000)),
            timestamp=__import__("datetime").datetime.now(),
        )

        await self.handle_message(event)


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def check_requirements() -> bool:
    """Check if IRC is configured.

    Only requires the server and channel — no external pip packages needed.
    """
    server = os.getenv("IRC_SERVER", "")
    channel = os.getenv("IRC_CHANNEL", "")
    # Also accept config.yaml-only configuration (no env vars).
    # The gateway passes PlatformConfig; we just check env for the
    # hermes setup / requirements check path.
    return bool(server and channel)


def validate_config(config) -> bool:
    """Validate that the platform config has enough info to connect."""
    extra = getattr(config, "extra", {}) or {}
    server = os.getenv("IRC_SERVER") or extra.get("server", "")
    channel = os.getenv("IRC_CHANNEL") or extra.get("channel", "")
    return bool(server and channel)


def register(ctx):
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="irc",
        label="IRC",
        adapter_factory=lambda cfg: IRCAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        required_env=["IRC_SERVER", "IRC_CHANNEL", "IRC_NICKNAME"],
        install_hint="No extra packages needed (stdlib only)",
        # Auth env vars for _is_user_authorized() integration
        allowed_users_env="IRC_ALLOWED_USERS",
        allow_all_env="IRC_ALLOW_ALL_USERS",
        # IRC line limit after protocol overhead
        max_message_length=450,
        # Display
        emoji="💬",
        # IRC doesn't have phone numbers to redact
        pii_safe=False,
        allow_update_command=True,
    )
