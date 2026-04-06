"""IRC gateway adapter.

Connects to any IRC server (IRCd) via asyncio sockets with IRCv3 protocol
support, including draft/multiline for multiline messages.

Environment variables:
    IRC_SERVER           IRC server hostname (e.g. irc.libera.chat)
    IRC_PORT             IRC server port (default: 6667, use 6697 for TLS)
    IRC_NICK             Bot nickname
    IRC_USERNAME          Username for USER command (default: IRC_NICK)
    IRC_REALNAME         Realname for USER command (default: "Hermes Agent")
    IRC_PASSWORD         Server password (optional)
    IRC_CHANNELS         Comma-separated list of channels to join (e.g. #bots,#help)
    IRC_USE_TLS          Set "true" to enable TLS (default: false)
    IRC_TLS_CA_CERT      Path to PEM-encoded certificate file to trust for TLS (optional)
    IRC_MESSAGE_CHUNK_LIMIT  Max characters per message when BATCH unavailable (default: 350)
    IRC_REQUIRE_MULTILINE Set "true" to fail connection if server doesn't support draft/multiline (default: false)
    IRC_NICKSERV_PASSWORD NickServ password for authentication (optional)
    IRC_NICKSERV_SERVICE NickServ service name (default: NickServ)
    IRC_ALLOWED_USERS    Comma-separated nicks/hosts allowed to command bot
    IRC_HOME_CHANNEL     Default channel for cron/notification delivery

Notes:
    - IRC is text-only; no media support (images, voice, documents)
    - Markdown is supported and will be rendered
    - Uses IRCv3 draft/multiline for sending multiline messages when available
    - Falls back to splitting multiline messages into separate PRIVMSG when
      draft/multiline is not supported by the server
    - Message length is limited to ~512 bytes per IRC protocol
    - NickServ authentication sends IDENTIFY after successful registration (001)
    - Nick collision (433/436 errors) attempts fallback nick with "_" suffix
      then gives up as fatal error
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import ssl
import time
from typing import Any, Dict, List, Optional, Set

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    SessionSource,
)

logger = logging.getLogger(__name__)

# Grace period: ignore messages older than this many seconds before startup.
_STARTUP_GRACE_SECONDS = 5

# Regex patterns for IRC protocol parsing
# Updated to support IRCv3 message tags: @tag=value;tag2=value2 PREFIX COMMAND ...
IRC_MESSAGE_RE = re.compile(
    r"(?:@(?P<tags>[^ ]+) )?(?::(?P<prefix>[^ ]+) )?(?P<command>[A-Z0-9]+)(?P<params>(?: [^ :][^ ]*)*)?(?: :(?P<trailing>.*))?"
)

IRC_PREFIX_RE = re.compile(
    r"(?P<nick>[^!@]+)(?:!(?P<user>[^@]+))?(?:@(?P<host>.+))?"
)


def parse_tags(tags_str: str) -> Dict[str, str]:
    """Parse IRCv3 message tags into a dictionary."""
    if not tags_str:
        return {}
    tags = {}
    for tag in tags_str.split(";"):
        tag = tag.strip()
        if not tag:
            continue
        if "=" in tag:
            key, value = tag.split("=", 1)
            # Unescape tag values per IRCv3 spec
            value = value.replace(r"\:", ";").replace(r"\s", " ").replace(r"\\", "\\").replace(r"\r", "\r").replace(r"\n", "\n").replace(r"\0", "\0")
            tags[key] = value
        else:
            tags[tag] = ""
    return tags


def check_irc_requirements() -> bool:
    """Return True if the IRC adapter can be used."""
    server = os.getenv("IRC_SERVER", "")
    nick = os.getenv("IRC_NICK", "")
    if not server or not nick:
        logger.debug("IRC: IRC_SERVER or IRC_NICK not set")
        return False
    return True


class IRCAdapter(BasePlatformAdapter):
    """Gateway adapter for IRC (any server)."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.IRC)

        # Connection settings
        self._server: str = os.getenv("IRC_SERVER", "")
        self._port: int = int(os.getenv("IRC_PORT", "6667"))
        self._nick: str = os.getenv("IRC_NICK", "")
        self._username: str = os.getenv("IRC_USERNAME", "") or self._nick  # Default to nick if username not set
        self._realname: str = os.getenv("IRC_REALNAME", "") or "Hermes Agent"  # Default to "Hermes Agent" if not set
        self._password: str = os.getenv("IRC_PASSWORD", "")
        self._use_tls: bool = os.getenv("IRC_USE_TLS", "").lower() in ("true", "1", "yes")

        # Channels to join
        channels_str = os.getenv("IRC_CHANNELS", "")
        self._channels: Set[str] = {
            ch.strip() if ch.strip().startswith("#") else f"#{ch.strip()}"
            for ch in channels_str.split(",")
            if ch.strip()
        }

        # Message chunk limit for long messages (when BATCH unavailable)
        self._message_chunk_limit: int = int(os.getenv("IRC_MESSAGE_CHUNK_LIMIT", "350"))

        # Require draft/multiline support (fail if server doesn't support it)
        self._require_multiline: bool = os.getenv("IRC_REQUIRE_MULTILINE", "").lower() in ("true", "1", "yes")

        # Warn if multiline is required but chunk limit is at default
        if self._require_multiline and self._message_chunk_limit == 350:
            logger.warning(
                "IRC: require_multiline is enabled but message_chunk_limit is at default (350). "
                "Consider increasing message_chunk_limit or disabling require_multiline to avoid unnecessary chunking."
            )

        # Connection state
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
        self._closing = False
        self._startup_ts: float = 0.0
        self._registered = False

        # Message deduplication (bounded)
        from collections import deque
        self._processed_msgs: deque = deque(maxlen=500)
        self._processed_msgs_set: set = set()

        # IRCv3 multiline/draft support
        self._multiline_cap: bool = False
        self._batch_counter: int = 0
        self._incoming_batches: Dict[str, List[Dict[str, Any]]] = {}
        self._multiline_batch_ids: Set[str] = set()
        self._cap_negotiated: bool = False
        self._cap_negotiation_complete = asyncio.Event()
        self._accumulated_caps: str = ""  # Accumulate caps from multi-chunk CAP LS

        # NickServ authentication
        self._nickserv_password: str = os.getenv("IRC_NICKSERV_PASSWORD", "")
        self._nickserv_service: str = os.getenv("IRC_NICKSERV_SERVICE", "NickServ")

        # Nick collision handling
        # _nick = original configured name (what we wanted)
        # _actual_nick = what server actually accepted (what we got)
        self._actual_nick: str = self._nick  # Initially same, changes on collision
        self._nick_collision_attempted: bool = False

    def _parse_prefix(self, prefix: str) -> Dict[str, str]:
        """Parse IRC prefix into nick, user, host components."""
        match = IRC_PREFIX_RE.match(prefix or "")
        if match:
            return {
                "nick": match.group("nick") or "",
                "user": match.group("user") or "",
                "host": match.group("host") or "",
            }
        return {"nick": prefix or "", "user": "", "host": ""}

    def _parse_message(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse an IRC protocol line into components."""
        match = IRC_MESSAGE_RE.match(line.rstrip("\r\n"))
        if not match:
            return None
        return {
            "tags": parse_tags(match.group("tags") or ""),
            "prefix": match.group("prefix") or "",
            "command": match.group("command") or "",
            "params": (match.group("params") or "").strip().split(),
            "trailing": match.group("trailing") or "",
        }

    def _build_source(self, prefix: str, channel: str) -> SessionSource:
        """Build a SessionSource from IRC message metadata."""
        parsed = self._parse_prefix(prefix)
        nick = parsed.get("nick", "unknown")
        host = parsed.get("host", "")

        return SessionSource(
            platform=Platform.IRC,
            chat_id=channel,
            user_id=f"{nick}@{host}" if host else nick,
            user_name=nick,
            chat_type="group" if channel.startswith("#") else "dm",
        )

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Connect to the IRC server and join channels."""
        logger.debug("IRC: connect() called - server=%s, nick=%s, port=%s, tls=%s", self._server, self._nick, self._port, self._use_tls)
        logger.debug("IRC: Channels to join: %s", list(self._channels))
        if not self._server or not self._nick:
            logger.error("IRC: server or nick not configured")
            return False

        try:
            logger.debug("IRC: Attempting to open_connection to %s:%s", self._server, self._port or 6667)
            # Open connection
            if self._use_tls:
                ssl_context = ssl.create_default_context()

                # Load custom certificate if provided
                custom_cert_path = os.getenv("IRC_TLS_CA_CERT", "")
                if custom_cert_path:
                    if os.path.exists(custom_cert_path):
                        ssl_context.load_verify_locations(cafile=custom_cert_path)
                        logger.info("IRC: Loaded custom TLS certificate from %s", custom_cert_path)
                    else:
                        logger.error("IRC: TLS certificate file not found: %s", custom_cert_path)
                        return False

                self._reader, self._writer = await asyncio.open_connection(
                    self._server, self._port or 6697, ssl=ssl_context
                )
            else:
                self._reader, self._writer = await asyncio.open_connection(
                    self._server, self._port or 6667
                )
            logger.debug("IRC: Connection established successfully")
        except Exception as exc:
            logger.error("IRC: failed to connect to %s:%s: %s", self._server, self._port, exc)
            return False

        self._closing = False
        self._startup_ts = time.time()

        # Reset CAP negotiation state
        self._cap_negotiated = False
        self._cap_negotiation_complete.clear()

        # Start reader loop
        logger.debug("IRC: About to start reader_task")
        self._reader_task = asyncio.create_task(self._reader_loop())
        logger.debug("IRC: reader_task created")

        # Add exception handler for reader task
        def _reader_exception_handler(task):
            try:
                exc = task.exception()
                logger.debug("IRC: Reader task died with exception: %s", exc)
            except asyncio.CancelledError:
                logger.debug("IRC: Reader task was cancelled")
            except Exception as e:
                logger.error("IRC: Reader task exception handler error: %s", e)

        self._reader_task.add_done_callback(_reader_exception_handler)

        # Start CAP negotiation for draft/multiline support
        self._send_line("CAP LS 302")

        # Send registration
        if self._password:
            self._send_line(f"PASS {self._password}")
        self._send_line(f"NICK {self._nick}")
        self._send_line(f"USER {self._username} 0 * :{self._realname}")

        # Wait for registration (001 reply) with timeout
        logger.debug("IRC: Waiting for registration (001 reply)...")
        for i in range(450):  # 45 seconds max (slow networks)
            if self._registered:
                logger.debug("IRC: Registered after %.1f seconds", i * 0.1)
                break
            await asyncio.sleep(0.1)
        else:
            logger.error("IRC: registration timeout")
            await self.disconnect()
            return False

        # Wait for CAP negotiation to complete (multiline support)
        try:
            await asyncio.wait_for(self._cap_negotiation_complete.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("IRC: CAP negotiation timeout, proceeding without multiline support")

        # Join configured channels
        logger.debug("IRC: Joining channels: %s", list(self._channels))
        for channel in self._channels:
            logger.debug("IRC: Sending JOIN %s", channel)
            self._send_line(f"JOIN {channel}")
            logger.info("IRC: joined %s", channel)

        # Start PING task
        self._ping_task = asyncio.create_task(self._ping_loop())

        self._mark_connected()
        logger.info("IRC: connected to %s as %s", self._server, self._nick)

        # Start a heartbeat to check if reader is still alive
        async def _reader_heartbeat():
            count = 0
            while not self._closing and self._reader:
                await asyncio.sleep(5)
                count += 1
                if self._reader_task and self._reader_task.done():
                    logger.warning("IRC: Reader task is DONE at %ds!", count * 5)
                elif not self._closing:
                    logger.debug("IRC: Reader still running at %ds", count * 5)

        asyncio.create_task(_reader_heartbeat())

        return True

    async def disconnect(self) -> None:
        """Disconnect from IRC."""
        self._closing = True

        if self._ping_task and not self._ping_task.done():
            self._ping_task.cancel()
            try:
                await self._ping_task
            except (asyncio.CancelledError, Exception):
                pass

        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass

        if self._writer:
            try:
                self._send_line("QUIT :Hermes Agent disconnecting")
                await self._writer.drain()
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None

        self._reader = None
        self._registered = False
        self._mark_disconnected()
        logger.info("IRC: disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a message to an IRC channel or user."""
        if not content:
            return SendResult(success=True)

        if not self._writer:
            return SendResult(success=False, error="Not connected")

        has_newlines = "\n" in content

        if self._multiline_cap and has_newlines:
            # Use BATCH for multiline messages
            # trim() removes trailing newlines but preserves blank lines in the middle (paragraph breaks)
            lines = content.strip().split("\n")
            self._batch_counter += 1
            batch_id = f"m{self._batch_counter}"
            self._send_line(f"BATCH +{batch_id} draft/multiline {chat_id}")
            for line in lines:
                # Send empty lines as a single space to preserve paragraph breaks
                # without sending invalid empty PRIVMSG (just `:` which some servers reject)
                line_content = line or " "
                # Write directly to avoid _send_line for tag support
                self._writer.write(f"@batch={batch_id} PRIVMSG {chat_id} :{line_content}\r\n".encode("utf-8"))
            self._send_line(f"BATCH -{batch_id}")
        else:
            # Fallback: flatten newlines to spaces, then split at message_chunk_limit
            flat = content.replace("\n", " ")
            remaining = flat
            while remaining:
                if len(remaining) <= self._message_chunk_limit:
                    # Send remaining text
                    piece = remaining.strip()
                    if piece:
                        self._send_line(f"PRIVMSG {chat_id} :{piece}")
                    break
                # Find last space before limit
                split_at = remaining.rfind(" ", 0, self._message_chunk_limit)
                # If no space found before halfway point, split at limit
                if split_at < self._message_chunk_limit // 2:
                    split_at = self._message_chunk_limit
                piece = remaining[:split_at].strip()
                if piece:
                    self._send_line(f"PRIVMSG {chat_id} :{piece}")
                remaining = remaining[split_at:].lstrip()

        try:
            await self._writer.drain()
            return SendResult(success=True)
        except Exception as exc:
            logger.error("IRC: failed to send to %s: %s", chat_id, exc)
            return SendResult(success=False, error=str(exc), retryable=True)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return channel/user info."""
        if chat_id.startswith("#"):
            return {"name": chat_id, "type": "group", "chat_id": chat_id}
        return {"name": chat_id, "type": "dm", "chat_id": chat_id}

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """IRC has no typing indicator - no-op."""
        pass

    def format_message(self, content: str) -> str:
        """Return content unchanged."""
        return content

    # ------------------------------------------------------------------
    # IRC protocol helpers
    # ------------------------------------------------------------------

    def _send_line(self, line: str) -> None:
        """Send a raw IRC line."""
        if self._writer:
            try:
                self._writer.write((line + "\r\n").encode("utf-8"))
            except Exception as exc:
                logger.warning("IRC: failed to send line: %s", exc)

    async def _reader_loop(self) -> None:
        """Read and process incoming IRC messages."""
        line_count = 0
        logger.info("IRC: Reader loop starting")
        while not self._closing and self._reader:
            try:
                line = await asyncio.wait_for(self._reader.readline(), timeout=300)
                line_count += 1
                if line_count < 5 or line_count % 100 == 0:
                    logger.debug("IRC: Read line #%d: %d bytes", line_count, len(line))
            except asyncio.TimeoutError:
                logger.warning("IRC: read timeout, connection may be dead")
                continue
            except asyncio.CancelledError:
                logger.info("IRC: Reader loop cancelled")
                break
            except Exception as exc:
                if not self._closing:
                    logger.error("IRC: read error: %s", exc)
                break

            if not line:
                if not self._closing:
                    logger.warning("IRC: connection closed by server")
                break

            try:
                line = line.decode("utf-8", errors="replace").strip()
            except Exception as exc:
                logger.error("IRC: decode error: %s", exc)
                continue

            if not line:
                continue

            await self._handle_line(line)

    async def _handle_line(self, line: str) -> None:
        """Parse and handle an incoming IRC line."""
        logger.debug("IRC: Received line: %s", line[:200])

        msg = self._parse_message(line)
        if not msg:
            logger.debug("IRC: Failed to parse line: %s", line[:100])
            return

        cmd = msg["command"]
        prefix = msg["prefix"]
        params = msg["params"]
        trailing = msg["trailing"]

        # Log ALL commands for debugging
        if cmd not in ("PING", "PONG"):
            logger.debug("IRC: Parsed cmd=%s prefix=%s params=%s trailing=%s",
                      cmd, prefix[:30] if prefix else "none",
                      params[:5] if params else "none",
                      trailing[:50] if trailing else "none")

        logger.debug("IRC: Parsed command=%s prefix=%s params=%s", cmd, prefix[:50] if prefix else "", params)

        # Handle PING
        if cmd == "PING":
            logger.debug("IRC: PING received, sending PONG")
            self._send_line(f"PONG :{trailing}")
            return

        # Handle CAP LS for draft/multiline support
        if cmd == "CAP" and len(params) >= 2 and params[1] == "LS":
            # Accumulate caps from each chunk - capability can be in trailing OR params[2]
            caps = trailing if trailing else (params[2] if len(params) >= 3 else "")
            self._accumulated_caps += " " + caps.lower()
            # params[2] === "*" means more chunks coming; wait for final chunk
            if len(params) >= 3 and params[2] == "*":
                return
            # Final chunk - process accumulated caps
            if "draft/multiline" in self._accumulated_caps:
                self._send_line("CAP REQ :draft/multiline")
            else:
                if self._require_multiline:
                    # Fail connection if multiline is required but not available
                    logger.error("IRC: draft/multiline required but not supported by server")
                    self._set_fatal_error("multiline_required", "Server does not support draft/multiline", retryable=False)
                    self._send_line("QUIT")
                    return
                self._cap_negotiated = True
                self._cap_negotiation_complete.set()
                self._send_line("CAP END")
            return

        # Handle CAP ACK
        if cmd == "CAP" and len(params) >= 2 and params[1] == "ACK":
            # Capability can be in trailing OR in params[2] depending on server
            acked = trailing.lower() if trailing else (params[2].lower() if len(params) >= 3 else "")
            if "draft/multiline" in acked:
                self._multiline_cap = True
                logger.info("IRC: draft/multiline capability enabled")
            self._cap_negotiated = True
            self._cap_negotiation_complete.set()
            self._send_line("CAP END")
            return

        # Handle CAP NAK
        if cmd == "CAP" and len(params) >= 2 and params[1] == "NAK":
            rejected = trailing.lower() if trailing else (params[2].lower() if len(params) >= 3 else "")
            if self._require_multiline and "draft/multiline" in rejected:
                # Fail connection if multiline is required but server rejected it
                logger.error("IRC: draft/multiline required but server rejected CAP REQ")
                self._set_fatal_error("multiline_required", "Server rejected draft/multiline capability", retryable=False)
                self._send_line("QUIT")
                return
            self._cap_negotiated = True
            self._cap_negotiation_complete.set()
            self._send_line("CAP END")
            return

        # Handle BATCH commands for draft/multiline
        if cmd == "BATCH":
            await self._handle_batch(params, trailing)
            return

        # Handle JOIN confirmation
        if cmd == "JOIN":
            logger.info("IRC: JOIN confirmed: %s (we are now in the channel)", trailing)
            return

        # Handle numeric 353 (RPL_NAMREPLY) - list of users in channel
        if cmd == "353":
            logger.info("IRC: Users in channel %s: %s", params[2] if len(params) > 2 else "unknown", trailing[:200])
            return

        # Handle nick collision (433/436 errors) - only before registration
        if cmd in ("433", "436") and not self._registered:
            if not self._nick_collision_attempted:
                # First attempt: try fallback nick with "_" suffix
                self._nick_collision_attempted = True
                fallback_nick = f"{self._nick}_"
                # Sanitize: max 30 chars, remove invalid chars, strip whitespace
                fallback_nick = fallback_nick.replace(" ", "")[:30]
                fallback_nick = "".join(c for c in fallback_nick if c.isalnum() or c in "_-[]\\`^{}|")

                if fallback_nick.lower() != self._actual_nick.lower():
                    logger.warning("IRC: nick collision (error %s), trying fallback nick: %s", cmd, fallback_nick)
                    self._actual_nick = fallback_nick
                    self._send_line(f"NICK {fallback_nick}")
                    return
                else:
                    logger.error("IRC: nick collision and fallback nick is same, cannot recover")
                    self._set_fatal_error("nick_collision", f"Nickname {self._nick} is in use", retryable=False)
                    self._send_line("QUIT")
                    return
            else:
                # Already tried fallback, give up
                logger.error("IRC: nick collision, already tried fallback nick, giving up")
                self._set_fatal_error("nick_collision", f"Nickname {self._nick} is in use and fallback also failed", retryable=False)
                self._send_line("QUIT")
                return

        # Handle registration confirmation
        if cmd == "001":
            self._registered = True
            logger.info("IRC: registered with server")
            # Fallback: if CAP negotiation didn't complete, resolve it now
            if not self._cap_negotiated:
                self._cap_negotiated = True
                self._cap_negotiation_complete.set()

            # Send NickServ IDENTIFY if configured
            if self._nickserv_password:
                nickserv_identify = f"PRIVMSG {self._nickserv_service} :IDENTIFY {self._nick} {self._nickserv_password}"
                self._send_line(nickserv_identify)
                logger.info("IRC: sent NickServ IDENTIFY")

            return

        # Handle PRIVMSG (actual chat messages)
        if cmd == "PRIVMSG":
            tags = msg["tags"]
            await self._handle_privmsg(prefix, params, trailing, tags)
            return

        # Handle other commands as needed
        if cmd == "KICK" and len(params) >= 2:
            # We were kicked from a channel
            channel = params[0]
            target = params[1]
            if target == self._actual_nick:
                logger.warning("IRC: kicked from %s, rejoining in 5s", channel)
                await asyncio.sleep(5)
                self._send_line(f"JOIN {channel}")

        elif cmd in ("ERROR",):
            logger.error("IRC: server error: %s", trailing)
            self._set_fatal_error("irc_error", trailing, retryable=True)

    async def _handle_batch(
        self, params: List[str], trailing: str
    ) -> None:
        """Handle BATCH commands for draft/multiline support."""
        if not params:
            return

        batch_param = params[0] or trailing
        if not batch_param:
            return

        if batch_param.startswith("+"):
            # Start of batch - only track draft/multiline batches
            batch_id = batch_param[1:]
            batch_type = params[1].lower() if len(params) >= 2 else ""
            if batch_type == "draft/multiline":
                self._incoming_batches[batch_id] = []
                self._multiline_batch_ids.add(batch_id)

        elif batch_param.startswith("-"):
            # End of batch - combine and emit only for multiline batches
            batch_id = batch_param[1:]
            messages = self._incoming_batches.pop(batch_id, [])
            self._multiline_batch_ids.discard(batch_id)

            if messages and self._message_handler:
                # Combine all messages in the batch
                first = messages[0]
                combined_text = "\n".join(m["text"] for m in messages)

                # Build session source
                source = self._build_source(first["prefix"], first["chat_id"])

                # Create message event
                event = MessageEvent(
                    text=combined_text,
                    message_type=MessageType.TEXT,
                    source=source,
                    raw_message={
                        "prefix": first["prefix"],
                        "params": first["params"],
                        "trailing": combined_text,
                        "batch_id": batch_id,
                    },
                )

                try:
                    await self._message_handler(event)
                except Exception as exc:
                    logger.error("IRC: message handler error: %s", exc)

    async def _handle_privmsg(
        self, prefix: str, params: List[str], trailing: str, tags: Dict[str, str]
    ) -> None:
        """Handle an incoming PRIVMSG."""
        if not params:
            return

        target = params[0]  # Channel or our nick (for DMs)
        text = trailing

        logger.info("IRC: PRIVMSG from %s to %s: %s", prefix, target, text[:100])

        # Check if this message is part of a multiline batch *before* filtering empty text
        batch_tag = tags.get("batch")
        if batch_tag and batch_tag in self._multiline_batch_ids:
            # Buffer the message for later
            parsed = self._parse_prefix(prefix)
            self._incoming_batches[batch_tag].append({
                "prefix": prefix,
                "params": params,
                "text": text,  # preserve blank lines; joining happens at BATCH -<id>
                "sender_nick": parsed.get("nick", ""),
                "sender_user": parsed.get("user", ""),
                "sender_host": parsed.get("host", ""),
                "chat_id": target if target.startswith("#") else parsed.get("nick", prefix),
            })
            return  # Don't emit yet - wait for BATCH -<id>

        if not text.strip():
            return

        # Determine if this is a channel message or DM
        if target.startswith("#"):
            chat_id = target

            # For channels: only respond to messages that mention us at the beginning
            # Must start with "<nickname>:" (colon immediately after nickname, no space)
            # Check against original nick (what users will type) in case of collision
            mention_pattern = re.compile(f"^{re.escape(self._nick)}:", re.IGNORECASE)
            if not mention_pattern.match(text):
                logger.debug("IRC: Filtered channel message (no mention): %s", text[:100])
                return
        else:
            # DM - use sender's nick as chat_id
            # DMs always respond, no mention filtering
            parsed = self._parse_prefix(prefix)
            chat_id = parsed.get("nick", prefix)

        # Filter self-messages
        parsed = self._parse_prefix(prefix)
        sender_nick = parsed.get("nick", "")
        if sender_nick == self._actual_nick:
            logger.debug("IRC: Filtered self-message from %s", sender_nick)
            return

        # Check for CTCP (actions, etc.) - skip them
        if text.startswith("\x01") and text.endswith("\x01"):
            logger.debug("IRC: Filtered CTCP message from %s", sender_nick)
            # CTCP message - could parse /me actions, but skip for now
            return

        logger.debug("IRC: Processing message from %s to %s: %s", sender_nick, chat_id, text[:100])

        # Build session source
        source = self._build_source(prefix, chat_id)

        # Create message event
        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message={"prefix": prefix, "params": params, "trailing": trailing},
        )

        # Dispatch to handler using base adapter's handle_message method
        # This spawns a background task that handles response sending
        if self._message_handler:
            logger.debug("IRC: About to dispatch to handler")
            logger.info("IRC: Dispatching message from %s (%s) to handler", sender_nick, chat_id)
            try:
                await self.handle_message(event)
                logger.debug("IRC: Message dispatched successfully")
                logger.info("IRC: Message from %s dispatched successfully", sender_nick)
            except Exception as exc:
                logger.error("IRC: message handler error: %s", exc, exc_info=True)
        else:
            logger.warning("IRC: No message handler set - message from %s will be dropped", sender_nick)

    async def _ping_loop(self) -> None:
        """Periodically send PING to keep connection alive."""
        while not self._closing:
            await asyncio.sleep(60)
            if not self._closing:
                self._send_line(f"PING :{self._server}")
