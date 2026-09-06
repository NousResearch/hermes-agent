"""IRC Platform Adapter for Hermes Agent — stdlib asyncio only, zero external dependencies.

config.yaml ``gateway.platforms.irc.extra`` keys: server, port (6697), nickname (hermes-bot), channel,
use_tls (true), server_password, nickserv_password, allowed_users ([] = allow all), max_message_length (450).
Env vars override config.yaml: IRC_SERVER, IRC_PORT, IRC_NICKNAME, IRC_CHANNEL, IRC_USE_TLS,
IRC_SERVER_PASSWORD, IRC_NICKSERV_PASSWORD.
"""

import asyncio
import datetime
import contextlib
import logging
import re
import ssl
import time
from typing import Any, Dict, List, Optional

from gateway.platforms._shared import coerce_port, get_scoped_secret as _get_scoped_secret
from gateway.platforms.base import BasePlatformAdapter, SendResult, MessageEvent, MessageType
from gateway.config import Platform


logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes"}
_EOF = object()  # _standalone_send: server closed the connection
_MARKDOWN_RULES = (
    (r"\*\*(.+?)\*\*", r"\1"),  # bold
    (r"__(.+?)__", r"\1"),
    (r"\*(.+?)\*", r"\1"),  # italic
    (r"(?<!\w)_(.+?)_(?!\w)", r"\1"),
    (r"`(.+?)`", r"\1"),  # inline code
    (r"```\w*\n?", ""),  # code fences
    (r"!\[([^\]]*)\]\(([^)]+)\)", r"\2"),  # images → url (must precede links)
    (r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)"),  # links → text (url)
)


# ── IRC protocol helpers ─────────────────────────────────────────────────────

def _parse_irc_message(raw: str) -> dict:
    """Parse a raw IRC line into ``{"prefix", "command", "params"}``."""
    prefix, trailing = "", ""
    if raw.startswith(":"):
        prefix, _, raw = raw[1:].partition(" ")
    if " :" in raw:
        raw, trailing = raw.split(" :", 1)
    parts = raw.split()
    params = parts[1:] if len(parts) > 1 else []
    if trailing:
        params.append(trailing)
    return {"prefix": prefix, "command": parts[0] if parts else "", "params": params}


def _extract_nick(prefix: str) -> str:
    """Extract nickname from IRC prefix (nick!user@host)."""
    return prefix.split("!")[0]


def _ms_id() -> str:
    return str(int(time.time() * 1000))


def _env_or_extra(extra: dict, env: str, key: str, default: Any = "") -> Any:
    """Env var overrides config.yaml ``extra``."""
    return _get_scoped_secret(env) or extra.get(key, default)


def _server_channel(config) -> tuple:
    extra = getattr(config, "extra", {}) or {}
    return _env_or_extra(extra, "IRC_SERVER", "server"), _env_or_extra(extra, "IRC_CHANNEL", "channel")


def _chunk_paragraph(paragraph: str, limit: int) -> List[str]:
    """Split one line into UTF-8 chunks of at most ``limit`` bytes, preferring space boundaries."""
    chunks: List[str] = []
    while paragraph:
        if len(paragraph.encode("utf-8")) <= limit:
            chunks.append(paragraph)
            break
        # Binary search for the largest character prefix that fits within limit
        low, high, split_at = 1, len(paragraph), 0
        while low <= high:
            mid = (low + high) // 2
            if len(paragraph[:mid].encode("utf-8")) <= limit:
                split_at, low = mid, mid + 1
            else:
                high = mid - 1
        space = paragraph.rfind(" ", 0, split_at)
        if space > split_at // 3:
            split_at = space
        chunks.append(paragraph[:split_at].rstrip())
        paragraph = paragraph[split_at:].lstrip()
    return chunks


def _privmsg_budget(target: str) -> int:
    """Payload bytes left in a 510-byte line after ``PRIVMSG <target> :`` and CRLF."""
    return 510 - (len(f"PRIVMSG {target} :".encode("utf-8")) + 2)


def _split_lines(paragraphs, limit: int) -> List[str]:
    return [chunk for paragraph in paragraphs for chunk in _chunk_paragraph(paragraph, limit)]


def _encode_line(line: str) -> bytes:
    return (line + "\r\n").encode("utf-8")


def _ssl_ctx(use_tls: bool) -> Optional[ssl.SSLContext]:
    return ssl.create_default_context() if use_tls else None


# ── IRC Adapter ──────────────────────────────────────────────────────────────

class IRCAdapter(BasePlatformAdapter):
    """Async IRC adapter implementing the BasePlatformAdapter interface."""

    def __init__(self, config, **kwargs):
        super().__init__(config=config, platform=Platform("irc"))
        extra = getattr(config, "extra", {}) or {}
        self.server = _env_or_extra(extra, "IRC_SERVER", "server")
        self.port = coerce_port(_env_or_extra(extra, "IRC_PORT", "port", 6697), 6697)
        self.nickname = _env_or_extra(extra, "IRC_NICKNAME", "nickname", "hermes-bot")
        self.channel = _env_or_extra(extra, "IRC_CHANNEL", "channel")
        _use_tls_raw = _get_scoped_secret("IRC_USE_TLS")
        self.use_tls = _use_tls_raw.lower() in _TRUTHY if _use_tls_raw else extra.get("use_tls", True)
        self.server_password = _env_or_extra(extra, "IRC_SERVER_PASSWORD", "server_password")
        self.nickserv_password = _env_or_extra(extra, "IRC_NICKSERV_PASSWORD", "nickserv_password")
        self.allowed_users: list = extra.get("allowed_users", [])
        # IRC nicks are case-insensitive — normalise for lookups
        self._allowed_users_lower: set = {u.lower() for u in self.allowed_users if isinstance(u, str)}
        max_msg = extra.get("max_message_length")
        if max_msg is None:
            with contextlib.suppress(Exception):
                from gateway.platform_registry import platform_registry
                max_msg = platform_registry.get("irc").max_message_length
        self.max_message_length = int(max_msg or 450)
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._current_nick = self.nickname
        self._registered = False  # IRC registration complete
        self._registration_event = asyncio.Event()

    @property
    def name(self) -> str:
        return "IRC"

    def _fail(self, code: str, message: str, *, retryable: bool) -> bool:
        self._set_fatal_error(code, message, retryable=retryable)
        return False

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect to the IRC server, register, and join the channel."""
        if not self.server or not self.channel:
            logger.error("IRC: server and channel must be configured")
            return self._fail("config_missing", "IRC_SERVER and IRC_CHANNEL must be set", retryable=False)
        # Prevent two profiles from using the same IRC identity
        try:
            from gateway.status import acquire_scoped_lock
            lock_key = f"{self.server}:{self.nickname}"
            if not acquire_scoped_lock("irc", lock_key):
                logger.error("IRC: %s@%s already in use by another profile", self.nickname, self.server)
                return self._fail("lock_conflict", "IRC identity in use by another profile", retryable=False)
            self._lock_key = lock_key
        except ImportError:
            self._lock_key = None  # status module not available (e.g. tests)
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.server, self.port, ssl=_ssl_ctx(self.use_tls)), timeout=30.0)
        except Exception as e:
            logger.error("IRC: failed to connect to %s:%s — %s", self.server, self.port, e)
            return self._fail("connect_failed", str(e), retryable=True)
        if self.server_password:
            await self._send_raw(f"PASS {self.server_password}")
        await self._send_raw(f"NICK {self.nickname}")
        await self._send_raw(f"USER {self.nickname} 0 * :Hermes Agent")
        self._recv_task = asyncio.create_task(self._receive_loop())
        try:  # wait for registration (001 RPL_WELCOME)
            await asyncio.wait_for(self._registration_event.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.error("IRC: registration timed out")
            await self.disconnect()
            return self._fail("registration_timeout", "IRC server did not send RPL_WELCOME", retryable=True)
        if self.nickserv_password:
            await self._send_raw(f"PRIVMSG NickServ :IDENTIFY {self.nickserv_password}")
            await asyncio.sleep(2)  # Give NickServ time to process
        await self._send_raw(f"JOIN {self.channel}")
        self._mark_connected()
        logger.info("IRC: connected to %s:%s as %s, joined %s", self.server, self.port, self._current_nick, self.channel)
        self._wire_plugin_handlers(None)
        return True

    async def disconnect(self) -> None:
        """Quit and close the connection."""
        if getattr(self, "_lock_key", None):
            with contextlib.suppress(Exception):
                from gateway.status import release_scoped_lock
                release_scoped_lock("irc", self._lock_key)
        self._mark_disconnected()
        if self._writer and not self._writer.is_closing():
            with contextlib.suppress(Exception):
                await self._send_raw("QUIT :Hermes Agent shutting down")
                await asyncio.sleep(0.5)
            with contextlib.suppress(Exception):
                self._writer.close()
                await self._writer.wait_closed()
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._recv_task
        self._reader = None
        self._writer = None
        self._registered = False
        self._registration_event.clear()

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None):
        if not self._writer or self._writer.is_closing():
            return SendResult(success=False, error="Not connected")
        for line in self._split_message(content, chat_id):
            try:
                await self._send_raw(f"PRIVMSG {chat_id} :{line}")
                await asyncio.sleep(0.3)  # Basic flood protection
            except Exception as e:
                return SendResult(success=False, error=str(e))
        return SendResult(success=True, message_id=_ms_id())

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """IRC has no typing indicator — no-op."""

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "group" if chat_id.startswith(("#", "&")) else "dm"}

    def _split_message(self, content: str, target: str) -> List[str]:
        """Split a long message into IRC-safe chunks (510-byte line limit minus PRIVMSG overhead)."""
        paragraphs = [p for p in self._strip_markdown(content).split("\n") if p.strip()]
        return _split_lines(paragraphs, min(self.max_message_length, _privmsg_budget(target))) or [""]

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Convert basic markdown to plain text for IRC."""
        for pattern, repl in _MARKDOWN_RULES:
            text = re.sub(pattern, repl, text)
        return text

    async def _send_raw(self, line: str) -> None:
        """Send a raw IRC protocol line."""
        if not self._writer or self._writer.is_closing():
            return
        self._writer.write(_encode_line(line))
        await self._writer.drain()

    async def _receive_loop(self) -> None:
        """Main receive loop — reads lines and dispatches them."""
        buffer = b""
        try:
            while self._reader and not self._reader.at_eof():
                if not (data := await self._reader.read(4096)):
                    break
                buffer += data
                while b"\r\n" in buffer:
                    line, buffer = buffer.split(b"\r\n", 1)
                    try:
                        await self._handle_line(line.decode("utf-8", errors="replace"))
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
        command, params = msg["command"], msg["params"]
        if command == "PING":
            await self._send_raw(f"PONG :{params[0] if params else ''}")
        elif command == "001":  # RPL_WELCOME — registration complete
            self._registered = True
            self._registration_event.set()
            if params:
                self._current_nick = params[0]  # server may confirm our nick
        elif command == "433":  # ERR_NICKNAMEINUSE — retry: hermes_, hermes_1, hermes_2...
            if suffix_match := re.search(r"_(\d+)$", self._current_nick):
                self._current_nick = f"{self.nickname.rstrip('_0123456789')}_{int(suffix_match.group(1)) + 1}"
            else:
                self._current_nick = self.nickname + ("_" if self._current_nick == self.nickname else "_1")
            await self._send_raw(f"NICK {self._current_nick}")
        elif command == "PRIVMSG" and len(params) >= 2:
            await self._handle_privmsg(_extract_nick(msg["prefix"]), params[0], params[1])
        elif command == "NICK" and params and _extract_nick(msg["prefix"]).lower() == self._current_nick.lower():
            self._current_nick = params[0]  # track our own nick changes

    async def _handle_privmsg(self, sender_nick: str, target: str, text: str) -> None:
        if sender_nick.lower() == self._current_nick.lower():
            return  # our own message
        if text.startswith("\x01ACTION ") and text.endswith("\x01"):
            text = f"* {sender_nick} {text[8:-1]}"  # CTCP ACTION (/me)
        if text.startswith("\x01"):
            return  # other CTCP
        is_channel = target.startswith(("#", "&"))
        # In channels, only respond if addressed (nick: / nick, / nick )
        if is_channel:
            for prefix in (f"{self._current_nick}:", f"{self._current_nick},", f"{self._current_nick} "):
                if text.lower().startswith(prefix.lower()):
                    text = text[len(prefix):].strip()
                    break
            else:
                return
        if self._allowed_users_lower and sender_nick.lower() not in self._allowed_users_lower:
            logger.debug("IRC: ignoring message from unauthorized user %s", sender_nick)
            return
        await self._dispatch_message(text=text, chat_id=target if is_channel else sender_nick,
                                     chat_type="group" if is_channel else "dm",
                                     user_id=sender_nick, user_name=sender_nick)

    async def _dispatch_message(self, text: str, chat_id: str, chat_type: str, user_id: str, user_name: str) -> None:
        """Build a MessageEvent and hand it to the base class handler."""
        if not self._message_handler:
            return
        source = self.build_source(
            chat_id=chat_id, chat_name=chat_id, chat_type=chat_type, user_id=user_id, user_name=user_name)
        await self.handle_message(MessageEvent(text=text, message_type=MessageType.TEXT, source=source,
                                               message_id=_ms_id(), timestamp=datetime.datetime.now()))


# ── Plugin registration ──────────────────────────────────────────────────────

def check_requirements() -> bool:
    """Check if IRC is configured via env (server + channel; no pip packages needed)."""
    return bool(_get_scoped_secret("IRC_SERVER", "") and _get_scoped_secret("IRC_CHANNEL", ""))


def validate_config(config) -> bool:
    """Validate that the platform config (env or config.yaml) has enough info to connect."""
    server, channel = _server_channel(config)
    return bool(server and channel)


def interactive_setup() -> None:
    """`hermes gateway setup` flow (lazy hermes_cli imports keep the plugin importable outside the CLI)."""
    from hermes_cli.setup import (
        prompt, prompt_yes_no, save_env_value, get_env_value, print_header, print_info, print_warning, print_success)

    def info(*lines: str) -> None:
        for line in lines:
            print_info(line)

    def _required(label: str, env: str, default: str, what: str) -> bool:
        value = prompt(label, default=default)
        if not value:
            print_warning(f"{what} is required — skipping IRC setup")
            return False
        save_env_value(env, value.strip())
        return True
    print_header("IRC")
    existing_server = get_env_value("IRC_SERVER")
    if existing_server:
        print_info(f"IRC: already configured (server: {existing_server})")
        if not prompt_yes_no("Reconfigure IRC?", False):
            return
    info("Connect Hermes to an IRC network. Uses Python stdlib — no extra packages needed.",
         "   Works with Libera.Chat, OFTC, your own ZNC/InspIRCd, etc.")
    print()
    if not _required("IRC server hostname (e.g. irc.libera.chat)", "IRC_SERVER", existing_server or "", "Server"):
        return
    use_tls = prompt_yes_no("Use TLS (recommended)?", True)
    save_env_value("IRC_USE_TLS", "true" if use_tls else "false")
    default_port = "6697" if use_tls else "6667"
    port = prompt(f"Port (default {default_port})", default=get_env_value("IRC_PORT") or "")
    if port:
        try:
            save_env_value("IRC_PORT", str(int(port)))
        except ValueError:
            print_warning(f"Invalid port — using default {default_port}")
    elif get_env_value("IRC_PORT"):
        save_env_value("IRC_PORT", "")  # user cleared the prompt; drop the override
    for label, env, what in (("Bot nickname (e.g. hermes-bot)", "IRC_NICKNAME", "Nickname"),
                             ("Channel to join (e.g. #hermes — comma-separate for multiple)", "IRC_CHANNEL", "Channel")):
        if not _required(label, env, get_env_value(env) or "", what):
            return
    print()
    info("🔑 Optional authentication", "   Leave blank to skip.")
    for question, label, env in (
        ("Configure a server password (PASS command)?", "Server password", "IRC_SERVER_PASSWORD"),
        ("Identify with NickServ on connect?", "NickServ password", "IRC_NICKSERV_PASSWORD")):
        if prompt_yes_no(question, False) and (secret := prompt(label, password=True)):
            save_env_value(env, secret)
    print()
    info("🔒 Access control: restrict who can message the bot",
         "   IRC nicks are not authenticated — anyone can claim any nick.",
         "   For public channels, pair with NickServ-only mode on your network",
         "   if you want stronger identity guarantees.")
    if prompt_yes_no("Allow all users in the channel to talk to the bot?", False):
        save_env_value("IRC_ALLOW_ALL_USERS", "true")
        save_env_value("IRC_ALLOWED_USERS", "")
        print_warning("⚠️  Open access — any nick in the channel can command the bot.")
    else:
        save_env_value("IRC_ALLOW_ALL_USERS", "false")
        allowed = prompt("Allowed nicks (comma-separated, leave empty to deny everyone)",
                         default=get_env_value("IRC_ALLOWED_USERS") or "")
        if allowed:
            save_env_value("IRC_ALLOWED_USERS", allowed.replace(" ", ""))
            print_success("Allowlist configured")
        else:
            save_env_value("IRC_ALLOWED_USERS", "")
            print_info("No nicks allowed — the bot will ignore all messages until you add nicks.")
    print()
    print_success("IRC configuration saved to ~/.hermes/.env")
    print_info("Restart the gateway for changes to take effect: hermes gateway restart")


def is_connected(config) -> bool:
    """Check whether IRC is configured (env or config.yaml)."""
    return validate_config(config)


def _env_enablement() -> dict | None:
    """Seed ``PlatformConfig.extra`` from env vars BEFORE adapter construction; ``None`` when IRC isn't
    minimally configured (caller skips auto-enabling). ``home_channel`` becomes a ``HomeChannel``."""
    server = _get_scoped_secret("IRC_SERVER", "").strip()
    channel = _get_scoped_secret("IRC_CHANNEL", "").strip()
    if not (server and channel):
        return None
    seed: dict = {"server": server, "channel": channel}
    for env, key, conv in (("IRC_PORT", "port", int), ("IRC_NICKNAME", "nickname", str),
                           ("IRC_USE_TLS", "use_tls", lambda v: v.lower() in _TRUTHY)):
        if raw := _get_scoped_secret(env, "").strip():
            with contextlib.suppress(ValueError):  # non-numeric IRC_PORT is dropped, not fatal
                seed[key] = conv(raw)
    # Passwords also live in extra for back-compat with config.yaml users; env wins at construct time.
    for env, key in (("IRC_SERVER_PASSWORD", "server_password"), ("IRC_NICKSERV_PASSWORD", "nickserv_password")):
        if secret := _get_scoped_secret(env):
            seed[key] = secret
    # Home channel defaults to IRC_CHANNEL so cron ``deliver=irc`` has a target without extra config.
    if home := _get_scoped_secret("IRC_HOME_CHANNEL") or channel:
        seed["home_channel"] = {"chat_id": home, "name": _get_scoped_secret("IRC_HOME_CHANNEL_NAME", home)}
    return seed


def _strip_irc_control_chars(text: str) -> str:
    """Neutralise CR/LF (IRC command injection vector) and the protocol-illegal NUL byte."""
    return text.replace("\r", " ").replace("\n", " ").replace("\x00", "")


def _is_irc_channel(target: str) -> bool:
    return bool(target) and target[0] in "#&+!"


def _sa_error(detail: str) -> Dict[str, Any]:
    return {"error": f"IRC standalone send: {detail}"}


class _StandaloneConn:
    """Raw line I/O for ``_standalone_send``; ``pump`` answers PINGs while waiting for a numeric."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader, self.writer = reader, writer
        self._loop = asyncio.get_running_loop()

    async def raw(self, line: str) -> None:
        self.writer.write(_encode_line(line))
        await self.writer.drain()

    async def pump(self, timeout: float, on_msg):
        """Feed commands to ``on_msg`` until it returns non-None; None on timeout, ``_EOF`` on close."""
        deadline = self._loop.time() + timeout
        while (remaining := deadline - self._loop.time()) > 0:
            try:
                raw_line = await asyncio.wait_for(self.reader.readuntil(b"\r\n"), timeout=remaining)
            except asyncio.TimeoutError:
                return None
            except asyncio.IncompleteReadError:
                return _EOF
            msg = _parse_irc_message(raw_line.decode("utf-8", errors="replace").rstrip("\r\n"))
            if msg["command"] == "PING":
                await self.raw(f"PONG :{msg['params'][0] if msg['params'] else ''}")
            elif (result := await on_msg(msg["command"])) is not None:
                return result
        return None

    async def close(self) -> None:
        with contextlib.suppress(Exception):
            self.writer.close()
            await asyncio.wait_for(self.writer.wait_closed(), timeout=5.0)


async def _sa_register(conn: _StandaloneConn, nick_base: str, server_password: str) -> Optional[Dict[str, Any]]:
    """PASS/NICK/USER and wait for 001, retrying nick collisions; returns an error dict or None on success."""
    nick_attempts = 0
    standalone_nick = f"{nick_base}-cron"[:30]

    async def _on_registration(cmd: str):
        nonlocal nick_attempts, standalone_nick
        if cmd in {"432", "433"}:
            nick_attempts += 1
            if nick_attempts > 5:
                return _sa_error("too many nick collisions")
            # Build from the stable base, not the mutated nick, so the suffix stays bounded.
            standalone_nick = f"{nick_base}-cron-{nick_attempts}"[:30]
            await conn.raw(f"NICK {standalone_nick}")
        elif cmd in {"464", "465"}:
            return _sa_error(f"server rejected client ({cmd})")
        return True if cmd == "001" else None
    if server_password:
        await conn.raw(f"PASS {_strip_irc_control_chars(server_password)}")
    await conn.raw(f"NICK {standalone_nick}")
    await conn.raw(f"USER {standalone_nick} 0 * :Hermes Agent (cron)")
    registered = await conn.pump(15.0, _on_registration)
    if registered is None:
        return _sa_error("registration timeout (no RPL_WELCOME)")
    if registered is _EOF:
        return _sa_error("server closed connection during registration")
    return None if registered is True else registered


async def _sa_join(conn: _StandaloneConn, target: str) -> Optional[Dict[str, Any]]:
    """JOIN a channel target (+n channels drop PRIVMSG from non-members); error dict only on explicit rejection."""
    async def _on_join(cmd: str):
        if cmd in {"403", "405", "471", "473", "474", "475"}:
            return _sa_error(f"JOIN {target} rejected ({cmd})")
        return True if cmd in {"366", "JOIN"} else None
    await conn.raw(f"JOIN {target}")
    # No JOIN ack within 5s (or EOF): proceed anyway, the server may still deliver.
    joined = await conn.pump(5.0, _on_join)
    return joined if isinstance(joined, dict) else None


async def _standalone_send(pconfig, chat_id: str, message: str, *, thread_id: Optional[str] = None,
                           media_files: Optional[List[str]] = None, force_document: bool = False) -> Dict[str, Any]:
    """Open an ephemeral IRC connection, send a PRIVMSG, and quit (out-of-process cron delivery via
    ``send_message_tool``). Uses a distinct ``-cron`` nick so it never collides with the live gateway adapter.
    ``thread_id``/``media_files`` are accepted for signature parity only."""
    extra = getattr(pconfig, "extra", {}) or {}
    server, channel = _server_channel(pconfig)
    if not server or not channel:
        return _sa_error("IRC_SERVER and IRC_CHANNEL must be configured")
    port_value = _env_or_extra(extra, "IRC_PORT", "port", 6697)
    try:
        port = int(port_value)
    except (TypeError, ValueError):
        return _sa_error(f"invalid port {port_value!r}")
    use_tls_env = _get_scoped_secret("IRC_USE_TLS")
    use_tls = use_tls_env.lower() in _TRUTHY if use_tls_env is not None else bool(extra.get("use_tls", True))
    # Reject control characters in chat_id to block IRC command injection.
    target = chat_id or channel
    if any(ch in target for ch in ("\r", "\n", "\x00", " ")):
        return _sa_error("chat_id contains illegal IRC characters")
    # Cap the base to 24 chars so collision retries stay within the 30-char NICKLEN most networks enforce.
    nick_base = _env_or_extra(extra, "IRC_NICKNAME", "nickname", "hermes-bot").rstrip("_0123456789-")[:24] or "hermes-bot"
    plain = IRCAdapter._strip_markdown(message)
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(server, port, ssl=_ssl_ctx(use_tls)), timeout=15.0)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        return {"error": f"IRC standalone connect failed: {e}"}
    conn = _StandaloneConn(reader, writer)
    try:
        if error := await _sa_register(conn, nick_base, _env_or_extra(extra, "IRC_SERVER_PASSWORD", "server_password")):
            return error
        if nickserv_password := _env_or_extra(extra, "IRC_NICKSERV_PASSWORD", "nickserv_password"):
            await conn.raw(f"PRIVMSG NickServ :IDENTIFY {_strip_irc_control_chars(nickserv_password)}")
            await asyncio.sleep(2)
        # JOIN before PRIVMSG; never JOIN bare nicks (DM target) or server queries.
        if _is_irc_channel(target) and (error := await _sa_join(conn, target)):
            return error
        # Bytes-aware per-line splitting (same algorithm as IRCAdapter._split_message),
        # with control-character stripping per line to block CRLF injection from content.
        paragraphs = [q for q in (_strip_irc_control_chars(p).rstrip() for p in plain.split("\n")) if q]
        lines = _split_lines(paragraphs, _privmsg_budget(target))
        for line in lines:
            await conn.raw(f"PRIVMSG {target} :{line}")
            await asyncio.sleep(0.3)
        if not lines:
            return _sa_error("empty message after stripping")
        await conn.raw("QUIT :delivered")
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(reader.read(1024), timeout=2.0)
        return {"success": True, "message_id": _ms_id()}
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.debug("IRC standalone send raised", exc_info=True)
        return {"error": f"IRC standalone send failed: {e}"}
    finally:
        await conn.close()


def register(ctx):
    """Plugin entry point: called by the Hermes plugin system."""
    ctx.register_platform(
        name="irc",
        label="IRC",
        adapter_factory=IRCAdapter,
        check_fn=check_requirements,
        # ACTIVE lazy-installer — create_adapter() calls this when check_fn is False, right before the
        # gateway connects Teams (#79812).
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["IRC_SERVER", "IRC_CHANNEL", "IRC_NICKNAME"],
        install_hint="No extra packages needed (stdlib only)",
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,  # env-only setups show in gateway status
        cron_deliver_env_var="IRC_HOME_CHANNEL",  # defaults to IRC_CHANNEL (see _env_enablement)
        standalone_sender_fn=_standalone_send,  # cron running separately from the gateway
        allowed_users_env="IRC_ALLOWED_USERS",
        allow_all_env="IRC_ALLOW_ALL_USERS",
        max_message_length=450,  # IRC line limit after protocol overhead
        emoji="💬",
        pii_safe=False,  # IRC doesn't have phone numbers to redact
        allow_update_command=True,
        platform_hint=(
            "You are chatting via IRC. IRC does not support markdown formatting "
            "— use plain text only. Messages are limited to ~450 characters per "
            "line (long messages are automatically split). In channels, users "
            "address you by prefixing your nick. Keep responses concise and "
            "conversational."))


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import os  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
