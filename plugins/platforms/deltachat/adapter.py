"""Delta Chat platform adapter (Hermes bundled plugin).

The adapter drives a local ``deltachat-rpc-server`` child process over its
newline-delimited JSON-RPC 2.0 stdio interface. Delta Chat Core owns account
credentials, encryption, network I/O, and attachment storage; Hermes only
normalizes messages into the gateway platform contract.

Configuration lives in ``config.yaml`` (the account database is the credential
store)::

    platforms:
      deltachat:
        enabled: true
        dm_policy: pairing
        group_policy: disabled
        require_mention: true
        extra:
          email: bot@example.chat
          data_dir: <HERMES_HOME>/platforms/deltachat
          display_name: Hermes Agent
          show_invite_link: true

The recommended first-run path is ``hermes gateway setup``. It creates a
Chatmail account, persists the generated address, and prints the secure-join
invite without exposing mailbox passwords to Hermes configuration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from email.utils import parseaddr
from pathlib import Path
from typing import Any, Dict, List, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_media_bytes,
    validate_inbound_media_size,
)
from hermes_constants import display_hermes_home, get_hermes_home

logger = logging.getLogger(__name__)

DEFAULT_CHATMAIL_RELAY = "nine.testrun.org"
CHATMAIL_RELAYS_URL = "https://chatmail.at/relays"
RPC_READY_ATTEMPTS = 40
RPC_READY_DELAY_SECONDS = 0.25
RPC_CALL_TIMEOUT_SECONDS = 30.0
ACCOUNT_CONFIGURE_TIMEOUT_SECONDS = 90.0

_EMAIL_RE = re.compile(r"(?i)([a-z0-9.!#$%&'*+/=?^_`{|}~-]{1,64})@([a-z0-9.-]+)")
_AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"}


class DeltaChatRPCError(RuntimeError):
    """JSON-RPC or child-process failure."""


class DeltaChatConfigurationError(RuntimeError):
    """Non-retryable account/configuration failure."""


class DeltaChatBootstrapCreated(RuntimeError):
    """A first-run marker created an account that must be saved in config."""

    def __init__(self, address: str):
        super().__init__(address)
        self.address = address


def _redact_email(value: str) -> str:
    """Redact email local-parts before writing identifiers to logs."""

    def _replace(match: re.Match[str]) -> str:
        local, domain = match.groups()
        visible = local[:1] if local else ""
        return f"{visible}***@{domain}"

    return _EMAIL_RE.sub(_replace, str(value or ""))


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        return []
    return [str(item).strip() for item in values if str(item).strip()]


def parse_email_setting(value: str) -> tuple[str, bool]:
    """Return ``(address_or_relay, is_bootstrap_marker)``.

    ``@relay.example`` is the first-run Chatmail marker used by PicoClaw's
    implementation. A normal value must parse as one complete email address.
    """

    email = str(value or "").strip()
    if not email:
        raise ValueError(
            "Delta Chat email is required. Run `hermes gateway setup` to "
            "create a Chatmail account, or configure an existing account store."
        )
    if email.startswith("@"):
        relay = email[1:].strip().lower()
        if not relay or "@" in relay or any(char in relay for char in "/\\ \t\r\n"):
            raise ValueError(
                f"Invalid Chatmail relay marker {email!r}; use "
                f"'@{DEFAULT_CHATMAIL_RELAY}' or a relay from {CHATMAIL_RELAYS_URL}."
            )
        return relay, True

    _name, parsed = parseaddr(email)
    if parsed.casefold() != email.casefold() or parsed.count("@") != 1:
        raise ValueError(f"Invalid Delta Chat account address: {email!r}")
    local, domain = parsed.rsplit("@", 1)
    if not local or not domain or any(char.isspace() for char in parsed):
        raise ValueError(f"Invalid Delta Chat account address: {email!r}")
    return parsed.lower(), False


def _chatmail_account_qr(relay: str) -> str:
    return f"DCACCOUNT:https://{relay}/new"


def _mention_pattern(token: str) -> Optional[re.Pattern[str]]:
    token = str(token or "").strip()
    if not token:
        return None
    return re.compile(rf"(?<!\w){re.escape(token)}(?!\w)", re.IGNORECASE)


def mentions_bot(content: str, display_name: str, email: str) -> bool:
    """Whether a group message names the bot by profile or ``@localpart``."""

    tokens = [display_name]
    if "@" in email:
        tokens.append(f"@{email.split('@', 1)[0]}")
    return any(
        pattern.search(content or "")
        for pattern in (_mention_pattern(token) for token in tokens)
        if pattern is not None
    )


def _strip_bot_mention(content: str, display_name: str, email: str) -> str:
    text = content
    tokens = [display_name]
    if "@" in email:
        tokens.append(f"@{email.split('@', 1)[0]}")
    for token in tokens:
        pattern = _mention_pattern(token)
        if pattern and pattern.search(text):
            text = pattern.sub("", text, count=1)
            return text.lstrip(" \t,:;-\u2014")
    return text


def _message_type_for_media(kind: str) -> MessageType:
    return {
        "image": MessageType.PHOTO,
        "video": MessageType.VIDEO,
        "audio": MessageType.VOICE,
        "document": MessageType.DOCUMENT,
    }.get(kind, MessageType.DOCUMENT)


def _raw_platform_config() -> dict:
    """Read Delta Chat's raw config block without recursing into gateway load."""

    path = get_hermes_home() / "config.yaml"
    if not path.is_file():
        return {}
    try:
        import yaml

        root = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    if not isinstance(root, dict):
        return {}

    merged: dict = {}
    for candidate in (
        (root.get("gateway") or {}).get("platforms", {}).get("deltachat")
        if isinstance(root.get("gateway"), dict)
        and isinstance((root.get("gateway") or {}).get("platforms"), dict)
        else None,
        (root.get("platforms") or {}).get("deltachat")
        if isinstance(root.get("platforms"), dict)
        else None,
        root.get("deltachat"),
    ):
        if isinstance(candidate, dict):
            extra = {**merged.get("extra", {}), **candidate.get("extra", {})}
            merged.update(candidate)
            if extra:
                merged["extra"] = extra
    return merged


def _configured_rpc_path(extra: Optional[dict] = None) -> Optional[str]:
    configured = str((extra or {}).get("rpc_server_path") or "").strip()
    if not configured:
        raw = _raw_platform_config()
        configured = str((raw.get("extra") or {}).get("rpc_server_path") or "").strip()
    if configured:
        path = Path(configured).expanduser()
        return str(path) if path.is_file() else None
    return shutil.which("deltachat-rpc-server")


class DeltaChatRPC:
    """Concurrent stdio JSON-RPC client for ``deltachat-rpc-server``."""

    def __init__(self, server_path: str, data_dir: Path):
        self.server_path = server_path
        self.data_dir = data_dir
        self.process: Optional[asyncio.subprocess.Process] = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._write_lock = asyncio.Lock()
        self._read_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._closed = False

    async def start(self) -> None:
        if self.process is not None:
            return
        env = os.environ.copy()
        env["DC_ACCOUNTS_PATH"] = str(self.data_dir)
        try:
            self.process = await asyncio.create_subprocess_exec(
                self.server_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except OSError as exc:
            raise DeltaChatRPCError(
                f"Could not start deltachat-rpc-server at {self.server_path}: {exc}"
            ) from exc
        self._read_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())

    async def call(
        self,
        method: str,
        *params: Any,
        timeout: Optional[float] = RPC_CALL_TIMEOUT_SECONDS,
    ) -> Any:
        process = self.process
        if self._closed or process is None or process.stdin is None:
            raise DeltaChatRPCError("Delta Chat RPC process is not running")
        if process.returncode is not None:
            raise DeltaChatRPCError(
                f"Delta Chat RPC process exited with status {process.returncode}"
            )

        self._next_id += 1
        request_id = self._next_id
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        payload = (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": list(params),
                },
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )

        try:
            async with self._write_lock:
                process.stdin.write(payload)
                await process.stdin.drain()
            if timeout is None:
                return await future
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise DeltaChatRPCError(f"Delta Chat RPC call {method} timed out") from exc
        except (BrokenPipeError, ConnectionError) as exc:
            raise DeltaChatRPCError(
                f"Delta Chat RPC write failed for {method}: {exc}"
            ) from exc
        finally:
            self._pending.pop(request_id, None)

    async def _read_stdout(self) -> None:
        process = self.process
        assert process is not None and process.stdout is not None
        failure: Optional[Exception] = None
        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    failure = DeltaChatRPCError("Delta Chat RPC stdout closed")
                    break
                try:
                    response = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    logger.debug("Delta Chat RPC emitted a non-JSON stdout line")
                    continue
                request_id = response.get("id")
                if not isinstance(request_id, int):
                    continue
                future = self._pending.get(request_id)
                if future is None or future.done():
                    continue
                error = response.get("error")
                if error:
                    code = error.get("code", -1) if isinstance(error, dict) else -1
                    message = (
                        error.get("message", str(error))
                        if isinstance(error, dict)
                        else str(error)
                    )
                    future.set_exception(
                        DeltaChatRPCError(f"Delta Chat RPC error {code}: {message}")
                    )
                else:
                    future.set_result(response.get("result"))
        except asyncio.CancelledError:
            return
        except Exception as exc:
            failure = DeltaChatRPCError(f"Delta Chat RPC reader failed: {exc}")
        finally:
            if failure is not None:
                self._fail_pending(failure)

    async def _read_stderr(self) -> None:
        process = self.process
        assert process is not None and process.stderr is not None
        try:
            while True:
                line = await process.stderr.readline()
                if not line:
                    return
                safe = _redact_email(line.decode("utf-8", errors="replace").strip())
                if safe:
                    logger.debug("Delta Chat Core: %s", safe[:1000])
        except asyncio.CancelledError:
            return

    def _fail_pending(self, error: Exception) -> None:
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(error)
        self._pending.clear()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._fail_pending(DeltaChatRPCError("Delta Chat RPC process closed"))

        process = self.process
        if process and process.stdin:
            try:
                process.stdin.close()
                await process.stdin.wait_closed()
            except (BrokenPipeError, ConnectionError):
                pass

        if process and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

        current = asyncio.current_task()
        for task in (self._read_task, self._stderr_task):
            if task and task is not current and not task.done():
                task.cancel()
        for task in (self._read_task, self._stderr_task):
            if task and task is not current:
                try:
                    await task
                except (asyncio.CancelledError, DeltaChatRPCError):
                    pass
        self.process = None


async def _wait_until_ready(rpc: DeltaChatRPC) -> None:
    last_error: Optional[Exception] = None
    for _attempt in range(RPC_READY_ATTEMPTS):
        try:
            await rpc.call("get_system_info", timeout=2.0)
            return
        except DeltaChatRPCError as exc:
            last_error = exc
            await asyncio.sleep(RPC_READY_DELAY_SECONDS)
    raise DeltaChatRPCError(
        f"deltachat-rpc-server did not become ready: {last_error or 'unknown error'}"
    )


async def _apply_profile(
    rpc: DeltaChatRPC,
    account_id: int,
    display_name: str,
    avatar_image: str,
) -> None:
    values: dict[str, str] = {}
    if display_name.strip():
        values["displayname"] = display_name.strip()
    if avatar_image.strip():
        avatar = Path(avatar_image).expanduser()
        if avatar.is_file():
            values["selfavatar"] = str(avatar.resolve())
        else:
            logger.warning(
                "Delta Chat avatar_image does not exist; keeping the current avatar: %s",
                avatar,
            )
    if values:
        await rpc.call("batch_set_config", account_id, values)


async def _create_chatmail_account(
    rpc: DeltaChatRPC,
    relay: str,
    display_name: str,
    avatar_image: str,
) -> tuple[int, str]:
    account_id = int(await rpc.call("add_account"))
    created = False
    try:
        await rpc.call(
            "add_transport_from_qr",
            account_id,
            _chatmail_account_qr(relay),
            timeout=ACCOUNT_CONFIGURE_TIMEOUT_SECONDS,
        )
        created = True
        try:
            await _apply_profile(rpc, account_id, display_name, avatar_image)
        except Exception as exc:
            # The transport and encryption identity already exist. Keep the
            # usable account and let connect() retry cosmetic profile settings.
            logger.warning(
                "Delta Chat account was created but profile setup failed: %s",
                _redact_email(str(exc)),
            )
        address = str(await rpc.call("get_config", account_id, "addr") or "").strip()
        if not address:
            raise DeltaChatRPCError(
                f"Chatmail account on {relay} was created without an address"
            )
        return account_id, address.lower()
    finally:
        if not created:
            try:
                await rpc.call("stop_ongoing_process", account_id, timeout=10.0)
            except Exception:
                pass
            try:
                await rpc.call("remove_account", account_id, timeout=10.0)
            except Exception:
                pass


async def _find_account(rpc: DeltaChatRPC, email: str) -> int:
    accounts = await rpc.call("get_all_accounts") or []
    for account in accounts:
        if not isinstance(account, dict):
            continue
        if str(account.get("kind") or "") != "Configured":
            continue
        if str(account.get("addr") or "").casefold() == email.casefold():
            return int(account.get("id") or 0)
    return 0


def _print_invite(address: str, invite: str) -> None:
    if not invite:
        return
    print()
    print(
        f"Delta Chat invite for {_redact_email(address)} "
        "(Delta Chat: + -> Scan QR Code):"
    )
    print(invite)
    if sys.stdout.isatty():
        try:
            import qrcode

            qr = qrcode.QRCode(border=1)
            qr.add_data(invite)
            qr.make(fit=True)
            qr.print_ascii(invert=True)
        except Exception:
            logger.debug(
                "Could not render Delta Chat invite as terminal QR", exc_info=True
            )
    print()


class DeltaChatAdapter(BasePlatformAdapter):
    """Hermes gateway adapter backed by Delta Chat Core JSON-RPC."""

    enforces_own_access_policy = True
    MAX_MESSAGE_LENGTH = 0

    def __init__(self, config: PlatformConfig, **kwargs: Any):
        super().__init__(config=config, platform=Platform("deltachat"))
        extra = config.extra or {}

        self.email = str(extra.get("email") or "").strip().lower()
        self.display_name = str(extra.get("display_name") or "Hermes Agent").strip()
        self.avatar_image = str(extra.get("avatar_image") or "").strip()
        self.data_dir = Path(
            str(extra.get("data_dir") or get_hermes_home() / "platforms" / "deltachat")
        ).expanduser()
        self.rpc_server_path = str(extra.get("rpc_server_path") or "").strip()
        self.join_invite_link = str(extra.get("join_invite_link") or "").strip()
        self.show_invite_link = _parse_bool(extra.get("show_invite_link"), True)

        self._dm_policy = str(extra.get("dm_policy") or "pairing").strip().lower()
        self._group_policy = (
            str(extra.get("group_policy") or "disabled").strip().lower()
        )
        self._allow_from = {
            item.casefold() for item in _parse_string_list(extra.get("allow_from"))
        }
        group_allow = _parse_string_list(extra.get("group_allow_from"))
        self._group_allow_from = {
            item.casefold()
            for item in (group_allow or _parse_string_list(extra.get("allow_from")))
        }
        self.require_mention = _parse_bool(extra.get("require_mention"), True)
        free_chats = extra.get(
            "free_response_chats", extra.get("free_response_channels")
        )
        self.free_response_chats = set(_parse_string_list(free_chats))

        self._rpc: Optional[DeltaChatRPC] = None
        self._account_id = 0
        self._listen_task: Optional[asyncio.Task] = None

    @property
    def name(self) -> str:
        return "Delta Chat"

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        try:
            account_or_relay, bootstrap = parse_email_setting(self.email)
        except ValueError as exc:
            self._set_fatal_error("deltachat_config_invalid", str(exc), retryable=False)
            return False

        server_path = _configured_rpc_path({"rpc_server_path": self.rpc_server_path})
        if not server_path:
            message = (
                "deltachat-rpc-server was not found. Install it with "
                "`pip install deltachat-rpc-server`, or set "
                "platforms.deltachat.extra.rpc_server_path in config.yaml."
            )
            self._set_fatal_error(
                "deltachat_dependency_missing", message, retryable=False
            )
            return False

        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.data_dir, 0o700)
        except OSError:
            pass

        if not self._acquire_platform_lock(
            "deltachat",
            str(self.data_dir.resolve()),
            "Delta Chat account store",
        ):
            return False

        rpc = DeltaChatRPC(server_path, self.data_dir)
        self._rpc = rpc
        try:
            await rpc.start()
            await _wait_until_ready(rpc)

            if bootstrap:
                _account_id, generated = await _create_chatmail_account(
                    rpc,
                    account_or_relay,
                    self.display_name,
                    self.avatar_image,
                )
                raise DeltaChatBootstrapCreated(generated)

            account_id = await _find_account(rpc, account_or_relay)
            if account_id <= 0 or not await rpc.call("is_configured", account_id):
                raise DeltaChatConfigurationError(
                    f"Account {_redact_email(account_or_relay)} is not configured "
                    f"in {self.data_dir}. Run `hermes gateway setup` to create "
                    "an account, or point data_dir at its existing account store."
                )

            await rpc.call("select_account", account_id)
            await _apply_profile(
                rpc,
                account_id,
                self.display_name,
                self.avatar_image,
            )
            await rpc.call(
                "batch_set_config",
                account_id,
                {"bot": "1", "mdns_enabled": "1"},
            )
            await rpc.call("start_io", account_id)

            self._account_id = account_id
            if self.join_invite_link:
                try:
                    chat_id = int(
                        await rpc.call("secure_join", account_id, self.join_invite_link)
                    )
                    if chat_id > 0:
                        await rpc.call("accept_chat", account_id, chat_id)
                except Exception as exc:
                    logger.warning(
                        "Delta Chat could not join the configured invite: %s", exc
                    )

            self._mark_connected()
            self._listen_task = asyncio.create_task(self._listen_loop())
            logger.info(
                "Delta Chat connected as %s",
                _redact_email(account_or_relay),
            )

            if self.show_invite_link:
                try:
                    invite = str(
                        await rpc.call("get_chat_securejoin_qr_code", account_id, None)
                        or ""
                    )
                    logger.info("Delta Chat secure-join invite: %s", invite)
                    _print_invite(account_or_relay, invite)
                except Exception as exc:
                    logger.warning("Delta Chat could not generate its invite: %s", exc)
            return True
        except DeltaChatBootstrapCreated as created:
            await rpc.close()
            self._rpc = None
            self._release_platform_lock()
            message = (
                f"Created Chatmail account {_redact_email(created.address)}. "
                "Replace the @relay marker in "
                "platforms.deltachat.extra.email with the generated full "
                f"address ({created.address!r}), then start the gateway again."
            )
            logger.error("Delta Chat: %s", message)
            self._set_fatal_error("deltachat_account_created", message, retryable=False)
            return False
        except DeltaChatConfigurationError as exc:
            await rpc.close()
            self._rpc = None
            self._release_platform_lock()
            message = _redact_email(str(exc))
            logger.error("Delta Chat configuration failed: %s", message)
            self._set_fatal_error(
                "deltachat_account_not_configured", message, retryable=False
            )
            return False
        except Exception as exc:
            await rpc.close()
            self._rpc = None
            self._release_platform_lock()
            message = _redact_email(str(exc))
            logger.error("Delta Chat connection failed: %s", message)
            self._set_fatal_error("deltachat_connect_failed", message, retryable=True)
            return False

    async def disconnect(self) -> None:
        self._mark_disconnected()
        listen_task = self._listen_task
        self._listen_task = None
        if listen_task and not listen_task.done():
            listen_task.cancel()

        rpc = self._rpc
        if rpc and self._account_id > 0:
            try:
                await rpc.call("stop_io", self._account_id, timeout=3.0)
            except Exception:
                pass
        if rpc:
            await rpc.close()
        self._rpc = None
        self._account_id = 0

        if listen_task:
            try:
                await listen_task
            except (asyncio.CancelledError, DeltaChatRPCError):
                pass
        self._release_platform_lock()

    async def _listen_loop(self) -> None:
        rpc = self._rpc
        if rpc is None:
            return
        try:
            while self.is_connected:
                message_ids = await rpc.call(
                    "wait_next_msgs", self._account_id, timeout=None
                )
                for message_id in message_ids or []:
                    try:
                        await self._handle_inbound_message(int(message_id))
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        logger.exception(
                            "Delta Chat failed to handle inbound message %s",
                            message_id,
                        )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            if self.is_connected:
                message = _redact_email(str(exc))
                logger.error("Delta Chat listener stopped: %s", message)
                self._set_fatal_error(
                    "deltachat_rpc_disconnected", message, retryable=True
                )
                await self._notify_fatal_error()

    async def _handle_inbound_message(self, message_id: int) -> None:
        rpc = self._require_rpc()
        message = await rpc.call("get_message", self._account_id, message_id)
        if not isinstance(message, dict):
            return

        text = str(message.get("text") or "").strip()
        file_path = str(message.get("file") or "").strip()
        if message.get("isInfo") or (not text and not file_path):
            return

        sender_id = int(message.get("fromId") or 0)
        if sender_id <= 0:
            return
        sender = await rpc.call("get_contact", self._account_id, sender_id)
        if not isinstance(sender, dict):
            return
        sender_address = str(sender.get("address") or "").strip().lower()
        if not sender_address or sender_address.casefold() == self.email.casefold():
            return

        chat_id = int(message.get("chatId") or 0)
        if chat_id <= 0:
            return
        chat = await rpc.call("get_full_chat_by_id", self._account_id, chat_id)
        if not isinstance(chat, dict) or chat.get("isDeviceChat"):
            return
        is_group = str(chat.get("chatType") or "Single") != "Single"

        if not self._sender_allowed(sender_address, is_group=is_group):
            logger.info(
                "Delta Chat ignored sender %s under %s policy",
                _redact_email(sender_address),
                self._group_policy if is_group else self._dm_policy,
            )
            return

        if (
            is_group
            and self.require_mention
            and str(chat_id) not in self.free_response_chats
        ):
            if not mentions_bot(text, self.display_name, self.email):
                return
            text = _strip_bot_mention(text, self.display_name, self.email)

        media_urls: list[str] = []
        media_types: list[str] = []
        message_type = MessageType.TEXT
        if file_path:
            cached = await self._cache_inbound_file(message)
            if cached is not None:
                media_urls.append(cached.path)
                media_types.append(cached.media_type)
                message_type = _message_type_for_media(cached.kind)

        if not text and not media_urls:
            return

        sender_name = (
            str(sender.get("displayName") or "").strip()
            or str(sender.get("name") or "").strip()
            or sender_address
        )
        source = self.build_source(
            chat_id=str(chat_id),
            chat_name=str(chat.get("name") or sender_name),
            chat_type="group" if is_group else "dm",
            user_id=sender_address,
            user_name=sender_name,
            message_id=str(message.get("id") or message_id),
        )

        try:
            timestamp = datetime.fromtimestamp(
                int(message.get("timestamp") or 0), tz=timezone.utc
            )
        except (OSError, OverflowError, TypeError, ValueError):
            timestamp = datetime.now(tz=timezone.utc)

        event = MessageEvent(
            source=source,
            text=text,
            message_type=message_type,
            message_id=str(message.get("id") or message_id),
            media_urls=media_urls,
            media_types=media_types,
            timestamp=timestamp,
            raw_message={
                "id": int(message.get("id") or message_id),
                "chat_id": chat_id,
            },
        )
        await self.handle_message(event)

        if chat.get("isContactRequest"):
            await rpc.call("accept_chat", self._account_id, chat_id)
        await rpc.call(
            "markseen_msgs",
            self._account_id,
            [int(message.get("id") or message_id)],
        )

    async def _cache_inbound_file(self, message: dict):
        path = Path(str(message.get("file") or "")).expanduser()
        if not path.is_file():
            logger.warning("Delta Chat attachment is missing from its account store")
            return None
        filename = str(message.get("fileName") or path.name)
        mime_type = str(message.get("fileMime") or "")
        if not mime_type:
            mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        try:
            size = path.stat().st_size
            validate_inbound_media_size(size, media_type="Delta Chat attachment")
            data = await asyncio.to_thread(path.read_bytes)
            return await asyncio.to_thread(
                cache_media_bytes,
                data,
                filename=filename,
                mime_type=mime_type,
                default_kind="audio"
                if path.suffix.lower() in _AUDIO_EXTENSIONS
                else None,
            )
        except Exception as exc:
            logger.warning(
                "Delta Chat could not cache inbound attachment %s: %s",
                filename,
                exc,
            )
            return None

    def _sender_allowed(self, sender: str, *, is_group: bool) -> bool:
        policy = self._group_policy if is_group else self._dm_policy
        allowed = self._group_allow_from if is_group else self._allow_from
        if policy == "disabled":
            return False
        if policy == "allowlist":
            folded = sender.casefold()
            return "*" in allowed or folded in allowed
        # ``pairing`` must reach GatewayRunner so it can issue a pairing code.
        # ``open`` still passes through GatewayRunner's global fail-closed auth.
        return policy == "open" or (policy == "pairing" and not is_group)

    def _require_rpc(self) -> DeltaChatRPC:
        if self._rpc is None or not self.is_connected:
            raise DeltaChatRPCError("Delta Chat adapter is not connected")
        return self._rpc

    async def _resolve_chat_id(self, target: str) -> int:
        rpc = self._require_rpc()
        value = str(target or "").strip()
        if not value:
            raise ValueError("Delta Chat target is empty")
        for prefix in ("chat:", "chatid:", "chat_id:", "deltachat:"):
            if value.lower().startswith(prefix):
                value = value[len(prefix) :].strip()
                break
        if value.isdigit() and int(value) > 0:
            return int(value)

        address = (
            value[7:].split("?", 1)[0] if value.lower().startswith("mailto:") else value
        )
        _name, parsed = parseaddr(address)
        if parsed and parsed.count("@") == 1:
            contact_id = await rpc.call(
                "lookup_contact_id_by_addr", self._account_id, parsed.lower()
            )
            if not contact_id:
                contact_id = await rpc.call(
                    "create_contact", self._account_id, parsed.lower(), None
                )
            chat_id = await rpc.call(
                "get_chat_id_by_contact_id", self._account_id, int(contact_id)
            )
            if not chat_id:
                chat_id = await rpc.call(
                    "create_chat_by_contact_id", self._account_id, int(contact_id)
                )
            return int(chat_id)

        queries = list(
            dict.fromkeys(
                query
                for query in (value, value.strip("<>"), value.removeprefix("@"))
                if query
            )
        )
        for query in queries:
            contacts = await rpc.call("get_contacts", self._account_id, 0, query) or []
            unique_contacts = {
                int(contact.get("id") or 0): contact
                for contact in contacts
                if isinstance(contact, dict) and int(contact.get("id") or 0) > 0
            }
            exact_contacts = [
                contact
                for contact in unique_contacts.values()
                if self._contact_matches(contact, query)
            ]
            contact_matches = exact_contacts or (
                list(unique_contacts.values()) if len(unique_contacts) == 1 else []
            )
            if len(contact_matches) == 1:
                return await self._chat_id_for_contact(int(contact_matches[0]["id"]))

            chat_ids = (
                await rpc.call("get_chatlist_entries", self._account_id, 0, query, None)
                or []
            )
            chat_matches = []
            for chat_id in dict.fromkeys(int(item) for item in chat_ids):
                chat = await rpc.call("get_full_chat_by_id", self._account_id, chat_id)
                if (
                    isinstance(chat, dict)
                    and not chat.get("isDeviceChat")
                    and str(chat.get("name") or "").strip().casefold()
                    == query.casefold()
                ):
                    chat_matches.append(chat)
            if len(chat_matches) == 1:
                return int(chat_matches[0]["id"])
            if len(unique_contacts) > 1 or len(chat_matches) > 1:
                raise ValueError(f"Delta Chat recipient {value!r} is ambiguous")

        raise ValueError(f"Delta Chat recipient {value!r} is unknown")

    async def _chat_id_for_contact(self, contact_id: int) -> int:
        rpc = self._require_rpc()
        chat_id = await rpc.call(
            "get_chat_id_by_contact_id", self._account_id, contact_id
        )
        if not chat_id:
            chat_id = await rpc.call(
                "create_chat_by_contact_id", self._account_id, contact_id
            )
        return int(chat_id)

    @staticmethod
    def _contact_matches(contact: dict, query: str) -> bool:
        address = str(contact.get("address") or "")
        aliases = {
            str(contact.get("displayName") or ""),
            str(contact.get("name") or ""),
            str(contact.get("nameAndAddr") or ""),
            address,
        }
        if "@" in address:
            local = address.split("@", 1)[0]
            aliases.update({local, f"@{local}"})
        folded = query.strip().casefold()
        return any(
            alias.strip().casefold() == folded for alias in aliases if alias.strip()
        )

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not str(content or "").strip():
            return SendResult(success=True)
        try:
            resolved = await self._resolve_chat_id(chat_id)
            quoted_id = int(reply_to) if str(reply_to or "").isdigit() else None
            result = await self._require_rpc().call(
                "misc_send_msg",
                self._account_id,
                resolved,
                content,
                None,
                None,
                None,
                quoted_id,
            )
            message_id = result[0] if isinstance(result, list) and result else None
            return SendResult(
                success=True,
                message_id=str(message_id) if message_id is not None else None,
                raw_response=result,
            )
        except Exception as exc:
            return SendResult(
                success=False, error=_redact_email(str(exc)), retryable=True
            )

    async def _send_file(
        self,
        chat_id: str,
        file_path: str,
        caption: str = "",
        *,
        file_name: Optional[str] = None,
        viewtype: str = "",
    ) -> SendResult:
        safe_path = self.validate_media_delivery_path(file_path)
        if not safe_path:
            return SendResult(
                success=False, error="File path is not allowed or does not exist"
            )
        try:
            resolved = await self._resolve_chat_id(chat_id)
            data = {
                "text": caption or "",
                "file": str(Path(safe_path).resolve()),
                "filename": file_name or Path(safe_path).name,
            }
            # An empty string is not a valid Delta Chat Viewtype. Omit the
            # field so Core can infer image/GIF/video/file from the payload;
            # voice notes are the one format Hermes must force explicitly.
            if viewtype:
                data["viewtype"] = viewtype
            result = await self._require_rpc().call(
                "send_msg",
                self._account_id,
                resolved,
                data,
            )
            return SendResult(success=True, message_id=str(result), raw_response=result)
        except Exception as exc:
            return SendResult(
                success=False, error=_redact_email(str(exc)), retryable=True
            )

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        if image_url.startswith(("http://", "https://")):
            from gateway.platforms.base import cache_image_from_url

            try:
                image_url = await cache_image_from_url(image_url)
            except Exception as exc:
                return SendResult(success=False, error=str(exc))
        return await self._send_file(chat_id, image_url, caption or "")

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        return await self._send_file(chat_id, image_path, caption or "")

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        return await self._send_file(
            chat_id, file_path, caption or "", file_name=file_name
        )

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        return await self._send_file(
            chat_id, audio_path, caption or "", viewtype="Voice"
        )

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        return await self._send_file(chat_id, video_path, caption or "")

    async def send_animation(
        self,
        chat_id: str,
        animation_url: str,
        caption: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        if animation_url.startswith(("http://", "https://")):
            from gateway.platforms.base import cache_image_from_url

            try:
                animation_url = await cache_image_from_url(animation_url, ext=".gif")
            except Exception as exc:
                return SendResult(success=False, error=str(exc))
        return await self._send_file(chat_id, animation_url, caption or "")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Delta Chat has no real-time typing indicator over email."""

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        try:
            resolved = await self._resolve_chat_id(chat_id)
            chat = await self._require_rpc().call(
                "get_full_chat_by_id", self._account_id, resolved
            )
            return {
                "chat_id": str(resolved),
                "name": str((chat or {}).get("name") or resolved),
                "type": (
                    "dm"
                    if str((chat or {}).get("chatType") or "Single") == "Single"
                    else "group"
                ),
            }
        except Exception:
            return {"chat_id": str(chat_id), "name": str(chat_id), "type": "dm"}


def check_requirements() -> bool:
    """Return whether a configured or PATH-visible RPC server exists."""

    return _configured_rpc_path() is not None


def validate_config(config: PlatformConfig) -> bool:
    extra = getattr(config, "extra", {}) or {}
    try:
        parse_email_setting(str(extra.get("email") or ""))
    except ValueError:
        return False
    return True


def is_connected(config: PlatformConfig) -> bool:
    extra = getattr(config, "extra", {}) or {}
    enabled = bool(getattr(config, "enabled", False))
    if not extra:
        raw = _raw_platform_config()
        extra = raw.get("extra") if isinstance(raw.get("extra"), dict) else raw
        enabled = _parse_bool(raw.get("enabled"), False)
    if not enabled:
        return False
    try:
        parse_email_setting(str(extra.get("email") or ""))
    except ValueError:
        return False
    # A valid ``@relay`` marker is intentionally startable: the adapter turns
    # it into a configured Chatmail account, then stops with the generated
    # address the operator must persist. This preserves PicoClaw's manual
    # bootstrap path in addition to Hermes' one-step setup wizard.
    return True


def _apply_yaml_config(_root_config: dict, platform_config: dict) -> None:
    """Bridge a YAML home channel to cron's internal delivery lookup.

    Delta Chat has no user-facing environment-variable configuration. Cron's
    plugin registry currently resolves default targets through an env-var
    name, so this hook supplies that implementation detail from ``config.yaml``.
    """

    home = platform_config.get("home_channel")
    if isinstance(home, dict) and home.get("chat_id"):
        os.environ["DELTACHAT_HOME_CHANNEL"] = str(home["chat_id"])
    else:
        # This is not a supported user override. Clear a value left by an
        # earlier in-process config load when the YAML home channel is removed.
        os.environ.pop("DELTACHAT_HOME_CHANNEL", None)
    return None


async def _standalone_send(
    pconfig: PlatformConfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[Any]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """One-shot delivery for an out-of-process cron or ``hermes send`` call."""

    adapter = DeltaChatAdapter(pconfig)
    if not await adapter.connect():
        return {"error": adapter.fatal_error_message or "Delta Chat connection failed"}
    try:
        last_result = (
            await adapter.send(chat_id, message)
            if message
            else SendResult(success=True)
        )
        if not last_result.success:
            return {"error": last_result.error or "Delta Chat send failed"}
        for item in media_files or []:
            path, is_voice = item if isinstance(item, (tuple, list)) else (item, False)
            if is_voice and not force_document:
                last_result = await adapter.send_voice(chat_id, str(path))
            else:
                last_result = await adapter.send_document(chat_id, str(path))
            if not last_result.success:
                return {"error": last_result.error or "Delta Chat media send failed"}
        return {
            "success": True,
            "platform": "deltachat",
            "chat_id": chat_id,
            "message_id": last_result.message_id,
        }
    finally:
        await adapter.disconnect()


async def _setup_account(
    server_path: str,
    data_dir: Path,
    email_or_relay: str,
    display_name: str,
    avatar_image: str,
) -> tuple[str, str]:
    data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(data_dir, 0o700)
    except OSError:
        pass
    rpc = DeltaChatRPC(server_path, data_dir)
    await rpc.start()
    try:
        await _wait_until_ready(rpc)
        account_or_relay, bootstrap = parse_email_setting(email_or_relay)
        if bootstrap:
            account_id, address = await _create_chatmail_account(
                rpc, account_or_relay, display_name, avatar_image
            )
        else:
            address = account_or_relay
            account_id = await _find_account(rpc, address)
            if account_id <= 0 or not await rpc.call("is_configured", account_id):
                raise DeltaChatRPCError(
                    f"Account {_redact_email(address)} was not found in {data_dir}"
                )
            await _apply_profile(rpc, account_id, display_name, avatar_image)
        try:
            await rpc.call(
                "batch_set_config",
                account_id,
                {"bot": "1", "mdns_enabled": "1"},
            )
        except Exception as exc:
            # connect() applies the same settings again. Do not lose a newly
            # generated address merely because this optional setup pass failed.
            logger.warning(
                "Delta Chat account created, but bot settings will be retried "
                "when the gateway starts: %s",
                _redact_email(str(exc)),
            )
        try:
            invite = str(
                await rpc.call("get_chat_securejoin_qr_code", account_id, None) or ""
            )
        except Exception as exc:
            # Gateway startup also prints the invite, so account setup is still
            # successful when QR generation is temporarily unavailable.
            logger.warning(
                "Delta Chat invite generation will be retried on gateway start: %s",
                _redact_email(str(exc)),
            )
            invite = ""
        return address, invite
    finally:
        await rpc.close()


def interactive_setup() -> None:
    """Create/select a Delta Chat account and persist config-only settings."""

    from hermes_cli.cli_output import (
        print_header,
        print_info,
        print_success,
        print_warning,
        prompt,
        prompt_yes_no,
    )
    from utils import atomic_roundtrip_yaml_update

    print_header("Delta Chat")
    print_info(
        "Hermes uses deltachat-rpc-server locally; Delta Chat Core stores the "
        "mailbox credentials and end-to-end encryption keys."
    )
    print_info("Install it first with: pip install deltachat-rpc-server")
    print()

    raw = _raw_platform_config()
    existing_extra = (
        dict(raw.get("extra")) if isinstance(raw.get("extra"), dict) else {}
    )
    for key in (
        "allow_from",
        "group_allow_from",
        "dm_policy",
        "group_policy",
        "require_mention",
    ):
        if key in raw:
            existing_extra.setdefault(key, raw[key])
    discovered = _configured_rpc_path(existing_extra)
    server_input = prompt(
        "Path to deltachat-rpc-server",
        default=discovered or "deltachat-rpc-server",
    ).strip()
    server_path = shutil.which(server_input) or str(Path(server_input).expanduser())
    if not Path(server_path).is_file():
        print_warning(
            "deltachat-rpc-server was not found. Install it, then run "
            "`hermes gateway setup` again."
        )
        return

    default_data_dir = str(
        Path(
            str(
                existing_extra.get("data_dir")
                or get_hermes_home() / "platforms" / "deltachat"
            )
        ).expanduser()
    )
    data_dir = Path(
        prompt("Delta Chat account data directory", default=default_data_dir).strip()
        or default_data_dir
    ).expanduser()
    display_name = prompt(
        "Bot display name",
        default=str(existing_extra.get("display_name") or "Hermes Agent"),
    ).strip()

    existing_email = str(existing_extra.get("email") or "").strip()
    create_new = not existing_email or prompt_yes_no(
        "Create a new Chatmail account?", False
    )
    if create_new:
        relay = (
            prompt(
                f"Chatmail relay (see {CHATMAIL_RELAYS_URL})",
                default=DEFAULT_CHATMAIL_RELAY,
            )
            .strip()
            .lstrip("@")
        )
        email_or_relay = f"@{relay}"
    else:
        email_or_relay = prompt(
            "Existing account email in that data directory",
            default=existing_email,
        ).strip()

    print_info("Preparing the Delta Chat account (this can take up to 90 seconds)...")
    try:
        address, invite = asyncio.run(
            _setup_account(
                server_path,
                data_dir,
                email_or_relay,
                display_name,
                str(existing_extra.get("avatar_image") or ""),
            )
        )
    except Exception as exc:
        print_warning(f"Delta Chat account setup failed: {_redact_email(str(exc))}")
        return

    allow_raw = prompt(
        "Allowed sender emails (comma-separated; blank uses secure DM pairing)",
        default=", ".join(_parse_string_list(existing_extra.get("allow_from"))),
    ).strip()
    allow_from = _parse_string_list(allow_raw)

    config_path = get_hermes_home() / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    values = {
        "platforms.deltachat.enabled": True,
        "platforms.deltachat.typing_indicator": False,
        "platforms.deltachat.dm_policy": "allowlist" if allow_from else "pairing",
        "platforms.deltachat.group_policy": "disabled",
        "platforms.deltachat.require_mention": True,
        "platforms.deltachat.allow_from": allow_from,
        "platforms.deltachat.extra.email": address,
        "platforms.deltachat.extra.display_name": display_name or "Hermes Agent",
        "platforms.deltachat.extra.data_dir": str(data_dir.resolve()),
        "platforms.deltachat.extra.rpc_server_path": str(Path(server_path).resolve()),
        "platforms.deltachat.extra.show_invite_link": True,
    }
    try:
        for key, value in values.items():
            atomic_roundtrip_yaml_update(config_path, key, value)
        os.chmod(config_path, 0o600)
    except Exception as exc:
        print_warning(f"Could not save Delta Chat configuration: {exc}")
        return

    print_success(
        f"Delta Chat configured as {_redact_email(address)} in "
        f"{display_hermes_home()}/config.yaml"
    )
    _print_invite(address, invite)
    print_info("Start the gateway with: hermes gateway")


def register(ctx) -> None:
    """Plugin entry point called by Hermes' bundled-plugin discovery."""

    ctx.register_platform(
        name="deltachat",
        label="Delta Chat",
        adapter_factory=lambda cfg: DeltaChatAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=[],
        install_hint="pip install deltachat-rpc-server",
        setup_fn=interactive_setup,
        apply_yaml_config_fn=_apply_yaml_config,
        cron_deliver_env_var="DELTACHAT_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        max_message_length=0,
        emoji="📨",
        pii_safe=True,
        allow_update_command=True,
        platform_hint=(
            "You are chatting over Delta Chat, an email-based end-to-end "
            "encrypted messenger. Keep replies conversational. Delta Chat has "
            "no typing indicator or practical message-length limit. Basic "
            "Markdown may render, but do not rely on elaborate formatting. "
            "You can receive and send native images, voice notes, video, and "
            "arbitrary file attachments."
        ),
    )
