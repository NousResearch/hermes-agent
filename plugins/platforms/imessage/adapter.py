"""Native macOS iMessage adapter backed by the local ``imsg`` JSON-RPC bridge."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import mimetypes
import os
import re
import shutil
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms._shared import apply_yaml_bridge as _apply_yaml_bridge
from gateway.platforms._shared import extra_or_secret as _extra_or_secret
from gateway.platforms._shared import get_scoped_secret as _get_scoped_secret
from gateway.platforms._shared import seed_extra_from_env as _seed_extra_from_env
from gateway.platforms._shared import send_error
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.helpers import (
    cancel_task,
    compile_mention_patterns,
    strip_markdown,
)
from gateway.session import SessionSource
from utils import atomic_json_write

logger = logging.getLogger(__name__)

_DEFAULT_CLI = "/opt/homebrew/bin/imsg"
_DEFAULT_DB = "~/Library/Messages/chat.db"
_DEFAULT_HISTORY_LIMIT = 100
_STATE_SEEN_CAP = 5000
_RPC_TIMEOUT = 15.0
_INBOUND_RETRY_BASE_SECONDS = 1.0
_RESTART_BASE_SECONDS = 1.0
_MAX_RPC_LINE = 4 * 1024 * 1024
_DEFAULT_MENTION_PATTERNS = [r"(?<!\w)@?jarvis\b", r"(?<!\w)@?hermes\b"]


def _state_path() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "runtime" / "imessage-imsg-state.json"


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _list_setting(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.split(",")
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_handle(value: Any) -> str:
    raw = str(value or "").strip().lower()
    for prefix in ("imessage:", "sms:", "tel:", "mailto:"):
        if raw.startswith(prefix):
            raw = raw[len(prefix) :]
            break
    return raw


def _parse_timestamp(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)


def _message_type(mime: str) -> MessageType:
    mime = (mime or "").lower()
    if mime.startswith("image/"):
        return MessageType.PHOTO
    if mime.startswith("video/"):
        return MessageType.VIDEO
    if mime.startswith("audio/"):
        return MessageType.AUDIO
    return MessageType.DOCUMENT


def _rpc_error_message(error: Any) -> str:
    """Return a bounded operator-safe RPC error, with an actionable FDA diagnostic."""
    if not isinstance(error, dict):
        return "imsg RPC error"
    message = str(error.get("message") or "imsg RPC error")
    detail = str(error.get("data") or "")
    combined = f"{message} {detail}".lower()
    if "full disk access" in combined and "chat.db" in combined:
        return "imsg cannot access Messages chat.db; grant Full Disk Access to the Hermes gateway launcher"
    return (f"{message}: {detail}" if detail else message)[:1000]


class ImsgRpcError(RuntimeError):
    pass


class ImsgRpcClient:
    """Line-delimited JSON-RPC client with deterministic child-process cleanup."""

    def __init__(
        self,
        *,
        cli_path: str,
        db_path: str,
        on_notification: Callable[[str, Any], None],
    ) -> None:
        self.cli_path = cli_path
        self.db_path = db_path
        self.on_notification = on_notification
        self.process: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._pending: Dict[int, asyncio.Future] = {}
        self._next_id = 1
        self._closed = asyncio.Event()
        self.last_error = ""

    async def start(self) -> None:
        args = [self.cli_path, "rpc", "--json"]
        if self.db_path:
            args += ["--db", self.db_path]
        self._closed.clear()
        try:
            self.process = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=_MAX_RPC_LINE,
            )
        except OSError as exc:
            raise ImsgRpcError(f"could not start imsg: {exc}") from exc
        self._reader_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())

    async def _read_stdout(self) -> None:
        assert self.process and self.process.stdout
        try:
            while line := await self.process.stdout.readline():
                if len(line) > _MAX_RPC_LINE:
                    logger.warning("iMessage: discarded oversized imsg RPC frame")
                    continue
                try:
                    payload = json.loads(line)
                except (UnicodeDecodeError, ValueError):
                    logger.warning("iMessage: discarded malformed imsg RPC frame")
                    continue
                if payload.get("id") is not None:
                    future = self._pending.pop(int(payload["id"]), None)
                    if future and not future.done():
                        if payload.get("error"):
                            future.set_exception(
                                ImsgRpcError(_rpc_error_message(payload["error"]))
                            )
                        else:
                            future.set_result(payload.get("result"))
                elif payload.get("method"):
                    try:
                        self.on_notification(
                            str(payload["method"]), payload.get("params")
                        )
                    except Exception:
                        logger.exception("iMessage: notification callback failed")
        finally:
            self._finish_pending(
                ImsgRpcError(self.last_error or "imsg RPC process closed")
            )
            self._closed.set()

    async def _read_stderr(self) -> None:
        assert self.process and self.process.stderr
        while line := await self.process.stderr.readline():
            text = line.decode("utf-8", errors="replace").strip()[:1000]
            if not text:
                continue
            if "full disk access" in text.lower() and "chat.db" in text.lower():
                self.last_error = "imsg cannot access Messages chat.db; grant Full Disk Access to the Hermes gateway launcher"
            else:
                self.last_error = text
            logger.warning("iMessage imsg: %s", self.last_error)

    def _finish_pending(self, exc: Exception) -> None:
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(exc)
        self._pending.clear()

    async def request(
        self, method: str, params: Optional[dict] = None, timeout: float = _RPC_TIMEOUT
    ) -> Any:
        if (
            not self.process
            or not self.process.stdin
            or self.process.returncode is not None
        ):
            raise ImsgRpcError("imsg RPC is not running")
        request_id = self._next_id
        self._next_id += 1
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        frame = (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params or {},
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        try:
            self.process.stdin.write(frame.encode("utf-8"))
            await self.process.stdin.drain()
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as exc:
            self._pending.pop(request_id, None)
            raise ImsgRpcError(f"imsg RPC timeout ({method})") from exc

    async def wait_closed(self) -> None:
        await self._closed.wait()

    async def stop(self) -> None:
        process = self.process
        self.process = None
        if process and process.stdin:
            process.stdin.close()
        if process and process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                process.terminate()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                if process.returncode is None:
                    process.kill()
                    await process.wait()
        await cancel_task(self._reader_task)
        await cancel_task(self._stderr_task)
        self._reader_task = self._stderr_task = None
        self._finish_pending(ImsgRpcError("imsg RPC stopped"))
        self._closed.set()


class ImsgAdapter(BasePlatformAdapter):
    """Local Apple Messages transport using the current macOS Messages identity."""

    def __init__(self, config: PlatformConfig, *, rpc_factory=ImsgRpcClient) -> None:
        super().__init__(config=config, platform=Platform("imessage"))
        extra = getattr(config, "extra", {}) or {}
        self._extra = extra
        self.cli_path = str(
            _extra_or_secret(extra, "cli_path", "IMESSAGE_CLI_PATH", _DEFAULT_CLI)
        )
        self.db_path = str(
            Path(
                str(_extra_or_secret(extra, "db_path", "IMESSAGE_DB_PATH", _DEFAULT_DB))
            ).expanduser()
        )
        try:
            self.history_limit = max(
                0, min(500, int(extra.get("history_limit", _DEFAULT_HISTORY_LIMIT)))
            )
        except (TypeError, ValueError):
            self.history_limit = _DEFAULT_HISTORY_LIMIT
        self.require_mention = _truthy(
            _extra_or_secret(
                extra,
                "require_mention",
                "IMESSAGE_REQUIRE_MENTION",
                True,
                blank_is_unset=False,
            ),
            True,
        )
        self._mentions = compile_mention_patterns(
            extra.get("mention_patterns"),
            log_prefix="imessage",
            defaults=_DEFAULT_MENTION_PATTERNS,
        )
        env_dm = _get_scoped_secret("IMESSAGE_DM_ALLOWED_USERS", None)
        env_group = _get_scoped_secret("IMESSAGE_GROUP_ALLOWED_USERS", None)
        env_legacy = _get_scoped_secret("IMESSAGE_ALLOWED_USERS", None)
        if env_dm is not None or env_group is not None:
            dm_values = _list_setting(env_dm)
            group_values = _list_setting(env_group)
            policy_configured = True
        elif env_legacy is not None:
            dm_values = group_values = _list_setting(env_legacy)
            policy_configured = True
        elif "dm_allowed_users" in extra or "group_allowed_users" in extra:
            dm_values = _list_setting(extra.get("dm_allowed_users"))
            group_values = _list_setting(extra.get("group_allowed_users"))
            policy_configured = True
        elif "allowed_users" in extra:
            dm_values = group_values = _list_setting(extra.get("allowed_users"))
            policy_configured = True
        else:
            dm_values = group_values = []
            policy_configured = False
        self._dm_policy_configured = policy_configured
        self._group_policy_configured = policy_configured
        self._allowed_users = {_normalize_handle(v) for v in dm_values}
        self._group_allowed_users = {_normalize_handle(v) for v in group_values}
        self._allowed_chats = set(_list_setting(extra.get("allowed_chats")))
        self._rpc_factory = rpc_factory
        self._rpc: Optional[ImsgRpcClient] = None
        self._subscription: Optional[str] = None
        self._supervisor_task: Optional[asyncio.Task] = None
        self._notification_task: Optional[asyncio.Task] = None
        self._notification_queue: asyncio.Queue = asyncio.Queue()
        self._stopping = False
        self._lifecycle_lock = asyncio.Lock()
        self._rpc_candidate: Optional[ImsgRpcClient] = None
        self._intake_lock = asyncio.Lock()
        self._last_error = ""
        self._restarts = 0
        self._last_rowid = 0
        self._cursor_initialized = False
        self._seen: "OrderedDict[str, None]" = OrderedDict()
        self._load_state()

    @property
    def name(self) -> str:
        return "iMessage (imsg)"

    def _load_state(self) -> None:
        try:
            data = json.loads(_state_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        try:
            self._last_rowid = max(0, int(data.get("last_rowid", 0)))
        except (TypeError, ValueError):
            self._last_rowid = 0
        self._cursor_initialized = bool(
            data.get("cursor_initialized", self._last_rowid > 0)
        )
        guids = data.get("seen_guids")
        if isinstance(guids, list):
            self._seen = OrderedDict(
                (str(guid), None) for guid in guids[-_STATE_SEEN_CAP:] if guid
            )

    def _save_state(self) -> None:
        atomic_json_write(
            _state_path(),
            {
                "version": 1,
                "cursor_initialized": self._cursor_initialized,
                "last_rowid": self._last_rowid,
                "seen_guids": list(self._seen)[-_STATE_SEEN_CAP:],
            },
            mode=0o600,
        )

    def _commit(self, guid: str, rowid: int) -> None:
        self._last_rowid = max(self._last_rowid, rowid)
        if guid:
            self._seen.pop(guid, None)
            self._seen[guid] = None
            while len(self._seen) > _STATE_SEEN_CAP:
                self._seen.popitem(last=False)
        self._save_state()

    def _on_notification(self, method: str, params: Any) -> None:
        self._notification_queue.put_nowait((method, params))

    async def _consume_notifications(self) -> None:
        """Process events in bridge order; failures block later cursor advancement."""
        while True:
            method, params = await self._notification_queue.get()
            delay = _INBOUND_RETRY_BASE_SECONDS
            try:
                while True:
                    try:
                        await self._handle_notification(method, params)
                        break
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        self._last_error = str(exc)[:1000]
                        logger.warning(
                            "iMessage: retrying ordered inbound handoff in %.0fs: %s",
                            delay,
                            self._last_error,
                        )
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, 30.0)
            finally:
                self._notification_queue.task_done()

    async def _new_rpc(self):
        rpc = self._rpc_factory(
            cli_path=self.cli_path,
            db_path=self.db_path,
            on_notification=self._on_notification,
        )
        async with self._lifecycle_lock:
            if self._stopping:
                raise ImsgRpcError("iMessage adapter is stopping")
            self._rpc_candidate = rpc
        try:
            await rpc.start()
            if not self._cursor_initialized:
                await self._initialize_cursor(rpc)
            params: Dict[str, Any] = {
                "attachments": True,
                "include_reactions": False,
                "since_rowid": self._last_rowid if self._last_rowid > 0 else -1,
            }
            result = await rpc.request("watch.subscribe", params, timeout=_RPC_TIMEOUT)
            subscription = (
                result.get("subscription") if isinstance(result, dict) else None
            )
            if not subscription:
                raise ImsgRpcError("imsg watch.subscribe returned no subscription")
        except BaseException:
            async with self._lifecycle_lock:
                owns_candidate = self._rpc_candidate is rpc
                if owns_candidate:
                    self._rpc_candidate = None
            if owns_candidate:
                await rpc.stop()
            raise
        async with self._lifecycle_lock:
            if self._stopping or self._rpc_candidate is not rpc:
                owns_candidate = self._rpc_candidate is rpc
                if owns_candidate:
                    self._rpc_candidate = None
            else:
                self._rpc_candidate = None
                self._rpc, self._subscription = rpc, str(subscription)
                return
        if owns_candidate:
            await rpc.stop()
        raise ImsgRpcError("iMessage adapter stopped during bridge startup")

    async def _initialize_cursor(self, rpc: ImsgRpcClient) -> None:
        """Snapshot the current tail once, without replaying pre-install history."""
        cursor = 0
        while True:
            result = await rpc.request(
                "messages.after",
                {
                    "since_rowid": cursor,
                    "limit": 500,
                    "attachments": False,
                    "include_reactions": False,
                },
                timeout=_RPC_TIMEOUT,
            )
            if not isinstance(result, dict):
                raise ImsgRpcError("imsg messages.after returned an invalid baseline")
            try:
                next_cursor = int(result.get("next_rowid", cursor))
            except (TypeError, ValueError) as exc:
                raise ImsgRpcError(
                    "imsg messages.after returned an invalid cursor"
                ) from exc
            has_more = result.get("has_more") is True
            if next_cursor < cursor or (has_more and next_cursor == cursor):
                raise ImsgRpcError("imsg messages.after baseline did not advance")
            cursor = next_cursor
            if not has_more:
                break
        self._last_rowid = cursor
        self._cursor_initialized = True
        self._save_state()

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not Path(self.cli_path).is_file() and not shutil.which(self.cli_path):
            self._set_fatal_error(
                "cli_missing",
                f"imsg binary not found: {self.cli_path}",
                retryable=False,
            )
            return False
        if not self._acquire_platform_lock(
            "imessage", self.db_path, f"iMessage database {self.db_path}"
        ):
            return False
        async with self._lifecycle_lock:
            self._stopping = False
        try:
            await self._new_rpc()
        except Exception as exc:
            self._last_error = str(exc)
            self._set_fatal_error("connect_failed", self._last_error, retryable=True)
            self._release_platform_lock()
            return False
        async with self._lifecycle_lock:
            if self._stopping or self._rpc is None:
                self._release_platform_lock()
                return False
            self._mark_connected()
            self._wire_plugin_handlers(None)
            if self._notification_task is None or self._notification_task.done():
                self._notification_task = asyncio.create_task(
                    self._consume_notifications()
                )
            if hasattr(self._rpc, "wait_closed"):
                self._supervisor_task = asyncio.create_task(self._supervise())
        logger.info(
            "iMessage: watching Messages via imsg (checkpoint rowid=%d)",
            self._last_rowid,
        )
        return True

    async def _supervise(self) -> None:
        delay = _RESTART_BASE_SECONDS
        while not self._stopping:
            current = self._rpc
            if current is not None:
                await current.wait_closed()
            if self._stopping:
                return
            self._mark_disconnected()
            self._rpc = None
            self._subscription = None
            if current is not None:
                self._last_error = (
                    getattr(current, "last_error", "") or "imsg RPC exited"
                )
            while not self._stopping and self._rpc is None:
                logger.warning("iMessage: bridge stopped; restarting in %.0fs", delay)
                await asyncio.sleep(delay)
                if self._stopping:
                    return
                try:
                    await self._new_rpc()
                except Exception as exc:
                    self._last_error = str(exc)
                    delay = min(delay * 2, 30.0)
                    continue
                self._restarts += 1
                delay = _RESTART_BASE_SECONDS
                self._mark_connected()
                logger.info("iMessage: imsg bridge restarted")

    async def disconnect(self) -> None:
        async with self._lifecycle_lock:
            self._stopping = True
            self._mark_disconnected()
            rpc, subscription = self._rpc, self._subscription
            candidate = self._rpc_candidate
            self._rpc = None
            self._subscription = None
            self._rpc_candidate = None
        await cancel_task(self._supervisor_task)
        self._supervisor_task = None
        if rpc and subscription:
            with contextlib.suppress(Exception):
                await rpc.request(
                    "watch.unsubscribe", {"subscription": subscription}, timeout=3.0
                )
        if rpc:
            await rpc.stop()
        if candidate is not None and candidate is not rpc:
            await candidate.stop()
        await cancel_task(self._notification_task)
        self._notification_task = None
        with contextlib.suppress(Exception):
            self._release_platform_lock()

    def health_status(self) -> dict:
        connected = bool(
            self._rpc is not None
            and not self._stopping
            and getattr(self, "_running", False)
        )
        return {
            "status": "healthy" if connected else "unhealthy",
            "connected": connected,
            "last_rowid": self._last_rowid,
            "dedupe_entries": len(self._seen),
            "bridge_restarts": self._restarts,
            "last_error": self._last_error,
        }

    def _chat_allowed(self, msg: dict) -> bool:
        if not msg.get("is_group") or not self._allowed_chats:
            return True
        refs = {
            str(msg.get(key) or "")
            for key in ("chat_id", "chat_guid", "chat_identifier")
        }
        return bool(refs & self._allowed_chats)

    def _sender_allowed(self, msg: dict) -> bool:
        sender = _normalize_handle(msg.get("sender"))
        is_group = bool(msg.get("is_group"))
        allowed = self._group_allowed_users if is_group else self._allowed_users
        configured = (
            self._group_policy_configured if is_group else self._dm_policy_configured
        )
        return (not configured and not allowed) or sender in allowed or "*" in allowed

    def _is_mentioned(self, text: str) -> bool:
        return any(pattern.search(text or "") for pattern in self._mentions)

    def _attachment_data(self, msg: dict) -> tuple[List[str], List[str], MessageType]:
        paths: List[str] = []
        types: List[str] = []
        primary = MessageType.TEXT
        attachments = msg.get("attachments")
        for item in attachments if isinstance(attachments, list) else []:
            if not isinstance(item, dict) or item.get("missing") is True:
                continue
            raw_path = item.get("original_path") or item.get("path")
            if not raw_path:
                continue
            path = Path(str(raw_path)).expanduser()
            if not path.is_file():
                continue
            mime = str(
                item.get("mime_type")
                or mimetypes.guess_type(path.name)[0]
                or "application/octet-stream"
            )
            paths.append(str(path))
            types.append(mime)
            if primary == MessageType.TEXT:
                primary = _message_type(mime)
        return paths, types, primary

    async def _history_context(self, msg: dict) -> Optional[str]:
        if not self._rpc or self.history_limit <= 0:
            return None
        try:
            result = await self._rpc.request(
                "messages.history",
                {
                    "chat_id": int(msg["chat_id"]),
                    "limit": self.history_limit + 1,
                    "attachments": False,
                },
                timeout=_RPC_TIMEOUT,
            )
        except Exception as exc:
            logger.warning(
                "iMessage: history fetch failed for chat %s: %s",
                msg.get("chat_id"),
                exc,
            )
            return None
        rows = result.get("messages", []) if isinstance(result, dict) else []
        rows = [
            row
            for row in rows
            if isinstance(row, dict) and row.get("guid") != msg.get("guid")
        ]
        rows.sort(key=lambda row: int(row.get("id") or 0))
        lines = []
        for row in rows[-self.history_limit :]:
            text = str(row.get("text") or "").strip()
            if text:
                lines.append(f"[{row.get('sender') or 'unknown'}] {text}")
        return "\n".join(lines) or None

    async def _handle_notification(self, method: str, params: Any) -> None:
        if method == "error":
            self._last_error = str(params)[:1000]
            logger.warning("iMessage: watch error: %s", self._last_error)
            return
        if method == "watch.overflow":
            await self._recover_watch_overflow(params)
            return
        if (
            method != "message"
            or not isinstance(params, dict)
            or not isinstance(params.get("message"), dict)
        ):
            return
        msg = params["message"]
        guid = str(msg.get("guid") or "").strip()
        try:
            rowid = int(msg.get("id") or 0)
        except (TypeError, ValueError):
            rowid = 0
        if not guid or rowid <= 0:
            return
        async with self._intake_lock:
            if guid in self._seen:
                return
            # Every terminal gate advances the durable cursor so restarts cannot replay it forever.
            if (
                msg.get("is_from_me") is True
                or not self._chat_allowed(msg)
                or not self._sender_allowed(msg)
            ):
                self._commit(guid, rowid)
                return
            text = str(msg.get("text") or "").strip()
            is_group = bool(msg.get("is_group"))
            if is_group and self.require_mention and not self._is_mentioned(text):
                self._commit(guid, rowid)
                return
            media_urls, media_types, message_type = self._attachment_data(msg)
            if not text and not media_urls:
                self._commit(guid, rowid)
                return
            chat_id = str(msg.get("chat_id") or "")
            sender = str(msg.get("sender") or "").strip()
            source = SessionSource(
                platform=self.platform,
                chat_id=chat_id,
                chat_name=str(
                    msg.get("chat_name") or msg.get("chat_identifier") or chat_id
                ),
                chat_type="group" if is_group else "dm",
                user_id=sender,
                user_name=sender,
                message_id=guid,
            )
            event = MessageEvent(
                text=text,
                message_type=message_type,
                source=source,
                user_id=sender,
                user_name=sender,
                raw_message=msg,
                message_id=guid,
                media_urls=media_urls,
                media_types=media_types,
                reply_to_message_id=str(msg.get("reply_to_guid") or "") or None,
                reply_to_text=str(msg.get("reply_to_text") or "") or None,
                timestamp=_parse_timestamp(msg.get("created_at")),
                channel_context=await self._history_context(msg) if is_group else None,
                metadata={
                    "chat_guid": msg.get("chat_guid"),
                    "chat_identifier": msg.get("chat_identifier"),
                },
            )
            if (
                getattr(self.handle_message, "__func__", None)
                is BasePlatformAdapter.handle_message
                and self._message_handler is None
            ):
                raise RuntimeError("gateway message handler is not installed")
            uses_base_handler = (
                getattr(self.handle_message, "__func__", None)
                is BasePlatformAdapter.handle_message
            )
            was_active = bool(
                uses_base_handler
                and self._event_session_key(event) in self._active_sessions
            )
            try:
                await self.handle_message(event)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "iMessage: inbound handoff failed; leaving checkpoint unchanged"
                )
                raise
            if uses_base_handler and not event._gateway_accepted and not was_active:
                raise RuntimeError("gateway did not accept iMessage event")
            self._commit(guid, rowid)

    async def _recover_watch_overflow(self, params: Any) -> None:
        if not isinstance(params, dict) or params.get("terminal") is not True:
            return
        if str(params.get("subscription")) != str(self._subscription):
            return
        try:
            cursor = int(params["resume_after_rowid"])
        except (KeyError, TypeError, ValueError):
            self._last_error = "imsg watch overflow omitted its resume cursor"
            logger.error("iMessage: %s", self._last_error)
            if self._rpc:
                await self._rpc.stop()
            return
        rpc = self._rpc
        if rpc is None:
            return
        self._last_error = (
            f"imsg watch overflow ({params.get('reason') or 'unknown'}); resubscribing"
        )
        try:
            result = await rpc.request(
                "watch.subscribe",
                {
                    "attachments": True,
                    "include_reactions": False,
                    "since_rowid": cursor,
                },
                timeout=_RPC_TIMEOUT,
            )
            subscription = (
                result.get("subscription") if isinstance(result, dict) else None
            )
            if not subscription:
                raise ImsgRpcError("imsg overflow recovery returned no subscription")
        except Exception as exc:
            self._last_error = f"imsg overflow recovery failed: {exc}"
            logger.error("iMessage: %s", self._last_error)
            await rpc.stop()
            return
        if rpc is self._rpc and not self._stopping:
            self._subscription = str(subscription)
            logger.warning(
                "iMessage: recovered terminal watch overflow from rowid %d", cursor
            )

    @staticmethod
    def _chat_param(chat_id: str) -> dict:
        value = str(chat_id).strip()
        if value.startswith("chat_guid:"):
            return {"chat_guid": value.split(":", 1)[1]}
        if value.startswith("chat_identifier:"):
            return {"chat_identifier": value.split(":", 1)[1]}
        try:
            return {"chat_id": int(value)}
        except ValueError:
            return {"to": value}

    async def _send_payload(
        self,
        chat_id: str,
        content: str,
        *,
        reply_to: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> SendResult:
        if not self._rpc:
            return SendResult(
                success=False, error="iMessage bridge is not connected", retryable=True
            )
        params = {
            **self._chat_param(chat_id),
            "text": strip_markdown(content or ""),
            "service": "auto",
            "transport": "auto",
        }
        if reply_to:
            params["reply_to"] = str(reply_to)
        if file_path:
            path = Path(file_path).expanduser()
            if not path.is_file():
                return SendResult(success=False, error=f"Attachment not found: {path}")
            params["file"] = str(path)
        try:
            result = await self._rpc.request("send", params, timeout=150.0)
        except Exception as exc:
            return SendResult(success=False, error=str(exc), retryable=True)
        message_id = result.get("guid") if isinstance(result, dict) else None
        return SendResult(success=True, message_id=str(message_id or ""))

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        return await self._send_payload(chat_id, content, reply_to=reply_to)

    async def send_media(
        self, chat_id: str, caption: str, file_path: str, reply_to: Optional[str] = None
    ) -> SendResult:
        return await self._send_payload(
            chat_id, caption, reply_to=reply_to, file_path=file_path
        )

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self.send_media(chat_id, caption or "", file_path, reply_to)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata=None,
        **kwargs,
    ) -> SendResult:
        return await self.send_media(chat_id, caption or "", image_path, reply_to)

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata=None,
        **kwargs,
    ) -> SendResult:
        return await self.send_media(chat_id, caption or "", video_path, reply_to)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata=None,
        **kwargs,
    ) -> SendResult:
        return await self.send_media(chat_id, caption or "", audio_path, reply_to)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Resolve best-effort chat metadata without changing the stable chat id."""
        fallback = {"id": str(chat_id), "name": str(chat_id), "type": "dm"}
        if not self._rpc:
            return fallback
        try:
            result = await self._rpc.request(
                "chats.list", {"limit": 500}, timeout=_RPC_TIMEOUT
            )
        except Exception:
            return fallback
        chats = result.get("chats", []) if isinstance(result, dict) else []
        for chat in chats:
            if not isinstance(chat, dict) or str(chat.get("id")) != str(chat_id):
                continue
            return {
                "id": str(chat_id),
                "name": str(chat.get("name") or chat.get("identifier") or chat_id),
                "type": "group" if chat.get("is_group") else "dm",
            }
        return fallback


def check_requirements() -> bool:
    cli = _get_scoped_secret("IMESSAGE_CLI_PATH", _DEFAULT_CLI, external_fallback=True)
    return bool(Path(cli).is_file() or shutil.which(cli)) and sys.platform == "darwin"


def validate_config(cfg: PlatformConfig) -> bool:
    extra = getattr(cfg, "extra", {}) or {}
    return bool(getattr(cfg, "enabled", False) and extra.get("_configured") is True)


def is_connected(cfg: PlatformConfig) -> bool:
    return validate_config(cfg) and check_requirements()


def _env_enablement() -> Optional[dict]:
    if not check_requirements():
        return None
    explicit = any(
        _get_scoped_secret(name, None, external_fallback=True) is not None
        for name in (
            "IMESSAGE_CLI_PATH",
            "IMESSAGE_DB_PATH",
            "IMESSAGE_ALLOWED_USERS",
            "IMESSAGE_DM_ALLOWED_USERS",
            "IMESSAGE_GROUP_ALLOWED_USERS",
            "IMESSAGE_ALLOWED_CHATS",
            "IMESSAGE_HOME_CHANNEL",
            "IMESSAGE_HISTORY_LIMIT",
            "IMESSAGE_REQUIRE_MENTION",
        )
    )
    if not explicit:
        return None
    seeded = _seed_extra_from_env(
        (
            ("IMESSAGE_CLI_PATH", "cli_path", None),
            ("IMESSAGE_DB_PATH", "db_path", None),
            ("IMESSAGE_HISTORY_LIMIT", "history_limit", int),
            ("IMESSAGE_ALLOWED_USERS", "allowed_users", _list_setting),
            ("IMESSAGE_DM_ALLOWED_USERS", "dm_allowed_users", _list_setting),
            ("IMESSAGE_GROUP_ALLOWED_USERS", "group_allowed_users", _list_setting),
            ("IMESSAGE_ALLOWED_CHATS", "allowed_chats", _list_setting),
            ("IMESSAGE_REQUIRE_MENTION", "require_mention", _truthy),
        ),
        home_env="IMESSAGE_HOME_CHANNEL",
    )
    scoped_policy = any(
        _get_scoped_secret(name, None, external_fallback=True) is not None
        for name in ("IMESSAGE_DM_ALLOWED_USERS", "IMESSAGE_GROUP_ALLOWED_USERS")
    )
    legacy_policy = (
        _get_scoped_secret("IMESSAGE_ALLOWED_USERS", None, external_fallback=True)
        is not None
    )
    if scoped_policy:
        dm = _list_setting(seeded.get("dm_allowed_users"))
        groups = _list_setting(seeded.get("group_allowed_users"))
        seeded["dm_allowed_users"] = dm
        seeded["group_allowed_users"] = groups
        seeded["allowed_users"] = list(dict.fromkeys([*dm, *groups]))
    elif legacy_policy:
        legacy = _list_setting(seeded.get("allowed_users"))
        seeded["allowed_users"] = legacy
        seeded["dm_allowed_users"] = list(legacy)
        seeded["group_allowed_users"] = list(legacy)
    seeded["_configured"] = True
    return seeded


_YAML_BRIDGE = (
    ("cli_path", "IMESSAGE_CLI_PATH", "str"),
    ("db_path", "IMESSAGE_DB_PATH", "str"),
    ("home_channel", "IMESSAGE_HOME_CHANNEL", "str"),
    ("history_limit", "IMESSAGE_HISTORY_LIMIT", "str"),
    ("allowed_users", "IMESSAGE_ALLOWED_USERS", "csv"),
    ("dm_allowed_users", "IMESSAGE_DM_ALLOWED_USERS", "csv"),
    ("group_allowed_users", "IMESSAGE_GROUP_ALLOWED_USERS", "csv"),
    ("allowed_chats", "IMESSAGE_ALLOWED_CHATS", "csv"),
    ("require_mention", "IMESSAGE_REQUIRE_MENTION", "lower"),
)


def _apply_yaml_config(yaml_cfg: dict, platform_cfg: dict) -> Optional[dict]:
    extra = platform_cfg.get("extra", platform_cfg) or {}
    if not isinstance(extra, dict):
        return None
    normalized = dict(extra)
    # Bridge only keys the user actually supplied. Derived scoped allowlists belong in
    # PlatformConfig.extra; exporting them would outrank an explicit legacy env policy.
    bridge_source = dict(normalized)
    if any(
        _get_scoped_secret(name, None) is not None
        for name in (
            "IMESSAGE_ALLOWED_USERS",
            "IMESSAGE_DM_ALLOWED_USERS",
            "IMESSAGE_GROUP_ALLOWED_USERS",
        )
    ):
        for key in ("allowed_users", "dm_allowed_users", "group_allowed_users"):
            bridge_source.pop(key, None)
    bridged = _apply_yaml_bridge(bridge_source, _YAML_BRIDGE) or {}
    if "dm_allowed_users" in normalized or "group_allowed_users" in normalized:
        dm = _list_setting(normalized.get("dm_allowed_users"))
        groups = _list_setting(normalized.get("group_allowed_users"))
        normalized["dm_allowed_users"] = dm
        normalized["group_allowed_users"] = groups
        normalized["allowed_users"] = list(dict.fromkeys([*dm, *groups]))
    elif "allowed_users" in normalized:
        legacy = _list_setting(normalized.get("allowed_users"))
        normalized["allowed_users"] = legacy
        normalized["dm_allowed_users"] = list(legacy)
        normalized["group_allowed_users"] = list(legacy)
    bridged.update({
        key: normalized[key]
        for key in (
            "allowed_users",
            "dm_allowed_users",
            "group_allowed_users",
            "allowed_chats",
        )
        if key in normalized
    })
    bridged["_configured"] = True
    return bridged or None


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id=None,
    media_files=None,
    force_document=False,
) -> Dict[str, Any]:
    extra = getattr(pconfig, "extra", {}) or {}
    client = ImsgRpcClient(
        cli_path=str(
            _extra_or_secret(extra, "cli_path", "IMESSAGE_CLI_PATH", _DEFAULT_CLI)
        ),
        db_path=str(
            Path(
                str(_extra_or_secret(extra, "db_path", "IMESSAGE_DB_PATH", _DEFAULT_DB))
            ).expanduser()
        ),
        on_notification=lambda *_: None,
    )
    try:
        await client.start()
        files = list(media_files or [])
        sends = files or [None]
        message_id = ""
        for index, media in enumerate(sends):
            params = {
                **ImsgAdapter._chat_param(chat_id),
                "text": strip_markdown(message or "") if index == 0 else "",
                "service": "auto",
                "transport": "auto",
            }
            if thread_id:
                params["reply_to"] = str(thread_id)
            if media is not None:
                params["file"] = str(
                    media[0] if isinstance(media, (list, tuple)) else media
                )
            result = await client.request("send", params, timeout=150.0)
            message_id = message_id or str((result or {}).get("guid") or "")
        return {
            "success": True,
            "message_id": message_id,
            **({"media_delivered": True} if media_files else {}),
        }
    except Exception as exc:
        return send_error(f"iMessage send failed: {exc}")
    finally:
        await client.stop()


def register(ctx) -> None:
    ctx.register_platform(
        name="imessage",
        label="iMessage (local imsg)",
        adapter_factory=lambda cfg: ImsgAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=[],
        install_hint="Requires macOS and the imsg CLI with Full Disk Access and Messages Automation permission",
        env_enablement_fn=_env_enablement,
        apply_yaml_config_fn=_apply_yaml_config,
        cron_deliver_env_var="IMESSAGE_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="IMESSAGE_ALLOWED_USERS",
        allow_all_env="IMESSAGE_ALLOW_ALL_USERS",
        max_message_length=8000,
        pii_safe=True,
        emoji="💬",
        allow_update_command=True,
        platform_hint=(
            "You are replying in Apple Messages through the local imsg bridge. "
            "Replies stay in the originating DM or group thread. Use plain text."
        ),
    )
