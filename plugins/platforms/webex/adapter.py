"""Webex gateway adapter.

Uses the Webex Messaging REST API for outbound sends plus either:
- WebSocket events via the official Webex JavaScript SDK (default)
- Webhook callbacks (optional fallback mode)

Credentials are read from the active profile's secret scope. Behavioral
settings are loaded from ``config.yaml`` and retain scoped environment
fallbacks for existing installations.
"""

from __future__ import annotations

import asyncio
import filecmp
import hashlib
import hmac
import json
import logging
import mimetypes
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

try:
    import aiohttp
    from aiohttp import FormData, web

    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None  # type: ignore[assignment]
    FormData = Any  # type: ignore[misc,assignment]
    web = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_audio_from_bytes,
    cache_document_from_bytes,
    cache_image_from_bytes,
    cache_video_from_bytes,
    get_inbound_media_max_bytes,
    safe_url_for_log,
    validate_inbound_media_size,
)
from gateway.platforms._shared import (
    coerce_port,
    get_scoped_secret as _get_scoped_secret,
    seed_extra_from_env,
)
from gateway.platforms.helpers import MessageDeduplicator, send_chunks
from hermes_cli.commands import resolve_command
from hermes_constants import (
    find_node_executable,
    get_hermes_home,
    with_hermes_node_path,
)
from .listener_runtime import (
    _finish_owned_task,
    install_listener_dependencies,
    listener_installation_complete,
)
from .text import MAX_MESSAGE_BYTES, split_message, utf8_length

logger = logging.getLogger(__name__)

API_BASE = "https://webexapis.com/v1"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8646
DEFAULT_PATH = "/webex/webhook"
DEFAULT_WEBHOOK_PREFIX = "hermes-webex"
DEFAULT_CONNECTION_MODE = "websocket"
LISTENER_READY_TIMEOUT_SECONDS = 30
ROOM_CACHE_TTL_SECONDS = 300
THREAD_CONTEXT_CACHE_TTL_SECONDS = 60.0
THREAD_CONTEXT_DEFAULT_LIMIT = 30
THREAD_CONTEXT_MAX_CHARS = 6000
ATTACHMENT_READ_CHUNK_SIZE = 64 * 1024
WEBHOOK_MAX_BODY_BYTES = 1024 * 1024
MESSAGE_QUEUE_MAX_SIZE = 1000
_PLUGIN_DIR = Path(__file__).resolve().parent
_LISTENER_FILES = ("webex_listener.js", "package.json", "package-lock.json", ".npmrc")
_LISTENER_DEPENDENCIES = {
    "@babel/runtime-corejs2": "7.29.7",
    "@webex/internal-plugin-device": "3.12.0",
    "@webex/internal-plugin-mercury": "3.12.0",
    "@webex/plugin-logger": "3.12.0",
    "@webex/plugin-messages": "3.12.0",
    "@webex/plugin-people": "3.12.0",
    "@webex/plugin-rooms": "3.12.0",
    "@webex/webex-core": "3.12.0",
}


class _WebexAPIRequestError(RuntimeError):
    """Webex REST request failure whose HTTP status is safe to classify."""

    def __init__(
        self, path: str, status: Optional[int] = None, *, retry_after: Optional[float] = None,
        timed_out: bool = False, transport_error: bool = False,
    ):
        self.path = path
        self.status = status
        self.retry_after = retry_after
        self.timed_out = timed_out
        self.transport_error = transport_error
        detail = "timed out" if timed_out else f"HTTP {status}" if status is not None else "transport error"
        if status == 429:
            detail += " rate limited"
        super().__init__(f"Webex API {path} failed ({detail})")

    def as_send_result(self) -> SendResult:
        retryable = not self.timed_out and (
            self.transport_error or self.status in {408, 429} or (self.status is not None and self.status >= 500)
        )
        kind = {401: "forbidden", 403: "forbidden", 404: "not_found", 413: "too_long", 429: "rate_limited"}.get(
            self.status, "transient" if retryable else "unknown"
        )
        error = str(self)
        if self.status == 429:
            error = f"flood_control:{self.retry_after or 0.0}"
        return SendResult(
            success=False, error=error, retryable=retryable,
            error_kind=kind, retry_after=self.retry_after,
        )


class _WebexStartupError(RuntimeError):
    """Typed startup failure consumed by the gateway fatal-error contract."""

    def __init__(self, code: str, message: str, *, retryable: bool):
        self.code = code
        self.retryable = retryable
        super().__init__(message)


@dataclass
class _WebexThreadContextCache:
    """Cache entry for fetched Webex thread context."""

    content: str
    fetched_at: float = field(default_factory=time.monotonic)
    parent_text: str = ""


def check_webex_requirements() -> bool:
    """Passively verify dependencies shared by both connection modes."""
    return AIOHTTP_AVAILABLE


def _listener_dependencies_available(listener_dir: Path) -> bool:
    """Return whether the exact locked SDK packages exist in *listener_dir*."""
    for package, expected_version in _LISTENER_DEPENDENCIES.items():
        package_json = listener_dir / "node_modules" / package / "package.json"
        try:
            installed_version = json.loads(
                package_json.read_text(encoding="utf-8-sig")
            ).get("version")
        except (OSError, json.JSONDecodeError, AttributeError):
            return False
        if installed_version != expected_version:
            return False
    return True


def _listener_runtime_dir() -> Optional[Path]:
    """Use bundled deps or mirror package inputs into profile-local storage."""
    if _listener_dependencies_available(_PLUGIN_DIR):
        return _PLUGIN_DIR
    runtime_dir = get_hermes_home() / "platforms" / "webex" / "listener"
    try:
        runtime_dir.mkdir(parents=True, exist_ok=True)
        for name in _LISTENER_FILES:
            source = _PLUGIN_DIR / name
            target = runtime_dir / name
            if source.exists() and (
                not target.exists()
                or not filecmp.cmp(str(source), str(target), shallow=False)
            ):
                shutil.copy2(source, target)
    except OSError as exc:
        logger.error("[Webex] Could not prepare writable listener directory: %s", exc)
        return None
    return runtime_dir


async def ensure_websocket_listener_requirements() -> Optional[Path]:
    """Install locked listener packages for WebSocket mode only."""
    listener_dir = await asyncio.to_thread(_listener_runtime_dir)
    if listener_dir is None:
        return None

    def ready() -> bool:
        return _listener_dependencies_available(listener_dir) and (
            listener_dir == _PLUGIN_DIR or listener_installation_complete(listener_dir)
        )

    if await asyncio.to_thread(ready):
        return listener_dir
    if not await install_listener_dependencies(listener_dir):
        return None
    return listener_dir if await asyncio.to_thread(ready) else None


class WebexAdapter(BasePlatformAdapter):
    """Gateway adapter for Webex Messaging bots."""

    SUPPORTS_MESSAGE_EDITING = True
    serves_profile_prefix = True
    MAX_MESSAGE_LENGTH = MAX_MESSAGE_BYTES

    @property
    def message_len_fn(self) -> Callable[[str], int]:
        return utf8_length

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("webex"))
        extra = config.extra or {}
        self._token: str = config.token or _get_scoped_secret("WEBEX_BOT_TOKEN", "")
        self._connection_mode = _configured_connection_mode(config)
        self._host: str = str(
            extra.get("host") or _get_scoped_secret("WEBEX_WEBHOOK_HOST", DEFAULT_HOST)
        )
        self._port: int = coerce_port(
            extra.get("port") or _get_scoped_secret("WEBEX_WEBHOOK_PORT"),
            DEFAULT_PORT,
        )
        self._path: str = self._normalize_path(
            str(
                extra.get("path")
                or _get_scoped_secret("WEBEX_WEBHOOK_PATH", DEFAULT_PATH)
            )
        )
        self._public_url: str = str(
            extra.get("public_url")
            or _get_scoped_secret("WEBEX_WEBHOOK_PUBLIC_URL", "")
        ).rstrip("/")
        self._secret: str = str(
            extra.get("secret") or _get_scoped_secret("WEBEX_WEBHOOK_SECRET", "")
        )
        self._require_mention: bool = extra.get("require_mention", True)
        endpoint_digest = hashlib.sha256(self._target_url.encode("utf-8")).hexdigest()[
            :12
        ]
        self._webhook_prefix: str = str(
            extra.get("webhook_prefix") or f"{DEFAULT_WEBHOOK_PREFIX}-{endpoint_digest}"
        )

        self._runner: Optional[web.AppRunner] = None
        self._app: Optional[web.Application] = None
        self._session: Any = None  # aiohttp.ClientSession
        self._poll_task: Optional[asyncio.Task] = None
        self._listener_process: Optional[asyncio.subprocess.Process] = None
        self._listener_stdout_task: Optional[asyncio.Task] = None
        self._listener_stderr_task: Optional[asyncio.Task] = None
        self._listener_exit_task: Optional[asyncio.Task] = None
        self._listener_ready: Optional[asyncio.Future] = None
        self._listener_dir: Optional[Path] = None
        self._message_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=MESSAGE_QUEUE_MAX_SIZE
        )
        self._dedup = MessageDeduplicator()
        self._bot_id: str = ""
        self._bot_email: str = ""
        self._bot_display_name: str = ""
        self._room_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._thread_context_cache: dict[str, _WebexThreadContextCache] = {}
        self._managed_webhook_ids: list[str] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not AIOHTTP_AVAILABLE:
            return self._fail_startup(
                "webex_missing_dependency",
                "aiohttp is not installed. Reinstall Hermes with gateway dependencies.",
                retryable=False,
            )
        if not self._token:
            return self._fail_startup(
                "webex_missing_token",
                "WEBEX_BOT_TOKEN is not configured.",
                retryable=False,
            )
        if self._connection_mode not in {"websocket", "webhook"}:
            return self._fail_startup(
                "webex_invalid_config",
                f"Unsupported Webex connection mode {self._connection_mode!r}; "
                "supported modes are websocket and webhook.",
                retryable=False,
            )
        if self._connection_mode == "webhook" and not self._public_url:
            return self._fail_startup(
                "webex_invalid_config",
                "A public URL is required for Webex webhook mode.",
                retryable=False,
            )
        if self._connection_mode == "webhook" and not self._public_url.startswith(
            "https://"
        ):
            return self._fail_startup(
                "webex_invalid_config",
                "The Webex webhook public URL must use HTTPS.",
                retryable=False,
            )
        if self._connection_mode == "webhook" and not self._secret:
            return self._fail_startup(
                "webex_invalid_config",
                "WEBEX_WEBHOOK_SECRET is required for Webex webhook mode.",
                retryable=False,
            )
        if self._connection_mode == "webhook" and not self._path.startswith("/"):
            return self._fail_startup(
                "webex_invalid_config",
                f"Invalid Webex webhook path {self._path!r}.",
                retryable=False,
            )
        if self._connection_mode == "webhook":
            from gateway.platforms.shared_ingress import shared_ingress_profile

            profile = shared_ingress_profile(self)
            if profile and not urlsplit(self._target_url).path.endswith(
                f"/p/{profile}{self._path}"
            ):
                return self._fail_startup(
                    "webex_invalid_config",
                    f"Shared Webex callbacks must end with /p/{profile}{self._path}; "
                    f"include /p/{profile} in public_url.",
                    retryable=False,
                )
        if not self._acquire_platform_lock(
            "webex-bot-token", self._token, "Webex bot token"
        ):
            return False

        try:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
            await self._load_bot_identity()

            if self._connection_mode == "webhook":
                await self._start_webhook_listener()
                await self._ensure_managed_webhooks()

            else:
                self._listener_dir = await ensure_websocket_listener_requirements()
                if self._listener_dir is None:
                    raise _WebexStartupError(
                        "webex_missing_dependency",
                        "Webex listener dependencies could not be installed.",
                        retryable=False,
                    )
                await self._start_websocket_listener()

            self._poll_task = asyncio.create_task(self._poll_loop())
            self._mark_connected()
            if self._connection_mode == "webhook" and self._runner is not None:
                logger.info(
                    "[Webex] Listening on %s:%s%s and delivering via %s",
                    self._host,
                    self._port,
                    self._path,
                    safe_url_for_log(self._target_url),
                )
            elif self._connection_mode == "websocket":
                self._wire_plugin_handlers(None)
                logger.info(
                    "[Webex] Connected in websocket mode via local SDK listener"
                )
            return True
        except _WebexAPIRequestError as exc:
            is_auth_error = exc.status in {401, 403}
            code = "webex_auth_error" if is_auth_error else "webex_api_unavailable"
            message = (
                f"Webex bot authentication failed (HTTP {exc.status})."
                if is_auth_error
                else "Webex API was unavailable during startup; the gateway will retry."
            )
            logger.exception("[Webex] Failed to start")
            self._set_fatal_error(code, message, retryable=not is_auth_error)
            await self._cleanup()
            self._release_platform_lock()
            return False
        except _WebexStartupError as exc:
            logger.error("[Webex] Failed to start: %s", exc)
            self._set_fatal_error(exc.code, str(exc), retryable=exc.retryable)
            await self._cleanup()
            self._release_platform_lock()
            return False
        except Exception:
            logger.exception("[Webex] Failed to start")
            self._set_fatal_error(
                "webex_startup_failed",
                "Webex startup failed; the gateway will retry.",
                retryable=True,
            )
            await self._cleanup()
            self._release_platform_lock()
            return False

    def _fail_startup(self, code: str, message: str, *, retryable: bool) -> bool:
        logger.error("[Webex] %s", message)
        self._set_fatal_error(code, message, retryable=retryable)
        return False

    async def disconnect(self) -> None:
        self._running = False
        try:
            try:
                if self._poll_task and not self._poll_task.done():
                    self._poll_task.cancel()
                    try:
                        await self._poll_task
                    except asyncio.CancelledError:
                        if asyncio.current_task().cancelling():
                            raise
            finally:
                self._poll_task = None
                await self._cleanup()
        finally:
            self._release_platform_lock()
            self._mark_disconnected()
            logger.info("[Webex] Disconnected")

    async def _cleanup(self) -> None:
        try:
            await self._stop_websocket_listener()
        finally:
            self._shared_ingress_app = None
            try:
                if self._runner:
                    await _finish_owned_task(
                        asyncio.create_task(self._runner.cleanup())
                    )
            finally:
                self._runner = None
                self._app = None
                try:
                    if self._session:
                        await _finish_owned_task(
                            asyncio.create_task(self._session.close())
                        )
                finally:
                    self._session = None

    async def _start_webhook_listener(self) -> None:
        from gateway.platforms.shared_ingress import bind_listener

        self._app = web.Application(client_max_size=WEBHOOK_MAX_BODY_BYTES)
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_post(self._path, self._handle_webhook)
        self._wire_plugin_handlers(self._app)
        self._runner = await bind_listener(
            self, self._app, self._host, self._port, self._path)

    # ------------------------------------------------------------------
    # Outbound messaging
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        message = self.format_message(content).strip()
        if not message:
            return SendResult(success=True)

        sent_ids = []

        async def send_one(chunk):
            payload = self._build_text_payload(chat_id, chunk, metadata=metadata)
            if payload is None:
                return SendResult(success=False, error="Missing room target", error_kind="not_found")
            try:
                data = await self._api_post_json("messages", payload)
            except _WebexAPIRequestError as exc:
                return exc.as_send_result()
            if not data or not data.get("id"):
                return SendResult(
                    success=False,
                    error=json.dumps(data) if data else "Webex send returned no data",
                    error_kind="unknown", raw_response=data,
                )
            sent_ids.append(str(data["id"]))
            return SendResult(success=True, message_id=str(data["id"]), raw_response=data)

        result = await send_chunks(split_message(message), send_one)
        if sent_ids:
            result.message_id = sent_ids[-1]
            result.continuation_message_ids = tuple(sent_ids[:-1])
        return result

    def _send_retry_is_final(self, result: SendResult) -> bool:
        return self._is_partial_delivery(result) or result.error_kind in {
            "forbidden", "not_found", "too_long", "unknown",
        }

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
    ) -> SendResult:
        message = self.format_message(content).strip()
        if not message:
            return SendResult(success=True, message_id=message_id)
        if utf8_length(message) > MAX_MESSAGE_BYTES:
            return SendResult(
                success=False, message_id=message_id, error_kind="too_long",
                error=f"Webex edits are limited to {MAX_MESSAGE_BYTES} UTF-8 bytes",
            )

        room_id = await self._resolve_edit_room_id(chat_id, message_id)
        if not room_id:
            return SendResult(success=False, error="Missing room target for Webex edit")

        try:
            data = await self._api_put_json(
                f"messages/{message_id}",
                {"roomId": room_id, "markdown": message},
            )
        except _WebexAPIRequestError as exc:
            return exc.as_send_result()
        if not data:
            return SendResult(success=False, error="Webex edit returned no data")
        if data.get("id"):
            return SendResult(
                success=True, message_id=str(data["id"]), raw_response=data
            )
        return SendResult(success=False, error=json.dumps(data))

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        formatted_caption = self.format_message(caption or "")
        caption_ids = ()
        if utf8_length(formatted_caption) > MAX_MESSAGE_BYTES:
            sent_caption = await self.send(chat_id, caption or "", reply_to=reply_to, metadata=metadata)
            if not sent_caption.success:
                return sent_caption
            caption_ids = (*sent_caption.continuation_message_ids, sent_caption.message_id)
            formatted_caption = ""
            caption = None
        payload = self._build_text_payload(chat_id, formatted_caption, metadata=metadata)
        if payload is None:
            return SendResult(success=False, error="Missing room target")
        payload["files"] = [image_url]
        try:
            data = await self._api_post_json("messages", payload)
        except _WebexAPIRequestError as exc:
            if exc.status != 400:
                result = exc.as_send_result()
                result.continuation_message_ids = caption_ids
                return result
            data = {}
        if data and data.get("id"):
            return SendResult(
                success=True, message_id=str(data["id"]), raw_response=data,
                continuation_message_ids=caption_ids,
            )
        # Fall back to a plain text URL if the API rejects file-url delivery.
        text = f"{caption}\n{image_url}".strip() if caption else image_url
        result = await self.send(
            chat_id=chat_id, content=text, reply_to=reply_to, metadata=metadata
        )
        result.continuation_message_ids = (*caption_ids, *result.continuation_message_ids)
        return result

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> SendResult:
        return await self._send_local_file(
            chat_id, image_path, caption=caption, metadata=metadata
        )

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> SendResult:
        return await self._send_local_file(
            chat_id,
            file_path,
            caption=caption,
            metadata=metadata,
            file_name=file_name,
        )

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> SendResult:
        return await self._send_local_file(
            chat_id, audio_path, caption=caption, metadata=metadata
        )

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> SendResult:
        return await self._send_local_file(
            chat_id, video_path, caption=caption, metadata=metadata
        )

    async def _send_local_file(
        self,
        chat_id: str,
        file_path: str,
        *,
        caption: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        file_name: Optional[str] = None,
    ) -> SendResult:
        path = Path(file_path)
        if not await asyncio.to_thread(path.exists):
            return SendResult(success=False, error=f"File not found: {file_path}")

        formatted_caption = self.format_message(caption or "")
        caption_ids = ()
        if utf8_length(formatted_caption) > MAX_MESSAGE_BYTES:
            sent_caption = await self.send(chat_id, caption or "", metadata=metadata)
            if not sent_caption.success:
                return sent_caption
            caption_ids = (*sent_caption.continuation_message_ids, sent_caption.message_id)
            formatted_caption = ""

        form = FormData()
        self._populate_target_fields(form, chat_id)
        parent_id = self._extract_parent_id(metadata)
        if parent_id:
            form.add_field("parentId", parent_id)
        if formatted_caption:
            form.add_field("markdown", formatted_caption)

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        form.add_field(
            "files",
            await asyncio.to_thread(path.read_bytes),
            filename=file_name or path.name,
            content_type=content_type,
        )

        try:
            data = await self._api_post_form("messages", form)
        except _WebexAPIRequestError as exc:
            result = exc.as_send_result()
            result.continuation_message_ids = caption_ids
            return result
        if data and data.get("id"):
            return SendResult(
                success=True, message_id=str(data["id"]), raw_response=data,
                continuation_message_ids=caption_ids,
            )
        return SendResult(
            success=False, error=json.dumps(data or {"error": "upload failed"}),
            continuation_message_ids=caption_ids,
        )

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        room = await self._get_room(chat_id)
        room_type = room.get("type", "group")
        return {
            "name": room.get("title") or chat_id,
            "type": "dm" if room_type == "direct" else room_type,
            "chat_id": chat_id,
        }

    # ------------------------------------------------------------------
    # Inbound transport processing
    # ------------------------------------------------------------------

    async def _start_websocket_listener(self) -> None:
        listener_dir = self._listener_dir or _PLUGIN_DIR
        script_path = listener_dir / "webex_listener.js"
        if not script_path.exists():
            raise _WebexStartupError(
                "webex_missing_dependency",
                f"Missing Webex listener script: {script_path}",
                retryable=False,
            )

        node = find_node_executable("node")
        if not node:
            raise _WebexStartupError(
                "webex_missing_dependency",
                "Node.js is required for Webex websocket mode.",
                retryable=False,
            )
        env = with_hermes_node_path()
        env["WEBEX_BOT_TOKEN"] = self._token
        try:
            process = await asyncio.create_subprocess_exec(
                node,
                str(script_path),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(listener_dir),
                env=env,
            )
        except FileNotFoundError as exc:
            raise _WebexStartupError(
                "webex_missing_dependency",
                "Node.js is required for Webex websocket mode.",
                retryable=False,
            ) from exc
        self._listener_process = process
        self._listener_ready = asyncio.get_running_loop().create_future()
        self._listener_stdout_task = asyncio.create_task(
            self._listener_stdout_loop(process)
        )
        self._listener_stderr_task = asyncio.create_task(
            self._listener_stderr_loop(process)
        )
        self._listener_exit_task = asyncio.create_task(
            self._listener_exit_loop(process)
        )
        await asyncio.wait_for(
            self._listener_ready, timeout=LISTENER_READY_TIMEOUT_SECONDS
        )
        if process.returncode is not None:
            raise RuntimeError(
                f"Webex listener exited during startup (code {process.returncode})"
            )

    async def _stop_websocket_listener(self) -> None:
        process = self._listener_process
        try:
            if process and process.returncode is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except TimeoutError:
                    pass
        finally:
            try:
                if process and process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    # health: allow HX006 -- cancelled shutdown must reap its killed child before releasing ownership
                    reap = asyncio.create_task(process.wait())
                    await _finish_owned_task(reap)
            finally:
                if self._listener_process is process and (
                    process is None or process.returncode is not None
                ):
                    self._listener_process = None
                tasks = []
                for attr in (
                    "_listener_stdout_task",
                    "_listener_stderr_task",
                    "_listener_exit_task",
                ):
                    task = getattr(self, attr)
                    if task is not None:
                        task.cancel()
                        tasks.append(task)
                    setattr(self, attr, None)
                try:
                    results = await _finish_owned_task(
                        asyncio.gather(*tasks, return_exceptions=True)
                    )
                    for result in results:
                        if isinstance(result, BaseException) and not isinstance(
                            result, asyncio.CancelledError
                        ):
                            logger.warning(
                                "[Webex] Listener task failed during shutdown: %s", result
                            )
                finally:
                    if self._listener_ready and not self._listener_ready.done():
                        self._listener_ready.cancel()
                    self._listener_ready = None

    async def _listener_stdout_loop(self, process: asyncio.subprocess.Process) -> None:
        stream = process.stdout
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                break
            raw = line.decode("utf-8", errors="replace").strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                logger.debug("[Webex] Listener stdout: %s", raw)
                continue
            if not isinstance(payload, dict):
                logger.debug(
                    "[Webex] Ignoring non-object listener payload: %r", payload
                )
                continue
            await self._handle_listener_message(payload)

    async def _listener_stderr_loop(self, process: asyncio.subprocess.Process) -> None:
        stream = process.stderr
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                break
            raw = line.decode("utf-8", errors="replace").strip()
            if raw:
                logger.warning("[Webex] Listener stderr: %s", raw)

    async def _listener_exit_loop(self, process: asyncio.subprocess.Process) -> None:
        returncode = await process.wait()  # health: allow HX006 -- Lifetime watcher; disconnect kills and reaps the child.
        if self._listener_ready and not self._listener_ready.done():
            self._listener_ready.set_exception(
                RuntimeError(
                    f"Webex listener exited before startup completed (code {returncode})"
                )
            )
            return
        if not self._running:
            return
        message = f"Webex websocket listener exited unexpectedly (code {returncode})."
        logger.error("[Webex] %s", message)
        if not self.has_fatal_error:
            self._set_fatal_error("webex_listener_exited", message, retryable=True)
            await self._notify_fatal_error()

    async def _handle_listener_message(self, payload: dict[str, Any]) -> None:
        msg_type = str(payload.get("type") or "").strip().lower()
        if msg_type == "ready":
            if self._listener_ready and not self._listener_ready.done():
                self._listener_ready.set_result(payload)
            return

        if msg_type == "log":
            level = str(payload.get("level") or "info").lower()
            message = str(payload.get("message") or "").strip()
            if not message:
                return
            log_fn = {
                "debug": logger.debug,
                "warning": logger.warning,
                "error": logger.error,
            }.get(level, logger.info)
            log_fn("[Webex] %s", message)
            return

        if msg_type == "fatal":
            message = str(payload.get("message") or "Webex listener failed").strip()
            failure = self._classify_listener_failure(message)
            if self._listener_ready and not self._listener_ready.done():
                self._listener_ready.set_exception(failure)
                return
            if not self.has_fatal_error:
                self._set_fatal_error(
                    failure.code,
                    str(failure),
                    retryable=failure.retryable,
                )
                await self._notify_fatal_error()
            logger.error("[Webex] %s", message)
            return

        if msg_type != "event":
            return

        event = payload.get("event")
        if not isinstance(event, dict):
            return
        if event.get("resource") != "messages" or event.get("event") != "created":
            return
        try:
            self._message_queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("[Webex] Dropping websocket event because the queue is full")

    @staticmethod
    def _classify_listener_failure(message: str) -> _WebexStartupError:
        normalized = message.lower()
        if (
            "webex js sdk packages are not installed" in normalized
            or "webex_bot_token is required" in normalized
        ):
            code = (
                "webex_missing_dependency"
                if "sdk packages" in normalized
                else "webex_missing_token"
            )
            return _WebexStartupError(code, message, retryable=False)
        if re.search(r"\b(?:401|403)\b", normalized) and (
            "unauthorized" in normalized or "forbidden" in normalized
        ):
            return _WebexStartupError("webex_auth_error", message, retryable=False)
        return _WebexStartupError("webex_listener_fatal", message, retryable=True)

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "platform": "webex"})

    async def _handle_webhook(self, request: web.Request) -> web.Response:
        body = await request.read()
        if not self._verify_signature(request.headers, body):
            return web.Response(status=403, text="invalid signature")

        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return web.Response(status=400, text="invalid json")

        if payload.get("resource") != "messages" or payload.get("event") != "created":
            return web.Response(status=200, text="ignored")

        try:
            self._message_queue.put_nowait(payload)
        except asyncio.QueueFull:
            logger.warning("[Webex] Dropping webhook event because the queue is full")
            return web.Response(status=503, text="queue full")
        return web.Response(status=200, text="ok")

    async def _poll_loop(self) -> None:
        while True:
            payload = await self._message_queue.get()
            try:
                event = await self._build_event(payload)
                if event is None:
                    continue
                task = asyncio.create_task(self.handle_message(event))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            except Exception:
                logger.exception("[Webex] Failed to process inbound payload")

    async def _build_event(self, payload: dict[str, Any]) -> Optional[MessageEvent]:
        data = payload.get("data") or {}
        message_id = str(data.get("id") or "")
        if not message_id:
            return None

        message = self._coerce_event_message(data)
        if not message or self._needs_full_message_fetch(message):
            fetched_message = await self._api_get_json(f"messages/{message_id}")
            if fetched_message:
                message = {**message, **fetched_message}
            elif not message:
                return None
        if self._dedup.is_duplicate(message_id):
            return None

        sender_person_id = str(message.get("personId") or "")
        if sender_person_id and sender_person_id == self._bot_id:
            return None

        room_id = str(message.get("roomId") or data.get("roomId") or "")
        if not room_id:
            return None
        room = await self._get_room(room_id)
        room_type = str(room.get("type") or data.get("roomType") or "group")
        chat_type = "dm" if room_type == "direct" else "group"

        sender_email = str(message.get("personEmail") or data.get("personEmail") or "")
        sender_name = sender_email
        sender_display_name = await self._lookup_display_name(sender_person_id)
        if sender_display_name:
            sender_name = sender_display_name

        parent_id = str(message.get("parentId") or "") or None
        raw_text = str(message.get("text") or message.get("markdown") or "").strip()
        text = raw_text
        if room_type != "direct":
            text = self._strip_bot_mention(text)
            if (
                self._require_mention
                and not self._group_message_mentions_bot(message)
                and not self._is_gateway_command(text)
            ):
                return None

        media_urls, media_types = await self._download_attachments(
            message.get("files") or []
        )
        msg_type = self._detect_message_type(text, media_types)

        source = self.build_source(
            chat_id=room_id,
            chat_name=room.get("title") or room_id,
            chat_type=chat_type,
            user_id=sender_email or sender_person_id,
            user_name=sender_name,
            thread_id=parent_id,
            user_id_alt=sender_person_id or None,
        )

        reply_to_text = None
        channel_context = None
        if parent_id:
            if (
                msg_type != MessageType.COMMAND
                and not self._has_active_session_for_thread(
                    room_id=room_id,
                    chat_type=chat_type,
                    parent_id=parent_id,
                    user_id=sender_email or sender_person_id,
                    user_id_alt=sender_person_id or None,
                )
            ):
                thread_context = await self._fetch_thread_context(
                    room_id=room_id,
                    parent_id=parent_id,
                    current_message_id=message_id,
                    chat_type=chat_type,
                )
                if thread_context:
                    channel_context = thread_context
            reply_to_text = (
                await self._fetch_thread_parent_text(room_id, parent_id) or None
            )

        return MessageEvent(
            text=text,
            message_type=msg_type,
            source=source,
            raw_message=message,
            message_id=message_id,
            media_urls=media_urls,
            media_types=media_types,
            reply_to_message_id=parent_id,
            reply_to_text=reply_to_text,
            channel_context=channel_context,
        )

    @staticmethod
    def _coerce_event_message(data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            return {}
        if not data.get("id") or not data.get("roomId") or not data.get("personId"):
            return {}
        return dict(data)

    @staticmethod
    def _needs_full_message_fetch(message: dict[str, Any]) -> bool:
        return not bool(
            message.get("text") or message.get("markdown") or message.get("files")
        )

    # ------------------------------------------------------------------
    # Webex API helpers
    # ------------------------------------------------------------------

    async def _load_bot_identity(self) -> None:
        me = await self._api_get_json("people/me", raise_on_error=True)
        if not me or not me.get("id"):
            raise RuntimeError("Failed to load Webex bot identity from /people/me")
        self._bot_id = str(me.get("id") or "")
        emails = me.get("emails") or []
        self._bot_email = str(emails[0]) if emails else ""
        self._bot_display_name = str(
            me.get("displayName") or self._bot_email or self._bot_id
        )

    async def _get_room(self, room_id: str) -> dict[str, Any]:
        cached = self._room_cache.get(room_id)
        now = time.time()
        if cached and cached[0] > now:
            return cached[1]
        room = await self._api_get_json(f"rooms/{room_id}")
        if room:
            self._room_cache[room_id] = (now + ROOM_CACHE_TTL_SECONDS, room)
        return room or {}

    async def _lookup_display_name(self, person_id: str) -> str:
        if not person_id:
            return ""
        person = await self._api_get_json(f"people/{person_id}")
        return str(person.get("displayName") or "")

    async def _api_get_json(
        self,
        path: str,
        params: Optional[dict[str, Any]] = None,
        *,
        raise_on_error: bool = False,
    ) -> dict[str, Any]:
        if not self._session:
            if raise_on_error:
                raise _WebexAPIRequestError(path)
            return {}
        url = f"{API_BASE}/{path.lstrip('/')}"
        try:
            async with self._session.get(
                url, headers=self._headers(), params=params
            ) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    logger.warning("[Webex] GET %s -> HTTP %s", path, resp.status)
                    if raise_on_error:
                        raise _WebexAPIRequestError(path, resp.status)
                    return {}
                return json.loads(text) if text else {}
        except _WebexAPIRequestError:
            raise
        except Exception as exc:
            logger.warning("[Webex] GET %s failed: %s", path, exc)
            if raise_on_error:
                raise _WebexAPIRequestError(path) from exc
            return {}

    async def _api_post_json(
        self, path: str, payload: dict[str, Any], *, raise_on_error: bool = True,
    ) -> dict[str, Any]:
        return await self._api_write("post", path, payload, raise_on_error=raise_on_error)

    async def _api_post_form(self, path: str, form: FormData) -> dict[str, Any]:
        return await self._api_write("post", path, form, form=True)

    async def _api_put_json(
        self, path: str, payload: dict[str, Any], *, raise_on_error: bool = True,
    ) -> dict[str, Any]:
        return await self._api_write("put", path, payload, raise_on_error=raise_on_error)

    async def _api_write(
        self, method: str, path: str, payload: Any, *, form: bool = False, raise_on_error: bool = True,
    ) -> dict[str, Any]:
        if not self._session:
            if raise_on_error:
                raise _WebexAPIRequestError(path, transport_error=True)
            return {}
        url = f"{API_BASE}/{path.lstrip('/')}"
        headers = self._headers() if form else self._json_headers()
        try:
            async with getattr(self._session, method)(
                url, headers=headers, **{"data" if form else "json": payload},
            ) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    logger.warning("[Webex] %s %s -> HTTP %s", method.upper(), path, resp.status)
                    if raise_on_error:
                        try:
                            retry_after = max(0.0, float(resp.headers["Retry-After"]))
                        except (KeyError, ValueError):
                            retry_after = None
                        raise _WebexAPIRequestError(path, resp.status, retry_after=retry_after)
                    return {}
                return json.loads(text) if text else {}
        except _WebexAPIRequestError:
            raise
        except (TimeoutError, aiohttp.ClientError, ValueError) as exc:
            logger.warning("[Webex] %s %s failed: %s", method.upper(), path, exc)
            if raise_on_error:
                raise _WebexAPIRequestError(
                    path, timed_out=isinstance(exc, TimeoutError),
                    transport_error=isinstance(exc, aiohttp.ClientConnectionError),
                ) from exc
            return {}

    async def _api_delete(self, path: str) -> bool:
        if not self._session:
            return False
        url = f"{API_BASE}/{path.lstrip('/')}"
        try:
            async with self._session.delete(url, headers=self._headers()) as resp:
                if resp.status >= 400:
                    logger.warning("[Webex] DELETE %s -> HTTP %s", path, resp.status)
                    return False
                return True
        except (TimeoutError, aiohttp.ClientError, ValueError) as exc:
            logger.warning("[Webex] DELETE %s failed: %s", path, exc)
            return False

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
        }

    def _json_headers(self) -> dict[str, str]:
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        return headers

    # ------------------------------------------------------------------
    # Managed webhook helpers
    # ------------------------------------------------------------------

    @property
    def _target_url(self) -> str:
        return f"{self._public_url}{self._path}"

    async def _ensure_managed_webhooks(self) -> None:
        existing = await self._api_get_json("webhooks")
        items = existing.get("items") if isinstance(existing, dict) else None
        items = items if isinstance(items, list) else []

        desired = []
        desired.append({
            "name": f"{self._webhook_prefix}-direct",
            "resource": "messages",
            "event": "created",
            "filter": "roomType=direct",
        })
        desired.append({
            "name": f"{self._webhook_prefix}-mentions",
            "resource": "messages",
            "event": "created",
            "filter": "mentionedPeople=me" if self._require_mention else "",
        })

        matched_ids: dict[tuple[str, str, str, str], str] = {}
        desired_keys = {
            (spec["resource"], spec["event"], spec["filter"], self._target_url)
            for spec in desired
        }

        for webhook in items:
            if not isinstance(webhook, dict):
                continue
            name = str(webhook.get("name") or "")
            if not name.startswith(self._webhook_prefix):
                continue
            key = (
                str(webhook.get("resource") or ""),
                str(webhook.get("event") or ""),
                str(webhook.get("filter") or ""),
                str(webhook.get("targetUrl") or ""),
            )
            webhook_id = str(webhook.get("id") or "")
            if key in desired_keys and key not in matched_ids:
                matched_ids[key] = webhook_id
                continue
            if webhook_id:
                await self._api_delete(f"webhooks/{webhook_id}")

        self._managed_webhook_ids = []
        for spec in desired:
            key = (spec["resource"], spec["event"], spec["filter"], self._target_url)
            if key in matched_ids:
                webhook_id = matched_ids[key]
                # Webex does not return the signing key when listing hooks.
                # Always reconcile it and recover hooks disabled after an outage.
                updated = await self._api_put_json(f"webhooks/{webhook_id}", {
                    "name": spec["name"],
                    "targetUrl": self._target_url,
                    "secret": self._secret,
                    "status": "active",
                }, raise_on_error=False)
                if not updated.get("id") or updated.get("status") != "active":
                    raise RuntimeError("Failed to update managed Webex webhook")
                self._managed_webhook_ids.append(webhook_id)
                continue
            payload = {
                "name": spec["name"],
                "targetUrl": self._target_url,
                "resource": spec["resource"],
                "event": spec["event"],
            }
            if spec["filter"]:
                payload["filter"] = spec["filter"]
            if self._secret:
                payload["secret"] = self._secret
            created = await self._api_post_json("webhooks", payload, raise_on_error=False)
            webhook_id = str(created.get("id") or "")
            if not webhook_id:
                raise RuntimeError(
                    f"Failed to create managed Webex webhook for filter {spec['filter']!r}"
                )
            self._managed_webhook_ids.append(webhook_id)

    # ------------------------------------------------------------------
    # Signature verification
    # ------------------------------------------------------------------

    def _verify_signature(self, headers: Any, body: bytes) -> bool:
        if not self._secret:
            return False

        secret = self._secret.encode("utf-8")
        spark_sig = str(headers.get("X-Spark-Signature", "")).strip()
        if spark_sig:
            digest = hmac.new(secret, body, hashlib.sha1).hexdigest()
            if hmac.compare_digest(spark_sig, digest):
                return True

        webex_sig = str(headers.get("X-Webex-Signature", "")).strip()
        if not webex_sig:
            return False

        expected = {
            "sha1": hmac.new(secret, body, hashlib.sha1).hexdigest(),
            "sha256": hmac.new(secret, body, hashlib.sha256).hexdigest(),
            "sha512": hmac.new(secret, body, hashlib.sha512).hexdigest(),
        }
        for alg, candidate in self._parse_signature_header(webex_sig):
            if alg:
                digest = expected.get(alg)
                if digest and hmac.compare_digest(candidate, digest):
                    return True
                continue
            for digest in (expected["sha256"], expected["sha512"], expected["sha1"]):
                if hmac.compare_digest(candidate, digest):
                    return True
        return False

    @staticmethod
    def _parse_signature_header(value: str) -> list[tuple[Optional[str], str]]:
        parts = []
        for token in re.split(r"[,\s;]+", value):
            token = token.strip()
            if not token:
                continue
            if "=" in token:
                alg, digest = token.split("=", 1)
                parts.append((alg.strip().lower(), digest.strip().lower()))
            else:
                parts.append((None, token.lower()))
        return parts

    # ------------------------------------------------------------------
    # Content helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_path(path: str) -> str:
        path = path.strip() or DEFAULT_PATH
        if not path.startswith("/"):
            path = "/" + path
        return path.rstrip("/") or "/"

    def _build_text_payload(
        self,
        chat_id: str,
        content: str,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        chat_id = str(chat_id or "").strip()
        if not chat_id:
            return None
        payload: dict[str, Any] = {}
        if "@" in chat_id and " " not in chat_id:
            payload["toPersonEmail"] = chat_id
        else:
            payload["roomId"] = chat_id
        parent_id = self._extract_parent_id(metadata)
        if parent_id:
            payload["parentId"] = parent_id
        if content:
            payload["markdown"] = content
        return payload

    def _populate_target_fields(self, form: FormData, chat_id: str) -> None:
        chat_id = str(chat_id or "").strip()
        if "@" in chat_id and " " not in chat_id:
            form.add_field("toPersonEmail", chat_id)
        else:
            form.add_field("roomId", chat_id)

    @staticmethod
    def _extract_parent_id(metadata: Optional[dict[str, Any]]) -> Optional[str]:
        if not metadata:
            return None
        parent_id = metadata.get("thread_id")
        if parent_id:
            return str(parent_id)
        return None

    async def _resolve_edit_room_id(
        self, chat_id: str, message_id: str
    ) -> Optional[str]:
        target = str(chat_id or "").strip()
        if target and "@" not in target:
            return target

        message = await self._api_get_json(f"messages/{message_id}")
        room_id = str((message or {}).get("roomId") or "").strip()
        if room_id:
            return room_id
        return None

    def _strip_bot_mention(self, text: str) -> str:
        if not text:
            return text
        candidate = text
        names = [
            self._bot_display_name,
            self._bot_email.split("@")[0] if self._bot_email else "",
            self._bot_email,
        ]
        for name in names:
            name = name.strip()
            if not name:
                continue
            candidate = re.sub(
                rf"^\s*@?{re.escape(name)}(?:\s*[:,\-]\s*|\s+|[).!?]\s*|$)",
                "",
                candidate,
                flags=re.IGNORECASE,
            )
        return candidate.strip()

    @staticmethod
    def _is_gateway_command(text: str) -> bool:
        text = str(text or "").strip()
        if not text.startswith("/"):
            return False
        command = text.split(maxsplit=1)[0]
        return resolve_command(command) is not None

    def _group_message_mentions_bot(self, message: dict[str, Any]) -> bool:
        mentioned = message.get("mentionedPeople")
        if isinstance(mentioned, list):
            for candidate in mentioned:
                value = str(candidate or "").strip()
                if value in {"me", self._bot_id, self._bot_email}:
                    return True

        haystacks = [
            str(message.get("text") or ""),
            str(message.get("markdown") or ""),
        ]
        names = [
            self._bot_display_name,
            self._bot_email.split("@")[0] if self._bot_email else "",
            self._bot_email,
        ]
        for text in haystacks:
            if not text:
                continue
            for name in names:
                name = name.strip()
                if not name:
                    continue
                if re.search(
                    rf"(^|[\s(])@?{re.escape(name)}(?=$|[\s:,\-).!?])",
                    text,
                    flags=re.IGNORECASE,
                ):
                    return True
        return False

    def _detect_message_type(self, text: str, media_types: list[str]) -> MessageType:
        if text.startswith("/"):
            return MessageType.COMMAND
        if not media_types:
            return MessageType.TEXT
        if any(mt.startswith("image/") for mt in media_types):
            return MessageType.PHOTO
        if any(mt.startswith("audio/") for mt in media_types):
            return MessageType.VOICE
        if any(mt.startswith("video/") for mt in media_types):
            return MessageType.VIDEO
        return MessageType.DOCUMENT

    # ------------------------------------------------------------------
    # Thread context helpers
    # ------------------------------------------------------------------

    def _has_active_session_for_thread(
        self,
        *,
        room_id: str,
        chat_type: str,
        parent_id: str,
        user_id: str,
        user_id_alt: Optional[str] = None,
    ) -> bool:
        """Return True when Hermes already has a session for this Webex thread."""

        session_store = getattr(self, "_session_store", None)
        if not session_store:
            return False

        source = self.build_source(
            chat_id=room_id,
            chat_type=chat_type,
            user_id=user_id,
            thread_id=parent_id,
            user_id_alt=user_id_alt,
        )
        session_key = self._source_session_key(source)
        return session_store.lookup_by_session_key(session_key) is not None

    async def _fetch_thread_context(
        self,
        *,
        room_id: str,
        parent_id: str,
        current_message_id: str,
        chat_type: str,
        limit: int = THREAD_CONTEXT_DEFAULT_LIMIT,
    ) -> str:
        """Fetch prior Webex thread messages for first-contact thread context."""

        cache_key = f"{room_id}:{parent_id}"
        now = time.monotonic()
        cached = self._thread_context_cache.get(cache_key)
        if cached and (now - cached.fetched_at) < THREAD_CONTEXT_CACHE_TTL_SECONDS:
            return cached.content

        parent = await self._api_get_json(f"messages/{parent_id}")
        if not parent:
            return ""

        params = {"roomId": room_id, "parentId": parent_id, "max": limit + 1}
        if chat_type == "group":
            params["mentionedPeople"] = "me"
        replies_result = await self._api_get_json("messages", params=params)
        replies = replies_result.get("items")
        if not isinstance(replies, list):
            replies = []

        parts: list[str] = []
        parent_text = self._message_text(parent)
        parent_line = await self._format_thread_message(parent, is_parent=True)
        if parent_line:
            parts.append(parent_line)

        for reply in reversed(replies):
            if not isinstance(reply, dict):
                continue
            if str(reply.get("id") or "") == current_message_id:
                continue
            if str(reply.get("personId") or "") == self._bot_id:
                continue
            line = await self._format_thread_message(reply, is_parent=False)
            if line:
                parts.append(line)

        content = ""
        if parts:
            content = (
                "[Webex thread context - prior messages in this thread "
                "(not yet in conversation history):]\n"
                + "\n".join(parts)
                + "\n[End of Webex thread context]\n\n"
            )
            if len(content) > THREAD_CONTEXT_MAX_CHARS:
                content = (
                    content[: THREAD_CONTEXT_MAX_CHARS - 18].rstrip()
                    + "\n... [truncated]\n\n"
                )

        self._thread_context_cache[cache_key] = _WebexThreadContextCache(
            content=content,
            fetched_at=now,
            parent_text=parent_text,
        )
        return content

    async def _fetch_thread_parent_text(self, room_id: str, parent_id: str) -> str:
        """Return the raw text of a Webex thread parent for reply context."""

        cache_key = f"{room_id}:{parent_id}"
        now = time.monotonic()
        cached = self._thread_context_cache.get(cache_key)
        if cached and (now - cached.fetched_at) < THREAD_CONTEXT_CACHE_TTL_SECONDS:
            return cached.parent_text

        parent = await self._api_get_json(f"messages/{parent_id}")
        if not parent:
            return ""
        return self._message_text(parent)

    async def _format_thread_message(
        self, message: dict[str, Any], *, is_parent: bool
    ) -> str:
        text = self._message_text(message)
        if not text:
            return ""
        text = self._strip_bot_mention(text)
        if len(text) > 1000:
            text = text[:997].rstrip() + "..."

        person_id = str(message.get("personId") or "")
        person_email = str(message.get("personEmail") or "")
        if person_id == self._bot_id:
            name = self._bot_display_name or self._bot_email or "Hermes"
        else:
            name = await self._lookup_display_name(person_id)
            if not name:
                name = person_email or person_id or "unknown"

        prefix = "[thread parent] " if is_parent else ""
        return f"{prefix}{name}: {text}"

    @staticmethod
    def _message_text(message: dict[str, Any]) -> str:
        return str(message.get("text") or message.get("markdown") or "").strip()

    async def _download_attachments(
        self, files: list[str]
    ) -> tuple[list[str], list[str]]:
        media_urls: list[str] = []
        media_types: list[str] = []
        for file_url in files:
            try:
                local_path, mime = await self._download_attachment(file_url)
                if local_path:
                    media_urls.append(local_path)
                    media_types.append(mime)
            except (TimeoutError, aiohttp.ClientError, OSError, RuntimeError, ValueError) as exc:
                logger.warning(
                    "[Webex] Failed to download attachment %s: %s",
                    safe_url_for_log(file_url),
                    exc,
                )
        return media_urls, media_types

    async def _download_attachment(self, file_url: str) -> tuple[str, str]:
        if not self._session:
            raise RuntimeError("HTTP session is not available")
        async with self._session.get(file_url, headers=self._headers()) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status}")
            mime = (
                resp.headers.get("Content-Type") or "application/octet-stream"
            ).split(";", 1)[0]
            filename = self._attachment_filename(file_url, resp.headers)
            ext = Path(filename).suffix or mimetypes.guess_extension(mime) or ".bin"
            media_type = (
                "image"
                if mime.startswith("image/")
                else "audio"
                if mime.startswith("audio/")
                else "video"
                if mime.startswith("video/")
                else "document"
            )
            data = await self._read_attachment_body(resp, media_type=media_type)
            if mime.startswith("image/"):
                return cache_image_from_bytes(data, ext), mime
            if mime.startswith("audio/"):
                return cache_audio_from_bytes(data, ext), mime
            if mime.startswith("video/"):
                return cache_video_from_bytes(data, ext), mime
            return cache_document_from_bytes(data, filename), mime

    @staticmethod
    async def _read_attachment_body(resp: Any, *, media_type: str) -> bytes:
        """Read an aiohttp body while enforcing the shared inbound-media cap."""
        max_bytes = get_inbound_media_max_bytes()
        content_length = resp.headers.get("Content-Length") or resp.headers.get(
            "content-length"
        )
        if content_length:
            try:
                declared_size = int(content_length)
            except (TypeError, ValueError):
                logger.debug(
                    "[Webex] Ignoring invalid attachment Content-Length: %r",
                    content_length,
                )
            else:
                validate_inbound_media_size(
                    declared_size,
                    media_type=media_type,
                    max_bytes=max_bytes,
                )

        chunks: list[bytes] = []
        total = 0
        async for chunk in resp.content.iter_chunked(ATTACHMENT_READ_CHUNK_SIZE):
            total += len(chunk)
            validate_inbound_media_size(
                total,
                media_type=media_type,
                max_bytes=max_bytes,
            )
            chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    def _attachment_filename(file_url: str, headers: Any) -> str:
        content_disp = str(headers.get("Content-Disposition", ""))
        match = re.search(r'filename="?([^";]+)"?', content_disp)
        if match:
            return Path(match.group(1)).name
        path_name = Path(urlsplit(file_url).path).name
        return path_name or "attachment"


def _env_enablement() -> Optional[dict[str, Any]]:
    token = _get_scoped_secret("WEBEX_BOT_TOKEN", "").strip()
    if not token:
        return None
    return seed_extra_from_env(
        (
            ("WEBEX_CONNECTION_MODE", "connection_mode", None),
            ("WEBEX_WEBHOOK_PUBLIC_URL", "public_url", None),
            ("WEBEX_WEBHOOK_SECRET", "secret", None),
            ("WEBEX_WEBHOOK_HOST", "host", None),
            ("WEBEX_WEBHOOK_PORT", "port", int),
            ("WEBEX_WEBHOOK_PATH", "path", None),
        ),
        home_env="WEBEX_HOME_CHANNEL",
    )


def _validate_config(config: PlatformConfig) -> bool:
    if not (config.token or _get_scoped_secret("WEBEX_BOT_TOKEN", "")):
        return False
    extra = config.extra or {}
    mode = _configured_connection_mode(config)
    if mode not in {"websocket", "webhook"}:
        return False
    if mode == "webhook":
        public_url = str(
            extra.get("public_url")
            or _get_scoped_secret("WEBEX_WEBHOOK_PUBLIC_URL", "")
        )
        secret = str(
            extra.get("secret") or _get_scoped_secret("WEBEX_WEBHOOK_SECRET", "")
        )
        return public_url.startswith("https://") and bool(secret)
    return True


def _configured_connection_mode(config: PlatformConfig) -> str:
    return str(
        (config.extra or {}).get("connection_mode")
        or _get_scoped_secret("WEBEX_CONNECTION_MODE", DEFAULT_CONNECTION_MODE)
    ).strip().lower() or DEFAULT_CONNECTION_MODE


def _is_port_binding(config: PlatformConfig) -> bool:
    return _configured_connection_mode(config) == "webhook"


async def _standalone_send(
    pconfig: PlatformConfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[list[tuple[str, bool]]] = None,
    force_document: bool = False,
) -> dict[str, Any]:
    del force_document
    token = pconfig.token or _get_scoped_secret("WEBEX_BOT_TOKEN", "")
    if not token:
        return {"error": "WEBEX_BOT_TOKEN is not configured"}
    target_field = (
        "toPersonEmail" if "@" in chat_id and " " not in chat_id else "roomId"
    )
    headers = {"Authorization": f"Bearer {token}"}
    delivered_files = 0
    sent_messages = 0

    async def post_message(session, text, media_path=None):
        payload = {target_field: chat_id}
        if text or media_path is None:
            payload["markdown"] = text
        if thread_id:
            payload["parentId"] = thread_id
        handle = None
        try:
            if media_path is not None:
                form = FormData()
                for key, value in payload.items():
                    form.add_field(key, value)
                handle = await asyncio.to_thread(open, media_path, "rb")
                form.add_field("files", handle, filename=Path(media_path).name)
                response = await session.post(f"{API_BASE}/messages", headers=headers, data=form)
            else:
                response = await session.post(f"{API_BASE}/messages", headers=headers, json=payload)
            async with response:
                if response.status >= 400:
                    raise _WebexAPIRequestError("messages", response.status)
                return await response.json(content_type=None)
        finally:
            if handle is not None:
                handle.close()

    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            chunks = split_message(message)
            for chunk in chunks[:-1] if media_files else chunks:
                body = await post_message(session, chunk)
                sent_messages += 1
            # Webex permits one file per message, including multipart uploads:
            # https://github.com/WebexCommunity/WebexPythonSDK/blob/main/src/webexpythonsdk/api/messages.py
            for index, (path, _is_voice) in enumerate(media_files or []):
                body = await post_message(session, chunks[-1] if index == 0 else "", path)
                delivered_files += 1
                sent_messages += 1
        result = {"success": True, "message_id": body.get("id", "")}
        if media_files:
            result["media_delivered"] = True
        return result
    except (TimeoutError, aiohttp.ClientError, OSError, ValueError, _WebexAPIRequestError) as exc:
        detail = f"Webex HTTP {exc.status}" if isinstance(exc, _WebexAPIRequestError) else str(exc)
        result = {"error": detail}
        if media_files:
            result.update(error=f"Webex media delivery failed after {delivered_files}/{len(media_files)} files: {detail}",
                          media_delivered=False)
        if sent_messages:
            result["partial_success"] = True
        return result


def register(ctx: Any) -> None:
    ctx.register_platform(
        name="webex",
        label="Webex",
        adapter_factory=lambda cfg: WebexAdapter(cfg),
        check_fn=check_webex_requirements,
        validate_config=_validate_config,
        is_connected=_validate_config,
        required_env=["WEBEX_BOT_TOKEN"],
        install_hint="reinstall Hermes with messaging dependencies",
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="WEBEX_HOME_CHANNEL",
        allowed_users_env="WEBEX_ALLOWED_USERS",
        allow_all_env="WEBEX_ALLOW_ALL_USERS",
        allowlist_user_id_alt=True,
        # Live and standalone senders split in UTF-8 bytes; the host splitter counts characters.
        max_message_length=0,
        platform_hint=(
            "You are chatting via Webex. Markdown and threaded replies are supported."
        ),
        standalone_sender_fn=_standalone_send,
        standalone_sender_supports_media=True,
        is_port_binding_fn=_is_port_binding,
        pii_safe=True,
        emoji="🟢",
    )
