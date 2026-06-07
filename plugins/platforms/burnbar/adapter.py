"""BurnBar Cloud platform adapter for Hermes Agent.

The adapter uses the BurnBar Hermes Gateway API:

    https://api.burnbar.ai/v1/hermes-gateway

It intentionally depends only on ``httpx``, already present in Hermes core.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import mimetypes
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import display_hermes_home, get_hermes_home

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover - Hermes installs httpx in core.
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult

logger = logging.getLogger(__name__)

DEFAULT_API_BASE_URL = "https://api.burnbar.ai/v1/hermes-gateway"
DEFAULT_HOME_CHANNEL = "burnbar:home"
MAX_MESSAGE_LENGTH = 64000
CURSOR_FILE = Path(
    os.getenv("HERMES_BURNBAR_CURSOR_FILE", str(get_hermes_home() / "cache" / "burnbar_cursor.json"))
).expanduser()
MAX_ATTACHMENT_BYTES = 50 * 1024 * 1024
MAX_RUNTIME_MODELS = 100
RUNTIME_STATUS_INTERVAL_SECONDS = 30.0
# How often to refresh the human-in-the-loop oversight toggle from /state.
OVERSIGHT_REFRESH_SECONDS = 15.0
_SAFE_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,179}$")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"\b(access[_-]?token|api[_-]?key|auth[_-]?token|device[_-]?secret|signature|sig|token)\s*=\s*([^\s,;&]+)",
    re.IGNORECASE,
)
_SECRET_JSON_RE = re.compile(
    r'("(?:accessToken|deviceSecret|uploadURL|token)"\s*:\s*")[^"]+(")',
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"\bBearer\s+[^,\s]+", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s'\"<>]+")


def _is_safe_model_id(model_id: str) -> bool:
    """True for model ids that are safe to interpolate into ``/model <id>``.

    Rejects empty ids, ids outside the conservative character class
    (``_SAFE_MODEL_ID``), and — critically — any id containing a ``--`` run.
    The double-dash guard matters because the ``/model`` slash parser
    (:func:`hermes_cli.model_switch.parse_model_flags`) matches ``--global`` and
    ``--refresh`` as *substrings* rather than whitespace-delimited tokens, so a
    no-whitespace id such as ``sonnet--global`` would otherwise pass the
    character class yet smuggle a persistent global-config flag into the slash
    command. Real model ids never contain consecutive dashes, so this rejects
    nothing legitimate.
    """
    if not model_id or "--" in model_id:
        return False
    return bool(_SAFE_MODEL_ID.fullmatch(model_id))


def _safe_exception_message(exc: BaseException) -> str:
    """Return a concise error string without bearer tokens or signed URLs."""
    text = str(exc) or exc.__class__.__name__
    text = _SECRET_JSON_RE.sub(r"\1[redacted]\2", text)
    text = _SECRET_ASSIGNMENT_RE.sub(lambda m: f"{m.group(1)}=[redacted]", text)
    text = _BEARER_RE.sub("Bearer [redacted]", text)
    text = _URL_RE.sub("[redacted-url]", text)
    return text[:500]


def _agent_version() -> str:
    """Best-effort Hermes Agent build string for truthful gateway-version display.

    Reads the installed hermes_agent/hermes_cli version when available; falls back
    to an env override or empty (the server simply omits the version then).
    """
    override = os.getenv("HERMES_BURNBAR_AGENT_VERSION")
    if override:
        return override[:120]
    for module_name in ("hermes_agent", "hermes_cli", "hermes"):
        try:
            from importlib import metadata as _metadata

            return f"{module_name}/{_metadata.version(module_name)}"[:120]
        except Exception:
            continue
    return ""


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _api_base(config: PlatformConfig | None = None) -> str:
    extra = getattr(config, "extra", {}) or {}
    return (extra.get("api_base_url") or os.getenv("BURNBAR_API_BASE_URL") or DEFAULT_API_BASE_URL).rstrip("/")


def _access_token(config: PlatformConfig | None = None) -> str:
    extra = getattr(config, "extra", {}) or {}
    return (extra.get("access_token") or os.getenv("BURNBAR_ACCESS_TOKEN") or "").strip()


def _home_channel(config: PlatformConfig | None = None) -> str:
    extra = getattr(config, "extra", {}) or {}
    return (extra.get("home_channel") or os.getenv("BURNBAR_HOME_CHANNEL") or DEFAULT_HOME_CHANNEL).strip()


def check_requirements() -> bool:
    return HTTPX_AVAILABLE and bool(_access_token())


def validate_config(config: PlatformConfig) -> bool:
    return bool(_access_token(config))


def is_connected(config: PlatformConfig) -> bool:
    return bool(_access_token(config))


def _home_channel_payload(config: PlatformConfig | None = None) -> dict:
    home = _home_channel(config)
    return {"chat_id": home, "name": os.getenv("BURNBAR_HOME_CHANNEL_NAME") or "BurnBar Home"}


def _env_enablement() -> Optional[dict]:
    token = _access_token()
    if not token:
        return None
    return {
        "api_base_url": _api_base(),
        "access_token": token,
        "home_channel": _home_channel_payload(),
    }


def _headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "hermes-agent-burnbar-platform/1.0",
    }


def _read_cursor() -> int:
    try:
        data = json.loads(CURSOR_FILE.read_text())
        value = int(data.get("cursor", 0))
        return value if value > 0 else 0
    except Exception:
        return 0


def _write_cursor(cursor: int) -> None:
    CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    CURSOR_FILE.write_text(json.dumps({"cursor": cursor}))


def _guess_content_type(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


async def _init_attachment(
    client: "httpx.AsyncClient",
    *,
    api_base: str,
    token: str,
    destination_id: str,
    file_path: Path,
    content_type: str,
) -> tuple[str, str]:
    byte_count = file_path.stat().st_size
    if byte_count < 1:
        raise ValueError(f"{file_path} is empty")
    if byte_count > MAX_ATTACHMENT_BYTES:
        raise ValueError(f"{file_path} exceeds BurnBar's {MAX_ATTACHMENT_BYTES} byte attachment limit")

    response = await client.post(
        f"{api_base}/attachments/init",
        headers=_headers(token),
        json={
            "destinationId": destination_id,
            "fileName": file_path.name,
            "contentType": content_type,
            "byteCount": byte_count,
        },
    )
    response.raise_for_status()
    payload = response.json()
    attachment = payload.get("attachment") or {}
    attachment_id = attachment.get("id")
    upload_url = payload.get("uploadURL")
    if not attachment_id or not upload_url:
        raise RuntimeError("BurnBar attachment init response was missing attachment.id or uploadURL")
    return str(attachment_id), str(upload_url)


async def _upload_attachment(
    client: "httpx.AsyncClient",
    *,
    upload_url: str,
    file_path: Path,
    content_type: str,
) -> None:
    data = file_path.read_bytes()
    response = await client.put(upload_url, content=data, headers={"Content-Type": content_type})
    response.raise_for_status()


async def _create_attachments(
    client: "httpx.AsyncClient",
    *,
    api_base: str,
    token: str,
    destination_id: str,
    media_files: list | None,
) -> list[str]:
    attachment_ids: list[str] = []
    for item in media_files or []:
        raw_path = item[0] if isinstance(item, (tuple, list)) else item
        file_path = Path(str(raw_path)).expanduser()
        if not file_path.is_file():
            raise FileNotFoundError(f"Attachment not found: {file_path}")
        content_type = _guess_content_type(file_path)
        attachment_id, upload_url = await _init_attachment(
            client,
            api_base=api_base,
            token=token,
            destination_id=destination_id,
            file_path=file_path,
            content_type=content_type,
        )
        await _upload_attachment(client, upload_url=upload_url, file_path=file_path, content_type=content_type)
        attachment_ids.append(attachment_id)
    return attachment_ids


async def _post_message(
    client: "httpx.AsyncClient",
    *,
    api_base: str,
    token: str,
    destination_id: str,
    text: str,
    thread_id: str | None = None,
    reply_to: str | None = None,
    attachment_ids: list[str] | None = None,
) -> dict:
    response = await client.post(
        f"{api_base}/messages",
        headers=_headers(token),
        json={
            "destinationId": destination_id,
            "threadId": thread_id,
            "replyToEventId": reply_to,
            "text": text[:MAX_MESSAGE_LENGTH],
            "attachmentIds": attachment_ids or [],
        },
    )
    response.raise_for_status()
    return response.json().get("message", {})


def _runtime_status_payload() -> dict:
    """Build a compact current-model/catalog status payload for BurnBar.

    The gateway adapter sits inside Hermes Agent, so it can reuse Hermes'
    own curated model inventory instead of inventing a second catalog. If the
    local checkout is older or inventory probing fails, the adapter still
    works as a messaging platform; it just skips catalog publication.
    """
    try:
        from hermes_cli.inventory import build_models_payload, load_picker_context

        payload = build_models_payload(load_picker_context(), max_models=MAX_RUNTIME_MODELS)
    except Exception:
        logger.debug("[%s] Could not build Hermes model inventory", "burnbar", exc_info=True)
        return {}

    options: list[dict[str, str]] = []
    for provider in payload.get("providers") or []:
        if not isinstance(provider, dict):
            continue
        provider_id = str(provider.get("slug") or provider.get("provider") or "hermes").strip() or "hermes"
        provider_name = str(provider.get("label") or provider.get("name") or provider_id).strip() or provider_id
        for model in provider.get("models") or []:
            if isinstance(model, dict):
                model_id = str(model.get("id") or model.get("model") or model.get("name") or "").strip()
                display_name = str(model.get("display_name") or model.get("displayName") or model_id).strip()
            else:
                model_id = str(model or "").strip()
                display_name = model_id
            if not model_id:
                continue
            options.append(
                {
                    "providerId": provider_id[:80],
                    "providerName": provider_name[:120],
                    "modelId": model_id[:180],
                    "displayName": (display_name or model_id)[:180],
                }
            )
            if len(options) >= MAX_RUNTIME_MODELS:
                break
        if len(options) >= MAX_RUNTIME_MODELS:
            break

    current_model = str(payload.get("model") or "").strip()
    current_provider = str(payload.get("provider") or "").strip()
    body: dict[str, object] = {"modelOptions": options}
    if current_model:
        body["currentModelId"] = current_model[:180]
    if current_provider:
        body["currentProviderId"] = current_provider[:80]
    agent_version = _agent_version()
    if agent_version:
        body["agentVersion"] = agent_version
    return body


class BurnBarAdapter(BasePlatformAdapter):
    """BurnBar Cloud adapter backed by the BurnBar Hermes Gateway API."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config=config, platform=Platform("burnbar"))
        self._api_base = _api_base(config)
        self._token = _access_token(config)
        self._home_channel = _home_channel(config)
        self._client: Optional["httpx.AsyncClient"] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._cursor = _read_cursor()
        self._last_runtime_publish = 0.0
        # Human-in-the-loop oversight. The toggle lives on the server (the phone
        # sets it); the adapter mirrors it here and obeys it. Default is the safe
        # option (supervised) until /state says otherwise.
        self._oversight_mode = "supervised"
        self._oversight_checked_at = 0.0
        # Armed approval gates awaiting a phone decision: actionId -> context.
        self._pending_confirms: Dict[str, Dict[str, Any]] = {}

    async def connect(self) -> bool:
        if not HTTPX_AVAILABLE:
            logger.warning("[%s] httpx is unavailable", self.name)
            return False
        if not self._token:
            logger.warning("[%s] BURNBAR_ACCESS_TOKEN is not configured", self.name)
            return False
        # A send path (e.g. _send_local_file) may have lazily created a client
        # before connect(); close it first so reconnects don't leak sockets.
        if self._client is not None:
            await self._client.aclose()
        self._client = httpx.AsyncClient(timeout=30)
        try:
            response = await self._client.get(f"{self._api_base}/destinations", headers=_headers(self._token))
            response.raise_for_status()
        except Exception as exc:
            logger.warning("[%s] BurnBar connection check failed: %s", self.name, _safe_exception_message(exc))
            await self.disconnect()
            return False
        await self._publish_runtime_status(force=True)
        self._poll_task = asyncio.create_task(self._poll_loop())
        self._mark_connected()
        logger.info("[%s] Connected to BurnBar Cloud", self.name)
        return True

    async def disconnect(self) -> None:
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        self._poll_task = None
        if self._client:
            await self._client.aclose()
        self._client = None
        self._mark_disconnected()

    async def _poll_loop(self) -> None:
        backoff = 1.0
        while self._running:
            try:
                await self._poll_once()
                backoff = 1.0
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[%s] BurnBar event poll failed: %s", self.name, exc)
                await asyncio.sleep(backoff)
                backoff = min(30.0, backoff * 2)

    async def _poll_once(self) -> None:
        assert self._client is not None
        await self._publish_runtime_status()
        await self._refresh_oversight_mode()
        # Resolve any oversight gates the phone has decided since the last poll.
        if self._pending_confirms:
            await self._resolve_pending_confirms()
        response = await self._client.get(
            f"{self._api_base}/events",
            headers=_headers(self._token),
            params={"cursor": str(self._cursor), "limit": "50"},
        )
        response.raise_for_status()
        payload = response.json()
        for raw in payload.get("events", []):
            # Isolate each event: one malformed/poison event must not abort the
            # batch (which would leave the cursor unadvanced and replay the whole
            # page forever). Log and skip it; siblings still process and the
            # cursor advances past it below.
            try:
                await self._handle_burnbar_event(raw)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "[%s] dropped malformed BurnBar event %r",
                    self.name,
                    raw.get("id") if isinstance(raw, dict) else None,
                    exc_info=True,
                )
        next_cursor = int(payload.get("nextCursor") or self._cursor)
        if next_cursor > self._cursor:
            self._cursor = next_cursor
            _write_cursor(self._cursor)

    async def _handle_burnbar_event(self, raw: dict) -> None:
        is_model_switch = raw.get("kind") == "model_switch"
        if is_model_switch:
            model_id = str(raw.get("modelId") or "").strip()
            if not _is_safe_model_id(model_id):
                logger.warning("[%s] dropped model_switch with unsafe modelId %r", self.name, model_id)
                return
            text = f"/model {model_id}".strip()
        else:
            text = str(raw.get("text") or "").strip()
        if not text:
            return
        destination_id = str(raw.get("destinationId") or self._home_channel)
        sender_id = str(raw.get("senderId") or "burnbar-user")
        source = self.build_source(
            chat_id=destination_id,
            chat_name=destination_id,
            chat_type="dm",
            user_id=sender_id,
            user_name=raw.get("senderDisplayName") or sender_id,
            thread_id=raw.get("threadId"),
            message_id=raw.get("id"),
        )
        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=raw,
            message_id=raw.get("id"),
        )
        await self.handle_message(event)
        if is_model_switch:
            # Republish immediately so the server reflects the newly applied model
            # in ~1s instead of waiting out the 30s heartbeat. Also reset the
            # throttle so the next poll re-confirms once Hermes has fully applied.
            self._last_runtime_publish = 0.0
            await self._publish_runtime_status(force=True)

    async def _publish_runtime_status(self, *, force: bool = False) -> None:
        if self._client is None:
            return
        now = time.monotonic()
        if not force and now - self._last_runtime_publish < RUNTIME_STATUS_INTERVAL_SECONDS:
            return
        body = _runtime_status_payload()
        if not body:
            self._last_runtime_publish = now
            return
        try:
            response = await self._client.post(
                f"{self._api_base}/runtime",
                headers=_headers(self._token),
                json=body,
            )
            response.raise_for_status()
            self._last_runtime_publish = now
        except Exception:
            logger.debug("[%s] BurnBar runtime status publish failed", self.name, exc_info=True)

    # ------------------------------------------------------------------
    # Human-in-the-loop oversight
    # ------------------------------------------------------------------
    async def _refresh_oversight_mode(self) -> None:
        """Mirror the server-owned oversight toggle (the phone sets it)."""
        if self._client is None:
            return
        now = time.monotonic()
        if now - self._oversight_checked_at < OVERSIGHT_REFRESH_SECONDS:
            return
        self._oversight_checked_at = now
        try:
            response = await self._client.get(f"{self._api_base}/state", headers=_headers(self._token))
            response.raise_for_status()
            mode = str(response.json().get("oversightMode") or "").strip()
            if mode in ("supervised", "autonomous"):
                self._oversight_mode = mode
        except Exception:
            logger.debug("[%s] BurnBar oversight refresh failed", self.name, exc_info=True)

    async def send_slash_confirm(
        self,
        chat_id: str,
        title: str,
        message: str,
        session_key: str,
        confirm_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Gate Hermes slash-confirm prompts through BurnBar oversight.

        Autonomous mode auto-approves so the agent runs unattended. Supervised
        mode arms an approval gate on the BurnBar gateway and surfaces it to the
        phone; the decision is applied from the poll loop via
        ``tools.slash_confirm.resolve``. If the gateway is unreachable while
        supervised, we fall back to Hermes' built-in text confirm (which still
        requires an explicit ``/approve``) rather than silently proceeding.
        """
        if self._client is None:
            return SendResult(success=False, error="Not connected")
        if self._oversight_mode == "autonomous":
            await self._resolve_slash_confirm(session_key, confirm_id, "once", chat_id, metadata)
            return SendResult(success=True)
        armed = await self._arm_approval(
            action_id=confirm_id, summary=message, tool_name=title, destination_id=chat_id
        )
        if not armed:
            return await super().send_slash_confirm(
                chat_id, title, message, session_key, confirm_id, metadata
            )
        self._pending_confirms[confirm_id] = {
            "session_key": session_key,
            "chat_id": chat_id,
            "metadata": metadata,
        }
        card = f"{title}\n\n{message}\n\nApprove this action on your BurnBar device to continue."
        await self._post_confirm_followup(chat_id, card, metadata)
        return SendResult(success=True)

    async def _arm_approval(
        self, *, action_id: str, summary: str, tool_name: str, destination_id: str
    ) -> bool:
        if self._client is None:
            return False
        body: Dict[str, Any] = {"actionId": action_id, "summary": summary}
        if tool_name:
            body["toolName"] = tool_name
        if destination_id:
            body["destinationId"] = destination_id
        try:
            response = await self._client.post(
                f"{self._api_base}/approvals", headers=_headers(self._token), json=body
            )
            response.raise_for_status()
            return True
        except Exception:
            logger.debug("[%s] BurnBar approval arm failed", self.name, exc_info=True)
            return False

    async def _resolve_pending_confirms(self) -> None:
        if self._client is None:
            return
        for action_id, ctx in list(self._pending_confirms.items()):
            try:
                response = await self._client.get(
                    f"{self._api_base}/approvals",
                    headers=_headers(self._token),
                    params={"actionId": action_id},
                )
                if response.status_code == 404:
                    logger.debug(
                        "[%s] BurnBar approval %s not found while polling; retaining pending confirm",
                        self.name,
                        action_id,
                    )
                    continue
                response.raise_for_status()
                status = str((response.json().get("approval") or {}).get("status") or "").strip()
            except Exception:
                logger.debug("[%s] BurnBar approval poll failed", self.name, exc_info=True)
                continue
            if status == "waiting_for_approval":
                continue
            self._pending_confirms.pop(action_id, None)
            choice = "once" if status == "approved" else "cancel"
            fallback = None
            if status == "rejected":
                fallback = "Action denied from your BurnBar device."
            elif status == "expired":
                fallback = "Approval request expired without a decision."
            await self._resolve_slash_confirm(
                ctx["session_key"], action_id, choice, ctx.get("chat_id"), ctx.get("metadata"), fallback
            )

    async def _resolve_slash_confirm(
        self,
        session_key: str,
        confirm_id: str,
        choice: str,
        chat_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
        fallback: Optional[str] = None,
    ) -> None:
        output: Optional[str] = None
        try:
            from tools import slash_confirm as _slash_confirm

            output = await _slash_confirm.resolve(session_key, confirm_id, choice)
        except Exception:
            logger.debug("[%s] slash-confirm resolve failed", self.name, exc_info=True)
        message = output or fallback
        if message and chat_id:
            await self._post_confirm_followup(chat_id, message, metadata)

    async def _post_confirm_followup(
        self, chat_id: Optional[str], text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        if self._client is None or not text:
            return
        try:
            await self._client.post(
                f"{self._api_base}/messages",
                headers=_headers(self._token),
                json={
                    "destinationId": chat_id or self._home_channel,
                    "text": str(text)[:MAX_MESSAGE_LENGTH],
                },
            )
        except Exception:
            logger.debug("[%s] BurnBar confirm follow-up post failed", self.name, exc_info=True)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30)
        destination_id = chat_id or self._home_channel
        try:
            message = await _post_message(
                self._client,
                api_base=self._api_base,
                token=self._token,
                destination_id=destination_id,
                text=content,
                thread_id=(metadata or {}).get("thread_id"),
                reply_to=reply_to,
            )
            return SendResult(success=True, message_id=message.get("id"))
        except Exception as exc:
            error = _safe_exception_message(exc)
            logger.warning("[%s] BurnBar send failed: %s", self.name, error)
            return SendResult(success=False, error=error)

    async def _send_local_file(
        self,
        chat_id: str,
        file_path: str,
        *,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30)
        destination_id = chat_id or self._home_channel
        try:
            attachment_ids = await _create_attachments(
                self._client,
                api_base=self._api_base,
                token=self._token,
                destination_id=destination_id,
                media_files=[file_path],
            )
            message = await _post_message(
                self._client,
                api_base=self._api_base,
                token=self._token,
                destination_id=destination_id,
                text=caption or "",
                thread_id=(metadata or {}).get("thread_id"),
                reply_to=reply_to,
                attachment_ids=attachment_ids,
            )
            return SendResult(success=True, message_id=message.get("id"))
        except Exception as exc:
            error = _safe_exception_message(exc)
            logger.warning("[%s] BurnBar attachment send failed: %s", self.name, error)
            return SendResult(success=False, error=error)

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
        return await self._send_local_file(chat_id, file_path, caption=caption, reply_to=reply_to, metadata=metadata)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_local_file(chat_id, image_path, caption=caption, reply_to=reply_to, metadata=metadata)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_local_file(chat_id, audio_path, caption=caption, reply_to=reply_to, metadata=metadata)

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_local_file(chat_id, video_path, caption=caption, reply_to=reply_to, metadata=metadata)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=15)
        try:
            await self._client.post(
                f"{self._api_base}/typing",
                headers=_headers(self._token),
                json={"destinationId": chat_id or self._home_channel, "threadId": (metadata or {}).get("thread_id")},
            )
        except Exception:
            logger.debug("[%s] BurnBar typing failed", self.name, exc_info=True)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"chat_id": chat_id, "name": chat_id or "BurnBar Home", "type": "dm"}


async def _standalone_send(
    pconfig,
    chat_id,
    message,
    *,
    thread_id=None,
    media_files=None,
    force_document=False,
) -> dict:
    if not HTTPX_AVAILABLE:
        return {"error": "httpx is not available"}
    token = _access_token(pconfig)
    if not token:
        return {"error": "BURNBAR_ACCESS_TOKEN is not configured"}
    destination_id = chat_id or _home_channel(pconfig)
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            attachment_ids = await _create_attachments(
                client,
                api_base=_api_base(pconfig),
                token=token,
                destination_id=destination_id,
                media_files=media_files,
            )
            posted = await _post_message(
                client,
                api_base=_api_base(pconfig),
                token=token,
                destination_id=destination_id,
                thread_id=thread_id,
                text=message,
                attachment_ids=attachment_ids,
            )
        except Exception as exc:
            return {"error": f"BurnBar standalone send failed: {_safe_exception_message(exc)}"}
        return {
            "success": True,
            "platform": "burnbar",
            "chat_id": destination_id,
            "message_id": posted.get("id"),
            "attachment_ids": attachment_ids,
        }


def _apply_yaml_config(yaml_cfg: dict, platform_cfg: dict) -> Optional[dict]:
    """Translate BurnBar's config.yaml keys into PlatformConfig.extra.

    Environment variables still win; this hook only lets users configure
    BurnBar in structured YAML without core Hermes knowing BurnBar-specific
    field names.
    """
    extra = dict(platform_cfg.get("extra") or {})
    for yaml_key, env_key, extra_key in (
        ("api_base_url", "BURNBAR_API_BASE_URL", "api_base_url"),
        ("access_token", "BURNBAR_ACCESS_TOKEN", "access_token"),
        ("home_channel", "BURNBAR_HOME_CHANNEL", "home_channel"),
    ):
        value = yaml_cfg.get(yaml_key)
        if value is None:
            continue
        value = str(value).strip()
        if not value:
            continue
        extra[extra_key] = value
        os.environ.setdefault(env_key, value)
    return extra or None


def _poll_device_authorization(
    api_base: str,
    device_code: str,
    device_secret: str,
    interval: int,
    timeout_seconds: float = 600.0,
) -> dict:
    # Bound the wait so `hermes gateway setup` cannot hang forever on a stuck
    # `pending` if the server never reports denied/expired. The deadline tracks the
    # grant's own `expiresIn` (capped at a sane default by the caller).
    deadline = time.monotonic() + max(1.0, timeout_seconds)
    with httpx.Client(timeout=30) as client:
        while True:
            poll = client.post(
                f"{api_base}/device/poll",
                json={"deviceCode": device_code, "deviceSecret": device_secret},
            )
            poll.raise_for_status()
            status = poll.json()
            if status.get("status") == "approved":
                return status
            if status.get("status") in {"denied", "expired"}:
                raise RuntimeError(f"BurnBar link {status['status']}")
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "BurnBar link approval timed out; re-run `hermes gateway setup` and "
                    "approve the device code in BurnBar before it expires"
                )
            time.sleep(interval)


def _configure_interactive_access_control(
    *,
    get_env_value,
    print_info,
    print_success,
    print_warning,
    prompt,
    prompt_yes_no,
    save_env_value,
) -> None:
    if get_env_value("BURNBAR_ALLOWED_USERS") or get_env_value("BURNBAR_ALLOW_ALL_USERS"):
        return

    print()
    print_info("Access control")
    print_info("BurnBar senders are denied unless you configure an allowlist or explicitly allow all.")
    allow_all = prompt_yes_no(
        "Allow every BurnBar sender in this workspace to talk to Hermes?",
        False,
    )
    if allow_all:
        save_env_value("BURNBAR_ALLOW_ALL_USERS", "true")
        save_env_value("BURNBAR_ALLOWED_USERS", "")
        print_warning(
            "Open access enabled for BurnBar. Any sender in the connected workspace can command Hermes."
        )
        return

    save_env_value("BURNBAR_ALLOW_ALL_USERS", "false")
    allowed = prompt(
        "Allowed BurnBar sender IDs (comma-separated, leave empty to deny everyone)",
        default="",
    )
    allowed_ids = ",".join(part.strip() for part in allowed.split(",") if part.strip())
    if allowed_ids:
        save_env_value("BURNBAR_ALLOWED_USERS", allowed_ids)
        print_success("BurnBar sender allowlist configured.")
    else:
        print_warning(
            "No BurnBar sender allowlist configured. Unknown BurnBar senders will be denied "
            "until BURNBAR_ALLOWED_USERS is set or BURNBAR_ALLOW_ALL_USERS=true is selected."
        )


def interactive_setup() -> None:
    """Device-code setup helper for ``hermes gateway setup``."""
    from hermes_cli.setup import (
        get_env_value,
        print_header,
        print_info,
        print_success,
        print_warning,
        prompt,
        prompt_yes_no,
        save_env_value,
    )

    print_header("BurnBar Cloud")
    existing_token = get_env_value("BURNBAR_ACCESS_TOKEN")
    if existing_token:
        print_info("BurnBar Cloud is already configured.")
        if not prompt_yes_no("Reconfigure BurnBar Cloud?", False):
            return

    api_base = (
        prompt(
            "BurnBar Hermes Gateway API base URL",
            default=get_env_value("BURNBAR_API_BASE_URL") or DEFAULT_API_BASE_URL,
        ).strip()
        or DEFAULT_API_BASE_URL
    ).rstrip("/")

    device_secret = secrets.token_urlsafe(32)
    payload = {
        "clientName": "Hermes Agent",
        "deviceSecretHash": _sha256(device_secret),
        "scopes": ["hermes.gateway.read", "hermes.gateway.write", "hermes.gateway.manage"],
    }
    try:
        with httpx.Client(timeout=30) as client:
            start = client.post(f"{api_base}/device/start", json=payload)
            start.raise_for_status()
            body = start.json()
    except Exception as exc:
        print_warning(f"Could not start BurnBar device authorization: {exc}")
        return

    print()
    print_info("Open BurnBar, sign in, and approve this code:")
    print_info(f"  {body['userCode']}")
    print_info(f"  {body['verificationUriComplete']}")
    print_info("Waiting for approval...")

    try:
        approved = _poll_device_authorization(
            api_base,
            body["deviceCode"],
            device_secret,
            int(body.get("interval", 3)),
            timeout_seconds=float(body.get("expiresIn", 600)),
        )
    except Exception as exc:
        print_warning(f"BurnBar authorization failed: {exc}")
        return

    save_env_value("BURNBAR_API_BASE_URL", api_base)
    save_env_value("BURNBAR_ACCESS_TOKEN", approved["accessToken"])
    save_env_value("BURNBAR_HOME_CHANNEL", approved.get("homeDestinationId") or DEFAULT_HOME_CHANNEL)
    _configure_interactive_access_control(
        get_env_value=get_env_value,
        print_info=print_info,
        print_success=print_success,
        print_warning=print_warning,
        prompt=prompt,
        prompt_yes_no=prompt_yes_no,
        save_env_value=save_env_value,
    )
    print_success(f"BurnBar Cloud configuration saved to {display_hermes_home()}/.env")
    print_info("Restart the gateway for changes to take effect: hermes gateway restart")


def register(ctx) -> None:
    ctx.register_platform(
        name="burnbar",
        label="BurnBar Cloud",
        adapter_factory=lambda cfg: BurnBarAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["BURNBAR_ACCESS_TOKEN"],
        install_hint="Configure from BurnBar Cloud with `hermes gateway setup`.",
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        apply_yaml_config_fn=_apply_yaml_config,
        allowed_users_env="BURNBAR_ALLOWED_USERS",
        allow_all_env="BURNBAR_ALLOW_ALL_USERS",
        cron_deliver_env_var="BURNBAR_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        supports_standalone_media=True,
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="🔥",
        platform_hint=(
            "You are speaking through BurnBar Cloud. Keep replies concise, "
            "mobile-friendly, and explicit about completed actions. Use plain "
            "Markdown that renders cleanly in compact app chat surfaces."
        ),
        allow_update_command=True,
    )
