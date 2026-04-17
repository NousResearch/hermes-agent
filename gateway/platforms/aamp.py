"""AAMP platform adapter backed by the Python AAMP SDK."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.platforms.helpers import strip_markdown
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://meshmail.ai"
DEFAULT_SLUG = "hermes"
DEFAULT_RECONNECT_INTERVAL = 10.0
DEFAULT_CREDENTIALS_PATH = "aamp/mailbox_identity.json"
DEFAULT_DESCRIPTION = "Hermes Agent AAMP mailbox"
MAX_MESSAGE_LENGTH = 50_000


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in ("true", "1", "yes", "on")


def _normalize_base_url(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        return ""
    if not value.startswith(("http://", "https://")):
        value = f"http://{value}"
    return value.rstrip("/")


def _resolve_credentials_path(raw_path: str | None) -> Path:
    value = (raw_path or "").strip()
    if not value:
        value = DEFAULT_CREDENTIALS_PATH
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = get_hermes_home() / path
    return path


def _build_mailbox_token(email: str, password: str) -> str:
    raw = f"{email}:{password}".encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def _decode_mailbox_token(token: str) -> tuple[str, str]:
    try:
        decoded = base64.b64decode(token).decode("utf-8")
    except Exception as exc:
        raise ValueError(f"Invalid AAMP mailbox token: {exc}") from exc
    if ":" not in decoded:
        raise ValueError("Invalid AAMP mailbox token: expected base64(email:password)")
    email, password = decoded.split(":", 1)
    if not email or not password:
        raise ValueError("Invalid AAMP mailbox token: empty email or password")
    return email, password


@dataclass
class AampIdentity:
    """Resolved mailbox identity used by the adapter."""

    base_url: str
    email: str
    mailbox_token: str
    smtp_password: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AampIdentity":
        return cls(
            base_url=_normalize_base_url(str(data.get("base_url", "") or "")),
            email=str(data.get("email", "") or "").strip(),
            mailbox_token=str(data.get("mailbox_token", "") or "").strip(),
            smtp_password=str(data.get("smtp_password", "") or "").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AampSenderPolicy:
    """Per-sender AAMP authorization policy."""

    sender: str
    dispatch_context_rules: dict[str, list[str]]


@dataclass(frozen=True)
class AampAuthorizationResult:
    """Authorization decision for an inbound AAMP task."""

    allowed: bool
    reason: Optional[str] = None


def _load_identity(path: Path) -> Optional[AampIdentity]:
    if not path.exists():
        return None
    try:
        return AampIdentity.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        logger.warning("[aamp] Failed to read cached identity %s: %s", path, exc)
        return None


def _save_identity(path: Path, identity: AampIdentity) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = identity.to_dict()
    payload["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _normalize_sender_policies_payload(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    payload = raw
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return []
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("[aamp] Failed to parse sender policies JSON: %s", exc)
            return []
    if isinstance(payload, dict):
        payload = payload.get("senderPolicies") or payload.get("sender_policies") or []
    if not isinstance(payload, list):
        logger.warning("[aamp] Ignoring invalid sender policies payload of type %s", type(payload).__name__)
        return []
    return [item for item in payload if isinstance(item, dict)]


def load_aamp_sender_policies(config: PlatformConfig | None) -> list[AampSenderPolicy]:
    """Load sender policies from config or environment."""
    extra = (config.extra if config else {}) or {}
    raw = extra.get("sender_policies")
    if raw is None:
        raw = os.getenv("AAMP_SENDER_POLICIES")

    policies: list[AampSenderPolicy] = []
    for item in _normalize_sender_policies_payload(raw):
        sender = str(item.get("sender", "") or "").strip().lower()
        if not sender:
            continue

        raw_rules = item.get("dispatchContextRules")
        if raw_rules is None:
            raw_rules = item.get("dispatch_context_rules")

        normalized_rules: dict[str, list[str]] = {}
        if isinstance(raw_rules, dict):
            for raw_key, raw_values in raw_rules.items():
                key = str(raw_key or "").strip().lower()
                if not key:
                    continue
                if isinstance(raw_values, (list, tuple, set)):
                    values = [str(value).strip() for value in raw_values if str(value).strip()]
                elif raw_values is None:
                    values = []
                else:
                    value = str(raw_values).strip()
                    values = [value] if value else []
                if values:
                    normalized_rules[key] = values

        policies.append(
            AampSenderPolicy(
                sender=sender,
                dispatch_context_rules=normalized_rules,
            )
        )

    return policies


def match_aamp_sender_policy(
    task: dict[str, Any],
    sender_policies: list[AampSenderPolicy],
) -> AampAuthorizationResult | None:
    """Evaluate OpenClaw-style sender + dispatchContext authorization."""
    if not sender_policies:
        return None

    sender = str(task.get("from", "") or "").strip()
    sender_lower = sender.lower()
    policy = next((item for item in sender_policies if item.sender == sender_lower), None)
    if policy is None:
        return AampAuthorizationResult(
            allowed=False,
            reason=f"sender {sender or '(unknown sender)'} is not allowed by senderPolicies",
        )

    if not policy.dispatch_context_rules:
        return AampAuthorizationResult(allowed=True)

    raw_context = task.get("dispatchContext")
    context: dict[str, str] = {}
    if isinstance(raw_context, dict):
        for raw_key, raw_value in raw_context.items():
            key = str(raw_key or "").strip().lower()
            value = str(raw_value or "").strip()
            if key and value:
                context[key] = value

    for key, allowed_values in policy.dispatch_context_rules.items():
        context_value = context.get(key)
        if not context_value:
            return AampAuthorizationResult(
                allowed=False,
                reason=f'dispatchContext missing required key "{key}"',
            )
        if context_value not in allowed_values:
            return AampAuthorizationResult(
                allowed=False,
                reason=f"dispatchContext {key}={context_value} is not allowed",
            )

    return AampAuthorizationResult(allowed=True)


def _import_aamp_sdk(config: PlatformConfig | None = None) -> Any:
    del config
    import importlib

    return importlib.import_module("aamp_sdk")


def _get_aamp_client_cls(config: PlatformConfig | None = None) -> Any:
    return _import_aamp_sdk(config).AampClient


def check_aamp_requirements() -> bool:
    """Check if the runtime can use the AAMP adapter."""
    try:
        _get_aamp_client_cls(None)
    except Exception:
        return False
    return True


def _resolve_reconnect_interval(config: PlatformConfig) -> float:
    extra = config.extra or {}
    raw = extra.get("poll_interval") or os.getenv("AAMP_POLL_INTERVAL") or DEFAULT_RECONNECT_INTERVAL
    try:
        return max(1.0, float(str(raw)))
    except (TypeError, ValueError):
        return DEFAULT_RECONNECT_INTERVAL


def _resolve_reject_unauthorized(config: PlatformConfig) -> bool:
    extra = config.extra or {}
    raw = extra.get("reject_unauthorized")
    if isinstance(raw, bool):
        return raw
    if raw is not None and str(raw).strip():
        return str(raw).strip().lower() in ("true", "1", "yes", "on")
    return _env_flag("AAMP_REJECT_UNAUTHORIZED", True)


def _identity_from_config(config: PlatformConfig) -> tuple[Optional[AampIdentity], Path, str, str]:
    extra = config.extra or {}
    base_url = _normalize_base_url(
        str(extra.get("base_url") or os.getenv("AAMP_BASE_URL") or os.getenv("AAMP_HOST") or DEFAULT_BASE_URL)
    )
    slug = str(extra.get("slug") or os.getenv("AAMP_SLUG") or DEFAULT_SLUG).strip()
    description = str(
        extra.get("description")
        or os.getenv("AAMP_DESCRIPTION")
        or DEFAULT_DESCRIPTION
    ).strip()
    credentials_path = _resolve_credentials_path(
        str(extra.get("credentials_file") or os.getenv("AAMP_CREDENTIALS_FILE") or "")
    )

    cached = _load_identity(credentials_path)
    if cached and (not base_url or cached.base_url == base_url):
        return cached, credentials_path, slug, description

    email = str(extra.get("email") or os.getenv("AAMP_EMAIL") or "").strip()
    mailbox_token = str(extra.get("mailbox_token") or os.getenv("AAMP_MAILBOX_TOKEN") or "").strip()
    smtp_password = str(
        extra.get("smtp_password")
        or os.getenv("AAMP_PASSWORD")
        or os.getenv("AAMP_SMTP_PASSWORD")
        or ""
    ).strip()

    if mailbox_token and (not email or not smtp_password):
        decoded_email, decoded_password = _decode_mailbox_token(mailbox_token)
        email = email or decoded_email
        smtp_password = smtp_password or decoded_password
    elif email and smtp_password and not mailbox_token:
        mailbox_token = _build_mailbox_token(email, smtp_password)

    if base_url and email and mailbox_token and smtp_password:
        return (
            AampIdentity(
                base_url=base_url,
                email=email,
                mailbox_token=mailbox_token,
                smtp_password=smtp_password,
            ),
            credentials_path,
            slug,
            description,
        )

    return None, credentials_path, slug, description


async def resolve_aamp_identity(config: PlatformConfig) -> tuple[AampIdentity, Path]:
    identity, credentials_path, slug, description = _identity_from_config(config)
    if identity is not None:
        return identity, credentials_path

    base_url = _normalize_base_url(
        str((config.extra or {}).get("base_url") or os.getenv("AAMP_BASE_URL") or os.getenv("AAMP_HOST") or DEFAULT_BASE_URL)
    )
    if not slug:
        raise RuntimeError(
            "AAMP identity is not configured. Set AAMP_EMAIL/AAMP_PASSWORD, "
            "AAMP_MAILBOX_TOKEN, or let Hermes auto-register the default "
            f"AAMP mailbox slug '{DEFAULT_SLUG}'."
        )

    client_cls = _get_aamp_client_cls(config)
    reject_unauthorized = _resolve_reject_unauthorized(config)

    registered = await asyncio.to_thread(
        client_cls.register_mailbox,
        aamp_host=base_url,
        slug=slug,
        description=description,
        reject_unauthorized=reject_unauthorized,
    )
    identity = AampIdentity(
        base_url=_normalize_base_url(str(registered.get("baseUrl", "") or base_url)),
        email=str(registered.get("email", "") or "").strip(),
        mailbox_token=str(registered.get("mailboxToken", "") or "").strip(),
        smtp_password=str(registered.get("smtpPassword", "") or "").strip(),
    )
    if not identity.email or not identity.mailbox_token or not identity.smtp_password:
        raise RuntimeError("AAMP mailbox registration returned an incomplete identity payload")

    _save_identity(credentials_path, identity)
    logger.info("[aamp] Registered mailbox %s and cached credentials at %s", identity.email, credentials_path)
    return identity, credentials_path


def _build_sdk_client(config: PlatformConfig, identity: AampIdentity) -> Any:
    client_cls = _get_aamp_client_cls(config)
    return client_cls(
        email=identity.email,
        mailbox_token=identity.mailbox_token,
        base_url=identity.base_url,
        smtp_password=identity.smtp_password,
        reconnect_interval=_resolve_reconnect_interval(config),
        reject_unauthorized=_resolve_reject_unauthorized(config),
    )


async def send_aamp_direct(
    config: PlatformConfig,
    chat_id: str,
    content: str,
    *,
    reply_to: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Send a one-off AAMP message without a long-lived adapter instance."""
    identity, _ = await resolve_aamp_identity(config)
    text = strip_markdown(content or "").strip() or "(empty message)"
    task_id = (
        (reply_to or "").strip()
        or str((metadata or {}).get("thread_id") or "").strip()
        or None
    )
    in_reply_to = str((metadata or {}).get("aamp_message_id") or "").strip() or None
    status = str((metadata or {}).get("aamp_status") or "completed").strip() or "completed"
    error_msg = str((metadata or {}).get("aamp_error_msg") or "").strip() or None
    structured_result = (metadata or {}).get("aamp_structured_result")

    def _send() -> dict[str, Any]:
        client = _build_sdk_client(config, identity)
        if task_id:
            client.send_result(
                to=chat_id,
                task_id=task_id,
                status=status,
                output=text,
                error_msg=error_msg,
                structured_result=structured_result,
                in_reply_to=in_reply_to,
            )
            return {
                "success": True,
                "platform": Platform.AAMP.value,
                "chat_id": chat_id,
                "message_id": task_id,
                "task_id": task_id,
                "intent": "task.result",
                "status": status,
            }

        subject = str((metadata or {}).get("subject") or "").strip()
        title = subject or text.splitlines()[0][:120].strip() or "Hermes message"
        new_task_id, new_message_id = client.send_task(
            to=chat_id,
            title=title,
            body_text=text,
        )
        return {
            "success": True,
            "platform": Platform.AAMP.value,
            "chat_id": chat_id,
            "message_id": new_message_id,
            "task_id": new_task_id,
            "intent": "task.dispatch",
        }

    return await asyncio.to_thread(_send)


class AampAdapter(BasePlatformAdapter):
    """Gateway adapter for AAMP mailboxes."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH
    SUPPORTS_MESSAGE_EDITING = False
    SUPPORTS_STREAMING = False
    SUPPORTS_INTERIM_MESSAGES = False
    SUPPORTS_TOOL_PROGRESS = False

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.AAMP)
        extra = config.extra or {}
        self._base_url = _normalize_base_url(
            str(extra.get("base_url") or os.getenv("AAMP_BASE_URL") or os.getenv("AAMP_HOST") or DEFAULT_BASE_URL)
        )
        self._credentials_path = _resolve_credentials_path(
            str(extra.get("credentials_file") or os.getenv("AAMP_CREDENTIALS_FILE") or "")
        )
        self._identity: Optional[AampIdentity] = None
        self._sdk_client: Any = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _build_event(self, task: dict[str, Any]) -> Optional[MessageEvent]:
        task_id = str(task.get("taskId", "") or "").strip()
        from_agent = str(task.get("from", "") or "").strip()
        title = str(task.get("title", "") or task.get("subject", "") or "").strip()
        body_text = str(task.get("bodyText", "") or "").strip()
        context_links = [str(item).strip() for item in (task.get("contextLinks") or []) if str(item).strip()]
        attachments = task.get("attachments") or []
        attachment_names = [
            str(item.get("filename") or item.get("name") or "").strip()
            for item in attachments
            if str(item.get("filename") or item.get("name") or "").strip()
        ]
        if not task_id or not from_agent or not (title or body_text or context_links or attachment_names):
            return None

        text_parts = []
        if title:
            text_parts.append(title)
        if body_text and body_text != title:
            text_parts.append(body_text)
        if context_links:
            text_parts.append("Context:\n" + "\n".join(context_links))
        if attachment_names:
            text_parts.append("Attachments:\n" + "\n".join(attachment_names))

        source = self.build_source(
            chat_id=from_agent,
            chat_name=from_agent,
            chat_type="dm",
            user_id=from_agent,
            user_name=from_agent,
            thread_id=task_id,
        )
        return MessageEvent(
            text="\n\n".join(text_parts).strip(),
            message_type=MessageType.TEXT,
            source=source,
            raw_message=task,
            message_id=task_id,
        )

    def _schedule_message(self, task: dict[str, Any]) -> None:
        event = self._build_event(task)
        if event is None or self._loop is None or self._loop.is_closed():
            return

        def _runner() -> None:
            asyncio.create_task(self.handle_message(event))

        self._loop.call_soon_threadsafe(_runner)

    def _handle_connected(self, *_args: Any) -> None:
        logger.info("[aamp] SDK connected using %s", "polling fallback" if self._sdk_client and self._sdk_client.is_using_polling_fallback() else "JMAP push")

    def _handle_disconnected(self, *args: Any) -> None:
        reason = str(args[0]) if args else "disconnected"
        logger.info("[aamp] SDK disconnected: %s", reason)

    def _handle_error(self, *args: Any) -> None:
        if not args:
            logger.warning("[aamp] SDK emitted an unspecified error")
            return
        err = args[0]
        logger.warning("[aamp] SDK error: %s", err)

    async def connect(self) -> bool:
        if not self._base_url:
            logger.error("[aamp] AAMP_BASE_URL (or AAMP_HOST) is required")
            return False

        try:
            self._loop = asyncio.get_running_loop()
            self._identity, self._credentials_path = await resolve_aamp_identity(self.config)
            if not self._acquire_platform_lock("aamp-mailbox", self._identity.email, "AAMP mailbox"):
                return False

            self._sdk_client = _build_sdk_client(self.config, self._identity)
            self._sdk_client.on("task.dispatch", self._schedule_message)
            self._sdk_client.on("connected", self._handle_connected)
            self._sdk_client.on("disconnected", self._handle_disconnected)
            self._sdk_client.on("error", self._handle_error)

            await asyncio.to_thread(self._sdk_client.connect)
            await asyncio.to_thread(self._sdk_client.reconcile_recent_emails, 10)
        except Exception as exc:
            logger.error("[aamp] Failed to connect: %s", exc)
            if self._sdk_client is not None:
                try:
                    await asyncio.to_thread(self._sdk_client.disconnect)
                except Exception:
                    pass
            self._sdk_client = None
            self._release_platform_lock()
            return False

        self._mark_connected()
        logger.info(
            "[aamp] Connected as %s via %s (reconnect=%ss, credentials=%s)",
            self._identity.email,
            self._identity.base_url,
            _resolve_reconnect_interval(self.config),
            self._credentials_path,
        )
        return True

    async def disconnect(self) -> None:
        self._running = False
        if self._sdk_client is not None:
            try:
                await asyncio.to_thread(self._sdk_client.disconnect)
            except Exception:
                pass
        self._sdk_client = None
        self._release_platform_lock()
        self._mark_disconnected()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        try:
            result = await send_aamp_direct(
                self.config,
                chat_id,
                content,
                reply_to=reply_to,
                metadata=metadata,
            )
            return SendResult(
                success=bool(result.get("success")),
                message_id=result.get("message_id"),
                raw_response=result,
            )
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        return {"name": chat_id, "type": "dm"}

    def format_message(self, content: str) -> str:
        return strip_markdown(content)
