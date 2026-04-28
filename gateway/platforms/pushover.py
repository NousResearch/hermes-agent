"""Pushover platform adapter.

Outbound-only notification adapter for Pushover.  Pushover does not provide an
incoming chat transport, so this adapter exists for Hermes notifications,
`send_message`, cron delivery, and webhook delivery.

Configuration can either provide `token` + `extra.user` directly, or point at a
Hermes-managed credential JSON file with `extra.credentials_file` and
`extra.app`/`extra.app_label`.  The latter keeps secrets out of config.yaml.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency probe
    aiohttp = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult

logger = logging.getLogger(__name__)

DEFAULT_SEND_ENDPOINT = "https://api.pushover.net/1/messages.json"
DEFAULT_VALIDATE_ENDPOINT = "https://api.pushover.net/1/users/validate.json"
DEFAULT_CREDENTIALS_FILE = "~/.hermes/credentials/claude-migration/pushover.json"
MAX_MESSAGE_LENGTH = 1024


def check_pushover_requirements() -> bool:
    """Check whether the adapter can make HTTP requests."""
    return AIOHTTP_AVAILABLE


def _redacted_error(exc: Exception | str) -> str:
    """Return a safe error string without credential material."""
    text = str(exc)
    return re.sub(
        r"(?i)\b(token|user|api_key|api_token|app_token|user_key)([=:])([^\s&]+)",
        r"\1\2[REDACTED]",
        text,
    )


def _redact_values(text: str, *values: str) -> str:
    """Redact known credential values from a diagnostic string."""
    redacted = text
    for value in values:
        if value:
            redacted = redacted.replace(value, "[REDACTED]")
    return redacted


class PushoverAdapter(BasePlatformAdapter):
    """Outbound-only Pushover notification adapter."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.PUSHOVER)
        self.extra: Dict[str, Any] = config.extra or {}
        self.send_endpoint = str(self.extra.get("endpoint") or DEFAULT_SEND_ENDPOINT)
        self.validate_endpoint = str(
            self.extra.get("validate_endpoint") or DEFAULT_VALIDATE_ENDPOINT
        )
        self.default_title = str(self.extra.get("title") or "Hermes")
        self.default_device = str(self.extra.get("device") or "")
        self.default_sound = str(self.extra.get("sound") or "")
        self.default_priority = self.extra.get("priority")
        self.app_label = str(
            self.extra.get("app")
            or self.extra.get("app_label")
            or os.getenv("PUSHOVER_APP")
            or ""
        ).strip()
        self._token, self._user = self._resolve_credentials()

    @property
    def name(self) -> str:
        return "Pushover"

    async def connect(self) -> bool:
        if not AIOHTTP_AVAILABLE:
            self._set_fatal_error(
                "pushover_dependency_missing",
                "Pushover requires aiohttp to send notifications.",
                retryable=False,
            )
            return False
        if not self._token or not self._user:
            self._set_fatal_error(
                "pushover_credentials_missing",
                "Pushover token/user missing. Configure token + extra.user, or extra.credentials_file + extra.app.",
                retryable=False,
            )
            return False
        self._mark_connected()
        logger.info("[pushover] Outbound notification adapter ready")
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return logical chat info for Pushover's outbound-only target."""
        return {"name": chat_id or "Pushover", "type": "notification"}

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not content.strip():
            return SendResult(success=False, error="Pushover message content is empty")
        if not AIOHTTP_AVAILABLE:
            return SendResult(
                success=False,
                error="Pushover requires aiohttp to send notifications.",
                retryable=False,
            )
        if not self._token or not self._user:
            return SendResult(success=False, error="Pushover credentials are not configured")

        metadata = metadata or {}
        chunks = self.truncate_message(content, self.MAX_MESSAGE_LENGTH)
        if not chunks:
            return SendResult(success=False, error="Pushover message content is empty after truncation")

        last_response: Any = None
        request_ids: list[str] = []
        delivered_chunks = 0
        try:
            async with aiohttp.ClientSession(  # type: ignore[union-attr]
                timeout=aiohttp.ClientTimeout(total=30)  # type: ignore[union-attr]
            ) as session:
                for chunk in chunks:
                    payload = self._build_payload(chat_id, chunk, metadata)
                    async with session.post(self.send_endpoint, data=payload) as resp:
                        text = await resp.text()
                        try:
                            data = json.loads(text) if text else {}
                        except json.JSONDecodeError:
                            data = {"raw": text[:200]}
                        last_response = data
                        request_id = str(data.get("request") or "")
                        if request_id:
                            request_ids.append(request_id)
                        if resp.status >= 400 or data.get("status") == 0:
                            errors = data.get("errors") or [f"HTTP {resp.status}"]
                            if isinstance(errors, list):
                                err = "; ".join(str(e) for e in errors)
                            else:
                                err = str(errors)
                            last_request_id = request_ids[-1] if request_ids else None
                            return SendResult(
                                success=False,
                                error=f"Pushover send failed: {err}",
                                raw_response={
                                    "status": resp.status,
                                    "request": last_request_id,
                                    "requests": request_ids,
                                    "delivered_chunks": delivered_chunks,
                                    "total_chunks": len(chunks),
                                },
                                retryable=resp.status >= 500,
                            )
                        delivered_chunks += 1
        except Exception as exc:
            err = _redact_values(_redacted_error(exc), self._token, self._user)
            logger.error("[pushover] send failed: %s", err)
            return SendResult(
                success=False,
                error=f"Pushover send failed: {err}",
                raw_response={"delivered_chunks": delivered_chunks, "total_chunks": len(chunks)},
                retryable=True,
            )

        last_request_id = request_ids[-1] if request_ids else None
        return SendResult(
            success=True,
            message_id=last_request_id,
            raw_response={
                "request": last_request_id,
                "requests": request_ids,
                "status": last_response.get("status") if isinstance(last_response, dict) else None,
                "delivered_chunks": delivered_chunks,
                "total_chunks": len(chunks),
            },
        )

    async def validate(self) -> SendResult:
        """Validate the configured token/user pair without sending a message."""
        if not self._token or not self._user:
            return SendResult(success=False, error="Pushover credentials are not configured")
        try:
            async with aiohttp.ClientSession(  # type: ignore[union-attr]
                timeout=aiohttp.ClientTimeout(total=15)  # type: ignore[union-attr]
            ) as session:
                async with session.post(
                    self.validate_endpoint,
                    data={"token": self._token, "user": self._user},
                ) as resp:
                    text = await resp.text()
                    try:
                        data = json.loads(text) if text else {}
                    except json.JSONDecodeError:
                        data = {"raw": text[:200]}
                    if resp.status >= 400 or data.get("status") == 0:
                        errors = data.get("errors") or [f"HTTP {resp.status}"]
                        if isinstance(errors, list):
                            err = "; ".join(str(e) for e in errors)
                        else:
                            err = str(errors)
                        return SendResult(success=False, error=f"Pushover validation failed: {err}")
                    safe = {
                        "status": data.get("status"),
                        "devices_count": len(data.get("devices") or []),
                        "licenses": data.get("licenses") or [],
                    }
                    return SendResult(success=True, raw_response=safe)
        except Exception as exc:
            err = _redact_values(_redacted_error(exc), self._token, self._user)
            logger.error("[pushover] validation failed: %s", err)
            return SendResult(
                success=False,
                error=f"Pushover validation failed: {err}",
                retryable=True,
            )

    def _build_payload(self, chat_id: str, message: str, metadata: Dict[str, Any]) -> Dict[str, str]:
        payload: Dict[str, str] = {
            "token": self._token,
            "user": self._user,
            "message": message,
        }
        title = metadata.get("title") or self.extra.get("title") or self.default_title
        if title:
            payload["title"] = str(title)

        # Pushover has no chat/channel concept. Use explicit metadata/config for
        # device routing rather than coercing arbitrary Hermes chat IDs into
        # Pushover device names.
        device = metadata.get("device") or self.default_device
        if device:
            payload["device"] = str(device)

        priority = metadata.get("priority", self.default_priority)
        if priority not in (None, ""):
            payload["priority"] = str(priority)
        sound = metadata.get("sound") or self.default_sound
        if sound:
            payload["sound"] = str(sound)
        url = metadata.get("url") or self.extra.get("url")
        if url:
            payload["url"] = str(url)
        url_title = metadata.get("url_title") or self.extra.get("url_title")
        if url_title:
            payload["url_title"] = str(url_title)
        return payload

    def _resolve_credentials(self) -> Tuple[str, str]:
        token = str(self.config.token or self.config.api_key or self.extra.get("token") or "").strip()
        user = str(self.extra.get("user") or self.extra.get("user_key") or "").strip()
        if token and user:
            return token, user

        cred_file = str(
            self.extra.get("credentials_file")
            or os.getenv("PUSHOVER_CREDENTIALS_FILE")
            or DEFAULT_CREDENTIALS_FILE
        )
        app_label = self.app_label
        try:
            path = Path(os.path.expanduser(cred_file))
            data = json.loads(path.read_text())
            if not user:
                user = str(data.get("user") or data.get("user_key") or "").strip()
            if not token:
                raw_apps = data.get("apps")
                apps = raw_apps if isinstance(raw_apps, dict) else {}
                if app_label and app_label in apps:
                    token = str(apps[app_label] or "").strip()
                else:
                    token = str(
                        data.get("token") or data.get("api_token") or data.get("app_token") or ""
                    ).strip()
            endpoint = str(data.get("endpoint") or "").strip()
            if endpoint and self.send_endpoint == DEFAULT_SEND_ENDPOINT:
                self.send_endpoint = endpoint
        except FileNotFoundError:
            logger.warning("[pushover] credentials file not found: %s", cred_file)
        except (json.JSONDecodeError, OSError, TypeError, AttributeError, UnicodeDecodeError) as exc:
            logger.warning("[pushover] could not read credentials file: %s", _redacted_error(exc))
        return token, user
