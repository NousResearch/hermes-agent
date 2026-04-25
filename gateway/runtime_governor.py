"""Runtime governor client for HermesOS managed instances.

The dashboard sidecar owns the local enforcement API. This module keeps the
agent runtime independent from dashboard internals: it signs local requests,
fails closed when required, and returns small typed decisions to gateway/cron.
"""

from __future__ import annotations

import hmac
import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

from utils import is_truthy_value

logger = logging.getLogger(__name__)

DEFAULT_UNAVAILABLE_MESSAGE = (
    "HermesOS runtime limits could not be verified. Try again in a moment."
)
DEFAULT_LIMIT_MESSAGE = (
    "HermesOS runtime limit reached. Upgrade or wait for your allowance to reset."
)


@dataclass(frozen=True)
class RuntimeGovernorDecision:
    allowed: bool
    lease_id: Optional[str] = None
    deadline_at: Optional[str] = None
    reason: str = ""
    user_message: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeGovernorHeartbeat:
    should_stop: bool = False
    reason: str = ""
    user_message: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class RuntimeGovernorError(RuntimeError):
    """Raised for required lifecycle calls that cannot be verified."""

    def __init__(self, message: str = DEFAULT_UNAVAILABLE_MESSAGE):
        super().__init__(message)
        self.user_message = message


class RuntimeGovernorClient:
    """Small stdlib-only client for the local sidecar runtime API."""

    def __init__(
        self,
        *,
        base_url: str,
        instance_id: str,
        api_key: str,
        required: bool,
        timeout_seconds: float = 5.0,
        default_user_id: str = "",
        default_tier: str = "free",
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.instance_id = (instance_id or "").strip()
        self.api_key = api_key or ""
        self.required = bool(required)
        self.timeout_seconds = max(float(timeout_seconds or 5.0), 0.5)
        self.default_user_id = (default_user_id or "").strip()
        self.default_tier = (default_tier or "free").strip() or "free"

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.instance_id and self.api_key)

    @property
    def enabled(self) -> bool:
        return self.required or self.configured

    def admit(
        self,
        *,
        platform: str,
        session_key: str,
        source_message_id: str = "",
        message_preview: str = "",
        user_id: str = "",
        tier: str = "",
    ) -> RuntimeGovernorDecision:
        if not self.enabled:
            return RuntimeGovernorDecision(allowed=True, reason="disabled")

        if not self.configured:
            return RuntimeGovernorDecision(
                allowed=False,
                reason="policy_unavailable",
                user_message=DEFAULT_UNAVAILABLE_MESSAGE,
            )

        payload = {
            "userId": (user_id or self.default_user_id).strip(),
            "tier": (tier or self.default_tier).strip() or "free",
            "platform": str(platform or "").strip(),
            "sessionKey": str(session_key or "").strip(),
            "sourceMessageId": str(source_message_id or "").strip(),
            "messagePreview": str(message_preview or "")[:500],
        }
        try:
            data = self._post("/admit", payload)
        except RuntimeGovernorError as exc:
            logger.warning(
                "Runtime governor admission failed closed: %s",
                exc.__class__.__name__,
            )
            return RuntimeGovernorDecision(
                allowed=False,
                reason="policy_unavailable",
                user_message=exc.user_message,
            )

        allowed = bool(data.get("allowed"))
        return RuntimeGovernorDecision(
            allowed=allowed,
            lease_id=_optional_string(data.get("leaseId")),
            deadline_at=_optional_string(data.get("deadlineAt")),
            reason=_optional_string(data.get("reason")) or ("allowed" if allowed else "denied"),
            user_message=_optional_string(data.get("userMessage"))
            or ("" if allowed else DEFAULT_LIMIT_MESSAGE),
            raw=data,
        )

    def start(self, lease_id: str) -> None:
        if not lease_id:
            return
        self._post_lifecycle("/start", lease_id)

    def heartbeat(self, lease_id: str) -> RuntimeGovernorHeartbeat:
        if not lease_id:
            return RuntimeGovernorHeartbeat()
        try:
            data = self._post_lifecycle("/heartbeat", lease_id)
        except RuntimeGovernorError as exc:
            logger.warning(
                "Runtime governor heartbeat failed closed: %s",
                exc.__class__.__name__,
            )
            return RuntimeGovernorHeartbeat(
                should_stop=True,
                reason="policy_unavailable",
                user_message=exc.user_message,
            )

        should_stop = bool(data.get("shouldStop"))
        return RuntimeGovernorHeartbeat(
            should_stop=should_stop,
            reason=_optional_string(data.get("reason")) or ("cutoff" if should_stop else "ok"),
            user_message=_optional_string(data.get("userMessage"))
            or (DEFAULT_LIMIT_MESSAGE if should_stop else ""),
            raw=data,
        )

    def finish(self, lease_id: str, *, reason: str = "agent_end") -> None:
        if not lease_id:
            return
        self._post_lifecycle("/finish", lease_id, reason=reason)

    def fail(self, lease_id: str, *, reason: str = "agent_error") -> None:
        if not lease_id:
            return
        self._post_lifecycle("/fail", lease_id, reason=reason)

    def _post_lifecycle(
        self,
        path: str,
        lease_id: str,
        *,
        reason: str = "",
    ) -> dict[str, Any]:
        if not self.enabled:
            return {}
        if not self.configured:
            raise RuntimeGovernorError()
        payload = {"leaseId": lease_id}
        if reason:
            payload["reason"] = reason
        return self._post(path, payload)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.configured:
            raise RuntimeGovernorError()

        request_payload = {
            **payload,
            "instanceId": self.instance_id,
        }
        body = json.dumps(
            request_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        timestamp = str(int(time.time() * 1000))
        signature = hmac.new(
            self.api_key.encode("utf-8"),
            timestamp.encode("utf-8") + b"." + body,
            hashlib.sha256,
        ).hexdigest()

        url = self.base_url + (path if path.startswith("/") else "/" + path)
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "content-type": "application/json",
                "x-hermes-timestamp": timestamp,
                "x-hermes-signature": signature,
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
                status = getattr(response, "status", response.getcode())
        except urllib.error.HTTPError as exc:
            status = exc.code
            try:
                response_body = exc.read().decode("utf-8")
            except Exception:
                response_body = ""
            raise RuntimeGovernorError(_safe_error_message(status, response_body)) from exc
        except Exception as exc:
            raise RuntimeGovernorError() from exc

        if status >= 400:
            raise RuntimeGovernorError(_safe_error_message(status, response_body))

        try:
            parsed = json.loads(response_body) if response_body else {}
        except json.JSONDecodeError as exc:
            raise RuntimeGovernorError() from exc

        if parsed.get("success") is False:
            raise RuntimeGovernorError(
                _optional_string(parsed.get("error")) or DEFAULT_UNAVAILABLE_MESSAGE
            )

        data = parsed.get("data", parsed)
        if not isinstance(data, dict):
            raise RuntimeGovernorError()
        return data


class DisabledRuntimeGovernorClient(RuntimeGovernorClient):
    def __init__(self) -> None:
        super().__init__(
            base_url="",
            instance_id="",
            api_key="",
            required=False,
        )

    @property
    def enabled(self) -> bool:
        return False


def get_runtime_governor() -> RuntimeGovernorClient:
    required = is_truthy_value(os.getenv("HERMES_RUNTIME_GOVERNOR_REQUIRED"), default=False)
    enabled = required or is_truthy_value(
        os.getenv("HERMES_RUNTIME_GOVERNOR_ENABLED"),
        default=False,
    )
    if not enabled:
        return DisabledRuntimeGovernorClient()

    timeout_raw = os.getenv("HERMES_RUNTIME_GOVERNOR_TIMEOUT_SECONDS", "5")
    try:
        timeout_seconds = float(timeout_raw)
    except (TypeError, ValueError):
        timeout_seconds = 5.0

    return RuntimeGovernorClient(
        base_url=os.getenv("HERMES_RUNTIME_GOVERNOR_URL", ""),
        instance_id=os.getenv("HERMES_RUNTIME_GOVERNOR_INSTANCE_ID", ""),
        api_key=os.getenv("API_SERVER_KEY", ""),
        required=required,
        timeout_seconds=timeout_seconds,
        default_user_id=os.getenv("HERMES_RUNTIME_GOVERNOR_USER_ID", ""),
        default_tier=os.getenv("HERMES_RUNTIME_GOVERNOR_TIER", "free"),
    )


def _optional_string(value: Any) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _safe_error_message(status: int, response_body: str) -> str:
    if status >= 500:
        return DEFAULT_UNAVAILABLE_MESSAGE
    try:
        parsed = json.loads(response_body) if response_body else {}
    except json.JSONDecodeError:
        parsed = {}
    message = _optional_string(parsed.get("error")) if isinstance(parsed, dict) else ""
    return message or DEFAULT_UNAVAILABLE_MESSAGE
