"""OpenClaw cron receiver delivery adapter."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from ipaddress import ip_address
from typing import Any, Dict, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from hermes_cli.config import get_hermes_home

DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_RECEIVER_ALLOWLIST = {"127.0.0.1", "localhost", "::1"}


def check_openclaw_requirements() -> bool:
    return True


def _profile_name() -> str:
    profile = os.getenv("HERMES_PROFILE", "").strip()
    if profile:
        return profile
    home = get_hermes_home()
    if home.parent.name == "profiles":
        return home.name
    return "default"


def _metadata_value(metadata: Optional[Dict[str, Any]], key: str, default: str = "") -> str:
    if not metadata:
        return default
    value = metadata.get(key)
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _payload_for(content: str, metadata: Optional[Dict[str, Any]]) -> dict[str, str]:
    job = _metadata_value(metadata, "job") or _metadata_value(metadata, "job_id", "unknown")
    run_id = _metadata_value(metadata, "run_id")
    if not run_id:
        run_id = f"{job}-{int(time.time() * 1000)}"
    return {
        "profile": _metadata_value(metadata, "profile", _profile_name()),
        "job": job,
        "run_id": run_id,
        "output": content,
        "severity": _metadata_value(metadata, "severity", "info"),
    }


def _sign(secret: str, body: bytes) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _coerce_allowlist(value: Any) -> set[str] | None:
    if value is None:
        return set(DEFAULT_RECEIVER_ALLOWLIST)
    if isinstance(value, str):
        entries = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        entries = [str(part).strip() for part in value]
    else:
        return set(DEFAULT_RECEIVER_ALLOWLIST)
    if not entries:
        return set()
    return {entry.lower() for entry in entries if entry}


def _validate_receiver_url(url: str, *, allowlist: set[str] | None) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("scheme must be http or https")
    if not parsed.hostname:
        raise ValueError("host is required")
    if parsed.username or parsed.password:
        raise ValueError("userinfo is not allowed")
    if parsed.fragment:
        raise ValueError("fragments are not allowed")

    host = parsed.hostname.lower()
    if allowlist is not None and len(allowlist) > 0 and host not in allowlist:
        raise ValueError(f"host {parsed.hostname!r} is not in the OpenClaw receiver allowlist")

    try:
        parsed_ip = ip_address(host)
    except ValueError:
        return
    if allowlist is not None and len(allowlist) > 0 and str(parsed_ip) not in allowlist:
        raise ValueError(f"host {parsed.hostname!r} is not in the OpenClaw receiver allowlist")


def _resolve_url(pconfig: PlatformConfig, chat_id: str) -> str:
    """Resolve and validate the OpenClaw receiver URL.

    Resolution order is explicit chat_id, ``gateway.platforms.openclaw.url``
    loaded into ``PlatformConfig.extra["url"]``, then the deprecated
    ``OPENCLAW_CRON_RECEIVER_URL`` environment fallback. The environment
    fallback remains supported for compatibility and emits a
    ``DeprecationWarning`` when it is the selected source.
    """
    source = ""
    candidate = str(chat_id or "").strip()
    if candidate:
        source = "chat_id"
    if not candidate:
        candidate = str(pconfig.extra.get("url") or "").strip()
        if candidate:
            source = str(pconfig.extra.get("_url_source") or "config")
    if not candidate:
        candidate = os.getenv("OPENCLAW_CRON_RECEIVER_URL", "").strip()
        if candidate:
            source = "env"
    if not candidate:
        return ""

    allowlist = _coerce_allowlist(pconfig.extra.get("allowlist"))
    _validate_receiver_url(candidate, allowlist=allowlist)
    if source == "env":
        warnings.warn(
            "OPENCLAW_CRON_RECEIVER_URL is deprecated; set gateway.platforms.openclaw.url in config.yaml",
            DeprecationWarning,
            stacklevel=2,
        )
    return candidate


def _resolve_secret(pconfig: PlatformConfig) -> str:
    return str(pconfig.extra.get("secret") or os.getenv("OPENCLAW_CRON_SHARED_SECRET", "")).strip()


def _resolve_timeout(pconfig: PlatformConfig) -> float:
    value = pconfig.extra.get("timeout_seconds", os.getenv("OPENCLAW_CRON_TIMEOUT_SECONDS", ""))
    try:
        return float(value)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_SECONDS


def _post_payload(url: str, secret: str, payload: dict[str, str], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "content-type": "application/json",
            "x-cron-signature": _sign(secret, body),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8", errors="replace")
            if response.status != 202:
                return {
                    "success": False,
                    "status": response.status,
                    "body": response_body,
                    "error": f"expected HTTP 202, got {response.status}",
                }
            try:
                parsed = json.loads(response_body)
            except json.JSONDecodeError:
                return {
                    "success": False,
                    "status": response.status,
                    "body": response_body,
                    "error": "receiver returned non-JSON body",
                }
            if not isinstance(parsed, dict) or parsed.get("accepted") is not True:
                return {
                    "success": False,
                    "status": response.status,
                    "body": response_body,
                    "error": "receiver did not acknowledge acceptance",
                }
            return {
                "success": True,
                "status": response.status,
                "body": response_body,
                "message_id": parsed.get("message_id"),
            }
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        return {"success": False, "status": exc.code, "body": response_body}


async def send_openclaw_direct(
    pconfig: PlatformConfig,
    chat_id: str,
    message: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        url = _resolve_url(pconfig, chat_id)
    except ValueError as exc:
        return {"error": f"OpenClaw receiver URL invalid: {exc}"}
    if not url:
        return {"error": "OpenClaw receiver URL missing. Set gateway.platforms.openclaw.url in config.yaml."}
    secret = _resolve_secret(pconfig)
    if not secret:
        return {"error": "OpenClaw shared secret missing. Set OPENCLAW_CRON_SHARED_SECRET."}

    payload = _payload_for(message, metadata)
    timeout = _resolve_timeout(pconfig)
    try:
        result = await asyncio.to_thread(_post_payload, url, secret, payload, timeout)
    except Exception as exc:
        return {"error": f"OpenClaw delivery failed: {exc}"}
    if not result.get("success"):
        return {
            "error": f"OpenClaw receiver returned HTTP {result.get('status')}: {result.get('error') or result.get('body')}",
            "status": result.get("status"),
        }
    return {
        "success": True,
        "platform": "openclaw",
        "chat_id": url,
        "message_id": payload["run_id"],
        "raw_response": result,
    }


class OpenClawAdapter(BasePlatformAdapter):
    """Outbound-only adapter that POSTs cron output to OpenClaw."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.OPENCLAW)

    async def connect(self) -> bool:
        try:
            url = _resolve_url(self.config, "")
        except ValueError as exc:
            self._set_fatal_error("openclaw_invalid_url", f"OpenClaw receiver URL invalid: {exc}", retryable=False)
            return False
        if not url:
            self._set_fatal_error("openclaw_missing_url", "OpenClaw receiver URL missing", retryable=False)
            return False
        if not _resolve_secret(self.config):
            self._set_fatal_error("openclaw_missing_secret", "OpenClaw shared secret missing", retryable=False)
            return False
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        result = await send_openclaw_direct(self.config, chat_id, content, metadata=metadata)
        if result.get("error"):
            return SendResult(success=False, error=str(result["error"]), raw_response=result)
        return SendResult(
            success=True,
            message_id=str(result.get("message_id") or ""),
            raw_response=result,
        )

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {
            "name": "OpenClaw Cron Receiver",
            "type": "webhook",
            "chat_id": chat_id or _resolve_url(self.config, ""),
        }
