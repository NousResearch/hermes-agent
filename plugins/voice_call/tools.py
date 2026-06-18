"""Hermes model tool handler for guarded voice calls."""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from .config import ALLOWED_ESCALATION_POLICIES, load_voice_call_config, voice_call_available

_PHONE_RE = re.compile(r"[^0-9+]")


def redact_phone(value: str) -> str:
    """Redact a phone number while preserving enough shape for debugging."""
    digits = re.sub(r"\D", "", value or "")
    if len(digits) < 4:
        return "[redacted-phone]"
    return f"[redacted-phone:*{digits[-4:]}]"


def normalize_phone(value: str) -> str:
    cleaned = _PHONE_RE.sub("", (value or "").strip())
    if cleaned.startswith("++"):
        cleaned = "+" + cleaned.lstrip("+")
    if cleaned and not cleaned.startswith("+") and len(re.sub(r"\D", "", cleaned)) == 10:
        cleaned = "+1" + cleaned
    return cleaned


def _validate_destination(number: str) -> tuple[bool, str]:
    cfg = load_voice_call_config()
    if not number or len(re.sub(r"\D", "", number)) < 7:
        return False, "destination phone number is missing or too short"
    if cfg.blocked_prefixes and any(number.startswith(prefix) for prefix in cfg.blocked_prefixes):
        return False, "destination is blocked by voice_call.blocked_prefixes"
    if cfg.allowed_prefixes and not any(number.startswith(prefix) for prefix in cfg.allowed_prefixes):
        return False, "destination is not allowed by voice_call.allowed_prefixes"
    return True, ""


def _error(message: str, **extra: Any) -> str:
    payload = {"success": False, "error": message}
    payload.update(extra)
    return json.dumps(payload)


def _redact_payload(obj: Any) -> Any:
    if isinstance(obj, dict):
        redacted = {}
        for key, value in obj.items():
            if key.lower() in {"to", "from", "phone", "phone_number", "transfer_number", "callback_number"} and isinstance(value, str):
                redacted[key] = redact_phone(value)
            else:
                redacted[key] = _redact_payload(value)
        return redacted
    if isinstance(obj, list):
        return [_redact_payload(item) for item in obj]
    return obj


def _request(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = load_voice_call_config()
    if not cfg.service_url:
        raise RuntimeError("VOICE_CALL_SERVICE_URL or voice_call.service_url is not configured")
    url = f"{cfg.service_url}{path}"
    with httpx.Client(timeout=cfg.timeout_seconds) as client:
        if method == "GET":
            response = client.get(url, params=payload or {})
        else:
            response = client.request(method, url, json=payload or {})
    response.raise_for_status()
    if response.headers.get("content-type", "").startswith("application/json"):
        return response.json()
    return {"success": True, "text": response.text[:2000]}


def _call(args: dict[str, Any]) -> str:
    to = normalize_phone(str(args.get("to") or ""))
    purpose = str(args.get("purpose") or "").strip()
    context = str(args.get("context") or "").strip()
    policy = str(args.get("escalation_policy") or "no_escalation").strip()

    if not purpose:
        return _error("purpose is required for voice_call action=call")
    if not context:
        return _error("context is required for voice_call action=call")
    if policy not in ALLOWED_ESCALATION_POLICIES:
        return _error("invalid escalation_policy", allowed=sorted(ALLOWED_ESCALATION_POLICIES))
    ok, reason = _validate_destination(to)
    if not ok:
        return _error(reason, to=redact_phone(to))

    cfg = load_voice_call_config()
    payload = {
        "to": to,
        "purpose": purpose,
        "context": context,
        "escalation_policy": policy,
        "caller_profile": cfg.caller_profile.__dict__,
    }
    result = _request("POST", "/twilio/voice/outbound", payload)
    return json.dumps(_redact_payload(result))


def voice_call(args: dict[str, Any], **_: Any) -> str:
    """Dispatch voice_call actions to the configured voice service."""
    if not voice_call_available():
        return _error("voice_call is not configured; set VOICE_CALL_SERVICE_URL or voice_call.service_url")
    if not isinstance(args, dict):
        return _error("voice_call expects object arguments")

    action = str(args.get("action") or "").strip()
    try:
        if action == "call":
            return _call(args)
        call_id = str(args.get("call_id") or "").strip()
        if not call_id:
            return _error("call_id is required for this action")
        if action == "hangup":
            result = _request("POST", f"/twilio/voice/{call_id}/hangup")
        elif action == "transfer_to_jason":
            result = _request("POST", f"/twilio/voice/{call_id}/transfer")
        elif action == "get_transcript":
            result = _request("GET", f"/twilio/voice/{call_id}/transcript")
        else:
            return _error("invalid action", allowed=["call", "hangup", "transfer_to_jason", "get_transcript"])
        return json.dumps(_redact_payload(result))
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:1000]
        return _error(f"voice service returned HTTP {exc.response.status_code}", detail=detail)
    except Exception as exc:
        return _error(f"voice_call failed: {exc}")
