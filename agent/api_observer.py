"""Canonical payload helpers for API request lifecycle observers."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional

from agent.redact import redact_sensitive_text
from agent.usage_pricing import normalize_usage


def hook_payload_max_chars() -> int:
    raw = os.getenv("HERMES_PLUGIN_PAYLOAD_MAX_CHARS", "50000")
    try:
        return max(1000, int(raw))
    except (TypeError, ValueError):
        return 50000


def is_sensitive_hook_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    lowered = key.lower().replace("-", "_")
    exact = {
        "api_key",
        "apikey",
        "authorization",
        "authentication",
        "proxy_authorization",
        "cookie",
        "set_cookie",
        "access_token",
        "refresh_token",
        "id_token",
        "auth_token",
        "token",
        "auth",
        "jwt",
        "credential",
        "client_secret",
        "password",
        "private_key",
        "key_material",
        "raw_secret",
        "secret_input",
        "secret_value",
        "secret",
    }
    return (
        lowered in exact
        or lowered.endswith("_api_key")
        or lowered.endswith("_token")
        or lowered.endswith("_access_token")
        or lowered.endswith("_refresh_token")
        or lowered.endswith("_client_secret")
        or lowered.endswith("_password")
    )


def hook_jsonable(
    value: Any,
    *,
    depth: int = 0,
    max_depth: int = 8,
    max_string: int = 8000,
    max_sequence: int = 200,
) -> Any:
    if depth > max_depth:
        return f"<{type(value).__name__} depth limit>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        redacted = redact_sensitive_text(
            value,
            force=True,
            redact_url_credentials=True,
        )
        if len(redacted) > max_string:
            return redacted[:max_string] + f"...[truncated {len(redacted) - max_string} chars]"
        return redacted
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    if isinstance(value, dict):
        output: Dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_sequence:
                output["_truncated_items"] = len(value) - max_sequence
                break
            string_key = str(key)
            if is_sensitive_hook_key(string_key):
                output[string_key] = "<redacted>"
                continue
            output[string_key] = hook_jsonable(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_string=max_string,
                max_sequence=max_sequence,
            )
        return output
    if isinstance(value, (list, tuple, set)):
        sequence = list(value)
        output = [
            hook_jsonable(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_string=max_string,
                max_sequence=max_sequence,
            )
            for item in sequence[:max_sequence]
        ]
        if len(sequence) > max_sequence:
            output.append({"_truncated_items": len(sequence) - max_sequence})
        return output
    try:
        if hasattr(value, "model_dump"):
            try:
                dumped = value.model_dump(mode="json")
            except TypeError:
                dumped = value.model_dump()
            return hook_jsonable(
                dumped,
                depth=depth + 1,
                max_depth=max_depth,
                max_string=max_string,
                max_sequence=max_sequence,
            )
    except Exception:
        pass
    try:
        if is_dataclass(value):
            return hook_jsonable(
                asdict(value),
                depth=depth + 1,
                max_depth=max_depth,
                max_string=max_string,
                max_sequence=max_sequence,
            )
    except Exception:
        pass
    if isinstance(value, SimpleNamespace):
        return hook_jsonable(
            vars(value),
            depth=depth + 1,
            max_depth=max_depth,
            max_string=max_string,
            max_sequence=max_sequence,
        )
    if hasattr(value, "__dict__"):
        try:
            public_attributes = {
                key: item
                for key, item in vars(value).items()
                if not str(key).startswith("_")
            }
            return hook_jsonable(
                public_attributes,
                depth=depth + 1,
                max_depth=max_depth,
                max_string=max_string,
                max_sequence=max_sequence,
            )
        except Exception:
            pass
    return redact_sensitive_text(
        str(value),
        force=True,
        redact_url_credentials=True,
    )[:max_string]


def sanitize_hook_payload(value: Any) -> Any:
    payload = hook_jsonable(value)
    limit = hook_payload_max_chars()
    try:
        encoded = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        return str(payload)[:limit]
    if len(encoded) <= limit:
        return payload
    payload = hook_jsonable(value, max_string=1000, max_sequence=50)
    try:
        encoded = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        return str(payload)[:limit]
    if len(encoded) <= limit:
        return payload
    return {
        "_truncated": True,
        "original_type": type(value).__name__,
        "preview": encoded[:limit],
    }


def api_request_payload_for_hook(api_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    body = {
        key: value
        for key, value in (api_kwargs or {}).items()
        if key not in {"timeout", "http_client"}
    }
    return sanitize_hook_payload({"method": "POST", "body": body})


def usage_summary_for_api_request_hook(
    response: Any,
    *,
    provider: Optional[str] = None,
    api_mode: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if response is None:
        return None
    raw_usage = getattr(response, "usage", None)
    if not raw_usage:
        return None
    canonical = normalize_usage(raw_usage, provider=provider, api_mode=api_mode)
    summary = asdict(canonical)
    summary.pop("raw_usage", None)
    summary["prompt_tokens"] = canonical.prompt_tokens
    summary["total_tokens"] = canonical.total_tokens
    return summary


def api_response_payload_for_hook(
    response: Any,
    assistant_message: Any,
    *,
    finish_reason: Optional[str],
    provider: Optional[str] = None,
    api_mode: Optional[str] = None,
) -> Dict[str, Any]:
    tool_calls = getattr(assistant_message, "tool_calls", None) or []
    return sanitize_hook_payload(
        {
            "model": getattr(response, "model", None),
            "finish_reason": finish_reason,
            "assistant_message": {
                "role": getattr(assistant_message, "role", "assistant"),
                "content": getattr(assistant_message, "content", None),
                "tool_calls": tool_calls,
            },
            "usage": usage_summary_for_api_request_hook(
                response,
                provider=provider,
                api_mode=api_mode,
            ),
        }
    )
