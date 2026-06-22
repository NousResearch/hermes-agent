"""Synchronous adapter for calling Ouroboros MCP tools from gateway code.

The gateway owns its own async dispatch boundary, so this module deliberately
stays synchronous: callers can wrap :func:`call_ouroboros_tool` with
``asyncio.to_thread`` when needed.  Calls are still gated through the Hermes
registry before reaching the MCP handler factory so disabled/excluded tools are
not bypassed.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from tools.mcp_tool import _make_tool_handler, sanitize_mcp_name_component
from tools.registry import registry

OUROBOROS_MCP_SERVER = "ouroboros"
OUROBOROS_MCP_TOOLSET = "mcp-ouroboros"

_SAFE_RAW_TOOL_RE = re.compile(r"^ouroboros_[A-Za-z0-9_]+$")
_REDACTED = "[REDACTED]"
_SENSITIVE_PAYLOAD_KEYS = frozenset(
    {
        "token",
        "api_key",
        "apikey",
        "authorization",
        "password",
        "secret",
        "access_token",
        "refresh_token",
        "credential",
        "credentials",
        "auth",
    }
)

# Test seam: tests can replace this with a fake factory without touching the
# runtime MCP client implementation.
_handler_factory = _make_tool_handler

try:  # Best-effort safety boundary; keep the adapter usable if unavailable.
    from agent.redact import redact_sensitive_text as _redact_sensitive_text
except Exception:  # pragma: no cover - optional dependency fallback
    _redact_sensitive_text = None


def ouroboros_registry_tool_name(tool_name: str) -> str:
    """Return the Hermes registry name for a raw Ouroboros MCP tool name."""
    safe_server = sanitize_mcp_name_component(OUROBOROS_MCP_SERVER)
    safe_tool = sanitize_mcp_name_component(tool_name)
    return f"mcp_{safe_server}_{safe_tool}"


def _redact_text(value: str) -> str:
    """Redact sensitive text when the redactor is importable."""
    if _redact_sensitive_text is None:
        return value
    try:
        return _redact_sensitive_text(value, force=True)
    except TypeError:  # Older redactor signatures may not accept force=.
        try:
            return _redact_sensitive_text(value)
        except Exception:
            return value
    except Exception:
        return value


def _is_sensitive_payload_key(key: Any) -> bool:
    return isinstance(key, str) and key.lower() in _SENSITIVE_PAYLOAD_KEYS


def _redact_payload(value: Any) -> Any:
    """Recursively redact strings and sensitive-key values in JSON-like payloads."""
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            redacted[key] = (
                _REDACTED if _is_sensitive_payload_key(key) else _redact_payload(item)
            )
        return redacted
    if isinstance(value, list):
        return [_redact_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_payload(item) for item in value)
    return value


def _safe_tool_value(tool_name: Any) -> Any:
    """Return a JSON-safe, non-secret tool identifier for error payloads."""
    if isinstance(tool_name, str):
        return _redact_text(tool_name)
    return f"<{type(tool_name).__name__}>"


def _error(
    message: str,
    *,
    tool_name: Any,
    registry_name: str | None = None,
) -> dict[str, Any]:
    """Build the structured error shape expected by gateway callers."""
    return {
        "success": False,
        "error": _redact_text(str(message)),
        "tool": _safe_tool_value(tool_name),
        "registry_name": registry_name,
        "server": OUROBOROS_MCP_SERVER,
    }


def _validate_raw_tool_name(tool_name: Any) -> tuple[str | None, dict[str, Any] | None]:
    if not isinstance(tool_name, str) or not tool_name.strip():
        return None, _error(
            "tool_name must be a non-empty raw Ouroboros MCP tool name",
            tool_name=tool_name,
            registry_name=None,
        )

    raw_tool_name = tool_name.strip()
    if raw_tool_name != tool_name:
        return None, _error(
            "raw Ouroboros MCP tool name must not contain leading or trailing whitespace",
            tool_name=tool_name,
            registry_name=None,
        )
    if raw_tool_name.startswith("mcp_"):
        return None, _error(
            "Pass the raw Ouroboros MCP tool name, not the mcp_ouroboros_ registry prefix",
            tool_name=raw_tool_name,
            registry_name=None,
        )
    safe_tool_name = sanitize_mcp_name_component(raw_tool_name)
    if safe_tool_name != raw_tool_name:
        return None, _error(
            (
                "raw Ouroboros MCP tool name contains unsupported characters; "
                "pass the exact safe raw name without hyphens, dots, or spaces"
            ),
            tool_name=raw_tool_name,
            registry_name=None,
        )
    if _SAFE_RAW_TOOL_RE.fullmatch(raw_tool_name) is None:
        return None, _error(
            (
                "raw Ouroboros MCP tool name must start with 'ouroboros_' "
                "and contain only ASCII letters, digits, and underscores"
            ),
            tool_name=raw_tool_name,
            registry_name=None,
        )
    return raw_tool_name, None


def _validate_args(
    args: dict | None,
    raw_tool_name: str,
    registry_name: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if args is None:
        return {}, None
    if not isinstance(args, dict):
        return None, _error(
            "args must be a dict or None",
            tool_name=raw_tool_name,
            registry_name=registry_name,
        )
    return args, None


def _validate_timeout(
    timeout: float,
    raw_tool_name: str,
    registry_name: str,
) -> tuple[float | None, dict[str, Any] | None]:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        return None, _error(
            "timeout must be a finite number greater than 0",
            tool_name=raw_tool_name,
            registry_name=registry_name,
        )
    timeout_value = float(timeout)
    if not math.isfinite(timeout_value) or timeout_value <= 0:
        return None, _error(
            "timeout must be a finite number greater than 0",
            tool_name=raw_tool_name,
            registry_name=registry_name,
        )
    return timeout_value, None


def _parse_handler_result(
    raw_result: Any,
    *,
    raw_tool_name: str,
    registry_name: str,
) -> dict[str, Any]:
    """Normalize an MCP handler return value into a dict response."""
    if isinstance(raw_result, dict):
        parsed: Any = raw_result
    else:
        raw_text = raw_result if isinstance(raw_result, str) else str(raw_result)
        try:
            parsed = json.loads(raw_text)
        except (json.JSONDecodeError, TypeError):
            return {"success": True, "result": _redact_text(raw_text)}

    if not isinstance(parsed, dict):
        return {"success": True, "result": _redact_payload(parsed)}

    response = dict(parsed)
    if "success" not in response:
        response["success"] = False if "error" in response else True

    if response.get("success") is False or "error" in response:
        response.setdefault("tool", raw_tool_name)
        response.setdefault("registry_name", registry_name)
        response.setdefault("server", OUROBOROS_MCP_SERVER)

    return _redact_payload(response)


def call_ouroboros_tool(
    tool_name: str,
    args: dict | None = None,
    timeout: float = 45.0,
) -> dict[str, Any]:
    """Call a raw Ouroboros MCP tool through the Hermes registry gate.

    The function never raises ordinary validation/availability/handler failures
    to gateway callers.  Instead it returns a structured ``{"success": False,
    ...}`` error with sensitive strings redacted.
    """
    raw_tool_name, validation_error = _validate_raw_tool_name(tool_name)
    if validation_error is not None:
        return validation_error
    assert raw_tool_name is not None

    registry_name = ouroboros_registry_tool_name(raw_tool_name)

    call_args, validation_error = _validate_args(args, raw_tool_name, registry_name)
    if validation_error is not None:
        return validation_error
    assert call_args is not None

    timeout_value, validation_error = _validate_timeout(timeout, raw_tool_name, registry_name)
    if validation_error is not None:
        return validation_error
    assert timeout_value is not None

    entry = registry.get_entry(registry_name)
    if entry is None:
        return _error(
            "Ouroboros MCP tool is not registered or available",
            tool_name=raw_tool_name,
            registry_name=registry_name,
        )

    if hasattr(entry, "toolset") and entry.toolset != OUROBOROS_MCP_TOOLSET:
        return _error(
            f"Registry entry toolset is {entry.toolset!r}, expected {OUROBOROS_MCP_TOOLSET!r}",
            tool_name=raw_tool_name,
            registry_name=registry_name,
        )

    check_fn = getattr(entry, "check_fn", None)
    if check_fn is not None:
        try:
            available = bool(check_fn())
        except Exception as exc:
            return _error(
                f"Ouroboros MCP tool availability check failed: {type(exc).__name__}: {exc}",
                tool_name=raw_tool_name,
                registry_name=registry_name,
            )
        if not available:
            return _error(
                "Ouroboros MCP server is not connected or the tool is not available",
                tool_name=raw_tool_name,
                registry_name=registry_name,
            )

    try:
        handler = _handler_factory(OUROBOROS_MCP_SERVER, raw_tool_name, timeout_value)
        raw_result = handler(call_args)
    except Exception as exc:
        return _error(
            f"MCP call failed: {type(exc).__name__}: {exc}",
            tool_name=raw_tool_name,
            registry_name=registry_name,
        )

    return _parse_handler_result(
        raw_result,
        raw_tool_name=raw_tool_name,
        registry_name=registry_name,
    )
