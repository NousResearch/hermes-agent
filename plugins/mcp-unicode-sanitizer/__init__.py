"""mcp-unicode-sanitizer — Hermes MCP Gateway plugin.

Hooks into the MCP tool-metadata pipeline right after the ``tools/list``
handshake completes. Every MCP tool description and inputSchema string surface
is sanitized before it can reach an approval dialog or model context.

The plugin registers a single ``sanitize_tool_metadata`` hook. The MCP gateway
core (``tools/mcp_tool.py``) invokes that hook per tool at discovery time, with
a mutable copy of the tool definition. The handler:

  1. Runs the vendored Unicode sanitization library over the whole tool
     (name + description + every inputSchema string surface).
  2. Returns ``{"tool": {...}}`` containing the sanitized definition when safe.
  3. Returns ``{"quarantine": "reason"}`` when the tool is not safe — the core
     then skips registering it (it is never delivered to approval dialogs or
     the model).

The handler is fail-closed and fail-safe: it never raises, never blocks the
gateway, and never lets a sanitized-dangerous tool through. Configuration is
read from ``plugins.entries.mcp-unicode-sanitizer.*`` in config.yaml.

Configuration keys (all optional):
    quarantine_on_flag (bool, default true): drop tools whose residual text
        trips the keyword detector OR which carried concealment (TAG/bidi/
        invisible). When false, concealment-carrying tools are still sanitized
        (encoding stripped) but are allowed through.
    log_level (str, default "info"): "debug" | "info" | "warning".
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

try:  # directory / dev-checkout import
    from .sanitizer import sanitize_tool_metadata
except ImportError:  # pragma: no cover - alternate layout
    from sanitizer import sanitize_tool_metadata  # type: ignore[no-redef]

_PLUGIN_ID = "mcp-unicode-sanitizer"

# Fail-closed default: quarantine anything that trips the detector or carried
# a concealment encoding (TAG/bidi/invisible).
_DEFAULT_QUARANTINE_ON_FLAG = True

_logger: logging.Logger = logging.getLogger(f"hermes_plugins.{_PLUGIN_ID}")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _load_config() -> Dict[str, Any]:
    """Read plugin config from config.yaml. Never raises.

    Layout:
        plugins:
          entries:
            mcp-unicode-sanitizer:
              quarantine_on_flag: true
              log_level: "info"
    """
    try:
        from hermes_cli.config import load_config
        config = load_config() or {}
    except Exception:  # pragma: no cover - environment-dependent
        config = {}
    try:
        entries = (config.get("plugins") or {}).get("entries") or {}
        return dict(entries.get(_PLUGIN_ID) or {})
    except Exception:
        return {}


def _quarantine_on_flag(cfg: Dict[str, Any]) -> bool:
    val = cfg.get("quarantine_on_flag", _DEFAULT_QUARANTINE_ON_FLAG)
    return bool(val) if isinstance(val, bool) else _DEFAULT_QUARANTINE_ON_FLAG


def _log_level(cfg: Dict[str, Any]) -> str:
    val = cfg.get("log_level", "info")
    return val if isinstance(val, str) and val in ("debug", "info", "warning") else "info"


# ---------------------------------------------------------------------------
# Hook handler
# ---------------------------------------------------------------------------


def _sanitize_tool(tool: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Sanitize a single MCP tool definition (dict). Returns the hook payload.

    The handler must never raise. On any unexpected input/error it fails
    closed to quarantine (never delivers an unsanitized tool).

    Hook contract (defined in tools/mcp_tool.py):
        * return ``{"tool": {...}}`` -> use the sanitized tool dict.
        * return ``{"quarantine": "<reason>"}`` -> skip registration.
        * return ``None`` -> leave the tool untouched (plugin disabled path
          never reaches here; None is returned only when the input is not a
          well-formed MCP tool, which we treat as "can't sanitize -> let the
          core's existing checks handle it").
    """
    if not isinstance(tool, dict):
        return None

    # Build a defensive copy so we never mutate the caller's object on error.
    definition = {
        "name": tool.get("name", ""),
        "description": tool.get("description", "") or "",
        "inputSchema": tool.get("inputSchema", tool.get("input_schema")) or {},
    }

    try:
        st = sanitize_tool_metadata(definition)
    except Exception as exc:  # fail-closed
        _logger.warning(
            "mcp-unicode-sanitizer: sanitizer raised for tool %r — quarantining (%s)",
            definition.get("name"), exc,
        )
        return {"quarantine": f"sanitizer failure ({exc})"}

    safe = st.safe
    flagged_fields = [p for p, r in st.field_results.items() if r.flagged]

    level = _log_level(cfg)
    if not safe:
        _logger.log(
            logging.DEBUG if level == "debug" else logging.INFO,
            "mcp-unicode-sanitizer: tool %r %s",
            definition.get("name"),
            _describe_unsafe(st, flagged_fields),
        )

    if not safe and _quarantine_on_flag(cfg):
        reasons = []
        if st.flagged:
            reasons.append(f"flagged fields: {', '.join(flagged_fields)}")
        if st.dangerous_defaults:
            reasons.append(f"dangerous schema defaults: {', '.join(st.dangerous_defaults)}")
        return {"quarantine": "; ".join(reasons) or "unsafe metadata"}

    # Safe (or quarantine disabled): deliver the sanitized definition.
    return {
        "tool": {
            "name": st.name,
            "description": st.description,
            "inputSchema": st.input_schema,
        }
    }


def _describe_unsafe(st, flagged_fields) -> str:
    parts = []
    if flagged_fields:
        parts.append("flagged=" + ",".join(flagged_fields))
    if st.dangerous_defaults:
        parts.append("dangerous_defaults=" + ",".join(st.dangerous_defaults))
    return "unsafe: " + ("; ".join(parts) if parts else "unspecified")


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Register the sanitize_tool_metadata hook (Hermes plugin convention)."""
    cfg = _load_config()

    def _handler(tool=None, **_kwargs) -> Optional[Dict[str, Any]]:
        # Hook contract: the MCP gateway passes the raw tool dict. Delegate to
        # the real handler so it can be unit-tested independently of ctx.
        return _sanitize_tool(tool, cfg)

    ctx.register_hook("sanitize_tool_metadata", _handler)
    _logger.log(
        logging.DEBUG if _log_level(cfg) == "debug" else logging.INFO,
        "mcp-unicode-sanitizer: registered sanitize_tool_metadata hook "
        "(quarantine_on_flag=%s)",
        _quarantine_on_flag(cfg),
    )
