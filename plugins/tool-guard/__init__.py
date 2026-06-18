"""
tool-guard — Hermes plugin to intercept and guard tool calls.

When a user specifies a tool like "Claude Code" or "Codex" in a
delegate_task goal, and that tool has recently failed with connectivity
or proxy errors, this plugin blocks the call and returns a clear error
message instead of allowing silent fallback or substitution.

Configuration
-------------
Reads from plugin.yaml ``config`` section, overridable via
``~/.hermes/tool-guard-config.yaml`` (external file takes precedence).

See plugin.yaml for all available settings.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Failure state — module-level singleton, thread-safe
# ---------------------------------------------------------------------------

_lock = Lock()
# {tool_name_lower: {"failures": int, "last_failure": float, "last_error": str}}
_failure_state: Dict[str, Dict[str, Any]] = {}


def _load_config() -> Dict[str, Any]:
    """Load config from plugin.yaml defaults, overridden by external file."""
    defaults = {
        "guarded_tools": ["claude-code", "claude code", "codex"],
        "failure_patterns": [
            "connection refused",
            "connect refused",
            "proxy error",
            "ERR_PROXY",
            "tunnel.*fail",
            "ECONNREFUSED",
            "ETIMEDOUT",
            "socket hang up",
            "502 Bad Gateway",
            "503 Service Unavailable",
            "504 Gateway Timeout",
            "Could not connect",
            "unable to connect",
            "No API key found",
            "provider.*unavailable",
            "rate limit",
        ],
        "cooldown_seconds": 300,
        "max_consecutive_failures": 1,
    }

    # Try external config file first
    external = Path.home() / ".hermes" / "tool-guard-config.yaml"
    if external.exists():
        try:
            import yaml

            with open(external) as f:
                user_cfg = yaml.safe_load(f) or {}
            if isinstance(user_cfg, dict):
                defaults.update(user_cfg)
                logger.debug("tool-guard: loaded external config from %s", external)
                return defaults
        except Exception as exc:
            logger.warning("tool-guard: failed to load %s: %s", external, exc)

    # Try plugin.yaml config section
    plugin_yaml = Path(__file__).parent / "plugin.yaml"
    if plugin_yaml.exists():
        try:
            import yaml

            with open(plugin_yaml) as f:
                manifest = yaml.safe_load(f) or {}
            plugin_cfg = manifest.get("config", {})
            if isinstance(plugin_cfg, dict):
                defaults.update(plugin_cfg)
        except Exception as exc:
            logger.debug("tool-guard: failed to read plugin.yaml config: %s", exc)

    return defaults


_config: Optional[Dict[str, Any]] = None


def _get_config() -> Dict[str, Any]:
    global _config
    if _config is None:
        _config = _load_config()
    return _config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mentions_guarded_tool(goal_text: str, guarded_tools: List[str]) -> Optional[str]:
    """Return the first guarded tool name mentioned in *goal_text*, or None."""
    text_lower = goal_text.lower()
    for tool_name in guarded_tools:
        if tool_name.lower() in text_lower:
            return tool_name
    return None


def _is_failure_active(tool_key: str, cooldown: int, min_failures: int) -> Optional[str]:
    """Check if the given tool has an active failure within cooldown window.

    Returns the last error string if blocked, or None if allowed.
    """
    with _lock:
        state = _failure_state.get(tool_key)
        if state is None:
            return None
        if state["failures"] < min_failures:
            return None
        elapsed = time.monotonic() - state["last_failure"]
        if elapsed > cooldown:
            # Cooldown expired — clear state
            del _failure_state[tool_key]
            return None
        return state["last_error"]


def _record_failure(tool_key: str, error_summary: str) -> None:
    """Record a connectivity failure for *tool_key*."""
    with _lock:
        if tool_key in _failure_state:
            _failure_state[tool_key]["failures"] += 1
            _failure_state[tool_key]["last_failure"] = time.monotonic()
            _failure_state[tool_key]["last_error"] = error_summary
        else:
            _failure_state[tool_key] = {
                "failures": 1,
                "last_failure": time.monotonic(),
                "last_error": error_summary,
            }
        logger.info(
            "tool-guard: recorded failure for '%s' (count=%d): %s",
            tool_key,
            _failure_state[tool_key]["failures"],
            error_summary[:200],
        )


def _clear_failure(tool_key: str) -> None:
    """Clear failure state for *tool_key* after a successful call."""
    with _lock:
        if tool_key in _failure_state:
            del _failure_state[tool_key]
            logger.info("tool-guard: cleared failure state for '%s'", tool_key)


def _compile_patterns(patterns: List[str]) -> List[re.Pattern]:
    """Compile a list of regex pattern strings (case-insensitive)."""
    compiled = []
    for p in patterns:
        try:
            compiled.append(re.compile(p, re.IGNORECASE))
        except re.error:
            logger.warning("tool-guard: invalid regex pattern skipped: %s", p)
    return compiled


def _matches_failure_pattern(text: str, patterns: List[re.Pattern]) -> Optional[str]:
    """Return the first matching pattern's string, or None."""
    for pat in patterns:
        m = pat.search(text)
        if m:
            return m.group(0)
    return None


# ---------------------------------------------------------------------------
# Hook: pre_tool_call
# ---------------------------------------------------------------------------

def _pre_tool_call(
    tool_name: str,
    args: Optional[Dict[str, Any]] = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    **_kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """Block delegate_task calls to recently-failed guarded tools.

    Returns ``{"action": "block", "message": "..."}`` when the tool should
    be blocked, or ``None`` to allow the call.
    """
    if tool_name != "delegate_task":
        return None
    if not isinstance(args, dict):
        return None

    cfg = _get_config()
    guarded_tools: List[str] = cfg.get("guarded_tools", [])
    cooldown: int = cfg.get("cooldown_seconds", 300)
    min_failures: int = cfg.get("max_consecutive_failures", 1)

    # Extract the goal text from delegate_task args
    goal = args.get("goal", "") or ""
    if not goal:
        return None

    mentioned = _mentions_guarded_tool(goal, guarded_tools)
    if mentioned is None:
        return None

    # Normalize the tool key for failure lookup
    tool_key = mentioned.lower().strip()

    error_info = _is_failure_active(tool_key, cooldown, min_failures)
    if error_info is not None:
        msg = (
            f"⚠️ **tool-guard**: Blocked delegate_task — the tool **{mentioned}** "
            f"has recently failed due to a connectivity issue and is being "
            f"guarded to prevent silent substitution.\n\n"
            f"**Last error:** {error_info}\n\n"
            f"**Cooldown:** {cooldown}s remaining (failures are tracked per-tool). "
            f"Retry after the cooldown expires, or resolve the connectivity issue.\n\n"
            f"To clear the block immediately, you can reset the failure state "
            f"or adjust config in ~/.hermes/tool-guard-config.yaml."
        )
        logger.warning("tool-guard: BLOCKED delegate_task — '%s' is down: %s", mentioned, error_info)
        return {"action": "block", "message": msg}

    return None


# ---------------------------------------------------------------------------
# Hook: post_tool_call
# ---------------------------------------------------------------------------

_failure_regexes: Optional[List[re.Pattern]] = None


def _post_tool_call(
    tool_name: str,
    args: Optional[Dict[str, Any]] = None,
    result: Any = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    duration_ms: int = 0,
    **_kwargs: Any,
) -> None:
    """Monitor delegate_task results for connectivity failure patterns.

    When a failure is detected, records the state so subsequent
    pre_tool_call hooks can block the same tool.
    """
    if tool_name != "delegate_task":
        return

    cfg = _get_config()
    guarded_tools: List[str] = cfg.get("guarded_tools", [])

    # Compile patterns lazily
    global _failure_regexes
    if _failure_regexes is None:
        _failure_regexes = _compile_patterns(cfg.get("failure_patterns", []))

    # Serialize result to string for pattern matching
    if result is None:
        return
    if isinstance(result, dict):
        result_text = json.dumps(result)
    elif isinstance(result, str):
        result_text = result
    else:
        result_text = str(result)

    # Check for failure patterns
    match = _matches_failure_pattern(result_text, _failure_regexes)
    if match is None:
        # No failure — if this was a guarded tool call, clear its state
        if isinstance(args, dict):
            goal = args.get("goal", "") or ""
            mentioned = _mentions_guarded_tool(goal, guarded_tools)
            if mentioned:
                _clear_failure(mentioned.lower().strip())
        return

    # Failure detected — figure out which guarded tool was involved
    if not isinstance(args, dict):
        return

    goal = args.get("goal", "") or ""
    mentioned = _mentions_guarded_tool(goal, guarded_tools)
    if mentioned is None:
        # Failure in delegate_task but not about a guarded tool — ignore
        return

    tool_key = mentioned.lower().strip()

    # Extract a concise error summary from the result
    if isinstance(result, dict):
        error_summary = result.get("error", "") or result.get("output", "") or match
    else:
        # Truncate long results for the error summary
        error_summary = result_text[:500] if len(result_text) > 500 else result_text

    _record_failure(tool_key, error_summary)


# ---------------------------------------------------------------------------
# Plugin registration entrypoint
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Called by the Hermes plugin system to register hooks.

    Args:
        ctx: PluginContext instance providing register_hook() and metadata.
    """
    ctx.register_hook("pre_tool_call", _pre_tool_call)
    ctx.register_hook("post_tool_call", _post_tool_call)

    cfg = _get_config()
    logger.info(
        "tool-guard: loaded — guarding tools %s, cooldown=%ds, min_failures=%d",
        cfg.get("guarded_tools", []),
        cfg.get("cooldown_seconds", 300),
        cfg.get("max_consecutive_failures", 1),
    )
