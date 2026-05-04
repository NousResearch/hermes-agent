"""b00t guard interposition plugin.

Intercepts terminal tool calls and routes them through ``b00t hive run --dry-run``
for guard evaluation. Handles:

- **warn + redirect** (🦨): rewrites the command (e.g. pip → uv pip)
- **block** (🚫💩): returns an error message, command never executes
- **pass** (no emoji): lets the command through unchanged

Config:
    Set ``terminal.backend: local`` in config.yaml — b00t intercepts at
    the plugin level, not as a full environment backend replacement.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import sys
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

# Emoji markers produced by b00t hive run guards
_BLOCK_EMOJIS = {"🚫", "💩"}
_WARN_EMOJI = "🦨"
_B00T_EMOJI = "🥾"  # b00t identity marker for all interposition output


def _run_b00t_guard(cmd: str) -> dict[str, Any]:
    """Run ``b00t hive run --dry-run <cmd>`` and parse the result.

    Returns:
        {"action": "pass"}                   — no guard matched
        {"action": "warn", "message": ...,
         "redirect": "rewritten cmd"}        — guard warned, may redirect
        {"action": "block", "message": ...}  — guard blocked
    """
    try:
        result = subprocess.run(
            ["b00t-cli", "hive", "run", "--dry-run", "--", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        # Forward guard output to user's stderr + capture for parsing
        if result.stderr:
            sys.stderr.write(result.stderr)
            sys.stderr.flush()
            for line in result.stderr.splitlines():
                if line.strip():
                    logger.info("%s", line.strip())
    except FileNotFoundError:
        logger.debug("b00t-cli not found — guard interposition disabled")
        return {"action": "pass"}
    except subprocess.TimeoutExpired:
        logger.warning("🥾 b00t guard timed out for cmd: %.60s", cmd)
        return {"action": "pass"}

    stdout = result.stdout or ""
    stderr = result.stderr or ""

    # Try JSONL parsing first (structured contract, v2)
    for line in stdout.split("\n"):
        stripped = line.strip()
        if stripped.startswith("{"):
            try:
                j = json.loads(stripped)
                action = j.get("action", "pass")
                if action == "block":
                    msg = j.get("message", "") or j.get("error", "blocked by guard")
                    return {"action": "block", "message": msg}
                if action == "warn":
                    return {"action": "warn", "message": j.get("message", ""), "redirect": j.get("redirect")}
                if action == "pass":
                    return {"action": "pass"}
            except (json.JSONDecodeError, KeyError):
                continue  # fall through to emoji scraping

    # Fallback: emoji scraping from combined output (v1 compat)
    combined = stdout + stderr

    for emoji in _BLOCK_EMOJIS:
        if emoji in combined:
            msg = ""
            for line in combined.split("\n"):
                if emoji in line:
                    msg = line.strip()
                    break
            return {"action": "block", "message": msg or "blocked by guard"}

    if _WARN_EMOJI in combined:
        redirect = None
        msg = ""
        for line in combined.split("\n"):
            stripped = line.strip()
            if _WARN_EMOJI in stripped:
                msg = stripped
            if "suggested:" in stripped.lower() or "redirect" in stripped.lower():
                parts = stripped.split(":", 1)
                if len(parts) > 1:
                    redirect = parts[1].strip()
        return {"action": "warn", "message": msg, "redirect": redirect}

    return {"action": "pass"}


def _b00t_pre_tool_hook(
    tool_name: str,
    args: dict[str, Any] | None,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Intercept terminal tool calls and route through b00t guards."""
    if tool_name != "terminal":
        return None  # only intercept terminal commands

    if not args:
        return None

    cmd = args.get("command", "")
    if not cmd:
        return None

    guard_result = _run_b00t_guard(cmd)

    if guard_result["action"] == "block":
        return {
            "action": "block",
            "message": guard_result.get(
                "message", "Command blocked by b00t guard"
            ),
        }

    if guard_result["action"] == "warn":
        redirect = guard_result.get("redirect")
        if redirect:
            # Rewrite the command to the redirected version
            new_args = dict(args)
            new_args["command"] = redirect
            return {"action": "rewrite", "args": new_args}
        # Warn only — let it through, message is logged by guard output
        return None

    # Pass through
    return None


def register(ctx) -> None:
    """Register the b00t guard interposition hook."""
    ctx.register_hook("pre_tool_call", _b00t_pre_tool_hook)
    logger.info("🥾 b00t guard interposition plugin registered")
