"""Shared logic for the /claude-runtime slash command.

Toggles ``model.claude_runtime`` between "auto" (= default Hermes runtime,
uses API key) and "claude_subprocess" (= hand turns to the ``claude`` CLI
subprocess, uses subscription tokens).

Both CLI (cli.py) and gateway (gateway/run.py) call into this module so the
behavior stays identical across surfaces.

Mirrors ``codex_runtime_switch.py`` for the OpenAI Codex runtime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


VALID_RUNTIMES = ("auto", "claude_subprocess")


@dataclass
class ClaudeRuntimeStatus:
    """Result of a /claude-runtime invocation.  Callers render this however
    suits their surface (CLI uses Rich panels, gateway sends a text message)."""

    success: bool
    new_value: Optional[str] = None
    old_value: Optional[str] = None
    message: str = ""
    requires_new_session: bool = False
    claude_binary_ok: bool = True
    claude_version: Optional[str] = None


def parse_args(arg_string: str) -> tuple[Optional[str], list[str]]:
    """Parse the slash-command argument string.  Returns (value, errors).

    No args         → return current state (value=None)
    'auto' / 'claude_subprocess' / 'on' / 'off' → return that value
    anything else   → error
    """
    raw = (arg_string or "").strip().lower()
    if not raw:
        return None, []
    # Accept human-friendly synonyms
    if raw in {"on", "claude", "enable", "sub", "subscription"}:
        return "claude_subprocess", []
    if raw in {"off", "default", "disable", "hermes", "api"}:
        return "auto", []
    if raw in VALID_RUNTIMES:
        return raw, []
    return None, [
        f"Unknown runtime {raw!r}. Use one of: auto, claude_subprocess, on, off"
    ]


def get_current_runtime(config: dict) -> str:
    """Read the current ``model.claude_runtime`` value from a config dict.
    Returns 'auto' for unset / empty / unrecognized values."""
    if not isinstance(config, dict):
        return "auto"
    model_cfg = config.get("model") or {}
    if not isinstance(model_cfg, dict):
        return "auto"
    value = str(model_cfg.get("claude_runtime") or "").strip().lower()
    if value in VALID_RUNTIMES:
        return value
    return "auto"


def set_runtime(config: dict, new_value: str) -> str:
    """Mutate the config dict in place to persist the new runtime value.
    Returns the previous value for callers that want to report a delta."""
    if new_value not in VALID_RUNTIMES:
        raise ValueError(
            f"invalid runtime {new_value!r}; must be one of {VALID_RUNTIMES}"
        )
    old = get_current_runtime(config)
    if not isinstance(config.get("model"), dict):
        config["model"] = {}
    config["model"]["claude_runtime"] = new_value
    return old


def check_claude_binary_ok() -> tuple[bool, Optional[str]]:
    """Best-effort verification that ``claude`` CLI is installed at
    acceptable version.  Returns (ok, version_or_message)."""
    try:
        from hermes_cli.claude_models import check_claude_binary
        return check_claude_binary()
    except Exception as exc:
        return False, f"claude check failed: {exc}"


def apply(
    config: dict,
    new_value: Optional[str],
    *,
    persist_callback=None,
) -> ClaudeRuntimeStatus:
    """Top-level entry point used by both CLI and gateway handlers.

    Args:
        config: in-memory config dict (will be mutated when new_value is set)
        new_value: desired runtime; None means "show current state only"
        persist_callback: optional callable taking the mutated config dict
            and persisting it to disk.  Skipped when None (used by tests).

    Returns: ClaudeRuntimeStatus describing the outcome.
    """
    current = get_current_runtime(config)

    # Cache the binary check for this apply() call.
    _binary_check: Optional[tuple[bool, Optional[str]]] = None

    def _check_binary_cached() -> tuple[bool, Optional[str]]:
        nonlocal _binary_check
        if _binary_check is None:
            _binary_check = check_claude_binary_ok()
        return _binary_check

    # Read-only call: just report state
    if new_value is None:
        ok, ver = _check_binary_cached()

        # Check auth status
        auth_status = _check_auth_status()

        msg_lines = [
            f"claude_runtime: {current}",
            f"claude CLI: {'OK ' + ver if ok else 'not available — ' + (ver or 'install with: npm i -g @anthropic-ai/claude-code')}",
        ]
        if auth_status:
            msg_lines.append(f"auth: {auth_status}")

        return ClaudeRuntimeStatus(
            success=True,
            new_value=current,
            old_value=current,
            message="\n".join(msg_lines),
            claude_binary_ok=ok,
            claude_version=ver if ok else None,
        )

    # No change requested
    if new_value == current:
        return ClaudeRuntimeStatus(
            success=True,
            new_value=current,
            old_value=current,
            message=f"claude_runtime already set to {current}",
        )

    # If switching ON, verify claude CLI is installed before persisting.
    if new_value == "claude_subprocess":
        ok, ver_or_msg = _check_binary_cached()
        if not ok:
            return ClaudeRuntimeStatus(
                success=False,
                new_value=None,
                old_value=current,
                message=(
                    "Cannot enable claude_subprocess runtime: "
                    f"{ver_or_msg or 'claude CLI not available'}\n"
                    "Install with: npm i -g @anthropic-ai/claude-code"
                ),
                claude_binary_ok=False,
                claude_version=None,
            )

    set_runtime(config, new_value)
    if persist_callback is not None:
        try:
            persist_callback(config)
        except Exception as exc:
            logger.exception("failed to persist claude_runtime change")
            return ClaudeRuntimeStatus(
                success=False,
                new_value=new_value,
                old_value=current,
                message=f"updated config in memory but persist failed: {exc}",
            )

    msg_lines = [
        f"claude_runtime: {current} → {new_value}",
    ]
    if new_value == "claude_subprocess":
        ok, ver = _check_binary_cached()
        if ok:
            msg_lines.append(f"claude CLI: {ver}")
        auth_status = _check_auth_status()
        if auth_status:
            msg_lines.append(f"auth: {auth_status}")
        msg_lines.append(
            "Anthropic/Claude turns now run through `claude -p` subprocess "
            "(uses your Claude Pro/Max subscription tokens, NOT API credits)."
        )
        msg_lines.append(
            "Effective on next session — current cached agent keeps "
            "the prior runtime to preserve prompt cache."
        )
    else:
        msg_lines.append(
            "Anthropic/Claude turns will use the default Hermes runtime "
            "(API key from .env)."
        )
        msg_lines.append("Effective on next session.")

    return ClaudeRuntimeStatus(
        success=True,
        new_value=new_value,
        old_value=current,
        message="\n".join(msg_lines),
        requires_new_session=True,
    )


def _check_auth_status() -> Optional[str]:
    """Check if Claude Code has valid OAuth credentials.

    Returns a human-readable status string or None.
    """
    try:
        from agent.anthropic_adapter import (
            read_claude_code_credentials,
            is_claude_code_token_valid,
        )

        creds = read_claude_code_credentials()
        if not creds:
            return "no Claude Code credentials found — run `claude login`"
        if is_claude_code_token_valid(creds):
            source = creds.get("source", "unknown")
            return f"valid OAuth token ({source})"
        return "OAuth token expired — run `claude login` to refresh"
    except Exception as exc:
        logger.debug("auth status check failed: %s", exc)
        return None
