# gateway/daimon/gateway_hooks.py
"""Gateway integration hooks for Daimon.

Provides the bridge between gateway/run.py's _run_agent() and the Daimon subsystem.
The gateway calls these functions at specific points in agent construction and response delivery.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from gateway.daimon.agent_overrides import AgentOverrides, compute_overrides
from gateway.daimon.tool_gate import register_limiter, unregister_limiter, check_tool_call
from gateway.daimon.tool_limiter import ToolLimiter
from gateway.daimon.config import load_daimon_config
from gateway.daimon.redaction import redact_response

logger = logging.getLogger(__name__)

# Path to the Daimon system prompt (relative to this file)
_SYSTEM_PROMPT_PATH = Path(__file__).parent / "daimon-system-prompt.md"


def get_agent_overrides(
    raw_config: dict,
    user_id: str,
    platform: str,
    role_ids: Optional[list[str]] = None,
) -> Optional[AgentOverrides]:
    """Get Daimon tier-based overrides for agent construction.

    Called by gateway/run.py before constructing AIAgent.
    Returns None if Daimon is not active or platform is not Discord.
    Returns AgentOverrides with tier=None if user should be silently ignored.
    """
    return compute_overrides(raw_config, user_id, platform, role_ids=role_ids)


def load_system_prompt() -> str:
    """Load the Daimon system prompt text.

    Returns empty string if file not found.
    """
    if _SYSTEM_PROMPT_PATH.exists():
        return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    return ""


def setup_tool_gate(session_id: str, raw_config: dict) -> None:
    """Register a tool limiter for a Daimon user session.

    Called after agent construction for non-admin sessions.
    The limiter is checked on every tool call via check_tool_call().
    """
    cfg = load_daimon_config(raw_config)
    limiter = ToolLimiter(cfg.tool_limits)
    register_limiter(session_id, limiter)
    logger.debug("[Daimon] Registered tool limiter for session %s", session_id)


def teardown_tool_gate(session_id: str) -> None:
    """Remove tool limiter for a session (cleanup on session end).

    Called in the finally block after agent.run_conversation().
    """
    unregister_limiter(session_id)


def gate_tool_call(session_id: str, tool_name: str) -> Optional[str]:
    """Check if a tool call is allowed.

    Returns None if allowed, or a denial message string if blocked.
    Called from the pre_tool_call hook path.
    """
    return check_tool_call(session_id, tool_name)


def redact_output(text: str) -> str:
    """Apply output redaction to agent response.

    Called before sending response to Discord for non-admin sessions.
    """
    return redact_response(text)


def apply_overrides(
    overrides: AgentOverrides,
    *,
    model: str,
    max_iterations: int,
    disabled_toolsets: list[str] | None,
    source=None,
) -> dict:
    """Apply AgentOverrides to the current agent construction params.

    Returns a dict with the modified values:
        - model: str
        - max_iterations: int
        - disabled_toolsets: list[str] | None
        - ephemeral_system_prompt: str | None

    The caller unpacks these into the AIAgent constructor.

    When *source* (a SessionSource) is provided, template variables in the
    system prompt are resolved:
        - <DISCORD_THREAD_URL> → full Discord thread URL
        - <THREAD_ID> → raw thread/channel ID
    """
    result_model = overrides.model or model
    result_iterations = overrides.max_iterations if overrides.max_iterations is not None else max_iterations

    # Merge disabled toolsets (additive)
    result_disabled = list(disabled_toolsets or [])
    if overrides.disabled_toolsets:
        result_disabled = list(set(result_disabled + overrides.disabled_toolsets))

    # Load system prompt for non-admin users
    prompt = None
    if not overrides.tier.is_admin:
        prompt = load_system_prompt() or None
        if prompt and source:
            prompt = _resolve_prompt_vars(prompt, source)

    return {
        "model": result_model,
        "max_iterations": result_iterations,
        "disabled_toolsets": result_disabled or None,
        "ephemeral_system_prompt": prompt,
    }


def _resolve_prompt_vars(prompt: str, source) -> str:
    """Resolve template variables in the Daimon system prompt.

    Variables:
        <DISCORD_THREAD_URL> — full clickable Discord thread URL
        <THREAD_ID> — raw thread/channel ID
    """
    # Thread ID is chat_id for thread-type sessions (the thread IS the channel)
    thread_id = source.thread_id or source.chat_id or ""
    guild_id = getattr(source, "guild_id", "") or ""

    # Build the Discord thread URL
    if guild_id and thread_id:
        thread_url = f"https://discord.com/channels/{guild_id}/{thread_id}"
    else:
        thread_url = f"(thread URL unavailable — guild_id={guild_id}, thread_id={thread_id})"

    prompt = prompt.replace("<DISCORD_THREAD_URL>", thread_url)
    prompt = prompt.replace("<THREAD_ID>", thread_id)
    return prompt


# ── Module-level turn counter (accessible from gateway/run.py) ──
# Same pattern as tool_gate.py — module-level registry keyed by thread_id.
import threading

_turn_lock = threading.Lock()
_turn_counts: dict[str, int] = {}


def increment_thread_turn(thread_id: str) -> None:
    """Increment turn counter for a thread after agent response delivery."""
    with _turn_lock:
        _turn_counts[thread_id] = _turn_counts.get(thread_id, 0) + 1
    # Persist to DB (best-effort, non-blocking)
    try:
        from gateway.daimon.persistence import DaimonDB
        from hermes_constants import get_hermes_home
        _db_path = get_hermes_home() / "daimon.db"
        if _db_path.exists():
            db = DaimonDB(_db_path)
            db.increment_turn(thread_id)
            db.close()
    except Exception:
        pass


def get_thread_turns(thread_id: str) -> int:
    """Get current turn count for a thread."""
    with _turn_lock:
        return _turn_counts.get(thread_id, 0)


def clear_thread_turns(thread_id: str) -> None:
    """Clear turn count for a thread (cleanup)."""
    with _turn_lock:
        _turn_counts.pop(thread_id, None)
