"""Registry of known ACP-compatible coding agents.

All agents are spawned via ``acpx`` (the standalone ACP headless client),
which handles protocol negotiation, authentication (including OAuth gateway
handshake), session management, and credential resolution.

Resolution order for a given agent name:
  1. Environment variable  HERMES_ACP_{NAME}_COMMAND  (uppercased name)
  2. Built-in registry      ACP_AGENT_REGISTRY[name]

The command string is split with shlex — shell features are NOT supported.
"""

from __future__ import annotations

import logging
import os
import shlex
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent entry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ACPAgentEntry:
    """Registry entry for an ACP agent."""

    command: str
    """Shell command that starts the ACP agent on stdin/stdout."""

    auth_env: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    """Auth environment variable mapping: ((target_var, (source_var, ...)), ...)

    For each pair, if *target_var* is not set in the environment, search
    *source_vars* in order and use the first non-empty value.  This bridges
    Hermes's credential names to what acpx/the ACP adapter expects.
    """


# ---------------------------------------------------------------------------
# Built-in agent registry — all routed through acpx
# ---------------------------------------------------------------------------

ACP_AGENT_REGISTRY: Dict[str, ACPAgentEntry] = {
    "claude": ACPAgentEntry(
        command="npx -y acpx claude",
        auth_env=(
            ("ANTHROPIC_API_KEY", ("ANTHROPIC_API_KEY",)),
            ("ANTHROPIC_AUTH_TOKEN", ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")),
        ),
    ),
    "codex": ACPAgentEntry(
        command="npx -y acpx codex",
        auth_env=(
            ("OPENAI_API_KEY", ("OPENAI_API_KEY",)),
        ),
    ),
    "gemini": ACPAgentEntry(
        command="npx -y acpx gemini",
    ),
    "copilot": ACPAgentEntry(
        command="npx -y acpx copilot",
    ),
    "cursor": ACPAgentEntry(command="npx -y acpx cursor"),
    "kiro": ACPAgentEntry(command="npx -y acpx kiro"),
    "kilocode": ACPAgentEntry(command="npx -y acpx kilocode"),
    "opencode": ACPAgentEntry(command="npx -y acpx opencode"),
    "kimi": ACPAgentEntry(command="npx -y acpx kimi"),
    "qwen": ACPAgentEntry(command="npx -y acpx qwen"),
    "droid": ACPAgentEntry(command="npx -y acpx droid"),
    "iflow": ACPAgentEntry(command="npx -y acpx iflow"),
    "cline": ACPAgentEntry(command="npx -y acpx cline"),
    "amp": ACPAgentEntry(command="npx -y acpx amp"),
}


def resolve_agent_command(agent_name: str) -> Optional[str]:
    """Return the ACP launch command for *agent_name*, or ``None``.

    Checks env-var override first, then the built-in registry.
    """
    name_upper = agent_name.upper().replace("-", "_")
    env_key = f"HERMES_ACP_{name_upper}_COMMAND"
    env_val = os.getenv(env_key, "").strip()
    if env_val:
        logger.debug("ACP agent '%s' resolved via %s", agent_name, env_key)
        return env_val

    normalized = agent_name.lower().strip()
    entry = ACP_AGENT_REGISTRY.get(normalized)
    if entry:
        logger.debug("ACP agent '%s' resolved from built-in registry", agent_name)
        return entry.command
    return None


def resolve_agent_env(agent_name: str) -> Dict[str, str]:
    """Build auth environment variables for *agent_name*.

    For each ``auth_env`` mapping on the agent entry, resolve the first
    non-empty source variable and set it as the target.  Returns a dict
    of env vars to inject into the subprocess (only includes vars that
    were actually resolved).
    """
    normalized = agent_name.lower().strip()
    entry = ACP_AGENT_REGISTRY.get(normalized)
    if not entry or not entry.auth_env:
        return {}

    env_patch: Dict[str, str] = {}
    for target_var, source_vars in entry.auth_env:
        # Skip if target is already set in the real environment
        if os.environ.get(target_var, "").strip():
            continue
        # Search source vars for a value
        for src in source_vars:
            val = os.environ.get(src, "").strip()
            if val:
                logger.debug(
                    "ACP agent '%s': bridging %s -> %s",
                    agent_name, src, target_var,
                )
                env_patch[target_var] = val
                break
    return env_patch


def split_agent_command(command: str) -> List[str]:
    """Split a command string into argv suitable for ``subprocess.Popen``."""
    return shlex.split(command)


def list_agents() -> List[str]:
    """Return sorted list of all known agent names."""
    return sorted(ACP_AGENT_REGISTRY.keys())
