"""Registry of known ACP-compatible coding agents.

Maps short agent names to their ACP launch commands and auth environment
variable requirements.  Each command spawns a process that speaks the Agent
Client Protocol over stdio (NDJSON / JSON-RPC).

Resolution order for a given agent name:
  1. Environment variable  HERMES_ACP_{NAME}_COMMAND  (uppercased name)
  2. User config.yaml      acp_agents.<name>.command
  3. Built-in registry      ACP_AGENT_REGISTRY[name]

The command string is split with shlex — shell features are NOT supported.
"""

from __future__ import annotations

import logging
import os
import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent entry with auth env mapping
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
    Hermes's credential names to what the ACP adapter expects.

    Example: (("ANTHROPIC_API_KEY", ("ANTHROPIC_TOKEN", "ANTHROPIC_API_KEY")),)
    means: if ANTHROPIC_API_KEY is empty, try ANTHROPIC_TOKEN.
    """


# ---------------------------------------------------------------------------
# Built-in agent registry
# ---------------------------------------------------------------------------

ACP_AGENT_REGISTRY: Dict[str, ACPAgentEntry] = {
    "claude": ACPAgentEntry(
        command="npx -y @agentclientprotocol/claude-agent-acp@0.25.0",
        auth_env=(
            ("ANTHROPIC_API_KEY", ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")),
        ),
    ),
    "codex": ACPAgentEntry(
        command="npx -y @zed-industries/codex-acp@0.9.5",
        auth_env=(
            ("OPENAI_API_KEY", ("OPENAI_API_KEY",)),
        ),
    ),
    "gemini": ACPAgentEntry(
        command="gemini --acp",
    ),
    "copilot": ACPAgentEntry(
        command="copilot --acp --stdio",
    ),
    "cursor": ACPAgentEntry(command="cursor-agent acp"),
    "kiro": ACPAgentEntry(command="kiro-cli acp"),
    "kilocode": ACPAgentEntry(command="npx -y @kilocode/cli acp"),
    "opencode": ACPAgentEntry(command="npx -y opencode-ai acp"),
    "kimi": ACPAgentEntry(command="kimi acp"),
    "qwen": ACPAgentEntry(command="qwen --acp"),
    "droid": ACPAgentEntry(command="droid exec --output-format acp"),
    "iflow": ACPAgentEntry(command="iflow --experimental-acp"),
    "cline": ACPAgentEntry(command="npx -y cline --acp"),
    "amp": ACPAgentEntry(command="amp --acp"),
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
