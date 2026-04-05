"""Registry of known ACP-compatible coding agents.

Maps short agent names to their ACP launch commands.  Each command spawns a
process that speaks the Agent Client Protocol over stdio (NDJSON / JSON-RPC).

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
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in agent commands
# ---------------------------------------------------------------------------
# Each value is a shell command string that, when executed, starts an ACP
# agent on stdin/stdout.  Entries using ``npx -y`` auto-install on first run.

ACP_AGENT_REGISTRY: Dict[str, str] = {
    "claude": "npx -y @agentclientprotocol/claude-agent-acp@0.21.0",
    "codex": "npx -y @zed-industries/codex-acp@0.9.5",
    "gemini": "gemini --acp",
    "copilot": "copilot --acp --stdio",
    "cursor": "cursor-agent acp",
    "kiro": "kiro-cli acp",
    "kilocode": "npx -y @kilocode/cli acp",
    "opencode": "npx -y opencode-ai acp",
    "kimi": "kimi acp",
    "qwen": "qwen --acp",
    "droid": "droid exec --output-format acp",
    "iflow": "iflow --experimental-acp",
    "cline": "npx -y cline --acp",
    "amp": "amp --acp",
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
    command = ACP_AGENT_REGISTRY.get(normalized)
    if command:
        logger.debug("ACP agent '%s' resolved from built-in registry", agent_name)
    return command


def split_agent_command(command: str) -> List[str]:
    """Split a command string into argv suitable for ``subprocess.Popen``."""
    return shlex.split(command)


def list_agents() -> List[str]:
    """Return sorted list of all known agent names."""
    return sorted(ACP_AGENT_REGISTRY.keys())
