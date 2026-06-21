"""Registry of external agent lanes Hermes can reason about."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AgentKind(Enum):
    """How an external agent is discovered and launched."""

    BINARY = "binary"
    SHELL_FUNCTION = "shell_function"


@dataclass(frozen=True)
class AgentSpec:
    """Static metadata for a candidate external-agent lane."""

    name: str
    kind: AgentKind
    version_argv: tuple[str, ...] | None = None
    secrets: bool = False
    sandbox: bool = False
    external_isolation: bool = False


KNOWN_AGENTS: tuple[AgentSpec, ...] = (
    AgentSpec("ccd", AgentKind.SHELL_FUNCTION),
    AgentSpec("codex", AgentKind.BINARY, ("codex", "--version"), sandbox=True, external_isolation=True),
    AgentSpec("ccg", AgentKind.SHELL_FUNCTION),
    AgentSpec("ccm", AgentKind.SHELL_FUNCTION, secrets=True),
)


def get_agent_spec(name: str) -> AgentSpec | None:
    """Return the registry entry for ``name`` if known."""

    normalized = (name or "").strip().lower()
    for spec in KNOWN_AGENTS:
        if spec.name == normalized:
            return spec
    return None
