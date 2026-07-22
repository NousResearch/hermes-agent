"""Build the Hermes A2A Agent Card.

The Agent Card is the discovery document an A2A client fetches from
``/.well-known/agent-card.json`` before sending any message. It mirrors the
intent of ``acp_registry/agent.json`` but follows the A2A schema.
"""

from __future__ import annotations

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentProvider,
    AgentSkill,
    TransportProtocol,
)

_DESCRIPTION = (
    "Self-improving open-source AI agent by Nous Research, with persistent "
    "memory, skills, and rich tool support (shell, filesystem, web search, "
    "code editing). Exposed over the Agent2Agent protocol so other agents can "
    "delegate coding and reasoning tasks to it."
)


def _hermes_version() -> str:
    """Best-effort Hermes version; falls back to a sentinel when the package
    metadata isn't importable (e.g. running the adapter in isolation)."""
    try:
        from hermes_cli import __version__

        return str(__version__)
    except Exception:
        return "0.0.0+local"


def build_skills() -> list[AgentSkill]:
    """The skills advertised on the card. Coarse-grained on purpose — Hermes is
    a general agent, not a fixed-function service."""
    return [
        AgentSkill(
            id="general-agent",
            name="General coding & reasoning agent",
            description=(
                "Plans and executes multi-step tasks: writes and edits code, "
                "runs shell commands, inspects and modifies files, and reasons "
                "over the results to reach a goal."
            ),
            tags=["coding", "shell", "filesystem", "reasoning", "autonomous"],
            examples=[
                "Refactor the auth module to use async and add tests.",
                "Find why the build is failing and fix it.",
                "Summarize what this repository does and how it's structured.",
            ],
        ),
        AgentSkill(
            id="research",
            name="Web research & synthesis",
            description=(
                "Searches the web, reads sources, and synthesizes a concise, "
                "cited answer to a question."
            ),
            tags=["research", "web-search", "summarization"],
            examples=[
                "What changed in the latest release of <library>?",
                "Compare these three approaches and recommend one.",
            ],
        ),
    ]


def build_agent_card(
    url: str,
    *,
    version: str | None = None,
    streaming: bool = True,
) -> AgentCard:
    """Construct the Hermes Agent Card.

    Args:
        url: The externally reachable base URL of this A2A service (the
             JSON-RPC endpoint is served at the root of it).
        version: Override the advertised version; defaults to the Hermes version.
        streaming: Whether to advertise SSE streaming support.
    """
    return AgentCard(
        name="Hermes Agent",
        description=_DESCRIPTION,
        url=url,
        version=version or _hermes_version(),
        protocol_version="0.3.0",
        preferred_transport=TransportProtocol.jsonrpc,
        provider=AgentProvider(
            organization="Nous Research",
            url="https://github.com/NousResearch/hermes-agent",
        ),
        capabilities=AgentCapabilities(
            streaming=streaming,
            push_notifications=False,
            state_transition_history=False,
        ),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=build_skills(),
    )
