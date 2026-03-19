"""
CLI Swarm Provider — ExternalProvider wrapping the CLI bridge router.

"Free compute, full citizen. Every CLI speaks to the swarm." - The Collective

Bridges the CLI capability router into Cosmos's provider system:
- Inherits circuit breaker + fault tolerance from ExternalProvider
- Routes queries through CLICapabilityRouter
- Provides chat(), search(), code_task() interfaces
- Plugs into persistent_agent.py and agent_registry.py

This provider is the glue between the CLI bridge layer and the rest
of the Cosmos swarm infrastructure.
"""

import asyncio
from typing import Any, Dict, List, Optional

from loguru import logger

from .base import ExternalProvider, IntegrationConfig, ConnectionStatus
from Cosmos.integration.cli_bridge.base import CLICapability
from Cosmos.integration.cli_bridge.capability_router import CLICapabilityRouter, get_cli_router


class CLISwarmProvider(ExternalProvider):
    """
    ExternalProvider that routes queries through CLI bridges.

    Supports:
    - chat() for general queries (routed by capability)
    - search() for web search (forces WEB_SEARCH → Gemini)
    - code_task() for code editing (forces CODE_EDIT → Claude Code)
    - execute_action() for the standard provider interface
    - swarm_respond() for swarm chat participation
    """

    def __init__(self, preferred_cli: Optional[str] = None):
        super().__init__(IntegrationConfig(
            name=f"cli_bridge_{preferred_cli or 'auto'}",
            circuit_breaker_enabled=True,
            failure_threshold=5,
            timeout_seconds=30.0,
        ))
        self.preferred_cli = preferred_cli
        self._router: Optional[CLICapabilityRouter] = None

    async def _get_router(self) -> CLICapabilityRouter:
        """Lazy-init the router."""
        if self._router is None:
            self._router = await get_cli_router()
        return self._router

    async def connect(self) -> bool:
        """Health-check all registered CLIs."""
        try:
            router = await self._get_router()
            bridges = router.get_available_bridges()
            available = [b for b in bridges if b.get("available")]

            if available:
                self.status = ConnectionStatus.CONNECTED
                logger.info(
                    f"CLISwarmProvider connected: "
                    f"{len(available)}/{len(bridges)} bridges available"
                )
                return True
            else:
                self.status = ConnectionStatus.ERROR
                logger.warning("CLISwarmProvider: No CLI bridges available")
                return False

        except Exception as e:
            self.status = ConnectionStatus.ERROR
            logger.error(f"CLISwarmProvider connection failed: {e}")
            return False

    async def sync(self):
        """CLI bridges don't need polling."""
        pass

    async def execute_action(self, action: str, params: Dict[str, Any]) -> Any:
        """Execute an action through the CLI router."""
        if action == "chat":
            return await self.chat(
                prompt=params.get("prompt", ""),
                system=params.get("system"),
                context=params.get("context"),
                model=params.get("model"),
                temperature=params.get("temperature"),
                max_tokens=params.get("max_tokens"),
            )
        elif action == "search":
            return await self.search(params.get("query", ""))
        elif action == "code":
            return await self.code_task(
                task=params.get("task", ""),
                context=params.get("context"),
            )
        else:
            return await self.chat(prompt=params.get("prompt", str(params)))

    async def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        context: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Chat through the CLI router.

        Matches the interface used by other providers (Grok, Gemini, etc.)
        so it can be used as a drop-in replacement.
        """
        router = await self._get_router()

        # Build full prompt with context
        full_prompt = ""
        if context:
            full_prompt = f"Context:\n{context}\n\n"
        full_prompt += prompt

        response = await router.query_with_fallback(
            prompt=full_prompt,
            system_prompt=system,
            model=model,
            max_tokens=max_tokens,
            preferred_cli=self.preferred_cli,
        )

        if response.success:
            return {
                "content": response.content,
                "model": response.model,
                "cli_name": response.cli_name,
                "cost_usd": response.cost_usd,
                "latency_ms": response.latency_ms,
                "session_id": response.session_id,
            }
        else:
            return {
                "content": "",
                "error": response.error,
                "cli_name": response.cli_name,
            }

    async def search(self, query: str) -> Dict[str, Any]:
        """
        Search using a CLI with WEB_SEARCH capability.

        Routes to Gemini CLI (free Google Search grounding).
        """
        router = await self._get_router()

        response = await router.query_with_fallback(
            prompt=f"Search and summarize: {query}",
            system_prompt="You have access to real-time web search. Provide current, accurate information with sources.",
            required_capabilities={CLICapability.WEB_SEARCH},
        )

        return {
            "content": response.content if response.success else "",
            "error": response.error if not response.success else None,
            "cli_name": response.cli_name,
        }

    async def code_task(
        self,
        task: str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a code task using a CLI with CODE_EDIT capability.

        Routes to Claude Code CLI.
        """
        router = await self._get_router()

        prompt = task
        if context:
            prompt = f"Context:\n{context}\n\nTask:\n{task}"

        response = await router.query_with_fallback(
            prompt=prompt,
            system_prompt="You are a code assistant. Provide precise, working code edits.",
            required_capabilities={CLICapability.CODE_EDIT},
        )

        return {
            "content": response.content if response.success else "",
            "error": response.error if not response.success else None,
            "cli_name": response.cli_name,
            "session_id": response.session_id,
        }

    async def swarm_respond(
        self,
        other_bots: List[str],
        last_speaker: str,
        last_content: str,
        chat_history: List[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Generate a swarm chat response via CLI bridges.

        Used when this provider participates in swarm conversations.
        """
        # Build context from recent history
        history_context = ""
        if chat_history:
            recent = chat_history[-5:]
            lines = []
            for msg in recent:
                name = msg.get("bot_name") or msg.get("user_name", "Unknown")
                content = msg.get("content", "")[:200]
                lines.append(f"{name}: {content}")
            history_context = "\n".join(lines)

        agent_label = self.preferred_cli or "CLI Agent"
        system = f"""You are {agent_label} in the Cosmos AI swarm chat with {', '.join(other_bots)}.
Be concise (1-3 sentences), natural, no roleplay actions."""

        prompt = f"""Recent conversation:
{history_context}

{last_speaker} just said: "{last_content[:300]}"

Respond naturally."""

        return await self.chat(prompt=prompt, system=system, max_tokens=200)


# =============================================================================
# FACTORY
# =============================================================================

_cli_providers: Dict[str, CLISwarmProvider] = {}


def get_cli_swarm_provider(preferred_cli: Optional[str] = None) -> CLISwarmProvider:
    """Get or create a CLI swarm provider, optionally pinned to a specific CLI."""
    key = preferred_cli or "auto"
    if key not in _cli_providers:
        _cli_providers[key] = CLISwarmProvider(preferred_cli=preferred_cli)
    return _cli_providers[key]
