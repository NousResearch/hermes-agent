"""
COSMOS Bridge — Connects COSMOS mesh to existing Cosmos systems.

Bridges:
  - PersistentAgent shadow agents → remote bots via COSMOS
  - DialogueBus → COSMOS mesh (remote agents appear in bus)
  - AgentSpawner → COSMOS-aware fallback chains
"""
import asyncio
from typing import Optional, Dict, List
from loguru import logger

from .cosmos_client import get_cosmos_client, COSMOSClient
from .cosmos_node import get_cosmos_node


class COSMOSRemoteProvider:
    """
    Provider that routes queries through the COSMOS mesh to a remote bot.

    Drop-in replacement for OllamaWithToolsProvider / API providers.
    Used by PersistentAgent for remote bot access.
    """

    def __init__(self, bot_name: str):
        self.bot_name = bot_name
        self.system_prompt = None
        self._client = get_cosmos_client()

    async def chat(self, prompt: str, max_tokens: int = 4000) -> Dict:
        """Query the remote bot via COSMOS mesh."""
        # Prepend system prompt if set
        full_prompt = prompt
        if self.system_prompt:
            full_prompt = f"{self.system_prompt}\n\n{prompt}"

        response = await self._client.query(self.bot_name, full_prompt, max_tokens)
        return {"content": response or ""}

    async def chat_stream(self, prompt: str, max_tokens: int = 4000):
        """Stream from the remote bot. Yields content chunks."""
        full_prompt = prompt
        if self.system_prompt:
            full_prompt = f"{self.system_prompt}\n\n{prompt}"

        async for chunk in self._client.stream(self.bot_name, full_prompt, max_tokens):
            yield chunk


def register_cosmos_bots_as_shadow_agents():
    """
    Register all remote COSMOS bots as shadow agents in the PersistentAgent system.

    This makes remote bots callable via:
        from Cosmos.core.collective.persistent_agent import call_shadow_agent
        result = await call_shadow_agent("qwen3-coder-next", "Write a parser")
    """
    try:
        from Cosmos.core.collective.persistent_agent import (
            AGENT_CONFIGS, PersistentAgent, _SHADOW_AGENTS, _SHADOW_LOCK,
        )

        client = get_cosmos_client()
        remote_bots = client.list_remote_bots()

        for bot_name in remote_bots:
            safe_name = bot_name.replace("-", "_").replace(":", "_").replace(".", "_")

            if safe_name not in AGENT_CONFIGS:
                AGENT_CONFIGS[safe_name] = {
                    "provider": "cosmos_remote",
                    "personality": f"Remote bot via COSMOS mesh ({bot_name})",
                    "thinking_interval": 60,
                    "specialties": ["remote", "cosmos"],
                    "cosmos_bot_name": bot_name,
                }
                logger.info(f"Registered COSMOS remote bot as shadow agent: {safe_name} → {bot_name}")

    except Exception as e:
        logger.warning(f"Could not register COSMOS bots as shadow agents: {e}")


def register_cosmos_bots_with_spawner():
    """
    Register remote COSMOS bots with the AgentSpawner system.
    """
    try:
        from Cosmos.core.agent_spawner import get_spawner, TaskType

        spawner = get_spawner()
        client = get_cosmos_client()
        remote_bots = client.list_remote_bots()

        for bot_name in remote_bots:
            display_name = bot_name.replace("-", " ").replace("_", " ").title().replace(" ", "")

            if display_name not in spawner.agent_capabilities:
                spawner.agent_capabilities[display_name] = [
                    TaskType.CHAT, TaskType.DEVELOPMENT, TaskType.RESEARCH,
                ]
                spawner.max_instances[display_name] = 2
                # Add to fallback chains
                spawner.fallback_chains[display_name] = ["DeepSeek", "ClaudeOpus"]
                logger.info(f"Registered COSMOS bot with spawner: {display_name}")

    except Exception as e:
        logger.warning(f"Could not register COSMOS bots with spawner: {e}")


async def start_cosmos_bridge():
    """
    Initialize the COSMOS bridge.

    Call this after the COSMOS node is started and connected to peers.
    Registers remote bots with all Cosmos systems.
    """
    # Wait a moment for peer discovery
    await asyncio.sleep(5)

    register_cosmos_bots_as_shadow_agents()
    register_cosmos_bots_with_spawner()

    # Register local shadow agents as COSMOS bots (reverse bridge)
    node = get_cosmos_node()
    if node:
        try:
            from Cosmos.core.collective.persistent_agent import (
                call_shadow_agent, get_shadow_agents,
            )
            for agent_id in get_shadow_agents():
                async def make_fn(aid=agent_id):
                    async def query_fn(prompt: str, max_tokens: int = 4000) -> str:
                        result = await call_shadow_agent(aid, prompt, max_tokens)
                        return result[1] if result else ""
                    return query_fn

                node.register_bot(agent_id, await make_fn())
                logger.debug(f"Registered shadow agent '{agent_id}' as COSMOS bot")

        except Exception as e:
            logger.debug(f"Could not register shadow agents as COSMOS bots: {e}")

    logger.info("COSMOS bridge initialized")
