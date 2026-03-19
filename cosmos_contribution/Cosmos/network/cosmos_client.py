"""
COSMOS Client — Async client for querying remote bots via the COSMOS mesh.

Simple API for the rest of the codebase:

    from Cosmos.network.cosmos_client import COSMOSClient

    client = COSMOSClient()
    await client.connect()

    # Complete response
    response = await client.query("qwen3-coder-next", "Write a parser")

    # Streaming (fast path)
    async for chunk in client.stream("qwen3-coder-next", "Write a parser"):
        print(chunk, end="", flush=True)
"""
import asyncio
from typing import Optional, AsyncIterator, Dict, List
from loguru import logger

from .cosmos_node import get_cosmos_node, COSMOSNode


class COSMOSClient:
    """
    High-level async client for COSMOS mesh queries.

    Uses the local COSMOS node to route requests to remote bots.
    If no node is running, attempts to connect directly.
    """

    def __init__(self, node: Optional[COSMOSNode] = None):
        self._node = node

    def _get_node(self) -> Optional[COSMOSNode]:
        """Get the COSMOS node (lazy)."""
        if self._node is None:
            self._node = get_cosmos_node()
        return self._node

    async def query(self, bot_name: str, prompt: str,
                    max_tokens: int = 4000,
                    timeout: float = 120.0) -> Optional[str]:
        """
        Query a remote bot and get the complete response.

        Args:
            bot_name: Name of the bot (e.g., "qwen3-coder-next")
            prompt: The prompt to send
            max_tokens: Max response tokens
            timeout: Max wait time

        Returns:
            Response string or None if unavailable
        """
        node = self._get_node()
        if not node:
            logger.warning("No COSMOS node available")
            return None

        # Check if bot is local first (faster)
        if bot_name in node.get_local_bots():
            local_bots = node._local_bots
            if bot_name in local_bots:
                return await local_bots[bot_name](prompt, max_tokens)

        # Route to remote
        return await node.query_remote_bot(bot_name, prompt, max_tokens, timeout)

    async def stream(self, bot_name: str, prompt: str,
                     max_tokens: int = 4000) -> AsyncIterator[str]:
        """
        Stream dialogue from a remote bot. Yields chunks.

        This is the FAST PATH — minimum latency per chunk.

        Usage:
            async for chunk in client.stream("qwen3-coder-next", prompt):
                print(chunk, end="", flush=True)
        """
        node = self._get_node()
        if not node:
            return

        async for chunk in node.stream_remote_bot(bot_name, prompt, max_tokens):
            yield chunk

    def list_bots(self) -> Dict[str, str]:
        """
        List all available bots with their location.

        Returns:
            Dict of bot_name → node_name
        """
        node = self._get_node()
        if not node:
            return {}
        return node.get_all_bots()

    def list_remote_bots(self) -> List[str]:
        """List only remote (non-local) bots."""
        node = self._get_node()
        if not node:
            return []
        local = set(node.get_local_bots())
        all_bots = node.get_all_bots()
        return [name for name in all_bots if name not in local]

    def status(self) -> Dict:
        """Get COSMOS mesh status."""
        node = self._get_node()
        if not node:
            return {"error": "No COSMOS node running"}
        return node.get_status()


# ── Global client singleton ───────────────────────────────────

_cosmos_client: Optional[COSMOSClient] = None


def get_cosmos_client() -> COSMOSClient:
    """Get or create the global COSMOS client."""
    global _cosmos_client
    if _cosmos_client is None:
        _cosmos_client = COSMOSClient()
    return _cosmos_client
