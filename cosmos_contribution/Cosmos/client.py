"""
cosmos Python SDK
        await client.remember("My favorite color is blue")
        memories = await client.recall("What is my favorite color?")

    # Sync usage (convenience wrapper)
    client = cosmosClient()
    client.connect()
    client.remember_sync("My favorite color is blue")
    memories = client.recall_sync("favorite color")
"""

import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass
class Memory:
    """A retrieved memory from Cosmos."""
    content: str
    score: float
    source: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DelegationResult:
    """Result from agent task delegation."""
    success: bool
    output: str
    confidence: float
    agent_used: str
    execution_time: Optional[float] = None


class cosmosClient:
    """
    Client for interacting with cosmos.

    Supports both async and sync operations. For best performance,
    use the async methods directly.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        local_mode: bool = True,
        server_url: str = "http://localhost:8000",
    ):
        """
        Initialize cosmos client.

        Args:
            data_dir: Directory for local data storage
            local_mode: If True, use direct local integration
            server_url: URL for remote server mode (if local_mode=False)
        """
        self.data_dir = Path(data_dir)
        self.local_mode = local_mode
        self.server_url = server_url

        # Lazy-loaded components
        self._memory_system = None
        self._swarm_orchestrator = None
        self._initialized = False
        self._loop = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect_async()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect_async()

    async def connect_async(self):
        """Connect to cosmos system asynchronously."""
        if self._initialized:
            return

        if self.local_mode:
            try:
                from Cosmos.memory.memory_system import MemorySystem
                from Cosmos.agents.swarm_orchestrator import SwarmOrchestrator

                self._memory_system = MemorySystem(data_dir=str(self.data_dir))
                await self._memory_system.initialize()

                self._swarm_orchestrator = SwarmOrchestrator()

                self._initialized = True
                logger.info(f"cosmos client connected (local mode, data_dir={self.data_dir})")

            except Exception as e:
                logger.error(f"Failed to initialize cosmos: {e}")
                raise
        else:
            # Remote mode - would use HTTP/WebSocket
            logger.info(f"cosmos client connected to {self.server_url}")
            self._initialized = True

    async def disconnect_async(self):
        """Disconnect from Cosmos system."""
        if self._memory_system:
            await self._memory_system.shutdown()
        self._initialized = False
        logger.info("cosmos client disconnected")

    def connect(self):
        """Synchronous connect (convenience wrapper)."""
        loop = self._get_or_create_loop()
        loop.run_until_complete(self.connect_async())

    def disconnect(self):
        """Synchronous disconnect."""
        if self._loop:
            self._loop.run_until_complete(self.disconnect_async())

    def _get_or_create_loop(self):
        """Get or create event loop for sync operations."""
        if self._loop is None:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    async def remember(
        self,
        content: str,
        tags: Optional[List[str]] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a memory asynchronously.

        Args:
            content: The content to remember
            tags: Optional tags for categorization
            importance: Importance score 0-1 (default 0.5)
            metadata: Optional additional metadata

        Returns:
            Memory ID
        """
        if not self._initialized:
            await self.connect_async()

        if self.local_mode and self._memory_system:
            memory_id = await self._memory_system.remember(
                content=content,
                tags=tags,
                importance=importance,
                metadata=metadata,
            )
            logger.debug(f"Stored memory: {memory_id}")
            return memory_id
        else:
            # Remote mode
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.server_url}/api/remember",
                    json={
                        "content": content,
                        "tags": tags,
                        "importance": importance,
                        "metadata": metadata,
                    }
                )
                result = response.json()
                return result.get("memory_id", "unknown")

    async def recall(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.3,
    ) -> List[Memory]:
        """
        Recall memories asynchronously.

        Args:
            query: Search query
            limit: Maximum number of results
            min_score: Minimum relevance score

        Returns:
            List of Memory objects
        """
        if not self._initialized:
            await self.connect_async()

        if self.local_mode and self._memory_system:
            results = await self._memory_system.recall(
                query=query,
                top_k=limit,
                min_score=min_score,
            )
            return [
                Memory(
                    content=r.content,
                    score=r.score,
                    source=r.source,
                    metadata=r.metadata,
                )
                for r in results
            ]
        else:
            # Remote mode
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.server_url}/api/recall",
                    json={"query": query, "limit": limit}
                )
                result = response.json()
                return [
                    Memory(
                        content=m["content"],
                        score=m["score"],
                        source=m.get("source", "remote"),
                    )
                    for m in result.get("memories", [])
                ]

    async def delegate(
        self,
        task: str,
        agent_type: str = "auto",
        context: Optional[Dict[str, Any]] = None,
        timeout: float = 120.0,
    ) -> DelegationResult:
        """
        Delegate a task to a specialist agent.

        Args:
            task: Task description
            agent_type: Agent type or "auto" for automatic selection
            context: Optional context for the task
            timeout: Maximum time to wait for result

        Returns:
            DelegationResult with outcome
        """
        if not self._initialized:
            await self.connect_async()

        if self.local_mode and self._swarm_orchestrator:
            try:
                task_id = await self._swarm_orchestrator.submit_task(
                    description=task,
                    context=context,
                )
                result = await self._swarm_orchestrator.wait_for_task(task_id, timeout=timeout)

                return DelegationResult(
                    success=result.success,
                    output=result.output,
                    confidence=result.confidence,
                    agent_used=agent_type,
                    execution_time=result.execution_time,
                )
            except Exception as e:
                logger.error(f"Delegation failed: {e}")
                return DelegationResult(
                    success=False,
                    output=str(e),
                    confidence=0.0,
                    agent_used=agent_type,
                )
        else:
            # Remote mode
            import httpx
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.server_url}/api/delegate",
                    json={"task": task, "agent_type": agent_type, "context": context}
                )
                result = response.json()
                return DelegationResult(
                    success=result.get("success", False),
                    output=result.get("output", ""),
                    confidence=result.get("confidence", 0.0),
                    agent_used=result.get("agent_used", agent_type),
                )

    async def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        if not self._initialized:
            await self.connect_async()

        if self.local_mode and self._memory_system:
            return {
                "connected": True,
                "mode": "local",
                "memory": self._memory_system.get_stats(),
            }
        else:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.server_url}/api/status")
                return response.json()

    # Synchronous convenience wrappers

    def remember_sync(
        self,
        content: str,
        tags: Optional[List[str]] = None,
        importance: float = 0.5,
    ) -> str:
        """Synchronous remember (convenience wrapper)."""
        loop = self._get_or_create_loop()
        return loop.run_until_complete(self.remember(content, tags, importance))

    def recall_sync(self, query: str, limit: int = 5) -> List[Memory]:
        """Synchronous recall (convenience wrapper)."""
        loop = self._get_or_create_loop()
        return loop.run_until_complete(self.recall(query, limit))

    def delegate_sync(
        self,
        task: str,
        agent_type: str = "auto",
    ) -> DelegationResult:
        """Synchronous delegate (convenience wrapper)."""
        loop = self._get_or_create_loop()
        return loop.run_until_complete(self.delegate(task, agent_type))

    def get_status_sync(self) -> Dict[str, Any]:
        """Synchronous status (convenience wrapper)."""
        loop = self._get_or_create_loop()
        return loop.run_until_complete(self.get_status())


# Convenience function for quick usage
def create_client(data_dir: str = "./data") -> cosmosClient:
    """Create and connect a cosmos client."""
    client = cosmosClient(data_dir=data_dir)
    client.connect()
    return client
