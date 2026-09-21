"""In-process Agent-to-Agent message bus (async queue)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class AgentMessage:
    """Single A2A envelope (tool-call shaped payloads allowed)."""

    kind: str
    from_agent: str
    to_agent: str
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None


class AgentCommunicationBus:
    """Lightweight pub/sub bus for sandboxed subagents in one process."""

    def __init__(self) -> None:
        self._queues: Dict[str, asyncio.Queue[AgentMessage]] = {}
        self._lock = asyncio.Lock()

    async def ensure_peer(self, agent_id: str) -> None:
        async with self._lock:
            if agent_id not in self._queues:
                self._queues[agent_id] = asyncio.Queue()

    async def publish(self, msg: AgentMessage) -> None:
        await self.ensure_peer(msg.to_agent)
        await self._queues[msg.to_agent].put(msg)

    async def drain_for(
        self,
        agent_id: str,
        *,
        max_messages: int = 32,
    ) -> list[AgentMessage]:
        await self.ensure_peer(agent_id)
        q = self._queues[agent_id]
        out: list[AgentMessage] = []
        for _ in range(max_messages):
            try:
                out.append(q.get_nowait())
            except asyncio.QueueEmpty:
                break
        return out

    def broadcast_tool_hint(
        self,
        *,
        from_agent: str,
        tool_name: str,
        args_preview: str,
    ) -> AgentMessage:
        return AgentMessage(
            kind="tool.hint",
            from_agent=from_agent,
            to_agent="*",
            payload={"tool": tool_name, "preview": args_preview},
        )
