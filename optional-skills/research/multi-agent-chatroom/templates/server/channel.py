# server/channel.py
"""Channel-based pub/sub message manager."""

import asyncio
from typing import Callable, Awaitable, Dict, List

Handler = Callable[[dict], Awaitable[None]]


class ChannelManager:
    """Manages named channels with pub/sub semantics."""

    def __init__(self):
        self._subscribers: Dict[str, List[Handler]] = {}
        self._messages: Dict[str, List[dict]] = {}

    async def create_channel(self, name: str):
        if name not in self._subscribers:
            self._subscribers[name] = []
            self._messages[name] = []

    async def subscribe(self, channel: str, handler: Handler):
        if channel not in self._subscribers:
            await self.create_channel(channel)
        self._subscribers[channel].append(handler)

    async def unsubscribe(self, channel: str, handler: Handler):
        if channel in self._subscribers and handler in self._subscribers[channel]:
            self._subscribers[channel].remove(handler)
            if not self._subscribers[channel]:
                del self._subscribers[channel]

    async def publish(self, channel: str, message: dict):
        if channel not in self._subscribers:
            await self.create_channel(channel)
        self._messages[channel].append(message)
        # Notify all subscribers concurrently
        tasks = [handler(message) for handler in self._subscribers[channel]]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def get_history(self, channel: str, limit: int = 50) -> List[dict]:
        if channel not in self._messages:
            return []
        return self._messages[channel][-limit:]

    async def drain(self):
        """Let pending async handlers complete."""
        await asyncio.sleep(0.1)
