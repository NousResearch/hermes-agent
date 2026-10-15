# agents/base.py
"""Base WebSocket client for all chatroom agents."""

import json
import asyncio
import websockets
from typing import Callable, Awaitable, Optional

MessageHandler = Callable[[dict], Awaitable[None]]


class BaseAgent:
    """Base agent that connects to the chatroom server via WebSocket."""

    def __init__(self, name: str, server_url: str = "ws://localhost:8765"):
        self.name = name
        self.server_url = server_url
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._handlers: list[MessageHandler] = []
        self._running = False
        self._reconnect_delay = 3

    def on_message(self, handler: MessageHandler):
        """Register a message handler."""
        self._handlers.append(handler)

    async def connect(self):
        """Connect to the chatroom server with retry."""
        while not self._running:
            try:
                self.ws = await websockets.connect(
                    f"{self.server_url}/ws/{self.name}",
                    ping_interval=30,
                    ping_timeout=10,
                )
                self._running = True
                print(f"[{self.name}] Connected to {self.server_url}")
            except (ConnectionRefusedError, OSError) as e:
                print(f"[{self.name}] Connection failed: {e}. Retrying in {self._reconnect_delay}s...")
                await asyncio.sleep(self._reconnect_delay)

    async def subscribe(self, channel: str):
        """Subscribe to a channel."""
        await self._send({"action": "subscribe", "channel": channel})

    async def publish(self, channel: str, content: str,
                      msg_type: str = "message", metadata: dict = None):
        """Publish a message to a channel."""
        await self._send({
            "action": "publish",
            "channel": channel,
            "content": content,
            "msg_type": msg_type,
            "metadata": metadata or {},
        })

    async def get_history(self, channel: str, limit: int = 50):
        """Get channel message history."""
        await self._send({"action": "history", "channel": channel, "limit": limit})

    async def _send(self, payload: dict):
        if not self.ws:
            raise RuntimeError(f"[{self.name}] Not connected")
        await self.ws.send(json.dumps(payload, ensure_ascii=False))

    async def listen(self):
        """Main listen loop — processes incoming messages."""
        if not self.ws:
            raise RuntimeError(f"[{self.name}] Not connected")
        try:
            async for raw in self.ws:
                message = json.loads(raw)
                for handler in self._handlers:
                    try:
                        await handler(message)
                    except Exception as e:
                        print(f"[{self.name}] Handler error: {e}")
        except websockets.ConnectionClosed:
            print(f"[{self.name}] Connection closed")
            self._running = False

    async def run(self, channels: list[str]):
        """Connect, subscribe to channels, and start listening."""
        await self.connect()
        for ch in channels:
            await self.subscribe(ch)
        await self.listen()

    async def disconnect(self):
        """Close the WebSocket connection."""
        self._running = False
        if self.ws:
            await self.ws.close()
