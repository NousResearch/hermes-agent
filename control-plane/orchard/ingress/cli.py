"""CLI ingress — a local test harness that simulates Mattermost users.

Type lines like:
    alice: how do I reset my VPN?
    bob: summarize the Q3 report

Each distinct prefix is treated as a separate sender (=> separate tenant),
so you can exercise routing + isolation without a Mattermost server.
Blank prefix reuses the last sender. Ctrl-D to quit.
"""
from __future__ import annotations

import asyncio
import sys

from .base import Handler, Ingress
from ..models import InboundMessage


class CLIIngress(Ingress):
    def __init__(self, default_sender: str = "tester") -> None:
        self._last_sender = default_sender

    async def run(self, handler: Handler) -> None:
        loop = asyncio.get_event_loop()
        print(f"orchard CLI — chatting as '{self._last_sender}'. Type a message + Enter "
              "(first reply can take ~60s). Ctrl-D to quit.\n", flush=True)
        while True:
            print("› ", end="", flush=True)  # visible prompt so it doesn't look dead
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:  # EOF
                print()
                return
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if ":" in line:
                sender, text = line.split(":", 1)
                sender, text = sender.strip() or self._last_sender, text.strip()
            else:
                sender, text = self._last_sender, line.strip()
            self._last_sender = sender
            msg = InboundMessage(sender_id=sender, channel_id=sender, text=text)
            await handler(msg)

    async def post(self, channel_id: str, text: str, thread_id: str | None = None) -> None:
        print(f"\n🤖 [{channel_id}] {text}\n", flush=True)

    async def typing(self, channel_id: str) -> None:
        print(f"… [{channel_id}] agent is thinking", flush=True)
