"""Additive session updates through the pinned SDK's public transport hooks."""

import asyncio
from acp.connection import StreamDirection
from acp.task import MessageSender


class SessionUpdateTransport:
    """Connection-owned capability snapshot and the SDK's ordered sender."""

    def __init__(self):
        self.capabilities = {}
        self.loop = None
        self.sender = None

    def connection_options(self):
        return {"observers": [self.observe], "sender_factory": self.sender_factory}

    def observe(self, event):
        if event.direction == StreamDirection.INCOMING and event.message.get("method") == "initialize":
            params = event.message.get("params")
            capabilities = params.get("clientCapabilities") if isinstance(params, dict) else None
            self.capabilities = capabilities if isinstance(capabilities, dict) else {}

    def sender_factory(self, writer, supervisor):
        self.loop = asyncio.get_running_loop()
        self.sender = MessageSender(writer, supervisor)
        return self.sender

    def supports_session(self, feature):
        session = self.capabilities.get("session")
        return isinstance(session, dict) and isinstance(session.get(feature), dict)

    async def send(self, session_id, update):
        await self.sender.send({"jsonrpc": "2.0", "method": "session/update",
                                "params": {"sessionId": session_id, "update": update}})
