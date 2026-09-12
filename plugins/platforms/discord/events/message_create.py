"""Discord ``on_message`` event and ingress orchestration."""

import asyncio


async def handle(message, adapter) -> bool:
    """Wait for readiness, apply admission policy, and dispatch the message."""
    if not adapter._ready_event.is_set():
        try:
            await asyncio.wait_for(adapter._ready_event.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            pass
    admitted, role_authorized = adapter._discord_message_admission(message, claim=True)
    if not admitted:
        return False
    return await adapter._handle_message(message, role_authorized=role_authorized)


def register(client, adapter) -> None:
    @client.event
    async def on_message(message):
        await handle(message, adapter)
