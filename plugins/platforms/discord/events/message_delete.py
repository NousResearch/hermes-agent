"""Discord ``on_message_delete`` event."""


async def handle(message, adapter) -> None:
    """Normalize a deleted user message into the platform-event boundary."""
    def extra(_message, author):
        return {"author_id": str(getattr(author, "id", "") or "")[:128] or None}

    await adapter._emit_platform_event(
        "message_deleted", lambda: adapter._message_event_parts(message, extra),
    )


def register(client, adapter) -> None:
    @client.event
    async def on_message_delete(message):
        await handle(message, adapter)
