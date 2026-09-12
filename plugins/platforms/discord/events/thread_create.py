"""Discord ``on_thread_create`` event."""


async def handle(thread, adapter) -> None:
    """Normalize a newly created thread into the platform-event boundary."""
    def extra(value, owner_id):
        name = getattr(value, "name", None)
        return {
            "name": name[:256] if isinstance(name, str) else None,
            "owner_id": str(owner_id)[:128] if owner_id is not None else None,
        }

    await adapter._emit_platform_event(
        "thread_created", lambda: adapter._thread_event_parts(thread, extra),
    )


def register(client, adapter) -> None:
    @client.event
    async def on_thread_create(thread):
        await handle(thread, adapter)
