"""Discord ``on_thread_update`` event."""


async def handle(before, after, adapter) -> None:
    """Normalize thread renames and drop unrelated updates."""
    def build():
        old_name = getattr(before, "name", None)
        new_name = getattr(after, "name", None)
        if old_name == new_name or not isinstance(new_name, str):
            return None
        return adapter._thread_event_parts(after, lambda _thread, _owner: {
            "old_name": old_name[:256] if isinstance(old_name, str) else None,
            "new_name": new_name[:256],
        })

    await adapter._emit_platform_event("thread_renamed", build)


def register(client, adapter) -> None:
    @client.event
    async def on_thread_update(before, after):
        await handle(before, after, adapter)
