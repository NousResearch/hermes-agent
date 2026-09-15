"""Turn-owned status delivery, including receipts arriving after final cleanup."""

import asyncio
import uuid
from collections import defaultdict
from contextlib import suppress


class StatusDelivery:
    """One turn, one adapter identity; never lend its status key to another turn."""

    def __init__(self, ctx, current_adapter):
        self.ctx = ctx
        self.current_adapter = current_adapter
        self.token = uuid.uuid4().hex
        self.closed = False
        self.cleaned = False
        self.owners = {}
        self.tasks = set()
        self.locks = defaultdict(asyncio.Lock)

    def live(self, adapter):
        return (
            not self.closed
            and self.ctx._run_still_current()
            and self.current_adapter() is adapter
        )

    def track(self, result, adapter):
        if not self.ctx._cleanup_progress or not getattr(result, "success", False):
            return
        mid = getattr(result, "message_id", None)
        if not mid:
            return
        mid = str(mid)
        if self.cleaned:
            task = asyncio.create_task(self.delete(adapter, mid))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
        else:
            self.owners[mid] = adapter
            self.ctx._cleanup_msg_ids.append(mid)

    async def delete(self, adapter, mid):
        # IDs are owned by the transport that returned them, not a replacement bot.
        with suppress(Exception):
            await asyncio.wait_for(
                adapter.delete_message(self.ctx.source.chat_id, mid), 5
            )

    async def send(self, adapter, chat_id, event_type, content, metadata):
        from gateway.run import _send_or_update_status_coro

        async with self.locks[event_type]:
            if not self.live(adapter):
                return
            # Recheck inside the lock: a queued callback may outlive this turn.
            result = await _send_or_update_status_coro(
                adapter,
                chat_id,
                f"{self.token}:{event_type}",
                content,
                dict(metadata) if metadata else None,
            )
            self.track(result, adapter)
            return result
