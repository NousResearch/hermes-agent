"""Cancellation cannot safely retry an initial status send with no receipt (#31106)."""
import asyncio

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from tests.gateway.test_todo_checklist_delivery import ChecklistAdapter, make_turn, snapshot


class PendingReceiptAdapter(ChecklistAdapter):
    def __init__(self, accepted):
        super().__init__()
        self.accepted = accepted
        self.send_started = asyncio.Event()
        self.attempts = self.inflight = 0

    async def send(self, **kwargs):
        if "Task List" not in kwargs["content"]:
            return await super().send(**kwargs)
        self.attempts += 1
        # A second attempt would return immediately, exposing the old cancellation-drain retry.
        if self.attempts != 1:
            return await super().send(**kwargs)
        self.inflight += 1
        try:
            if self.accepted:
                await super().send(**kwargs)
            self.send_started.set()
            await asyncio.Event().wait()
        finally:
            self.inflight -= 1


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", [True, False])
async def test_unknown_initial_receipt_is_not_retried_at_turn_end(accepted):
    adapter = PendingReceiptAdapter(accepted)

    async def handler(event):
        turn = make_turn(adapter, event.source)
        sender = asyncio.create_task(turn.send_progress_messages())
        try:
            turn.progress_callback("tool.completed", "todo", result=snapshot("a", "pending"))
            await asyncio.wait_for(adapter.send_started.wait(), 5)
            # Keep the queued terminal update: it must not cause a second send without a receipt.
            turn.progress_callback("tool.completed", "todo", result=snapshot("a", "completed"))
        finally:
            sender.cancel()
            await asyncio.wait_for(sender, 5)
        return "Final reply"

    adapter.set_message_handler(handler)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", thread_id="a")
    event = MessageEvent(text="work", source=source)
    await adapter._process_message_background(event, "session-a")

    assert adapter.attempts == 1
    checklists = [call for call in adapter.sent if "Task List" in call["content"]]
    assert len(checklists) == int(accepted)
    assert adapter.inflight == 0
    assert not adapter.edits
    assert adapter.sent[-1]["content"] == "Final reply"
