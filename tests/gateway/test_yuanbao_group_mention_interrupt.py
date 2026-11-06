"""An @bot Yuanbao group message reaches busy-input handling immediately, like a DM.

The group dispatch queue used to serialise every group message behind the active turn, so an
@-mention sent while the agent was busy could never interrupt or steer it.
"""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from plugins.platforms.yuanbao.adapter import DispatchMiddleware

from tests.gateway.test_yuanbao_pipeline import make_adapter, make_ctx


@pytest.mark.asyncio
async def test_addressed_group_message_bypasses_serial_queue():
    adapter = make_adapter()
    adapter._bot_id = "bot_123"
    first_started, second_started, release_first = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def handle_message(event):
        if event.message_id == "first":
            first_started.set()
            await release_first.wait()
        elif event.message_id == "second":
            second_started.set()

    adapter.handle_message = handle_message
    source = adapter.build_source(chat_id="group:grp-1", chat_type="group", user_id="alice", user_name="Alice")
    mention = {"msg_type": "TIMCustomElem",
               "msg_content": {"data": json.dumps({"elem_type": 1002, "text": "@bot", "user_id": "bot_123"})}}

    def addressed(message_id):
        return make_ctx(adapter=adapter, chat_type="group", chat_id="group:grp-1", msg_body=[mention],
                        msg_id=message_id, raw_text=f"message {message_id}", source=source)

    middleware = DispatchMiddleware()
    await middleware(addressed("first"), AsyncMock())
    await asyncio.wait_for(first_started.wait(), timeout=1)
    await middleware(addressed("second"), AsyncMock())
    await asyncio.wait_for(second_started.wait(), timeout=1)

    assert not adapter._group_queues
    release_first.set()
    await asyncio.gather(*list(adapter._inbound_tasks))
