"""Regression coverage for empty tool-progress bubbles."""

from queue import Queue
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import SendResult
from gateway.run_turn_progress import GatewayTurnProgressMixin


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def progress():
    adapter = SimpleNamespace(
        name="Discord",
        send=AsyncMock(return_value=SendResult(success=True, message_id="7")),
        edit_message=AsyncMock(),
        delete_message=AsyncMock(return_value=True),
    )
    loop = GatewayTurnProgressMixin()
    loop._ctx = SimpleNamespace(
        source=SimpleNamespace(chat_id="123"),
        progress_queue=Queue(),
        progress_grouping="edit",
        _progress_metadata={},
        _progress_reply_to=None,
        _cleanup_progress=False,
    )
    return loop, loop._progress_edit_state(adapter)


@pytest.mark.anyio
async def test_dedup_can_start_a_fresh_progress_bubble(progress):
    loop, st = progress
    base_line = "Running vision_analyze"
    msg = loop._progress_absorb(st, ("__dedup__", base_line, 1))

    assert await loop._progress_send_or_edit(st, msg)
    st.adapter.send.assert_awaited_once_with(
        chat_id="123", content=base_line, reply_to=None, metadata={}
    )
    assert st.progress_lines == [base_line]
    assert st.progress_msg_id == "7"


@pytest.mark.anyio
@pytest.mark.parametrize("lines", [[], ["", "  "]])
async def test_blank_first_send_is_suppressed(progress, lines):
    loop, st = progress
    st.progress_lines.extend(lines)

    assert not await loop._progress_send_or_edit(st, "Running vision_analyze")
    st.adapter.send.assert_not_awaited()
    assert st.progress_msg_id is None


@pytest.mark.anyio
@pytest.mark.parametrize("lines", [[], ["", "  "]])
async def test_cancel_deletes_blank_progress_bubble(progress, lines):
    loop, st = progress
    st.progress_msg_id = "7"
    st.progress_lines.extend(lines)

    await loop._drain_progress_on_cancel(st)

    st.adapter.delete_message.assert_awaited_once_with("123", "7")
    st.adapter.edit_message.assert_not_awaited()
