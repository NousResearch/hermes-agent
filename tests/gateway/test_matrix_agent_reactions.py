"""Agent reactions keep their annotations separate from lifecycle tapbacks."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plugins.platforms.matrix.adapter import MatrixAdapter
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


ROOM = "!room:matrix.test"
TARGET = "$inbound"


def _adapter() -> MatrixAdapter:
    adapter = object.__new__(MatrixAdapter)
    adapter._reactions_enabled = False
    adapter._pending_reactions = {}
    adapter._agent_reactions = {}
    adapter._send_reaction = AsyncMock(side_effect=["$first", "$second"])
    adapter._redact_reaction = AsyncMock(return_value=True)
    return adapter


@pytest.mark.asyncio
async def test_agent_reactions_retract_their_annotations():
    adapter = _adapter()

    first = await adapter.add_reaction(chat_id=ROOM, message_id=TARGET, emoji="👍")
    second = await adapter.add_reaction(chat_id=ROOM, message_id=TARGET, emoji="❤️")
    removed = await adapter.remove_reaction(chat_id=ROOM, message_id=TARGET)

    assert (first, second, removed) == (
        {"success": True, "message_id": TARGET},
        {"success": True, "message_id": TARGET},
        {"success": True, "message_id": TARGET},
    )
    assert [call.args for call in adapter._send_reaction.await_args_list] == [
        (ROOM, TARGET, "👍"),
        (ROOM, TARGET, "❤️"),
    ]
    assert [call.args[:2] for call in adapter._redact_reaction.await_args_list] == [
        (ROOM, "$first"),
        (ROOM, "$second"),
    ]
    assert adapter._agent_reactions == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "kwargs"),
    [("add_reaction", {"emoji": "❤️"}), ("remove_reaction", {})],
)
async def test_agent_reactions_require_an_explicit_target(method, kwargs):
    adapter = _adapter()
    await adapter.add_reaction(chat_id=ROOM, message_id=TARGET, emoji="👍")
    await adapter.on_processing_start(
        MessageEvent(text="", message_id=TARGET, source=SessionSource(platform=Platform.MATRIX, chat_id=ROOM))
    )

    result = await getattr(adapter, method)(chat_id=ROOM, **kwargs)

    assert (
        result,
        adapter._agent_reactions,
        adapter._send_reaction.await_count,
        adapter._redact_reaction.await_count,
    ) == (
        {"success": False, "error": "message_id is required"},
        {(ROOM, TARGET): ["$first"]},
        1,
        0,
    )


@pytest.mark.asyncio
async def test_failed_reaction_redaction_remains_available_for_retry():
    adapter = _adapter()
    adapter._redact_reaction = AsyncMock(side_effect=[False, True])
    await adapter.add_reaction(chat_id=ROOM, message_id=TARGET, emoji="👍")

    first = await adapter.remove_reaction(chat_id=ROOM, message_id=TARGET)
    retained = deepcopy(adapter._agent_reactions)
    second = await adapter.remove_reaction(chat_id=ROOM, message_id=TARGET)

    assert (first, retained, second, adapter._agent_reactions) == (
        {
            "success": False,
            "message_id": TARGET,
            "error": "reaction redaction failed (see gateway log)",
        },
        {(ROOM, TARGET): ["$first"]},
        {"success": True, "message_id": TARGET},
        {},
    )


@pytest.mark.asyncio
async def test_concurrent_unreact_calls_do_not_redact_the_same_annotation_twice():
    adapter = _adapter()
    await adapter.add_reaction(chat_id=ROOM, message_id=TARGET, emoji="👍")

    redaction_started = asyncio.Event()
    second_reached_or_finished = asyncio.Event()
    finish_redaction = asyncio.Event()
    redaction_calls = 0

    async def redact(*_args):
        nonlocal redaction_calls
        redaction_calls += 1
        redaction_started.set()
        if redaction_calls == 2:
            second_reached_or_finished.set()
        await finish_redaction.wait()
        return True

    adapter._redact_reaction = AsyncMock(side_effect=redact)
    first = asyncio.create_task(adapter.remove_reaction(chat_id=ROOM, message_id=TARGET))
    await asyncio.wait_for(redaction_started.wait(), timeout=2)
    second = asyncio.create_task(adapter.remove_reaction(chat_id=ROOM, message_id=TARGET))
    second.add_done_callback(lambda _: second_reached_or_finished.set())
    try:
        await asyncio.wait_for(second_reached_or_finished.wait(), timeout=2)
    finally:
        finish_redaction.set()

    results = await asyncio.gather(first, second)
    assert (results, redaction_calls, adapter._agent_reactions) == (
        [
            {"success": True, "message_id": TARGET},
            {"success": False, "error": "no reaction of ours recorded on that message"},
        ],
        1,
        {},
    )


@pytest.mark.asyncio
async def test_unreact_result_ignores_a_reaction_added_while_redacting():
    adapter = _adapter()
    await adapter.add_reaction(chat_id=ROOM, message_id=TARGET, emoji="👍")
    redaction_started = asyncio.Event()
    finish_redaction = asyncio.Event()

    async def redact(*_args):
        redaction_started.set()
        await finish_redaction.wait()
        return True

    adapter._redact_reaction = AsyncMock(side_effect=redact)
    adapter._send_reaction = AsyncMock(return_value="$new")
    removing = asyncio.create_task(adapter.remove_reaction(chat_id=ROOM, message_id=TARGET))
    await asyncio.wait_for(redaction_started.wait(), timeout=2)
    await adapter.add_reaction(chat_id=ROOM, message_id=TARGET, emoji="❤️")
    finish_redaction.set()

    assert (await removing, adapter._agent_reactions) == (
        {"success": True, "message_id": TARGET},
        {(ROOM, TARGET): ["$new"]},
    )


@pytest.mark.asyncio
async def test_agent_reaction_records_keep_the_most_recently_used_targets(monkeypatch):
    monkeypatch.setattr(MatrixAdapter, "_AGENT_REACTIONS_MAX", 2)
    adapter = _adapter()
    adapter._send_reaction = AsyncMock(side_effect=["$1", "$2", "$3", "$4"])

    for target in ("$a", "$b", "$a", "$c"):
        await adapter.add_reaction(chat_id=ROOM, message_id=target, emoji="👍")

    assert adapter._agent_reactions == {
        (ROOM, "$a"): ["$1", "$3"],
        (ROOM, "$c"): ["$4"],
    }
