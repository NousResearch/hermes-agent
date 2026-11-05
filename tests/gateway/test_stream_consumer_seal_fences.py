"""Sealing an overflowing preview mid code block keeps code rendered as code.

When deltas push a live preview past the platform limit, ``_seal_overflow_heads``
edits the preview down to a head and continues in a new message. The head goes out
fence-closed; the continuation must reopen the block (language tag included), or it
starts with bare code lines and its own closing fence becomes an opener that wraps
the prose after the code in a code block.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

LIMIT = 2000
CODE_LINES = [f"value_{i:03d} = compute({i})  # long code line" for i in range(150)]
PROSE = "After the code block, this sentence is prose, not code."


def _make_plain_adapter():
    from gateway.platforms.base import BasePlatformAdapter

    cls = type(
        "PlainAdapter",
        (BasePlatformAdapter,),
        {
            "MAX_MESSAGE_LENGTH": LIMIT,
            "preferred_final_chunks": lambda self, text, budget: None,
            "preferred_final_split_index": lambda self, text, budget: None,
        },
    )
    cls.__abstractmethods__ = frozenset()
    adapter = cls.__new__(cls)
    adapter._typing_paused = set()
    adapter._fatal_error_message = None
    return adapter


def _render(message: str) -> "tuple[dict[str, bool], bool]":
    """Map each line to whether a Markdown renderer shows it inside a code block, and
    whether the message ends inside one."""
    in_code, where = False, {}
    for line in message.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
        else:
            where[line] = in_code
    return where, in_code


@pytest.mark.asyncio
async def test_every_message_renders_code_as_code_and_prose_as_prose():
    adapter = _make_plain_adapter()
    screen: "dict[str, str]" = {}

    async def fake_send(**kw):
        message_id = f"m{len(screen) + 1}"
        screen[message_id] = kw.get("content", "")
        return SimpleNamespace(success=True, message_id=message_id)

    async def fake_edit(**kw):
        screen[kw["message_id"]] = kw.get("content", "")
        return SimpleNamespace(success=True, message_id=kw["message_id"])

    adapter.send = AsyncMock(side_effect=fake_send)
    adapter.edit_message = AsyncMock(side_effect=fake_edit)
    consumer = GatewayStreamConsumer(
        adapter, "chat_plain", StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor=""))
    consumer.on_delta("Here is the implementation:\n\n")
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.06)  # the preview message exists before the code arrives
    consumer.on_delta("```python\n" + "\n".join(CODE_LINES) + f"\n```\n\n{PROSE}\n")
    await asyncio.sleep(0.12)
    consumer.finish()
    await asyncio.wait_for(task, timeout=10)

    assert len(screen) >= 3, "the harness never sealed the preview mid code block"
    seen: "dict[str, bool]" = {}
    for message_id, message in screen.items():
        where, ends_in_code = _render(message)
        assert not ends_in_code, f"{message_id} leaves a code block open"
        seen.update(where)
    assert [line for line in CODE_LINES if seen.get(line) is not True] == []
    assert seen.get(PROSE) is False
    # Continuations reopen with the block's language tag, so highlighting survives the split.
    assert all(m.startswith("```python\n") for m in list(screen.values())[1:])
