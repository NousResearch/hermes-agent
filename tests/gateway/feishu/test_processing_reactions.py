"""Processing reaction state machine contract tests.

Covers:
  start → success  → typing reaction removed
  start → failure  → typing removed → CrossMark added
  start → cancelled → typing removed silently

Production constants (from ``gateway/platforms/feishu/adapter.py``):
  _FEISHU_REACTION_IN_PROGRESS = "Typing"
  _FEISHU_REACTION_FAILURE     = "CrossMark"
"""

import asyncio

import pytest

pytest.importorskip("lark_oapi.channel")


@pytest.fixture
def harness_with_reactions(adapter_harness):
    """Stub _add_reaction / _remove_reaction to record calls."""
    add_calls = []
    remove_calls = []

    async def fake_add(message_id, emoji_type):
        add_calls.append((message_id, emoji_type))
        return f"react_{len(add_calls)}"

    async def fake_remove(message_id, reaction_id):
        remove_calls.append((message_id, reaction_id))
        return True

    adapter_harness.adapter._add_reaction = fake_add  # type: ignore
    adapter_harness.adapter._remove_reaction = fake_remove  # type: ignore
    adapter_harness.adapter._add_calls = add_calls  # for test access
    adapter_harness.adapter._remove_calls = remove_calls
    return adapter_harness


def _build_event_with_message_id(message_id: str):
    from gateway.platforms.base import MessageEvent, MessageType, SessionSource
    from gateway.config import Platform
    return MessageEvent(
        text="hi", message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.FEISHU, chat_id="oc_test",
                             chat_name="t", chat_type="p2p", user_id="ou_alice"),
        message_id=message_id,
    )


def test_start_then_success_removes_typing(harness_with_reactions):
    from gateway.platforms.base import ProcessingOutcome
    event = _build_event_with_message_id("om_react_1")
    asyncio.run(harness_with_reactions.adapter.on_processing_start(event))
    assert ("om_react_1", "Typing") in harness_with_reactions.adapter._add_calls
    asyncio.run(harness_with_reactions.adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS))
    # Typing reaction id must be removed; no CrossMark added on SUCCESS
    assert len(harness_with_reactions.adapter._remove_calls) == 1
    add_emojis = [emoji for _, emoji in harness_with_reactions.adapter._add_calls]
    assert add_emojis == ["Typing"], (
        f"SUCCESS outcome must not add a follow-up reaction; got {add_emojis}"
    )


def test_start_then_failure_adds_cross_mark(harness_with_reactions):
    from gateway.platforms.base import ProcessingOutcome
    event = _build_event_with_message_id("om_react_2")
    asyncio.run(harness_with_reactions.adapter.on_processing_start(event))
    asyncio.run(harness_with_reactions.adapter.on_processing_complete(event, ProcessingOutcome.FAILURE))
    add_emojis = [emoji for _, emoji in harness_with_reactions.adapter._add_calls]
    assert "Typing" in add_emojis
    # Production constant is the literal string "CrossMark" (Lark identifier).
    assert any(emoji.lower() in ("crossmark", "cross_mark", "x") for emoji in add_emojis), (
        f"Failure outcome must add a cross-mark reaction; got {add_emojis}"
    )


def test_start_then_cancelled_silently_removes(harness_with_reactions):
    from gateway.platforms.base import ProcessingOutcome
    event = _build_event_with_message_id("om_react_3")
    asyncio.run(harness_with_reactions.adapter.on_processing_start(event))
    asyncio.run(harness_with_reactions.adapter.on_processing_complete(event, ProcessingOutcome.CANCELLED))
    # Typing removed, but no extra reaction added
    assert len(harness_with_reactions.adapter._remove_calls) == 1
    add_emojis = [emoji for _, emoji in harness_with_reactions.adapter._add_calls]
    assert add_emojis == ["Typing"], f"Cancelled outcome must not add CrossMark; got {add_emojis}"
