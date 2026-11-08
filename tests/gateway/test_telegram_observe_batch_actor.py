"""Telegram observe mode: two members addressing the bot inside one text-batch window are two turns.

The observed shared source is sender-less, so a session-only batch key merged the second member's
text into the first member's event and both requests ran under the first member as the actor.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from tests.gateway.test_telegram_group_gating import _bot_command_entity, _group_message, _make_adapter, _mention_entity


def test_two_members_in_one_batch_window_stay_separate_turns_with_their_own_actor():
    adapter = _make_adapter(require_mention=True, allowed_chats=["-100"], group_allowed_chats=["-100"],
                            observe_unmentioned_group_messages=True)
    adapter._text_batch_delay_seconds = 0.05
    adapter._text_batch_split_delay_seconds = 0.05
    flushed = []
    adapter.handle_message = AsyncMock(side_effect=flushed.append)
    adapter._ensure_forum_commands = AsyncMock()

    async def _run():
        for update_id, (uid, name, body) in enumerate(((111, "Alice", "save my notes"), (222, "Bob", "save mine too"))):
            text = f"@hermes_bot {body}"
            message = _group_message(text, from_user_id=uid, from_user_name=name, entities=[_mention_entity(text)])
            await adapter._handle_text_message(SimpleNamespace(update_id=update_id, message=message, effective_message=None), None)
        for _ in range(50):
            if len(flushed) >= 2:
                break
            await asyncio.sleep(0.05)

    asyncio.run(_run())
    assert sorted((e.user_id, e.text.split("\n")[-1]) for e in flushed) == [
        ("111", "save my notes"), ("222", "save mine too")]
    # Both still route to the ONE shared group session.
    assert {e.source.user_id for e in flushed} == {None}


def test_two_members_long_commands_never_coalesce_into_the_first_sender():
    """Near-limit commands keep source.user_id for slash access but share the group session;
    the batch key must still split per sender or Bob's command runs under Alice's identity."""
    adapter = _make_adapter(require_mention=False, allowed_chats=["-100"], group_allowed_chats=["-100"],
                            observe_unmentioned_group_messages=True)
    adapter._text_batch_delay_seconds = 0.05
    adapter._text_batch_split_delay_seconds = 0.05
    flushed = []
    adapter.handle_message = AsyncMock(side_effect=flushed.append)
    adapter._ensure_forum_commands = AsyncMock()

    async def _run():
        for uid, name in ((111, "Alice"), (222, "Bob")):
            text = ("/queue@hermes_bot " + name * 1500)[:4096]
            message = _group_message(text, from_user_id=uid, from_user_name=name,
                                     entities=[_bot_command_entity(text, "/queue@hermes_bot")])
            await adapter._handle_command(SimpleNamespace(update_id=uid, message=message, effective_message=None), None)
        for _ in range(50):
            if not adapter._pending_text_batch_tasks and len(flushed) >= 2:
                break
            await asyncio.sleep(0.05)

    asyncio.run(_run())
    by_sender = sorted((e.source.user_id, e.user_id, "Alice" in e.text, "Bob" in e.text) for e in flushed)
    assert by_sender == [("111", "111", True, False), ("222", "222", False, True)]
