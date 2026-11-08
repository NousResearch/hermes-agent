"""Telegram observe mode: two members addressing the bot inside one text-batch window are two turns.

The observed shared source is sender-less, so a session-only batch key merged the second member's
text into the first member's event and both requests ran under the first member as the actor.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from tests.gateway.test_telegram_group_gating import _group_message, _make_adapter, _mention_entity


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
