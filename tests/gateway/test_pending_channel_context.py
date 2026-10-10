"""Merged pending turns retain each contribution's untrusted channel context."""

from dataclasses import replace

import pytest

from gateway.config import Platform
from gateway.platforms.base_pending import merge_recorded, withdraw_from_event
from gateway.platforms.base_pending_merge import (
    _absorb_pending_media, _absorb_pending_text,
    _append_batched_text, _append_debounced_text,
)
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource


@pytest.mark.parametrize("merge", [
    _absorb_pending_media, _absorb_pending_text, _append_batched_text, _append_debounced_text,
])
def test_recorded_merges_preserve_and_withdraw_channel_context(merge):
    source = SessionSource(platform=Platform.MATRIX, chat_id="room", user_id="sender")
    kind = MessageType.PHOTO if merge is _absorb_pending_media else MessageType.TEXT
    first = MessageEvent(text="one", source=source, message_id="one", message_type=kind,
                         channel_context="untrusted history one", reply_to_message_id="parent",
                         reply_to_text="literal @file:private", reply_to_author_id="other",
                         reply_to_author_name="Other", reply_to_is_own_message=False,
                         reply_to_author_authorized=False)
    second = replace(first, text="two", message_id="two", channel_context="untrusted history two")
    expected = replace(first, text="one\n\ntwo" if kind is MessageType.PHOTO else "one\ntwo",
                       channel_context="untrusted history one\n\nuntrusted history two",
                       merged_message_ids=["two"])
    if merge is _append_debounced_text:
        expected = replace(expected, message_id="two", merged_message_ids=["one"])
    merge_recorded(first, second, merge)
    assert first == expected
    matched, remaining = withdraw_from_event(first, lambda part: part.message_id == "one")
    assert (matched, remaining) == (True, second)
