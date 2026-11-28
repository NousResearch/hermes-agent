"""Invisible format characters must not hide a slash command.

Copy-paste from an office suite, some IMEs and the WeCom/WeChat clients leaves zero-width
characters (ZWSP/ZWNJ/ZWJ/BOM/word-joiner) inside the pasted token: ``/wxip\u200b``.  The
gateway parses the command name from that token, so the invisible tail made an otherwise
valid command unrecognizable — the message was treated as prose and the command vanished.
"""

import pytest

from gateway.platforms.event import MessageEvent

ZERO_WIDTH = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u2060", "\u180e"]


def _event(text: str) -> MessageEvent:
    return MessageEvent(text=text)


@pytest.mark.parametrize("char", ZERO_WIDTH)
def test_trailing_zero_width_is_not_part_of_the_name(char):
    assert _event(f"/wxip{char}").get_command() == "wxip"


@pytest.mark.parametrize("char", ZERO_WIDTH)
def test_zero_width_inside_the_name_is_dropped(char):
    assert _event(f"/wx{char}ip").get_command() == "wxip"


def test_zero_width_between_command_and_args_keeps_the_args():
    event = _event("/ship_check\u200b 2026-11")
    assert event.get_command() == "ship_check"
    assert event.get_command_args() == "2026-11"


def test_zero_width_in_the_body_is_left_alone():
    """Only the command name is normalized; the user's text may legitimately carry them."""
    event = _event("/note a\u200bb")
    assert event.get_command() == "note"
    assert event.get_command_args() == "a\u200bb"
