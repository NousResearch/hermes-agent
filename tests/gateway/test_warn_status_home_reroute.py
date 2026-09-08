"""Gateway warn-status rerouting: internal diagnostics must not leak into
shared group chats when a home channel is configured.

Regression for a guest/staff-facing profile whose compression-overflow
warning (CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE) was delivered into the
WhatsApp group that triggered the turn, exposing framework internals
(token counts, /new, /compress) to every participant.
"""

import pytest

from gateway.run import _warn_status_reroute_target


class _FakeHomeChannel:
    def __init__(self, chat_id):
        self.chat_id = chat_id


@pytest.mark.parametrize(
    "event_type,chat_type,current,home,expected",
    [
        # warn in a group with a distinct home channel -> reroute to home
        ("warn", "group", "120363421112864421@g.us", "140905512218873@lid", "140905512218873@lid"),
        ("warn", "channel", "channel-1", "home-1", "home-1"),
        # warn in a DM / 1:1 thread -> operator is the only participant, keep in place
        ("warn", "dm", "140905512218873@lid", "140905512218873@lid", None),
        ("warn", "private", "user-1", "home-1", None),
        # non-warn statuses are never rerouted (lifecycle/progress stay in chat)
        ("lifecycle", "group", "grp-1", "home-1", None),
        ("progress", "group", "grp-1", "home-1", None),
        # warn in a group but no home channel configured -> keep in place
        ("warn", "group", "grp-1", None, None),
        # warn in a group whose home channel IS the group -> keep in place
        ("warn", "group", "home-1", "home-1", None),
        # warn in an unclassified chat -> keep in place
        ("warn", None, "grp-1", "home-1", None),
    ],
)
def test_warn_status_reroute_target(event_type, chat_type, current, home, expected):
    home_obj = _FakeHomeChannel(home) if home else None
    assert (
        _warn_status_reroute_target(event_type, chat_type, current, home_obj)
        == expected
    )
