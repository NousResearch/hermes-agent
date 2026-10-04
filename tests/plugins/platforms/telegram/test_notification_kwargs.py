"""Telegram notification-mode gating (#132516): "important" mode silences ordinary sends, but a
human-decision prompt (``is_approval_prompt``) must always push — a silently delivered prompt is
indistinguishable from "no prompt" and costs the full approvals.timeout before the command is refused.
"""

import pytest

from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.mark.parametrize("mode,metadata,expected", [
    ("important", None, {"disable_notification": True}),
    ("important", {"thread_id": "t1"}, {"disable_notification": True}),
    ("important", {"notify": True}, {}),
    ("important", {"thread_id": "t1", "is_approval_prompt": True}, {}),
    ("all", None, {}),
    ("all", {"is_approval_prompt": True}, {}),
])
def test_notification_kwargs(mode, metadata, expected):
    adapter = object.__new__(TelegramAdapter)
    adapter._notifications_mode = mode
    assert adapter._notification_kwargs(metadata) == expected
