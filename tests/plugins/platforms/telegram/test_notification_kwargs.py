"""Telegram notification-mode gating for approval prompts (#132516).

In "important" mode every send carries disable_notification=True unless the metadata
opts out. An approval prompt is the one payload that must ALWAYS opt out: a silently
delivered prompt is indistinguishable from "no prompt" and costs the full
approvals.timeout before the command is refused.
"""

from plugins.platforms.telegram.adapter import TelegramAdapter


def _adapter(mode):
    adapter = object.__new__(TelegramAdapter)
    adapter._notifications_mode = mode
    return adapter


def test_important_mode_silences_plain_sends():
    assert _adapter("important")._notification_kwargs({"thread_id": "t1"}) == {
        "disable_notification": True}


def test_important_mode_pushes_notify_marked_sends():
    assert _adapter("important")._notification_kwargs({"notify": True}) == {}


def test_important_mode_pushes_approval_prompts():
    """#132516: is_approval_prompt must defeat disable_notification even without an
    explicit notify marker — covers every surface that marks the prompt."""
    assert _adapter("important")._notification_kwargs({"is_approval_prompt": True}) == {}
    assert _adapter("important")._notification_kwargs(
        {"thread_id": "t1", "is_approval_prompt": True}) == {}


def test_none_metadata_in_important_mode_stays_silent_for_plain_sends():
    assert _adapter("important")._notification_kwargs(None) == {"disable_notification": True}


def test_all_mode_never_disables_notifications():
    for metadata in (None, {}, {"is_approval_prompt": True}, {"notify": False}):
        assert _adapter("all")._notification_kwargs(metadata) == {}
