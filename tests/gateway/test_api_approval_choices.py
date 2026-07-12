"""Approval choice rendering for the desktop/API gateway."""

from gateway.platforms.api_server import _approval_event_choices


def test_api_approval_choices_include_always_when_supported():
    assert _approval_event_choices(allow_permanent=True) == [
        "once", "session", "always", "deny"
    ]


def test_api_approval_choices_hide_always_when_unsupported():
    assert _approval_event_choices(allow_permanent=False) == [
        "once", "session", "deny"
    ]
