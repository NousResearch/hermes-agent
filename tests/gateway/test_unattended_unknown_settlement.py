"""Regression: a lost ``-z`` one-shot turn must settle, not pause its session forever.

``session_finite.admit_finite`` commits ``unattended: true`` onto the admission payload precisely
because ``hermes -z`` has no user available to answer a prompt. ``is_unattended`` did not read that
flag, so such a turn fenced ``unknown`` by a restart fell through to "a person's turn": the session's
FIFO paused and waited for a resend that could never come (#139532).
"""
import pytest


def _row(payload, principal='uid:1000', status='unknown'):
    return {'admission_id': 'a1', 'target_session_id': 's1', 'status': status,
            'principal_id': principal, 'payload': payload, 'generation': 1}


def test_unattended_one_shot_is_settled_not_paused():
    from gateway.session_unattended_unknown import is_unattended
    assert is_unattended(_row({'text': 'Reply with exactly: OK', 'finite': True, 'unattended': True}))


def test_a_persons_ordinary_turn_still_pauses():
    """The fence must keep protecting a human's turn: only they may decide to resend it."""
    from gateway.session_unattended_unknown import is_unattended
    assert not is_unattended(_row({'text': 'continue', 'finite': True}))
    assert not is_unattended(_row({'text': 'continue'}))


@pytest.mark.parametrize('payload', [
    {'text': 'x', 'finite': True, 'unattended': False},
    {'text': 'x', 'finite': True, 'unattended': 'true'},   # only a real bool counts
    {'text': 'x', 'finite': True, 'unattended': 1},
])
def test_only_a_true_bool_marks_unattended(payload):
    from gateway.session_unattended_unknown import is_unattended
    assert not is_unattended(_row(payload))


def test_existing_markers_still_settle():
    """No regression for the automation shapes #141357 already covered."""
    from gateway.session_unattended_unknown import is_unattended
    assert is_unattended(_row({'local_automation_v1': {}}))
    assert is_unattended(_row({'kanban_card_v1': {}}))
    assert is_unattended(_row({'native_text_v1': {'automation': {}}}))
    assert is_unattended(_row({'native_text_v1': {'webhook_delivery': {}}}))
    assert is_unattended(_row({'text': 'x'}, principal='cron-owner'))
    assert is_unattended(_row({'text': 'x'}, principal='automation:route'))
