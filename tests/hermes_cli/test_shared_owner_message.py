"""Ownership age is not evidence that a model turn is running."""

from hermes_cli.active_sessions import session_already_owned_message


def test_owner_refusal_distinguishes_lease_age_from_turn_activity():
    message = session_already_owned_message("session", {
        "surface": "desktop", "pid": 123, "started_at": 1,
    })
    assert "lease age" in message
    assert "turn activity is unknown" in message
    assert "close the session in its owning surface" in message
    assert "running " not in message
