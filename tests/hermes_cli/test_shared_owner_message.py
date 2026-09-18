"""The owner refusal never claims a turn is running, and leads with plain words."""

from hermes_cli.active_sessions import session_already_owned_message


def test_owner_refusal_is_plain_and_never_claims_a_running_turn():
    message = session_already_owned_message("session", {
        "surface": "desktop", "pid": 123, "started_at": 1,
    })
    first, details = message.splitlines()
    assert first.startswith("This chat is open in another Hermes window/terminal.")
    assert "start a new chat here" in first
    assert details.startswith("Details: ") and "desktop" in details and "ago" in details
    assert "running " not in message
    assert "lease" not in message and "pid" not in message
