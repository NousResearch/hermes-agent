"""The local-viewer card carries the plain-English explanation.

`PendingControls.register` is the ONE payload a local viewer (Desktop / Ink TUI) renders as an
approval card: it publishes `approval.request` with an explicit field list, so a field missing
here is a field the card can never show — regardless of what the gateway notification carries.
This is the path an earlier revision of the feature missed.
"""

from gateway.session_pending_controls import PendingControls


class _Events:
    """Minimal event bus stand-in: records what the control layer publishes."""

    def __init__(self):
        import threading

        self.lock = threading.Lock()
        self.published = []

    def publish(self, session_id, prompt, *, event_type):
        self.published.append((event_type, dict(prompt)))


def _data(**overrides):
    data = {
        "request_id": "req-1",
        "command": "rm -rf /tmp/hermes-explain-demo",
        "description": "recursive delete",
        "explanation": "Permanently deletes the folder and everything inside it.",
        "allow_session": True,
        "allow_permanent": True,
    }
    data.update(overrides)
    return data


def test_published_card_carries_the_explanation():
    events = _Events()
    controls = PendingControls(events)

    controls.register("sid-1", "route", 3, _data())

    event_type, prompt = events.published[-1]
    assert event_type == "approval.request"
    assert prompt["explanation"] == "Permanently deletes the folder and everything inside it."
    # Still an ordinary approval: the line adds context, it does not change the offered scopes.
    assert prompt["choices"] == ["once", "deny", "session", "always"]
    assert prompt["kind"] == "approval"


def test_a_request_without_an_explanation_publishes_an_empty_string():
    events = _Events()
    controls = PendingControls(events)

    controls.register("sid-2", "route", 4, _data(explanation=""))

    _, prompt = events.published[-1]
    assert prompt["explanation"] == ""
