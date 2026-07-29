import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageType
from gateway.relay.ws_transport import _event_from_wire


def _wire(text: str, *, platform: str = "slack") -> dict:
    return {
        "text": text,
        "message_type": "command",
        "source": {
            "platform": platform,
            "chat_id": "D123",
            "chat_type": "dm",
            "user_id": "U123",
        },
    }


@pytest.mark.parametrize(
    ("wire_text", "expected"),
    [
        ("/hermes sethome", "/sethome"),
        ("/hermes\tsethome", "/sethome"),
        (
            "/hermes model gpt-5.6 --provider openai",
            "/model gpt-5.6 --provider openai",
        ),
        ("/hermes", "/help"),
    ],
)
def test_slack_relay_parent_becomes_gateway_command(wire_text: str, expected: str):
    event = _event_from_wire(_wire(wire_text))

    assert event.text == expected
    assert event.message_type == MessageType.COMMAND
    assert event.source.platform == Platform.SLACK
    assert event.source.delivered_via_upstream_relay is True


def test_slack_relay_parent_freeform_text_matches_native_adapter():
    event = _event_from_wire(_wire("/hermes explain this"))

    assert event.text == "explain this"
    assert event.message_type == MessageType.TEXT


def test_non_slack_relay_message_is_not_rewritten():
    event = _event_from_wire(_wire("/hermes sethome", platform="discord"))

    assert event.text == "/hermes sethome"
    assert event.message_type == MessageType.COMMAND
