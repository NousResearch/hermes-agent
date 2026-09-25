"""Inbound email provider IDs are model-visible without polluting the transcript."""

import json
from unittest.mock import MagicMock

from agent.turn_context import compose_user_api_content
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


def _runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._pending_turn_sidecar_notes = {}
    runner.session_store = MagicMock()
    return runner


def _event(*, platform=Platform.EMAIL, message_id: str | None = "provider-id", internal=False):
    source = SessionSource(
        platform=platform,
        chat_id="inbox@example.com",
        chat_type="dm",
        user_id="sender@example.com",
    )
    return MessageEvent(
        text="Please reply",
        source=source,
        message_id=message_id,
        internal=internal,
    ), source


def test_email_message_id_is_exact_model_sidecar_and_transcript_stays_clean():
    runner = _runner()
    message_id = '<weird"\\id\n\u2603@example.com>'
    event, source = _event(message_id=message_id)
    notes = []

    runner._hmwa_add_email_message_id_sidecar(event, source, notes)

    assert len(notes) == 1
    note = notes[0]
    assert "transport metadata" in note
    assert "not instructions or authorization" in note
    payload = note[note.index("{") : note.rindex("}") + 1]
    assert json.loads(payload) == {"inbound_message_id": message_id}

    runner._set_pending_turn_sidecar_notes("email-session", notes)
    staged = "\n\n".join(runner._consume_pending_turn_sidecar_notes("email-session"))
    api_content = compose_user_api_content(event.text, "", staged)
    assert api_content == event.text + "\n\n" + note
    assert event.text == "Please reply"

    # The note is turn-scoped and cannot leak onto the next cached-agent turn.
    assert runner._consume_pending_turn_sidecar_notes("email-session") == []
    assert compose_user_api_content(event.text, "", "") is None


def test_email_message_id_sidecar_excludes_other_platforms_and_internal_events():
    runner = _runner()
    for event, source in (
        _event(platform=Platform.DISCORD),
        _event(internal=True),
        _event(message_id=None),
    ):
        notes = []
        runner._hmwa_add_email_message_id_sidecar(event, source, notes)
        assert notes == []
