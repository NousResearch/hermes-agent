"""The /new paused-queue note names a discard command the chat can actually send (#145177).

Slack registers no native ``/discard`` (its slash-command cap); the umbrella ``/hermes discard``
reaches it, as the lost-turn pause notice already tells Slack users.
"""
from agent.i18n import t
from gateway.config import Platform
from gateway.session import SessionSource
from gateway.slash_commands_session import discard_command_for


def _note(platform):
    source = SessionSource(platform=platform, chat_id="c1", chat_type="dm", user_id="u1")
    return t("gateway.reset.paused_left", count=2, session_id="sid-1", discard=discard_command_for(source))


def test_slack_is_told_the_umbrella_discard_command():
    note = _note(Platform.SLACK)
    assert "/hermes discard" in note and "/resume sid-1" in note


def test_telegram_is_told_the_native_discard_command():
    note = _note(Platform.TELEGRAM)
    assert "then /discard" in note and "/hermes" not in note
