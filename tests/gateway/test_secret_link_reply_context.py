"""A Telegram reply to the bot's secret-link message must not carry the one-time token (or the
full link) into the agent context: the quoted text is redacted, nothing else in it changes."""

from types import SimpleNamespace

import pytest

from gateway import secret_link
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from plugins.platforms.telegram.adapter import TelegramAdapter
from tests.gateway.secret_link_helpers import BASE

_TOKEN = "Zq3-Tok_" + "x" * 35  # 43 characters, like a real token
_LINK = f"{BASE}/secret-link/{_TOKEN}"
_BOT_MESSAGE = secret_link.link_message(_LINK, "12:00 UTC", "STRIPE_SECRET_KEY")


def _reply(quote: str | None = None, text: str = _BOT_MESSAGE):
    return SimpleNamespace(
        reply_to_message=SimpleNamespace(message_id=9, text=text, caption=None),
        quote=SimpleNamespace(text=quote) if quote is not None else None,
        chat=SimpleNamespace(id=4242))


def _agent_text(message) -> tuple[str, str]:
    """``(reply_to_text the adapter hands over, message text the agent receives)``."""
    adapter = object.__new__(TelegramAdapter)
    reply_to_id, reply_to_text = adapter._reply_context(message)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="4242", chat_type="dm", user_id="7")
    event = MessageEvent(text="what is this?", source=source, reply_to_message_id=reply_to_id,
                         reply_to_text=reply_to_text)
    return reply_to_text, GatewayRunner._prepend_inbound_reply_context(event, source, event.text)


@pytest.mark.parametrize("quote", [
    None,                                  # full replied-to message
    _LINK,                                 # native quote: the whole link
    f"/secret-link/{_TOKEN}",              # native quote: path only
    _TOKEN,                                # native quote: the bare token
    f"Open this link\n{_LINK}\nKey name",  # native quote: link inside a longer selection
], ids=["full-message", "quoted-link", "quoted-path", "quoted-token", "quoted-span"])
def test_replying_to_the_link_message_never_puts_the_token_in_the_agent_context(quote):
    reply_to_text, agent_text = _agent_text(_reply(quote))
    for seen in (reply_to_text, agent_text):
        assert _TOKEN not in seen
        assert _LINK not in seen
    assert "what is this?" in agent_text


def test_the_rest_of_the_quoted_link_message_is_kept():
    reply_to_text, _ = _agent_text(_reply())
    assert "Open this link to enter a secret." in reply_to_text
    assert "Key name: STRIPE_SECRET_KEY" in reply_to_text
    assert "expires at 12:00 UTC" in reply_to_text


def test_other_replies_are_not_touched():
    other = f"build id {_TOKEN} deployed to {BASE}/status"
    reply_to_text, _ = _agent_text(_reply(text=other))
    assert reply_to_text == other
    quoted, _ = _agent_text(_reply(quote=_TOKEN, text=other))
    assert quoted == _TOKEN
