"""Chat punctuation contract: accepted inbound, stripped from chat display.

Outbound: every chat surface shows no punctuation (Unicode P category) while
words and line breaks survive; raw/programmatic surfaces and persisted history
keep the original text. Both outbound choke points (interactive finals,
cron/artifact delivery) call the display helper under test. Inbound: slash
commands tolerate edge punctuation.
"""

import unicodedata

from gateway.platforms.event import MessageEvent
from gateway.response_filters import strip_punctuation_for_display
from hermes_cli.commands import resolve_command


def _punctuation_chars(text: str) -> set:
    return {ch for ch in text if unicodedata.category(ch).startswith("P")}


def test_chat_display_has_no_punctuation_but_keeps_words():
    raw = "Hello, world! How are you?\n*Bold* - dash... done."
    shown = strip_punctuation_for_display(raw)
    # Behaviour contract: no punctuation visible, words and lines preserved.
    assert _punctuation_chars(shown) == set()
    assert "Hello" in shown and "world" in shown and "done" in shown
    assert "\n" in shown
    assert strip_punctuation_for_display("") == ""


def test_punctuated_slash_commands_still_resolve():
    for text, name in [("/stop!", "stop"), ("/new?", "new"),
                       ("/status...", "status"), ("/reload-mcp!", "reload-mcp")]:
        command = MessageEvent(text=text).get_command()
        # Behaviour contract: edge punctuation accepted, inner punctuation kept,
        # and the name resolves through the real command registry.
        assert command == name
        assert resolve_command(command) is not None
    assert MessageEvent(text="just chatting!").get_command() is None
