"""Tests for Telegram native partial-quote handling in _build_message_event.

When a Telegram user replies using Telegram's native quote feature to
select only part of a prior message, the adapter must use ``message.quote.text``
(the user-selected substring) rather than ``message.reply_to_message.text``
(the entire replied-to message). Otherwise the agent receives the full prior
message as ``reply_to_text``, which can cause it to act on unrelated
actionable-looking text the user did not quote (#22619).
"""

import json
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter():
    return TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={}))


def _make_message(
    text="follow-up",
    reply_to_text=None,
    reply_to_caption=None,
    reply_to_id=42,
    quote_text=None,
    reply_to_entities=None,
    reply_to_caption_entities=None,
    quote_position=None,
    quote_rich_texts=None,
    quote_rich_texts_as_attribute=False,
):
    chat = SimpleNamespace(id=111, type="private", title=None, full_name="Alice")
    user = SimpleNamespace(id=42, full_name="Alice")

    reply_to_message = None
    if reply_to_text is not None or reply_to_caption is not None:
        reply_to_message = SimpleNamespace(
            message_id=reply_to_id,
            text=reply_to_text,
            caption=reply_to_caption,
            entities=reply_to_entities,
            caption_entities=reply_to_caption_entities,
        )

    quote = None
    if quote_text is not None or quote_rich_texts is not None:
        quote = SimpleNamespace(text=quote_text or "", position=quote_position)
        if quote_rich_texts is not None:
            if quote_rich_texts_as_attribute:
                # A future PTB release may model ``rich_texts`` as a first-class attribute
                # instead of leaving it in ``api_kwargs``.
                quote.rich_texts = quote_rich_texts
            else:
                quote.api_kwargs = {"rich_texts": quote_rich_texts}

    return SimpleNamespace(
        chat=chat,
        from_user=user,
        text=text,
        message_thread_id=None,
        message_id=1001,
        reply_to_message=reply_to_message,
        quote=quote,
        date=None,
        forum_topic_created=None,
    )


def test_native_partial_quote_used_as_reply_to_text():
    """When ``message.quote`` is present, prefer the selected substring."""
    from gateway.platforms.event import MessageType

    adapter = _make_adapter()
    msg = _make_message(
        text="mark this one as done",
        reply_to_text=(
            "Briefing:\n- Item A: deploy fix\n- Item B: rotate keys\n- Item C: update docs"
        ),
        quote_text="Item B: rotate keys",
    )

    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.reply_to_text == "Item B: rotate keys"
    assert event.reply_to_message_id == "42"


def _text_link(offset, length, url):
    return SimpleNamespace(type="text_link", offset=offset, length=length, url=url)


def test_hidden_link_in_replied_to_text_is_inlined():
    """A reply to a message whose link sits behind its anchor text still carries the URL."""
    from gateway.platforms.event import MessageType

    msg = _make_message(
        text="summarize the linked article",
        reply_to_text="New post: Read the full report here",
        reply_to_entities=[_text_link(19, 11, "https://example.com/report-2026")],
    )

    event = _make_adapter()._build_message_event(msg, MessageType.TEXT)

    assert event.reply_to_text == (
        "New post: Read the full report (https://example.com/report-2026) here"
    )


def test_hidden_link_in_replied_to_caption_is_inlined():
    from gateway.platforms.event import MessageType

    msg = _make_message(
        text="what is this?",
        reply_to_caption="Chart from the dashboard",
        reply_to_caption_entities=[_text_link(15, 9, "https://example.com/dash")],
    )

    event = _make_adapter()._build_message_event(msg, MessageType.TEXT)

    assert event.reply_to_text == "Chart from the dashboard (https://example.com/dash)"


def test_hidden_link_in_native_quote_is_inlined():
    """Quotes keep no ``text_link`` entities, so the link comes from the replied-to message."""
    from gateway.platforms.event import MessageType

    msg = _make_message(
        text="open this one",
        reply_to_text="Links: the docs and the changelog",
        reply_to_entities=[
            _text_link(11, 4, "https://example.com/docs"),
            _text_link(24, 9, "https://example.com/changes"),
        ],
        quote_text="the changelog",
        quote_position=20,
    )

    event = _make_adapter()._build_message_event(msg, MessageType.TEXT)

    assert event.reply_to_text == "the changelog (https://example.com/changes)"


def test_native_quote_at_a_mismatched_position_is_left_as_quoted():
    """``position`` is approximate; links are only mapped where the quote matches exactly."""
    from gateway.platforms.event import MessageType

    msg = _make_message(
        text="open this one",
        reply_to_text="Links: the docs and the changelog",
        reply_to_entities=[_text_link(24, 9, "https://example.com/changes")],
        quote_text="the changelog",
        quote_position=18,
    )

    event = _make_adapter()._build_message_event(msg, MessageType.TEXT)

    assert event.reply_to_text == "the changelog"


# --- Bot API 10.4: quotes chosen from rich messages -----------------------------------------


def test_rich_quote_used_when_quote_text_is_empty():
    """A rich quote has empty ``text``; its selection must still win over the whole reply."""
    from gateway.platforms.event import MessageType

    msg = _make_message(
        text="do the selected bit",
        reply_to_text=(
            "Briefing:\n- Item A: deploy fix\n- Item B: rotate keys\n- Item C: update docs"
        ),
        quote_text="",
        quote_rich_texts=[["Item B: ", [{"type": "bold", "text": "rotate"}], " keys"]],
    )

    event = _make_adapter()._build_message_event(msg, MessageType.TEXT)

    assert event.reply_to_text == "Item B: rotate keys"
    assert event.reply_to_message_id == "42"


def test_rich_quote_top_level_segments_join_with_newlines():
    """The top-level array holds distinct quoted segments, not inline runs."""
    from gateway.platforms.event import MessageType

    msg = _make_message(
        text="apply both",
        reply_to_text="unrelated body",
        quote_text="",
        quote_rich_texts=[["First segment"], "Second segment"],
    )

    event = _make_adapter()._build_message_event(msg, MessageType.TEXT)

    assert event.reply_to_text == "First segment\nSecond segment"


def test_rich_quote_nested_segments_concatenate_inline():
    """Nested arrays are inline runs: no separator is added inside one quoted segment."""
    from gateway.platforms.event import MessageType

    msg = _make_message(
        text="apply this",
        reply_to_text="unrelated body",
        quote_text="",
        quote_rich_texts=[[{"type": "italic", "text": "rotate "}, ["the ", "keys"]]],
    )

    event = _make_adapter()._build_message_event(msg, MessageType.TEXT)

    assert event.reply_to_text == "rotate the keys"


def test_rich_quote_flattens_math_custom_emoji_and_date_time():
    """Documented rich quote nodes carry their readable text, not just plain runs."""
    from gateway.platforms.event import MessageType

    msg = _make_message(
        text="check this",
        reply_to_text="unrelated body",
        quote_text="",
        quote_rich_texts=[
            [
                "See ",
                {"type": "mathematical_expression", "expression": "x^2"},
                " for ",
                {
                    "type": "custom_emoji",
                    "custom_emoji_id": "5368324170671202286",
                    "alternative_text": "😀",
                },
                " on ",
                {
                    "type": "date_time",
                    "text": "Nov 9",
                    "unix_time": 1762646400,
                    "date_time_format": "{month} {day}",
                },
            ]
        ],
    )

    event = _make_adapter()._build_message_event(msg, MessageType.TEXT)

    assert event.reply_to_text == "See x^2 for 😀 on Nov 9"


def test_empty_rich_quote_never_injects_full_replied_message():
    """A present but contentless rich selection is still a quote, not a cue to echo the reply."""
    from gateway.platforms.event import MessageType

    msg = _make_message(
        text="hm",
        reply_to_text="Briefing:\n- Item A: deploy fix\n- Item B: rotate keys",
        quote_text="",
        quote_rich_texts=[[]],
    )

    event = _make_adapter()._build_message_event(msg, MessageType.TEXT)

    assert event.reply_to_text is None
    assert event.reply_to_message_id == "42"


def test_rich_quote_read_from_attribute_when_a_future_ptb_models_it():
    """The adapter also accepts ``quote.rich_texts`` directly, not only ``api_kwargs``."""
    from gateway.platforms.event import MessageType

    msg = _make_message(
        text="apply this",
        reply_to_text="unrelated body",
        quote_text="",
        quote_rich_texts=[["Modelled field"]],
        quote_rich_texts_as_attribute=True,
    )

    event = _make_adapter()._build_message_event(msg, MessageType.TEXT)

    assert event.reply_to_text == "Modelled field"


def test_quote_without_rich_texts_still_falls_back_to_full_reply():
    """Quotes from ordinary messages keep the pre-10.4 fallback when they carry no text."""
    from gateway.platforms.event import MessageType

    msg = _make_message(
        text="continue",
        reply_to_text="Full briefing body",
        quote_text="",
    )

    event = _make_adapter()._build_message_event(msg, MessageType.TEXT)

    assert event.reply_to_text == "Full briefing body"


_REPO = pathlib.Path(__file__).resolve().parents[2]

_REAL_PTB_PROBE = r"""
import json, sys
sys.path.insert(0, __REPO__)
from telegram import Message
from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter

adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={}))
payload = {
    "message_id": 1001,
    "date": 1762646400,
    "chat": {"id": 111, "type": "private"},
    "from": {"id": 42, "is_bot": False, "first_name": "Alice"},
    "text": "do the selected bit",
    "reply_to_message": {
        "message_id": 42,
        "date": 1762646400,
        "chat": {"id": 111, "type": "private"},
        "text": "Full briefing:\n- Item A: deploy fix\n- Item B: rotate keys",
    },
    "quote": {
        "text": "",
        "position": 0,
        "is_manual": True,
        "rich_texts": [
            ["Item B: ", [{"type": "bold", "text": "rotate"}], " keys"],
            [{"type": "mathematical_expression", "expression": "x^2"}],
        ],
    },
}
message = Message.de_json(payload, None)
quote = message.quote
print("___RESULT___")
print(json.dumps({
    "quote_type": type(quote).__name__,
    "quote_text": quote.text,
    "rich_texts": quote.api_kwargs.get("rich_texts"),
    "reply_context": list(adapter._reply_context(message)),
}))
"""


def _probe_real_ptb():
    """Run the payload check in a clean interpreter (no conftest ``telegram`` mock)."""
    code = _REAL_PTB_PROBE.replace("__REPO__", repr(str(_REPO)))
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    if proc.returncode != 0:
        pytest.skip(f"real python-telegram-bot unavailable: {proc.stderr.strip()[-300:]}")
    marker = proc.stdout.rfind("___RESULT___")
    return json.loads(proc.stdout[marker + len("___RESULT___"):].strip())


def test_real_ptb_passes_rich_quote_payload_through():
    """Real PTB ``Message.de_json`` keeps ``rich_texts`` in ``api_kwargs`` and the adapter reads it."""
    result = _probe_real_ptb()

    assert result["quote_type"] == "TextQuote"
    assert result["quote_text"] == ""
    assert result["rich_texts"] == [
        ["Item B: ", [{"type": "bold", "text": "rotate"}], " keys"],
        [{"type": "mathematical_expression", "expression": "x^2"}],
    ]
    assert result["reply_context"] == ["42", "Item B: rotate keys\nx^2"]

