"""A text-less ``rich_message`` group message is addressed exactly like the same text sent as plain text (#151730):
every addressing check (other-bot leading mention, ``/cmd@other_bot``, wake words, the reply-expected rule and the
other-participant guard on handle stripping) reads the same text, so a message meant for another bot stays theirs."""

from types import SimpleNamespace

import pytest

from gateway.platforms.event import MessageType
from plugins.platforms.telegram.telegram_context import group_reply_expected, group_trigger_text
from tests.gateway.test_telegram_group_gating import _group_message, _make_adapter


def _rich(*paragraphs):
    msg = _group_message(None)
    blocks = [p if isinstance(p, dict) else {"type": "paragraph", "text": p} for p in paragraphs]
    msg.api_kwargs = {"rich_message": {"blocks": blocks}}
    return msg


def _plain(text, *spans):
    return _group_message(text, entities=[SimpleNamespace(type=t, offset=text.index(s), length=len(s)) for t, s in spans])


_LINK = ["see ", {"type": "url", "text": "docs", "url": "https://chompy.example.com/docs"}]
_BOTH = "@hermes_bot , @ops_bot are you both listening?"


def _gate(extra, msg):
    return _make_adapter(**extra)._should_process_message(msg)


def _reply_expected(extra, msg):
    return group_reply_expected(_make_adapter(**extra), msg, "group")


def _trigger_text(extra, msg):
    adapter = _make_adapter(**extra)
    return group_trigger_text(adapter, msg, adapter._build_message_event(msg, MessageType.TEXT).text.split("\n")[0])


@pytest.mark.parametrize("probe, extra, plain, rich, expected", [
    (_gate, {"require_mention": False}, _plain("@other_bot please deploy now", ("mention", "@other_bot")),
     _rich("@other_bot please deploy now", "p2"), False),
    (_gate, {"require_mention": False}, _plain("/status@other_bot now", ("bot_command", "/status@other_bot")),
     _rich("/status@other_bot now", "second paragraph"), False),
    (_gate, {"require_mention": True, "mention_patterns": [r"\bchompy\b"]},
     _group_message("see docs", entities=[SimpleNamespace(type="text_link", offset=4, length=4,
                                                          url="https://chompy.example.com/docs")]),
     _rich(_LINK), False),
    (_reply_expected, {"require_mention": False}, _plain("anyone around?"), _rich("anyone around?", "p2"), None),
    (_trigger_text, {"require_mention": True},
     _plain(_BOTH, ("mention", "@hermes_bot"), ("mention", "@ops_bot")), _rich(_BOTH), _BOTH),
    # Leading code is content, as in plain text: the later foreign handle is not the addressee.
    (_gate, {"require_mention": False}, _plain("x @other_bot please explain", ("code", "x"), ("mention", "@other_bot")),
     _rich([{"type": "code", "text": "x"}, " @other_bot please explain"]), True),
    (_gate, {"require_mention": False}, _plain("ls\n@other_bot run it", ("pre", "ls"), ("mention", "@other_bot")),
     _rich({"type": "pre", "text": "ls"}, "@other_bot run it"), True),
    # The opposite side: a rich message addressed to us is still answered, and our sole handle is still stripped.
    (_gate, {"require_mention": True}, _plain("@hermes_bot deploy", ("mention", "@hermes_bot")),
     _rich("@hermes_bot deploy", "p2"), True),
    (_gate, {"require_mention": True}, _plain("/status@hermes_bot", ("bot_command", "/status@hermes_bot")),
     _rich("/status@hermes_bot", "p2"), True),
    (_trigger_text, {"require_mention": True}, _plain("@hermes_bot deploy", ("mention", "@hermes_bot")),
     _rich("@hermes_bot deploy"), "deploy"),
])
def test_rich_only_group_message_is_addressed_like_plain_text(probe, extra, plain, rich, expected):
    assert probe(extra, plain) == expected
    assert probe(extra, rich) == expected

def test_rich_linked_foreign_command_does_not_override_our_mention():
    """A rich hyperlink label is text_link, not Telegram's bot_command entity (#151730)."""
    linked = "/status@other_bot"
    plain = _group_message(
        f"{linked} @hermes_bot",
        entities=[
            SimpleNamespace(type="text_link", offset=0, length=len(linked),
                            url="https://example.com/status"),
            SimpleNamespace(type="mention", offset=len(linked) + 1, length=len("@hermes_bot")),
        ],
    )
    rich = _rich([{"type": "url", "text": linked, "url": "https://example.com/status"}, " @hermes_bot"])

    # The user deliberately mentions us; a clickable command-shaped label cannot veto that.
    assert _gate({"require_mention": True}, plain) is True
    assert _gate({"require_mention": True}, rich) is True
