"""``exclusive_bot_mentions`` must only drop a message *addressed* to another bot.

A foreign bot handle that merely appears later in the text/caption talks *about* that bot and
must not silence this one (#136236): the gate reads a leading run of handles only. Our own
handle anywhere still wins, as before. Cases mirror the table in the issue, plus the entity-less
raw-text fallback and the skip logging the issue asks for.
"""

import logging
import re
from types import SimpleNamespace

import pytest

from tests.gateway.test_telegram_group_gating import (
    _group_message, _make_adapter, _mention_entities,
)


def _bot_command_entity(text, command):
    return SimpleNamespace(type="bot_command", offset=text.index(command), length=len(command))


def _handles_in(text):
    return [f"@{m.group(1)}" for m in re.finditer(r"@([A-Za-z0-9_]{2,31})", text)]


def _mentioning(text):
    return _group_message(text, entities=_mention_entities(text, _handles_in(text)))


def _adapter():
    return _make_adapter(require_mention=True, exclusive_bot_mentions=True)


@pytest.mark.parametrize("text", [
    "@OtherBot add a lead",
    "@BotA @OtherBot compare these",
    "@OtherBot: add a lead",
    "@alice @OtherBot do this",  # human handle first: the leading run keeps scanning
])
def test_leading_foreign_bot_handle_still_excludes(text):
    """A message whose head addresses other bots stays skipped (today's correct half)."""
    adapter = _adapter()

    assert adapter._explicit_bot_mentions_exclude_self(_mentioning(text)) is True


def test_leading_slash_command_for_other_bot_still_excludes():
    adapter = _adapter()
    text = "/status@OtherBot"
    message = _group_message(text, entities=[_bot_command_entity(text, text)])

    assert adapter._explicit_bot_mentions_exclude_self(message) is True


@pytest.mark.parametrize("text", [
    "Note: moved to @OtherBot, please confirm",
    "moved to @OtherBot",  # plain text, no entities: raw fallback must not silence us either
])
def test_passing_mention_no_longer_silences(text):
    """A foreign handle later in the message only talks about that bot (#136236)."""
    adapter = _adapter()
    assert adapter._explicit_bot_mentions_exclude_self(_mentioning(text)) is False

    plain = _group_message(text)  # Telegram sent no entities at all
    assert adapter._explicit_bot_mentions_exclude_self(plain) is False


def test_passing_mention_in_caption_no_longer_silences():
    adapter = _adapter()
    caption = "The roof that @OtherBot quoted"
    message = _group_message(
        None, caption=caption, caption_entities=_mention_entities(caption, ["@OtherBot"]),
    )

    assert adapter._explicit_bot_mentions_exclude_self(message) is False


def test_plain_leading_foreign_handle_still_skips():
    """Entity-less fallback keeps skipping a message that opens with a foreign bot handle."""
    adapter = _adapter()
    message = _group_message("@OtherBot moved the wiki")  # no entities

    assert adapter._explicit_bot_mentions_exclude_self(message) is True


def test_own_handle_in_leading_run_wins():
    """``@OtherBot @hermes_bot compare`` addresses us too — process."""
    adapter = _adapter()
    text = "@OtherBot @hermes_bot compare"
    message = _group_message(text, entities=_mention_entities(text, ["@OtherBot", "@hermes_bot"]))

    assert adapter._explicit_bot_mentions_exclude_self(message) is False


def test_own_handle_later_in_message_still_wins():
    """Our own handle anywhere wins, as today: a foreign leading run plus a later own mention
    is a collective address (``@OtherBot hey @hermes_bot``), not a foreign-only one."""
    adapter = _adapter()
    text = "@OtherBot hey @hermes_bot whatsup"
    message = _group_message(text, entities=_mention_entities(text, ["@OtherBot", "@hermes_bot"]))

    assert adapter._explicit_bot_mentions_exclude_self(message) is False


def test_code_span_handle_addresses_no_one():
    """With entities, a leading handle counts only where a mention entity starts."""
    adapter = _adapter()
    text = "`@OtherBot` add"
    message = _group_message(text, entities=[SimpleNamespace(type="code", offset=0, length=10)])

    assert adapter._explicit_bot_mentions_exclude_self(message) is False


def test_skip_is_logged_with_chat_message_and_handles(caplog):
    """A vanished message costs a long diagnosis — one INFO line per skip (#136236)."""
    adapter = _adapter()
    message = _mentioning("@OtherBot add a lead")
    message.message_id = 4242

    with caplog.at_level(logging.INFO, logger="plugins.platforms.telegram.adapter"):
        assert adapter._explicit_bot_mentions_exclude_self(message) is True

    record = next(r for r in caplog.records if "addressed to other bot" in r.getMessage())
    assert "-100" in record.getMessage() and "4242" in record.getMessage()
    assert "@otherbot" in record.getMessage()


def test_no_skip_log_when_message_is_processed(caplog):
    adapter = _adapter()
    text = "Note: moved to @OtherBot, please confirm"

    with caplog.at_level(logging.INFO, logger="plugins.platforms.telegram.adapter"):
        assert adapter._explicit_bot_mentions_exclude_self(_mentioning(text)) is False

    assert not [r for r in caplog.records if "addressed to other bot" in r.getMessage()]


def test_skip_still_schedules_bot_identity_recheck():
    """A skip caused by a stale own handle must keep self-correcting via getMe."""
    adapter = _adapter()
    rechecks = []
    adapter._schedule_bot_identity_recheck = lambda: rechecks.append(1)
    message = _mentioning("@OtherBot add a lead")

    assert adapter._explicit_bot_mentions_exclude_self(message) is True
    assert rechecks == [1]


def test_passing_mention_reaches_the_trigger_gate():
    """End to end (``require_mention`` off, so the exclusive gate is the only veto): the gate
    no longer swallows a message that only talks about another bot; a leading foreign handle
    still vetoes it there."""
    adapter = _make_adapter(require_mention=False, exclusive_bot_mentions=True)
    passing = _mentioning("Note: moved to @OtherBot, please confirm")
    addressed = _mentioning("@OtherBot add a lead")

    assert adapter._should_process_message(passing) is True
    assert adapter._should_process_message(addressed) is False
