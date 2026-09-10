"""Tests for Telegram bot mention detection (bug #12545).

The old implementation used a naive substring check
(`f"@{bot_username}" in text.lower()`), which incorrectly matched partial
substrings like 'foo@hermes_bot.example'.

Detection now relies entirely on the MessageEntity objects Telegram's server
emits for real mentions. A bare `@username` substring in message text without
a corresponding `MENTION` entity is NOT a mention — this correctly ignores
@handles that appear inside URLs, code blocks, email-like strings, or quoted
text, because Telegram's parser does not emit mention entities for any of
those contexts.
"""
from types import SimpleNamespace

from gateway.config import Platform, PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter():
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="***", extra={})
    adapter._bot = SimpleNamespace(id=999, username="hermes_bot")
    return adapter


def _mention_entity(text, mention="@hermes_bot"):
    """Build a MENTION entity pointing at a literal `@username` in `text`."""
    offset = text.index(mention)
    return SimpleNamespace(type="mention", offset=offset, length=len(mention))


def _telegram_mention_entity(text, mention="@hermes_bot", entity_type="mention"):
    """Build an entity with Telegram UTF-16 code-unit offset/length values."""
    start = text.index(mention)
    offset = len(text[:start].encode("utf-16-le")) // 2
    length = len(mention.encode("utf-16-le")) // 2
    return SimpleNamespace(type=entity_type, offset=offset, length=length)


def _text_mention_entity(offset, length, user_id):
    """Build a TEXT_MENTION entity (used when the target user has no public @handle)."""
    return SimpleNamespace(
        type="text_mention",
        offset=offset,
        length=length,
        user=SimpleNamespace(id=user_id),
    )


def _message(text=None, caption=None, entities=None, caption_entities=None):
    return SimpleNamespace(
        text=text,
        caption=caption,
        entities=entities or [],
        caption_entities=caption_entities or [],
        message_thread_id=None,
        chat=SimpleNamespace(id=-100, type="group"),
        reply_to_message=None,
    )


class TestRealMentionsAreDetected:
    """A real Telegram mention always comes with a MENTION entity — detect those."""

    def test_mention_at_start_of_message(self):
        adapter = _make_adapter()
        text = "@hermes_bot hello world"
        msg = _message(text=text, entities=[_mention_entity(text)])
        assert adapter._message_mentions_bot(msg) is True


    def test_mention_after_non_bmp_characters_uses_telegram_offsets(self):
        adapter = _make_adapter()
        text = "\U0001f680\U0001f680 @hermes_bot hello"
        msg = _message(text=text, entities=[_telegram_mention_entity(text)])
        assert adapter._message_mentions_bot(msg) is True

    def test_text_mention_entity_targets_bot(self):
        """TEXT_MENTION is Telegram's entity type for @FirstName -> user without a public handle."""
        adapter = _make_adapter()
        msg = _message(text="hey you", entities=[_text_mention_entity(4, 3, user_id=999)])
        assert adapter._message_mentions_bot(msg) is True


class TestSubstringFalsePositivesAreRejected:
    """Bare `@bot_username` substrings without a MENTION entity must NOT match.

    These are all inputs where the OLD substring check returned True incorrectly.
    A word-boundary regex would still over-match some of these (code blocks,
    URLs). Entity-based detection handles them all correctly because Telegram's
    parser does not emit mention entities for non-mention contexts.
    """

    def test_email_like_substring(self):
        """bug #12545 exact repro: 'foo@hermes_bot.example'."""
        adapter = _make_adapter()
        msg = _message(text="email me at foo@hermes_bot.example")
        assert adapter._message_mentions_bot(msg) is False


    def test_substring_inside_url_without_entity(self):
        """@handle inside a URL produces a URL entity, not a MENTION entity."""
        adapter = _make_adapter()
        msg = _message(text="see https://example.com/@hermes_bot for details")
        assert adapter._message_mentions_bot(msg) is False


    def test_email_substring_in_caption(self):
        adapter = _make_adapter()
        msg = _message(caption="foo@hermes_bot.example")
        assert adapter._message_mentions_bot(msg) is False


class TestEntityEdgeCases:
    """Malformed or mismatched entities should not crash or over-match."""


    def test_malformed_entity_with_negative_offset(self):
        adapter = _make_adapter()
        msg = _message(text="@hermes_bot hi",
                       entities=[SimpleNamespace(type="mention", offset=-1, length=11)])
        assert adapter._message_mentions_bot(msg) is False


class TestCaseInsensitivity:
    """Telegram usernames are case-insensitive; the slice-compare normalizes both sides."""

    def test_uppercase_mention(self):
        adapter = _make_adapter()
        text = "hi @HERMES_BOT"
        msg = _message(text=text, entities=[_mention_entity(text, mention="@HERMES_BOT")])
        assert adapter._message_mentions_bot(msg) is True

    def test_mixed_case_mention(self):
        adapter = _make_adapter()
        text = "hi @Hermes_Bot"
        msg = _message(text=text, entities=[_mention_entity(text, mention="@Hermes_Bot")])
        assert adapter._message_mentions_bot(msg) is True


class TestTelegramUtf16EntityOffsets:
    def test_extracts_bot_mention_username_after_non_bmp_characters(self):
        text = "\U0001f9ea\U0001f9ea @hermes_bot please"
        msg = _message(text=text, entities=[_telegram_mention_entity(text)])
        assert TelegramAdapter._extract_bot_mention_usernames(msg) == {"hermes_bot"}

    def test_bot_command_suffix_after_non_bmp_characters_mentions_bot(self):
        adapter = _make_adapter()
        text = "\U0001f9ea\U0001f9ea /new@hermes_bot"
        entity = _telegram_mention_entity(text, mention="/new@hermes_bot", entity_type="bot_command")
        msg = _message(text=text, entities=[entity])
        assert adapter._message_mentions_bot(msg) is True

    def test_extracts_bot_command_target_after_non_bmp_characters(self):
        text = "\U0001f9ea\U0001f9ea /new@hermes_bot"
        entity = _telegram_mention_entity(text, mention="/new@hermes_bot", entity_type="bot_command")
        msg = _message(text=text, entities=[entity])
        assert TelegramAdapter._extract_bot_mention_usernames(msg) == {"hermes_bot"}


class TestBotTriggerTextCleaning:
    """_clean_bot_trigger_text strips our own handle but must keep the whitespace
    separating a group command (/cmd@botname <args>) from its arguments (#107082).

    Telegram's group command menu auto-disambiguates to the /cmd@botname form; the
    old trailing ``\\s*`` in the strip regex swallowed the command↔args separator,
    so ``get_command()`` glued the args onto the command name and the message fell
    through to default busy handling (interrupt instead of queue)."""

    def test_group_command_keeps_arg_separator(self):
        adapter = _make_adapter()
        assert adapter._clean_bot_trigger_text("/queue@hermes_bot 帮我整理") == "/queue 帮我整理"

    def test_group_command_still_resolves_as_command_with_args(self):
        from gateway.platforms.event import MessageEvent

        adapter = _make_adapter()
        event = MessageEvent(text=adapter._clean_bot_trigger_text("/queue@hermes_bot do something"))
        assert event.get_command() == "queue"
        assert event.get_command_args() == "do something"

    def test_command_without_args_unchanged(self):
        adapter = _make_adapter()
        assert adapter._clean_bot_trigger_text("/new@hermes_bot") == "/new"

    def test_standalone_mention_still_cleaned(self):
        adapter = _make_adapter()
        assert adapter._clean_bot_trigger_text("@hermes_bot 你好") == "你好"

    def test_punctuated_mention_still_cleaned(self):
        adapter = _make_adapter()
        assert adapter._clean_bot_trigger_text("@hermes_bot: run it") == "run it"

    def test_mid_sentence_mention_tokens_preserved(self):
        adapter = _make_adapter()
        assert adapter._clean_bot_trigger_text("hey @hermes_bot do X") == "hey do X"

    def test_multiline_mention_keeps_newline_boundaries(self):
        adapter = _make_adapter()
        cleaned = adapter._clean_bot_trigger_text("line one\n@hermes_bot line two")
        assert cleaned.splitlines()[0] == "line one"
