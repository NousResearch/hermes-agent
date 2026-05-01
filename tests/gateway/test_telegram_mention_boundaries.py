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
from gateway.platforms.telegram import TelegramAdapter


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

    def test_mention_mid_sentence(self):
        adapter = _make_adapter()
        text = "hey @hermes_bot, can you help?"
        msg = _message(text=text, entities=[_mention_entity(text)])
        assert adapter._message_mentions_bot(msg) is True

    def test_mention_at_end_of_message(self):
        adapter = _make_adapter()
        text = "thanks for looking @hermes_bot"
        msg = _message(text=text, entities=[_mention_entity(text)])
        assert adapter._message_mentions_bot(msg) is True

    def test_mention_in_caption(self):
        adapter = _make_adapter()
        caption = "photo for @hermes_bot"
        msg = _message(caption=caption, caption_entities=[_mention_entity(caption)])
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

    def test_hostname_substring(self):
        adapter = _make_adapter()
        msg = _message(text="contact user@hermes_bot.domain.com")
        assert adapter._message_mentions_bot(msg) is False

    def test_superstring_username(self):
        """`@hermes_botx` is a different username; Telegram would emit a mention
        entity for `@hermes_botx`, not `@hermes_bot`."""
        adapter = _make_adapter()
        msg = _message(text="@hermes_botx hello")
        assert adapter._message_mentions_bot(msg) is False

    def test_underscore_suffix_substring(self):
        adapter = _make_adapter()
        msg = _message(text="see @hermes_bot_admin for help")
        assert adapter._message_mentions_bot(msg) is False

    def test_substring_inside_url_without_entity(self):
        """@handle inside a URL produces a URL entity, not a MENTION entity."""
        adapter = _make_adapter()
        msg = _message(text="see https://example.com/@hermes_bot for details")
        assert adapter._message_mentions_bot(msg) is False

    def test_substring_inside_code_block_without_entity(self):
        """Telegram doesn't emit mention entities inside code/pre entities."""
        adapter = _make_adapter()
        msg = _message(text="use the string `@hermes_bot` in config")
        assert adapter._message_mentions_bot(msg) is False

    def test_plain_text_with_no_at_sign(self):
        adapter = _make_adapter()
        msg = _message(text="just a normal group message")
        assert adapter._message_mentions_bot(msg) is False

    def test_email_substring_in_caption(self):
        adapter = _make_adapter()
        msg = _message(caption="foo@hermes_bot.example")
        assert adapter._message_mentions_bot(msg) is False


class TestEntityEdgeCases:
    """Malformed or mismatched entities should not crash or over-match."""

    def test_mention_entity_for_different_username(self):
        adapter = _make_adapter()
        text = "@someone_else hi"
        msg = _message(text=text, entities=[_mention_entity(text, mention="@someone_else")])
        assert adapter._message_mentions_bot(msg) is False

    def test_text_mention_entity_for_different_user(self):
        adapter = _make_adapter()
        msg = _message(text="hi there", entities=[_text_mention_entity(0, 2, user_id=12345)])
        assert adapter._message_mentions_bot(msg) is False

    def test_malformed_entity_with_negative_offset(self):
        adapter = _make_adapter()
        msg = _message(text="@hermes_bot hi",
                       entities=[SimpleNamespace(type="mention", offset=-1, length=11)])
        assert adapter._message_mentions_bot(msg) is False

    def test_malformed_entity_with_zero_length(self):
        adapter = _make_adapter()
        msg = _message(text="@hermes_bot hi",
                       entities=[SimpleNamespace(type="mention", offset=0, length=0)])
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
