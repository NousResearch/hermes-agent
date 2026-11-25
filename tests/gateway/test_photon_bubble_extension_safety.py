"""Tests for AirHermes Custom Extension Registration Safety and Concurrency (Part B).

Verifies:
- Idempotent registration (calling register 1x vs 2x produces identical state)
- Refusal of silent overwrites for different handlers
- Registry persistence across parser instances and repeated turns (finish does not clear registry)
- Concurrency and isolation across multiple chat_ids / turn_ids
- Safe public mode behavior after clear_custom_extensions()
- No duplicate cards, polls, or attachment dispatches
"""

import re
import sys
import tempfile
import unittest
from pathlib import Path



from gateway.bubble_actions import ImageAttachmentAction, PollAction, TextBubble
from gateway.bubble_parser import BubbleParser
from gateway.attachment_registry import registry


# Self-contained sample extension handlers for registration safety verification
_CUSTOM_CARD_RE = re.compile(
    r'<(?:custom_card|widget)(?:\s+(?P<attrs>[^\s>][^>]*?))?\s*/>',
    re.IGNORECASE,
)
_CUSTOM_SHEET_RE = re.compile(
    r'<custom_sheet(?:\s+(?P<attrs>[^>]*?))?\s*>(?P<body>.*?)</custom_sheet>',
    re.IGNORECASE | re.DOTALL,
)


def _custom_card_handler(parser, match, pre_clean, has_tags):
    actions = []
    if pre_clean:
        actions.extend(parser._emit_actions(pre_clean))
    _start, end = match.span()
    parser._buffer = parser._buffer[end:]

    # Stage dummy artifact
    tmp_path = getattr(parser, "_test_dummy_img", None)
    if tmp_path:
        rec, _err = registry.stage_and_authorize(
            chat_id=parser.chat_id,
            source_path=tmp_path,
            producer="test_custom_card",
            turn_id=parser.turn_id,
        )
        if rec:
            actions.append(ImageAttachmentAction(staged_path=rec.staged_path, token=rec.token))
    return actions


def _custom_sheet_handler(parser, match, pre_clean, has_tags):
    actions = []
    if pre_clean:
        actions.extend(parser._emit_actions(pre_clean))
    _start, end = match.span()
    parser._buffer = parser._buffer[end:]
    parser._custom_options = ["Option 1", "Option 2"]
    return actions


def _custom_finish_hook(parser):
    actions = []
    opts = getattr(parser, "_custom_options", None)
    if opts and not getattr(parser, "_has_emitted_poll", False):
        actions.append(PollAction(title="Custom Finish Poll", options=tuple(opts)))
    parser._custom_options = None
    return actions


def register_test_extensions(parser_cls):
    parser_cls.register_custom_action("custom_card", _CUSTOM_CARD_RE, _custom_card_handler)
    parser_cls.register_custom_action("custom_sheet", _CUSTOM_SHEET_RE, _custom_sheet_handler)
    parser_cls.register_custom_finish_hook(_custom_finish_hook)


class TestExtensionRegistrationSafety(unittest.TestCase):
    def setUp(self):
        BubbleParser.clear_custom_extensions()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dummy_img = Path(self.temp_dir.name) / "dummy.png"
        self.dummy_img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 1000)

    def tearDown(self):
        BubbleParser.clear_custom_extensions()
        self.temp_dir.cleanup()

    def test_idempotent_registration_1x_vs_2x(self):
        """Registering extensions once vs twice yields exactly 2 actions, 1 hook, and identical outputs."""
        register_test_extensions(BubbleParser)
        count_actions_1x = len(BubbleParser._custom_action_handlers)
        count_hooks_1x = len(BubbleParser._custom_finish_hooks)

        self.assertEqual(count_actions_1x, 2)
        self.assertEqual(count_hooks_1x, 1)

        # Register a second time
        register_test_extensions(BubbleParser)
        count_actions_2x = len(BubbleParser._custom_action_handlers)
        count_hooks_2x = len(BubbleParser._custom_finish_hooks)

        self.assertEqual(count_actions_2x, 2)
        self.assertEqual(count_hooks_2x, 1)

        # Verify parsing produces exactly 1 image action, no duplicates
        parser = BubbleParser(chat_id="chat-idemp", turn_id="turn-idemp")
        parser._test_dummy_img = str(self.dummy_img)
        actions = parser.feed_actions('<custom_card title="Test" />')
        actions.extend(parser.finish_actions())

        img_actions = [a for a in actions if isinstance(a, ImageAttachmentAction)]
        self.assertEqual(len(img_actions), 1)

    def test_refuse_silent_overwrite_of_different_handler(self):
        """Registering a different handler under an existing name refuses overwrite."""
        register_test_extensions(BubbleParser)
        original_handler = BubbleParser._custom_action_handlers["custom_card"][1]

        def rogue_handler(parser, match, pre, has_tags):
            return []

        # Attempt overwrite
        BubbleParser.register_custom_action("custom_card", re.compile(r"<rogue>"), rogue_handler)

        # Verify original handler is preserved
        current_handler = BubbleParser._custom_action_handlers["custom_card"][1]
        self.assertIs(current_handler, original_handler)

    def test_parser_finish_does_not_unregister_extensions(self):
        """Turn cleanup via finish() must not clear class-level extensions."""
        register_test_extensions(BubbleParser)

        # Turn 1
        parser1 = BubbleParser(chat_id="chat-1", turn_id="turn-1")
        parser1._test_dummy_img = str(self.dummy_img)
        actions1 = parser1.feed_actions('<custom_card title="Turn1" />')
        actions1.extend(parser1.finish_actions())
        self.assertEqual(len([a for a in actions1 if isinstance(a, ImageAttachmentAction)]), 1)

        # Verify class-level extensions are still intact
        self.assertEqual(len(BubbleParser._custom_action_handlers), 2)
        self.assertEqual(len(BubbleParser._custom_finish_hooks), 1)

        # Turn 2 with new parser instance
        parser2 = BubbleParser(chat_id="chat-1", turn_id="turn-2")
        parser2._test_dummy_img = str(self.dummy_img)
        actions2 = parser2.feed_actions('<custom_card title="Turn2" />')
        actions2.extend(parser2.finish_actions())
        self.assertEqual(len([a for a in actions2 if isinstance(a, ImageAttachmentAction)]), 1)

    def test_concurrent_conversations_isolation(self):
        """Multiple concurrent parser instances with different chat_ids do not cross-contaminate."""
        register_test_extensions(BubbleParser)

        parser_a = BubbleParser(chat_id="chat-A", turn_id="turn-A1")
        parser_b = BubbleParser(chat_id="chat-B", turn_id="turn-B1")
        parser_a._test_dummy_img = str(self.dummy_img)
        parser_b._test_dummy_img = str(self.dummy_img)

        # Interleaved feeds
        actions_a1 = parser_a.feed_actions('<custom_card title="A" />')
        actions_b1 = parser_b.feed_actions('<custom_card title="B" />')

        actions_a2 = parser_a.finish_actions()
        actions_b2 = parser_b.finish_actions()

        all_a = actions_a1 + actions_a2
        all_b = actions_b1 + actions_b2

        img_a = [a for a in all_a if isinstance(a, ImageAttachmentAction)]
        img_b = [a for a in all_b if isinstance(a, ImageAttachmentAction)]

        self.assertEqual(len(img_a), 1)
        self.assertEqual(len(img_b), 1)

        # Verify tokens/staged paths in AttachmentRegistry belong to respective chats
        rec_a = registry.get_record(img_a[0].token)
        rec_b = registry.get_record(img_b[0].token)

        self.assertIsNotNone(rec_a)
        self.assertIsNotNone(rec_b)
        self.assertEqual(rec_a.chat_id, "chat-A")
        self.assertEqual(rec_a.turn_id, "turn-A1")
        self.assertEqual(rec_b.chat_id, "chat-B")
        self.assertEqual(rec_b.turn_id, "turn-B1")

    def test_public_mode_without_registration(self):
        """In clean public mode (no extensions registered), custom tags fall back safely."""
        self.assertEqual(len(BubbleParser._custom_action_handlers), 0)
        self.assertEqual(len(BubbleParser._custom_finish_hooks), 0)

        parser = BubbleParser(chat_id="chat-pub", turn_id="turn-pub")
        content = "自定义内容测试：\n<custom_card title=\"Test\">\n内部保留文本。\n</custom_card>"
        actions = parser.feed_actions(content)
        actions.extend(parser.finish_actions())

        # No image attachment actions produced
        img_actions = [a for a in actions if isinstance(a, ImageAttachmentAction)]
        self.assertEqual(len(img_actions), 0)

        # The body text is safely preserved or degraded gracefully without throwing exception
        text_bubbles = [a for a in actions if isinstance(a, TextBubble)]
        self.assertTrue(any("内部保留文本" in b.content or "自定义内容测试" in b.content for b in text_bubbles))

    def test_no_duplicate_polls_or_cards(self):
        """Verify that personal finish hooks do not emit duplicate polls if explicit poll exists."""
        register_test_extensions(BubbleParser)

        parser = BubbleParser(chat_id="chat-poll", turn_id="turn-poll")
        content = (
            "<custom_sheet>\n"
            "| A | B |\n"
            "|---|---|\n"
            "| 1 | 2 |\n"
            "</custom_sheet>\n"
            "<poll title=\"你选哪项？\"><option>A</option><option>B</option></poll>"
        )
        actions = parser.feed_actions(content)
        actions.extend(parser.finish_actions())

        polls = [a for a in actions if isinstance(a, PollAction)]
        # Exactly 1 poll (the explicit one), no synthesized duplicate poll
        self.assertEqual(len(polls), 1)
        self.assertEqual(polls[0].title, "你选哪项？")


if __name__ == "__main__":
    unittest.main()
