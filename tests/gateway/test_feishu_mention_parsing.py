"""Feishu inbound mention and post-payload text parsing tests.

Pure-function coverage for the adapter's mention map / hint / edge-strip,
text normalization, and post (rich-text) payload parsing helpers -
byte-verbatim split out of ``tests/gateway/test_feishu.py`` (#79971).
"""

import json
import unittest
from types import SimpleNamespace


class TestFeishuMentionMap(unittest.TestCase):

    def test_build_mentions_map_marks_self_by_open_id(self):
        from plugins.platforms.feishu.adapter import _build_mentions_map, _FeishuBotIdentity

        mention = SimpleNamespace(
            key="@_user_1",
            id=SimpleNamespace(open_id="ou_bot", user_id=""),
            name="Hermes",
        )
        ref = _build_mentions_map([mention], _FeishuBotIdentity(open_id="ou_bot"))["@_user_1"]
        self.assertTrue(ref.is_self)
        self.assertEqual(ref.open_id, "ou_bot")
        self.assertEqual(ref.name, "Hermes")


    def test_build_mentions_map_name_match_does_not_override_mismatching_open_id(self):
        """Regression: a human user whose display name matches the bot must
        NOT be flagged as self when their open_id differs. Before the fix,
        name-match fired even when open_id was present and different, causing
        their messages to be silently stripped/dropped."""
        from plugins.platforms.feishu.adapter import _build_mentions_map, _FeishuBotIdentity

        human_with_same_name = SimpleNamespace(
            key="@_user_1",
            id=SimpleNamespace(open_id="ou_human", user_id=""),
            name="Hermes Bot",
        )
        result = _build_mentions_map(
            [human_with_same_name],
            _FeishuBotIdentity(open_id="ou_bot", name="Hermes Bot"),
        )
        self.assertFalse(result["@_user_1"].is_self)

    def test_build_mentions_map_falls_back_to_name_when_bot_open_id_not_hydrated(self):
        """Regression: right after gateway startup, _hydrate_bot_identity may
        not have populated _bot_open_id yet. During that window, a mention
        carrying a real open_id should still match via name — otherwise
        @bot messages silently fail admission."""
        from plugins.platforms.feishu.adapter import _build_mentions_map, _FeishuBotIdentity

        bot_mention = SimpleNamespace(
            key="@_user_1",
            id=SimpleNamespace(open_id="ou_bot_actual", user_id=""),
            name="Hermes Bot",
        )
        # Bot identity has name but no open_id yet (hydration pending).
        result = _build_mentions_map(
            [bot_mention],
            _FeishuBotIdentity(open_id="", name="Hermes Bot"),
        )
        self.assertTrue(result["@_user_1"].is_self)


class TestFeishuMentionHint(unittest.TestCase):
    def test_hint_single_user(self):
        from plugins.platforms.feishu.adapter import FeishuMentionRef, _build_mention_hint

        refs = [FeishuMentionRef(name="Alice", open_id="ou_alice")]
        self.assertEqual(
            _build_mention_hint(refs),
            "[Mentioned: Alice (open_id=ou_alice)]",
        )

    def test_hint_multiple_users(self):
        from plugins.platforms.feishu.adapter import FeishuMentionRef, _build_mention_hint

        refs = [
            FeishuMentionRef(name="Alice", open_id="ou_alice"),
            FeishuMentionRef(name="Bob", open_id="ou_bob"),
        ]
        self.assertEqual(
            _build_mention_hint(refs),
            "[Mentioned: Alice (open_id=ou_alice), Bob (open_id=ou_bob)]",
        )


    def test_hint_filters_self_mentions(self):
        from plugins.platforms.feishu.adapter import FeishuMentionRef, _build_mention_hint

        refs = [
            FeishuMentionRef(name="Hermes", open_id="ou_bot", is_self=True),
            FeishuMentionRef(name="Alice", open_id="ou_alice"),
        ]
        self.assertEqual(
            _build_mention_hint(refs),
            "[Mentioned: Alice (open_id=ou_alice)]",
        )


    def test_hint_dedupes_repeated_user(self):
        from plugins.platforms.feishu.adapter import FeishuMentionRef, _build_mention_hint

        refs = [
            FeishuMentionRef(name="Alice", open_id="ou_alice"),
            FeishuMentionRef(name="Alice", open_id="ou_alice"),
            FeishuMentionRef(name="Bob", open_id="ou_bob"),
        ]
        self.assertEqual(
            _build_mention_hint(refs),
            "[Mentioned: Alice (open_id=ou_alice), Bob (open_id=ou_bob)]",
        )


class TestFeishuStripLeadingSelf(unittest.TestCase):
    def _make_refs(self, *, self_name="Hermes", other_name=None):
        from plugins.platforms.feishu.adapter import FeishuMentionRef

        refs = [FeishuMentionRef(name=self_name, open_id="ou_bot", is_self=True)]
        if other_name:
            refs.append(FeishuMentionRef(name=other_name, open_id="ou_alice"))
        return refs


    def test_stops_at_first_non_self_token(self):
        from plugins.platforms.feishu.adapter import _strip_edge_self_mentions

        result = _strip_edge_self_mentions(
            "@Hermes @Alice make a group", self._make_refs(other_name="Alice")
        )
        self.assertEqual(result, "@Alice make a group")


    def test_strips_trailing_self_with_terminal_punct(self):
        from plugins.platforms.feishu.adapter import _strip_edge_self_mentions

        # Terminal punct after the mention — strip the mention, keep the punct.
        result = _strip_edge_self_mentions("look up docs @Hermes.", self._make_refs())
        self.assertEqual(result, "look up docs.")

    def test_preserves_trailing_self_before_non_terminal_char(self):
        from plugins.platforms.feishu.adapter import _strip_edge_self_mentions

        # Non-terminal char (here a Chinese particle) follows — preserve.
        result = _strip_edge_self_mentions(
            "please don't @Hermes anymore", self._make_refs()
        )
        self.assertEqual(result, "please don't @Hermes anymore")


    def test_returns_input_when_no_self_refs(self):
        from plugins.platforms.feishu.adapter import _strip_edge_self_mentions, FeishuMentionRef

        refs = [FeishuMentionRef(name="Alice", open_id="ou_alice")]
        self.assertEqual(_strip_edge_self_mentions("@Alice hi", refs), "@Alice hi")


class TestFeishuNormalizeText(unittest.TestCase):

    def test_renders_self_mention_with_name(self):
        from plugins.platforms.feishu.adapter import _normalize_feishu_text, FeishuMentionRef

        refs = {"@_user_1": FeishuMentionRef(name="Hermes", open_id="ou_bot", is_self=True)}
        self.assertEqual(
            _normalize_feishu_text("stop pinging @_user_1 please", refs),
            "stop pinging @Hermes please",
        )

    def test_at_all_rendered_as_english_literal(self):
        from plugins.platforms.feishu.adapter import _normalize_feishu_text

        self.assertEqual(_normalize_feishu_text("@_all notice", None), "@all notice")


class TestFeishuPostMentionParsing(unittest.TestCase):
    def test_post_at_tag_renders_via_mentions_map(self):
        """Post <at>.user_id is a placeholder ('@_user_N'); the real display
        name comes from the mentions_map lookup. Confirmed via live
        im.v1.message.get payload."""
        from plugins.platforms.feishu.adapter import parse_feishu_post_payload, FeishuMentionRef

        payload = {
            "en_us": {
                "content": [[
                    {"tag": "at", "user_id": "@_user_1", "user_name": "ignored"},
                    {"tag": "text", "text": " hello"},
                ]]
            }
        }
        mentions_map = {
            "@_user_1": FeishuMentionRef(name="Alice", open_id="ou_alice"),
        }
        result = parse_feishu_post_payload(payload, mentions_map=mentions_map)
        self.assertEqual(result.text_content, "@Alice hello")


class TestFeishuPostFileParsing(unittest.TestCase):
    def test_collects_valid_top_level_files_and_dedupes_inline_refs(self):
        from plugins.platforms.feishu.adapter import parse_feishu_post_payload

        result = parse_feishu_post_payload({
            "title": "",
            "content": [[
                {"tag": "text", "text": "Review these"},
                {"tag": "file", "file_key": "file_1", "file_name": "first.md"},
            ]],
            "files": [
                {"file_key": "file_1", "file_name": "first.md", "is_folder": False},
                {"file_key": "file_2", "file_name": "second.txt", "is_folder": False},
                {"file_key": "folder_1", "file_name": "folder", "is_folder": True},
                {"file_name": "missing-key.md", "is_folder": False},
            ],
        })

        self.assertEqual([ref.file_key for ref in result.media_refs], ["file_1", "file_2"])
        self.assertEqual(
            result.text_content,
            "Review these[Attachment: first.md]\n[Attachment: second.txt]",
        )


class TestFeishuPostTextIsNotMarkdownEscaped(unittest.TestCase):
    def test_text_elements_keep_markdown_characters_and_style_wrappers(self):
        """Inbound post text reaches the model verbatim (no backslash escapes) while the
        structured style flags still render as markdown (#9816)."""
        from plugins.platforms.feishu.adapter import parse_feishu_post_payload

        payload = {"zh_cn": {"content": [[
            {"tag": "text", "text": "run `print('hi')` for **emphasis** [x](y)"},
            {"tag": "text", "text": "strong", "style": {"bold": True}},
        ]]}}
        text = parse_feishu_post_payload(payload).text_content
        self.assertIn("run `print('hi')` for **emphasis** [x](y)", text)
        self.assertIn("**strong**", text)
        self.assertNotIn("\\", text)


class TestFeishuNormalizeWithMentions(unittest.TestCase):
    def test_text_message_renders_mention_by_name(self):
        from plugins.platforms.feishu.adapter import normalize_feishu_message, _FeishuBotIdentity

        mention = SimpleNamespace(
            key="@_user_1",
            id=SimpleNamespace(open_id="ou_alice", user_id=""),
            name="Alice",
        )
        normalized = normalize_feishu_message(
            message_type="text",
            raw_content=json.dumps({"text": "@_user_1 hello"}),
            mentions=[mention],
            bot=_FeishuBotIdentity(open_id="ou_bot"),
        )
        self.assertEqual(normalized.text_content, "@Alice hello")
        self.assertEqual(len(normalized.mentions), 1)
        self.assertEqual(normalized.mentions[0].open_id, "ou_alice")
        self.assertFalse(normalized.mentions[0].is_self)


    def test_post_message_marks_self_via_mentions_map_lookup(self):
        """Real Feishu post: <at user_id="@_user_N"> + top-level mentions array
        resolves to open_id via placeholder lookup, not direct tag fields."""
        from plugins.platforms.feishu.adapter import normalize_feishu_message, _FeishuBotIdentity

        raw = json.dumps({
            "en_us": {
                "content": [
                    [
                        {"tag": "at", "user_id": "@_user_1", "user_name": "Hermes"},
                        {"tag": "text", "text": " check this"},
                    ]
                ]
            }
        })
        bot_mention = SimpleNamespace(
            key="@_user_1",
            id=SimpleNamespace(open_id="ou_bot", user_id=""),
            name="Hermes",
        )
        normalized = normalize_feishu_message(
            message_type="post",
            raw_content=raw,
            mentions=[bot_mention],
            bot=_FeishuBotIdentity(open_id="ou_bot"),
        )
        self.assertEqual(len(normalized.mentions), 1)
        self.assertTrue(normalized.mentions[0].is_self)
        self.assertEqual(normalized.mentions[0].open_id, "ou_bot")


class TestFeishuPostMentionsBot(unittest.TestCase):
    def _build_adapter(self, bot_open_id="ou_bot", bot_user_id="", bot_name=""):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter.__new__(FeishuAdapter)
        adapter._bot_open_id = bot_open_id
        adapter._bot_user_id = bot_user_id
        adapter._bot_name = bot_name
        return adapter

    def test_post_mentions_bot_uses_is_self_flag(self):
        from plugins.platforms.feishu.adapter import FeishuMentionRef

        adapter = self._build_adapter()
        self.assertTrue(
            adapter._post_mentions_bot(
                [FeishuMentionRef(name="Hermes", open_id="ou_bot", is_self=True)]
            )
        )
        self.assertFalse(
            adapter._post_mentions_bot(
                [FeishuMentionRef(name="Alice", open_id="ou_alice")]
            )
        )
