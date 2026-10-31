"""Feishu inbound text normalisation (``plugins/platforms/feishu/feishu_text.py``)."""

import unittest

from plugins.platforms.feishu.adapter import FeishuMentionRef
from plugins.platforms.feishu.feishu_text import normalize_feishu_text


class TestFeishuNormalizeText(unittest.TestCase):

    def test_renders_self_mention_with_name(self):
        refs = {"@_user_1": FeishuMentionRef(name="Hermes", open_id="ou_bot", is_self=True)}
        self.assertEqual(
            normalize_feishu_text("stop pinging @_user_1 please", refs),
            "stop pinging @Hermes please",
        )

    def test_at_all_rendered_as_english_literal(self):
        self.assertEqual(normalize_feishu_text("@_all notice", None), "@all notice")
