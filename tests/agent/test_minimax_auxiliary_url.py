"""Tests for MiniMax auxiliary client URL normalization.

MiniMax and MiniMax-CN set inference_base_url to the /anthropic path.
The auxiliary client uses the OpenAI SDK, which needs /v1 instead.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agent.auxiliary_client import _to_openai_base_url


class TestToOpenaiBaseUrl:
    def test_minimax_global_anthropic_suffix_replaced(self):
        assert _to_openai_base_url("https://api.minimax.io/anthropic") == "https://api.minimax.io/v1"

    def test_minimax_chat_host_rewritten(self):
        assert _to_openai_base_url("https://api.minimax.chat/anthropic") == "https://api.minimax.chat/v1"

    def test_anthropic_only_custom_gateway_kept(self):
        """Alibaba Bailian Token Plan and similar have no sibling /v1 (#83642)."""
        url = "https://token-plan.cn-beijing.maas.aliyuncs.com/apps/anthropic"
        assert _to_openai_base_url(url) == url

    def test_marker_in_path_does_not_false_positive(self):
        """Host-anchored matching: a marker in the path must not trigger rewrite."""
        url = "https://gateway.example.com/api.minimax.io/anthropic"
        assert _to_openai_base_url(url) == url

    def test_minimax_cn_api_prefix_host_rewritten(self):
        assert _to_openai_base_url("https://api.minimaxi.com/anthropic") == "https://api.minimaxi.com/v1"

    def test_lookalike_host_suffix_not_matched(self):
        """evil-minimax.io.example.com must not match the minimax.io suffix."""
        url = "https://minimax.io.evil.example.com/anthropic"
        assert _to_openai_base_url(url) == url

    def test_none(self):
        assert _to_openai_base_url(None) == ""
