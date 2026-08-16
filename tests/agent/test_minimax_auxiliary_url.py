"""Tests for Anthropic-to-OpenAI auxiliary URL normalization.

MiniMax and MiniMax-CN set inference_base_url to the /anthropic path.
The auxiliary client uses the OpenAI SDK, which needs /v1 instead. ZAI's
Anthropic endpoint belongs to Coding Plan, whose OpenAI-compatible peer is the
separately billed /api/coding/paas/v4 endpoint.
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

    def test_zai_anthropic_routes_to_coding_plan_openai_endpoint(self):
        assert (
            _to_openai_base_url("https://api.z.ai/api/anthropic")
            == "https://api.z.ai/api/coding/paas/v4"
        )

    def test_bigmodel_anthropic_routes_to_coding_plan_openai_endpoint(self):
        assert (
            _to_openai_base_url("https://open.bigmodel.cn/api/anthropic")
            == "https://open.bigmodel.cn/api/coding/paas/v4"
        )

    def test_bigmodel_marker_in_path_does_not_false_positive(self):
        """Host-anchored matching: 'bigmodel' in the path must not trigger rewrite."""
        url = "https://gateway.example.com/proxy/bigmodel-fallback/anthropic"
        assert _to_openai_base_url(url) == url

    def test_zai_marker_in_path_does_not_false_positive(self):
        url = "https://gateway.example.com/api.z.ai-mirror/anthropic"
        assert _to_openai_base_url(url) == url

    def test_kimi_coding_host_rewritten(self):
        assert (
            _to_openai_base_url("https://api.kimi.com/coding")
            == "https://api.kimi.com/coding/v1"
        )

    def test_kimi_marker_in_path_does_not_false_positive(self):
        url = "https://gateway.example.com/some/api.kimi.com-proxy/coding"
        assert _to_openai_base_url(url) == url

    def test_none(self):
        assert _to_openai_base_url(None) == ""
