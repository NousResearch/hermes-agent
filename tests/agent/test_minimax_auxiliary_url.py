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

    def test_none(self):
        assert _to_openai_base_url(None) == ""
