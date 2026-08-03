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

    def test_minimax_cn_anthropic_suffix_replaced(self):
        assert _to_openai_base_url("https://api.minimaxi.com/anthropic") == "https://api.minimaxi.com/v1"

    def test_trailing_slash_stripped_before_replace(self):
        assert _to_openai_base_url("https://api.minimax.io/anthropic/") == "https://api.minimax.io/v1"

    def test_v1_url_unchanged(self):
        assert _to_openai_base_url("https://api.openai.com/v1") == "https://api.openai.com/v1"

    def test_openrouter_url_unchanged(self):
        assert _to_openai_base_url("https://openrouter.ai/api/v1") == "https://openrouter.ai/api/v1"

    def test_anthropic_domain_unchanged(self):
        """api.anthropic.com doesn't end with /anthropic — should be untouched."""
        assert _to_openai_base_url("https://api.anthropic.com") == "https://api.anthropic.com"

    def test_anthropic_in_subpath_unchanged(self):
        assert _to_openai_base_url("https://example.com/anthropic/extra") == "https://example.com/anthropic/extra"

    def test_empty_string(self):
        assert _to_openai_base_url("") == ""

    def test_none(self):
        assert _to_openai_base_url(None) == ""
