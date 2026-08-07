"""Tests for auxiliary client URL normalization (_to_openai_base_url).

Several providers expose their Anthropic Messages wire on a ``/anthropic``
path and their OpenAI chat-completions wire on a DIFFERENT path. The
auxiliary client uses the OpenAI SDK, so it must be pointed at the
provider's real OpenAI surface — the generic ``/v1`` rewrite is wrong for
providers whose OpenAI wire lives elsewhere:

  - MiniMax / MiniMax-CN        → /v1
  - ZAI (open.bigmodel.cn)      → /paas/v4
  - token-plan / Aliyun MaaS    → /compatible-mode/v1   (NOT /v1 — 404)
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agent.auxiliary_client import _to_openai_base_url


class TestToOpenaiBaseUrl:
    def test_minimax_global_anthropic_suffix_replaced(self):
        assert _to_openai_base_url("https://api.minimax.io/anthropic") == "https://api.minimax.io/v1"

    def test_tokenplan_anthropic_suffix_maps_to_compatible_mode(self):
        # token-plan / Aliyun MaaS exposes OpenAI wire at /compatible-mode/v1,
        # NOT /v1 — the generic rewrite produced a 404 for MoA reference models.
        assert _to_openai_base_url(
            "https://token-plan.cn-beijing.maas.aliyuncs.com/apps/anthropic"
        ) == "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"

    def test_aliyun_maas_anthropic_suffix_no_apps_segment(self):
        # Same provider family without the /apps segment — the base path must
        # survive intact (no accidental character-strip of 'apps').
        assert _to_openai_base_url(
            "https://dashscope.aliyuncs.com/anthropic"
        ) == "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def test_zai_anthropic_suffix_maps_to_paas_v4(self):
        assert _to_openai_base_url(
            "https://open.bigmodel.cn/api/anthropic"
        ) == "https://open.bigmodel.cn/api/paas/v4"

    def test_kimi_coding_suffix_gets_v1_appended(self):
        assert _to_openai_base_url("https://api.kimi.com/coding") == "https://api.kimi.com/coding/v1"

    def test_plain_openai_url_untouched(self):
        assert _to_openai_base_url("https://api.openai.com/v1") == "https://api.openai.com/v1"

    def test_none(self):
        assert _to_openai_base_url(None) == ""
