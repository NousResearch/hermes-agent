"""``_infer_provider_from_url`` matches a known vendor by hostname, never by substring.

The table drives context-length and catalog lookups, reasoning-effort vocabularies, attribution
headers, pricing identity and stale-route detection, so a lookalike host must stay unknown.
"""

import pytest

from agent.model_metadata import _infer_provider_from_url


@pytest.mark.parametrize("url, expected", [
    ("https://api.moonshot.ai/v1", "kimi-coding"),
    ("https://api.moonshot.cn/v1", "kimi-coding-cn"),
    ("https://open.bigmodel.cn/api/paas/v4", "zai"),
    ("https://api.minimax.io/v1", "minimax"),
    ("https://api.minimaxi.com/v1", "minimax"),
    ("https://api.minimax.chat/v1", "minimax"),
    ("https://us-central1-aiplatform.googleapis.com/v1beta1/projects/p/locations/us-central1", "vertex"),
    ("https://aiplatform.googleapis.com/v1", "vertex"),
    ("https://api.business.githubcopilot.com", "copilot"),
    ("https://eu.openrouter.ai/api/v1", "openrouter"),
    ("https://token-plan-cn.xiaomimimo.com/v1", "xiaomi"),
    ("https://api.openai.com:443/v1", "openai"),
    ("https://user:pw@api.openai.com/v1", "openai"),
    ("api.deepseek.com", "deepseek"),
    ("acp://copilot", "copilot-acp"),
])
def test_known_hosts_and_their_subdomains(url, expected):
    assert _infer_provider_from_url(url) == expected


@pytest.mark.parametrize("url", [
    "https://api.moonshot.ai.evil.example/v1",
    "https://openrouter.ai.attacker.invalid/api/v1",
    "https://notollama.com/v1",
    "https://fakechatgpt.com/backend-api",
    "https://myopencode.ai/zen/v1",
    "https://api.minimax.evil.example/v1",
    "https://mycopilot.example.com/v1",
    "https://githubcopilot.com.evil.example",
    "https://evil-aiplatform.googleapis.com.example/v1",
    "https://evil.example/api.moonshot.ai/v1",
    "http://localhost:8080/v1",
])
def test_lookalike_hosts_stay_unknown(url):
    assert _infer_provider_from_url(url) is None
