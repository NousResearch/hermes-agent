"""Claude Haiku 5.5 has a 1M context window, like the rest of the Claude 5 family; Haiku 4.5 is 200K
(https://platform.claude.com/docs/en/about-claude/models/overview). Without a live lookup it must not
land on the 200K ``claude`` catch-all (#154406), on the Claude API or on Bedrock."""

import pytest

from agent.bedrock_adapter import get_bedrock_context_length
from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS, _longest_key_match, get_model_context_length


@pytest.mark.parametrize("model", ["claude-haiku-5-5", "claude-haiku-5.5", "claude-haiku-5"])
def test_haiku_5_offline_window_is_1m(model):
    assert _longest_key_match(DEFAULT_CONTEXT_LENGTHS, model)[1] == 1_000_000


@pytest.mark.parametrize("model", ["claude-haiku-4-5", "claude-haiku-4-5-20251001", "claude-3-5-haiku-20241022"])
def test_older_haiku_keeps_the_200k_window(model):
    assert _longest_key_match(DEFAULT_CONTEXT_LENGTHS, model)[1] == 200_000


def test_custom_anthropic_endpoint_without_window_metadata_resolves_1m(monkeypatch):
    """The reported route: a custom endpoint whose /models rows carry no window, no config override."""
    import agent.model_metadata as mm
    monkeypatch.setattr(mm, "fetch_endpoint_model_metadata", lambda *a, **k: {"claude-haiku-5-5": {}})
    monkeypatch.setattr(mm, "get_cached_context_length", lambda *a, **k: None)
    assert get_model_context_length("claude-haiku-5-5", base_url="https://relay.example/v1", provider="custom") == 1_000_000


@pytest.mark.parametrize("scope", ["", "us.", "global."])
def test_haiku_5_bedrock_window_is_1m(scope):
    assert get_bedrock_context_length(f"{scope}anthropic.claude-haiku-5-5", probe=False) == 1_000_000


# Vendor-documented windows for every row audited in #154406 (each cited in the PR body): a custom
# endpoint whose /models carries no window lands on the hardcoded table, which must match the vendor.
_VENDOR_WINDOWS = {
    "claude-haiku-5-5": 1_000_000,  # platform.claude.com models overview
    "gpt-5.6": 1_050_000,  # developers.openai.com: the gpt-5.6 alias routes to Sol
    "solar-mini4": 524_288,  # console.upstage.ai solar-mini-4: 512K
    "grok-4.7": 500_000,  # docs.x.ai grok-4.7
    "grok-4.20": 1_000_000, "grok-4.20-multi-agent": 1_000_000,  # docs.x.ai grok-4.20 pages
    "glm-5.3": 1_000_000, "glm-5.3-flash": 1_000_000, "glm-5.3-flashx": 1_000_000,  # docs.bigmodel.cn
}


@pytest.mark.parametrize(("model", "window"), sorted(_VENDOR_WINDOWS.items()))
def test_offline_window_matches_vendor_docs(model, window, monkeypatch):
    import agent.model_metadata as mm
    monkeypatch.setattr(mm, "fetch_endpoint_model_metadata", lambda *a, **k: {model: {}})
    monkeypatch.setattr(mm, "get_cached_context_length", lambda *a, **k: None)
    assert get_model_context_length(model, base_url="https://relay.example/v1", provider="custom") == window


@pytest.mark.parametrize(("model", "persisted"), [("claude-haiku-5-5", 200_000), ("grok-4.7", 256_000), ("gpt-5.6", 400_000)])
def test_catch_all_value_persisted_before_the_fix_is_dropped(model, persisted):
    """Older builds cached the family catch-all; step 1 must not pin it after the specific key lands."""
    from agent.model_metadata import _stale_pre_catalog_cache_entry
    assert _stale_pre_catalog_cache_entry(model, persisted)
