"""Tests for the delegation fallback chain config helper.

These tests cover tools.delegate_tool_config._get_delegation_fallback_chain
(re-exported through tools.delegate_tool) applied to the delegation block
(delegation.fallback_providers / fallback_chain / fallback_model). The helper
is parsed locally for the delegation block on purpose: the shared top-level
get_fallback_chain() is left untouched.
"""

from tools.delegate_tool import _get_delegation_fallback_chain


class TestDelegationFallbackChainConfig:
    def test_empty_delegation_returns_none(self):
        assert _get_delegation_fallback_chain({}) is None
        assert _get_delegation_fallback_chain({"fallback_providers": []}) is None
        assert _get_delegation_fallback_chain(None) is None

    def test_fallback_providers_list_preserved(self):
        chain = [
            {"provider": "openrouter", "model": "gpt-4o-mini"},
            {"provider": "nous", "model": "hermes-4-405b"},
        ]
        assert _get_delegation_fallback_chain({"fallback_providers": chain}) == chain

    def test_fallback_model_single_dict_normalized(self):
        entry = {"provider": "nous", "model": "hermes-4-405b"}
        assert _get_delegation_fallback_chain({"fallback_model": entry}) == [entry]

    def test_fallback_chain_alias_honoured(self):
        chain = [{"provider": "nous", "model": "hermes-4-405b"}]
        assert _get_delegation_fallback_chain({"fallback_chain": chain}) == chain

    def test_fallback_providers_take_precedence_over_aliases(self):
        providers_chain = [{"provider": "openrouter", "model": "gpt-4o-mini"}]
        alias_chain = [{"provider": "nous", "model": "hermes-4-405b"}]
        cfg = {"fallback_providers": providers_chain, "fallback_chain": alias_chain}
        assert _get_delegation_fallback_chain(cfg) == providers_chain + alias_chain

    def test_invalid_entries_filtered(self):
        cfg = {"fallback_providers": [
            {"provider": "", "model": "x"},
            {"provider": "nous", "model": ""},
            {"provider": "nous", "model": "hermes-4-405b"},
        ]}
        assert _get_delegation_fallback_chain(cfg) == [
            {"provider": "nous", "model": "hermes-4-405b"}
        ]

    def test_duplicate_routes_deduped(self):
        entry = {"provider": "nous", "model": "hermes-4-405b"}
        cfg = {"fallback_providers": [entry], "fallback_model": dict(entry)}
        assert _get_delegation_fallback_chain(cfg) == [entry]

    def test_pinned_keys_coexist_with_chain(self):
        chain = [{"provider": "openrouter", "model": "gpt-4o-mini"}]
        cfg = {"provider": "minimax", "model": "minimax/m2", "fallback_providers": chain}
        assert _get_delegation_fallback_chain(cfg) == chain
        assert cfg["provider"] == "minimax"
