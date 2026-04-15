"""Tests for hermes_cli/provider_native_tools.py.

Covers the registry shape, active-provider lookup (incl. alias
normalisation), URL derivation, the setup-time defaults hook (override
preservation, idempotency), and the user-facing summary.

Uses MiniMax (the first registered provider) as the live fixture.
Adding another provider to ``NATIVE_TOOLS_BY_PROVIDER`` should let the
generic suite continue to pass — the only provider-specific assertions
are on the MiniMax phrasing in ``describe_changes``.
"""

from __future__ import annotations

import pytest

from hermes_cli.provider_native_tools import (
    NATIVE_TOOLS_BY_PROVIDER,
    _active_provider_id,
    active_provider_api_root,
    apply_provider_native_tool_defaults,
    describe_changes,
    get_native_tools,
    minimax_endpoint_and_key,
    native_credential_present,
    provider_has_native_tool,
)


class TestRegistryShape:
    def test_minimax_registered(self):
        assert "minimax" in NATIVE_TOOLS_BY_PROVIDER
        assert "minimax-cn" in NATIVE_TOOLS_BY_PROVIDER

    def test_minimax_capabilities(self):
        assert set(NATIVE_TOOLS_BY_PROVIDER["minimax"]) == {"tts", "image_gen", "vision"}
        assert set(NATIVE_TOOLS_BY_PROVIDER["minimax-cn"]) == {"tts", "image_gen", "vision"}

    def test_no_unknown_tool_categories(self):
        valid = {"tts", "image_gen", "vision"}
        for provider, tools in NATIVE_TOOLS_BY_PROVIDER.items():
            unknown = set(tools) - valid
            assert not unknown, (
                f"provider {provider!r} declares unknown tool categories {unknown}"
            )


class TestActiveProviderId:
    def test_dict_shape(self):
        assert _active_provider_id({"model": {"provider": "minimax"}}) == "minimax"

    def test_dict_shape_cn(self):
        assert _active_provider_id({"model": {"provider": "minimax-cn"}}) == "minimax-cn"

    def test_case_insensitive(self):
        assert _active_provider_id({"model": {"provider": "MiniMax"}}) == "minimax"

    def test_alias_normalised(self):
        assert _active_provider_id({"model": {"provider": "minimax-china"}}) == "minimax-cn"
        assert _active_provider_id({"model": {"provider": "minimax_cn"}}) == "minimax-cn"

    def test_legacy_string_shape(self):
        assert _active_provider_id({"model": "minimax/MiniMax-M2.7"}) == "minimax"

    def test_unset(self):
        assert _active_provider_id({}) == ""
        assert _active_provider_id({"model": {}}) == ""
        assert _active_provider_id({"model": {"default": "M2.7"}}) == ""


class TestQueryHelpers:
    def test_get_native_tools_minimax(self):
        out = get_native_tools({"model": {"provider": "minimax"}})
        assert set(out) == {"tts", "image_gen", "vision"}

    def test_get_native_tools_unregistered(self):
        assert get_native_tools({"model": {"provider": "anthropic"}}) == ()

    def test_get_native_tools_unset(self):
        assert get_native_tools({}) == ()

    @pytest.mark.parametrize("tool", ["tts", "image_gen", "vision"])
    def test_provider_has_native_tool_minimax(self, tool):
        assert provider_has_native_tool(tool, {"model": {"provider": "minimax"}}) is True

    def test_provider_has_native_tool_unregistered(self):
        assert provider_has_native_tool("tts", {"model": {"provider": "anthropic"}}) is False

    def test_unknown_tool(self):
        assert provider_has_native_tool("music_gen", {"model": {"provider": "minimax"}}) is False


class TestApiRoot:
    def test_strips_anthropic_suffix(self):
        cfg = {"model": {"base_url": "https://api.minimax.io/anthropic"}}
        assert active_provider_api_root(cfg) == "https://api.minimax.io"

    def test_strips_anthropic_suffix_with_trailing_slash(self):
        cfg = {"model": {"base_url": "https://api.minimax.io/anthropic/"}}
        assert active_provider_api_root(cfg) == "https://api.minimax.io"

    def test_cn_endpoint(self):
        cfg = {"model": {"base_url": "https://api.minimaxi.com/anthropic"}}
        assert active_provider_api_root(cfg) == "https://api.minimaxi.com"

    def test_no_anthropic_suffix(self):
        cfg = {"model": {"base_url": "https://api.openai.com/v1"}}
        assert active_provider_api_root(cfg) == "https://api.openai.com/v1"

    def test_unset(self):
        assert active_provider_api_root({}) == ""
        assert active_provider_api_root({"model": {}}) == ""


class TestApplyDefaults:
    def test_noop_when_provider_not_native(self):
        config = {"model": {"provider": "anthropic"}}
        assert apply_provider_native_tool_defaults(config) == set()
        assert "tts" not in config

    def test_noop_when_no_provider(self):
        assert apply_provider_native_tool_defaults({}) == set()

    def test_minimax_writes_tts_image_reports_vision(self):
        config = {"model": {"provider": "minimax"}}
        changed = apply_provider_native_tool_defaults(config)
        assert changed == {"tts", "image_gen", "vision"}
        assert config["tts"]["provider"] == "minimax"
        assert config["image_gen"]["provider"] == "minimax"
        # Vision is reported but no value persists — runtime dispatch handles it.
        assert config.get("auxiliary", {}).get("vision") in (None, {})

    def test_explicit_tts_preserved(self):
        config = {
            "model": {"provider": "minimax"},
            "tts": {"provider": "elevenlabs"},
        }
        changed = apply_provider_native_tool_defaults(config)
        assert "tts" not in changed
        assert config["tts"]["provider"] == "elevenlabs"
        assert "image_gen" in changed

    def test_edge_default_overridden(self):
        config = {"model": {"provider": "minimax"}, "tts": {"provider": "edge"}}
        changed = apply_provider_native_tool_defaults(config)
        assert "tts" in changed
        assert config["tts"]["provider"] == "minimax"

    def test_explicit_fal_overridden(self):
        config = {"model": {"provider": "minimax"}, "image_gen": {"provider": "fal"}}
        changed = apply_provider_native_tool_defaults(config)
        assert "image_gen" in changed
        assert config["image_gen"]["provider"] == "minimax"

    def test_explicit_third_party_image_preserved(self):
        config = {
            "model": {"provider": "minimax"},
            "image_gen": {"provider": "stability"},
        }
        changed = apply_provider_native_tool_defaults(config)
        assert "image_gen" not in changed
        assert config["image_gen"]["provider"] == "stability"

    def test_explicit_vision_provider_preserved(self):
        config = {
            "model": {"provider": "minimax"},
            "auxiliary": {"vision": {"provider": "openrouter"}},
        }
        changed = apply_provider_native_tool_defaults(config)
        assert "tts" in changed
        assert config["auxiliary"]["vision"]["provider"] == "openrouter"

    def test_idempotent(self):
        config = {"model": {"provider": "minimax"}}
        first = apply_provider_native_tool_defaults(config)
        assert first == {"tts", "image_gen", "vision"}
        second = apply_provider_native_tool_defaults(config)
        assert second == set()

    def test_cn_provider_same_behaviour(self):
        config = {"model": {"provider": "minimax-cn"}}
        changed = apply_provider_native_tool_defaults(config)
        assert changed == {"tts", "image_gen", "vision"}
        assert config["tts"]["provider"] == "minimax-cn"

    def test_alias_provider_resolved(self):
        config = {"model": {"provider": "minimax-china"}}
        changed = apply_provider_native_tool_defaults(config)
        assert "tts" in changed
        assert config["tts"]["provider"] == "minimax-cn"

    def test_other_auxiliary_keys_preserved(self):
        config = {
            "model": {"provider": "minimax"},
            "auxiliary": {"compression": {"provider": "openai"}},
        }
        apply_provider_native_tool_defaults(config)
        assert config["auxiliary"]["compression"]["provider"] == "openai"


class TestDescribeChanges:
    def test_empty_set(self):
        out = describe_changes(set(), {"model": {"provider": "minimax"}})
        assert "No changes" in out

    def test_minimax_three_tools(self):
        out = describe_changes(
            {"tts", "image_gen", "vision"},
            {"model": {"provider": "minimax"}},
        )
        # Provider name omitted from values (wizard's success line names it).
        assert "TTS" in out and "image-01" in out and "MiniMax-VL-01" in out
        assert out.count("•") == 3

    def test_minimax_cn_shares_minimax_phrasing(self):
        out_int = describe_changes({"tts"}, {"model": {"provider": "minimax"}})
        out_cn = describe_changes({"tts"}, {"model": {"provider": "minimax-cn"}})
        assert out_int == out_cn

    def test_unknown_provider_falls_back_to_generic(self):
        out = describe_changes({"tts"}, {"model": {"provider": "imaginary"}})
        assert "tts" in out


class TestNativeCredentialPresent:
    def test_false_when_provider_not_native(self):
        assert native_credential_present(
            "image_gen", {"model": {"provider": "anthropic"}},
        ) is False

    def test_false_when_tool_not_served_by_provider(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-test")
        assert native_credential_present(
            "search", {"model": {"provider": "minimax"}},
        ) is False  # minimax declares tts/image_gen/vision — not "search"

    def test_true_when_provider_native_and_key_set(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-test")
        assert native_credential_present(
            "image_gen", {"model": {"provider": "minimax"}},
        ) is True

    def test_false_when_provider_native_but_no_key(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        monkeypatch.delenv("MINIMAX_CN_API_KEY", raising=False)
        assert native_credential_present(
            "image_gen", {"model": {"provider": "minimax"}},
        ) is False


class TestMinimaxEndpointAndKey:
    def test_non_minimax_returns_empty_tuple(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-test")
        assert minimax_endpoint_and_key(
            "/v1/t2a_v2", {"model": {"provider": "anthropic"}},
        ) == ("", "")

    def test_derives_url_and_key(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-intl")
        url, key = minimax_endpoint_and_key("/v1/t2a_v2", {
            "model": {"provider": "minimax", "base_url": "https://api.minimax.io/anthropic"},
        })
        assert url == "https://api.minimax.io/v1/t2a_v2"
        assert key == "sk-intl"

    def test_cn_picks_cn_host_and_key(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-intl")
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "sk-cn")
        url, key = minimax_endpoint_and_key("/v1/t2a_v2", {
            "model": {"provider": "minimax-cn", "base_url": "https://api.minimaxi.com/anthropic"},
        })
        assert url == "https://api.minimaxi.com/v1/t2a_v2"
        assert key == "sk-cn"

    def test_empty_when_base_url_missing(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-test")
        assert minimax_endpoint_and_key(
            "/v1/t2a_v2", {"model": {"provider": "minimax"}},
        ) == ("", "")

    def test_empty_when_key_missing(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        monkeypatch.delenv("MINIMAX_CN_API_KEY", raising=False)
        assert minimax_endpoint_and_key("/v1/t2a_v2", {
            "model": {"provider": "minimax", "base_url": "https://api.minimax.io/anthropic"},
        }) == ("", "")
