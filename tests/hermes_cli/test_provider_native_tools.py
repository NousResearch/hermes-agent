"""Tests for hermes_cli/provider_native_tools.py."""

from __future__ import annotations

import pytest

from hermes_cli.provider_native_tools import (
    active_provider_api_root,
    apply_provider_native_tool_defaults,
    describe_changes,
    endpoint_and_key,
    get_native_tools,
    provider_has_native_tool,
)

_INTL = "https://api.minimax.io/anthropic"
_INTL_UW = "https://api-uw.minimax.io/anthropic"
_CN = "https://api.minimaxi.com/anthropic"
_CN_APAC = "https://api-apac.minimaxi.com/anthropic"


def _cfg(base_url):
    return {"model": {"base_url": base_url}}


class TestApiHost:
    def test_intl_detected(self):
        assert get_native_tools(_cfg(_INTL)) != ()

    def test_intl_uw_detected(self):
        assert get_native_tools(_cfg(_INTL_UW)) != ()

    def test_cn_detected(self):
        assert get_native_tools(_cfg(_CN)) != ()

    def test_cn_apac_detected(self):
        assert get_native_tools(_cfg(_CN_APAC)) != ()

    def test_non_native(self):
        assert get_native_tools(_cfg("https://api.openai.com/v1")) == ()

    def test_unset(self):
        assert get_native_tools({}) == ()
        assert get_native_tools({"model": {}}) == ()


class TestApiRoot:
    def test_strips_anthropic_suffix(self):
        assert active_provider_api_root(_cfg(_INTL)) == "https://api.minimax.io"

    def test_strips_trailing_slash(self):
        assert active_provider_api_root(
            _cfg("https://api.minimax.io/anthropic/")
        ) == "https://api.minimax.io"

    def test_cn_endpoint(self):
        assert active_provider_api_root(_cfg(_CN)) == "https://api.minimaxi.com"

    def test_no_anthropic_suffix(self):
        cfg = _cfg("https://api.openai.com/v1")
        assert active_provider_api_root(cfg) == "https://api.openai.com/v1"

    def test_unset(self):
        assert active_provider_api_root({}) == ""


class TestNativeTools:
    @pytest.mark.parametrize("tool", ["tts", "image_gen", "vision", "video_gen", "music_gen"])
    def test_all_categories_present(self, tool):
        assert provider_has_native_tool(tool, _cfg(_INTL)) is True

    def test_unknown_tool(self):
        assert provider_has_native_tool("search", _cfg(_INTL)) is False

    def test_non_native_provider(self):
        assert provider_has_native_tool("tts", _cfg("https://api.openai.com/v1")) is False


class TestApplyDefaults:
    def test_noop_for_non_native(self):
        config = _cfg("https://api.openai.com/v1")
        assert apply_provider_native_tool_defaults(config) == set()

    def test_noop_for_empty(self):
        assert apply_provider_native_tool_defaults({}) == set()

    def test_writes_all_config_slots(self):
        config = _cfg(_INTL)
        changed = apply_provider_native_tool_defaults(config)
        assert changed == {"tts", "image_gen", "vision", "video_gen", "music_gen"}
        assert config["tts"]["provider"] == "minimax"
        assert config["image_gen"]["provider"] == "minimax"
        assert config["video_gen"]["provider"] == "minimax"
        assert config["music_gen"]["provider"] == "minimax"

    def test_explicit_tts_preserved(self):
        config = {**_cfg(_INTL), "tts": {"provider": "elevenlabs"}}
        changed = apply_provider_native_tool_defaults(config)
        assert "tts" not in changed
        assert config["tts"]["provider"] == "elevenlabs"

    def test_edge_default_overridden(self):
        config = {**_cfg(_INTL), "tts": {"provider": "edge"}}
        changed = apply_provider_native_tool_defaults(config)
        assert "tts" in changed

    def test_fal_default_overridden(self):
        config = {**_cfg(_INTL), "image_gen": {"provider": "fal"}}
        changed = apply_provider_native_tool_defaults(config)
        assert "image_gen" in changed

    def test_explicit_image_preserved(self):
        config = {**_cfg(_INTL), "image_gen": {"provider": "stability"}}
        changed = apply_provider_native_tool_defaults(config)
        assert "image_gen" not in changed

    def test_explicit_video_preserved(self):
        config = {**_cfg(_INTL), "video_gen": {"provider": "runway"}}
        changed = apply_provider_native_tool_defaults(config)
        assert "video_gen" not in changed

    def test_explicit_music_preserved(self):
        config = {**_cfg(_INTL), "music_gen": {"provider": "suno"}}
        changed = apply_provider_native_tool_defaults(config)
        assert "music_gen" not in changed

    def test_explicit_vision_preserved(self):
        config = {
            **_cfg(_INTL),
            "auxiliary": {"vision": {"provider": "openrouter"}},
        }
        changed = apply_provider_native_tool_defaults(config)
        assert "vision" not in changed

    def test_idempotent(self):
        config = _cfg(_INTL)
        first = apply_provider_native_tool_defaults(config)
        assert len(first) == 5
        second = apply_provider_native_tool_defaults(config)
        assert second == set()

    def test_cn_provider(self):
        config = _cfg(_CN)
        changed = apply_provider_native_tool_defaults(config)
        assert "tts" in changed
        assert config["tts"]["provider"] == "minimax-cn"

    def test_other_auxiliary_keys_preserved(self):
        config = {
            **_cfg(_INTL),
            "auxiliary": {"compression": {"provider": "openai"}},
        }
        apply_provider_native_tool_defaults(config)
        assert config["auxiliary"]["compression"]["provider"] == "openai"


class TestDescribeChanges:
    def test_empty(self):
        assert "No changes" in describe_changes(set(), _cfg(_INTL))

    def test_all_five(self):
        out = describe_changes(
            {"tts", "image_gen", "vision", "video_gen", "music_gen"},
            _cfg(_INTL),
        )
        assert out.count("\u2022") == 5

    def test_contains_model_names(self):
        out = describe_changes({"image_gen", "video_gen"}, _cfg(_INTL))
        assert "image-01" in out
        assert "Hailuo" in out


class TestEndpointAndKey:
    def test_derives_url(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-test")
        url, key = endpoint_and_key("/v1/t2a_v2", _cfg(_INTL))
        assert url == "https://api.minimax.io/v1/t2a_v2"
        assert key == "sk-test"

    def test_cn_host(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "sk-cn")
        url, key = endpoint_and_key("/v1/t2a_v2", _cfg(_CN))
        assert url == "https://api.minimaxi.com/v1/t2a_v2"
        assert key == "sk-cn"

    def test_non_native_empty(self):
        assert endpoint_and_key("/v1/t2a_v2", _cfg("https://api.openai.com/v1")) == ("", "")

    def test_no_key_empty(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        monkeypatch.delenv("MINIMAX_CN_API_KEY", raising=False)
        assert endpoint_and_key("/v1/t2a_v2", _cfg(_INTL)) == ("", "")


