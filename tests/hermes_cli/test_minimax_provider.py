"""Unit tests for hermes_cli/minimax_provider.py.

Covers:
- ``is_minimax_provider`` for both dict and legacy string model shapes.
- ``apply_minimax_provider_defaults`` idempotency, honoring explicit overrides,
  and the change-set returned.
- ``describe_changes`` formatting.
"""

from __future__ import annotations

import pytest

from hermes_cli.minimax_provider import (
    MINIMAX_PROVIDERS,
    apply_minimax_provider_defaults,
    describe_changes,
    is_minimax_provider,
)


# ─── is_minimax_provider ─────────────────────────────────────────────────


class TestIsMiniMaxProvider:
    def test_dict_shape_international(self):
        assert is_minimax_provider({"model": {"provider": "minimax"}}) is True

    def test_dict_shape_cn(self):
        assert is_minimax_provider({"model": {"provider": "minimax-cn"}}) is True

    def test_case_insensitive(self):
        assert is_minimax_provider({"model": {"provider": "MiniMax"}}) is True
        assert is_minimax_provider({"model": {"provider": "MINIMAX"}}) is True

    def test_other_provider(self):
        assert is_minimax_provider({"model": {"provider": "anthropic"}}) is False
        assert is_minimax_provider({"model": {"provider": "openrouter"}}) is False

    def test_empty_config(self):
        assert is_minimax_provider({}) is False

    def test_string_shape_legacy(self):
        """Legacy config where model is 'provider/name' string."""
        assert is_minimax_provider({"model": "minimax/MiniMax-M2.7"}) is True
        assert is_minimax_provider({"model": "anthropic/claude-opus-4"}) is False

    def test_unset_provider(self):
        assert is_minimax_provider({"model": {"default": "M2.7"}}) is False


# ─── apply_minimax_provider_defaults ─────────────────────────────────────


class TestApplyDefaults:
    def test_noop_when_not_minimax(self):
        config = {"model": {"provider": "anthropic"}}
        changed = apply_minimax_provider_defaults(config)
        assert changed == set()
        assert "tts" not in config or config["tts"].get("provider") != "minimax"

    def test_sets_tts_image_vision_when_unset(self):
        config = {"model": {"provider": "minimax"}}
        changed = apply_minimax_provider_defaults(config)
        assert changed == {"tts", "image_gen", "vision"}
        assert config["tts"]["provider"] == "minimax"
        assert config["image_gen"]["provider"] == "minimax"
        # Vision does not persist a config value — dispatch is decided at
        # call time in tools/vision_tools.py via _use_minimax_vision().
        # "vision" appears in `changed` so the setup wizard can surface
        # it in the success summary, but the config is not modified.
        assert config.get("auxiliary", {}).get("vision") in (None, {}), \
            "vision config should remain empty (dispatch is runtime-only)"

    def test_respects_explicit_tts(self):
        config = {
            "model": {"provider": "minimax"},
            "tts": {"provider": "elevenlabs"},
        }
        changed = apply_minimax_provider_defaults(config)
        assert "tts" not in changed
        assert config["tts"]["provider"] == "elevenlabs"
        # But image/vision still get set because user didn't override them
        assert "image_gen" in changed
        assert "vision" in changed

    def test_edge_default_is_overridable(self):
        """'edge' is the built-in TTS default — treat as not-explicit."""
        config = {
            "model": {"provider": "minimax"},
            "tts": {"provider": "edge"},
        }
        changed = apply_minimax_provider_defaults(config)
        assert "tts" in changed
        assert config["tts"]["provider"] == "minimax"

    def test_explicit_image_fal_is_overridable(self):
        """FAL is the built-in image_gen default — treat as not-explicit."""
        config = {
            "model": {"provider": "minimax"},
            "image_gen": {"provider": "fal"},
        }
        changed = apply_minimax_provider_defaults(config)
        assert "image_gen" in changed
        assert config["image_gen"]["provider"] == "minimax"

    def test_respects_explicit_image_third_party(self):
        config = {
            "model": {"provider": "minimax"},
            "image_gen": {"provider": "stability"},
        }
        changed = apply_minimax_provider_defaults(config)
        assert "image_gen" not in changed
        assert config["image_gen"]["provider"] == "stability"

    def test_idempotent(self):
        """Second run is a no-op; already-set values aren't re-reported."""
        config = {"model": {"provider": "minimax"}}
        first = apply_minimax_provider_defaults(config)
        assert first == {"tts", "image_gen", "vision"}
        second = apply_minimax_provider_defaults(config)
        assert second == set()

    def test_cn_provider_triggers_same_defaults(self):
        config = {"model": {"provider": "minimax-cn"}}
        changed = apply_minimax_provider_defaults(config)
        assert changed == {"tts", "image_gen", "vision"}

    def test_auxiliary_already_populated(self):
        """Other auxiliary sections must be preserved; vision config is
        not touched (dispatch is runtime-only)."""
        config = {
            "model": {"provider": "minimax"},
            "auxiliary": {"compression": {"provider": "openai"}},
        }
        apply_minimax_provider_defaults(config)
        assert config["auxiliary"]["compression"]["provider"] == "openai"
        # Vision is reported in `changed` but config is not modified.
        assert "vision" not in config["auxiliary"] or \
               config["auxiliary"].get("vision") in (None, {})

    def test_respects_explicit_vision_provider(self):
        """If the user has explicitly set a non-MiniMax vision provider,
        we must not claim we wired vision to MiniMax — the runtime
        dispatcher will step aside for that override."""
        config = {
            "model": {"provider": "minimax"},
            "auxiliary": {"vision": {"provider": "openrouter"}},
        }
        changed = apply_minimax_provider_defaults(config)
        assert "vision" not in changed
        assert config["auxiliary"]["vision"]["provider"] == "openrouter"

    def test_vision_auto_override_is_treated_as_unset(self):
        """`auxiliary.vision.provider: auto` is the built-in default —
        treat it as not-explicit so MiniMax VLM can take over."""
        config = {
            "model": {"provider": "minimax"},
            "auxiliary": {"vision": {"provider": "auto"}},
        }
        changed = apply_minimax_provider_defaults(config)
        assert "vision" in changed


# ─── describe_changes ────────────────────────────────────────────────────


class TestDescribeChanges:
    def test_empty_set(self):
        out = describe_changes(set())
        assert "No changes" in out

    def test_all_three(self):
        out = describe_changes({"tts", "image_gen", "vision"})
        assert "TTS" in out
        assert "Image" in out
        assert "Vision" in out
        assert out.count("•") == 3

    def test_one(self):
        out = describe_changes({"tts"})
        assert out.count("•") == 1
        assert "TTS" in out


# ─── Constants ───────────────────────────────────────────────────────────


class TestMiniMaxProviderSet:
    def test_contains_international(self):
        assert "minimax" in MINIMAX_PROVIDERS

    def test_contains_cn(self):
        assert "minimax-cn" in MINIMAX_PROVIDERS

    def test_no_unexpected_entries(self):
        """Guard against accidental typos like 'mini-max' or 'MiniMax'."""
        assert MINIMAX_PROVIDERS == {"minimax", "minimax-cn"}
