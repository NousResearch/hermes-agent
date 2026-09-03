"""Tests for OpenRouter reasoning-capability metadata (prime-agent#1258 port).

Covers:
  - parse_openrouter_reasoning_capabilities: catalog-entry normalization
  - clamp_reasoning_effort_to_supported: nearest-lower-level clamping
  - openrouter_model_reasoning_capabilities: cache + tri-state contract
  - AIAgent._supports_reasoning_extra_body: metadata-first, static fallback
  - OpenRouterProfile._clamp_reasoning_to_catalog: emitted effort clamping
"""

import pytest

from hermes_cli.models import (
    clamp_reasoning_effort_to_supported,
    parse_openrouter_reasoning_capabilities,
)


class TestParseReasoningCapabilities:
    def test_reasoning_supported_with_efforts(self):
        item = {
            "id": "nvidia/nemotron-3-ultra",
            "supported_parameters": ["temperature", "tools", "reasoning"],
            "reasoning": {"mandatory": False, "supported_efforts": ["low", "medium", "high"]},
        }
        caps = parse_openrouter_reasoning_capabilities(item)
        assert caps == {
            "supports_reasoning": True,
            "supported_efforts": ["low", "medium", "high"],
            "mandatory": False,
        }

    def test_reasoning_supported_all_efforts_when_field_omitted(self):
        item = {
            "id": "deepseek/deepseek-chat",
            "supported_parameters": ["reasoning", "tools"],
            "reasoning": {},
        }
        caps = parse_openrouter_reasoning_capabilities(item)
        assert caps["supports_reasoning"] is True
        assert caps["supported_efforts"] is None  # None = every effort accepted
        assert caps["mandatory"] is False

    def test_reasoning_supported_without_reasoning_object(self):
        # supported_parameters alone is authoritative for the on/off question.
        item = {"supported_parameters": ["reasoning"]}
        caps = parse_openrouter_reasoning_capabilities(item)
        assert caps["supports_reasoning"] is True
        assert caps["supported_efforts"] is None

    def test_mandatory_flag(self):
        item = {
            "supported_parameters": ["reasoning"],
            "reasoning": {"mandatory": True, "supported_efforts": None},
        }
        caps = parse_openrouter_reasoning_capabilities(item)
        assert caps["mandatory"] is True

    def test_reasoning_object_untrusted_without_supported_parameters_entry(self):
        # Top-level reasoning object present but supported_parameters omits
        # "reasoning" → the route rejects reasoning controls.
        item = {
            "supported_parameters": ["temperature", "tools"],
            "reasoning": {"supported_efforts": ["high"]},
        }
        assert parse_openrouter_reasoning_capabilities(item) == {
            "supports_reasoning": False
        }

    def test_unknown_when_supported_parameters_missing(self):
        assert parse_openrouter_reasoning_capabilities({"id": "x"}) is None
        assert parse_openrouter_reasoning_capabilities({"supported_parameters": "bad"}) is None
        assert parse_openrouter_reasoning_capabilities("not-a-dict") is None

    def test_effort_list_normalized_and_deduped(self):
        item = {
            "supported_parameters": ["reasoning"],
            "reasoning": {"supported_efforts": [" High ", "high", "LOW", "", 3]},
        }
        caps = parse_openrouter_reasoning_capabilities(item)
        assert caps["supported_efforts"] == ["high", "low", "3"]


class TestClampReasoningEffort:
    @pytest.mark.parametrize(
        "effort,supported,expected",
        [
            # Supported as-is → unchanged.
            ("high", ["low", "medium", "high"], "high"),
            # Unknown supported list → pass through.
            ("ultra", None, "ultra"),
            ("ultra", [], "ultra"),
            # Clamp DOWN to nearest lower supported level.
            ("ultra", ["low", "medium", "high"], "high"),
            ("xhigh", ["low", "medium", "high"], "high"),
            ("max", ["minimal", "low"], "low"),
            # No lower level exists → nearest (lowest) supported.
            ("minimal", ["low", "medium"], "low"),
            ("none", ["low", "high"], "low"),
            # Unrecognized custom level → pass through untouched.
            ("turbo-think", ["low", "high"], "turbo-think"),
            # Supported list containing only unrecognized names → pass through.
            ("high", ["banana"], "high"),
            # Empty/None effort → pass through.
            (None, ["low"], None),
            ("", ["low"], ""),
        ],
    )
    def test_clamping(self, effort, supported, expected):
        assert clamp_reasoning_effort_to_supported(effort, supported) == expected

    def test_never_escalates(self):
        # A clamp must never pick a HIGHER level when a lower one exists.
        assert clamp_reasoning_effort_to_supported("medium", ["low", "xhigh"]) == "low"


class TestOpenRouterModelReasoningCapabilities:
    def _prime_cache(self, monkeypatch, caps_by_id):
        import hermes_cli.models as models_mod
        monkeypatch.setattr(models_mod, "_openrouter_reasoning_caps_cache", caps_by_id)
        monkeypatch.setattr(models_mod, "_openrouter_reasoning_caps_failed_at", None)

    def test_known_model(self, monkeypatch):
        from hermes_cli.models import openrouter_model_reasoning_capabilities
        self._prime_cache(monkeypatch, {
            "nvidia/nemotron-3-ultra": {
                "supports_reasoning": True,
                "supported_efforts": ["low", "high"],
                "mandatory": False,
            },
        })
        caps = openrouter_model_reasoning_capabilities("nvidia/nemotron-3-ultra")
        assert caps["supports_reasoning"] is True

    def test_unlisted_model_returns_none(self, monkeypatch):
        from hermes_cli.models import openrouter_model_reasoning_capabilities
        self._prime_cache(monkeypatch, {"a/b": {"supports_reasoning": True}})
        assert openrouter_model_reasoning_capabilities("private/custom") is None

    def test_empty_model_returns_none(self, monkeypatch):
        from hermes_cli.models import openrouter_model_reasoning_capabilities
        self._prime_cache(monkeypatch, {"a/b": {"supports_reasoning": True}})
        assert openrouter_model_reasoning_capabilities("") is None
        assert openrouter_model_reasoning_capabilities(None) is None

    def test_catalog_unreachable_returns_none_and_rate_limits(self, monkeypatch):
        import hermes_cli.models as models_mod
        from hermes_cli.models import openrouter_model_reasoning_capabilities

        monkeypatch.setattr(models_mod, "_openrouter_reasoning_caps_cache", None)
        monkeypatch.setattr(models_mod, "_openrouter_reasoning_caps_failed_at", None)
        calls = {"n": 0}

        def _boom(req, *, timeout):
            calls["n"] += 1
            raise OSError("offline")

        monkeypatch.setattr(models_mod, "_urlopen_model_catalog_request", _boom)
        assert openrouter_model_reasoning_capabilities("a/b", allow_fetch=True) is None
        assert openrouter_model_reasoning_capabilities("a/b", allow_fetch=True) is None
        # Second call inside the 60s failure TTL must not re-fetch.
        assert calls["n"] == 1

    def test_cache_only_by_default_never_fetches(self, monkeypatch):
        import hermes_cli.models as models_mod
        from hermes_cli.models import openrouter_model_reasoning_capabilities

        monkeypatch.setattr(models_mod, "_openrouter_reasoning_caps_cache", None)
        monkeypatch.setattr(models_mod, "_openrouter_reasoning_caps_failed_at", None)

        def _boom(req, *, timeout):
            raise AssertionError("hot path must not fetch")

        monkeypatch.setattr(models_mod, "_urlopen_model_catalog_request", _boom)
        # Default (allow_fetch=False) → cache-only, no HTTP even when cold.
        assert openrouter_model_reasoning_capabilities("a/b") is None


class TestSupportsReasoningExtraBodyMetadataGate:
    """AIAgent._supports_reasoning_extra_body: metadata-first with fallback."""

    def _make_agent(self, model):
        from run_agent import AIAgent
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model=model,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        return agent

    def test_metadata_positive_overrides_static_list(self, monkeypatch):
        import hermes_cli.models as models_mod
        monkeypatch.setattr(models_mod, "_openrouter_reasoning_caps_failed_at", None)
        monkeypatch.setattr(models_mod, "_openrouter_reasoning_caps_cache", {
            # nvidia/ is NOT in the static prefix allowlist (#75386) —
            # metadata must make it work without a code change.
            "nvidia/nemotron-3-ultra": {
                "supports_reasoning": True,
                "supported_efforts": ["low", "medium", "high"],
                "mandatory": False,
            },
        })
        agent = self._make_agent("nvidia/nemotron-3-ultra")
        assert agent._supports_reasoning_extra_body() is True

    def test_metadata_negative_overrides_static_list(self, monkeypatch):
        import hermes_cli.models as models_mod
        monkeypatch.setattr(models_mod, "_openrouter_reasoning_caps_failed_at", None)
        monkeypatch.setattr(models_mod, "_openrouter_reasoning_caps_cache", {
            # openai/ IS in the static prefix list, but the catalog says this
            # route rejects reasoning controls — the definitive negative wins.
            "openai/gpt-4o-mini": {"supports_reasoning": False},
        })
        agent = self._make_agent("openai/gpt-4o-mini")
        assert agent._supports_reasoning_extra_body() is False

    def test_unknown_falls_back_to_static_prefixes(self, monkeypatch):
        import hermes_cli.models as models_mod
        monkeypatch.setattr(models_mod, "_openrouter_reasoning_caps_cache", {"a/b": None})
        monkeypatch.setattr(models_mod, "_openrouter_reasoning_caps_failed_at", None)
        # deepseek/ is in the static list; unlisted in catalog → fallback True.
        agent = self._make_agent("deepseek/deepseek-chat")
        assert agent._supports_reasoning_extra_body() is True
        # unknown vendor absent from both → False.
        agent2 = self._make_agent("someveryunknown/model-x")
        assert agent2._supports_reasoning_extra_body() is False


class TestOpenRouterProfileClamp:
    def test_clamp_applied_in_build_api_kwargs_extras(self, monkeypatch):
        import hermes_cli.models as models_mod
        from providers import get_provider_profile

        monkeypatch.setattr(models_mod, "_openrouter_reasoning_caps_failed_at", None)
        monkeypatch.setattr(models_mod, "_openrouter_reasoning_caps_cache", {
            "qwen/qwen3-plus": {
                "supports_reasoning": True,
                "supported_efforts": ["low", "medium", "high"],
                "mandatory": False,
            },
        })
        profile = get_provider_profile("openrouter")
        assert profile is not None
        extra_body, _top = profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "ultra"},
            supports_reasoning=True,
            model="qwen/qwen3-plus",
        )
        assert extra_body["reasoning"]["effort"] == "high"

    def test_no_clamp_when_catalog_unknown(self, monkeypatch):
        import hermes_cli.models as models_mod
        from providers import get_provider_profile

        monkeypatch.setattr(models_mod, "_openrouter_reasoning_caps_cache", {"a/b": None})
        monkeypatch.setattr(models_mod, "_openrouter_reasoning_caps_failed_at", None)
        profile = get_provider_profile("openrouter")
        assert profile is not None
        extra_body, _top = profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "ultra"},
            supports_reasoning=True,
            model="unlisted/model",
        )
        # Unknown capability → passthrough unchanged (no silent downgrade).
        assert extra_body["reasoning"]["effort"] == "ultra"

    def test_disable_omitted_for_mandatory_route_kept_otherwise(self, monkeypatch):
        """A reasoning-mandatory route 400s on ``{enabled: false}`` — omit it;
        a route that can disable still gets the user's disable verbatim."""
        import hermes_cli.models as models_mod
        from providers import get_provider_profile

        monkeypatch.setattr(models_mod, "_openrouter_reasoning_caps_failed_at", None)
        monkeypatch.setattr(models_mod, "_openrouter_reasoning_caps_cache", {
            "z-ai/glm-5.3-flash": {
                "supports_reasoning": True,
                "supported_efforts": ["max", "high", "low"],
                "mandatory": True,
            },
            "z-ai/glm-5.1": {"supports_reasoning": True, "supported_efforts": None, "mandatory": False},
        })
        profile = get_provider_profile("openrouter")
        for disable in ({"enabled": False}, {"enabled": True, "effort": "none"}):
            extra_body, _top = profile.build_api_kwargs_extras(
                reasoning_config=disable, supports_reasoning=True, model="z-ai/glm-5.3-flash",
            )
            assert "reasoning" not in extra_body, disable
        extra_body, _top = profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, supports_reasoning=True, model="z-ai/glm-5.1",
        )
        assert extra_body["reasoning"] == {"enabled": False}
