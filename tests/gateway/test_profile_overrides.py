"""Tests for per-agent profile runtime / toolset overrides in GatewayRunner.

These cover the wiring that lets ``AgentProfile.model / provider /
enabled_toolsets / disabled_toolsets`` actually take effect at AIAgent
construction time — without this wiring those fields are dead-letter.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.profile import AgentProfile, use_profile
from gateway.run import GatewayRunner


def _bound_method(name):
    """Return the unbound GatewayRunner method to invoke against a stub instance."""
    return getattr(GatewayRunner, name)


# =========================================================================
# _apply_profile_toolsets
# =========================================================================


class TestApplyProfileToolsets:
    def test_no_active_profile_is_passthrough(self):
        runner = MagicMock(spec=GatewayRunner)
        fn = _bound_method("_apply_profile_toolsets")
        enabled, disabled = fn(runner, ["a", "b"], ["c"])
        assert enabled == ["a", "b"]
        assert disabled == ["c"]

    def test_default_main_profile_is_passthrough(self):
        runner = MagicMock(spec=GatewayRunner)
        fn = _bound_method("_apply_profile_toolsets")
        with use_profile(AgentProfile(id="main", enabled_toolsets=["x"])):
            enabled, disabled = fn(runner, ["a"], ["b"])
        # main is the default — never overrides, to preserve legacy behavior.
        assert enabled == ["a"]
        assert disabled == ["b"]

    def test_non_main_profile_enabled_toolsets_overrides(self):
        runner = MagicMock(spec=GatewayRunner)
        fn = _bound_method("_apply_profile_toolsets")
        profile = AgentProfile(id="coder", enabled_toolsets=["filesystem", "terminal"])
        with use_profile(profile):
            enabled, disabled = fn(runner, ["a", "b"], ["c"])
        assert enabled == ["filesystem", "terminal"]
        assert disabled == ["c"]  # untouched — profile only set enabled

    def test_non_main_profile_disabled_toolsets_overrides(self):
        runner = MagicMock(spec=GatewayRunner)
        fn = _bound_method("_apply_profile_toolsets")
        profile = AgentProfile(id="research", disabled_toolsets=["terminal"])
        with use_profile(profile):
            enabled, disabled = fn(runner, ["a"], ["c"])
        assert enabled == ["a"]
        assert disabled == ["terminal"]

    def test_profile_none_fields_preserve_defaults(self):
        """Profile with all-None toolset fields is identity."""
        runner = MagicMock(spec=GatewayRunner)
        fn = _bound_method("_apply_profile_toolsets")
        with use_profile(AgentProfile(id="coder")):
            enabled, disabled = fn(runner, ["a"], ["b"])
        assert enabled == ["a"]
        assert disabled == ["b"]

    def test_empty_list_is_explicit_override(self):
        """An empty list is an explicit choice, not None — must override."""
        runner = MagicMock(spec=GatewayRunner)
        fn = _bound_method("_apply_profile_toolsets")
        with use_profile(AgentProfile(id="coder", enabled_toolsets=[])):
            enabled, _ = fn(runner, ["a", "b"], None)
        assert enabled == []  # explicit empty wins


# =========================================================================
# _apply_profile_runtime_overrides
# =========================================================================


class TestApplyProfileRuntimeOverrides:
    def test_no_active_profile_is_passthrough(self):
        runner = MagicMock(spec=GatewayRunner)
        fn = _bound_method("_apply_profile_runtime_overrides")
        model, runtime = fn(runner, "gateway-model", {"provider": "anthropic"})
        assert model == "gateway-model"
        assert runtime == {"provider": "anthropic"}

    def test_default_main_profile_is_passthrough(self):
        runner = MagicMock(spec=GatewayRunner)
        fn = _bound_method("_apply_profile_runtime_overrides")
        with use_profile(AgentProfile(id="main", model="should-not-win")):
            model, runtime = fn(runner, "gateway-model", {"provider": "anthropic"})
        assert model == "gateway-model"

    def test_profile_model_overrides_when_provider_unset(self):
        """When the profile pins only ``model`` (no provider/key/base_url),
        the model swaps but the gateway runtime credentials are preserved."""
        runner = MagicMock(spec=GatewayRunner)
        fn = _bound_method("_apply_profile_runtime_overrides")
        with use_profile(AgentProfile(id="coder", model="anthropic/claude-opus-4-7")):
            model, runtime = fn(runner, "gateway-model", {"provider": "anthropic", "api_key": "k"})
        assert model == "anthropic/claude-opus-4-7"
        assert runtime == {"provider": "anthropic", "api_key": "k"}

    def test_profile_provider_triggers_runtime_resolution(self, monkeypatch):
        """Setting ``profile.provider`` re-resolves runtime via
        ``resolve_runtime_provider`` so api_mode/base_url stay consistent."""
        called_with = {}

        def fake_resolve(*, requested, explicit_api_key, explicit_base_url, target_model):
            called_with.update(
                requested=requested,
                explicit_api_key=explicit_api_key,
                explicit_base_url=explicit_base_url,
                target_model=target_model,
            )
            return {
                "provider": "openai",
                "api_key": "resolved-key",
                "base_url": "https://api.openai.com",
                "api_mode": "chat_completions",
            }

        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve
        )

        runner = MagicMock(spec=GatewayRunner)
        fn = _bound_method("_apply_profile_runtime_overrides")
        profile = AgentProfile(
            id="coder",
            model="gpt-5",
            provider="openai",
            base_url="https://api.openai.com",
        )
        with use_profile(profile):
            model, runtime = fn(runner, "gateway-model", {"provider": "anthropic", "api_key": "old"})

        assert model == "gpt-5"
        assert runtime["provider"] == "openai"
        assert runtime["api_key"] == "resolved-key"
        assert runtime["base_url"] == "https://api.openai.com"
        assert runtime["api_mode"] == "chat_completions"
        assert called_with["requested"] == "openai"
        assert called_with["explicit_base_url"] == "https://api.openai.com"
        assert called_with["target_model"] == "gpt-5"

    def test_api_key_env_reads_from_environment(self, monkeypatch):
        captured = {}

        def fake_resolve(*, requested, explicit_api_key, explicit_base_url, target_model):
            captured["explicit_api_key"] = explicit_api_key
            return {"provider": "anthropic", "api_key": explicit_api_key, "base_url": None, "api_mode": None}

        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve
        )
        monkeypatch.setenv("CODER_AGENT_KEY", "sk-coder-secret")

        runner = MagicMock(spec=GatewayRunner)
        fn = _bound_method("_apply_profile_runtime_overrides")
        profile = AgentProfile(id="coder", provider="anthropic", api_key_env="CODER_AGENT_KEY")
        with use_profile(profile):
            fn(runner, "m", {})

        assert captured["explicit_api_key"] == "sk-coder-secret"

    def test_resolve_failure_falls_back_to_gateway_runtime(self, monkeypatch):
        """If provider resolution raises, keep the gateway runtime — never
        return half-broken credentials to AIAgent."""

        def boom(**_kw):
            raise RuntimeError("auth pool empty")

        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider", boom
        )

        runner = MagicMock(spec=GatewayRunner)
        fn = _bound_method("_apply_profile_runtime_overrides")
        profile = AgentProfile(id="coder", model="gpt-5", provider="openai")
        gateway_runtime = {"provider": "anthropic", "api_key": "gateway-key"}
        with use_profile(profile):
            model, runtime = fn(runner, "gw-model", gateway_runtime)
        # Model already pinned by profile before resolve was attempted — keep it.
        assert model == "gpt-5"
        assert runtime is gateway_runtime  # untouched fallback
