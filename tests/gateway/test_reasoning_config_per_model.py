"""Tests for per-model reasoning_effort override in gateway _load_reasoning_config."""

import pytest

import gateway.run as gateway_run


class TestGatewayPerModelReasoningConfig:
    """Test GatewayRunner._load_reasoning_config respects per-model overrides."""

    def test_per_model_override_takes_precedence(self, monkeypatch):
        """Per-model override wins over global reasoning_effort."""
        from hermes_cli.config import DEFAULT_CONFIG

        fake_cfg = {
            "model": {"default": "anthropic/claude-opus-4.5"},
            "agent": {
                "reasoning_effort": "medium",
                "reasoning_overrides": {
                    "anthropic/claude-opus-4.5": "xhigh",
                },
            },
        }
        monkeypatch.setattr(gateway_run, "_load_gateway_runtime_config", lambda: fake_cfg)

        result = gateway_run.GatewayRunner._load_reasoning_config()
        assert result is not None
        assert result["enabled"] is True
        assert result["effort"] == "xhigh"


    def test_global_fallback_with_yaml_false(self, monkeypatch):
        """YAML boolean False must reach parse_reasoning_effort uncoerced.

        Regression: str(... or "").strip() turned False into "", silently
        re-enabling thinking. The raw value must pass through so
        parse_reasoning_effort(False) returns {'enabled': False}.
        """
        fake_cfg = {
            "model": {"default": "gpt-5"},
            "agent": {
                "reasoning_effort": False,  # YAML boolean, not string
            },
        }
        monkeypatch.setattr(gateway_run, "_load_gateway_runtime_config", lambda: fake_cfg)

        result = gateway_run.GatewayRunner._load_reasoning_config()
        assert result is not None
        assert result.get("enabled") is False


class TestGatewaySessionEffectiveModel:
    """The reasoning override must track the SESSION's effective model.

    Regression guard: _load_reasoning_config used to always read
    model.default from config.yaml, so a session-only /model switch to a
    different model kept resolving the config default's override.
    """

    def test_explicit_model_beats_config_default(self, monkeypatch):
        """_load_reasoning_config(model=...) resolves for that model, not model.default."""
        fake_cfg = {
            "model": {"default": "gpt-5"},
            "agent": {
                "reasoning_effort": "medium",
                "reasoning_overrides": {
                    "gpt-5": "low",
                    "claude-opus-4.5": "xhigh",
                },
            },
        }
        monkeypatch.setattr(gateway_run, "_load_gateway_runtime_config", lambda: fake_cfg)

        # Session switched (session-only) to claude-opus-4.5 — its override
        # must win over the config default model's override.
        result = gateway_run.GatewayRunner._load_reasoning_config("claude-opus-4.5")
        assert result is not None
        assert result["effort"] == "xhigh"

        # And without a model arg, the config default's override applies.
        result_default = gateway_run.GatewayRunner._load_reasoning_config()
        assert result_default is not None
        assert result_default["effort"] == "low"

