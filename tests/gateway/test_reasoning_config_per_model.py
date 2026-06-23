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

    def test_global_fallback_when_no_override(self, monkeypatch):
        """Global reasoning_effort applies when no per-model override matches."""
        fake_cfg = {
            "model": {"default": "gpt-5"},
            "agent": {
                "reasoning_effort": "high",
                "reasoning_overrides": {
                    "anthropic/claude-opus-4.5": "xhigh",
                },
            },
        }
        monkeypatch.setattr(gateway_run, "_load_gateway_runtime_config", lambda: fake_cfg)

        result = gateway_run.GatewayRunner._load_reasoning_config()
        assert result is not None
        assert result["effort"] == "high"

    def test_spelling_tolerant_match_in_gateway(self, monkeypatch):
        """Override matches even with different spelling (dots vs dashes)."""
        fake_cfg = {
            "model": {"default": "claude-opus-4-5"},
            "agent": {
                "reasoning_effort": "medium",
                "reasoning_overrides": {
                    "claude-opus-4.5": "xhigh",  # key has dots, model has dashes
                },
            },
        }
        monkeypatch.setattr(gateway_run, "_load_gateway_runtime_config", lambda: fake_cfg)

        result = gateway_run.GatewayRunner._load_reasoning_config()
        assert result is not None
        assert result["effort"] == "xhigh"

    def test_no_overrides_dict(self, monkeypatch):
        """Works fine when reasoning_overrides key is absent."""
        fake_cfg = {
            "model": {"default": "gpt-5"},
            "agent": {
                "reasoning_effort": "low",
            },
        }
        monkeypatch.setattr(gateway_run, "_load_gateway_runtime_config", lambda: fake_cfg)

        result = gateway_run.GatewayRunner._load_reasoning_config()
        assert result is not None
        assert result["effort"] == "low"

    def test_empty_overrides(self, monkeypatch):
        """Empty overrides dict falls back to global."""
        fake_cfg = {
            "model": {"default": "gpt-5"},
            "agent": {
                "reasoning_effort": "medium",
                "reasoning_overrides": {},
            },
        }
        monkeypatch.setattr(gateway_run, "_load_gateway_runtime_config", lambda: fake_cfg)

        result = gateway_run.GatewayRunner._load_reasoning_config()
        assert result is not None
        assert result["effort"] == "medium"

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
