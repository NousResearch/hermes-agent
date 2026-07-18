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

    def test_resolve_session_reasoning_forwards_model(self, monkeypatch):
        """_resolve_session_reasoning_config passes the effective model through
        (and session-scoped /reasoning overrides still win over it)."""
        fake_cfg = {
            "model": {"default": "gpt-5"},
            "agent": {
                "reasoning_effort": "medium",
                "reasoning_overrides": {"claude-opus-4.5": "xhigh"},
            },
        }
        monkeypatch.setattr(gateway_run, "_load_gateway_runtime_config", lambda: fake_cfg)

        runner = object.__new__(gateway_run.GatewayRunner)
        runner._session_reasoning_overrides = {}

        # No session override → per-model override for the effective model.
        result = runner._resolve_session_reasoning_config(
            session_key="agent:main:telegram:private:1", model="claude-opus-4.5"
        )
        assert result is not None
        assert result["effort"] == "xhigh"

        # Session-scoped /reasoning override still wins over per-model.
        runner._session_reasoning_overrides = {
            "agent:main:telegram:private:1": {"enabled": True, "effort": "minimal"}
        }
        result = runner._resolve_session_reasoning_config(
            session_key="agent:main:telegram:private:1", model="claude-opus-4.5"
        )
        assert result == {"enabled": True, "effort": "minimal"}
