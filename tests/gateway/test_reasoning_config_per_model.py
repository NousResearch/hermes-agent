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



class TestApiServerPerModelReasoning:
    """The OpenAI-compatible surface must resolve reasoning for the model the
    request actually runs, not ``model.default``.

    e81d18dfb removed exactly this defect from the native gateway paths
    ("the gateway resolving reasoning against config model.default instead of
    the session's effective model"); the API server adapter kept it because it
    resolved reasoning at function entry, before the model precedence chain.
    """

    @staticmethod
    def _cfg():
        return {
            "model": {"default": "cheap/model-mini"},
            "agent": {
                "reasoning_effort": "low",
                "reasoning_overrides": {"premium/model-max": "xhigh"},
            },
        }

    def _run(self, monkeypatch, effective_model):
        from unittest.mock import MagicMock, patch

        from gateway.config import PlatformConfig
        from gateway.platforms.api_server import APIServerAdapter

        monkeypatch.setattr(
            gateway_run, "_load_gateway_runtime_config", lambda: self._cfg(),
        )
        adapter = APIServerAdapter(PlatformConfig())

        with patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True), \
             patch("gateway.run._resolve_runtime_agent_kwargs") as kwargs_mock, \
             patch("gateway.run._resolve_gateway_model") as model_mock, \
             patch("gateway.run._load_gateway_config") as cfg_mock, \
             patch("run_agent.AIAgent") as agent_cls:
            kwargs_mock.return_value = {
                "api_key": "k", "base_url": None, "provider": None,
                "api_mode": None, "command": None, "args": [],
            }
            model_mock.return_value = effective_model
            cfg_mock.return_value = self._cfg()
            agent_cls.return_value = MagicMock()

            adapter._create_agent()

            return agent_cls.call_args.kwargs

    def test_reasoning_follows_the_requested_model(self, monkeypatch):
        kwargs = self._run(monkeypatch, "premium/model-max")

        assert kwargs["model"] == "premium/model-max"
        assert kwargs["reasoning_config"] == {"enabled": True, "effort": "xhigh"}

    def test_model_without_an_override_falls_back_to_global(self, monkeypatch):
        kwargs = self._run(monkeypatch, "cheap/model-mini")

        assert kwargs["reasoning_config"] == {"enabled": True, "effort": "low"}
