"""Tests for model.generation_params config → request_overrides flow.

Covers CLI path, gateway path, dump display, and edge cases.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# CLI path: _resolve_turn_agent_config in CLIAgentSetupMixin
# ---------------------------------------------------------------------------

class TestCLIGenerationParams:
    """CLI _resolve_turn_agent_config merges generation_params into request_overrides."""

    def _make_cli_stub(self, model_name="test-model", svc_tier=None):
        """Create a minimal stub with the attributes _resolve_turn_agent_config reads."""
        from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin

        class Stub(CLIAgentSetupMixin):
            api_key = "sk-test"
            base_url = "http://localhost:8080/v1"
            provider = "custom"
            api_mode = "chat_completions"
            acp_command = None
            acp_args = []
            _credential_pool = None
            _fallback_model = None

        s = Stub()
        s.model = model_name
        s.service_tier = svc_tier
        return s

    def _mock_cli_config(self, cli_config):
        """Return a context manager that patches 'cli' module with given CLI_CONFIG."""
        return patch.dict("sys.modules", {"cli": MagicMock(CLI_CONFIG=cli_config)})

    def test_generation_params_merged(self):
        gen_params = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
        stub = self._make_cli_stub()
        with self._mock_cli_config({"model": {"generation_params": gen_params}}):
            route = stub._resolve_turn_agent_config("hello")
        assert route["request_overrides"] is not None
        assert route["request_overrides"]["temperature"] == 1.0
        assert route["request_overrides"]["top_p"] == 0.95
        assert route["request_overrides"]["top_k"] == 64

    def test_no_generation_params_returns_none(self):
        stub = self._make_cli_stub()
        with self._mock_cli_config({"model": {}}):
            route = stub._resolve_turn_agent_config("hello")
        assert route["request_overrides"] is None

    def test_string_model_returns_none(self):
        stub = self._make_cli_stub()
        with self._mock_cli_config({"model": "openrouter/claude"}):
            route = stub._resolve_turn_agent_config("hello")
        assert route["request_overrides"] is None

    def test_empty_generation_params_returns_none(self):
        stub = self._make_cli_stub()
        with self._mock_cli_config({"model": {"generation_params": {}}}):
            route = stub._resolve_turn_agent_config("hello")
        assert route["request_overrides"] is None

    def test_non_dict_generation_params_ignored(self):
        stub = self._make_cli_stub()
        with self._mock_cli_config({"model": {"generation_params": "not-a-dict"}}):
            route = stub._resolve_turn_agent_config("hello")
        assert route["request_overrides"] is None

    def test_fast_mode_merges_with_generation_params(self):
        gen_params = {"temperature": 1.0}
        stub = self._make_cli_stub(svc_tier="fast")
        with self._mock_cli_config({"model": {"generation_params": gen_params}}):
            with patch("hermes_cli.models.resolve_fast_mode_overrides", return_value={"service_tier": "fast"}):
                route = stub._resolve_turn_agent_config("hello")
        assert route["request_overrides"]["temperature"] == 1.0
        assert route["request_overrides"]["service_tier"] == "fast"


# ---------------------------------------------------------------------------
# Gateway path: _resolve_turn_agent_config in GatewayRunner
# ---------------------------------------------------------------------------

class TestGatewayGenerationParams:
    """Gateway _resolve_turn_agent_config merges generation_params into request_overrides."""

    def test_generation_params_merged(self):
        gen_params = {"temperature": 0.7, "top_p": 0.9}
        mock_cfg = {"model": {"generation_params": gen_params}}
        with patch("hermes_cli.config.load_config", return_value=mock_cfg):
            from gateway.run import GatewayRunner
            gw = object.__new__(GatewayRunner)
            gw._service_tier = None
            route = gw._resolve_turn_agent_config("hello", "test-model", {"api_key": "k", "base_url": "u", "provider": "p"})
        assert route["request_overrides"]["temperature"] == 0.7
        assert route["request_overrides"]["top_p"] == 0.9

    def test_no_generation_params_returns_empty(self):
        mock_cfg = {"model": {}}
        with patch("hermes_cli.config.load_config", return_value=mock_cfg):
            from gateway.run import GatewayRunner
            gw = object.__new__(GatewayRunner)
            gw._service_tier = None
            route = gw._resolve_turn_agent_config("hello", "test-model", {"api_key": "k", "base_url": "u", "provider": "p"})
        assert route["request_overrides"] == {}

    def test_fast_mode_merges_with_generation_params(self):
        gen_params = {"temperature": 1.0}
        mock_cfg = {"model": {"generation_params": gen_params}}
        with patch("hermes_cli.config.load_config", return_value=mock_cfg):
            with patch("hermes_cli.models.resolve_fast_mode_overrides", return_value={"service_tier": "fast"}):
                from gateway.run import GatewayRunner
                gw = object.__new__(GatewayRunner)
                gw._service_tier = "fast"
                route = gw._resolve_turn_agent_config("hello", "test-model", {"api_key": "k", "base_url": "u", "provider": "p"})
        assert route["request_overrides"]["temperature"] == 1.0
        assert route["request_overrides"]["service_tier"] == "fast"


# ---------------------------------------------------------------------------
# Dump path: _config_overrides shows generation_params
# ---------------------------------------------------------------------------

class TestDumpGenerationParams:
    """hermes config display shows model.generation_params when configured."""

    def test_generation_params_in_dump(self):
        from hermes_cli.dump import _config_overrides
        config = {"model": {"generation_params": {"temperature": 1.0, "top_p": 0.95}}}
        overrides = _config_overrides(config)
        assert "model.generation_params" in overrides
        assert "temperature" in overrides["model.generation_params"]

    def test_empty_generation_params_not_in_dump(self):
        from hermes_cli.dump import _config_overrides
        config = {"model": {"generation_params": {}}}
        overrides = _config_overrides(config)
        assert "model.generation_params" not in overrides

    def test_string_model_no_error(self):
        from hermes_cli.dump import _config_overrides
        config = {"model": "openrouter/claude"}
        overrides = _config_overrides(config)
        assert "model.generation_params" not in overrides
