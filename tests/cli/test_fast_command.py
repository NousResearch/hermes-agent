"""Tests for the /fast CLI command and service-tier config handling."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _import_cli():
    import hermes_cli.config as config_mod

    if not hasattr(config_mod, "save_env_value_secure"):
        config_mod.save_env_value_secure = lambda key, value: {
            "success": True,
            "stored_as": key,
            "validated": False,
        }

    import cli as cli_mod

    return cli_mod


class TestParseServiceTierConfig(unittest.TestCase):
    def _parse(self, raw):
        cli_mod = _import_cli()
        return cli_mod._parse_service_tier_config(raw)

    def test_fast_maps_to_priority(self):
        self.assertEqual(self._parse("fast"), "priority")
        self.assertEqual(self._parse("priority"), "priority")

    def test_normal_disables_service_tier(self):
        self.assertIsNone(self._parse("normal"))
        self.assertIsNone(self._parse("off"))
        self.assertIsNone(self._parse(""))


class TestHandleFastCommand(unittest.TestCase):
    def _make_cli(self, service_tier=None):
        return SimpleNamespace(
            service_tier=service_tier,
            provider="openai-codex",
            requested_provider="openai-codex",
            model="gpt-5.4",
            _fast_command_available=lambda: True,
            agent=MagicMock(),
        )

    def test_no_args_shows_status(self):
        cli_mod = _import_cli()
        stub = self._make_cli(service_tier=None)
        with (
            patch.object(cli_mod, "_cprint") as mock_cprint,
            patch.object(cli_mod, "save_config_value") as mock_save,
        ):
            cli_mod.HermesCLI._handle_fast_command(stub, "/fast")

        # Bare /fast shows status, does not change config
        mock_save.assert_not_called()
        # Should have printed the status line
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertIn("normal", printed)

    def test_no_args_shows_fast_when_enabled(self):
        cli_mod = _import_cli()
        stub = self._make_cli(service_tier="priority")
        with (
            patch.object(cli_mod, "_cprint") as mock_cprint,
            patch.object(cli_mod, "save_config_value") as mock_save,
        ):
            cli_mod.HermesCLI._handle_fast_command(stub, "/fast")

        mock_save.assert_not_called()
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertIn("fast", printed)

    def test_normal_argument_clears_service_tier(self):
        cli_mod = _import_cli()
        stub = self._make_cli(service_tier="priority")
        with (
            patch.object(cli_mod, "_cprint"),
            patch.object(cli_mod, "save_config_value", return_value=True) as mock_save,
        ):
            cli_mod.HermesCLI._handle_fast_command(stub, "/fast normal")

        mock_save.assert_called_once_with("agent.service_tier", "normal")
        self.assertIsNone(stub.service_tier)
        self.assertIsNone(stub.agent)

    def test_unsupported_model_does_not_expose_fast(self):
        cli_mod = _import_cli()
        stub = SimpleNamespace(
            service_tier=None,
            provider="openai-codex",
            requested_provider="openai-codex",
            model="gpt-5.3-codex",
            _fast_command_available=lambda: False,
            agent=MagicMock(),
        )

        with (
            patch.object(cli_mod, "_cprint") as mock_cprint,
            patch.object(cli_mod, "save_config_value") as mock_save,
        ):
            cli_mod.HermesCLI._handle_fast_command(stub, "/fast")

        mock_save.assert_not_called()
        self.assertTrue(mock_cprint.called)


class TestFastModeRegistry(unittest.TestCase):
    def test_only_gpt_5_4_is_enabled_for_codex(self):
        from hermes_cli.models import fast_mode_backend_config

        assert fast_mode_backend_config("gpt-5.4") == {
            "provider": "openai-codex",
            "request_overrides": {"service_tier": "priority"},
        }
        assert fast_mode_backend_config("gpt-5.3-codex") is None


class TestFastModeRouting(unittest.TestCase):
    def test_fast_command_exposed_for_model_even_when_provider_is_auto(self):
        cli_mod = _import_cli()
        stub = SimpleNamespace(provider="auto", requested_provider="auto", model="gpt-5.4", agent=None)

        assert cli_mod.HermesCLI._fast_command_available(stub) is True

    def test_turn_route_switches_to_model_backend_when_fast_enabled(self):
        cli_mod = _import_cli()
        stub = SimpleNamespace(
            model="gpt-5.4",
            api_key="primary-key",
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            api_mode="chat_completions",
            acp_command=None,
            acp_args=[],
            _credential_pool=None,
            _smart_model_routing={},
            service_tier="priority",
        )

        with (
            patch("agent.smart_model_routing.resolve_turn_route", return_value={
                "model": "gpt-5.4",
                "runtime": {
                    "api_key": "primary-key",
                    "base_url": "https://openrouter.ai/api/v1",
                    "provider": "openrouter",
                    "api_mode": "chat_completions",
                    "command": None,
                    "args": [],
                    "credential_pool": None,
                },
                "label": None,
                "signature": ("gpt-5.4", "openrouter", "https://openrouter.ai/api/v1", "chat_completions", None, ()),
            }),
            patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
                "provider": "openai-codex",
                "api_mode": "codex_responses",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_key": "codex-key",
                "command": None,
                "args": [],
                "credential_pool": None,
            }),
        ):
            route = cli_mod.HermesCLI._resolve_turn_agent_config(stub, "hi")

        assert route["runtime"]["provider"] == "openai-codex"
        assert route["runtime"]["api_mode"] == "codex_responses"
        assert route["request_overrides"] == {"service_tier": "priority"}

    def test_turn_route_keeps_primary_runtime_when_model_has_no_fast_backend(self):
        cli_mod = _import_cli()
        stub = SimpleNamespace(
            model="gpt-5.3-codex",
            api_key="primary-key",
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            api_mode="chat_completions",
            acp_command=None,
            acp_args=[],
            _credential_pool=None,
            _smart_model_routing={},
            service_tier="priority",
        )

        primary_route = {
            "model": "gpt-5.3-codex",
            "runtime": {
                "api_key": "primary-key",
                "base_url": "https://openrouter.ai/api/v1",
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "command": None,
                "args": [],
                "credential_pool": None,
            },
            "label": None,
            "signature": ("gpt-5.3-codex", "openrouter", "https://openrouter.ai/api/v1", "chat_completions", None, ()),
        }
        with patch("agent.smart_model_routing.resolve_turn_route", return_value=primary_route):
            route = cli_mod.HermesCLI._resolve_turn_agent_config(stub, "hi")

        assert route["runtime"]["provider"] == "openrouter"
        assert route.get("request_overrides") is None


class TestConfigDefault(unittest.TestCase):
    def test_default_config_has_service_tier(self):
        from hermes_cli.config import DEFAULT_CONFIG

        agent = DEFAULT_CONFIG.get("agent", {})
        self.assertIn("service_tier", agent)
        self.assertEqual(agent["service_tier"], "")
