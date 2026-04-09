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

    def test_no_args_uses_dropdown_selection(self):
        cli_mod = _import_cli()
        stub = self._make_cli(service_tier=None)
        with (
            patch.object(cli_mod, "_cprint"),
            patch.object(cli_mod, "save_config_value", return_value=True) as mock_save,
            patch.object(cli_mod, "_prompt_fast_mode_selection", return_value="fast") as mock_prompt,
        ):
            cli_mod.HermesCLI._handle_fast_command(stub, "/fast")

        mock_prompt.assert_called_once_with(current_tier=None)
        mock_save.assert_called_once_with("agent.service_tier", "fast")
        self.assertEqual(stub.service_tier, "priority")
        self.assertIsNone(stub.agent)

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

        assert fast_mode_backend_config("openai-codex", "gpt-5.4") == {"service_tier": "priority"}
        assert fast_mode_backend_config("openai-codex", "gpt-5.3-codex") is None
        assert fast_mode_backend_config("openai", "gpt-5.4") is None


class TestConfigDefault(unittest.TestCase):
    def test_default_config_has_service_tier(self):
        from hermes_cli.config import DEFAULT_CONFIG

        agent = DEFAULT_CONFIG.get("agent", {})
        self.assertIn("service_tier", agent)
        self.assertEqual(agent["service_tier"], "")