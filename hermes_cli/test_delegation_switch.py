"""Basic unit tests for hermes_cli/delegation_switch.py."""

import sys
import unittest
from unittest.mock import MagicMock, patch

from hermes_cli.delegation_switch import (
    apply_api_d_switch,
    format_api_d_status,
    parse_api_d_args,
)


class ParseApiDArgsTests(unittest.TestCase):
    def test_empty(self):
        args, errors = parse_api_d_args("")
        self.assertEqual(errors, [])
        self.assertEqual(args.key, "")
        self.assertEqual(args.provider, "")
        self.assertEqual(args.model, "")
        self.assertFalse(args.save)
        self.assertFalse(args.reload)

    def test_key_only(self):
        args, errors = parse_api_d_args("sk-test-key")
        self.assertEqual(errors, [])
        self.assertEqual(args.key, "sk-test-key")

    def test_key_with_flags(self):
        args, errors = parse_api_d_args("--provider openrouter --model sonnet --save sk-key")
        self.assertEqual(errors, [])
        self.assertEqual(args.provider, "openrouter")
        self.assertEqual(args.model, "sonnet")
        self.assertEqual(args.key, "sk-key")
        self.assertTrue(args.save)

    def test_quoted_key(self):
        args, errors = parse_api_d_args('"sk-test-key with spaces"')
        self.assertEqual(errors, [])
        self.assertEqual(args.key, "sk-test-key with spaces")


class FormatStatusTests(unittest.TestCase):
    def test_all_set(self):
        text = format_api_d_status("openrouter", "sonnet", "sk-abc123")
        self.assertIn("openrouter", text)
        self.assertIn("sonnet", text)
        self.assertIn("***", text)

    def test_inherit(self):
        text = format_api_d_status("", "", "")
        self.assertIn("(inherit from parent)", text)
        self.assertIn("(not set)", text)


class ApplySwitchTests(unittest.TestCase):
    def _patch_runtime(self, cfg):
        """Install a fake ``cli`` module with the given CLI_CONFIG."""
        cli_module = MagicMock()
        cli_module.CLI_CONFIG = cfg
        return patch.dict(sys.modules, {"cli": cli_module})

    def test_updates_runtime_config(self):
        cli_config = {"delegation": {"provider": "deepseek", "model": "deepseek-v4-flash"}}
        with self._patch_runtime(cli_config):
            result = apply_api_d_switch(
                provider="openrouter",
                model="sonnet",
                api_key="sk-new",
                save_to_config=False,
            )
        self.assertTrue(result.success)
        self.assertEqual(cli_config["delegation"]["provider"], "openrouter")
        self.assertEqual(cli_config["delegation"]["model"], "sonnet")
        self.assertEqual(cli_config["delegation"]["api_key"], "sk-new")
        self.assertFalse(result.saved_to_config)

    def test_partial_update(self):
        cli_config = {"delegation": {"provider": "deepseek", "model": "deepseek-v4-flash", "api_key": "old"}}
        with self._patch_runtime(cli_config):
            result = apply_api_d_switch(
                provider="",
                model="",
                api_key="sk-new",
                save_to_config=False,
            )
        self.assertTrue(result.success)
        self.assertEqual(cli_config["delegation"]["provider"], "deepseek")
        self.assertEqual(cli_config["delegation"]["model"], "deepseek-v4-flash")
        self.assertEqual(cli_config["delegation"]["api_key"], "sk-new")

    def test_save_to_config(self):
        cli_config = {"delegation": {}}
        with self._patch_runtime(cli_config), patch(
            "hermes_cli.config.save_config_value",
            return_value=True,
        ) as mock_save:
            result = apply_api_d_switch(
                provider="openrouter",
                model="sonnet",
                api_key="sk-new",
                save_to_config=True,
            )
        self.assertTrue(result.success)
        self.assertTrue(result.saved_to_config)
        mock_save.assert_any_call("delegation.provider", "openrouter")
        mock_save.assert_any_call("delegation.model", "sonnet")
        mock_save.assert_any_call("delegation.api_key", "sk-new")

    def test_no_runtime_config_requires_save_flag(self):
        """When cli is not loaded (gateway mode), --save is required to persist."""
        # Ensure cli is not in sys.modules.
        modules = {k: v for k, v in sys.modules.items() if k != "cli"}
        with patch.dict(sys.modules, modules, clear=True), patch(
            "hermes_cli.delegation_switch._persistent_config",
            return_value={"delegation": {"provider": "deepseek", "model": "deepseek-v4-flash"}},
        ), patch(
            "hermes_cli.config.save_config_value",
            return_value=True,
        ) as mock_save:
            # Without --save: rejected with guidance
            result = apply_api_d_switch(
                provider="",
                model="",
                api_key="sk-new",
                save_to_config=False,
            )
        self.assertFalse(result.success)
        self.assertIn("--save", result.message)
        mock_save.assert_not_called()

        # With --save: persisted
        with patch.dict(sys.modules, modules, clear=True), patch(
            "hermes_cli.delegation_switch._persistent_config",
            return_value={"delegation": {"provider": "deepseek", "model": "deepseek-v4-flash"}},
        ), patch(
            "hermes_cli.config.save_config_value",
            return_value=True,
        ) as mock_save2:
            result = apply_api_d_switch(
                provider="",
                model="",
                api_key="sk-new",
                save_to_config=True,
            )
        self.assertTrue(result.success)
        self.assertTrue(result.saved_to_config)
        mock_save2.assert_called_once_with("delegation.api_key", "sk-new")


if __name__ == "__main__":
    unittest.main()
