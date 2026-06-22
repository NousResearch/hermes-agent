"""Basic unit tests for hermes_cli/delegation_switch.py."""

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
    def test_updates_runtime_config(self):
        cli_config = {"delegation": {"provider": "deepseek", "model": "deepseek-v4-flash"}}
        cli_module = MagicMock()
        cli_module.CLI_CONFIG = cli_config
        with patch.dict("sys.modules", {"cli": cli_module}):
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
        cli_module = MagicMock()
        cli_module.CLI_CONFIG = cli_config
        with patch.dict("sys.modules", {"cli": cli_module}):
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
        cli_module = MagicMock()
        cli_module.CLI_CONFIG = cli_config
        cli_module.save_config_value = MagicMock(return_value=True)
        with patch.dict("sys.modules", {"cli": cli_module}):
            result = apply_api_d_switch(
                provider="openrouter",
                model="sonnet",
                api_key="sk-new",
                save_to_config=True,
            )
        self.assertTrue(result.success)
        self.assertTrue(result.saved_to_config)
        cli_module.save_config_value.assert_any_call("delegation.provider", "openrouter")
        cli_module.save_config_value.assert_any_call("delegation.model", "sonnet")
        cli_module.save_config_value.assert_any_call("delegation.api_key", "sk-new")


if __name__ == "__main__":
    unittest.main()
