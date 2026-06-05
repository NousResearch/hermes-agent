"""Tests for the delegation-router plugin.

Covers the bundled plugin at ``plugins/delegation-router/``:
  * Routing rule compilation and matching
  * Router state routing logic
  * Hook callback behavior
  * register() wiring with real config loading
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _load_plugin():
    """Import the plugin's __init__.py using the PluginManager naming convention."""
    repo_root = Path(__file__).resolve().parents[2]
    plugin_dir = repo_root / "plugins" / "delegation-router"
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.delegation_router",
        plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.delegation_router"
    mod.__path__ = [str(plugin_dir)]
    sys.modules["hermes_plugins.delegation_router"] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_plugin()
_RoutingRule = _mod._RoutingRule
_RouterState = _mod._RouterState
_on_pre_delegate_build = _mod._on_pre_delegate_build
register = _mod.register


class TestRoutingRule(unittest.TestCase):
    """Unit tests for _RoutingRule matching."""

    def test_matches_simple_pattern(self):
        rule = _RoutingRule(match="review")
        self.assertTrue(rule.matches("Please review this code"))
        self.assertFalse(rule.matches("Write a function"))

    def test_matches_case_insensitive(self):
        rule = _RoutingRule(match="REVIEW")
        self.assertTrue(rule.matches("please review"))

    def test_matches_word_boundary(self):
        rule = _RoutingRule(match=r"\btest\b")
        self.assertTrue(rule.matches("run the test"))
        self.assertFalse(rule.matches("latest feature"))

    def test_matches_regex(self):
        rule = _RoutingRule(match=r"\b(code.?review|security)\b")
        self.assertTrue(rule.matches("do a code review"))
        self.assertTrue(rule.matches("security audit"))
        self.assertFalse(rule.matches("write code"))

    def test_to_override_model_only(self):
        rule = _RoutingRule(match="x", model="gpt-4")
        self.assertEqual(rule.to_override(), {"model": "gpt-4"})

    def test_to_override_full(self):
        rule = _RoutingRule(
            match="x", model="gpt-4", provider="openrouter", reasoning_effort="high"
        )
        result = rule.to_override()
        self.assertEqual(result["model"], "gpt-4")
        self.assertEqual(result["provider"], "openrouter")
        self.assertEqual(result["reasoning_effort"], "high")

    def test_to_override_empty_when_no_targets(self):
        rule = _RoutingRule(match="x")
        self.assertEqual(rule.to_override(), {})

    def test_matches_none_goal(self):
        rule = _RoutingRule(match="review")
        self.assertFalse(rule.matches(None))

    def test_coerces_non_string_model(self):
        rule = _RoutingRule(match="x", model=123)
        # Non-string values are rejected, not coerced
        self.assertEqual(rule.to_override(), {})


class TestRouterState(unittest.TestCase):
    """Unit tests for _RouterState routing logic."""

    def test_reload_compiles_rules(self):
        router = _RouterState()
        router.reload({
            "rules": [
                {"match": "review", "model": "gpt-4"},
                {"match": "test", "model": "haiku"},
            ]
        })
        self.assertEqual(len(router.rules), 2)

    def test_reload_skips_invalid_regex(self):
        router = _RouterState()
        router.reload({
            "rules": [
                {"match": "[invalid", "model": "gpt-4"},
                {"match": "review", "model": "haiku"},
            ]
        })
        self.assertEqual(len(router.rules), 1)

    def test_reload_skips_entries_without_match(self):
        router = _RouterState()
        router.reload({
            "rules": [
                {"model": "gpt-4"},
                {"match": "", "model": "haiku"},
            ]
        })
        self.assertEqual(len(router.rules), 0)

    def test_reload_handles_non_dict_config(self):
        router = _RouterState()
        router.reload("not a dict")
        self.assertEqual(len(router.rules), 0)

    def test_reload_handles_none_config(self):
        router = _RouterState()
        router.reload(None)
        self.assertEqual(len(router.rules), 0)

    def test_reload_handles_non_list_rules(self):
        router = _RouterState()
        router.reload({"rules": "not a list"})
        self.assertEqual(len(router.rules), 0)

    def test_reload_handles_non_dict_default(self):
        router = _RouterState()
        router.reload({"default": "not a dict"})
        self.assertEqual(len(router.rules), 0)
        self.assertEqual(router.default, {})

    def test_reload_skips_non_dict_rule_entry(self):
        router = _RouterState()
        router.reload({"rules": ["not a dict", 42, None]})
        self.assertEqual(len(router.rules), 0)

    def test_reload_skips_non_string_match(self):
        router = _RouterState()
        router.reload({"rules": [{"match": 123, "model": "gpt-4"}]})
        self.assertEqual(len(router.rules), 0)

    def test_route_first_match_wins(self):
        router = _RouterState()
        router.reload({
            "rules": [
                {"match": "review", "model": "first"},
                {"match": "review", "model": "second"},
            ]
        })
        result = router.route("please review this")
        self.assertEqual(result["model"], "first")

    def test_route_no_match_returns_default(self):
        router = _RouterState()
        router.reload({
            "rules": [{"match": "review", "model": "gpt-4"}],
            "default": {"model": "default-model"},
        })
        result = router.route("write a function")
        self.assertEqual(result["model"], "default-model")

    def test_route_no_match_no_default_returns_none(self):
        router = _RouterState()
        router.reload({
            "rules": [{"match": "review", "model": "gpt-4"}],
        })
        self.assertIsNone(router.route("write a function"))

    def test_route_empty_rules_returns_none(self):
        router = _RouterState()
        router.reload({})
        self.assertIsNone(router.route("anything"))

    def test_route_match_with_empty_override_returns_none(self):
        router = _RouterState()
        router.reload({
            "rules": [{"match": "review"}],
        })
        self.assertIsNone(router.route("review this"))

    def test_route_default_with_empty_values_ignored(self):
        router = _RouterState()
        router.reload({
            "default": {"model": None, "provider": ""},
        })
        self.assertIsNone(router.route("anything"))

    def test_route_default_returns_copy(self):
        """Default override dict should be a copy, not a reference."""
        router = _RouterState()
        router.reload({"default": {"model": "x"}})
        result1 = router.route("a")
        result2 = router.route("b")
        self.assertIsNot(result1, result2)


class TestHookCallback(unittest.TestCase):
    """Tests for the _on_pre_delegate_build hook callback."""

    def test_returns_override_for_matching_goal(self):
        router = _RouterState()
        router.reload({"rules": [{"match": "review", "model": "gpt-4"}]})
        original = _mod._router
        _mod._router = router
        try:
            result = _on_pre_delegate_build(goal="please review this")
            self.assertEqual(result["model"], "gpt-4")
        finally:
            _mod._router = original

    def test_returns_none_for_non_matching_goal(self):
        router = _RouterState()
        router.reload({"rules": [{"match": "review", "model": "gpt-4"}]})
        original = _mod._router
        _mod._router = router
        try:
            result = _on_pre_delegate_build(goal="write a function")
            self.assertIsNone(result)
        finally:
            _mod._router = original

    def test_passes_goal_from_kwargs(self):
        router = _RouterState()
        router.reload({"rules": [{"match": "deploy", "model": "fast-model"}]})
        original = _mod._router
        _mod._router = router
        try:
            result = _on_pre_delegate_build(
                goal="deploy to production",
                context="some context",
                model="original",
            )
            self.assertEqual(result["model"], "fast-model")
        finally:
            _mod._router = original


class TestRegister(unittest.TestCase):
    """Tests for the plugin register() function."""

    def _make_ctx(self):
        return MagicMock()

    @patch("hermes_cli.config.load_config")
    def test_register_with_config(self, mock_load):
        mock_load.return_value = {
            "plugins": {
                "delegation_router": {
                    "enabled": True,
                    "rules": [{"match": "review", "model": "gpt-4"}],
                }
            }
        }
        ctx = self._make_ctx()
        register(ctx)
        ctx.register_hook.assert_called_once()
        args = ctx.register_hook.call_args
        self.assertEqual(args[0][0], "pre_delegate_build")

    @patch("hermes_cli.config.load_config")
    def test_register_disabled_by_config(self, mock_load):
        mock_load.return_value = {
            "plugins": {"delegation_router": {"enabled": False}}
        }
        ctx = self._make_ctx()
        register(ctx)
        ctx.register_hook.assert_not_called()

    @patch("hermes_cli.config.load_config")
    def test_register_handles_missing_config(self, mock_load):
        mock_load.side_effect = RuntimeError("no config")
        ctx = self._make_ctx()
        register(ctx)
        # Should still register with empty rules (graceful fallback)
        ctx.register_hook.assert_called_once()

    @patch("hermes_cli.config.load_config")
    def test_register_with_no_plugin_config(self, mock_load):
        mock_load.return_value = {"plugins": {}}
        ctx = self._make_ctx()
        register(ctx)
        # Enabled by default when no explicit config
        ctx.register_hook.assert_called_once()

    @patch("hermes_cli.config.load_config")
    def test_register_with_malformed_rules(self, mock_load):
        mock_load.return_value = {
            "plugins": {
                "delegation_router": {
                    "rules": "not a list",
                    "default": "not a dict",
                }
            }
        }
        ctx = self._make_ctx()
        register(ctx)
        ctx.register_hook.assert_called_once()


if __name__ == "__main__":
    unittest.main()
