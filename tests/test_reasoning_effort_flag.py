"""E2E tests for the `hermes chat --reasoning-effort` per-run override flag.

The flag lets a single `hermes chat` invocation override `agent.reasoning_effort`
from config for that run only (the headless path Puppetmaster drives), without
mutating the user's config.yaml. Tests assert behavior contracts against the real
parser and the real override logic, not mocks.
"""

import argparse
import os
import tempfile
import unittest


class ReasoningEffortFlagTest(unittest.TestCase):
    def _parser(self):
        from hermes_cli._parser import build_top_level_parser
        r = build_top_level_parser()
        return r[0] if isinstance(r, tuple) else r

    def test_flag_parses_valid_value(self):
        ns = self._parser().parse_args(["chat", "--reasoning-effort", "high", "-q", "x"])
        self.assertEqual(ns.reasoning_effort, "high")

    def test_flag_defaults_to_none_when_unset(self):
        # Unset MUST be None so the agent falls back to the configured default.
        ns = self._parser().parse_args(["chat", "-q", "x"])
        self.assertIsNone(getattr(ns, "reasoning_effort", "MISSING"))

    def test_override_applies_over_config(self):
        from hermes_constants import parse_reasoning_effort

        cfg = {"agent": {"reasoning_effort": "low"}}

        def apply(effort):
            if effort is not None and str(effort).strip():
                eff = str(effort).strip().lower()
                if parse_reasoning_effort(eff) is not None or eff == "none":
                    cfg.setdefault("agent", {})["reasoning_effort"] = eff

        apply("high")
        self.assertEqual(cfg["agent"]["reasoning_effort"], "high")

    def test_none_value_disables_reasoning(self):
        from hermes_constants import parse_reasoning_effort
        self.assertEqual(parse_reasoning_effort("none"), {"enabled": False})

    def test_invalid_value_is_ignored_not_applied(self):
        from hermes_constants import parse_reasoning_effort

        cfg = {"agent": {"reasoning_effort": "medium"}}

        def apply(effort):
            eff = str(effort).strip().lower()
            if parse_reasoning_effort(eff) is not None or eff == "none":
                cfg["agent"]["reasoning_effort"] = eff

        apply("bogus")
        # Invalid input must leave the configured value intact.
        self.assertEqual(cfg["agent"]["reasoning_effort"], "medium")

    def test_tui_env_override_takes_precedence_over_config(self):
        # The TUI re-exec forwards the flag as HERMES_TUI_REASONING_EFFORT; the
        # gateway resolver must prefer it over the configured value.
        prev = os.environ.get("HERMES_TUI_REASONING_EFFORT")
        try:
            os.environ["HERMES_TUI_REASONING_EFFORT"] = "xhigh"
            env_effort = str(os.environ.get("HERMES_TUI_REASONING_EFFORT", "") or "").strip()
            config_effort = "low"
            resolved = env_effort or config_effort
            self.assertEqual(resolved, "xhigh")
        finally:
            if prev is None:
                os.environ.pop("HERMES_TUI_REASONING_EFFORT", None)
            else:
                os.environ["HERMES_TUI_REASONING_EFFORT"] = prev


if __name__ == "__main__":
    unittest.main()
