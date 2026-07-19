"""Regression coverage for machine-readable MoA quiet output."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

from agent.agent_init import _moa_reference_output_allowed


class MoAQuietReferenceOutputTests(unittest.TestCase):
    def test_quiet_agent_suppresses_moa_reference_display(self) -> None:
        self.assertFalse(_moa_reference_output_allowed(SimpleNamespace(quiet_mode=True)))

    def test_interactive_agent_keeps_moa_reference_display(self) -> None:
        self.assertTrue(_moa_reference_output_allowed(SimpleNamespace(quiet_mode=False)))

    def test_missing_quiet_flag_keeps_existing_interactive_behavior(self) -> None:
        self.assertTrue(_moa_reference_output_allowed(SimpleNamespace()))


if __name__ == "__main__":
    unittest.main()
