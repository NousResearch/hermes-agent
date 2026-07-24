"""Regression coverage for machine-readable MoA quiet output."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

from agent.agent_init import _relay_moa_reference_event


class MoAQuietReferenceOutputTests(unittest.TestCase):
    @staticmethod
    def _agent(*, platform: str, tool_progress_mode: str, quiet_mode: bool = True):
        calls = []

        def callback(*args, **kwargs):
            calls.append((args, kwargs))

        return SimpleNamespace(
            platform=platform,
            tool_progress_mode=tool_progress_mode,
            quiet_mode=quiet_mode,
            tool_progress_callback=callback,
        ), calls

    def test_machine_readable_cli_suppresses_reference_relay(self) -> None:
        agent, calls = self._agent(platform="cli", tool_progress_mode="off")
        _relay_moa_reference_event(
            agent,
            "moa.reference",
            label="local:advisor",
            text="hidden",
            index=1,
            count=1,
        )
        self.assertEqual(calls, [])

    def test_interactive_cli_delivers_reference_relay(self) -> None:
        agent, calls = self._agent(
            platform="cli",
            tool_progress_mode="all",
            quiet_mode=True,
        )
        _relay_moa_reference_event(
            agent,
            "moa.reference",
            label="local:advisor",
            text="visible",
            index=1,
            count=2,
        )
        self.assertEqual(
            calls,
            [
                (
                    ("moa.reference", "local:advisor", "visible", None),
                    {"moa_index": 1, "moa_count": 2},
                )
            ],
        )

    def test_gateway_delivers_even_when_progress_mode_is_off(self) -> None:
        agent, calls = self._agent(platform="discord", tool_progress_mode="off")
        _relay_moa_reference_event(
            agent,
            "moa.aggregating",
            aggregator="local:aggregator",
            ref_count=2,
        )
        self.assertEqual(
            calls,
            [
                (
                    ("moa.aggregating", "local:aggregator", None, None),
                    {"moa_ref_count": 2},
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
