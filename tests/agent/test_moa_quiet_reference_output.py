"""MoA display events honour the ``-Q`` quiet contract.

``-Q`` (machine-readable CLI output) nulls ``agent.tool_progress_callback`` and sets
``tool_progress_mode = "off"``; the MoA reference relay reads the callback at emit time, so
quiet sessions must emit nothing and interactive sessions must relay every event.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.moa_loop import build_moa_facade


def _facade(agent):
    with patch("agent.moa_loop.MoAClient") as client_cls:
        build_moa_facade(agent, "default")
    return client_cls.call_args.kwargs["reference_callback"]


def test_quiet_cli_emits_no_moa_display_events():
    agent = SimpleNamespace(platform="cli", tool_progress_mode="off", tool_progress_callback=None, provider="moa", model="default")
    relay = _facade(agent)
    relay("moa.reference", label="m1", text="answer", index=0, count=2)  # must not raise
    relay("moa.aggregating", aggregator="agg")


@pytest.mark.parametrize("platform", ["cli", "telegram"])
def test_interactive_surfaces_receive_moa_events(platform):
    cb = MagicMock()
    agent = SimpleNamespace(platform=platform, tool_progress_mode="all", tool_progress_callback=cb, provider="moa", model="default")
    relay = _facade(agent)
    relay("moa.reference", label="m1", text="answer", index=0, count=2)
    relay("moa.aggregating", aggregator="agg")
    events = [c.args[0] for c in cb.call_args_list]
    assert events == ["moa.reference", "moa.aggregating"]
    assert cb.call_args_list[0].kwargs == {"moa_index": 0, "moa_count": 2}
