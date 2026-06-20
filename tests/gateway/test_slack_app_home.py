"""Slack App Home tab — a read-only orientation view (identity, how-to, command list)."""
from __future__ import annotations

from gateway.platforms.slack import build_app_home_view
from hermes_cli.commands import gateway_command_summaries


def test_gateway_command_summaries_nonempty_and_unique():
    summaries = gateway_command_summaries()
    assert len(summaries) > 1
    names = [n for n, _ in summaries]
    assert len(names) == len(set(names))  # canonical names, deduped
    assert all(isinstance(n, str) and isinstance(d, str) for n, d in summaries)


def test_app_home_view_shape_and_content():
    view = build_app_home_view("Welchman", "welchman", [("model", "Set the model"), ("stop", "Stop the run")])
    assert view["type"] == "home"
    blocks = view["blocks"]
    assert blocks[0]["type"] == "header" and "Welchman" in blocks[0]["text"]["text"]
    section_text = " ".join(b.get("text", {}).get("text", "") for b in blocks if b.get("type") == "section")
    assert "/welchman" in section_text  # how-to references the configured command
    assert "`model`" in section_text and "`stop`" in section_text  # commands are listed


def test_app_home_view_respects_slack_caps():
    # a huge command list with long descriptions must still fit Slack's limits
    big = [(f"cmd{i}", "d" * 300) for i in range(500)]
    view = build_app_home_view("Bot", "bot", big)
    assert len(view["blocks"]) <= 100  # home view block cap
    for b in view["blocks"]:
        if b.get("type") == "section":
            assert len(b["text"]["text"]) <= 3000  # section text cap
