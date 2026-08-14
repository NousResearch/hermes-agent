"""Regression tests: CLI display toggles honor quoted YAML ``"false"``.

The CLI init block read each display.* boolean with bare ``bool()`` /
truthiness; ``bool("false")`` is ``True``, so a hand-edited config.yaml
with a quoted value silently KEPT the feature on (focus view, bell,
streaming, timestamps, inline diffs, turn summary, token flow, persistent
output, reasoning recap, compact) against the operator's explicit intent.
"""

from unittest.mock import patch

import cli as _cli_mod
from cli import HermesCLI


def _make_cli(display, **kwargs):
    """Construct HermesCLI with a patched CLI_CONFIG display section."""
    _clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": display,
        "agent": {},
        "terminal": {"env_type": "local"},
    }
    clean_env = {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}
    with (
        patch("cli.get_tool_definitions", return_value=[]),
        patch.dict("os.environ", clean_env, clear=False),
        patch.dict(_cli_mod.__dict__, {"CLI_CONFIG": _clean_config}),
    ):
        return HermesCLI(model="anthropic/claude-opus-4.6", **kwargs)


def test_display_toggles_quoted_false_disables():
    """Quoted ``"false"`` for every display toggle must disable it."""
    display = {
        "compact": "false",
        "focus_view": "false",
        "bell_on_complete": "false",
        "reasoning_full": "false",
        "persistent_output": "false",
        "streaming": "false",
        "timestamps": "false",
        "inline_diffs": "false",
        "turn_summary": "false",
        "spinner_token_flow": "false",
        "battery": "false",
    }
    cli = _make_cli(display)
    assert cli.compact is False
    assert cli._focus_view_enabled is False
    assert cli.bell_on_complete is False
    assert cli.reasoning_full is False
    assert cli.streaming_enabled is False
    assert cli.show_timestamps is False
    assert cli._inline_diffs_enabled is False
    assert cli._turn_summary_enabled is False
    assert cli._spinner_token_flow_enabled is False
    assert cli._battery_visible is False


def test_display_toggles_quoted_true_and_defaults():
    """Quoted ``"true"`` enables; absent keys keep the historical defaults."""
    cli = _make_cli({"streaming": "true", "turn_summary": "true"})
    assert cli.streaming_enabled is True
    assert cli._turn_summary_enabled is True
    # Historical defaults when absent.
    assert cli._inline_diffs_enabled is True
    assert cli.compact is False
    assert cli._focus_view_enabled is False
    assert cli.bell_on_complete is False
    assert cli.show_timestamps is False
    assert cli._spinner_token_flow_enabled is True
