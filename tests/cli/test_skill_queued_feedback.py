"""Regression tests for skill-command queue feedback while the agent is busy (#83209).

A skill command carrying a user instruction (e.g. ``/my-skill review this
PR``) submitted while the agent is running used to be queued silently into
``_pending_input``: the user saw only "⚡ Loading skill", got no feedback
that their instruction was registered, and re-sending replayed N copies of
the same expanded turn. The fix prints explicit "Queued for the next turn
(skill): <instruction>" feedback when the agent is busy.

These tests exercise the detector without starting a prompt_toolkit app.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

from cli import _command_base


def _make_cli():
    """Create a HermesCLI instance with prompt_toolkit stubbed out.

    Returns ``(cli, module)``. The module is returned so tests can
    ``patch.object(module, ...)``: the instance's methods close over THIS
    module object's globals, and ``patch.dict(sys.modules, ...)`` (used to
    stub prompt_toolkit) removes ``cli`` from ``sys.modules`` on exit, so a
    string-target ``patch("cli.X")`` would re-import a *fresh* module the
    instance never references.
    """
    _clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": "all"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }
    clean_env = {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}
    prompt_toolkit_stubs = {
        "prompt_toolkit": MagicMock(),
        "prompt_toolkit.history": MagicMock(),
        "prompt_toolkit.styles": MagicMock(),
        "prompt_toolkit.patch_stdout": MagicMock(),
        "prompt_toolkit.application": MagicMock(),
        "prompt_toolkit.layout": MagicMock(),
        "prompt_toolkit.layout.processors": MagicMock(),
        "prompt_toolkit.filters": MagicMock(),
        "prompt_toolkit.layout.dimension": MagicMock(),
        "prompt_toolkit.layout.menus": MagicMock(),
        "prompt_toolkit.widgets": MagicMock(),
        "prompt_toolkit.key_binding": MagicMock(),
        "prompt_toolkit.completion": MagicMock(),
        "prompt_toolkit.formatted_text": MagicMock(),
        "prompt_toolkit.auto_suggest": MagicMock(),
    }
    with patch.dict(sys.modules, prompt_toolkit_stubs), patch.dict(
        "os.environ", clean_env, clear=False
    ):
        import cli as _cli_mod

        _cli_mod = importlib.reload(_cli_mod)
        with patch.object(_cli_mod, "get_tool_definitions", return_value=[]), patch.dict(
            _cli_mod.__dict__, {"CLI_CONFIG": _clean_config}
        ):
            return _cli_mod.HermesCLI(), _cli_mod


class TestSkillCommandInstructionDetector:
    """_skill_command_instruction recognizes skill commands with payloads.

    NOTE: scan_skill_commands() keys are slash-prefixed (``"/my-skill"``),
    matching the production dispatch check (cli.py ``base_cmd in
    skill_commands``). Tests must use that format — a slash-less stub once
    masked a membership bug that disabled the feedback in production.
    """

    def test_returns_payload_for_skill_command_with_instruction(self):
        cli, mod = _make_cli()
        with patch.object(
            mod, "_ensure_skill_commands",
            return_value={"/my-skill": {"name": "my-skill"}},
        ), patch("hermes_cli.commands.resolve_command", return_value=None):
            assert cli._skill_command_instruction("/my-skill review this PR") == "review this PR"

    def test_matches_slash_prefixed_skill_keys_like_production(self):
        """Regression: slash-less comparison made the feedback dead code."""
        cli, mod = _make_cli()
        with patch.object(
            mod, "_ensure_skill_commands",
            return_value={"/gif-search": {"name": "gif-search"}},
        ), patch("hermes_cli.commands.resolve_command", return_value=None):
            assert cli._skill_command_instruction("/gif-search find a cat gif") == "find a cat gif"

    def test_returns_empty_for_skill_command_without_payload(self):
        cli, mod = _make_cli()
        with patch.object(
            mod, "_ensure_skill_commands",
            return_value={"/my-skill": {"name": "my-skill"}},
        ), patch("hermes_cli.commands.resolve_command", return_value=None):
            assert cli._skill_command_instruction("/my-skill") == ""

    def test_returns_empty_for_builtin_command(self):
        cli, mod = _make_cli()
        # resolve_command returns a truthy CommandDef for built-ins.
        with patch.object(
            mod, "_ensure_skill_commands",
            return_value={"/steer": {"name": "steer"}},
        ), patch(
            "hermes_cli.commands.resolve_command",
            return_value=MagicMock(name="steer"),
        ):
            assert cli._skill_command_instruction("/steer focus on errors") == ""

    def test_returns_empty_for_plain_text(self):
        cli, _mod = _make_cli()
        assert cli._skill_command_instruction("review this PR") == ""

    def test_returns_empty_for_unknown_slash_command(self):
        cli, mod = _make_cli()
        with patch.object(mod, "_ensure_skill_commands", return_value={}), patch(
            "hermes_cli.commands.resolve_command", return_value=None
        ):
            assert cli._skill_command_instruction("/no-such-skill do something") == ""

    def test_command_base_strips_slash_and_lowercases(self):
        assert _command_base("/My-Skill foo") == "my-skill"
        assert _command_base("/steer focus") == "steer"
        assert _command_base("") == ""
