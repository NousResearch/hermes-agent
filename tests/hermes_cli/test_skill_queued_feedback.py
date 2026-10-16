"""Regression tests for skill-command queue feedback while the agent is busy (#83209).

A skill command carrying a user instruction (e.g. ``/my-skill review this
PR``) submitted while the agent is running used to be queued silently into
``_pending_input``: the user got no feedback that their instruction was
registered, and re-sending replayed N copies of the same expanded turn. The
busy-enter path now prints explicit "Queued for the next turn (skill):
<instruction>" feedback.

The wiring tests drive the real ``_tui_handle_enter`` with a real
prompt_toolkit ``Buffer`` (mirroring tests/hermes_cli/test_tui_rapid_enter_paste.py).
The detector tests exercise ``cli._skill_command_instruction`` directly, with
skill-map keys in the production shape.
"""

from __future__ import annotations

import queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from prompt_toolkit.buffer import Buffer

SKILLS = {"/my-skill": {"name": "my-skill"}}


def _shell(*, agent_running: bool):
    """HermesCLI with the minimal surface ``_tui_handle_enter`` touches."""
    from cli import HermesCLI

    shell = object.__new__(HermesCLI)
    shell._tui_enter_overlay = lambda event: False
    shell._tui_multiline_shortcuts = False
    shell._attached_images = []
    shell._agent_running = agent_running
    shell._tui_enter_while_busy = MagicMock()
    shell._tui_enter_inline_command = lambda *a, **k: False
    shell._inline_pastes = lambda buf: None
    shell._pending_input = queue.Queue()
    shell._tui_last_text_change = 0.0
    return shell


def _event(buf: Buffer):
    return SimpleNamespace(app=SimpleNamespace(current_buffer=buf, invalidate=lambda: None, is_running=False))


def _submit(shell, text: str) -> list[str]:
    """Type *text* into a live Buffer, press Enter, return the printed lines."""
    import cli as cli_mod

    printed: list[str] = []
    buf = Buffer()
    buf.insert_text(text)
    with (
        patch.object(cli_mod, "_cprint", side_effect=printed.append),
        patch.object(cli_mod, "_DIM", ""),
        patch.object(cli_mod, "_RST", ""),
        patch.object(cli_mod, "_ensure_skill_commands", return_value=SKILLS),
    ):
        shell._tui_handle_enter(_event(buf))
    return printed


class TestBusyEnterSkillFeedback:
    """The real enter path prints queue feedback for a busy skill submit."""

    def test_busy_skill_submit_prints_queue_feedback(self):
        shell = _shell(agent_running=True)
        printed = _submit(shell, "/my-skill review this PR")
        assert any("Queued for the next turn (skill): review this PR" in line for line in printed), printed
        assert shell._pending_input.get_nowait() == "/my-skill review this PR"

    def test_busy_skill_submit_without_instruction_prints_no_feedback(self):
        shell = _shell(agent_running=True)
        printed = _submit(shell, "/my-skill")
        assert not any("Queued for the next turn (skill)" in line for line in printed), printed

    def test_busy_non_skill_slash_command_prints_no_feedback(self):
        shell = _shell(agent_running=True)
        printed = _submit(shell, "/unknown-command do things")
        assert not any("Queued for the next turn (skill)" in line for line in printed), printed

    def test_idle_skill_submit_prints_no_feedback(self):
        """Idle submits keep the normal dispatch path - no queue feedback."""
        shell = _shell(agent_running=False)
        printed = _submit(shell, "/my-skill review this PR")
        assert not any("Queued for the next turn (skill)" in line for line in printed), printed

    def test_long_instruction_preview_is_truncated(self):
        shell = _shell(agent_running=True)
        printed = _submit(shell, "/my-skill review " + "x" * 200)
        assert any("Queued for the next turn (skill): review " in line and line.endswith("...") for line in printed), printed


class TestSkillCommandInstructionDetector:
    """``cli._skill_command_instruction`` matches the production skill-map shape.

    NOTE: scan_skill_commands() keys are slash-prefixed lowercased slugs
    (``"/my-skill"``), matching the production dispatch check (cli.py
    ``base_cmd in skill_commands``). Tests must use that format - a
    slash-less stub once masked a membership bug that disabled the feedback
    in production.
    """

    def _detect(self, text: str, skills=SKILLS) -> str:
        import cli as cli_mod

        with patch.object(cli_mod, "_ensure_skill_commands", return_value=skills):
            return cli_mod._skill_command_instruction(text)

    def test_returns_payload_for_skill_command_with_instruction(self):
        assert self._detect("/my-skill review this PR") == "review this PR"

    def test_slash_prefixed_keys_like_production(self):
        """Regression: the membership check must use the slash-prefixed key form."""
        skills = {"/gif-search": {"name": "gif-search"}}
        assert self._detect("/gif-search find a cat gif", skills) == "find a cat gif"

    def test_case_insensitive_key_lookup(self):
        """Regression: a case-preserving skill-map key must not disable the branch."""
        case_keys = {"/My-Skill": {"name": "My-Skill"}}
        assert self._detect("/My-Skill review", case_keys) == "review"

    def test_returns_empty_for_skill_command_without_payload(self):
        assert self._detect("/my-skill") == ""

    def test_returns_empty_for_builtin_command(self):
        # /model resolves in the real command registry, so it is not a skill.
        assert self._detect("/model opus review this") == ""

    def test_returns_empty_for_plain_text(self):
        assert self._detect("review this PR") == ""

    def test_returns_empty_for_unknown_slash_command(self):
        assert self._detect("/no-such-skill do something") == ""

    def test_command_base_strips_slash_and_lowercases(self):
        import cli as cli_mod

        assert cli_mod._command_base("/My-Skill foo") == "my-skill"
        assert cli_mod._command_base("/steer focus") == "steer"
        assert cli_mod._command_base("") == ""
