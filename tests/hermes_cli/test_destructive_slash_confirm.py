"""Tests for cli.HermesCLI._confirm_destructive_slash.

Drives the helper directly via __get__ on a SimpleNamespace stand-in so we
don't have to construct a full HermesCLI (which requires extensive setup).
"""

from __future__ import annotations

import os
import queue
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from prompt_toolkit.utils import get_cwidth


def _bound(fn, instance):
    """Bind an unbound method to a stand-in instance."""
    return fn.__get__(instance, type(instance))


def _make_self(prompt_response):
    """Build a minimal stand-in 'self' for _confirm_destructive_slash."""
    from cli import HermesCLI

    self_ = SimpleNamespace(
        _app=None,
        _prompt_text_input=lambda _prompt: prompt_response,
        _prompt_text_input_modal=lambda **_kw: prompt_response,
    )
    self_._normalize_slash_confirm_choice = _bound(
        HermesCLI._normalize_slash_confirm_choice, self_,
    )
    return self_


@pytest.mark.parametrize(
    ("choice_count", "quick_pick_count"),
    ((2, 2), (3, 3), (4, 4), (10, 9)),
)
def test_slash_confirm_hints_match_immediate_numeric_choices(choice_count, quick_pick_count):
    from cli import HermesCLI

    choices = [(str(index), f"Choice {index + 1}", "detail") for index in range(choice_count)]
    cli = HermesCLI.__new__(HermesCLI)
    setattr(cli, "_slash_confirm_state", {
        "title": "Choose",
        "detail": "Pick one.",
        "choices": choices,
        "selected": 0,
    })
    setattr(cli, "_slash_confirm_deadline", time.monotonic() + 120)
    cli._sudo_state = cli._secret_state = cli._approval_state = None
    cli._clarify_state = None
    cli._clarify_freetext = cli._voice_recording = cli._voice_processing = False
    cli._command_running = cli._agent_running = cli._voice_mode = False

    numbers = "/".join(str(index) for index in range(1, quick_pick_count + 1))
    hint = f"{numbers} quick pick · ↑/↓ then Enter"
    with patch("cli.shutil.get_terminal_size", return_value=os.terminal_size((80, 40))):
        panel = "".join(text for _style, text in cli._get_slash_confirm_display_fragments())
    countdown = "".join(text for _style, text in cli._tui_hint_text())

    rows = panel.splitlines()
    assert all(get_cwidth(row) <= get_cwidth(rows[0]) for row in rows)
    assert f"{hint} · Esc/Ctrl+C cancel" in panel
    assert hint in countdown
    assert cli._tui_placeholder_text() == hint
    assert "10" not in hint

    cli._submit_slash_confirm_response = MagicMock()
    buffer = SimpleNamespace(reset=MagicMock())
    event = SimpleNamespace(app=SimpleNamespace(current_buffer=buffer, invalidate=MagicMock()))
    cli._tui_make_slash_confirm_number_handler(quick_pick_count - 1)(event)

    cli._submit_slash_confirm_response.assert_called_once_with(choices[quick_pick_count - 1][0])
    buffer.reset.assert_called_once_with()




def test_gate_on_choice_once_returns_once():
    """When the gate is on and the user picks '1', return 'once'."""
    from cli import HermesCLI

    self_ = _make_self(prompt_response="1")

    with patch(
        "cli.load_cli_config",
        return_value={"approvals": {"destructive_slash_confirm": True}},
    ):
        result = _bound(HermesCLI._confirm_destructive_slash, self_)(
            "clear", "detail",
        )

    assert result == "once"








def test_gate_on_choice_always_persists_and_returns_always():
    """User picks 'always' → returns 'always' AND
    save_config_value('approvals.destructive_slash_confirm', False) was called."""
    from cli import HermesCLI

    self_ = _make_self(prompt_response="2")

    saves = []
    def _fake_save(key, value):
        saves.append((key, value))
        return True

    with patch(
        "cli.load_cli_config",
        return_value={"approvals": {"destructive_slash_confirm": True}},
    ), patch("cli.save_config_value", _fake_save):
        result = _bound(HermesCLI._confirm_destructive_slash, self_)(
            "clear", "detail",
        )

    assert result == "always"
    assert ("approvals.destructive_slash_confirm", False) in saves








# ---------------------------------------------------------------------------
# Inline-skip escape hatch (issue #30768)
#
# Users on platforms where the prompt_toolkit modal doesn't dispatch keys
# (currently native Windows PowerShell) need a way to bypass the confirmation
# without flipping the config gate.  ``/reset now``, ``/new --yes``, ``/clear
# -y`` all skip the modal and return "once" immediately.
# ---------------------------------------------------------------------------


def test_split_destructive_skip_recognized_tokens():
    """``now``, ``--yes``, and ``-y`` are recognized as skip tokens."""
    from cli import HermesCLI

    assert HermesCLI._split_destructive_skip("/reset now") == ("", True)
    assert HermesCLI._split_destructive_skip("/clear --yes") == ("", True)
    assert HermesCLI._split_destructive_skip("/undo -y") == ("", True)






def test_split_destructive_skip_handles_empty_and_none():
    """Defensive against missing/empty input."""
    from cli import HermesCLI

    assert HermesCLI._split_destructive_skip(None) == ("", False)
    assert HermesCLI._split_destructive_skip("") == ("", False)
    assert HermesCLI._split_destructive_skip("   ") == ("", False)


def test_confirm_destructive_slash_now_skips_modal():
    """``/reset now`` skips the modal even when the gate is on."""
    from cli import HermesCLI

    # Build a prompt stub that fails the test if invoked — proving the modal
    # was never reached.
    def _explode(**_kw):
        raise AssertionError("modal must not be invoked when inline-skip present")

    self_ = SimpleNamespace(
        _app=None,
        _prompt_text_input_modal=_explode,
    )
    self_._normalize_slash_confirm_choice = _bound(
        HermesCLI._normalize_slash_confirm_choice, self_,
    )
    self_._split_destructive_skip = HermesCLI._split_destructive_skip  # classmethod

    with patch(
        "cli.load_cli_config",
        return_value={"approvals": {"destructive_slash_confirm": True}},
    ):
        result = _bound(HermesCLI._confirm_destructive_slash, self_)(
            "new", "detail", cmd_original="/reset now",
        )

    assert result == "once"


def test_confirm_destructive_slash_yes_flag_skips_modal():
    """``--yes`` flag is equivalent to ``now``."""
    from cli import HermesCLI

    def _explode(**_kw):
        raise AssertionError("modal must not be invoked when --yes present")

    self_ = SimpleNamespace(
        _app=None,
        _prompt_text_input_modal=_explode,
    )
    self_._normalize_slash_confirm_choice = _bound(
        HermesCLI._normalize_slash_confirm_choice, self_,
    )
    self_._split_destructive_skip = HermesCLI._split_destructive_skip

    with patch(
        "cli.load_cli_config",
        return_value={"approvals": {"destructive_slash_confirm": True}},
    ):
        result = _bound(HermesCLI._confirm_destructive_slash, self_)(
            "new", "detail", cmd_original="/new --yes My Session",
        )

    assert result == "once"


def test_confirm_destructive_slash_no_skip_token_still_prompts():
    """Without a skip token the gate-on path still consults the modal."""
    from cli import HermesCLI

    self_ = _make_self(prompt_response="3")  # cancel
    self_._split_destructive_skip = HermesCLI._split_destructive_skip

    with patch(
        "cli.load_cli_config",
        return_value={"approvals": {"destructive_slash_confirm": True}},
    ):
        result = _bound(HermesCLI._confirm_destructive_slash, self_)(
            "new", "detail", cmd_original="/new My Session",
        )

    # Prompt was reached and returned cancel → None.
    assert result is None
