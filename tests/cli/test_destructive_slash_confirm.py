"""Tests for cli.HermesCLI._confirm_destructive_slash.

Drives the helper directly via __get__ on a SimpleNamespace stand-in so we
don't have to construct a full HermesCLI (which requires extensive setup).
"""

from __future__ import annotations

import queue
from types import SimpleNamespace
from unittest.mock import patch


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
