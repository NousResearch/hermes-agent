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


def test_gate_off_returns_once_without_prompting():
    """When approvals.destructive_slash_confirm is False, return 'once'
    immediately (caller proceeds without showing a prompt)."""
    from cli import HermesCLI

    self_ = _make_self(prompt_response="should not be called")

    with patch(
        "cli.load_cli_config",
        return_value={"approvals": {"destructive_slash_confirm": False}},
    ):
        result = _bound(HermesCLI._confirm_destructive_slash, self_)(
            "clear", "detail",
        )

    assert result == "once"


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


def test_destructive_slash_modal_uses_text_warning_glyph():
    """Kitty must not receive an emoji variation selector in modal titles."""
    from cli import HermesCLI

    captured = {}
    self_ = _make_self(prompt_response="3")

    def _capture_modal(**kwargs):
        captured.update(kwargs)
        return "3"

    self_._prompt_text_input_modal = _capture_modal
    with patch(
        "cli.load_cli_config",
        return_value={"approvals": {"destructive_slash_confirm": True}},
    ):
        _bound(HermesCLI._confirm_destructive_slash, self_)("new", "detail")

    assert captured["title"] == "⚠  /new — destroys conversation state"


def test_reload_mcp_modal_uses_text_warning_glyph():
    """The MCP confirmation title must also avoid emoji-width rendering."""
    from cli import HermesCLI

    captured = {}
    self_ = SimpleNamespace(
        _prompt_text_input_modal=lambda **kwargs: captured.update(kwargs) or "cancel",
    )
    self_._normalize_slash_confirm_choice = _bound(
        HermesCLI._normalize_slash_confirm_choice, self_,
    )

    with patch(
        "cli.load_cli_config",
        return_value={"approvals": {"mcp_reload_confirm": True}},
    ):
        _bound(HermesCLI._confirm_and_reload_mcp, self_)()

    assert captured["title"] == "⚠  /reload-mcp — Prompt cache invalidation warning"


def test_gate_on_choice_cancel_returns_none():
    """When the user picks '3' (cancel), return None — caller must abort."""
    from cli import HermesCLI

    self_ = _make_self(prompt_response="3")

    with patch(
        "cli.load_cli_config",
        return_value={"approvals": {"destructive_slash_confirm": True}},
    ):
        result = _bound(HermesCLI._confirm_destructive_slash, self_)(
            "clear", "detail",
        )

    assert result is None


def test_gate_on_no_input_returns_none():
    """No input (None / EOF / Ctrl-C) treated as cancel."""
    from cli import HermesCLI

    self_ = _make_self(prompt_response=None)

    with patch(
        "cli.load_cli_config",
        return_value={"approvals": {"destructive_slash_confirm": True}},
    ):
        result = _bound(HermesCLI._confirm_destructive_slash, self_)(
            "clear", "detail",
        )

    assert result is None


def test_gate_on_unknown_choice_returns_none():
    """Garbage input is treated as cancel — fail safe, don't destroy state."""
    from cli import HermesCLI

    self_ = _make_self(prompt_response="maybe")

    with patch(
        "cli.load_cli_config",
        return_value={"approvals": {"destructive_slash_confirm": True}},
    ):
        result = _bound(HermesCLI._confirm_destructive_slash, self_)(
            "clear", "detail",
        )

    assert result is None


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


def test_gate_default_true_when_config_missing():
    """If load_cli_config raises or returns malformed data, treat as
    'gate on' (default safe) — must prompt."""
    from cli import HermesCLI

    self_ = _make_self(prompt_response="3")  # cancel

    with patch("cli.load_cli_config", side_effect=Exception("boom")):
        result = _bound(HermesCLI._confirm_destructive_slash, self_)(
            "clear", "detail",
        )

    # Got prompted (returned None from cancel) — meaning the gate was
    # treated as on despite the config error.  If the gate had been off
    # this would have returned 'once' without consulting the prompt.
    assert result is None


def test_slash_confirm_modal_number_selection_submits_without_raw_input():
    """Pressing 2 in the TUI modal should resolve to Always Approve directly."""
    from cli import HermesCLI

    q = queue.Queue()
    self_ = SimpleNamespace(
        _slash_confirm_state={
            "choices": [
                ("once", "Approve Once", "proceed once"),
                ("always", "Always Approve", "persist opt-out"),
                ("cancel", "Cancel", "abort"),
            ],
            "selected": 0,
            "response_queue": q,
        },
        _slash_confirm_deadline=123,
        _invalidate=lambda: None,
    )

    _bound(HermesCLI._submit_slash_confirm_response, self_)("always")

    assert q.get_nowait() == "always"
    assert self_._slash_confirm_state is None
    assert self_._slash_confirm_deadline == 0


def test_slash_confirm_display_fragments_include_choice_mapping():
    """The modal itself must show what 1/2/3 mean, not only 'Choice [1/2/3]'."""
    from cli import HermesCLI

    self_ = SimpleNamespace(
        _slash_confirm_state={
            "title": "⚠️  /new — destroys conversation state",
            "detail": "This starts a fresh session.",
            "choices": [
                ("once", "Approve Once", "proceed once"),
                ("always", "Always Approve", "persist opt-out"),
                ("cancel", "Cancel", "abort"),
            ],
            "selected": 1,
        },
    )

    fragments = _bound(HermesCLI._get_slash_confirm_display_fragments, self_)()
    rendered = "".join(fragment for _style, fragment in fragments)

    assert "[1] Approve Once" in rendered
    assert "[2] Always Approve" in rendered
    assert "[3] Cancel" in rendered
    assert "Type 1/2/3" in rendered


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


def test_split_destructive_skip_strips_command_word():
    """Leading ``/cmd`` token is stripped; remaining args survive."""
    from cli import HermesCLI

    assert HermesCLI._split_destructive_skip("/new My title") == ("My title", False)
    assert HermesCLI._split_destructive_skip("/new --yes My title") == ("My title", True)


def test_split_destructive_skip_case_insensitive():
    """Token matching is case-insensitive but not a substring match."""
    from cli import HermesCLI

    assert HermesCLI._split_destructive_skip("/new NOW") == ("", True)
    # Substring match must NOT trigger — "Now-Title" is a literal title token.
    assert HermesCLI._split_destructive_skip("/new Now-Title") == ("Now-Title", False)


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
