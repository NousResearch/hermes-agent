"""Tests for cli.HermesCLI._confirm_destructive_slash.

Drives the helper directly via __get__ on a SimpleNamespace stand-in so we
don't have to construct a full HermesCLI (which requires extensive setup).
"""

from __future__ import annotations

import queue
from types import SimpleNamespace
from unittest.mock import patch


def _import_cli():
    """Import the real cli module from the checkout under test."""
    import cli as cli_mod

    return cli_mod


def _bound(fn, instance):
    """Bind an unbound method to a stand-in instance."""
    return fn.__get__(instance, type(instance))


def _make_self(
    prompt_response,
    *,
    agent=None,
    background_tasks=None,
    capture=None,
    cli_mod=None,
):
    """Build a minimal stand-in 'self' for _confirm_destructive_slash."""
    cli_mod = cli_mod or _import_cli()
    HermesCLI = cli_mod.HermesCLI

    def _modal(**kwargs):
        if capture is not None:
            capture.append(kwargs)
        return prompt_response

    self_ = SimpleNamespace(
        _app=None,
        agent=agent,
        _background_tasks=background_tasks or {},
        _prompt_text_input=lambda _prompt: prompt_response,
        _prompt_text_input_modal=_modal,
    )
    self_._normalize_slash_confirm_choice = _bound(
        HermesCLI._normalize_slash_confirm_choice, self_,
    )
    return self_


def test_gate_off_returns_once_without_prompting():
    """When approvals.destructive_slash_confirm is False, return 'once'
    immediately (caller proceeds without showing a prompt)."""
    cli_mod = _import_cli()
    HermesCLI = cli_mod.HermesCLI
    self_ = _make_self(prompt_response="should not be called", cli_mod=cli_mod)

    with patch.object(
        cli_mod,
        "load_cli_config",
        return_value={"approvals": {"destructive_slash_confirm": False}},
    ):
        result = _bound(HermesCLI._confirm_destructive_slash, self_)(
            "clear", "detail",
        )

    assert result == "once"


def test_gate_on_choice_once_returns_once():
    """When the gate is on and the user picks '1', return 'once'."""
    cli_mod = _import_cli()
    HermesCLI = cli_mod.HermesCLI
    self_ = _make_self(prompt_response="1", cli_mod=cli_mod)

    with patch.object(
        cli_mod,
        "load_cli_config",
        return_value={"approvals": {"destructive_slash_confirm": True}},
    ):
        result = _bound(HermesCLI._confirm_destructive_slash, self_)(
            "clear", "detail",
        )

    assert result == "once"


def test_gate_on_choice_cancel_returns_none():
    """When the user picks '3' (cancel), return None — caller must abort."""
    cli_mod = _import_cli()
    HermesCLI = cli_mod.HermesCLI
    self_ = _make_self(prompt_response="3", cli_mod=cli_mod)

    with patch.object(
        cli_mod,
        "load_cli_config",
        return_value={"approvals": {"destructive_slash_confirm": True}},
    ):
        result = _bound(HermesCLI._confirm_destructive_slash, self_)(
            "clear", "detail",
        )

    assert result is None


def test_gate_on_no_input_returns_none():
    """No input (None / EOF / Ctrl-C) treated as cancel."""
    cli_mod = _import_cli()
    HermesCLI = cli_mod.HermesCLI
    self_ = _make_self(prompt_response=None, cli_mod=cli_mod)

    with patch.object(
        cli_mod,
        "load_cli_config",
        return_value={"approvals": {"destructive_slash_confirm": True}},
    ):
        result = _bound(HermesCLI._confirm_destructive_slash, self_)(
            "clear", "detail",
        )

    assert result is None


def test_gate_on_unknown_choice_returns_none():
    """Garbage input is treated as cancel — fail safe, don't destroy state."""
    cli_mod = _import_cli()
    HermesCLI = cli_mod.HermesCLI
    self_ = _make_self(prompt_response="maybe", cli_mod=cli_mod)

    with patch.object(
        cli_mod,
        "load_cli_config",
        return_value={"approvals": {"destructive_slash_confirm": True}},
    ):
        result = _bound(HermesCLI._confirm_destructive_slash, self_)(
            "clear", "detail",
        )

    assert result is None


def test_gate_on_choice_always_persists_and_returns_always():
    """User picks 'always' → returns 'always' AND
    save_config_value('approvals.destructive_slash_confirm', False) was called."""
    cli_mod = _import_cli()
    HermesCLI = cli_mod.HermesCLI
    self_ = _make_self(prompt_response="2", cli_mod=cli_mod)

    saves = []
    def _fake_save(key, value):
        saves.append((key, value))
        return True

    with patch.object(
        cli_mod,
        "load_cli_config",
        return_value={"approvals": {"destructive_slash_confirm": True}},
    ), patch.object(cli_mod, "save_config_value", _fake_save):
        result = _bound(HermesCLI._confirm_destructive_slash, self_)(
            "clear", "detail",
        )

    assert result == "always"
    assert ("approvals.destructive_slash_confirm", False) in saves


def test_gate_default_true_when_config_missing():
    """If load_cli_config raises or returns malformed data, treat as
    'gate on' (default safe) — must prompt."""
    cli_mod = _import_cli()
    HermesCLI = cli_mod.HermesCLI
    self_ = _make_self(prompt_response="3", cli_mod=cli_mod)  # cancel

    with patch.object(cli_mod, "load_cli_config", side_effect=Exception("boom")):
        result = _bound(HermesCLI._confirm_destructive_slash, self_)(
            "clear", "detail",
        )

    # Got prompted (returned None from cancel) — meaning the gate was
    # treated as on despite the config error.  If the gate had been off
    # this would have returned 'once' without consulting the prompt.
    assert result is None


def test_gate_off_still_prompts_when_active_work_exists():
    """Active delegated/background work must force a confirmation prompt even
    when the global destructive-slash gate is disabled."""
    cli_mod = _import_cli()
    HermesCLI = cli_mod.HermesCLI

    captures = []
    agent = SimpleNamespace(_active_children=[object(), object()])
    self_ = _make_self(
        prompt_response="1",
        agent=agent,
        background_tasks={"bg-1": object()},
        capture=captures,
        cli_mod=cli_mod,
    )

    with patch.object(
        cli_mod,
        "load_cli_config",
        return_value={"approvals": {"destructive_slash_confirm": False}},
    ):
        result = _bound(HermesCLI._confirm_destructive_slash, self_)(
            "new", "This starts a fresh session.",
        )

    assert result == "once"
    assert captures, "active work should force the prompt modal to open"
    rendered_detail = captures[0]["detail"]
    assert "Active work is still running" in rendered_detail
    assert "2 delegated subagents still running" in rendered_detail
    assert "1 background task still running" in rendered_detail


def test_slash_confirm_modal_number_selection_submits_without_raw_input():
    """Pressing 2 in the TUI modal should resolve to Always Approve directly."""
    HermesCLI = _import_cli().HermesCLI

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
    HermesCLI = _import_cli().HermesCLI

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
