"""Tests that destructive/interactive slash commands are routed inline on the
UI thread so _prompt_text_input uses run_in_terminal instead of a daemon-thread
input() that races with prompt_toolkit for stdin."""

from __future__ import annotations

from unittest.mock import MagicMock


def _make_cli():
    """Build a minimal HermesCLI-like stand-in for gate checks."""
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli._app = MagicMock()  # truthy so the inline path is active
    return cli


def test_model_commands_routed_inline():
    """/model still routes inline."""
    cli = _make_cli()
    assert cli._should_handle_interactive_slash_inline("/model") is True
    assert cli._should_handle_interactive_slash_inline("/model gpt-5") is True


def test_new_routed_inline():
    cli = _make_cli()
    assert cli._should_handle_interactive_slash_inline("/new") is True
    assert cli._should_handle_interactive_slash_inline("/new my session") is True


def test_clear_routed_inline():
    cli = _make_cli()
    assert cli._should_handle_interactive_slash_inline("/clear") is True


def test_undo_routed_inline():
    cli = _make_cli()
    assert cli._should_handle_interactive_slash_inline("/undo") is True


def test_reset_routed_inline():
    cli = _make_cli()
    assert cli._should_handle_interactive_slash_inline("/reset") is True


def test_non_interactive_commands_not_routed_inline():
    """Non-interactive slash commands (no _prompt_text_input) should NOT
    route inline — they work fine through the process_loop daemon thread."""
    cli = _make_cli()
    assert cli._should_handle_interactive_slash_inline("/help") is False
    assert cli._should_handle_interactive_slash_inline("/status") is False
    assert cli._should_handle_interactive_slash_inline("/stop") is False
    assert cli._should_handle_interactive_slash_inline("/title") is False


def test_non_slash_input_not_routed_inline():
    cli = _make_cli()
    assert cli._should_handle_interactive_slash_inline("hello") is False
    assert cli._should_handle_interactive_slash_inline("") is False


def test_images_attached_not_routed_inline():
    """When images are attached, the inline path is skipped — the command
    can't be handled reliably inline with image data pending."""
    cli = _make_cli()
    assert cli._should_handle_interactive_slash_inline("/new", has_images=True) is False


def test_reload_mcp_routed_inline():
    cli = _make_cli()
    assert cli._should_handle_interactive_slash_inline("/reload-mcp") is True


def test_old_gate_still_works():
    """_should_handle_model_command_inline still works (backward compat)."""
    cli = _make_cli()
    assert cli._should_handle_model_command_inline("/model") is True
    assert cli._should_handle_model_command_inline("/new") is False  # old gate excludes non-model
