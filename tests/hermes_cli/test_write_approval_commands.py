"""Tests for hermes_cli/write_approval_commands.py — approval command helpers."""


def test_fmt_state_returns_string():
    from hermes_cli.write_approval_commands import _fmt_state
    assert isinstance(_fmt_state("terminal"), str)


def test_fmt_pending_list_returns_string():
    from hermes_cli.write_approval_commands import _fmt_pending_list
    assert isinstance(_fmt_pending_list("terminal"), str)
