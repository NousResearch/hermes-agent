"""Tests for hermes_cli/worktree_cmd.py — _fmt_size helper."""


def test_fmt_size_none():
    from hermes_cli.worktree_cmd import _fmt_size
    assert _fmt_size(None) == "?"


def test_fmt_size_small():
    from hermes_cli.worktree_cmd import _fmt_size
    assert _fmt_size(0) == "0M"
    assert _fmt_size(5) == "5M"
    assert _fmt_size(1023) == "1023M"


def test_fmt_size_gigabytes():
    from hermes_cli.worktree_cmd import _fmt_size
    assert _fmt_size(1024) == "1.0G"
    assert _fmt_size(2048) == "2.0G"
    assert _fmt_size(2560) == "2.5G"
