"""Tests for hermes_cli/suggestions_cmd.py — suggestions command helpers."""


def test_fmt_pending_returns_string():
    from hermes_cli.suggestions_cmd import _fmt_pending
    assert isinstance(_fmt_pending([]), str)
    assert isinstance(_fmt_pending(["task 1"]), str)


def test_fmt_pending_empty():
    from hermes_cli.suggestions_cmd import _fmt_pending
    result = _fmt_pending([])
    assert isinstance(result, str)
