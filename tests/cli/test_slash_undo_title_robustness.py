"""Regression tests for slash-command dispatch robustness (round-3 QA).

Covers:
- SC-01: `/undo <non-numeric>` must NOT quit the CLI (process_command returns
  True to keep the REPL alive), and must not raise.
- SC-06: `/undo` on an empty session reports "nothing to undo" without popping
  a destructive-confirmation dialog.
- SC-05: `/title <too-long>` prints exactly one error, not the contradictory
  "too long" + "empty after cleanup" pair.
"""

from unittest.mock import patch

from tests.cli.test_cli_init import _make_cli


def test_undo_non_numeric_count_keeps_repl_alive():
    """`/undo abc` returned a bare None → REPL treated it as exit (SC-01)."""
    cli = _make_cli()
    cli.conversation_history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    # Must not raise, and must return True (truthy → REPL continues).
    result = cli.process_command("/undo abc")
    assert result is True


def test_undo_empty_session_no_confirmation_dialog():
    """`/undo` with no history must short-circuit before the destructive prompt (SC-06)."""
    cli = _make_cli()
    cli.conversation_history = []

    called = {"confirm": False}

    def _spy_confirm(*a, **k):
        called["confirm"] = True
        return "approved"

    with patch.object(cli, "_confirm_destructive_slash", _spy_confirm):
        result = cli.process_command("/undo")

    assert result is True
    assert called["confirm"] is False  # no confirmation for a guaranteed no-op


def test_title_too_long_prints_single_error(capsys):
    """`/title <500 chars>` must print one error, not two contradictory ones (SC-05)."""
    cli = _make_cli()
    result = cli.process_command("/title " + "A" * 500)
    out = capsys.readouterr().out
    assert result is True
    # The length error should fire; the contradictory "empty after cleanup"
    # message must NOT also appear.
    assert "empty after cleanup" not in out.lower()
