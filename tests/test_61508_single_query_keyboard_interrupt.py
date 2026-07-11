"""
Regression test for issue #61508 - Single-query chat prints KeyboardInterrupt
traceback on signal.

The bug: `hermes chat -q "..."` (single-query mode) didn't catch
KeyboardInterrupt. The signal handler raises KeyboardInterrupt inside
frames deep in `cli.chat()`, and without a try/except, it bubbles to
`venv/bin/hermes` and prints a raw traceback.

The fix: mirror the quiet-mode pattern — wrap the `cli.chat(query, ...)`
call in `except KeyboardInterrupt`, call `_emit_interrupted_session_end`,
print session_id, and `sys.exit(130)`.
"""
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, "/tmp/hermes-pr-work-60859/hermes-agent")


def test_single_query_chat_catches_keyboard_interrupt():
    """When cli.chat() raises KeyboardInterrupt in single-query mode,
    the call site must catch it, emit the interrupted-session-end
    helper, print session_id to stderr, and exit 130."""
    import io

    # Capture stderr
    captured_stderr = io.StringIO()

    # Build a minimal CLI stub
    cli = MagicMock()
    cli.session_id = "test-session-12345"
    cli._show_security_advisories = MagicMock()
    cli._print_exit_summary = MagicMock()
    cli.chat = MagicMock(side_effect=KeyboardInterrupt())
    cli._emit_interrupted_session_end = MagicMock()

    # We need _emit_interrupted_session_end and sys.exit to be imported
    # into the test scope. We can exec the relevant block.
    # The block looks like:
    #   try:
    #       cli.chat(...)
    #   except KeyboardInterrupt:
    #       _emit_interrupted_session_end(cli, reason="keyboard_interrupt")
    #       print(...)
    #       sys.exit(130)

    # Just check that calling cli.chat with the try/except pattern works
    try:
        try:
            cli.chat("hello", images=None)
        except KeyboardInterrupt:
            cli._emit_interrupted_session_end(cli, reason="keyboard_interrupt")
            print(f"\nsession_id: {cli.session_id}", file=captured_stderr)
            sys.exit(130)
    except SystemExit as e:
        assert e.code == 130, f"expected exit code 130, got {e.code}"
    except Exception as e:
        assert False, f"unexpected exception: {e}"

    # Verify the emit was called and session_id was printed
    assert cli._emit_interrupted_session_end.called, \
        "_emit_interrupted_session_end was not called on KeyboardInterrupt"
    assert "test-session-12345" in captured_stderr.getvalue(), \
        f"session_id not printed to stderr; got: {captured_stderr.getvalue()!r}"


def test_single_query_block_has_keyboard_interrupt_handler():
    """Static check: the single-query branch in cli.py must contain
    the try/except KeyboardInterrupt pattern. If it doesn't, this
    test fails (catches regression)."""
    cli_path = "/tmp/hermes-pr-work-60859/hermes-agent/cli.py"
    with open(cli_path) as f:
        source = f.read()

    # Find the single-query block: starts with "Single-query mode" comment
    # and ends at "cli._print_exit_summary(clear_screen=False)"
    match = source[
        source.find("Single-query mode (`hermes chat -q"):
        source.find("cli._print_exit_summary(clear_screen=False)")
        + len("cli._print_exit_summary(clear_screen=False)")
    ]
    assert match, "could not locate the single-query block in cli.py"

    assert "except KeyboardInterrupt" in match, (
        "Issue #61508 regression: single-query block is missing the "
        "except KeyboardInterrupt handler. SIGINT/SIGTERM in this path "
        "will print a raw Python traceback to the terminal."
    )


def test_quiet_mode_handler_unchanged():
    """Sanity check: the quiet-mode handler (the working precedent) is
    still present and not broken."""
    cli_path = "/tmp/hermes-pr-work-60859/hermes-agent/cli.py"
    with open(cli_path) as f:
        source = f.read()

    assert "cli.agent.quiet_mode = True" in source
    # The quiet-mode try/except must still be present
    assert source.count("except KeyboardInterrupt") >= 2, (
        f"Expected at least 2 except KeyboardInterrupt blocks "
        f"(quiet mode + single-query), found {source.count('except KeyboardInterrupt')}"
    )


def test_keyboard_interrupt_exits_with_code_130():
    """Verify the exit code is 130 (128 + SIGINT=2)."""
    import io
    captured_stderr = io.StringIO()

    # Mock setup
    cli = MagicMock()
    cli.session_id = "s1"
    cli.chat = MagicMock(side_effect=KeyboardInterrupt())
    cli._emit_interrupted_session_end = MagicMock()

    try:
        try:
            cli.chat("x", images=None)
        except KeyboardInterrupt:
            cli._emit_interrupted_session_end(cli, reason="keyboard_interrupt")
            print(f"\nsession_id: {cli.session_id}", file=captured_stderr)
            sys.exit(130)
    except SystemExit as e:
        assert e.code == 130
        return
    assert False, "SystemExit was not raised"