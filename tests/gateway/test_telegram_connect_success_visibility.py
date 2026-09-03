"""Regression test for #90835 (logging-illusion half).

The Telegram adapter's "Connecting to Telegram (attempt N/8)…" line is
emitted at WARNING and therefore reaches the gateway's default stderr
handler (WARNING-only at default verbosity). The matching
"Connected to Telegram (… mode)" success line was INFO and went to the log
file only, so a healthy startup looked permanently stalled at "attempt 1/8"
on the terminal.

This test pins the invariant that both sides of the connect transition are
emitted at the same terminal-visible level, by inspecting the log calls in
the adapter source (the connect path needs a live bot token to execute, so
the levels are asserted at the AST level rather than by running it).
"""

import ast
from pathlib import Path

ADAPTER = (
    Path(__file__).resolve().parents[2]
    / "plugins"
    / "platforms"
    / "telegram"
    / "adapter.py"
)


def _logger_call_level(tree: ast.AST, needle: str) -> str:
    """Return the logger method name for the log call whose format string
    contains *needle*."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "logger"
            and node.args
        ):
            first = node.args[0]
            if (
                isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and needle in first.value
            ):
                return node.func.attr
    raise AssertionError(f"no logger call containing {needle!r} found")


def test_connect_success_line_matches_attempt_line_visibility():
    tree = ast.parse(ADAPTER.read_text(encoding="utf-8"))
    attempt_level = _logger_call_level(tree, "Connecting to Telegram (attempt")
    success_level = _logger_call_level(tree, "Connected to Telegram")

    # The attempt line is terminal-visible (WARNING at default verbosity).
    assert attempt_level == "warning"
    # The success line must reach the same console sink — otherwise a healthy
    # connect is indistinguishable from a hang at "attempt 1/8" (#90835).
    assert success_level == attempt_level, (
        "Telegram connect success line logs at a different level "
        f"({success_level!r}) than the attempt line ({attempt_level!r}); "
        "a healthy startup will look hung on the terminal (#90835)"
    )
