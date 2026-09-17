"""Regression tests for #53009: chat -q final response erased by exit-summary clear."""

from types import SimpleNamespace

import pytest

import cli as cli_mod


# ── A3.1 Test-First: verify _clear_terminal_on_exit gating ──────────────────

def test_print_exit_summary_clears_screen_by_default(monkeypatch):
    """Default behavior: _print_exit_summary() calls _clear_terminal_on_exit()."""
    calls = []

    class FakeCLI:
        conversation_history = []
        session_start = None

        def _clear_terminal_on_exit(self):
            calls.append("clear")

    monkeypatch.setattr(cli_mod, "datetime", SimpleNamespace(
        now=lambda: SimpleNamespace(
            __sub__=lambda self, other: SimpleNamespace(
                total_seconds=lambda: 0
            )
        )
    ))

    fake = FakeCLI()
    cli_mod.HermesCLI._print_exit_summary(fake)  # default clear_screen=True

    assert "clear" in calls, "_clear_terminal_on_exit should be called by default"


def test_print_exit_summary_skips_clear_when_clear_screen_false(monkeypatch):
    """With clear_screen=False, _print_exit_summary() does NOT clear."""
    calls = []

    class FakeCLI:
        conversation_history = []
        session_start = None

        def _clear_terminal_on_exit(self):
            calls.append("clear")

    monkeypatch.setattr(cli_mod, "datetime", SimpleNamespace(
        now=lambda: SimpleNamespace(
            __sub__=lambda self, other: SimpleNamespace(
                total_seconds=lambda: 0
            )
        )
    ))

    fake = FakeCLI()
    cli_mod.HermesCLI._print_exit_summary(fake, clear_screen=False)

    assert "clear" not in calls, (
        "_clear_terminal_on_exit should NOT be called when clear_screen=False"
    )


# ── Production-path test: single-query -q path skips the clear ──────────────


def test_print_exit_summary_still_clears_in_interactive_path(monkeypatch):
    """Interactive mode should still clear the screen (preserving #38928)."""
    from datetime import datetime as real_datetime

    calls = []

    class FakeCLI:
        conversation_history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        session_start = real_datetime(2026, 1, 1, 12, 0, 0)
        session_id = "test-session"
        _session_db = None
        agent = None

        def _clear_terminal_on_exit(self):
            calls.append("clear")

    monkeypatch.setattr(cli_mod, "datetime", SimpleNamespace(
        now=lambda: real_datetime(2026, 1, 1, 12, 1, 0)  # 1 min elapsed
    ))

    fake = FakeCLI()
    cli_mod.HermesCLI._print_exit_summary(fake)  # default clear_screen=True

    assert "clear" in calls, (
        "Interactive mode should still clear the screen (regression test for #38928)"
    )
