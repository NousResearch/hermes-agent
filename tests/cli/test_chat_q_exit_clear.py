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

def test_single_query_main_skips_clear_on_exit_summary(monkeypatch):
    """The single-query (-q) path calls _print_exit_summary without clearing."""
    calls = []
    clear_calls = []

    class FakeCLI:
        def __init__(self, **_kwargs):
            self.console = SimpleNamespace(print=lambda *_a, **_kw: calls.append("query-label"))
            self.session_id = "sq-test"
            self.agent = SimpleNamespace(
                session_id="sq-test",
                platform="cli",
            )

        def _claim_active_session(self, surface, *, stderr=False):
            calls.append(("claim", surface, stderr))
            return True

        def _show_security_advisories(self):
            calls.append("advisories")

        def chat(self, query, images=None):
            calls.append(("chat", query, images))
            return "done"

        def _print_exit_summary(self, clear_screen=True):
            calls.append(("summary", clear_screen))
            if clear_screen:
                clear_calls.append("CLEARED")  # should NOT happen

    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        cli_mod,
        "_finalize_single_query",
        lambda fake_cli: calls.append(("finalize", fake_cli.session_id)),
    )

    cli_mod.main(query="hello", quiet=False, toolsets="terminal")

    assert calls == [
        ("claim", "cli", False),
        "query-label",
        "advisories",
        ("chat", "hello", None),
        ("summary", False),  # <-- clear_screen=False for single-query
        ("finalize", "sq-test"),
    ]
    assert len(clear_calls) == 0, (
        "_clear_terminal_on_exit must NOT be called in single-query mode"
    )


def test_single_query_main_handles_keyboard_interrupt_without_traceback(monkeypatch):
    """The human-facing -q path exits 130 instead of leaking KeyboardInterrupt."""
    calls = []

    class FakeCLI:
        def __init__(self, **_kwargs):
            self.console = SimpleNamespace(print=lambda *_a, **_kw: calls.append("query-label"))
            self.session_id = "sq-interrupt"
            self.agent = SimpleNamespace(
                session_id="sq-interrupt",
                platform="cli",
            )

        def _claim_active_session(self, surface, *, stderr=False):
            calls.append(("claim", surface, stderr))
            return True

        def _show_security_advisories(self):
            calls.append("advisories")

        def chat(self, query, images=None):
            calls.append(("chat", query, images))
            raise KeyboardInterrupt

        def _print_exit_summary(self, clear_screen=True):
            calls.append(("summary", clear_screen))

    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        cli_mod,
        "_emit_interrupted_session_end",
        lambda fake_cli, *, reason: calls.append(("interrupted", fake_cli.session_id, reason)),
    )
    monkeypatch.setattr(
        cli_mod,
        "_finalize_single_query",
        lambda fake_cli: calls.append(("finalize", fake_cli.session_id)),
    )

    with pytest.raises(SystemExit) as exc:
        cli_mod.main(query="hello", quiet=False, toolsets="terminal")

    assert exc.value.code == 130
    assert calls == [
        ("claim", "cli", False),
        "query-label",
        "advisories",
        ("chat", "hello", None),
        ("interrupted", "sq-interrupt", "keyboard_interrupt"),
        ("summary", False),
        ("finalize", "sq-interrupt"),
    ]


# ── Verify interactive mode still clears ────────────────────────────────────

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
