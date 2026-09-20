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
            self._last_turn_result = {"final_response": "done", "completed": True}
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

    with pytest.raises(SystemExit) as exc_info:  # the one-shot path exits with the turn's outcome
        cli_mod.main(query="hello", quiet=False, toolsets="terminal")

    assert exc_info.value.code == 0
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


# ── #116904: the escape-sequence fallback must not go through os.system() ───

def _fallback_cli():
    """A stdout that is a tty but whose write() raises, forcing the clear fallback."""

    class ExplodingStdout:
        def isatty(self):
            return True

        def write(self, _data):
            raise OSError("terminal rejects the escape sequence")

        def flush(self):
            pass

    return SimpleNamespace(), ExplodingStdout()


@pytest.mark.parametrize("is_windows", [True, False])
def test_clear_fallback_spawns_no_shell(monkeypatch, is_windows):
    """#116904: fallback used os.system() — a shell spawn (console flash on Windows,
    silent no-op without `clear`). It must now be argv subprocess.run with hidden
    Windows console flags."""
    import hermes_cli.cli_session_mixin as mixin_mod

    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))

    monkeypatch.setattr(mixin_mod, "os", SimpleNamespace(
        name="nt" if is_windows else "posix"), raising=False)
    monkeypatch.setattr(mixin_mod.shutil, "which",
                        lambda exe: f"/usr/bin/{exe}" if exe == "clear" else None)
    import hermes_cli._subprocess_compat as compat_mod
    monkeypatch.setattr(compat_mod, "windows_hide_flags",
                        lambda: 0x08000000 if is_windows else 0)
    import subprocess as sp
    monkeypatch.setattr(sp, "run", fake_run)

    _, stdout = _fallback_cli()
    monkeypatch.setattr(mixin_mod.sys, "stdout", stdout)

    mixin_mod.CLISessionMixin._clear_terminal_on_exit(SimpleNamespace())

    assert len(calls) == 1, "fallback must spawn exactly one subprocess"
    argv, kwargs = calls[0]
    assert isinstance(argv, list), "argv must be a list — no shell string"
    assert kwargs.get("creationflags") == (0x08000000 if is_windows else 0), (
        "CREATE_NO_WINDOW must be applied on Windows so no console flashes"
    )
    if is_windows:
        assert argv == ["cmd", "/c", "cls"]
    else:
        assert argv == ["/usr/bin/clear"]


def test_clear_fallback_skips_spawn_when_no_clear(monkeypatch):
    """POSIX without `clear` on PATH: skip the spawn entirely instead of letting a
    shell swallow the failure (#116904's silent no-op)."""
    import hermes_cli.cli_session_mixin as mixin_mod

    calls = []
    monkeypatch.setattr(mixin_mod, "os", SimpleNamespace(name="posix"), raising=False)
    monkeypatch.setattr(mixin_mod.shutil, "which", lambda _exe: None)
    import subprocess as sp
    monkeypatch.setattr(sp, "run", lambda *a, **k: calls.append(a))

    _, stdout = _fallback_cli()
    monkeypatch.setattr(mixin_mod.sys, "stdout", stdout)

    mixin_mod.CLISessionMixin._clear_terminal_on_exit(SimpleNamespace())

    assert calls == [], "no clear binary => nothing to spawn"
