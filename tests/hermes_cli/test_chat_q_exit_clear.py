"""Regression tests for #53009: chat -q final response erased by exit-summary clear."""

import os
from types import SimpleNamespace

import pytest

import cli as cli_mod


# ── A3.1 Test-First: verify _clear_terminal_on_exit gating ──────────────────



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


@pytest.mark.skipif(os.name == "nt", reason="POSIX `clear` path; the nt path has its own test")
def test_clear_fallback_spawns_no_shell(monkeypatch):
    """#116904: fallback used os.system() — a shell spawn (console flash on Windows,
    silent no-op without `clear`). It must now be an argv subprocess.run of the
    resolved `clear` binary, with the real (0 off-Windows) hide flags."""
    import subprocess as sp

    import hermes_cli.cli_session_mixin as mixin_mod

    calls = []
    monkeypatch.setattr(mixin_mod.shutil, "which",
                        lambda exe: f"/usr/bin/{exe}" if exe == "clear" else None)
    monkeypatch.setattr(sp, "run", lambda argv, **kwargs: calls.append((argv, kwargs)))

    _, stdout = _fallback_cli()
    monkeypatch.setattr(mixin_mod.sys, "stdout", stdout)

    mixin_mod.CLISessionMixin._clear_terminal_on_exit(SimpleNamespace())

    assert calls == [(["/usr/bin/clear"], {"stdin": sp.DEVNULL, "creationflags": 0, "check": False})]


@pytest.mark.windows_only
def test_clear_fallback_windows_runs_cls_with_hidden_console(monkeypatch):
    """Native Windows: `cls` is a cmd builtin, so the argv is cmd /c cls, run with the
    real windows_hide_flags() (CREATE_NO_WINDOW) so no console flashes (#116904)."""
    import subprocess as sp

    import hermes_cli.cli_session_mixin as mixin_mod
    from hermes_cli._subprocess_compat import windows_hide_flags

    calls = []
    monkeypatch.setattr(sp, "run", lambda argv, **kwargs: calls.append((argv, kwargs)))

    _, stdout = _fallback_cli()
    monkeypatch.setattr(mixin_mod.sys, "stdout", stdout)

    mixin_mod.CLISessionMixin._clear_terminal_on_exit(SimpleNamespace())

    assert windows_hide_flags() == sp.CREATE_NO_WINDOW
    assert calls == [(["cmd", "/c", "cls"], {"stdin": sp.DEVNULL, "creationflags": sp.CREATE_NO_WINDOW, "check": False})]


@pytest.mark.skipif(os.name == "nt", reason="POSIX `clear` lookup")
def test_clear_fallback_skips_spawn_when_no_clear(monkeypatch):
    """POSIX without `clear` on PATH: skip the spawn entirely instead of letting a
    shell swallow the failure (#116904's silent no-op)."""
    import hermes_cli.cli_session_mixin as mixin_mod

    calls = []
    monkeypatch.setattr(mixin_mod.shutil, "which", lambda _exe: None)
    import subprocess as sp
    monkeypatch.setattr(sp, "run", lambda *a, **k: calls.append(a))

    _, stdout = _fallback_cli()
    monkeypatch.setattr(mixin_mod.sys, "stdout", stdout)

    mixin_mod.CLISessionMixin._clear_terminal_on_exit(SimpleNamespace())

    assert calls == [], "no clear binary => nothing to spawn"
