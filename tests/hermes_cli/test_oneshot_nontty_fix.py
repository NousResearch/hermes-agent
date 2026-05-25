"""Regression tests for oneshot non-TTY startup ordering (#30623)."""

import sys

import pytest

from hermes_cli import main as main_mod
from hermes_cli import oneshot as oneshot_mod


def _boom_prepare(_args):
    raise AssertionError("_prepare_agent_startup must not run before oneshot dispatch")


def test_main_dispatches_oneshot_without_prepare_agent_startup(monkeypatch):
    """Top-level `hermes -z` must enter run_oneshot before agent startup.

    The bug was that `_prepare_agent_startup()` ran first for `args.command is
    None`, allowing prompt_toolkit setup to touch a piped stdout before
    oneshot's quiet redirect was active.
    """
    calls = []

    def fake_run_oneshot(prompt, *, model=None, provider=None, toolsets=None):
        calls.append((prompt, model, provider, toolsets))
        return 0

    monkeypatch.setattr(sys, "argv", ["hermes", "-z", "Reply OK"])
    monkeypatch.setattr(main_mod, "_try_termux_fast_tui_launch", lambda: False)
    monkeypatch.setattr(main_mod, "_try_termux_fast_cli_launch", lambda: False)
    monkeypatch.setattr(main_mod, "_cleanup_quarantined_exes", lambda: None)
    monkeypatch.setattr(main_mod, "_prepare_agent_startup", _boom_prepare)
    monkeypatch.setattr(oneshot_mod, "run_oneshot", fake_run_oneshot)

    with pytest.raises(SystemExit) as exc:
        main_mod.main()

    assert exc.value.code == 0
    assert calls == [("Reply OK", None, None, None)]


def test_termux_fast_path_dispatches_oneshot_without_prepare_agent_startup(monkeypatch):
    """The Termux fast parser must preserve the same non-interactive invariant."""
    calls = []

    def fake_run_oneshot(prompt, *, model=None, provider=None, toolsets=None):
        calls.append((prompt, model, provider, toolsets))
        return 0

    monkeypatch.setattr(sys, "argv", ["hermes", "--oneshot", "Reply OK"])
    monkeypatch.setattr(main_mod, "_is_termux_startup_environment", lambda: True)
    monkeypatch.setattr(main_mod, "_prepare_agent_startup", _boom_prepare)
    monkeypatch.setattr(oneshot_mod, "run_oneshot", fake_run_oneshot)

    with pytest.raises(SystemExit) as exc:
        main_mod._try_termux_fast_cli_launch()

    assert exc.value.code == 0
    assert calls == [("Reply OK", None, None, None)]


def test_oneshot_run_redirects_stdout_before_running_agent(monkeypatch, tmp_path, capsys):
    """Output emitted inside the agent call tree must not leak to stdout/stderr."""

    def noisy_agent(*_args, **_kwargs):
        print("leaked stdout")
        print("leaked stderr", file=sys.stderr)
        return "final response"

    monkeypatch.setattr(oneshot_mod, "_validate_explicit_toolsets", lambda _toolsets: (None, None))
    monkeypatch.setattr(oneshot_mod, "_run_agent", noisy_agent)

    rc = oneshot_mod.run_oneshot("prompt")

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == "final response\n"
    assert captured.err == ""
