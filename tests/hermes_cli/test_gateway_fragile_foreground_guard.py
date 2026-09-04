"""``_guard_fragile_foreground_gateway`` — refuse an interactive Windows
console-attached ``hermes gateway run`` so a fleet of sessions can't churn the
gateway by each starting one that dies when its terminal closes.
"""

from __future__ import annotations

import types

import pytest

import hermes_cli.gateway as gateway_cli


def _force_conditions(monkeypatch, *, windows, tty, console, detached_env):
    monkeypatch.setattr(gateway_cli, "is_windows", lambda: windows)
    monkeypatch.setattr(gateway_cli, "_windows_console_window_attached", lambda: console)
    monkeypatch.setattr(gateway_cli, "_running_under_gateway_supervisor", lambda: False)
    monkeypatch.setattr(
        gateway_cli.sys, "stdin", types.SimpleNamespace(isatty=lambda: tty)
    )
    if detached_env:
        monkeypatch.setenv("HERMES_GATEWAY_DETACHED", "1")
    else:
        monkeypatch.delenv("HERMES_GATEWAY_DETACHED", raising=False)


def test_refuses_interactive_windows_console_attached_run(monkeypatch, capsys):
    _force_conditions(monkeypatch, windows=True, tty=True, console=True, detached_env=False)

    with pytest.raises(SystemExit) as exc:
        gateway_cli._guard_fragile_foreground_gateway(replace=False, force=False)

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "hermes gateway start" in out
    assert "--force" in out


@pytest.mark.parametrize(
    "kwargs, cond",
    [
        # explicit escape hatches
        (dict(replace=True, force=False), dict(windows=True, tty=True, console=True, detached_env=False)),
        (dict(replace=False, force=True), dict(windows=True, tty=True, console=True, detached_env=False)),
        # service launchers set the marker
        (dict(replace=False, force=False), dict(windows=True, tty=True, console=True, detached_env=True)),
        # legit foreground run off Windows (WSL / Docker / Termux)
        (dict(replace=False, force=False), dict(windows=False, tty=True, console=True, detached_env=False)),
        # detached spawn: no console window / not a tty
        (dict(replace=False, force=False), dict(windows=True, tty=True, console=False, detached_env=False)),
        (dict(replace=False, force=False), dict(windows=True, tty=False, console=True, detached_env=False)),
    ],
)
def test_passes_through_for_non_fragile_cases(monkeypatch, kwargs, cond):
    _force_conditions(monkeypatch, **cond)
    # must return without raising
    assert gateway_cli._guard_fragile_foreground_gateway(**kwargs) is None


def test_supervised_run_is_exempt(monkeypatch):
    _force_conditions(monkeypatch, windows=True, tty=True, console=True, detached_env=False)
    monkeypatch.setattr(gateway_cli, "_running_under_gateway_supervisor", lambda: True)
    assert gateway_cli._guard_fragile_foreground_gateway() is None
