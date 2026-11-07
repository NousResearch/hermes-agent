"""ANSI colors require a Windows console that can actually process them."""

import ctypes
import sys
from types import SimpleNamespace

import pytest

from hermes_cli import colors


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("initial_mode,console_available,enable_succeeds", [
    (1, True, True), (1, True, False), (1, False, False), (5, True, False),
])
def test_windows_color_follows_console_support(monkeypatch, initial_mode, console_available, enable_succeeds):
    import msvcrt

    calls = []

    def get_mode(handle, mode):
        mode._obj.value = initial_mode
        return console_available

    def set_mode(handle, mode):
        calls.append((handle, mode))
        return enable_succeeds

    kernel = SimpleNamespace(GetConsoleMode=get_mode, SetConsoleMode=set_mode)
    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **k: kernel, raising=False)
    monkeypatch.setattr(msvcrt, "get_osfhandle", lambda fd: 123)
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(isatty=lambda: True, fileno=lambda: 1))
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)

    text = colors.color("(●)", colors.Colors.GREEN)
    assert ("\x1b[" in text) == (console_available and (bool(initial_mode & 4) or enable_succeeds))
    assert calls == ([(123, initial_mode | 4)] if console_available and not initial_mode & 4 else [])


@pytest.mark.parametrize("tty,env", [(False, {}), (True, {"NO_COLOR": ""}), (True, {"TERM": "dumb"})])
def test_disabled_colors_do_not_touch_console(monkeypatch, tty, env):
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(isatty=lambda: tty))
    for name in ("NO_COLOR", "TERM"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    def unexpected_console_access():
        pytest.fail("Disabled colors must not access the Windows console")

    monkeypatch.setattr(colors, "enable_windows_ansi", unexpected_console_access)
    assert colors.color("selected", colors.Colors.GREEN) == "selected"
