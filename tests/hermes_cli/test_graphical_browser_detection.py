"""Tests for `_can_open_graphical_browser()` in hermes_cli.auth.

Guards the fix for the May 2026 report where `hermes auth add xai-oauth`
launched a text-mode browser (w3m) INSIDE the terminal on a headless Linux
box — `_is_remote_session()` only checked SSH/cloud-shell env vars, so a plain
local box with no GUI browser still called `webbrowser.open()`, which resolved
to a console browser and hijacked the TTY.

The helper distinguishes "a real windowed browser will pop up" from "a console
browser will hijack the terminal" so OAuth callsites can fall back to printing
the URL / manual paste instead of auto-opening.
"""

from __future__ import annotations

import webbrowser

import pytest

from hermes_cli.auth import _can_open_graphical_browser


class _FakeController:
    def __init__(self, name: str) -> None:
        self.name = name

    def open(self, *_a, **_kw):  # pragma: no cover - never invoked
        return True


@pytest.fixture(autouse=True)
def _clean_browser_env(monkeypatch):
    """Each test controls DISPLAY / WAYLAND_DISPLAY / BROWSER explicitly."""
    for var in ("DISPLAY", "WAYLAND_DISPLAY", "BROWSER"):
        monkeypatch.delenv(var, raising=False)
    yield


def _force_platform_linux(monkeypatch):
    monkeypatch.setattr("hermes_cli.auth.sys.platform", "linux")


def _force_resolved_browser(monkeypatch, name: str):
    monkeypatch.setattr(webbrowser, "get", lambda *_a, **_kw: _FakeController(name))


def test_headless_linux_no_display_refuses(monkeypatch):
    """The reported bug: headless Linux, no display server → don't auto-open."""
    _force_platform_linux(monkeypatch)
    # Even if a GUI browser somehow resolved, no display means no GUI.
    _force_resolved_browser(monkeypatch, "google-chrome")
    assert _can_open_graphical_browser() is False


def test_browser_env_pointing_at_console_browser_refuses(monkeypatch):
    """$BROWSER=w3m must refuse even with a display server present."""
    _force_platform_linux(monkeypatch)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("BROWSER", "/usr/bin/w3m")
    assert _can_open_graphical_browser() is False


@pytest.mark.parametrize("console", ["w3m", "lynx", "links", "elinks", "browsh"])
def test_resolved_console_browser_refuses(monkeypatch, console):
    """When webbrowser resolves to a console browser, refuse to auto-open."""
    _force_platform_linux(monkeypatch)
    monkeypatch.setenv("DISPLAY", ":0")
    _force_resolved_browser(monkeypatch, console)
    assert _can_open_graphical_browser() is False


def test_graphical_browser_with_display_allows(monkeypatch):
    """Real GUI browser + display server → auto-open is fine."""
    _force_platform_linux(monkeypatch)
    monkeypatch.setenv("DISPLAY", ":0")
    _force_resolved_browser(monkeypatch, "firefox")
    assert _can_open_graphical_browser() is True


def test_webbrowser_get_raises_refuses(monkeypatch):
    """No resolvable browser at all → don't auto-open."""
    _force_platform_linux(monkeypatch)
    monkeypatch.setenv("DISPLAY", ":0")

    def _boom(*_a, **_kw):
        raise webbrowser.Error("no browser")

    monkeypatch.setattr(webbrowser, "get", _boom)
    assert _can_open_graphical_browser() is False


def test_non_linux_with_gui_allows(monkeypatch):
    """macOS / Windows always have a usable default GUI browser."""
    monkeypatch.setattr("hermes_cli.auth.sys.platform", "darwin")
    _force_resolved_browser(monkeypatch, "MacOSX")
    assert _can_open_graphical_browser() is True
