"""Tests for Windows cua-driver post-update refresh decisions (#132709)."""

from __future__ import annotations

from hermes_cli.update_cmd_maint import _windows_cua_refresh_action


def test_windows_cua_refresh_noop_when_pin_current():
    assert _windows_cua_refresh_action(pin_current=True, autostart_opt_in=True) == "noop"
    assert _windows_cua_refresh_action(pin_current=True, autostart_opt_in=False) == "noop"


def test_windows_cua_refresh_ensure_without_autostart():
    assert _windows_cua_refresh_action(pin_current=False, autostart_opt_in=False) == "ensure"


def test_windows_cua_refresh_defer_when_outdated_and_autostart():
    assert _windows_cua_refresh_action(pin_current=False, autostart_opt_in=True) == "defer"
