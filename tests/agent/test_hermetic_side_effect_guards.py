"""Regression tests for hermetic guards around local desktop side effects."""

from __future__ import annotations

import webbrowser

import pytest

def test_webbrowser_open_calls_are_neutralized(monkeypatch):
    """OAuth/browser tests should never reach the real browser registry."""

    def _real_browser_lookup_reached(*_args, **_kwargs):
        raise AssertionError("test reached the real webbrowser registry")

    monkeypatch.setattr(webbrowser, "get", _real_browser_lookup_reached)

    url = "https://provider.example.invalid/oauth/authorize"

    assert webbrowser.open(url) is True
    assert webbrowser.open_new(url) is True
    assert webbrowser.open_new_tab(url) is True

def test_webbrowser_get_controller_is_neutralized(_neutralize_webbrowser):
    """Direct controller access should still stay inside the test recorder."""
    url = "https://provider.example.invalid/oauth/authorize"

    controller = webbrowser.get("hermes-test-browser")

    assert controller.open(url) is True
    assert controller.open_new(url) is True
    assert controller.open_new_tab(url) is True
    assert _neutralize_webbrowser == [url, url, url]

@pytest.mark.platforms("windows")
def test_user_path_registration_never_writes_the_real_registry(tmp_path, _neutralize_windows_registry_writes):
    """Launch/update tests reach the User PATH publication; it must stay recorded."""
    import winreg

    from hermes_cli import _launchers

    entry = tmp_path / "home" / "bin"

    assert _launchers._register_windows_user_path(entry) == "added"
    [(name, (value_name, _reserved, _kind, merged))] = _neutralize_windows_registry_writes
    assert (name, value_name) == ("SetValueEx", "Path")
    assert merged.split(";")[0] == str(entry)
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
        real, _ = winreg.QueryValueEx(key, "Path")
    assert str(entry) not in real
