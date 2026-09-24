"""Regression tests for hermetic guards around local desktop side effects."""

from __future__ import annotations

import webbrowser


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


