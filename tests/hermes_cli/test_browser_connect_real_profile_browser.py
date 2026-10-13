"""browser.real_profile_browser: choose the Chromium browser used for real-profile browsing.

Unset follows the OS default exactly as before; a supported key overrides it; an unsupported
value fails closed (never silently drives a different browser's logins).
"""
from unittest.mock import patch

import pytest

import hermes_cli.browser_connect as bc


def _cfg(value):
    return patch.object(bc, "_browser_setting", lambda key: value if key == "real_profile_browser" else None)


@pytest.mark.parametrize("value", [None, "", "   ", 7])
def test_unset_follows_system_default(value):
    with _cfg(value), patch.object(bc, "detect_default_chromium", return_value="brave") as det:
        assert bc.resolve_real_profile_browser() == ("brave", None)
    det.assert_called_once()


def test_unset_with_no_chromium_default_returns_none():
    with _cfg(None), patch.object(bc, "detect_default_chromium", return_value=None):
        assert bc.resolve_real_profile_browser() == (None, None)


@pytest.mark.parametrize("value,key", [("chrome", "chrome"), (" Chrome ", "chrome"), ("EDGE", "edge"),
                                       ("brave-origin", "brave-origin")])
def test_configured_browser_overrides_default_without_detection(value, key):
    with _cfg(value), patch.object(bc, "detect_default_chromium", return_value="brave") as det:
        assert bc.resolve_real_profile_browser() == (key, None)
    det.assert_not_called()


@pytest.mark.parametrize("value", ["safari", "firefox", "chrome-beta"])
def test_unsupported_value_fails_closed(value):
    with _cfg(value), patch.object(bc, "detect_default_chromium", return_value="chrome") as det:
        browser, err = bc.resolve_real_profile_browser()
    assert browser is None and err is not None
    assert "browser.real_profile_browser" in err and repr(value) in err and "chrome" in err
    det.assert_not_called()

