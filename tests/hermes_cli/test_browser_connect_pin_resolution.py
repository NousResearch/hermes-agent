"""resolve_real_profile_browser: with a pin, the pin resolves the browser, not just the profile.

``browser.real_profile_pin`` is an explicit identity choice that already fails closed on a
missing dir — demanding the OS default ALSO be Chromium on top of it (#131589) forces a
system-wide change for a per-app need. These tests pin down the resolution order: the OS
default wins when it can serve the pin (today's behavior), otherwise exactly one installed
stable Chromium with the pinned dir wins, and zero/several matches fail closed.
"""
from unittest.mock import patch

import pytest

import hermes_cli.browser_connect as bc
from tools import browser_tool_cloud as bt_cloud
from tools import browser_tool_lightpanda_fallback as bt_lp
from tools import browser_tool_real_profile as bt_real_profile


def _user_data_dirs(tmp_path, *, chrome_pins=("Default",), brave_pins=()):
    """Real user-data dirs for chrome/brave containing the given profile dirs."""
    dirs = {}
    for key, pins in (("chrome", chrome_pins), ("brave", brave_pins)):
        root = tmp_path / f"{key}-user-data"
        root.mkdir(parents=True, exist_ok=True)
        for pin in pins:
            (root / pin).mkdir(parents=True, exist_ok=True)
        dirs[key] = str(root)
    return dirs


def _resolve(tmp_path, dirs, *, default, pin, installed=("chrome",)):
    """Run resolve_real_profile_browser with the OS default, pin and installed set mocked."""
    with patch("hermes_cli.browser_connect.detect_default_chromium", return_value=default), \
         patch("hermes_cli.browser_connect._real_profile_pin", return_value=pin), \
         patch("hermes_cli.browser_connect.real_profile_data_dir",
               side_effect=lambda key, system=None: dirs.get(key)), \
         patch("hermes_cli.browser_connect.chromium_executable",
               side_effect=lambda key, system=None: f"/usr/bin/{key}" if key in installed else None):
        return bc.resolve_real_profile_browser()


class TestResolveRealProfileBrowser:

    def test_no_pin_returns_os_default_verbatim(self, tmp_path):
        dirs = _user_data_dirs(tmp_path)
        for default in (None, bc.UNSUPPORTED_CHANNEL, "chrome"):
            browser, err = _resolve(tmp_path, dirs, default=default, pin=None)
            assert (browser, err) == (default, None)

    def test_pin_under_default_browser_wins(self, tmp_path):
        # Both browsers have Default; the OS default must stay authoritative (no regression
        # for the common default=chrome + pin=Default setup).
        dirs = _user_data_dirs(tmp_path, brave_pins=("Default",))
        browser, err = _resolve(tmp_path, dirs, default="chrome", pin="Default",
                                installed=("chrome", "brave"))
        assert (browser, err) == ("chrome", None)

    def test_pin_resolves_installed_chromium_when_default_not_chromium(self, tmp_path):
        # #131589: Firefox default must not fail closed when the pin names an identity in an
        # installed stable Chrome.
        dirs = _user_data_dirs(tmp_path)
        browser, err = _resolve(tmp_path, dirs, default=None, pin="Default")
        assert (browser, err) == ("chrome", None)

    def test_pin_missing_from_default_dir_enumerates_other_browsers(self, tmp_path):
        dirs = _user_data_dirs(tmp_path, brave_pins=("Profile 2",))
        browser, err = _resolve(tmp_path, dirs, default="chrome", pin="Profile 2",
                                installed=("chrome", "brave"))
        assert (browser, err) == ("brave", None)

    def test_channel_default_with_pin_enumerates_stable_instead(self, tmp_path):
        dirs = _user_data_dirs(tmp_path)
        browser, err = _resolve(tmp_path, dirs, default=bc.UNSUPPORTED_CHANNEL, pin="Default")
        assert (browser, err) == ("chrome", None)

    def test_uninstalled_browser_never_matches(self, tmp_path):
        # A user-data dir without the browser binary is not an installed browser; only the
        # installed one may win.
        dirs = _user_data_dirs(tmp_path, brave_pins=("Profile 2",))
        browser, err = _resolve(tmp_path, dirs, default=None, pin="Profile 2",
                                installed=("brave",))
        assert (browser, err) == ("brave", None)

    def test_no_installed_match_fails_closed_naming_the_pin(self, tmp_path):
        dirs = _user_data_dirs(tmp_path)
        browser, err = _resolve(tmp_path, dirs, default=None, pin="Profile 9")
        assert browser is None
        assert err and "Profile 9" in err and "no installed Chromium browser" in err

    def test_several_matches_fail_closed_listing_candidates(self, tmp_path):
        dirs = _user_data_dirs(tmp_path, brave_pins=("Default",))
        browser, err = _resolve(tmp_path, dirs, default=None, pin="Default",
                                installed=("chrome", "brave"))
        assert browser is None
        assert err and "several installed Chromium browsers" in err
        assert "chrome (" in err and "brave (" in err  # candidates are listed, never guessed


class TestRealProfileCdpWiring:
    """_real_profile_cdp must hand the pin-resolved browser to the snapshot, and surface the
    fail-closed pin errors with the standard real-profile prefix."""

    def setup_method(self):
        import tools.browser_tool as bt
        bt._real_profile_cdp_cache.clear()

    def teardown_method(self):
        import tools.browser_tool as bt
        bt._real_profile_cdp_cache.clear()

    def test_pin_resolved_browser_reaches_the_snapshot(self, tmp_path):
        dirs = _user_data_dirs(tmp_path)
        with patch.object(bt_cloud, "_use_real_profile", return_value=True), \
             patch.object(bt_lp, "_using_lightpanda_engine", return_value=False), \
             patch("hermes_cli.browser_connect.detect_default_chromium", return_value=None), \
             patch("hermes_cli.browser_connect._real_profile_pin", return_value="Default"), \
             patch("hermes_cli.browser_connect.real_profile_data_dir",
                   side_effect=lambda key, system=None: dirs.get(key)), \
             patch("hermes_cli.browser_connect.chromium_executable",
                   side_effect=lambda key, system=None: "/usr/bin/chrome" if key == "chrome" else None), \
             patch("hermes_cli.browser_connect.snapshot_real_profile",
                   return_value=(None, "boom")) as snapshot:
            cdp, err = bt_real_profile._real_profile_cdp()
        assert cdp is None
        assert snapshot.call_args[0][0] == "chrome"  # browser resolved by the pin, not the OS default
        assert err and "boom" in err  # resolution passed; the flow continued to the snapshot

    def test_pin_zero_matches_fails_closed_with_prefix(self, tmp_path):
        dirs = _user_data_dirs(tmp_path)
        with patch.object(bt_cloud, "_use_real_profile", return_value=True), \
             patch.object(bt_lp, "_using_lightpanda_engine", return_value=False), \
             patch("hermes_cli.browser_connect.detect_default_chromium", return_value=None), \
             patch("hermes_cli.browser_connect._real_profile_pin", return_value="Profile 7"), \
             patch("hermes_cli.browser_connect.real_profile_data_dir",
                   side_effect=lambda key, system=None: dirs.get(key)), \
             patch("hermes_cli.browser_connect.chromium_executable", return_value=None):
            cdp, err = bt_real_profile._real_profile_cdp()
        assert cdp is None
        assert err is not None and err.startswith("browser.use_real_profile is on, but ")
        assert "Profile 7" in err
