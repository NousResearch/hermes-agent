"""browser.real_profile_browser: choose the Chromium browser used for real-profile browsing.

Unset follows the OS default exactly as before; a supported key overrides it; an unsupported
value fails closed (never silently drives a different browser's logins).
"""
from unittest.mock import patch

import pytest

import hermes_cli.browser_connect as bc


def _cfg(value):
    return patch.object(bc, "_browser_setting", lambda key: value if key == "real_profile_browser" else None)


@pytest.mark.parametrize("value", [None, "", "   "])
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


@pytest.mark.parametrize("value", ["safari", "firefox", "chrome-beta", 7, True, ["chrome"]])
def test_unsupported_value_fails_closed(value):
    with _cfg(value), patch.object(bc, "detect_default_chromium", return_value="chrome") as det:
        browser, err = bc.resolve_real_profile_browser()
    assert browser is None and err is not None
    assert "browser.real_profile_browser" in err and repr(value) in err and "brave-origin" in err
    det.assert_not_called()



class TestToolsLayer:
    """The launcher (tools.browser_tool_real_profile) must honour the setting end to end."""

    def _reset(self):
        import tools.browser_tool as bt

        bt._real_profile_cdp_cache.clear()

    def test_bad_value_fails_closed_before_any_snapshot(self):
        from tools import browser_tool_cloud as cloud
        from tools import browser_tool_real_profile as rp

        self._reset()
        with patch.object(cloud, "_use_real_profile", return_value=True), _cfg("safari"), \
             patch.object(bc, "detect_default_chromium", return_value="brave"), \
             patch.object(bc, "snapshot_real_profile") as snap:
            cdp, err = rp._real_profile_cdp()
        assert cdp is None and "browser.real_profile_browser" in err
        snap.assert_not_called()

    def test_configured_browser_is_the_one_snapshotted(self):
        from tools import browser_tool_cloud as cloud
        from tools import browser_tool_real_profile as rp

        self._reset()
        with patch.object(cloud, "_use_real_profile", return_value=True), _cfg("chrome"), \
             patch.object(bc, "detect_default_chromium", return_value="brave"), \
             patch.object(bc, "snapshot_real_profile", return_value=(None, "stop here")) as snap:
            rp._real_profile_cdp()
        snap.assert_called_once_with("chrome")


class TestCloseProfileCli:
    def _run(self, argv):
        import argparse

        from hermes_cli.subcommands.browser import build_browser_parser

        parser = argparse.ArgumentParser()
        build_browser_parser(parser.add_subparsers(dest="cmd"))
        args = parser.parse_args(argv)
        return args.func(args)

    def test_uses_configured_browser(self):
        with _cfg("chrome"), patch.object(bc, "detect_default_chromium", return_value="brave"), \
             patch.object(bc, "real_profile_data_dir", return_value="/x") as data_dir, \
             patch.object(bc, "close_browser_holding_profile", return_value=(True, "closed")):
            assert self._run(["browser", "close-profile"]) == 0
        data_dir.assert_called_once_with("chrome")

    def test_flag_still_wins_over_config(self):
        with _cfg("chrome"), patch.object(bc, "real_profile_data_dir", return_value="/x") as data_dir, \
             patch.object(bc, "close_browser_holding_profile", return_value=(True, "closed")):
            assert self._run(["browser", "close-profile", "--browser", "edge"]) == 0
        data_dir.assert_called_once_with("edge")

    def test_bad_value_exits_nonzero_without_closing_anything(self, capsys):
        with _cfg("safari"), patch.object(bc, "close_browser_holding_profile") as close:
            assert self._run(["browser", "close-profile"]) == 1
        close.assert_not_called()
        assert "browser.real_profile_browser" in capsys.readouterr().err
