"""Windows 10 default-font tofu fallbacks (#67151).

The update-banner glyph and the doctor's legacy-console detection are pure functions of
(platform, env) so every lane can prove the win32 behaviour without faking sys.platform.
"""

import pytest

from hermes_cli.doctor_platform import _check_windows_font_coverage, _legacy_windows_console
from hermes_cli.update_cmd import _print_update_check_result, _update_banner_glyph


class TestUpdateBannerGlyph:
    def test_plain_on_win32(self):
        # Stock conhost/Consolas has no U+2624 coverage; a cp1252 pipe cannot encode it either.
        assert _update_banner_glyph("win32") == ""
        assert "\u2624" not in _update_banner_glyph("win32")

    def test_kept_on_other_platforms(self):
        assert _update_banner_glyph("darwin") == "\u2624 "
        assert _update_banner_glyph("linux") == "\u2624 "


class TestUpdateBannerPrint:
    def test_prints_without_glyph_when_plain(self, capsys):
        _print_update_check_result(2, "origin/main", glyph="")
        out = capsys.readouterr().out
        assert "\u2624" not in out
        assert "Update available: 2 commits behind origin/main." in out

    def test_prints_unknown_count_without_glyph_when_plain(self, capsys):
        _print_update_check_result(None, "origin/main", glyph="")
        out = capsys.readouterr().out
        assert "\u2624" not in out
        assert "Update available (behind origin/main)." in out

    def test_default_keeps_the_glyph_here(self, capsys):
        # This host is not win32, so the default resolution keeps the branded banner.
        _print_update_check_result(2, "origin/main")
        assert "\u2624" in capsys.readouterr().out


class TestLegacyWindowsConsole:
    def test_detected_without_terminal_markers(self):
        assert _legacy_windows_console("win32", {}) is True

    def test_windows_terminal_markers_disarm_it(self):
        assert _legacy_windows_console("win32", {"WT_SESSION": "guid"}) is False
        assert _legacy_windows_console("win32", {"WT_PROFILE_ID": "profile"}) is False

    def test_other_platforms_never_legacy(self):
        assert _legacy_windows_console("darwin", {}) is False
        assert _legacy_windows_console("linux", {"WT_SESSION": "guid"}) is False


@pytest.mark.platforms("windows")
class TestCheckWindowsFontCoverage:
    def test_reports_a_remedy_under_a_legacy_console(self):
        # The Windows CI lane runs under a plain console (no WT_SESSION/WT_PROFILE_ID).
        finding = _check_windows_font_coverage(False)
        assert finding.manual_issues, "expected a manual remedy row under a legacy console"
        assert "Cascadia" in finding.manual_issues[0]
