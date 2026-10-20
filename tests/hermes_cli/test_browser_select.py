"""``hermes browser select`` + preferred-browser resolution.

Resolution contracts (not snapshots): an explicit valid preference beats OS-default
detection; unknown keys and not-installed preferences fail closed; unset falls through
to detection. The select command round-trips through a temp ``HERMES_HOME`` with real
imports — the config write and the resolver read the same file the CLI uses.
"""

import argparse
import sys
import types
from types import SimpleNamespace

import pytest
import yaml

import hermes_cli.browser_connect as bc
from hermes_cli.subcommands.browser import _select_browser, build_browser_parser


@pytest.fixture
def home(tmp_path, monkeypatch):
    from hermes_cli.config import ensure_hermes_home

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ensure_hermes_home()
    return tmp_path


def _write_config(home, browser_cfg):
    (home / "config.yaml").write_text(
        yaml.safe_dump({"browser": browser_cfg}), encoding="utf-8")


class TestResolveRealProfileBrowser:
    def test_explicit_preference_beats_detection(self, home, monkeypatch):
        _write_config(home, {"preferred_browser": "chrome"})
        monkeypatch.setattr(bc, "chromium_executable", lambda key, system=None: "/bin/chrome")
        monkeypatch.setattr(bc, "detect_default_chromium", lambda system=None: "edge")
        assert bc.resolve_real_profile_browser() == ("chrome", None)

    def test_unset_preference_falls_through_to_detection(self, home, monkeypatch):
        _write_config(home, {})
        monkeypatch.setattr(bc, "detect_default_chromium", lambda system=None: "edge")
        assert bc.resolve_real_profile_browser() == ("edge", None)

    def test_unknown_preference_fails_closed(self, home):
        _write_config(home, {"preferred_browser": "firefox"})
        browser, error = bc.resolve_real_profile_browser()
        assert browser is None
        assert isinstance(error, str)
        assert "firefox" in error and "hermes browser select" in error

    def test_preference_without_binary_fails_closed(self, home, monkeypatch):
        _write_config(home, {"preferred_browser": "brave"})
        monkeypatch.setattr(bc, "chromium_executable", lambda key, system=None: None)
        browser, error = bc.resolve_real_profile_browser()
        assert browser is None
        assert isinstance(error, str) and "brave" in error


class TestInstalledChromiumBrowsers:
    def test_lists_only_present_binaries_in_candidate_order(self, monkeypatch):
        monkeypatch.setattr(
            bc, "chromium_executable",
            lambda key, system=None: f"/bin/{key}" if key in {"edge", "chrome"} else None)
        assert bc.installed_chromium_browsers() == [
            ("chrome", "/bin/chrome"), ("edge", "/bin/edge")]


class TestSelectCommand:
    def test_flag_sets_known_browser(self, home):
        args = SimpleNamespace(browser="chrome")
        assert _select_browser(args, lambda system=None: [], bc._BROWSER_BY_KEY) == 0
        raw = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
        assert raw["browser"]["preferred_browser"] == "chrome"

    def test_flag_rejects_unknown_browser(self, home):
        args = SimpleNamespace(browser="firefox")
        assert _select_browser(args, lambda system=None: [], bc._BROWSER_BY_KEY) == 1
        assert not (home / "config.yaml").exists()

    def test_no_installed_browser_errors(self, home):
        assert _select_browser(SimpleNamespace(browser=None),
                               lambda system=None: [], bc._BROWSER_BY_KEY) == 1

    def test_single_installed_browser_saves_without_prompting(self, home):
        args = SimpleNamespace(browser=None)
        installed = lambda system=None: [("edge", "C:/edge/msedge.exe")]
        assert _select_browser(args, installed, bc._BROWSER_BY_KEY) == 0
        raw = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
        assert raw["browser"]["preferred_browser"] == "edge"

    def test_multiple_browsers_without_tty_lists_and_hints(self, home, monkeypatch, capsys):
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        installed = lambda system=None: [("chrome", "/c/chrome"), ("edge", "/e/edge")]
        assert _select_browser(SimpleNamespace(browser=None), installed,
                               bc._BROWSER_BY_KEY) == 2
        err = capsys.readouterr().err
        assert "--browser <key>" in err and "chrome" in err and "edge" in err

    def test_picker_choice_is_saved(self, home, monkeypatch):
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        fake_ui = types.ModuleType("hermes_cli.curses_ui")
        setattr(fake_ui, "curses_radiolist", lambda *a, **k: 1)
        monkeypatch.setitem(sys.modules, "hermes_cli.curses_ui", fake_ui)
        installed = lambda system=None: [("chrome", "/c/chrome"), ("edge", "/e/edge")]
        assert _select_browser(SimpleNamespace(browser=None), installed,
                               bc._BROWSER_BY_KEY) == 0
        raw = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
        assert raw["browser"]["preferred_browser"] == "edge"

    def test_parser_wires_select_end_to_end(self, home):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        build_browser_parser(subparsers)
        args = parser.parse_args(["browser", "select", "--browser", "brave"])
        assert args.func(args) == 0
        raw = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
        assert raw["browser"]["preferred_browser"] == "brave"
