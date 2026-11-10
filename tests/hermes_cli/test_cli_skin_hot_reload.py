"""Tests for TUI hot-reload when display.skin or the skin file changes."""
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

SKIN = "sync-theme"
OTHER = "other-theme"

SKIN_YAML = """name: {name}
description: test skin
colors:
  background: "#1a1b26"
  banner_text: "#c0caf5"
"""


def _write_skin(home, name, bg="#1a1b26"):
    skins = home / "skins"
    skins.mkdir(parents=True, exist_ok=True)
    (skins / f"{name}.yaml").write_text(SKIN_YAML.format(name=name).replace("#1a1b26", bg))


def _write_config(cfg_file, skin):
    import hermes_yaml as yaml
    cfg_file.write_text(yaml.safe_dump({"display": {"skin": skin}}))


def _make_cli(tmp_path, skin=SKIN):
    """Bare HermesCLI with skin-watcher state baselined to the current files."""
    import cli as cli_mod
    from hermes_cli.skin_engine import user_skin_file_signature
    obj = object.__new__(cli_mod.HermesCLI)
    obj.config = {"display": {"skin": skin}}
    obj._agent_running = False
    obj._last_skin_check = 0.0

    cfg_file = tmp_path / "config.yaml"
    _write_config(cfg_file, skin)
    with patch("hermes_cli.skin_engine.get_hermes_home", return_value=tmp_path):
        obj._skin_sig = (skin, user_skin_file_signature(skin))
    return obj, cfg_file


def _run_check(obj, cfg_file, home):
    """Run one watcher tick; return (activate_mock, accent_mock, repaint_mock)."""
    with patch("hermes_cli.config.get_config_path", return_value=cfg_file), \
         patch("hermes_cli.skin_engine.get_hermes_home", return_value=home), \
         patch("hermes_cli.skin_engine.set_active_skin") as activate, \
         patch("cli._ACCENT") as accent, \
         patch.object(obj, "_apply_tui_skin_style") as repaint:
        obj._last_skin_check = 0.0  # force past the fallback throttle
        obj._check_skin_reload()
    return activate, accent, repaint


def _alive_watcher(obj):
    """Pretend a live fsnotify thread is armed (fallback clock ignored)."""
    obj._skin_dirty = threading.Event()
    watcher = MagicMock()
    watcher.is_alive.return_value = True
    obj._skin_watcher_thread = watcher


class TestSkinHotReload:
    def test_skin_name_change_reloads(self, tmp_path, capsys):
        """Switching display.skin activates the new skin and repaints the TUI."""
        _write_skin(tmp_path, SKIN)
        _write_skin(tmp_path, OTHER)
        obj, cfg_file = _make_cli(tmp_path, skin=SKIN)
        _write_config(cfg_file, OTHER)

        activate, accent, repaint = _run_check(obj, cfg_file, tmp_path)

        activate.assert_called_once_with(OTHER)
        accent.reset.assert_called_once_with()
        repaint.assert_called_once_with()
        assert OTHER in capsys.readouterr().out

    def test_same_name_file_rewrite_reloads(self, tmp_path, capsys):
        """Rewriting the same-named skin file (theme-sync regeneration) reloads."""
        _write_skin(tmp_path, SKIN, bg="#1a1b26")
        obj, cfg_file = _make_cli(tmp_path, skin=SKIN)
        _write_skin(tmp_path, SKIN, bg="#2a2a36")  # same name, new palette

        activate, _, repaint = _run_check(obj, cfg_file, tmp_path)

        activate.assert_called_once_with(SKIN)
        repaint.assert_called_once_with()
        assert SKIN in capsys.readouterr().out

    def test_no_change_no_reload(self, tmp_path, capsys):
        """An idle tick with nothing changed stays silent."""
        _write_skin(tmp_path, SKIN)
        obj, cfg_file = _make_cli(tmp_path, skin=SKIN)

        activate, _, repaint = _run_check(obj, cfg_file, tmp_path)

        activate.assert_not_called()
        repaint.assert_not_called()
        assert capsys.readouterr().out == ""

    def test_unknown_skin_name_skipped_once(self, tmp_path, capsys):
        """A config naming a missing skin warns once and keeps the current skin."""
        _write_skin(tmp_path, SKIN)
        obj, cfg_file = _make_cli(tmp_path, skin=SKIN)
        _write_config(cfg_file, "no-such-skin")

        activate, _, repaint = _run_check(obj, cfg_file, tmp_path)

        activate.assert_not_called()
        repaint.assert_not_called()
        assert "no-such-skin" in capsys.readouterr().out
        # Second tick with the same unknown name stays silent (sig updated).
        activate2, _, _ = _run_check(obj, cfg_file, tmp_path)
        activate2.assert_not_called()
        assert capsys.readouterr().out == ""

    def test_throttle_uses_independent_clock(self, tmp_path):
        """A fresh fallback clock suppresses the check on its own clock."""
        _write_skin(tmp_path, SKIN)
        _write_skin(tmp_path, OTHER)
        obj, cfg_file = _make_cli(tmp_path, skin=SKIN)
        _write_config(cfg_file, OTHER)
        obj._last_skin_check = time.monotonic()  # just checked

        with patch("hermes_cli.config.get_config_path", return_value=cfg_file), \
             patch("hermes_cli.skin_engine.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.skin_engine.set_active_skin") as activate:
            obj._check_skin_reload()

        activate.assert_not_called()

    def test_event_path_ignores_fallback_throttle(self, tmp_path, capsys):
        """An fsnotify event reloads even when the fallback clock is fresh."""
        _write_skin(tmp_path, SKIN)
        obj, cfg_file = _make_cli(tmp_path, skin=SKIN)
        _write_skin(tmp_path, SKIN, bg="#2a2a36")  # same name, new palette
        _alive_watcher(obj)
        obj._last_skin_check = time.monotonic()  # fallback would suppress
        obj._skin_dirty.set()  # ...but fsnotify fired

        with patch("hermes_cli.config.get_config_path", return_value=cfg_file), \
             patch("hermes_cli.skin_engine.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.skin_engine.set_active_skin") as activate, \
             patch("cli._ACCENT"), \
             patch.object(obj, "_apply_tui_skin_style"):
            obj._check_skin_reload()

        activate.assert_called_once_with(SKIN)
        assert SKIN in capsys.readouterr().out

    def test_no_event_no_stat_no_reload(self, tmp_path, capsys):
        """Without an fsnotify event the tick does nothing (proves no polling)."""
        _write_skin(tmp_path, SKIN)
        obj, cfg_file = _make_cli(tmp_path, skin=SKIN)
        _write_config(cfg_file, OTHER)
        _write_skin(tmp_path, OTHER)
        _alive_watcher(obj)  # armed but silent — event never set

        with patch("hermes_cli.config.get_config_path", return_value=cfg_file) as get_path, \
             patch("hermes_cli.skin_engine.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.skin_engine.set_active_skin") as activate:
            obj._check_skin_reload()

        activate.assert_not_called()
        get_path.assert_not_called()  # config file not even read
        assert capsys.readouterr().out == ""

    def test_dead_watcher_falls_back_to_interval(self, tmp_path, capsys):
        """A dead fsnotify thread degrades to interval polling."""
        _write_skin(tmp_path, SKIN)
        obj, cfg_file = _make_cli(tmp_path, skin=SKIN)
        _write_skin(tmp_path, SKIN, bg="#2a2a36")
        obj._skin_dirty = threading.Event()
        dead = MagicMock()
        dead.is_alive.return_value = False
        obj._skin_watcher_thread = dead
        obj._last_skin_check = 0.0

        with patch("hermes_cli.config.get_config_path", return_value=cfg_file), \
             patch("hermes_cli.skin_engine.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.skin_engine.set_active_skin") as activate, \
             patch("cli._ACCENT"), \
             patch.object(obj, "_apply_tui_skin_style"):
            obj._check_skin_reload()

        activate.assert_called_once_with(SKIN)
        assert SKIN in capsys.readouterr().out

    def test_fsnotify_thread_fires_on_skin_write(self, tmp_path):
        """A real watchfiles thread sets the dirty event on skin save."""
        pytest.importorskip("watchfiles")
        _write_skin(tmp_path, SKIN)
        obj, cfg_file = _make_cli(tmp_path, skin=SKIN)
        obj._skin_dirty = threading.Event()

        with patch("hermes_cli.config.get_config_path", return_value=cfg_file), \
             patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            obj._start_skin_watcher()
            assert obj._skin_watcher_thread.is_alive()
            _write_skin(tmp_path, SKIN, bg="#3a3a46")
            assert obj._skin_dirty.wait(timeout=10), "fsnotify event never arrived"

    def _run_skin_command(self, obj, home, cmd):
        """Run /skin against a fake home; return (activate, accent, save, repaint)."""
        with patch("hermes_cli.skin_engine.get_hermes_home", return_value=home), \
             patch("hermes_cli.skin_engine.set_active_skin") as activate, \
             patch("cli._ACCENT") as accent, \
             patch("hermes_cli.cli_commands_mixin._save") as save, \
             patch.object(obj, "_apply_tui_skin_style") as repaint:
            obj._handle_skin_command(cmd)
        return activate, accent, save, repaint

    def test_skin_window_pin_does_not_reload_on_next_idle_tick(self, tmp_path, capsys):
        """An explicit window pin must not be activated a second time by the watcher."""
        _write_skin(tmp_path, SKIN)
        _write_skin(tmp_path, OTHER)
        obj, cfg_file = _make_cli(tmp_path, skin=SKIN)

        self._run_skin_command(obj, tmp_path, f"/skin {OTHER} --window")
        capsys.readouterr()  # discard the command's intentional confirmation
        activate, _, repaint = _run_check(obj, cfg_file, tmp_path)

        activate.assert_not_called()
        repaint.assert_not_called()
        assert capsys.readouterr().out == ""

    def test_skin_window_flag_pins_override(self, tmp_path, capsys):
        """/skin NAME --window pins this window without touching config."""
        _write_skin(tmp_path, SKIN)
        _write_skin(tmp_path, OTHER)
        obj, _ = _make_cli(tmp_path, skin=SKIN)

        activate, _, save, repaint = self._run_skin_command(obj, tmp_path, f"/skin {OTHER} --window")

        assert obj._session_skin_override == OTHER
        activate.assert_called_once_with(OTHER)
        save.assert_not_called()
        repaint.assert_called_once_with()
        assert "this window only" in capsys.readouterr().out

    def test_skin_plain_clears_override(self, tmp_path, capsys):
        """/skin NAME drops a window pin and saves the new default."""
        _write_skin(tmp_path, SKIN)
        _write_skin(tmp_path, OTHER)
        obj, _ = _make_cli(tmp_path, skin=SKIN)
        obj._session_skin_override = OTHER

        activate, _, save, _ = self._run_skin_command(obj, tmp_path, f"/skin {SKIN}")

        assert obj._session_skin_override is None
        activate.assert_called_once_with(SKIN)
        save.assert_called_once_with("display.skin", SKIN)
        assert "this window only" not in capsys.readouterr().out

    def test_skin_window_flag_unknown_sets_nothing(self, tmp_path, capsys):
        """/skin BOGUS --window reports the error and pins nothing."""
        _write_skin(tmp_path, SKIN)
        obj, _ = _make_cli(tmp_path, skin=SKIN)

        activate, _, save, _ = self._run_skin_command(obj, tmp_path, "/skin no-such-skin --window")

        assert getattr(obj, "_session_skin_override", None) is None
        activate.assert_not_called()
        save.assert_not_called()
        assert "no-such-skin" in capsys.readouterr().out

    def _baseline_override(self, obj, home, name):
        """Pretend the window was pinned earlier (sig already caught up)."""
        from hermes_cli.skin_engine import user_skin_file_signature
        with patch("hermes_cli.skin_engine.get_hermes_home", return_value=home):
            obj._session_skin_override = name
            obj._skin_sig = (name, user_skin_file_signature(name))

    def test_override_ignores_default_change(self, tmp_path, capsys):
        """A pinned window keeps its skin when the default changes."""
        _write_skin(tmp_path, SKIN)
        _write_skin(tmp_path, OTHER)
        obj, cfg_file = _make_cli(tmp_path, skin=SKIN)
        self._baseline_override(obj, tmp_path, OTHER)
        _write_config(cfg_file, "third-theme")

        activate, _, _ = _run_check(obj, cfg_file, tmp_path)

        activate.assert_not_called()
        assert capsys.readouterr().out == ""

    def test_override_file_rewrite_reloads(self, tmp_path, capsys):
        """A pinned window follows its own skin file's rewrites."""
        _write_skin(tmp_path, SKIN)
        _write_skin(tmp_path, OTHER)
        obj, cfg_file = _make_cli(tmp_path, skin=SKIN)
        self._baseline_override(obj, tmp_path, OTHER)
        _write_skin(tmp_path, OTHER, bg="#3a3a46")

        activate, _, _ = _run_check(obj, cfg_file, tmp_path)

        activate.assert_called_once_with(OTHER)
        assert OTHER in capsys.readouterr().out

    def test_override_unknown_name_warns_once(self, tmp_path, capsys):
        """A pin pointing at a removed skin warns once and stays put."""
        _write_skin(tmp_path, SKIN)
        obj, cfg_file = _make_cli(tmp_path, skin=SKIN)
        # Pin was set when the file existed; the file has since been deleted.
        obj._session_skin_override = "gone-skin"

        activate, _, _ = _run_check(obj, cfg_file, tmp_path)

        activate.assert_not_called()
        assert "gone-skin" in capsys.readouterr().out
        activate2, _, _ = _run_check(obj, cfg_file, tmp_path)
        activate2.assert_not_called()
        assert capsys.readouterr().out == ""

    def test_missing_file_signature(self, tmp_path):
        """Built-in / absent skin files report (False, 0, 0)."""
        from hermes_cli.skin_engine import user_skin_file_signature
        with patch("hermes_cli.skin_engine.get_hermes_home", return_value=tmp_path):
            assert user_skin_file_signature("default") == (False, 0, 0)
            _write_skin(tmp_path, SKIN)
            exists, mtime_ns, size = user_skin_file_signature(SKIN)
            assert exists and mtime_ns > 0 and size > 0
