"""Tests for the light-mode terminal detection + color remap in cli.py.

Covers the env-override path and the SkinConfig.get_color() wrapper that
the resize / light-mode salvage installs at module import time.  We don't
try to fake an OSC 11 reply — the env-override branch short-circuits
before the terminal query, which is the path most users hit.
"""

from __future__ import annotations


import pytest


@pytest.fixture
def cli_mod(monkeypatch):
    """Import cli with the light-mode cache cleared each test."""
    import cli as _cli

    # The module-level _install_skin_light_mode_hook() and import-time
    # _detect_light_mode() prime ran once at first import.  We just reset
    # the detection cache so the per-test env override takes effect.
    monkeypatch.setattr(_cli, "_LIGHT_MODE_CACHE", None)
    return _cli


class TestLightModeDetection:
    def test_hermes_light_env_true_forces_light(self, cli_mod, monkeypatch):
        monkeypatch.setenv("HERMES_LIGHT", "1")
        assert cli_mod._detect_light_mode() is True

    def test_hermes_light_env_false_forces_dark(self, cli_mod, monkeypatch):
        monkeypatch.setenv("HERMES_LIGHT", "0")
        # Also blank out other signals so nothing else flips it light.
        monkeypatch.delenv("HERMES_TUI_LIGHT", raising=False)
        monkeypatch.delenv("HERMES_TUI_THEME", raising=False)
        monkeypatch.delenv("HERMES_TUI_BACKGROUND", raising=False)
        monkeypatch.delenv("COLORFGBG", raising=False)
        assert cli_mod._detect_light_mode() is False

    def test_theme_hint_light(self, cli_mod, monkeypatch):
        monkeypatch.delenv("HERMES_LIGHT", raising=False)
        monkeypatch.delenv("HERMES_TUI_LIGHT", raising=False)
        monkeypatch.setenv("HERMES_TUI_THEME", "light")
        assert cli_mod._detect_light_mode() is True

    def test_background_hex_hint_light(self, cli_mod, monkeypatch):
        monkeypatch.delenv("HERMES_LIGHT", raising=False)
        monkeypatch.delenv("HERMES_TUI_LIGHT", raising=False)
        monkeypatch.delenv("HERMES_TUI_THEME", raising=False)
        monkeypatch.setenv("HERMES_TUI_BACKGROUND", "#FFFFFF")
        assert cli_mod._detect_light_mode() is True

    def test_background_hex_hint_dark(self, cli_mod, monkeypatch):
        monkeypatch.delenv("HERMES_LIGHT", raising=False)
        monkeypatch.delenv("HERMES_TUI_LIGHT", raising=False)
        monkeypatch.delenv("HERMES_TUI_THEME", raising=False)
        monkeypatch.setenv("HERMES_TUI_BACKGROUND", "#1a1a2e")
        monkeypatch.delenv("COLORFGBG", raising=False)
        assert cli_mod._detect_light_mode() is False

    def test_colorfgbg_light_bg_slot(self, cli_mod, monkeypatch):
        monkeypatch.delenv("HERMES_LIGHT", raising=False)
        monkeypatch.delenv("HERMES_TUI_LIGHT", raising=False)
        monkeypatch.delenv("HERMES_TUI_THEME", raising=False)
        monkeypatch.delenv("HERMES_TUI_BACKGROUND", raising=False)
        monkeypatch.setenv("COLORFGBG", "0;15")  # bg slot 15 = light
        assert cli_mod._detect_light_mode() is True

    def test_cache_is_sticky(self, cli_mod, monkeypatch):
        monkeypatch.setenv("HERMES_LIGHT", "1")
        assert cli_mod._detect_light_mode() is True
        # Even if the env flips, the cached result wins until reset.
        monkeypatch.setenv("HERMES_LIGHT", "0")
        assert cli_mod._detect_light_mode() is True


class TestOsc11Probe:
    """The OSC 11 background probe must never run where its reply can leak
    into prompt_toolkit's input (a late BEL-terminated reply reads as Ctrl+G
    = open-editor, trapping the user in a stray editor). Guard the cases we
    refuse to probe in.
    """

    @pytest.mark.parametrize("var", ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"))
    def test_skips_over_ssh(self, cli_mod, monkeypatch, var):
        monkeypatch.setattr(cli_mod.sys.stdin, "isatty", lambda: True, raising=False)
        monkeypatch.setattr(cli_mod.sys.stdout, "isatty", lambda: True, raising=False)
        for v in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"):
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv(var, "1.2.3.4 5555 22")
        assert cli_mod._query_osc11_background() is None

    def test_skips_when_not_a_tty(self, cli_mod, monkeypatch):
        monkeypatch.setattr(cli_mod.sys.stdin, "isatty", lambda: False, raising=False)
        assert cli_mod._query_osc11_background() is None


class TestLightModeRemap:
    def test_remap_no_op_in_dark_mode(self, cli_mod, monkeypatch):
        monkeypatch.setenv("HERMES_LIGHT", "0")
        # Cache is None from the fixture; first call sticks at False.
        assert cli_mod._maybe_remap_for_light_mode("#FFF8DC") == "#FFF8DC"

    def test_remap_known_dark_color(self, cli_mod, monkeypatch):
        monkeypatch.setenv("HERMES_LIGHT", "1")
        # Force the detect cache to True for this test.
        cli_mod._LIGHT_MODE_CACHE = True
        assert cli_mod._maybe_remap_for_light_mode("#FFF8DC") == "#1A1A1A"
        assert cli_mod._maybe_remap_for_light_mode("#FFD700") == "#9A6B00"

    def test_remap_case_insensitive(self, cli_mod, monkeypatch):
        cli_mod._LIGHT_MODE_CACHE = True
        # Lowercase input should still remap.
        assert cli_mod._maybe_remap_for_light_mode("#fff8dc") == "#1A1A1A"

    def test_remap_unknown_color_passthrough(self, cli_mod, monkeypatch):
        cli_mod._LIGHT_MODE_CACHE = True
        # A color not in the remap table is returned unchanged.
        assert cli_mod._maybe_remap_for_light_mode("#ABCDEF") == "#ABCDEF"

    def test_remap_skips_statusbar_paired_colors(self, cli_mod, monkeypatch):
        """Colors that live on a dark bg (status bar fg) MUST NOT be
        remapped — otherwise they go dark-on-dark and disappear.

        Regression guard for the patch-11 fix (intentional table omission).
        """
        cli_mod._LIGHT_MODE_CACHE = True
        for fg in ("#C0C0C0", "#888888", "#555555", "#8B8682"):
            assert cli_mod._maybe_remap_for_light_mode(fg) == fg, (
                f"{fg} is a status-bar fg paired with dark bg; remapping it "
                "would produce dark-on-dark"
            )


class TestOsc11ProbeAllowList:
    """Guards the OSC 11 background-color probe behind a terminal allow-list
    so the reply does not leak into prompt_toolkit's input buffer on
    terminals like macOS Terminal.app.  Regression for issue #30092.
    """

    def _clear_env(self, monkeypatch):
        for var in (
            "HERMES_DISABLE_OSC11",
            "TERM_PROGRAM",
            "TERM",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_apple_terminal_is_unsafe(self, cli_mod, monkeypatch):
        self._clear_env(monkeypatch)
        monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
        assert cli_mod._osc11_probe_is_safe() is False

    def test_unknown_term_program_is_unsafe(self, cli_mod, monkeypatch):
        self._clear_env(monkeypatch)
        monkeypatch.setenv("TERM_PROGRAM", "SomeOtherTerminalEmulator")
        assert cli_mod._osc11_probe_is_safe() is False

    def test_no_terminal_env_is_unsafe(self, cli_mod, monkeypatch):
        self._clear_env(monkeypatch)
        assert cli_mod._osc11_probe_is_safe() is False

    @pytest.mark.parametrize("prog", ["iTerm.app", "WezTerm", "ghostty", "Hyper"])
    def test_known_safe_term_program(self, cli_mod, monkeypatch, prog):
        self._clear_env(monkeypatch)
        monkeypatch.setenv("TERM_PROGRAM", prog)
        assert cli_mod._osc11_probe_is_safe() is True

    @pytest.mark.parametrize(
        "term",
        ["xterm-kitty", "foot", "alacritty", "tmux-256color", "screen.xterm-256color"],
    )
    def test_known_safe_term_prefix(self, cli_mod, monkeypatch, term):
        self._clear_env(monkeypatch)
        monkeypatch.setenv("TERM", term)
        assert cli_mod._osc11_probe_is_safe() is True

    @pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
    def test_hermes_disable_osc11_forces_unsafe(self, cli_mod, monkeypatch, val):
        self._clear_env(monkeypatch)
        monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")  # would otherwise be safe
        monkeypatch.setenv("HERMES_DISABLE_OSC11", val)
        assert cli_mod._osc11_probe_is_safe() is False

    def test_hermes_disable_osc11_falsey_keeps_safe(self, cli_mod, monkeypatch):
        self._clear_env(monkeypatch)
        monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
        monkeypatch.setenv("HERMES_DISABLE_OSC11", "0")
        assert cli_mod._osc11_probe_is_safe() is True

    def test_detect_light_mode_skips_osc11_on_unsafe_terminal(
        self, cli_mod, monkeypatch
    ):
        """Mode S regression: when the probe is unsafe, _detect_light_mode
        must NOT call _query_osc11_background — otherwise the reply leaks
        into prompt_toolkit input on terminals like Apple_Terminal."""
        self._clear_env(monkeypatch)
        monkeypatch.delenv("HERMES_LIGHT", raising=False)
        monkeypatch.delenv("HERMES_TUI_LIGHT", raising=False)
        monkeypatch.delenv("HERMES_TUI_THEME", raising=False)
        monkeypatch.delenv("HERMES_TUI_BACKGROUND", raising=False)
        monkeypatch.delenv("COLORFGBG", raising=False)
        monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")

        called = {"n": 0}

        def _fake_probe():
            called["n"] += 1
            return "#FFFFFF"

        monkeypatch.setattr(cli_mod, "_query_osc11_background", _fake_probe)
        cli_mod._detect_light_mode()
        assert called["n"] == 0, "OSC 11 probe must not run on Apple_Terminal"

    def test_detect_light_mode_runs_osc11_on_safe_terminal(
        self, cli_mod, monkeypatch
    ):
        """Counterpart: on a known-safe terminal, the OSC 11 probe still
        runs so light-mode auto-detection keeps working."""
        self._clear_env(monkeypatch)
        monkeypatch.delenv("HERMES_LIGHT", raising=False)
        monkeypatch.delenv("HERMES_TUI_LIGHT", raising=False)
        monkeypatch.delenv("HERMES_TUI_THEME", raising=False)
        monkeypatch.delenv("HERMES_TUI_BACKGROUND", raising=False)
        monkeypatch.delenv("COLORFGBG", raising=False)
        monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")

        called = {"n": 0}

        def _fake_probe():
            called["n"] += 1
            return "#FFFFFF"  # a light bg

        monkeypatch.setattr(cli_mod, "_query_osc11_background", _fake_probe)
        result = cli_mod._detect_light_mode()
        assert called["n"] == 1
        assert result is True


class TestSkinConfigHook:
    """The salvage wraps SkinConfig.get_color at module import time so
    every skin color read goes through the light-mode remap.  Verify
    the hook installed and functions correctly.
    """

    def test_hook_installed(self, cli_mod):
        from hermes_cli.skin_engine import SkinConfig

        assert getattr(SkinConfig, "_hermes_light_mode_hook_installed", False) is True

    def test_hook_is_idempotent(self, cli_mod):
        # Calling the installer twice must not double-wrap (the marker
        # attribute is the guard).
        from hermes_cli.skin_engine import SkinConfig

        before = SkinConfig.get_color
        cli_mod._install_skin_light_mode_hook()
        after = SkinConfig.get_color
        assert before is after

    def test_skin_color_remaps_through_wrapper_in_light_mode(
        self, cli_mod, monkeypatch
    ):
        from hermes_cli.skin_engine import SkinConfig

        cli_mod._LIGHT_MODE_CACHE = True
        skin = SkinConfig(
            name="test",
            colors={"banner_text": "#FFF8DC", "response_border": "#FFD700"},
        )
        # The wrapper kicks in at get_color, not at construction time.
        assert skin.get_color("banner_text") == "#1A1A1A"
        assert skin.get_color("response_border") == "#9A6B00"

    def test_skin_color_passthrough_in_dark_mode(self, cli_mod, monkeypatch):
        from hermes_cli.skin_engine import SkinConfig

        cli_mod._LIGHT_MODE_CACHE = False
        skin = SkinConfig(name="test", colors={"banner_text": "#FFF8DC"})
        assert skin.get_color("banner_text") == "#FFF8DC"
