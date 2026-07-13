"""Tests for hermes_cli.skin_engine — the data-driven skin/theme system."""

import os
import pytest


@pytest.fixture(autouse=True)
def reset_skin_state():
    """Reset skin engine state between tests."""
    from hermes_cli import skin_engine
    skin_engine._active_skin = None
    skin_engine._active_skin_name = "default"
    skin_engine._system_dark_mode = None
    # Stop any running monitor
    if skin_engine._skin_monitor_thread is not None:
        skin_engine.stop_skin_auto_switch_monitor()
    skin_engine._skin_monitor_thread = None
    skin_engine._skin_monitor_stop = None
    skin_engine._skin_switch_event = None
    yield
    skin_engine._active_skin = None
    skin_engine._active_skin_name = "default"
    skin_engine._system_dark_mode = None
    if skin_engine._skin_monitor_thread is not None:
        skin_engine.stop_skin_auto_switch_monitor()


class TestSkinConfig:
    def test_default_skin_has_required_fields(self):
        from hermes_cli.skin_engine import load_skin
        skin = load_skin("default")
        assert skin.name == "default"
        assert skin.tool_prefix == "┊"
        assert "banner_title" in skin.colors
        assert "banner_border" in skin.colors
        assert "agent_name" in skin.branding

    def test_get_color_with_fallback(self):
        from hermes_cli.skin_engine import load_skin
        skin = load_skin("default")
        assert skin.get_color("banner_title") == "#FFD700"
        assert skin.get_color("nonexistent", "#000") == "#000"

    def test_get_branding_with_fallback(self):
        from hermes_cli.skin_engine import load_skin
        skin = load_skin("default")
        assert skin.get_branding("agent_name") == "Hermes Agent"
        assert skin.get_branding("nonexistent", "fallback") == "fallback"

    def test_get_spinner_wings_empty_for_default(self):
        from hermes_cli.skin_engine import load_skin
        skin = load_skin("default")
        assert skin.get_spinner_wings() == []


class TestBuiltinSkins:
    def test_ares_skin_loads(self):
        from hermes_cli.skin_engine import load_skin
        skin = load_skin("ares")
        assert skin.name == "ares"
        assert skin.tool_prefix == "╎"
        assert skin.get_color("banner_border") == "#9F1C1C"
        assert skin.get_color("response_border") == "#C7A96B"
        assert skin.get_color("session_label") == "#C7A96B"
        assert skin.get_color("session_border") == "#6E584B"
        assert skin.get_branding("agent_name") == "Ares Agent"

    def test_ares_has_spinner_customization(self):
        from hermes_cli.skin_engine import load_skin
        skin = load_skin("ares")
        wings = skin.get_spinner_wings()
        assert len(wings) > 0
        assert isinstance(wings[0], tuple)
        assert len(wings[0]) == 2

    def test_mono_skin_loads(self):
        from hermes_cli.skin_engine import load_skin
        skin = load_skin("mono")
        assert skin.name == "mono"
        assert skin.get_color("banner_title") == "#e6edf3"

    def test_slate_skin_loads(self):
        from hermes_cli.skin_engine import load_skin
        skin = load_skin("slate")
        assert skin.name == "slate"
        assert skin.get_color("banner_title") == "#7eb8f6"

    def test_daylight_skin_loads(self):
        from hermes_cli.skin_engine import load_skin

        skin = load_skin("daylight")
        assert skin.name == "daylight"
        assert skin.tool_prefix == "│"
        assert skin.get_color("banner_title") == "#0F172A"
        assert skin.get_color("status_bar_bg") == "#E5EDF8"
        assert skin.get_color("voice_status_bg") == "#E5EDF8"
        assert skin.get_color("completion_menu_bg") == "#F8FAFC"
        assert skin.get_color("completion_menu_current_bg") == "#DBEAFE"
        assert skin.get_color("completion_menu_meta_bg") == "#EEF2FF"
        assert skin.get_color("completion_menu_meta_current_bg") == "#BFDBFE"

    def test_warm_lightmode_skin_loads(self):
        from hermes_cli.skin_engine import load_skin

        skin = load_skin("warm-lightmode")
        assert skin.name == "warm-lightmode"
        assert skin.get_color("banner_text") == "#2C1810"
        assert skin.get_color("completion_menu_bg") == "#F5EFE0"

    def test_charizard_skin_has_dark_ember_completion_menu(self):
        from hermes_cli.skin_engine import load_skin

        skin = load_skin("charizard")
        assert skin.name == "charizard"
        assert skin.get_color("banner_dim") == "#C58A45"
        assert skin.get_color("completion_menu_bg") == "#0B0503"
        assert skin.get_color("completion_menu_current_bg") == "#4A1B07"
        assert skin.get_color("completion_menu_meta_bg") == "#120806"
        assert skin.get_color("completion_menu_meta_current_bg") == "#5A260D"
        assert skin.get_color("selection_bg") == "#5A260D"

    def test_unknown_skin_falls_back_to_default(self):
        from hermes_cli.skin_engine import load_skin
        skin = load_skin("nonexistent_skin_xyz")
        assert skin.name == "default"

    def test_all_builtin_skins_have_complete_colors(self):
        from hermes_cli.skin_engine import _BUILTIN_SKINS, _build_skin_config
        required_keys = ["banner_border", "banner_title", "banner_accent",
                         "banner_dim", "banner_text", "ui_accent"]
        for name, data in _BUILTIN_SKINS.items():
            skin = _build_skin_config(data)
            for key in required_keys:
                assert key in skin.colors, f"Skin '{name}' missing color '{key}'"


class TestSkinManagement:
    def test_set_active_skin(self):
        from hermes_cli.skin_engine import set_active_skin, get_active_skin, get_active_skin_name
        skin = set_active_skin("ares")
        assert skin.name == "ares"
        assert get_active_skin_name() == "ares"
        assert get_active_skin().name == "ares"

    def test_get_active_skin_defaults(self):
        from hermes_cli.skin_engine import get_active_skin
        skin = get_active_skin()
        assert skin.name == "default"

    def test_list_skins_includes_builtins(self):
        from hermes_cli.skin_engine import list_skins
        skins = list_skins()
        names = [s["name"] for s in skins]
        assert "default" in names
        assert "ares" in names
        assert "mono" in names
        assert "slate" in names
        assert "daylight" in names
        assert "warm-lightmode" in names
        for s in skins:
            assert "source" in s
            assert s["source"] == "builtin"

    def test_init_skin_from_config(self):
        from hermes_cli.skin_engine import init_skin_from_config, get_active_skin_name
        init_skin_from_config({"display": {"skin": "ares"}})
        assert get_active_skin_name() == "ares"

    def test_init_skin_from_empty_config(self):
        from hermes_cli.skin_engine import init_skin_from_config, get_active_skin_name
        init_skin_from_config({})
        assert get_active_skin_name() == "default"

    def test_init_skin_from_null_display(self):
        """display: null should fall back to default, not crash."""
        from hermes_cli.skin_engine import init_skin_from_config, get_active_skin_name
        init_skin_from_config({"display": None})
        assert get_active_skin_name() == "default"

    def test_init_skin_from_non_dict_display(self):
        """display: <non-dict> should fall back to default."""
        from hermes_cli.skin_engine import init_skin_from_config, get_active_skin_name
        init_skin_from_config({"display": "invalid"})
        assert get_active_skin_name() == "default"

        init_skin_from_config({"display": 42})
        assert get_active_skin_name() == "default"

        init_skin_from_config({"display": []})
        assert get_active_skin_name() == "default"


class TestUserSkins:
    def test_load_user_skin_from_yaml(self, tmp_path, monkeypatch):
        from hermes_cli.skin_engine import load_skin
        # Create a user skin YAML
        skins_dir = tmp_path / "skins"
        skins_dir.mkdir()
        skin_file = skins_dir / "custom.yaml"
        skin_data = {
            "name": "custom",
            "description": "A custom test skin",
            "colors": {"banner_title": "#FF0000"},
            "branding": {"agent_name": "Custom Agent"},
            "tool_prefix": "▸",
        }
        import yaml
        skin_file.write_text(yaml.dump(skin_data))

        # Patch skins dir
        monkeypatch.setattr("hermes_cli.skin_engine._skins_dir", lambda: skins_dir)

        skin = load_skin("custom")
        assert skin.name == "custom"
        assert skin.get_color("banner_title") == "#FF0000"
        assert skin.get_branding("agent_name") == "Custom Agent"
        assert skin.tool_prefix == "▸"
        # Should inherit defaults for unspecified colors
        assert skin.get_color("banner_border") == "#CD7F32"  # from default

    def test_load_user_skin_invalid_section_types_fall_back_to_defaults(self, tmp_path, monkeypatch):
        from hermes_cli.skin_engine import load_skin

        skins_dir = tmp_path / "skins"
        skins_dir.mkdir()
        import yaml

        (skins_dir / "broken.yaml").write_text(
            yaml.dump(
                {
                    "name": "broken",
                    "colors": ["not", "a", "mapping"],
                    "spinner": "invalid",
                    "branding": ["also", "invalid"],
                    "tool_emojis": ["invalid"],
                    "tool_prefix": "!",
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr("hermes_cli.skin_engine._skins_dir", lambda: skins_dir)

        skin = load_skin("broken")

        assert skin.name == "broken"
        assert skin.get_color("banner_title") == "#FFD700"
        assert skin.get_branding("agent_name") == "Hermes Agent"
        assert skin.spinner.get("waiting_faces", []) == []
        assert skin.tool_emojis == {}
        assert skin.tool_prefix == "!"

    def test_list_skins_includes_user_skins(self, tmp_path, monkeypatch):
        from hermes_cli.skin_engine import list_skins
        skins_dir = tmp_path / "skins"
        skins_dir.mkdir()
        import yaml
        (skins_dir / "pirate.yaml").write_text(yaml.dump({
            "name": "pirate",
            "description": "Arr matey",
        }))
        monkeypatch.setattr("hermes_cli.skin_engine._skins_dir", lambda: skins_dir)

        skins = list_skins()
        names = [s["name"] for s in skins]
        assert "pirate" in names
        pirate = [s for s in skins if s["name"] == "pirate"][0]
        assert pirate["source"] == "user"


class TestDisplayIntegration:
    def test_get_skin_tool_prefix_default(self):
        from agent.display import get_skin_tool_prefix
        assert get_skin_tool_prefix() == "┊"

    def test_get_skin_tool_prefix_custom(self):
        from hermes_cli.skin_engine import set_active_skin
        from agent.display import get_skin_tool_prefix
        set_active_skin("ares")
        assert get_skin_tool_prefix() == "╎"

    def test_tool_message_uses_skin_prefix(self):
        from hermes_cli.skin_engine import set_active_skin
        from agent.display import get_cute_tool_message
        set_active_skin("ares")
        msg = get_cute_tool_message("terminal", {"command": "ls"}, 0.5)
        assert msg.startswith("╎")
        assert "┊" not in msg

    def test_tool_message_default_prefix(self):
        from agent.display import get_cute_tool_message
        msg = get_cute_tool_message("terminal", {"command": "ls"}, 0.5)
        assert msg.startswith("┊")


class TestCliBrandingHelpers:
    def test_active_prompt_symbol_default(self):
        from hermes_cli.skin_engine import get_active_prompt_symbol

        assert get_active_prompt_symbol() == "❯ "

    def test_active_prompt_symbol_ares(self):
        from hermes_cli.skin_engine import set_active_skin, get_active_prompt_symbol

        set_active_skin("ares")
        assert get_active_prompt_symbol() == "⚔ "

    def test_active_help_header_ares(self):
        from hermes_cli.skin_engine import set_active_skin, get_active_help_header

        set_active_skin("ares")
        assert get_active_help_header() == "(⚔) Available Commands"

    def test_active_goodbye_ares(self):
        from hermes_cli.skin_engine import set_active_skin, get_active_goodbye

        set_active_skin("ares")
        assert get_active_goodbye() == "Farewell, warrior! ⚔"

    def test_prompt_toolkit_style_overrides_cover_tui_classes(self):
        from hermes_cli.skin_engine import set_active_skin, get_prompt_toolkit_style_overrides
        set_active_skin("ares")
        overrides = get_prompt_toolkit_style_overrides()
        required = {
            "input-area",
            "placeholder",
            "prompt",
            "prompt-working",
            "hint",
            "status-bar",
            "status-bar-strong",
            "status-bar-dim",
            "status-bar-good",
            "status-bar-warn",
            "status-bar-bad",
            "status-bar-critical",
            "input-rule",
            "image-badge",
            "completion-menu",
            "completion-menu.completion",
            "completion-menu.completion.current",
            "completion-menu.meta.completion",
            "completion-menu.meta.completion.current",
            "status-bar",
            "status-bar-strong",
            "status-bar-dim",
            "status-bar-good",
            "status-bar-warn",
            "status-bar-bad",
            "status-bar-critical",
            "voice-status",
            "voice-status-recording",
            "clarify-border",
            "clarify-title",
            "clarify-question",
            "clarify-choice",
            "clarify-selected",
            "clarify-active-other",
            "clarify-countdown",
            "sudo-prompt",
            "sudo-border",
            "sudo-title",
            "sudo-text",
            "approval-border",
            "approval-title",
            "approval-desc",
            "approval-cmd",
            "approval-choice",
            "approval-selected",
        }
        assert required.issubset(overrides.keys())

    def test_prompt_toolkit_style_overrides_use_skin_colors(self):
        from hermes_cli.skin_engine import (
            set_active_skin,
            get_active_skin,
            get_prompt_toolkit_style_overrides,
        )

        set_active_skin("ares")
        skin = get_active_skin()
        overrides = get_prompt_toolkit_style_overrides()
        assert overrides["prompt"] == skin.get_color("prompt")
        assert overrides["input-rule"] == skin.get_color("input_rule")
        assert overrides["status-bar"] == (
            f"bg:{skin.get_color('status_bar_bg')} {skin.get_color('status_bar_text')}"
        )
        assert overrides["status-bar-strong"] == (
            f"bg:{skin.get_color('status_bar_bg')} {skin.get_color('status_bar_strong')} bold"
        )
        assert overrides["status-bar-critical"] == (
            f"bg:{skin.get_color('status_bar_bg')} {skin.get_color('status_bar_critical')} bold"
        )
        assert overrides["clarify-title"] == f"{skin.get_color('banner_title')} bold"
        assert overrides["sudo-prompt"] == f"{skin.get_color('ui_error')} bold"
        assert overrides["approval-title"] == f"{skin.get_color('ui_warn')} bold"

        set_active_skin("daylight")
        skin = get_active_skin()
        overrides = get_prompt_toolkit_style_overrides()
        assert overrides["status-bar"] == f"bg:{skin.get_color('status_bar_bg')} {skin.get_color('banner_text')}"
        assert overrides["voice-status"] == f"bg:{skin.get_color('voice_status_bg')} {skin.get_color('ui_label')}"


# =============================================================================
# Auto-switch skin tests
# =============================================================================


_ENV_KEYS = ("COLORFGBG", "HERMES_TUI_THEME", "HERMES_LIGHT", "HERMES_TUI_LIGHT")


@pytest.fixture()
def clean_env():
    """Remove all detection env vars and restore after test."""
    saved = {k: os.environ.pop(k, None) for k in _ENV_KEYS}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v
        elif k in os.environ:
            del os.environ[k]


class TestDetectSystemDarkMode:
    """Tests for _detect_system_dark_mode()."""

    def test_returns_bool(self):
        from hermes_cli.skin_engine import _detect_system_dark_mode
        result = _detect_system_dark_mode()
        assert isinstance(result, bool)

    def test_detects_dark_via_defaults(self):
        """When `defaults read` returns 'Dark', detects dark mode."""
        import subprocess
        orig_run = subprocess.run
        def mock_run(cmd, **kwargs):
            if isinstance(cmd, list) and "AppleInterfaceStyle" in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout="Dark\n", stderr="")
            return orig_run(cmd, **kwargs)
        subprocess.run = mock_run
        try:
            from hermes_cli.skin_engine import _detect_system_dark_mode
            assert _detect_system_dark_mode() is True
        finally:
            subprocess.run = orig_run

    def test_detects_light_via_defaults(self, clean_env):
        """When `defaults read` returns exit 1, detects light mode."""
        import subprocess
        orig_run = subprocess.run
        def mock_run(cmd, **kwargs):
            if isinstance(cmd, list) and "AppleInterfaceStyle" in cmd:
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="does not exist")
            return orig_run(cmd, **kwargs)
        subprocess.run = mock_run
        try:
            import hermes_cli.skin_engine as se
            # Mock the lazy-imported detector so it doesn't override with
            # a stale cached value from a prior test.
            import types
            fake_cli = types.ModuleType("cli")
            fake_cli._detect_light_mode = lambda: True  # say "light" → dark=False
            import sys
            sys.modules["cli"] = fake_cli
            try:
                assert se._detect_system_dark_mode() is False
            finally:
                del sys.modules["cli"]
        finally:
            subprocess.run = orig_run

    def test_colorfgbg_light(self, clean_env):
        """COLORFGBG with bg=15 means light mode."""
        os.environ["COLORFGBG"] = "0;15"
        import subprocess
        orig_run = subprocess.run
        subprocess.run = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("skip"))
        try:
            from hermes_cli.skin_engine import _detect_system_dark_mode
            assert _detect_system_dark_mode() is False
        finally:
            subprocess.run = orig_run

    def test_colorfgbg_dark(self, clean_env):
        """COLORFGBG with bg=0 means dark mode."""
        os.environ["COLORFGBG"] = "0;0"
        import subprocess
        orig_run = subprocess.run
        subprocess.run = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("skip"))
        try:
            from hermes_cli.skin_engine import _detect_system_dark_mode
            assert _detect_system_dark_mode() is True
        finally:
            subprocess.run = orig_run

    def test_theme_env_dark(self, clean_env):
        """HERMES_TUI_THEME=dark means dark mode."""
        os.environ["HERMES_TUI_THEME"] = "dark"
        import subprocess
        orig_run = subprocess.run
        subprocess.run = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("skip"))
        try:
            from hermes_cli.skin_engine import _detect_system_dark_mode
            assert _detect_system_dark_mode() is True
        finally:
            subprocess.run = orig_run

    def test_theme_env_light(self, clean_env):
        """HERMES_TUI_THEME=light means light mode."""
        os.environ["HERMES_TUI_THEME"] = "light"
        import subprocess
        orig_run = subprocess.run
        subprocess.run = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("skip"))
        try:
            from hermes_cli.skin_engine import _detect_system_dark_mode
            assert _detect_system_dark_mode() is False
        finally:
            subprocess.run = orig_run

    def test_hermes_light_env(self, clean_env):
        """HERMES_LIGHT=1 means light mode."""
        os.environ["HERMES_LIGHT"] = "1"
        import subprocess
        orig_run = subprocess.run
        subprocess.run = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("skip"))
        try:
            from hermes_cli.skin_engine import _detect_system_dark_mode
            assert _detect_system_dark_mode() is False
        finally:
            subprocess.run = orig_run

    def test_defaults_to_dark(self, clean_env):
        """When no detection method works, defaults to dark."""
        import subprocess
        orig_run = subprocess.run
        subprocess.run = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("nope"))
        try:
            from hermes_cli.skin_engine import _detect_system_dark_mode
            assert _detect_system_dark_mode() is True  # defaults to dark
        finally:
            subprocess.run = orig_run


class TestDoSkinSwitch:
    """Tests for _do_skin_switch()."""

    def test_switches_to_light_when_light_mode(self, monkeypatch):
        """When system is light, switches to skin_light."""
        monkeypatch.setattr("hermes_cli.skin_engine._detect_system_dark_mode", lambda: False)

        from hermes_cli.skin_engine import _do_skin_switch, set_active_skin
        set_active_skin("default")

        config = {"display": {"skin_auto_switch": True, "skin_dark": "default", "skin_light": "daylight"}}
        result = _do_skin_switch(config)
        assert result is True
        from hermes_cli.skin_engine import get_active_skin_name
        assert get_active_skin_name() == "daylight"

    def test_switches_to_dark_when_dark_mode(self, monkeypatch):
        """When system is dark, switches to skin_dark."""
        monkeypatch.setattr("hermes_cli.skin_engine._detect_system_dark_mode", lambda: True)

        from hermes_cli.skin_engine import _do_skin_switch, set_active_skin
        set_active_skin("daylight")

        config = {"display": {"skin_auto_switch": True, "skin_dark": "default", "skin_light": "daylight"}}
        result = _do_skin_switch(config)
        assert result is True
        from hermes_cli.skin_engine import get_active_skin_name
        assert get_active_skin_name() == "default"

    def test_no_switch_when_already_correct(self, monkeypatch):
        """No switch when skin already matches system mode."""
        monkeypatch.setattr("hermes_cli.skin_engine._detect_system_dark_mode", lambda: True)

        from hermes_cli.skin_engine import _do_skin_switch, set_active_skin
        set_active_skin("default")

        config = {"display": {"skin_auto_switch": True, "skin_dark": "default", "skin_light": "daylight"}}
        result = _do_skin_switch(config)
        assert result is False
        from hermes_cli.skin_engine import get_active_skin_name
        assert get_active_skin_name() == "default"

    def test_disabled_config_returns_false(self, monkeypatch):
        """Returns False when skin_auto_switch is disabled."""
        monkeypatch.setattr("hermes_cli.skin_engine._detect_system_dark_mode", lambda: False)

        from hermes_cli.skin_engine import _do_skin_switch, set_active_skin
        set_active_skin("default")

        config = {"display": {"skin_auto_switch": False, "skin_dark": "default", "skin_light": "daylight"}}
        result = _do_skin_switch(config)
        assert result is False

    def test_missing_config_returns_false(self):
        """Returns False with empty/missing config."""
        from hermes_cli.skin_engine import _do_skin_switch
        assert _do_skin_switch({}) is False
        assert _do_skin_switch({"display": None}) is False

    def test_default_skin_names(self, monkeypatch):
        """Uses default skin_dark/skin_light when not specified."""
        monkeypatch.setattr("hermes_cli.skin_engine._detect_system_dark_mode", lambda: False)

        from hermes_cli.skin_engine import _do_skin_switch, set_active_skin
        set_active_skin("default")

        config = {"display": {"skin_auto_switch": True}}
        result = _do_skin_switch(config)
        assert result is True
        from hermes_cli.skin_engine import get_active_skin_name
        assert get_active_skin_name() == "daylight"  # default skin_light


class TestSkinMonitor:
    """Tests for start/stop_skin_auto_switch_monitor."""

    def test_start_creates_thread(self):
        from hermes_cli.skin_engine import start_skin_auto_switch_monitor, _skin_monitor_thread
        config = {"display": {"skin_auto_switch": True, "skin_dark": "default", "skin_light": "daylight"}}
        start_skin_auto_switch_monitor(config)
        import hermes_cli.skin_engine as se
        assert se._skin_monitor_thread is not None
        assert se._skin_monitor_thread.is_alive()
        assert se._skin_monitor_thread.name == "skin-auto-switch"

    def test_start_disabled_config_does_nothing(self):
        from hermes_cli.skin_engine import start_skin_auto_switch_monitor
        config = {"display": {"skin_auto_switch": False}}
        start_skin_auto_switch_monitor(config)
        import hermes_cli.skin_engine as se
        assert se._skin_monitor_thread is None

    def test_start_empty_config_does_nothing(self):
        from hermes_cli.skin_engine import start_skin_auto_switch_monitor
        start_skin_auto_switch_monitor({})
        import hermes_cli.skin_engine as se
        assert se._skin_monitor_thread is None

    def test_stop_kills_thread(self):
        from hermes_cli.skin_engine import start_skin_auto_switch_monitor, stop_skin_auto_switch_monitor
        config = {"display": {"skin_auto_switch": True, "skin_dark": "default", "skin_light": "daylight"}}
        start_skin_auto_switch_monitor(config)
        import hermes_cli.skin_engine as se
        assert se._skin_monitor_thread.is_alive()
        stop_skin_auto_switch_monitor()
        assert se._skin_monitor_thread is None

    def test_start_is_idempotent(self):
        from hermes_cli.skin_engine import start_skin_auto_switch_monitor
        config = {"display": {"skin_auto_switch": True, "skin_dark": "default", "skin_light": "daylight"}}
        start_skin_auto_switch_monitor(config)
        import hermes_cli.skin_engine as se
        t1 = se._skin_monitor_thread
        start_skin_auto_switch_monitor(config)
        t2 = se._skin_monitor_thread
        assert t1 is t2  # same thread, not restarted


class TestMaybeAutoSwitchSkin:
    """Tests for maybe_auto_switch_skin()."""

    def test_polling_fallback_when_no_monitor(self, monkeypatch):
        """Without monitor thread, falls back to polling."""
        monkeypatch.setattr("hermes_cli.skin_engine._detect_system_dark_mode", lambda: False)

        from hermes_cli.skin_engine import maybe_auto_switch_skin, set_active_skin
        set_active_skin("default")

        config = {"display": {"skin_auto_switch": True, "skin_dark": "default", "skin_light": "daylight"}}
        result = maybe_auto_switch_skin(config)
        assert result is True
        from hermes_cli.skin_engine import get_active_skin_name
        assert get_active_skin_name() == "daylight"

    def test_delegates_to_monitor_event(self, monkeypatch):
        """When monitor is running, only acts on event signal."""
        import threading
        from hermes_cli.skin_engine import (
            start_skin_auto_switch_monitor, stop_skin_auto_switch_monitor,
            maybe_auto_switch_skin, set_active_skin,
        )

        set_active_skin("default")
        config = {"display": {"skin_auto_switch": True, "skin_dark": "default", "skin_light": "daylight"}}
        start_skin_auto_switch_monitor(config)

        import hermes_cli.skin_engine as se

        # No event set — should not switch
        result = maybe_auto_switch_skin(config)
        assert result is False

        # Simulate notification setting the event
        se._skin_switch_event.set()
        monkeypatch.setattr("hermes_cli.skin_engine._detect_system_dark_mode", lambda: False)
        result = maybe_auto_switch_skin(config)
        assert result is True
        from hermes_cli.skin_engine import get_active_skin_name
        assert get_active_skin_name() == "daylight"

        stop_skin_auto_switch_monitor()


class TestMacOSNotificationCallback:
    """Tests for _macos_theme_observer_callback."""

    def test_callback_sets_event(self):
        import threading
        from hermes_cli.skin_engine import _macos_theme_observer_callback, _skin_switch_event
        import hermes_cli.skin_engine as se

        se._skin_switch_event = threading.Event()
        _macos_theme_observer_callback(None)
        assert se._skin_switch_event.is_set()

    def test_callback_no_event_does_not_crash(self):
        """Callback with _skin_switch_event=None should not crash."""
        from hermes_cli.skin_engine import _macos_theme_observer_callback
        import hermes_cli.skin_engine as se
        se._skin_switch_event = None
        _macos_theme_observer_callback(None)  # should not raise
