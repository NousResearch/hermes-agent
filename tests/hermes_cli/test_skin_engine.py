"""Tests for hermes_cli.skin_engine — the data-driven skin/theme system."""

import pytest


@pytest.fixture(autouse=True)
def reset_skin_state():
    """Reset skin engine state between tests."""
    from hermes_cli import skin_engine
    skin_engine._active_skin = None
    skin_engine._active_skin_name = "default"
    yield
    skin_engine._active_skin = None
    skin_engine._active_skin_name = "default"


class TestSkinConfig:
    def test_default_skin_has_required_fields(self):
        from hermes_cli.skin_engine import load_skin
        skin = load_skin("default")
        assert skin.name == "default"
        assert skin.tool_prefix == "┊"
        assert "banner_title" in skin.colors
        assert "banner_border" in skin.colors
        assert "agent_name" in skin.branding


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
        # Crimson identity: border stays red-dominant (exact values are owned
        # by the palette audit in test_skin_palettes.py, which enforces
        # contrast floors — don't pin literals here).
        border = skin.get_color("banner_border")
        r, g, b = (int(border[i:i + 2], 16) for i in (1, 3, 5))
        assert r > g and r > b, f"ares border lost its crimson: {border}"
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








class TestSkinManagement:
    def test_set_active_skin(self):
        from hermes_cli.skin_engine import set_active_skin, get_active_skin, get_active_skin_name
        skin = set_active_skin("ares")
        assert skin.name == "ares"
        assert get_active_skin_name() == "ares"
        assert get_active_skin().name == "ares"


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


    def test_tool_message_uses_skin_prefix(self):
        from hermes_cli.skin_engine import set_active_skin
        from agent.display import get_cute_tool_message
        set_active_skin("ares")
        msg = get_cute_tool_message("terminal", {"command": "ls"}, 0.5)
        assert msg.startswith("╎")
        assert "┊" not in msg


class TestCliBrandingHelpers:


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


class TestAutoSkinSelector:
    """Regression tests for the 'auto' virtual skin selector (issue #16330).

    'auto' must bypass the installed-skin validation in /skin and resolve
    to a concrete skin via the existing terminal detection chain rather than
    an OS-specific probe. The persisted config value stays 'auto' so the
    preference is re-evaluated on every session start. A genuine
    user-installed auto.yaml skin takes precedence over the virtual
    resolution (#49783).
    """

    def test_set_active_skin_auto_resolves_to_concrete_skin(self, monkeypatch, tmp_path):
        """set_active_skin('auto') must store a concrete name, not 'auto'."""
        from unittest.mock import patch
        from hermes_cli.skin_engine import set_active_skin, get_active_skin_name

        monkeypatch.setattr("hermes_cli.skin_engine._skins_dir", lambda: tmp_path)
        with patch("hermes_cli.skin_engine.resolve_auto_skin", return_value="default"):
            set_active_skin("auto")

        assert get_active_skin_name() != "auto", (
            "set_active_skin('auto') must resolve to a concrete skin name"
        )
        assert get_active_skin_name() == "default"

    def test_resolve_auto_skin_light_mode(self):
        """resolve_auto_skin() returns 'daylight' when _detect_light_mode() is True.

        Regression (review of #63730): the original port imported the
        wrong name (_is_light_mode); current main defines _detect_light_mode().
        A wrong name here means EVERY call falls through the except clause
        to 'default', silently disabling the whole feature -- so this test
        must patch the exact real name, not a stand-in.
        """
        from unittest.mock import MagicMock, patch
        from hermes_cli.skin_engine import resolve_auto_skin

        fake_cli = MagicMock()
        fake_cli._detect_light_mode.return_value = True
        with patch.dict("sys.modules", {"cli": fake_cli}):
            result = resolve_auto_skin()

        assert result == "daylight", (
            "'light' is not a built-in skin name -- the actual light-background "
            "skin is 'daylight'"
        )

    def test_resolve_auto_skin_dark_mode(self):
        """resolve_auto_skin() returns 'default' when _detect_light_mode() is False."""
        from unittest.mock import MagicMock, patch
        from hermes_cli.skin_engine import resolve_auto_skin

        fake_cli = MagicMock()
        fake_cli._detect_light_mode.return_value = False
        with patch.dict("sys.modules", {"cli": fake_cli}):
            result = resolve_auto_skin()

        assert result == "default"

    def test_resolve_auto_skin_real_cli_module_has_detect_light_mode(self, monkeypatch):
        """Import-safety net: assert the real cli module actually exposes
        _detect_light_mode, so this regression (wrong function name) can't
        silently reappear even if the mocked tests above don't catch a
        future rename."""
        import sys as _sys
        import types as _types

        try:
            import cli as _cli_mod
        except Exception:
            # Only stub optional heavy deps as a last resort, and only for
            # the duration of this test (monkeypatch auto-reverts), so a
            # real prompt_toolkit install elsewhere isn't shadowed for
            # other tests in this file.
            for _mod in ("prompt_toolkit", "dotenv"):
                if _mod not in _sys.modules:
                    stub = _types.ModuleType(_mod)
                    if _mod == "dotenv":
                        stub.load_dotenv = lambda *a, **k: None
                    monkeypatch.setitem(_sys.modules, _mod, stub)
            try:
                import cli as _cli_mod
            except Exception:
                import pytest as _pytest
                _pytest.skip("cli module could not be imported in this sandbox")
        assert hasattr(_cli_mod, "_detect_light_mode"), (
            "resolve_auto_skin() depends on cli._detect_light_mode existing "
            "-- if this is renamed again, resolve_auto_skin() must be updated too"
        )

    def test_resolve_auto_skin_falls_back_on_error(self):
        """resolve_auto_skin() returns 'default' when detection raises."""
        from unittest.mock import patch
        from hermes_cli.skin_engine import resolve_auto_skin

        # When cli import itself fails, must still return a safe default.
        with patch.dict("sys.modules", {"cli": None}):
            result = resolve_auto_skin()

        assert result == "default"

    def test_init_skin_from_config_persists_auto(self, monkeypatch, tmp_path):
        """display.skin: auto in config must resolve at startup without crashing."""
        from unittest.mock import patch
        from hermes_cli.skin_engine import init_skin_from_config, get_active_skin_name

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr("hermes_cli.skin_engine._skins_dir", lambda: tmp_path)
        with patch("hermes_cli.skin_engine.resolve_auto_skin", return_value="default"):
            init_skin_from_config({"display": {"skin": "auto"}})

        assert get_active_skin_name() == "default"

    def test_user_auto_yaml_skin_takes_precedence_over_virtual_resolution(self, monkeypatch, tmp_path):
        """Regression (review of #63730, issue #49783): a genuine
        user-installed skin literally named auto.yaml must NOT be shadowed
        by the virtual 'auto' selector. resolve_auto_skin() must not even
        be consulted when the file exists."""
        from unittest.mock import patch
        from hermes_cli.skin_engine import set_active_skin, get_active_skin_name

        skins_dir = tmp_path
        (skins_dir / "auto.yaml").write_text(
            "name: auto\ndescription: My custom auto theme\ncolors: {}\n"
        )
        monkeypatch.setattr("hermes_cli.skin_engine._skins_dir", lambda: skins_dir)

        with patch("hermes_cli.skin_engine.resolve_auto_skin") as mock_resolve:
            set_active_skin("auto")
            mock_resolve.assert_not_called()

        assert get_active_skin_name() == "auto", (
            "A real user auto.yaml skin must be loaded as-is, not "
            "virtual-resolved to daylight/default"
        )

    def test_skin_command_bypasses_validation_for_auto(self, monkeypatch, tmp_path):
        """/skin auto must reach set_active_skin() instead of being rejected
        by the 'Unknown skin' check (list_skins() never lists 'auto')."""
        from unittest.mock import MagicMock, patch

        try:
            from hermes_cli.cli_commands_mixin import CLICommandsMixin
            import cli as _cli_mod  # noqa: F401 -- exercise the full import chain
        except Exception:
            import pytest as _pytest
            _pytest.skip("cli module could not be imported in this sandbox")

        monkeypatch.setattr("hermes_cli.skin_engine._skins_dir", lambda: tmp_path)

        class _Host(CLICommandsMixin):
            def _apply_tui_skin_style(self):
                return False

        host = _Host()
        with patch("cli._ACCENT", MagicMock()), \
             patch("cli.save_config_value", return_value=True), \
             patch("hermes_cli.skin_engine.resolve_auto_skin", return_value="default"):
            host._handle_skin_command("/skin auto")

        from hermes_cli.skin_engine import get_active_skin_name
        assert get_active_skin_name() == "default", (
            "/skin auto must reach set_active_skin(), not be rejected as an "
            "unknown skin name"
        )

    def test_real_startup_path_display_skin_auto_with_hermes_light_resolves_daylight(
        self, monkeypatch, tmp_path
    ):
        """Regression (review of #68610): a full, real cli.py module
        (re)import with display.skin: auto configured and HERMES_LIGHT=1
        set must end up on the "daylight" skin -- not silently fall back
        to "default" via resolve_auto_skin()'s own except-swallow.

        This is the exact path the prior tests in this class don't cover:
        they either mock resolve_auto_skin() directly, or check that
        cli._detect_light_mode exists as an attribute on the ALREADY-fully-
        imported module (post test-collection). Neither exercises cli.py's
        own top-level execution ORDER: init_skin_from_config() used to run
        (indirectly, via set_active_skin("auto") -> resolve_auto_skin() ->
        `from cli import _detect_light_mode`) BEFORE cli.py had reached the
        line defining that function, which failed with the classic
        partially-initialized-module ImportError, got silently swallowed,
        and left the active skin on "default" even with HERMES_LIGHT=1 set.
        importlib.reload() re-executes cli.py's module-scope code exactly
        as a fresh process import would, so this is the faithful
        reproduction of the real startup sequence.
        """
        import sys as _sys
        import types as _types
        import importlib
        import textwrap

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_LIGHT", "1")
        monkeypatch.delenv("HERMES_TUI_LIGHT", raising=False)
        monkeypatch.delenv("HERMES_TUI_THEME", raising=False)
        monkeypatch.delenv("HERMES_TUI_BACKGROUND", raising=False)
        monkeypatch.delenv("COLORFGBG", raising=False)
        (tmp_path / "config.yaml").write_text(
            textwrap.dedent(
                """
                display:
                  skin: auto
                """
            ).lstrip()
        )
        # No user-installed auto.yaml, so the virtual selector resolves.
        skins_dir = tmp_path / "skins"
        skins_dir.mkdir(exist_ok=True)

        try:
            import cli as _cli_mod  # noqa: F401 -- import for its side effect / availability check
        except Exception:
            for _mod in ("prompt_toolkit", "dotenv"):
                if _mod not in _sys.modules:
                    stub = _types.ModuleType(_mod)
                    if _mod == "dotenv":
                        stub.load_dotenv = lambda *a, **k: None
                    monkeypatch.setitem(_sys.modules, _mod, stub)
            try:
                import cli as _cli_mod  # noqa: F401
            except Exception:
                import pytest as _pytest
                _pytest.skip("cli module could not be imported in this sandbox")

        # importlib.reload() re-executes cli.py's top-level statements, but
        # reuses the SAME module object in place -- its namespace still has
        # every attribute from the PRIOR full execution (including
        # _detect_light_mode) while the reload is running, so the circular
        # `from cli import _detect_light_mode` inside resolve_auto_skin()
        # would succeed immediately regardless of where the init call sits
        # in this file. That does NOT reproduce the real bug: a fresh
        # process's first import starts with an EMPTY module namespace, so
        # the circular import only succeeds once execution has actually
        # reached the `def _detect_light_mode` line. Deleting cli (and its
        # own submodule cache entries it may have created) from
        # sys.modules before re-importing forces Python to treat this as a
        # genuinely first-time import, faithfully reproducing the ordering
        # bug this test guards against.
        for _mod_name in list(_sys.modules):
            if _mod_name == "cli" or _mod_name.startswith("cli."):
                del _sys.modules[_mod_name]

        import cli as _cli_mod

        from hermes_cli.skin_engine import get_active_skin_name

        assert get_active_skin_name() == "daylight", (
            "display.skin: auto with HERMES_LIGHT=1 must resolve to "
            "'daylight' via the real cli.py startup sequence -- if this is "
            "'default', init_skin_from_config() ran before "
            "cli._detect_light_mode was defined, and the resulting "
            "ImportError was silently swallowed"
        )
