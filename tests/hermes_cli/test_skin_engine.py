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












class TestSkinManagement:
    def test_set_active_skin(self):
        from hermes_cli.skin_engine import set_active_skin, get_active_skin, get_active_skin_name
        skin = set_active_skin("ares")
        assert skin.name == "ares"
        assert get_active_skin_name() == "ares"
        assert get_active_skin().name == "ares"






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
        import hermes_yaml as yaml
        skin_file.write_text(yaml.safe_dump(skin_data))

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
        import hermes_yaml as yaml

        (skins_dir / "broken.yaml").write_text(
            yaml.safe_dump(
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
        import hermes_yaml as yaml
        (skins_dir / "pirate.yaml").write_text(yaml.safe_dump({
            "name": "pirate",
            "description": "Arr matey",
        }))
        monkeypatch.setattr("hermes_cli.skin_engine._skins_dir", lambda: skins_dir)

        skins = list_skins()
        names = [s["name"] for s in skins]
        assert "pirate" in names
        pirate = [s for s in skins if s["name"] == "pirate"][0]
        assert pirate["source"] == "user"


class TestCustomCSS:
    """customCSS passthrough: parsed from user YAML, whitespace-stripped,
    capped at 32 KiB, empty when the field is absent."""

    def _load(self, tmp_path, monkeypatch, **skin_data):
        from hermes_cli.skin_engine import load_skin

        skins_dir = tmp_path / "skins"
        skins_dir.mkdir()
        import hermes_yaml as yaml

        data = {"name": "styled", "colors": {"background": "#101010"}}
        data.update(skin_data)
        (skins_dir / "styled.yaml").write_text(yaml.safe_dump(data))
        monkeypatch.setattr("hermes_cli.skin_engine._skins_dir", lambda: skins_dir)
        return load_skin("styled")

    def test_user_skin_custom_css_passthrough(self, tmp_path, monkeypatch):
        skin = self._load(tmp_path, monkeypatch, customCSS=".chat-input { font-size: 16px; }")

        assert skin.custom_css == ".chat-input { font-size: 16px; }"

    def test_user_skin_custom_css_whitespace_stripped(self, tmp_path, monkeypatch):
        skin = self._load(tmp_path, monkeypatch, customCSS="\n  .status-bar { background: black; }  \n")

        assert skin.custom_css == ".status-bar { background: black; }"

    def test_user_skin_custom_css_empty_when_missing(self, tmp_path, monkeypatch):
        skin = self._load(tmp_path, monkeypatch)

        assert skin.custom_css == ""

    def test_user_skin_custom_css_capped_at_32_ki_b(self, tmp_path, monkeypatch):
        skin = self._load(tmp_path, monkeypatch, customCSS="x" * 40000)

        assert len(skin.custom_css) == 32768

    def test_builtin_skin_has_no_custom_css(self, tmp_path, monkeypatch):
        from hermes_cli.skin_engine import load_skin

        skins_dir = tmp_path / "skins"
        skins_dir.mkdir()
        monkeypatch.setattr("hermes_cli.skin_engine._skins_dir", lambda: skins_dir)

        assert load_skin("default").custom_css == ""
        assert load_skin("mono").custom_css == ""


class TestDisplayIntegration:


    def test_tool_message_uses_skin_prefix(self):
        from hermes_cli.skin_engine import set_active_skin
        from agent.display import get_cute_tool_message
        set_active_skin("ares")
        msg = get_cute_tool_message("terminal", {"command": "ls"}, 0.5)
        assert msg.startswith("╎")
        assert "┊" not in msg


class TestCliBrandingHelpers:



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
        assert overrides["status-bar-session-title"] == (
            f"bg:{skin.get_color('status_bar_strong')} {skin.get_color('status_bar_bg')} bold"
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


class TestSkinAppliesOutsideTheCLI:
    """``display.skin`` must reach every process, not only the interactive CLI.

    ``init_skin_from_config()`` is called from ``cli.py`` and ``tui_gateway/change_watcher.py``
    and nowhere else, and ``get_active_skin()`` lazily resolved the configured skin only for
    routed/multiplex homes — so a gateway or cron process kept ``_active_skin_name = "default"``
    and silently ignored the user's skin (#36040).
    """

    @staticmethod
    def _home_with_skin(monkeypatch, tmp_path, skin):
        """A temp HERMES_HOME whose config.yaml selects ``skin`` (or none when falsy)."""
        import yaml

        home = tmp_path / ".hermes"
        home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        config = {"display": {"skin": skin}} if skin else {"display": {}}
        (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        return home

    def test_configured_skin_applies_without_any_cli_init(self, monkeypatch, tmp_path):
        """No init_skin_from_config() call anywhere: the config still decides."""
        from hermes_cli.skin_engine import get_active_skin, get_active_skin_name

        self._home_with_skin(monkeypatch, tmp_path, "ares")

        assert get_active_skin().name == "ares"
        assert get_active_skin_name() == "ares"

    def test_name_accessor_resolves_on_its_own(self, monkeypatch, tmp_path):
        """get_active_skin_name() alone: the answer must not depend on get_active_skin() running first."""
        from hermes_cli.skin_engine import get_active_skin_name

        self._home_with_skin(monkeypatch, tmp_path, "ares")

        assert get_active_skin_name() == "ares"

    def test_unloadable_skin_keeps_its_configured_name(self, monkeypatch, tmp_path):
        """The pair an initialized CLI reports: the default skin, under the name the user set."""
        from hermes_cli.skin_engine import get_active_skin, get_active_skin_name

        self._home_with_skin(monkeypatch, tmp_path, "lunaobt")

        assert get_active_skin_name() == "lunaobt"
        assert get_active_skin().name == "default"

    def test_user_yaml_skin_applies_without_any_cli_init(self, monkeypatch, tmp_path):
        """The reported symptom: a user skin file, not a built-in, ignored in a cold process."""
        import yaml
        from hermes_cli.skin_engine import get_active_skin

        home = self._home_with_skin(monkeypatch, tmp_path, "lunabot")
        skins = home / "skins"
        skins.mkdir(parents=True, exist_ok=True)
        (skins / "lunabot.yaml").write_text(
            yaml.safe_dump({"name": "lunabot", "description": "user skin",
                            "branding": {"agent_name": "Lunabot"}}),
            encoding="utf-8",
        )

        skin = get_active_skin()
        assert skin.name == "lunabot"
        assert skin.get_branding("agent_name", "") == "Lunabot"

    def test_explicit_choice_wins(self, monkeypatch, tmp_path):
        """The lazy resolve must not override an explicit set_active_skin()."""
        from hermes_cli.skin_engine import get_active_skin, set_active_skin

        self._home_with_skin(monkeypatch, tmp_path, "ares")
        set_active_skin("mono")

        assert get_active_skin().name == "mono"

    def test_unconfigured_stays_default(self, monkeypatch, tmp_path):
        """With nothing configured the resolve must not invent a skin."""
        from hermes_cli.skin_engine import get_active_skin

        self._home_with_skin(monkeypatch, tmp_path, None)

        assert get_active_skin().name == "default"
