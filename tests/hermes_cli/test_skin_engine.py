"""Tests for hermes_cli.skin_engine — the data-driven skin/theme system."""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch


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


class TestUserSkins:
    def test_load_user_skin_from_yaml(self, tmp_path, monkeypatch):
        from hermes_cli.skin_engine import load_skin, _skins_dir
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
        assert get_active_prompt_symbol() == "⚔ ❯ "

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
            "input-rule",
            "image-badge",
            "completion-menu",
            "completion-menu.completion",
            "completion-menu.completion.current",
            "completion-menu.meta.completion",
            "completion-menu.meta.completion.current",
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
        assert overrides["clarify-title"] == f"{skin.get_color('banner_title')} bold"
        assert overrides["sudo-prompt"] == f"{skin.get_color('ui_error')} bold"
        assert overrides["approval-title"] == f"{skin.get_color('ui_warn')} bold"
