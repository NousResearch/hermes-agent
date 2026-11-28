"""Tests for hermes_cli.skin_engine — the data-driven skin/theme system."""

import os
import time

import pytest


@pytest.fixture(autouse=True)
def reset_skin_state():
    """Reset skin engine state between tests."""
    from hermes_cli import skin_engine
    skin_engine._active_skin = None
    skin_engine._active_skin_name = "default"
    getattr(skin_engine, "_skin_cache", {}).clear()
    yield
    skin_engine._active_skin = None
    skin_engine._active_skin_name = "default"
    getattr(skin_engine, "_skin_cache", {}).clear()












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
        pirate = next(s for s in skins if s["name"] == "pirate")
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


class TestSkinParseCache:
    """``load_skin`` sits on every gateway websocket connect's path
    (``change_watcher.resolve_skin`` → ``init_skin_from_config`` → ``set_active_skin``): the file
    must be parsed once and re-read after a real edit."""

    @staticmethod
    def _write_skin(skins_dir, monkeypatch, *, title: str, age_s: int):
        import hermes_yaml as yaml

        path = skins_dir / "cached.yaml"
        path.write_text(yaml.safe_dump({"name": "cached", "colors": {"banner_title": title}}),
                        encoding="utf-8")
        # Backdate by default: a file written this instant is deliberately not cacheable.
        stamp = time.time_ns() - age_s * 10**9
        os.utime(path, ns=(stamp, stamp))
        monkeypatch.setattr("hermes_cli.skin_engine._skins_dir", lambda: skins_dir)
        return path

    def test_repeated_loads_parse_the_file_once(self, tmp_path, monkeypatch):
        from hermes_cli import skin_engine

        skins_dir = tmp_path / "skins"
        skins_dir.mkdir()
        self._write_skin(skins_dir, monkeypatch, title="#FF0000", age_s=60)

        parses = []
        real = skin_engine._load_skin_from_yaml

        def counting(path):
            parses.append(path)
            return real(path)

        monkeypatch.setattr(skin_engine, "_load_skin_from_yaml", counting)

        first = skin_engine.load_skin("cached")
        second = skin_engine.load_skin("cached")
        skin_engine.set_active_skin("cached")
        skin_engine.set_active_skin("cached")

        assert len(parses) == 1, f"parsed the unchanged skin {len(parses)} times"
        assert first is second
        assert second.get_color("banner_title") == "#FF0000"

    def test_an_edited_skin_file_reloads(self, tmp_path, monkeypatch):
        from hermes_cli.skin_engine import load_skin

        skins_dir = tmp_path / "skins"
        skins_dir.mkdir()
        path = self._write_skin(skins_dir, monkeypatch, title="#FF0000", age_s=60)
        assert load_skin("cached").get_color("banner_title") == "#FF0000"

        # New mtime, same size: the mtime alone has to miss.
        self._write_skin(skins_dir, monkeypatch, title="#00FF00", age_s=30)
        assert load_skin("cached").get_color("banner_title") == "#00FF00"

        # Same mtime, larger file: the size alone has to miss.
        stamp = path.stat().st_mtime_ns
        self._write_skin(skins_dir, monkeypatch, title="#0000FF00", age_s=30)
        os.utime(path, ns=(stamp, stamp))
        assert load_skin("cached").get_color("banner_title") == "#0000FF00"

    def test_atomic_replacement_preserving_mtime_and_size_reloads(self, tmp_path, monkeypatch):
        """An atomic replacement must invalidate the cache even with preserved metadata."""
        from hermes_cli import skin_engine
        import hermes_yaml as yaml

        skins_dir = tmp_path / "skins"
        skins_dir.mkdir()
        path = self._write_skin(skins_dir, monkeypatch, title="#112233", age_s=60)

        first = skin_engine.load_skin("cached")
        assert first is skin_engine.load_skin("cached")  # Baseline without caching is RED.
        before = path.stat()

        replacement = skins_dir / "replacement.tmp"
        replacement.write_text(
            yaml.safe_dump({"name": "cached", "colors": {"banner_title": "#445566"}}),
            encoding="utf-8",
        )
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert replacement.stat().st_size == before.st_size
        assert replacement.stat().st_mtime_ns == before.st_mtime_ns

        os.replace(replacement, path)
        assert skin_engine.set_active_skin("cached").get_color("banner_title") == "#445566"

    def test_an_atomic_replace_with_preserved_size_and_mtime_reloads(self, tmp_path, monkeypatch):
        """A temp-file + ``os.replace`` swap keeps size and mtime but changes the inode: a reconnect
        must not serve the previous skin's colours. The inode alone changes here, so a
        ``(path, mtime, size)`` key would hit and hand back ``#112233``."""
        from hermes_cli import skin_engine
        import hermes_yaml as yaml

        skins_dir = tmp_path / "skins"
        skins_dir.mkdir()
        path = self._write_skin(skins_dir, monkeypatch, title="#112233", age_s=60)

        parses = []
        real = skin_engine._load_skin_from_yaml

        def counting(p):
            parses.append(p)
            return real(p)

        monkeypatch.setattr(skin_engine, "_load_skin_from_yaml", counting)

        assert skin_engine.load_skin("cached").get_color("banner_title") == "#112233"
        assert skin_engine.load_skin("cached").get_color("banner_title") == "#112233"
        assert len(parses) == 1, "an unchanged skin must still hit the cache"

        before = path.stat()
        body = yaml.safe_dump({"name": "cached", "colors": {"banner_title": "#445566"}})
        assert len(body.encode()) == before.st_size, "the test needs an equal-length replacement"

        replacement = skins_dir / ".cached.yaml.swp"
        replacement.write_text(body, encoding="utf-8")
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        os.replace(replacement, path)

        after = path.stat()
        assert after.st_size == before.st_size and after.st_mtime_ns == before.st_mtime_ns
        assert after.st_ino != before.st_ino, "os.replace must change the inode for this to be a test"

        assert skin_engine.load_skin("cached").get_color("banner_title") == "#445566"
        assert len(parses) == 2, "the replaced file has to be parsed again"
        assert skin_engine.set_active_skin("cached").get_color("banner_title") == "#445566"

    def test_a_just_written_skin_file_is_never_cached(self, tmp_path, monkeypatch):
        """A same-tick rewrite can land after the stat, so a fresh mtime cannot be a cache key."""
        from hermes_cli.skin_engine import load_skin

        skins_dir = tmp_path / "skins"
        skins_dir.mkdir()
        self._write_skin(skins_dir, monkeypatch, title="#FF0000", age_s=0)
        assert load_skin("cached").get_color("banner_title") == "#FF0000"

        # Same length, same tick: the key would be identical, so this must not come from the cache.
        self._write_skin(skins_dir, monkeypatch, title="#0000FF", age_s=0)
        assert load_skin("cached").get_color("banner_title") == "#0000FF"
