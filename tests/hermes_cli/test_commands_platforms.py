"""Focused tests for platform-specific command derivation."""

from agent import i18n
from hermes_cli import commands_platforms as platforms


class TestTelegramLocalizedDescriptions:
    def setup_method(self):
        i18n.reset_language_cache()

    def teardown_method(self):
        i18n.reset_language_cache()

    def test_russian_menu_uses_canonical_name_before_sanitizing(self, monkeypatch):
        monkeypatch.setenv("HERMES_LANGUAGE", "ru")
        descriptions = dict(platforms.telegram_bot_commands(include_plugins=False))
        assert descriptions["help"] == "Помощь по командам"
        assert descriptions["new"] == "Новая сессия"
        assert descriptions["reload_mcp"] == "Перезагрузить MCP-серверы"

    def test_untranslated_menu_tracks_registry(self, monkeypatch):
        monkeypatch.setenv("HERMES_LANGUAGE", "ja")
        descriptions = dict(platforms.telegram_bot_commands(include_plugins=False))
        for cmd in platforms._gateway_available_commands():
            telegram_name = platforms._sanitize_telegram_name(cmd.name)
            assert descriptions[telegram_name] == cmd.description


class TestTelegramDescriptionClamp:
    def test_short_and_boundary_descriptions_are_untouched(self):
        assert platforms._clamp_telegram_description("help", "Short") == "Short"
        exact = "x" * platforms._TELEGRAM_MAX_DESCRIPTION_CHARS
        assert platforms._clamp_telegram_description("help", exact) == exact

    def test_oversized_description_is_truncated(self):
        clamped = platforms._clamp_telegram_description("help", "x" * 400)
        assert len(clamped) == platforms._TELEGRAM_MAX_DESCRIPTION_CHARS
        assert clamped.endswith("…")

    def test_plugin_descriptions_are_clamped(self, monkeypatch):
        monkeypatch.setattr(
            platforms,
            "_iter_plugin_command_entries",
            lambda: [("noisy-plugin", "y" * 500, "")],
        )
        descriptions = dict(platforms.telegram_bot_commands())
        assert len(descriptions["noisy_plugin"]) == platforms._TELEGRAM_MAX_DESCRIPTION_CHARS


class TestTelegramMenuPriority:
    def test_localized_core_descriptions_keep_existing_priority(self, monkeypatch):
        monkeypatch.setenv("HERMES_LANGUAGE", "ru")
        monkeypatch.setattr(
            platforms,
            "_telegram_command_menu_config",
            lambda: {"max_commands": 2, "priority_mode": "replace", "priority": ["reload-mcp"]},
        )
        monkeypatch.setattr(platforms, "_collect_gateway_skill_entries", lambda **kwargs: ([], 0))

        menu, hidden = platforms.telegram_menu_commands(max_commands=2)

        assert menu[0] == ("reload_mcp", "Перезагрузить MCP-серверы")
        assert hidden == len(platforms.telegram_bot_commands(include_plugins=False)) - 2
