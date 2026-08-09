"""Tests for /personality none — clearing personality overlay."""
import pytest
from unittest.mock import MagicMock, patch
import yaml


# ── CLI tests ──────────────────────────────────────────────────────────────

class TestCLIPersonalityNone:

    def _make_cli(self, personalities=None):
        from cli import HermesCLI
        cli = HermesCLI.__new__(HermesCLI)
        cli.personalities = personalities or {
            "helpful": "You are helpful.",
            "concise": "You are concise.",
        }
        cli.system_prompt = "You are kawaii~"
        cli.agent = MagicMock()
        cli.console = MagicMock()
        return cli



    def test_set_persists_display_personality_not_system_prompt(self):
        cli = self._make_cli()
        saves = []

        def _save(key, value):
            saves.append((key, value))
            return True

        with patch("cli.save_config_value", side_effect=_save):
            cli._handle_personality_command("/personality helpful")

        assert cli.system_prompt == "You are helpful."
        assert ("display.personality", "helpful") in saves
        assert not any(k == "agent.system_prompt" for k, _ in saves)

    def test_neutral_restores_manual_system_prompt_without_wiping_config(self):
        cli = self._make_cli()
        saves = []

        def _save(key, value):
            saves.append((key, value))
            return True

        with (
            patch("cli.save_config_value", side_effect=_save),
            patch(
                "hermes_cli.config.read_raw_config",
                return_value={"agent": {"system_prompt": "manual forever"}},
            ),
        ):
            cli._handle_personality_command("/personality neutral")

        assert cli.system_prompt == "manual forever"
        assert ("display.personality", "") in saves
        assert not any(k == "agent.system_prompt" for k, _ in saves)







# ── Gateway tests ──────────────────────────────────────────────────────────

class TestGatewayPersonalityNone:

    def _make_event(self, args=""):
        event = MagicMock()
        event.get_command.return_value = "personality"
        event.get_command_args.return_value = args
        return event

    def _make_runner(self, personalities=None):
        from gateway.run import GatewayRunner
        runner = GatewayRunner.__new__(GatewayRunner)
        runner._ephemeral_system_prompt = "You are kawaii~"
        runner.config = {
            "agent": {
                "personalities": personalities or {"helpful": "You are helpful."}
            }
        }
        return runner


    @pytest.mark.asyncio
    async def test_default_clears_ephemeral_prompt(self, tmp_path):
        runner = self._make_runner()
        config_data = {
            "agent": {
                "system_prompt": "manual forever",
                "personalities": {"helpful": "You are helpful."},
            },
            "display": {"personality": "helpful"},
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("gateway.run._hermes_home", tmp_path):
            event = self._make_event("default")
            result = await runner._handle_personality_command(event)

        saved = yaml.safe_load(config_file.read_text())
        assert saved["agent"]["system_prompt"] == "manual forever"
        assert saved.get("display", {}).get("personality", None) == ""
        assert runner._ephemeral_system_prompt == "manual forever"

    @pytest.mark.asyncio
    async def test_set_persists_display_personality_not_system_prompt(self, tmp_path):
        runner = self._make_runner()
        config_data = {
            "agent": {
                "system_prompt": "manual forever",
                "personalities": {"helpful": "You are helpful."},
            }
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("gateway.run._hermes_home", tmp_path):
            event = self._make_event("helpful")
            result = await runner._handle_personality_command(event)

        saved = yaml.safe_load(config_file.read_text())
        assert saved["agent"]["system_prompt"] == "manual forever"
        assert saved["display"]["personality"] == "helpful"
        assert runner._ephemeral_system_prompt == "You are helpful."
        assert "helpful" in result.lower()


    @pytest.mark.asyncio
    async def test_unknown_shows_none_in_available(self, tmp_path):
        runner = self._make_runner()
        config_data = {"agent": {"personalities": {"helpful": "You are helpful."}}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("gateway.run._hermes_home", tmp_path):
            event = self._make_event("nonexistent")
            result = await runner._handle_personality_command(event)

        assert "none" in result.lower()

    @pytest.mark.asyncio
    async def test_empty_personality_list_uses_profile_display_path(self, tmp_path):
        runner = self._make_runner(personalities={})
        (tmp_path / "config.yaml").write_text(yaml.dump({"agent": {"personalities": {}}}))

        with patch("gateway.run._hermes_home", tmp_path), \
             patch("hermes_constants.display_hermes_home", return_value="~/.hermes/profiles/coder"):
            event = self._make_event("")
            result = await runner._handle_personality_command(event)

        assert result == "No personalities configured in `~/.hermes/profiles/coder/config.yaml`"


class TestPersonalityDictFormat:
    """Test dict-format custom personalities with description, tone, style."""

    def _make_cli(self, personalities):
        from cli import HermesCLI
        cli = HermesCLI.__new__(HermesCLI)
        cli.personalities = personalities
        cli.system_prompt = ""
        cli.agent = None
        cli.console = MagicMock()
        return cli

    def test_dict_personality_uses_system_prompt(self):
        cli = self._make_cli({
            "coder": {
                "description": "Expert programmer",
                "system_prompt": "You are an expert programmer.",
                "tone": "technical",
                "style": "concise",
            }
        })
        with patch("cli.save_config_value", return_value=True):
            cli._handle_personality_command("/personality coder")
        assert "You are an expert programmer." in cli.system_prompt


    def test_dict_personality_includes_style(self):
        cli = self._make_cli({
            "coder": {
                "system_prompt": "You are an expert programmer.",
                "style": "use code examples",
            }
        })
        with patch("cli.save_config_value", return_value=True):
            cli._handle_personality_command("/personality coder")
        assert "Style: use code examples" in cli.system_prompt

    def test_string_personality_still_works(self):
        cli = self._make_cli({"helper": "You are helpful."})
        with patch("cli.save_config_value", return_value=True):
            cli._handle_personality_command("/personality helper")
        assert cli.system_prompt == "You are helpful."

    def test_resolve_prompt_dict_no_tone_no_style(self):
        from cli import HermesCLI
        result = HermesCLI._resolve_personality_prompt({
            "description": "A helper",
            "system_prompt": "You are helpful.",
        })
        assert result == "You are helpful."

    def test_resolve_prompt_string(self):
        from cli import HermesCLI
        result = HermesCLI._resolve_personality_prompt("You are helpful.")
        assert result == "You are helpful."
