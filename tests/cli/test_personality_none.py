"""Tests for /personality none — clearing personality overlay.

Updated for the single-owner unification (hermes_cli.personality): built-ins
always exist, resolution reads config (agent.personalities overlays), and
persistence flows exclusively through persist_personality().
"""
import os
import pytest
from unittest.mock import MagicMock, patch
import yaml


# ── CLI tests ──────────────────────────────────────────────────────────────

class TestCLIPersonalityNone:

    def _make_cli(self, personalities=None):
        from cli import HermesCLI
        from hermes_cli.personality import available_personalities

        cli = HermesCLI.__new__(HermesCLI)
        user = personalities or {
            "helpful": "You are helpful.",
            "concise": "You are concise.",
        }
        cli.config = {"agent": {"personalities": user}}
        cli.personalities = available_personalities(cli.config)
        cli.system_prompt = "You are kawaii~"
        cli.agent = MagicMock()
        cli.console = MagicMock()
        return cli

    def test_set_persists_display_personality_not_system_prompt(self):
        cli = self._make_cli()
        saves = []

        def _persist(name):
            saves.append(("display.personality", name))
            return True

        with patch("hermes_cli.personality.persist_personality", side_effect=_persist):
            cli._handle_personality_command("/personality helpful")

        assert cli.system_prompt == "You are helpful."
        assert ("display.personality", "helpful") in saves
        assert not any(k == "agent.system_prompt" for k, _ in saves)

    def test_neutral_restores_manual_system_prompt_without_wiping_config(self):
        cli = self._make_cli()
        saves = []

        def _persist(name):
            saves.append(("display.personality", name))
            return True

        with (
            patch("hermes_cli.personality.persist_personality", side_effect=_persist),
            patch(
                "hermes_cli.config.read_raw_config",
                return_value={"agent": {"system_prompt": "manual forever"}},
            ),
        ):
            cli._handle_personality_command("/personality neutral")

        assert cli.system_prompt == "manual forever"
        assert ("display.personality", "") in saves
        assert not any(k == "agent.system_prompt" for k, _ in saves)

    def test_builtin_personality_works_without_config_entry(self):
        # Built-ins come from hermes_cli.personality, not from config.
        cli = self._make_cli(personalities={})
        with patch("hermes_cli.personality.persist_personality", return_value=True):
            cli._handle_personality_command("/personality kawaii")
        assert "kawaii" in cli.system_prompt.lower()


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
        runner.config = {
            "agent": {
                "personalities": personalities or {"helpful": "You are helpful."}
            }
        }
        return runner

    def _gateway_env(self, tmp_path):
        # The gateway reads via _load_gateway_config (rooted at
        # gateway.run._hermes_home) and persists via persist_personality
        # (rooted at HERMES_HOME) — point both at the same tmp dir.
        return (
            patch("gateway.run._hermes_home", tmp_path),
            patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}),
        )

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

        p1, p2 = self._gateway_env(tmp_path)
        with p1, p2:
            event = self._make_event("default")
            result = await runner._handle_personality_command(event)

        saved = yaml.safe_load(config_file.read_text())
        assert saved["agent"]["system_prompt"] == "manual forever"
        assert saved.get("display", {}).get("personality", None) == ""
        # The next turn re-resolves from config (no in-memory snapshot).
        with p1, p2:
            assert runner._get_system_prompt_for_channel(None, "c") == "manual forever"

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

        p1, p2 = self._gateway_env(tmp_path)
        with p1, p2:
            event = self._make_event("helpful")
            result = await runner._handle_personality_command(event)

        saved = yaml.safe_load(config_file.read_text())
        assert saved["agent"]["system_prompt"] == "manual forever"
        assert saved["display"]["personality"] == "helpful"
        with p1, p2:
            assert runner._get_system_prompt_for_channel(None, "c") == "You are helpful."
        assert "helpful" in result.lower()

    @pytest.mark.asyncio
    async def test_unknown_shows_none_in_available(self, tmp_path):
        runner = self._make_runner()
        config_data = {"agent": {"personalities": {"helpful": "You are helpful."}}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        p1, p2 = self._gateway_env(tmp_path)
        with p1, p2:
            event = self._make_event("nonexistent")
            result = await runner._handle_personality_command(event)

        assert "none" in result.lower()

    @pytest.mark.asyncio
    async def test_empty_personality_list_still_lists_builtins(self, tmp_path):
        # Built-ins are always available — an empty agent.personalities no
        # longer means "no personalities configured".
        runner = self._make_runner(personalities={})
        (tmp_path / "config.yaml").write_text(yaml.dump({"agent": {"personalities": {}}}))

        p1, p2 = self._gateway_env(tmp_path)
        with p1, p2:
            event = self._make_event("")
            result = await runner._handle_personality_command(event)

        assert "kawaii" in result.lower()
        assert "pirate" in result.lower()


class TestPersonalityDictFormat:
    """Test dict-format custom personalities with description, tone, style."""

    def _make_cli(self, personalities):
        from cli import HermesCLI
        from hermes_cli.personality import available_personalities

        cli = HermesCLI.__new__(HermesCLI)
        cli.config = {"agent": {"personalities": personalities}}
        cli.personalities = available_personalities(cli.config)
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
        with patch("hermes_cli.personality.persist_personality", return_value=True):
            cli._handle_personality_command("/personality coder")
        assert "You are an expert programmer." in cli.system_prompt

    def test_dict_personality_includes_style(self):
        cli = self._make_cli({
            "coder": {
                "system_prompt": "You are an expert programmer.",
                "style": "use code examples",
            }
        })
        with patch("hermes_cli.personality.persist_personality", return_value=True):
            cli._handle_personality_command("/personality coder")
        assert "Style: use code examples" in cli.system_prompt

    def test_string_personality_still_works(self):
        cli = self._make_cli({"helper": "You are helpful."})
        with patch("hermes_cli.personality.persist_personality", return_value=True):
            cli._handle_personality_command("/personality helper")
        assert cli.system_prompt == "You are helpful."
