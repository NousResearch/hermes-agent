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

    def test_none_clears_system_prompt(self):
        cli = self._make_cli()
        with patch("cli.save_config_values", return_value=True):
            cli._handle_personality_command("/personality none")
        assert cli.system_prompt == ""

    def test_default_clears_system_prompt(self):
        cli = self._make_cli()
        with patch("cli.save_config_values", return_value=True):
            cli._handle_personality_command("/personality default")
        assert cli.system_prompt == ""

    def test_neutral_clears_system_prompt(self):
        cli = self._make_cli()
        with patch("cli.save_config_values", return_value=True):
            cli._handle_personality_command("/personality neutral")
        assert cli.system_prompt == ""

    def test_none_forces_agent_reinit(self):
        cli = self._make_cli()
        with patch("cli.save_config_values", return_value=True):
            cli._handle_personality_command("/personality none")
        assert cli.agent is None

    def test_none_saves_to_config(self):
        cli = self._make_cli()
        with patch("cli.save_config_values", return_value=True) as mock_save:
            cli._handle_personality_command("/personality none")
        # Saves both agent.system_prompt (the resolved prompt body, empty when
        # clearing) AND display.personality (the canonical name, empty when
        # clearing) in a single atomic call. Without the second key, the TUI
        # status bar and the `/personality` listing would read stale state;
        # without a single call, a partial failure could persist only one of
        # the two keys.
        assert mock_save.call_args_list == [
            (({"agent.system_prompt": "", "display.personality": ""},),),
        ]

    def test_none_reports_session_only_when_save_fails(self, capsys):
        """CR-01/CR-02 regression: if the config write fails, the CLI must
        say so (session only) rather than claim it was saved."""
        cli = self._make_cli()
        with patch("cli.save_config_values", return_value=False):
            cli._handle_personality_command("/personality none")
        output = capsys.readouterr().out
        assert "session only" in output.lower()
        assert "saved to config" not in output.lower()

    def test_known_personality_still_works(self):
        cli = self._make_cli()
        with patch("cli.save_config_values", return_value=True):
            cli._handle_personality_command("/personality helpful")
        assert cli.system_prompt == "You are helpful."

    def test_known_personality_saves_both_keys(self):
        """Setting a named personality persists both the prompt body and the
        canonical name in a single atomic write. Regression: previously only
        agent.system_prompt was written, leaving display.personality stale so
        the status bar and `/personality` listing disagreed with the active
        overlay; and previously two separate writes meant one could succeed
        while the other failed, leaving the pair inconsistent."""
        cli = self._make_cli()
        with patch("cli.save_config_values", return_value=True) as mock_save:
            cli._handle_personality_command("/personality helpful")
        assert mock_save.call_args_list == [
            (({"agent.system_prompt": "You are helpful.", "display.personality": "helpful"},),),
        ]

    def test_known_personality_reports_session_only_when_save_fails(self, capsys):
        cli = self._make_cli()
        with patch("cli.save_config_values", return_value=False):
            cli._handle_personality_command("/personality helpful")
        output = capsys.readouterr().out
        assert "session only" in output.lower()
        assert "saved to config" not in output.lower()

    def test_unknown_personality_shows_none_in_available(self, capsys):
        cli = self._make_cli()
        cli._handle_personality_command("/personality nonexistent")
        output = capsys.readouterr().out
        assert "none" in output.lower()

    def test_list_shows_none_option(self):
        cli = self._make_cli()
        with patch("builtins.print") as mock_print:
            cli._handle_personality_command("/personality")
        output = " ".join(str(c) for c in mock_print.call_args_list)
        assert "none" in output.lower()


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
    async def test_none_clears_ephemeral_prompt(self, tmp_path):
        runner = self._make_runner()
        config_data = {"agent": {"personalities": {"helpful": "You are helpful."}, "system_prompt": "kawaii"}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("gateway.run._hermes_home", tmp_path):
            event = self._make_event("none")
            result = await runner._handle_personality_command(event)

        assert runner._ephemeral_system_prompt == ""
        assert "cleared" in result.lower()

    @pytest.mark.asyncio
    async def test_default_clears_ephemeral_prompt(self, tmp_path):
        runner = self._make_runner()
        config_data = {"agent": {"personalities": {"helpful": "You are helpful."}}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("gateway.run._hermes_home", tmp_path):
            event = self._make_event("default")
            result = await runner._handle_personality_command(event)

        assert runner._ephemeral_system_prompt == ""

    @pytest.mark.asyncio
    async def test_list_includes_none(self, tmp_path):
        runner = self._make_runner()
        config_data = {"agent": {"personalities": {"helpful": "You are helpful."}}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("gateway.run._hermes_home", tmp_path):
            event = self._make_event("")
            result = await runner._handle_personality_command(event)

        assert "none" in result.lower()

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
    async def test_set_named_persists_both_keys_via_save_config_values(self, tmp_path):
        """Gateway must persist BOTH agent.system_prompt and
        display.personality via the comment-preserving save_config_values
        helper, in a single atomic call. Previously it called
        atomic_yaml_write on the whole config dict (clobbering comments) and
        only wrote agent.system_prompt so display.personality stayed stale;
        later it wrote the two keys via two separate save_config_value calls,
        which meant a failure between them could leave the pair
        inconsistent."""
        runner = self._make_runner()
        config_data = {"agent": {"personalities": {"helpful": "You are helpful."}}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("gateway.run._hermes_home", tmp_path), \
             patch("cli.save_config_values", return_value=True) as mock_save:
            event = self._make_event("helpful")
            await runner._handle_personality_command(event)

        assert mock_save.call_args_list == [
            (({"agent.system_prompt": "You are helpful.", "display.personality": "helpful"},),),
        ]
        assert runner._ephemeral_system_prompt == "You are helpful."

    @pytest.mark.asyncio
    async def test_clear_persists_both_keys_via_save_config_values(self, tmp_path):
        """Clearing (`/personality none`) must also clear display.personality,
        via the same single atomic call."""
        runner = self._make_runner()
        config_data = {"agent": {"personalities": {"helpful": "You are helpful."}}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("gateway.run._hermes_home", tmp_path), \
             patch("cli.save_config_values", return_value=True) as mock_save:
            event = self._make_event("none")
            await runner._handle_personality_command(event)

        assert mock_save.call_args_list == [
            (({"agent.system_prompt": "", "display.personality": ""},),),
        ]

    @pytest.mark.asyncio
    async def test_set_named_reports_failure_when_save_fails(self, tmp_path):
        """CR-01 regression: save_config_values() returns False rather than
        raising on failure. A try/except around the call would never fire,
        so the gateway must check the return value and report failure
        instead of silently claiming success."""
        runner = self._make_runner()
        config_data = {"agent": {"personalities": {"helpful": "You are helpful."}}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("gateway.run._hermes_home", tmp_path), \
             patch("cli.save_config_values", return_value=False):
            event = self._make_event("helpful")
            result = await runner._handle_personality_command(event)

        assert "failed" in result.lower() or "error" in result.lower()
        # In-memory state must not silently flip to "set" when the write failed.
        assert runner._ephemeral_system_prompt != "You are helpful."

    @pytest.mark.asyncio
    async def test_clear_reports_failure_when_save_fails(self, tmp_path):
        runner = self._make_runner()
        config_data = {"agent": {"personalities": {"helpful": "You are helpful."}}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("gateway.run._hermes_home", tmp_path), \
             patch("cli.save_config_values", return_value=False):
            event = self._make_event("none")
            result = await runner._handle_personality_command(event)

        assert "failed" in result.lower() or "error" in result.lower()

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

    def test_dict_personality_includes_tone(self):
        cli = self._make_cli({
            "coder": {
                "system_prompt": "You are an expert programmer.",
                "tone": "technical and precise",
            }
        })
        with patch("cli.save_config_value", return_value=True):
            cli._handle_personality_command("/personality coder")
        assert "Tone: technical and precise" in cli.system_prompt

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
