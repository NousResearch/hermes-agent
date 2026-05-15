"""Tests for gateway /set-prompt and /clear-prompt slash commands.

Covers:
- resolve_channel_prompt() runtime override precedence over config.yaml
- resolve_channel_prompt() falls back to config.yaml when no runtime file
- resolve_channel_prompt() silently degrades on corrupted runtime file
- _handle_set_prompt_command writes atomically to channel_prompts.json
- _handle_set_prompt_command rejects empty text (returns usage hint)
- _handle_clear_prompt_command removes entry and returns confirmation
- _handle_clear_prompt_command returns a notice when no runtime prompt exists
- set-prompt and clear-prompt are in GATEWAY_KNOWN_COMMANDS
"""

import json
import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, resolve_channel_prompt
from gateway.session import SessionSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(text: str, chat_id: str = "C0123456") -> MessageEvent:
    """Build a minimal MessageEvent for testing."""
    source = SessionSource(
        platform=Platform.SLACK,
        user_id="U9999",
        chat_id=chat_id,
        user_name="tester",
        chat_type="channel",
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    """Create a bare GatewayRunner without calling __init__."""
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner.session_store = None
    runner.config = None
    return runner


# ---------------------------------------------------------------------------
# resolve_channel_prompt — unit tests
# ---------------------------------------------------------------------------


class TestResolveChannelPrompt:
    """Tests for resolve_channel_prompt() precedence and fallback logic."""

    def test_runtime_file_wins_over_config(self, tmp_path, monkeypatch):
        """A runtime channel_prompts.json entry overrides config.yaml's channel_prompts."""
        import gateway.platforms.base as base_module
        import hermes_constants

        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)

        runtime = tmp_path / "channel_prompts.json"
        runtime.write_text(json.dumps({"C0123456": "runtime prompt"}), encoding="utf-8")

        config_extra = {"channel_prompts": {"C0123456": "config prompt"}}
        result = resolve_channel_prompt(config_extra, "C0123456")
        assert result == "runtime prompt"

    def test_falls_back_to_config_when_no_runtime_file(self, tmp_path, monkeypatch):
        """When channel_prompts.json does not exist, config.yaml value is used."""
        import hermes_constants

        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
        # No runtime file created

        config_extra = {"channel_prompts": {"C0123456": "config prompt"}}
        result = resolve_channel_prompt(config_extra, "C0123456")
        assert result == "config prompt"

    def test_falls_back_to_config_when_channel_not_in_runtime(self, tmp_path, monkeypatch):
        """Runtime file exists but doesn't have the channel — falls back to config."""
        import hermes_constants

        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)

        runtime = tmp_path / "channel_prompts.json"
        runtime.write_text(json.dumps({"C0OTHER": "other prompt"}), encoding="utf-8")

        config_extra = {"channel_prompts": {"C0123456": "config prompt"}}
        result = resolve_channel_prompt(config_extra, "C0123456")
        assert result == "config prompt"

    def test_corrupted_runtime_file_degrades_gracefully(self, tmp_path, monkeypatch):
        """A corrupted runtime JSON file does not crash — falls back to config."""
        import hermes_constants

        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)

        runtime = tmp_path / "channel_prompts.json"
        runtime.write_text("NOT VALID JSON }{", encoding="utf-8")

        config_extra = {"channel_prompts": {"C0123456": "config prompt"}}
        result = resolve_channel_prompt(config_extra, "C0123456")
        assert result == "config prompt"

    def test_returns_none_when_no_prompt_anywhere(self, tmp_path, monkeypatch):
        """Returns None when neither runtime file nor config has a prompt."""
        import hermes_constants

        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)

        result = resolve_channel_prompt({}, "C0123456")
        assert result is None

    def test_parent_id_fallback_in_runtime(self, tmp_path, monkeypatch):
        """Runtime file falls back to parent_id when channel_id isn't present."""
        import hermes_constants

        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)

        runtime = tmp_path / "channel_prompts.json"
        runtime.write_text(json.dumps({"PARENT_CHAN": "parent prompt"}), encoding="utf-8")

        result = resolve_channel_prompt({}, "CHILD_THREAD", parent_id="PARENT_CHAN")
        assert result == "parent prompt"

    def test_blank_runtime_entry_treated_as_absent(self, tmp_path, monkeypatch):
        """Blank/whitespace-only runtime entry is skipped (treated as absent)."""
        import hermes_constants

        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)

        runtime = tmp_path / "channel_prompts.json"
        runtime.write_text(json.dumps({"C0123456": "   "}), encoding="utf-8")

        config_extra = {"channel_prompts": {"C0123456": "config prompt"}}
        result = resolve_channel_prompt(config_extra, "C0123456")
        assert result == "config prompt"


# ---------------------------------------------------------------------------
# _handle_set_prompt_command
# ---------------------------------------------------------------------------


class TestHandleSetPromptCommand:
    """Tests for GatewayRunner._handle_set_prompt_command."""

    @pytest.mark.asyncio
    async def test_creates_json_file_on_first_use(self, tmp_path, monkeypatch):
        """When channel_prompts.json doesn't exist yet, it is created."""
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        runner = _make_runner()

        event = _make_event("/set-prompt You are a coding assistant.", chat_id="C0123456")
        result = await runner._handle_set_prompt_command(event)

        assert "set" in result.lower()
        written = json.loads((tmp_path / "channel_prompts.json").read_text())
        assert written["C0123456"] == "You are a coding assistant."

    @pytest.mark.asyncio
    async def test_updates_existing_entry(self, tmp_path, monkeypatch):
        """Calling /set-prompt again updates the existing entry."""
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        runner = _make_runner()

        # Pre-populate with an existing entry for a different channel
        existing = {"C0OTHER": "other prompt"}
        (tmp_path / "channel_prompts.json").write_text(json.dumps(existing))

        event = _make_event("/set-prompt New prompt text.", chat_id="C0123456")
        await runner._handle_set_prompt_command(event)

        written = json.loads((tmp_path / "channel_prompts.json").read_text())
        assert written["C0123456"] == "New prompt text."
        # Other channel must be preserved
        assert written["C0OTHER"] == "other prompt"

    @pytest.mark.asyncio
    async def test_empty_args_returns_usage_hint(self, tmp_path, monkeypatch):
        """When called with no text, returns a usage hint without writing the file."""
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        runner = _make_runner()

        event = _make_event("/set-prompt")
        result = await runner._handle_set_prompt_command(event)

        assert "usage" in result.lower() or "Usage" in result
        assert not (tmp_path / "channel_prompts.json").exists()

    @pytest.mark.asyncio
    async def test_overwrites_previous_entry_for_same_channel(self, tmp_path, monkeypatch):
        """Calling /set-prompt twice on the same channel replaces the first value."""
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        runner = _make_runner()

        event1 = _make_event("/set-prompt First prompt.", chat_id="C0123456")
        await runner._handle_set_prompt_command(event1)

        event2 = _make_event("/set-prompt Second prompt.", chat_id="C0123456")
        await runner._handle_set_prompt_command(event2)

        written = json.loads((tmp_path / "channel_prompts.json").read_text())
        assert written["C0123456"] == "Second prompt."

    @pytest.mark.asyncio
    async def test_preserves_other_channels_when_updating(self, tmp_path, monkeypatch):
        """Setting a prompt for one channel does not touch other channels."""
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        runner = _make_runner()

        event_a = _make_event("/set-prompt Prompt A.", chat_id="CHAN_A")
        await runner._handle_set_prompt_command(event_a)

        event_b = _make_event("/set-prompt Prompt B.", chat_id="CHAN_B")
        await runner._handle_set_prompt_command(event_b)

        written = json.loads((tmp_path / "channel_prompts.json").read_text())
        assert written["CHAN_A"] == "Prompt A."
        assert written["CHAN_B"] == "Prompt B."


# ---------------------------------------------------------------------------
# _handle_clear_prompt_command
# ---------------------------------------------------------------------------


class TestHandleClearPromptCommand:
    """Tests for GatewayRunner._handle_clear_prompt_command."""

    @pytest.mark.asyncio
    async def test_removes_existing_entry(self, tmp_path, monkeypatch):
        """Removing an existing entry clears it from the JSON file."""
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        runner = _make_runner()

        (tmp_path / "channel_prompts.json").write_text(
            json.dumps({"C0123456": "old prompt", "C0OTHER": "other"})
        )

        event = _make_event("/clear-prompt", chat_id="C0123456")
        result = await runner._handle_clear_prompt_command(event)

        assert "cleared" in result.lower()
        written = json.loads((tmp_path / "channel_prompts.json").read_text())
        assert "C0123456" not in written
        assert written["C0OTHER"] == "other"  # Other channel preserved

    @pytest.mark.asyncio
    async def test_no_file_returns_notice(self, tmp_path, monkeypatch):
        """When channel_prompts.json doesn't exist, returns a helpful notice."""
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        runner = _make_runner()

        event = _make_event("/clear-prompt", chat_id="C0123456")
        result = await runner._handle_clear_prompt_command(event)

        assert "no runtime" in result.lower() or "not set" in result.lower()

    @pytest.mark.asyncio
    async def test_channel_not_in_file_returns_notice(self, tmp_path, monkeypatch):
        """When the channel has no runtime prompt, returns a helpful notice."""
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        runner = _make_runner()

        (tmp_path / "channel_prompts.json").write_text(
            json.dumps({"C0OTHER": "other prompt"})
        )

        event = _make_event("/clear-prompt", chat_id="C0123456")
        result = await runner._handle_clear_prompt_command(event)

        assert "not set" in result.lower() or "no runtime" in result.lower()

    @pytest.mark.asyncio
    async def test_clears_then_set_prompt_restores(self, tmp_path, monkeypatch):
        """After /clear-prompt, /set-prompt can set a new prompt."""
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        runner = _make_runner()

        set_event = _make_event("/set-prompt Initial prompt.", chat_id="C0123456")
        await runner._handle_set_prompt_command(set_event)

        clear_event = _make_event("/clear-prompt", chat_id="C0123456")
        await runner._handle_clear_prompt_command(clear_event)

        set_event2 = _make_event("/set-prompt Restored prompt.", chat_id="C0123456")
        await runner._handle_set_prompt_command(set_event2)

        written = json.loads((tmp_path / "channel_prompts.json").read_text())
        assert written["C0123456"] == "Restored prompt."


# ---------------------------------------------------------------------------
# Command registry sanity checks
# ---------------------------------------------------------------------------


class TestCommandRegistry:
    """Verify the new commands are properly registered."""

    def test_set_prompt_in_gateway_known_commands(self):
        from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS
        assert "set-prompt" in GATEWAY_KNOWN_COMMANDS

    def test_clear_prompt_in_gateway_known_commands(self):
        from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS
        assert "clear-prompt" in GATEWAY_KNOWN_COMMANDS

    def test_set_prompt_aliases_in_gateway_known_commands(self):
        from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS
        assert "setprompt" in GATEWAY_KNOWN_COMMANDS

    def test_clear_prompt_aliases_in_gateway_known_commands(self):
        from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS
        assert "clearprompt" in GATEWAY_KNOWN_COMMANDS

    def test_set_prompt_is_gateway_only(self):
        from hermes_cli.commands import resolve_command
        cmd = resolve_command("set-prompt")
        assert cmd is not None
        assert cmd.gateway_only is True
        assert cmd.cli_only is False

    def test_clear_prompt_is_gateway_only(self):
        from hermes_cli.commands import resolve_command
        cmd = resolve_command("clear-prompt")
        assert cmd is not None
        assert cmd.gateway_only is True
        assert cmd.cli_only is False

    def test_set_prompt_has_args_hint(self):
        from hermes_cli.commands import resolve_command
        cmd = resolve_command("set-prompt")
        assert cmd is not None
        assert "<text>" in cmd.args_hint

    def test_set_prompt_resolve_by_alias(self):
        from hermes_cli.commands import resolve_command
        cmd = resolve_command("setprompt")
        assert cmd is not None
        assert cmd.name == "set-prompt"

    def test_clear_prompt_resolve_by_alias(self):
        from hermes_cli.commands import resolve_command
        cmd = resolve_command("clearprompt")
        assert cmd is not None
        assert cmd.name == "clear-prompt"
