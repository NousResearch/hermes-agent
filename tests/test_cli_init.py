"""Tests for HermesCLI initialization -- catches configuration bugs
that only manifest at runtime (not in mocked unit tests)."""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_cli(env_overrides=None, config_overrides=None, **kwargs):
    """Create a HermesCLI instance with minimal mocking."""
    import importlib

    _clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": "all"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }
    if config_overrides:
        _clean_config.update(config_overrides)
    clean_env = {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}
    if env_overrides:
        clean_env.update(env_overrides)
    prompt_toolkit_stubs = {
        "prompt_toolkit": MagicMock(),
        "prompt_toolkit.history": MagicMock(),
        "prompt_toolkit.styles": MagicMock(),
        "prompt_toolkit.patch_stdout": MagicMock(),
        "prompt_toolkit.application": MagicMock(),
        "prompt_toolkit.layout": MagicMock(),
        "prompt_toolkit.layout.processors": MagicMock(),
        "prompt_toolkit.filters": MagicMock(),
        "prompt_toolkit.layout.dimension": MagicMock(),
        "prompt_toolkit.layout.menus": MagicMock(),
        "prompt_toolkit.widgets": MagicMock(),
        "prompt_toolkit.key_binding": MagicMock(),
        "prompt_toolkit.completion": MagicMock(),
        "prompt_toolkit.formatted_text": MagicMock(),
    }
    with patch.dict(sys.modules, prompt_toolkit_stubs), \
         patch.dict("os.environ", clean_env, clear=False):
        import cli as _cli_mod
        _cli_mod = importlib.reload(_cli_mod)
        with patch.object(_cli_mod, "get_tool_definitions", return_value=[]), \
             patch.dict(_cli_mod.__dict__, {"CLI_CONFIG": _clean_config}):
            return _cli_mod.HermesCLI(**kwargs)


class TestMaxTurnsResolution:
    """max_turns must always resolve to a positive integer, never None."""

    def test_default_max_turns_is_integer(self):
        cli = _make_cli()
        assert isinstance(cli.max_turns, int)
        assert cli.max_turns == 90

    def test_explicit_max_turns_honored(self):
        cli = _make_cli(max_turns=25)
        assert cli.max_turns == 25

    def test_none_max_turns_gets_default(self):
        cli = _make_cli(max_turns=None)
        assert isinstance(cli.max_turns, int)
        assert cli.max_turns == 90

    def test_env_var_max_turns(self):
        """Env var is used when config file doesn't set max_turns."""
        cli_obj = _make_cli(env_overrides={"HERMES_MAX_ITERATIONS": "42"})
        assert cli_obj.max_turns == 42

    def test_legacy_root_max_turns_is_used_when_agent_key_exists_without_value(self):
        cli_obj = _make_cli(config_overrides={"agent": {}, "max_turns": 77})
        assert cli_obj.max_turns == 77

    def test_max_turns_never_none_for_agent(self):
        """The value passed to AIAgent must never be None (causes TypeError in run_conversation)."""
        cli = _make_cli()
        assert isinstance(cli.max_turns, int) and cli.max_turns == 90


class TestVerboseAndToolProgress:
    def test_default_verbose_is_bool(self):
        cli = _make_cli()
        assert isinstance(cli.verbose, bool)

    def test_tool_progress_mode_is_string(self):
        cli = _make_cli()
        assert isinstance(cli.tool_progress_mode, str)
        assert cli.tool_progress_mode in ("off", "new", "all", "verbose")


class TestHistoryDisplay:
    def test_history_numbers_only_visible_messages_and_summarizes_tools(self, capsys):
        cli = _make_cli()
        cli.conversation_history = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1"}, {"id": "call_2"}],
            },
            {"role": "tool", "content": "tool output 1"},
            {"role": "tool", "content": "tool output 2"},
            {"role": "assistant", "content": "All set."},
            {"role": "user", "content": "A" * 250},
        ]

        cli.show_history()
        output = capsys.readouterr().out

        assert "[You #1]" in output
        assert "[Hermes #2]" in output
        assert "(requested 2 tool calls)" in output
        assert "[Tools]" in output
        assert "(2 tool messages hidden)" in output
        assert "[Hermes #3]" in output
        assert "[You #4]" in output
        assert "[You #5]" not in output
        assert "A" * 250 in output
        assert "A" * 250 + "..." not in output


class TestProviderResolution:
    def test_api_key_is_string_or_none(self):
        cli = _make_cli()
        assert cli.api_key is None or isinstance(cli.api_key, str)

    def test_base_url_is_string(self):
        cli = _make_cli()
        assert isinstance(cli.base_url, str)
        assert cli.base_url.startswith("http")

    def test_model_is_string(self):
        cli = _make_cli()
        assert isinstance(cli.model, str)
        assert isinstance(cli.model, str) and '/' in cli.model
