"""Measured-work pins for the slash-completer config reads.

The /tools and /personality completers run on every keystroke while the
user types those commands (complete_while_typing). They used to re-read +
re-parse the full config on every keypress: load_config()'s defensive
deepcopy (~345us tax) in _tools_completions, and load_cli_config()'s full
YAML parse + defaults deep-merge (~110us) in _personality_completions.
These pins hold the per-keystroke cost down:
- _tools_completions uses the read-only loader (no deepcopy).
- _personality_completions memoises the personalities source keyed on the
  config file's mtime, so the parse+merge runs once per config state.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

import hermes_cli.commands_completion as commands_mod


def _reset_personalities_memo():
    commands_mod._personalities_memo = None


@pytest.fixture(autouse=True)
def _reset_memo():
    _reset_personalities_memo()
    yield
    _reset_personalities_memo()


class TestToolsCompletionsReadonlyConfig:
    def test_uses_readonly_loader(self):
        """_tools_completions must not pay the defensive deepcopy.

        The completer only reads the config (toolset enable state + MCP
        server names). Using load_config_readonly() skips the ~345us
        deepcopy that load_config() applies on every cache hit — a
        per-keystroke cost while completing /tools enable|disable.
        """
        calls = {"deepcopy": 0, "readonly": 0}

        def counting_deepcopy(*a, **k):
            calls["deepcopy"] += 1
            return {}

        def counting_readonly(*a, **k):
            calls["readonly"] += 1
            return {}

        # The completer imports the loader inside the function, so patch the
        # source module. Portable-MCP lookup is stubbed because it triggers
        # one-time plugin discovery (which legitimately calls load_config
        # during process init) — this test asserts on the completer's own
        # per-keystroke reads, not discovery's one-off startup reads.
        with patch("hermes_cli.config.load_config", counting_deepcopy), \
             patch("hermes_cli.config.load_config_readonly", counting_readonly), \
             patch("hermes_cli.plugins.get_portable_mcp_server_names_nowait", lambda: set()), \
             patch("hermes_cli.tools_config._get_plugin_toolset_keys", lambda: set()), \
             patch("hermes_cli.tools_config._homeassistant_credentials_present", lambda: False), \
             patch("hermes_cli.tools_config._xai_credentials_present", lambda: False):
            list(commands_mod.SlashCommandCompleter._tools_completions("enable ", "enable "))

        assert calls["readonly"] == 1, "completer should use the readonly loader"
        assert calls["deepcopy"] == 0, (
            "completer must not call the deepcopy loader on a read-only path"
        )


class TestPersonalityCompletionsMemo:
    def test_load_cli_config_called_once_per_config_state(self, monkeypatch):
        """The /personality completer parses the config once per state.

        load_cli_config() does a full YAML parse + deep merge of the
        built-in defaults; the completer runs on every keystroke. The
        mtime-keyed memo keeps that parse to once per config change.
        """
        calls = {"n": 0}

        def counting_load_cli_config():
            calls["n"] += 1
            return {
                "agent": {
                    "personalities": {
                        "helpful": "You are helpful.",
                        "concise": "You are concise.",
                    }
                }
            }

        monkeypatch.setattr(commands_mod, "_personalities_memo", None)
        with patch("cli.load_cli_config", counting_load_cli_config):
            # First call: cache miss -> one parse.
            list(commands_mod.SlashCommandCompleter._personality_completions("hel", "hel"))
            assert calls["n"] == 1, "first call should parse once"

            # Subsequent keystrokes: cache hit -> no re-parse.
            for _ in range(10):
                list(commands_mod.SlashCommandCompleter._personality_completions("hel", "hel"))
            assert calls["n"] == 1, (
                "repeated keystrokes must reuse the memoised personalities, "
                f"got {calls['n']} parses"
            )

    def test_mtime_change_reparses(self, monkeypatch, tmp_path):
        """A config file change on disk invalidates the memo."""
        from pathlib import Path

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "agent:\n  personalities:\n    helpful: \"v1\"\n",
            encoding="utf-8",
        )
        # Pin an explicit mtime so the change below is a guaranteed mtime_ns
        # bump regardless of filesystem timestamp granularity.
        os.utime(cfg_path, (1_700_000_000, 1_700_000_000))

        calls = {"n": 0}

        def counting_load_cli_config():
            calls["n"] += 1
            return {"agent": {"personalities": {"helpful": "v1"}}}

        def fake_config_path():
            return cfg_path

        monkeypatch.setattr(commands_mod, "_personalities_memo", None)
        with patch("cli.load_cli_config", counting_load_cli_config), \
             patch("hermes_cli.config.get_config_path", fake_config_path):
            list(commands_mod.SlashCommandCompleter._personality_completions("h", "h"))
            assert calls["n"] == 1

            # Bump the file mtime -> memo invalidates -> re-parse once.
            os.utime(cfg_path, (1_800_000_000, 1_800_000_000))
            list(commands_mod.SlashCommandCompleter._personality_completions("h", "h"))
            assert calls["n"] == 2, "config mtime change should re-parse once"
