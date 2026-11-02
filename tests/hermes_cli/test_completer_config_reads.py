"""The /personality completer memoises the parsed config per keystroke, but a
config edit on disk must show up in the next completion (never a stale list)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import hermes_yaml as yaml

import hermes_cli.commands_completion as commands_mod


@pytest.fixture(autouse=True)
def _reset_memo():
    commands_mod._personalities_memo = None
    yield
    commands_mod._personalities_memo = None


class TestPersonalityCompletionsMemo:
    def test_config_edit_on_disk_refreshes_completions(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("agent:\n  personalities:\n    zzfirst: v1\n", encoding="utf-8")
        # Pin mtimes so the edit below is a guaranteed signature change
        # regardless of filesystem timestamp granularity.
        os.utime(cfg_path, (1_700_000_000, 1_700_000_000))

        def load_cli_config_from_disk():
            return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

        def names():
            return {c.text for c in commands_mod.SlashCommandCompleter._personality_completions("zz", "zz")}

        with patch("cli.load_cli_config", load_cli_config_from_disk), \
             patch("hermes_cli.config.get_config_path", lambda: cfg_path):
            assert "zzfirst" in names()

            cfg_path.write_text("agent:\n  personalities:\n    zzsecond: v2\n", encoding="utf-8")
            os.utime(cfg_path, (1_800_000_000, 1_800_000_000))
            after = names()
            assert "zzsecond" in after
            assert "zzfirst" not in after
