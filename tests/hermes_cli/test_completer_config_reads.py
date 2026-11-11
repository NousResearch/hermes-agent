"""The /personality completer runs per keystroke, but a config edit on disk must show up in the
next completion (never a stale list), and reading the config must not move the session's cwd."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from hermes_cli.commands_completion import SlashCommandCompleter


def _names(text):
    return {c.text for c in SlashCommandCompleter().get_completions(Document(text), CompleteEvent())}


def test_config_edit_on_disk_refreshes_completions(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("agent:\n  personalities:\n    zzfirst: v1\n", encoding="utf-8")
    # Pin mtimes so the edit below is a guaranteed signature change
    # regardless of filesystem timestamp granularity.
    os.utime(cfg_path, (1_700_000_000, 1_700_000_000))

    with patch("hermes_cli.config.get_config_path", lambda: cfg_path):
        assert "zzfirst" in _names("/personality zz")

        cfg_path.write_text("agent:\n  personalities:\n    zzsecond: v2\n", encoding="utf-8")
        os.utime(cfg_path, (1_800_000_000, 1_800_000_000))
        after = _names("/personality zz")
        assert "zzsecond" in after
        assert "zzfirst" not in after


def test_completing_a_personality_keeps_the_worktree_terminal_cwd(tmp_path, monkeypatch):
    """``hermes -w`` selects the worktree through TERMINAL_CWD without a chdir; typing
    ``/personality <partial>`` before the first turn must not reset it to the checkout."""
    checkout, worktree = tmp_path / "checkout", tmp_path / "checkout-wt"
    checkout.mkdir()
    worktree.mkdir()
    monkeypatch.chdir(checkout)
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.setenv("TERMINAL_CWD", str(worktree))
    monkeypatch.delitem(sys.modules, "gateway.run", raising=False)

    assert "kawaii" in _names("/personality ka")
    assert os.environ["TERMINAL_CWD"] == str(worktree)
