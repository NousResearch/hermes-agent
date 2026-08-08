"""Tests for agent/runtime_cwd.py — the single source of truth for the agent working directory."""

import os
from pathlib import Path

import pytest

import agent.runtime_cwd as rt
from agent.runtime_cwd import (
    clear_session_cwd,
    resolve_agent_cwd,
    resolve_context_cwd,
    set_session_cwd,
)


def _raise_oserror(*args, **kwargs):
    raise OSError("cwd gone")


class TestResolveAgentCwd:
    def test_prefers_terminal_cwd_over_getcwd(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        monkeypatch.chdir(os.path.expanduser("~"))
        assert resolve_agent_cwd() == tmp_path





    def test_propagates_oserror_from_getcwd(self, monkeypatch):
        # The fallback arm calls os.getcwd(), which can raise OSError (deleted cwd).
        # The resolver must NOT swallow it — build_environment_hints owns the
        # try/except OSError guard at the call site (prompt_builder.py:805).
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.setattr(rt.os, "getcwd", _raise_oserror)
        with pytest.raises(OSError):
            resolve_agent_cwd()


class TestResolveContextCwd:
    def test_returns_dir_when_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        assert resolve_context_cwd() == tmp_path




    def test_expands_leading_tilde(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_CWD", "~")
        assert resolve_context_cwd() == Path(os.path.expanduser("~"))



class TestSessionCwdOverride:
    """The #29531 per-session arm: a contextvar cwd wins over TERMINAL_CWD so a
    multi-session gateway can pin each session to its own folder."""

    def test_session_cwd_overrides_terminal_cwd(self, monkeypatch, tmp_path):
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        token = set_session_cwd(str(other))
        try:
            assert resolve_agent_cwd() == other
            assert resolve_context_cwd() == other
        finally:
            rt._SESSION_CWD.reset(token)


    def test_clear_session_cwd_restores_terminal_cwd(self, monkeypatch, tmp_path):
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        token = set_session_cwd(str(other))
        try:
            clear_session_cwd()
            assert resolve_agent_cwd() == tmp_path
        finally:
            rt._SESSION_CWD.reset(token)



class TestTerminalCwdFallbackBridge:
    """config.yaml must reach cwd resolution even when no launcher bridged it.

    This module's contract assumes ``terminal.cwd`` was bridged to
    ``TERMINAL_CWD`` once at gateway/cron startup. A process that skips those
    launcher bridges read an unset variable and fell back to ``os.getcwd()``
    — the daemon's working directory — so context-file discovery loaded the
    wrong ``AGENTS.md`` while the user's configured workspace was ignored
    (#74116). ``tools/terminal_tool.py`` already closes this hole for the
    terminal backend (#63141, #54449, #61115, #65696); these cover the same
    guarantee for cwd resolution.
    """

    @pytest.fixture(autouse=True)
    def _unattempted_bridge(self, monkeypatch):
        monkeypatch.setattr(rt, "_config_bridge_attempted", False)
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        yield

    @staticmethod
    def _write_config(hermes_home: Path, cwd: Path) -> None:
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "config.yaml").write_text(
            f"terminal:\n  backend: local\n  cwd: {cwd}\n"
        )

    def test_context_cwd_backfills_from_config_when_env_is_unset(
        self, monkeypatch, tmp_path,
    ):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        self._write_config(tmp_path / "hermes", workspace)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        monkeypatch.chdir(tmp_path)

        assert resolve_context_cwd() == workspace

    def test_agent_cwd_backfills_from_config_when_env_is_unset(
        self, monkeypatch, tmp_path,
    ):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        self._write_config(tmp_path / "hermes", workspace)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        monkeypatch.chdir(tmp_path)

        assert resolve_agent_cwd() == workspace

    def test_explicit_env_still_wins_over_config(self, monkeypatch, tmp_path):
        """A launcher's bridge or the user's .env stays authoritative."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        explicit = tmp_path / "explicit"
        explicit.mkdir()
        self._write_config(tmp_path / "hermes", workspace)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        monkeypatch.setenv("TERMINAL_CWD", str(explicit))

        assert resolve_context_cwd() == explicit

    def test_session_override_still_wins_and_skips_the_bridge(
        self, monkeypatch, tmp_path,
    ):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        pinned = tmp_path / "pinned"
        pinned.mkdir()
        self._write_config(tmp_path / "hermes", workspace)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

        set_session_cwd(str(pinned))
        try:
            assert resolve_context_cwd() == pinned
        finally:
            clear_session_cwd()

    def test_no_terminal_config_leaves_context_cwd_unset(
        self, monkeypatch, tmp_path,
    ):
        """Without terminal.cwd, None still means 'use the launch dir'."""
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("model:\n  default: x\n")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.chdir(tmp_path)

        assert resolve_context_cwd() is None

    def test_bridge_is_attempted_only_once(self, monkeypatch, tmp_path):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("model:\n  default: x\n")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.chdir(tmp_path)
        calls = []

        def _counting(**kwargs):
            calls.append(kwargs)

        monkeypatch.setattr(
            "hermes_cli.config.apply_terminal_config_to_env", _counting,
        )
        resolve_context_cwd()
        resolve_context_cwd()
        resolve_agent_cwd()

        assert len(calls) == 1

    def test_config_failure_never_breaks_resolution(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        monkeypatch.chdir(tmp_path)

        def _boom(**kwargs):
            raise RuntimeError("config exploded")

        monkeypatch.setattr(
            "hermes_cli.config.apply_terminal_config_to_env", _boom,
        )
        assert resolve_context_cwd() is None
        assert resolve_agent_cwd() == tmp_path
