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

    def test_falls_back_to_getcwd_when_unset(self, monkeypatch, tmp_path):
        # The #19242 local-CLI contract: TERMINAL_CWD is unset, so the launch dir wins.
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.chdir(tmp_path)
        assert resolve_agent_cwd() == tmp_path

    def test_skips_nonexistent_terminal_cwd(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path / "gone"))
        monkeypatch.chdir(tmp_path)
        assert resolve_agent_cwd() == tmp_path

    def test_expands_leading_tilde(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_CWD", "~")
        assert resolve_agent_cwd() == Path(os.path.expanduser("~"))

    def test_whitespace_only_terminal_cwd_falls_back_to_getcwd(self, monkeypatch, tmp_path):
        # "   ".strip() → "" → falsy, so the launch dir wins (not a "   " path).
        monkeypatch.setenv("TERMINAL_CWD", "   ")
        monkeypatch.chdir(tmp_path)
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

    def test_returns_none_when_unset(self, monkeypatch):
        # Unset → None; the caller (build_context_files_prompt) then getcwds —
        # the local-CLI #19242 contract. Discovery still runs; it is NOT skipped.
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        assert resolve_context_cwd() is None

    def test_returns_none_for_nonexistent_dir(self, monkeypatch, tmp_path):
        # A configured but missing dir must not be returned. It previously was,
        # which diverged from resolve_agent_cwd and let an invalid cwd steer
        # context discovery. Now it is validated and drops to None.
        missing = tmp_path / "gone"
        monkeypatch.setenv("TERMINAL_CWD", str(missing))
        assert resolve_context_cwd() is None

    def test_returns_install_tree_when_explicitly_configured(self, monkeypatch):
        # An EXPLICITLY configured install-tree cwd is honored verbatim — the
        # Hermes source tree is a legitimate workspace when the user is
        # developing Hermes. Only the fallback path (cwd=None → os.getcwd())
        # is policed, in build_context_files_prompt (#64590).
        monkeypatch.setenv("TERMINAL_CWD", str(rt._PACKAGE_ROOT))
        assert resolve_context_cwd() == rt._PACKAGE_ROOT

    def test_expands_leading_tilde(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_CWD", "~")
        assert resolve_context_cwd() == Path(os.path.expanduser("~"))

    def test_whitespace_only_terminal_cwd_returns_none(self, monkeypatch):
        # "   ".strip() → "" → None, so the caller getcwds for discovery rather
        # than building Path("   ") and resolving garbage under the launch dir.
        monkeypatch.setenv("TERMINAL_CWD", "   ")
        assert resolve_context_cwd() is None

    def test_missing_authoritative_cwd_never_falls_back_for_context(
        self, monkeypatch, tmp_path
    ):
        from agent.prompt_builder import build_context_files_prompt

        process_dir = tmp_path / "foreign-process-workspace"
        process_dir.mkdir()
        (process_dir / "AGENTS.md").write_text("FOREIGN_CONTEXT_SENTINEL")
        missing = tmp_path / "deleted-cron-workspace"
        monkeypatch.chdir(process_dir)

        tokens = rt.set_authoritative_session_cwd(str(missing))
        try:
            resolved = resolve_context_cwd()
            assert resolved == missing
            prompt = build_context_files_prompt(cwd=str(resolved), skip_soul=True)
            assert "FOREIGN_CONTEXT_SENTINEL" not in prompt
        finally:
            rt.reset_authoritative_session_cwd(tokens)


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

    def test_terminal_and_file_tools_follow_session_cwd(self, monkeypatch, tmp_path):
        from tools import file_tools, terminal_tool

        session_dir = tmp_path / "session"
        process_dir = tmp_path / "process"
        session_dir.mkdir()
        process_dir.mkdir()
        monkeypatch.setenv("TERMINAL_CWD", str(process_dir))
        monkeypatch.setenv("TERMINAL_ENV", "local")

        token = set_session_cwd(str(session_dir))
        try:
            assert terminal_tool._get_env_config()["cwd"] == str(session_dir)
            assert file_tools._configured_terminal_cwd() == str(session_dir)
        finally:
            rt._SESSION_CWD.reset(token)

    def test_execute_code_follows_session_cwd(self, monkeypatch, tmp_path):
        from tools.code_execution_tool import _resolve_child_cwd

        session_dir = tmp_path / "session"
        process_dir = tmp_path / "process"
        staging_dir = tmp_path / "staging"
        session_dir.mkdir()
        process_dir.mkdir()
        staging_dir.mkdir()
        monkeypatch.setenv("TERMINAL_CWD", str(process_dir))

        token = set_session_cwd(str(session_dir))
        try:
            assert _resolve_child_cwd("project", str(staging_dir)) == str(session_dir)
        finally:
            rt._SESSION_CWD.reset(token)

    def test_empty_session_cwd_falls_back_to_terminal_cwd(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        token = set_session_cwd("")
        try:
            assert resolve_agent_cwd() == tmp_path
            assert resolve_context_cwd() == tmp_path
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

    def test_nonexistent_session_cwd_falls_back(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        token = set_session_cwd(str(tmp_path / "gone"))
        try:
            # resolve_agent_cwd guards on isdir; a missing session cwd must not win.
            assert resolve_agent_cwd() == tmp_path
        finally:
            rt._SESSION_CWD.reset(token)
