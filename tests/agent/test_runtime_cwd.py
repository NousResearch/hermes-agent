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


class TestRemoteBackendIgnoresTerminalCwd:
    """TERMINAL_CWD on a remote backend (ssh/docker/...) is a path on the
    REMOTE machine — it must never be stat'd or returned from the host
    process (root@ssh profiles set TERMINAL_CWD=/root, unreadable locally;
    that used to abort system-prompt construction with PermissionError)."""

    @pytest.mark.parametrize("backend", ["ssh", "docker", "modal"])
    def test_context_cwd_ignored_for_remote_backend(self, monkeypatch, tmp_path, backend):
        monkeypatch.setenv("TERMINAL_ENV", backend)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        assert resolve_context_cwd() is None

    @pytest.mark.parametrize("backend", ["ssh", "docker"])
    def test_agent_cwd_falls_back_to_getcwd_for_remote_backend(self, monkeypatch, tmp_path, backend):
        monkeypatch.setenv("TERMINAL_ENV", backend)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        assert resolve_agent_cwd() == tmp_path

    def test_local_backend_still_uses_terminal_cwd(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TERMINAL_ENV", "local")
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        assert resolve_context_cwd() == tmp_path

    def test_unset_env_is_local(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TERMINAL_ENV", raising=False)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        assert resolve_context_cwd() == tmp_path



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

