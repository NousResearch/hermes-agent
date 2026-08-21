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

    def test_preserves_opaque_terminal_cwd_when_local_validation_is_denied(
        self, monkeypatch
    ):
        remote_cwd = Path("/home/red-worker/workspace")
        monkeypatch.setenv("TERMINAL_CWD", str(remote_cwd))
        original_is_dir = Path.is_dir

        def deny_remote_cwd(path):
            if path == remote_cwd:
                raise PermissionError(13, "Permission denied", str(path))
            return original_is_dir(path)

        monkeypatch.setattr(Path, "is_dir", deny_remote_cwd)

        assert resolve_context_cwd() == remote_cwd

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

    def test_preserves_opaque_session_cwd_when_local_validation_is_denied(
        self, monkeypatch, tmp_path
    ):
        remote_cwd = Path("/home/red-worker/workspace")
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        original_is_dir = Path.is_dir

        def deny_remote_cwd(path):
            if path == remote_cwd:
                raise PermissionError(13, "Permission denied", str(path))
            return original_is_dir(path)

        monkeypatch.setattr(Path, "is_dir", deny_remote_cwd)
        token = set_session_cwd(str(remote_cwd))
        try:
            assert resolve_context_cwd() == remote_cwd
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

