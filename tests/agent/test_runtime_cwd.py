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


class TestRemoteBackendSkipsLocalValidation:
    """#83515: an SSH (or other remote) backend's cwd lives on the backend's
    filesystem, not this host's — Path.is_dir() is always False for a path
    that only exists on the remote target, so it must not be used to reject
    an otherwise-valid remote cwd."""

    def test_resolve_agent_cwd_honors_nonexistent_remote_session_cwd(self, monkeypatch):
        remote = "/workspace/research"  # does not exist on this host
        assert not Path(remote).is_dir()
        monkeypatch.setenv("TERMINAL_ENV", "ssh")
        monkeypatch.setenv("TERMINAL_CWD", "/workspace")
        token = set_session_cwd(remote)
        try:
            assert resolve_agent_cwd() == Path(remote)
        finally:
            rt._SESSION_CWD.reset(token)

    def test_resolve_agent_cwd_honors_nonexistent_remote_terminal_cwd(self, monkeypatch):
        remote = "/workspace"  # does not exist on this host either
        assert not Path(remote).is_dir()
        monkeypatch.setenv("TERMINAL_ENV", "ssh")
        monkeypatch.setenv("TERMINAL_CWD", remote)
        assert resolve_agent_cwd() == Path(remote)

    def test_local_backend_still_rejects_nonexistent_cwd(self, monkeypatch, tmp_path):
        missing = tmp_path / "does-not-exist"
        monkeypatch.setenv("TERMINAL_ENV", "local")
        monkeypatch.setenv("TERMINAL_CWD", str(missing))
        assert resolve_agent_cwd() == Path(os.getcwd())

    def test_remote_session_cwd_is_not_expanded_locally(self, monkeypatch):
        # `~` in a remote path means the backend's home, not this host's —
        # expanding it here (as the local-backend arm does) would silently
        # substitute the wrong directory. Must match session.create's
        # verbatim raw_cwd handling in tui_gateway/methods_session.py.
        monkeypatch.setenv("TERMINAL_ENV", "ssh")
        monkeypatch.setenv("TERMINAL_CWD", "/workspace")
        token = set_session_cwd("~/project")
        try:
            assert resolve_agent_cwd() == Path("~/project")
        finally:
            rt._SESSION_CWD.reset(token)

    def test_remote_terminal_cwd_is_not_expanded_locally(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "ssh")
        monkeypatch.setenv("TERMINAL_CWD", "~/project")
        assert resolve_agent_cwd() == Path("~/project")


class TestRemoteBackendListStaysInSync:
    """_REMOTE_TERMINAL_BACKENDS is duplicated (to dodge a circular import)
    between agent/runtime_cwd.py and agent/prompt_builder.py. A backend added
    to only one would keep being treated as local by the other — assert they
    agree."""

    def test_matches_prompt_builder(self):
        import agent.prompt_builder as pb

        assert rt._REMOTE_TERMINAL_BACKENDS == pb._REMOTE_TERMINAL_BACKENDS

