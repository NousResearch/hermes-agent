"""Remote-backend cwds are honored verbatim (#83515).

A non-local terminal backend (ssh, docker, ...) runs its filesystem elsewhere:
``Path.is_dir()`` is always False for a directory that only exists on the remote
host, so validating the configured cwd against the local filesystem discarded a
valid remote workspace (the "TERMINAL_CWD does not exist" warning the issue
quotes) and every cwd consumer fell back to the gateway host's launch dir.
``~`` in a remote cwd names the REMOTE user's home — expanding it against this
host would silently point the agent at the wrong directory.
"""
from __future__ import annotations

import os
from pathlib import Path

import agent.prompt_builder as pb
import agent.runtime_cwd as rt
from agent.runtime_cwd import (
    clear_session_cwd,
    resolve_agent_cwd,
    resolve_context_cwd,
    set_session_cwd,
)


class TestRemoteBackendSkipsLocalValidation:
    def test_resolve_agent_cwd_honors_remote_session_cwd(self, monkeypatch):
        """The issue's exact reproduction: session pinned to a dir that only exists on
        the SSH target; TERMINAL_CWD holds the configured global remote cwd."""
        remote = "/workspace/research"  # does not exist on this host
        assert not Path(remote).is_dir()
        monkeypatch.setenv("TERMINAL_ENV", "ssh")
        monkeypatch.setenv("TERMINAL_CWD", "/workspace")
        token = set_session_cwd(remote)
        try:
            assert resolve_agent_cwd() == Path(remote)
            assert resolve_context_cwd() == Path(remote)
        finally:
            rt._SESSION_CWD.reset(token)

    def test_resolve_agent_cwd_honors_remote_terminal_cwd(self, monkeypatch):
        """The global fallback: a configured remote terminal.cwd (no session override)
        must survive too, not just the session pin."""
        remote = "/workspace"  # does not exist on this host
        assert not Path(remote).is_dir()
        monkeypatch.setenv("TERMINAL_ENV", "ssh")
        monkeypatch.setenv("TERMINAL_CWD", remote)
        monkeypatch.delenv("HERMES_SESSION_CWD", raising=False)
        assert resolve_agent_cwd() == Path(remote)
        assert resolve_context_cwd() == Path(remote)

    def test_remote_cwd_tilde_stays_verbatim(self, monkeypatch):
        """``~`` in a remote cwd is the REMOTE user's home; expanding it here (the
        local-backend arm) would silently substitute this host's home (#83515)."""
        monkeypatch.setenv("TERMINAL_ENV", "ssh")
        monkeypatch.setenv("TERMINAL_CWD", "/workspace")
        token = set_session_cwd("~/project")
        try:
            assert resolve_agent_cwd() == Path("~/project")
        finally:
            rt._SESSION_CWD.reset(token)

    def test_local_backend_still_rejects_missing_cwd(self, monkeypatch, tmp_path):
        """Local keeps the host validation: a configured dir that is really gone still
        falls back (and the missing-dir warning keeps its diagnostic value)."""
        missing = tmp_path / "does-not-exist"
        monkeypatch.setenv("TERMINAL_ENV", "local")
        monkeypatch.setenv("TERMINAL_CWD", str(missing))
        monkeypatch.chdir(tmp_path)
        assert resolve_agent_cwd() == tmp_path
        assert resolve_context_cwd() is None

    def test_remote_backend_detected_through_terminal_scope(self, monkeypatch):
        """Under gateway multiplexing the per-turn terminal scope carries the ACTIVE
        profile's backend; a scope-aware read must drive the remote decision, not the
        process-global env var (which holds the launch profile's)."""
        from tools.terminal_scope import set_terminal_scope, _terminal_scope_var

        remote = "/workspace/research"
        monkeypatch.setenv("TERMINAL_ENV", "local")  # launch profile runs local
        monkeypatch.setenv("TERMINAL_CWD", "/workspace")
        token = set_terminal_scope({"TERMINAL_ENV": "ssh", "TERMINAL_CWD": "/workspace"})
        try:
            assert resolve_agent_cwd() == Path("/workspace")
        finally:
            _terminal_scope_var.reset(token)


class TestRemoteBackendListParity:
    """``_REMOTE_TERMINAL_BACKENDS`` is duplicated (circular import) between
    agent/runtime_cwd.py and agent/prompt_builder.py — a backend added to only one
    would keep being treated as local by the other."""

    def test_matches_prompt_builder(self):
        assert rt._REMOTE_TERMINAL_BACKENDS == pb._REMOTE_TERMINAL_BACKENDS
