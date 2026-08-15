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


class TestContextScopeSurvivesMissingDir:
    """A pinned context scope must not silently degrade into the execution cwd.

    Returning None here does NOT mean "no context files": build_context_files_prompt
    treats None as "fall back to os.getcwd()", which in multiplex is the shared gateway
    launch directory — exactly the cross-profile leak the scope exists to prevent. A
    missing profile directory must therefore stay authoritative (discovery simply finds
    nothing there) and say so loudly.
    """

    def test_missing_context_dir_stays_authoritative(self, monkeypatch, tmp_path, caplog):
        import logging
        import sys

        # Set and read through ONE module object. Other suites in this directory
        # reload sibling modules, and a duplicated agent.runtime_cwd would carry
        # its own ContextVars — the setter would then write a scope the resolver
        # cannot see, and this test would pass alone and fail in the full run.
        mod = sys.modules["agent.runtime_cwd"]

        launch_dir = tmp_path / "gateway-launch"
        launch_dir.mkdir()
        monkeypatch.setenv("TERMINAL_CWD", str(launch_dir))
        missing = tmp_path / "profiles" / "never-created"

        token = mod.set_context_file_cwd(str(missing))
        try:
            assert mod.is_context_file_cwd_scoped(), (
                "scope did not take effect; module identity split "
                f"(rt is sys.modules entry: {mod is rt})"
            )
            with caplog.at_level(logging.ERROR, logger="agent.runtime_cwd"):
                resolved = mod.resolve_context_cwd()
        finally:
            mod.reset_context_file_cwd(token)

        assert resolved == missing, "must not fall back to the shared execution cwd"
        assert resolved != launch_dir
        assert any("does not exist" in r.getMessage() for r in caplog.records)



