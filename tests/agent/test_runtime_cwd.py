"""Tests for agent/runtime_cwd.py — the single source of truth for the agent working directory."""

import os
from pathlib import Path

import pytest

import agent.runtime_cwd as rt
from agent.runtime_cwd import resolve_agent_cwd, resolve_context_cwd


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

    def test_returns_nonexistent_dir_unguarded(self, monkeypatch, tmp_path):
        # Deliberate asymmetry vs resolve_agent_cwd: context discovery has no isdir
        # guard, so a missing dir is returned (not None) — discovery just finds nothing.
        missing = tmp_path / "gone"
        monkeypatch.setenv("TERMINAL_CWD", str(missing))
        assert resolve_context_cwd() == missing

    def test_expands_leading_tilde(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_CWD", "~")
        assert resolve_context_cwd() == Path(os.path.expanduser("~"))

    def test_whitespace_only_terminal_cwd_returns_none(self, monkeypatch):
        # "   ".strip() → "" → None, so the caller getcwds for discovery rather
        # than building Path("   ") and resolving garbage under the launch dir.
        monkeypatch.setenv("TERMINAL_CWD", "   ")
        assert resolve_context_cwd() is None
