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


class TestSessionContextSnapshot:
    """Regression tests for #81451: a concurrent workdir cron job that
    writes ``os.environ['TERMINAL_CWD']`` must not be able to redirect a
    gateway session whose cwd was bound BEFORE the cron started.

    The fix lives in :func:`gateway.session_context.set_session_vars`: when
    the caller does not pin a cwd, the function snapshots the current
    process-global ``TERMINAL_CWD`` into the per-context
    ``_SESSION_CWD`` contextvar so subsequent process-global writes (the
    cron's) are invisible to the already-bound session.
    """

    def test_unbound_cwd_snapshots_process_global(self, monkeypatch, tmp_path):
        # Arrange: gateway-style baseline.  Session binds BEFORE any cron
        # writes.
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.setenv("TERMINAL_CWD", str(baseline))

        from gateway.session_context import set_session_vars, clear_session_vars

        # Act: bound session sees the baseline cwd by snapshotting it.
        tokens = set_session_vars(platform="telegram", chat_id="123")
        try:
            # Now a concurrent workdir cron writes a different path.
            monkeypatch.setenv("TERMINAL_CWD", str(other))

            # Resolve cwd paths — must NOT pick up the cron's override.
            assert resolve_context_cwd() == baseline
            assert resolve_agent_cwd() == baseline
        finally:
            clear_session_vars(tokens)

    def test_explicit_cwd_wins_over_cron_env_mutation(self, monkeypatch, tmp_path):
        # When the caller does pin a cwd, the pin is authoritative — the
        # snapshot is bypassed, and a later cron's env mutation is also
        # ignored (this is the cron path's behavior).
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        cron_dir = tmp_path / "cron_other"
        cron_dir.mkdir()
        monkeypatch.setenv("TERMINAL_CWD", str(baseline))

        from gateway.session_context import set_session_vars, clear_session_vars

        tokens = set_session_vars(
            platform="telegram",
            chat_id="123",
            cwd=str(baseline),
        )
        try:
            # Concurrent cron dirties the env.
            monkeypatch.setenv("TERMINAL_CWD", str(cron_dir))
            assert resolve_context_cwd() == baseline
        finally:
            clear_session_vars(tokens)

    def test_agents_md_isolation_under_workdir_cron(self, monkeypatch, tmp_path):
        # Simulates the AGENTS.md leakage scenario from #81451: a session
        # bound in directory A must not pick up AGENTS.md from cron
        # workdir B, even after the cron writes process-global
        # TERMINAL_CWD=B.
        a_dir = tmp_path / "A"
        b_dir = tmp_path / "B"
        a_dir.mkdir()
        b_dir.mkdir()
        (a_dir / "AGENTS.md").write_text("project A instructions")
        (b_dir / "AGENTS.md").write_text("cron repo B instructions")

        # Baseline env matches A.
        monkeypatch.setenv("TERMINAL_CWD", str(a_dir))

        from gateway.session_context import set_session_vars, clear_session_vars

        tokens = set_session_vars(platform="telegram", chat_id="123")
        try:
            # Concurrent workdir cron writes B to the process-global.
            monkeypatch.setenv("TERMINAL_CWD", str(b_dir))

            # The session's cwd is still A — its AGENTS.md is the one
            # that should be discovered, not B's.
            cwd = resolve_context_cwd()
            assert cwd == a_dir
            assert (cwd / "AGENTS.md").read_text() == "project A instructions"
        finally:
            clear_session_vars(tokens)

