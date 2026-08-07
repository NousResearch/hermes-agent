"""Regression tests for cron session cwd provenance (#79623).

Cron jobs run with a configured ``workdir`` (pinned via the ``_SESSION_CWD``
contextvar), but the durable ``sessions`` row left ``cwd`` NULL because
``_launch_cwd_for_session`` only stamped local CLI sessions. This broke
``hermes sessions list --workspace`` and resume-to-workspace for cron runs.
"""

import os

import pytest

import run_agent
from agent.runtime_cwd import _SESSION_CWD, set_session_cwd


def _with_session_cwd(cwd, fn):
    """Run ``fn`` with the _SESSION_CWD contextvar pinned to ``cwd``."""
    token = set_session_cwd(cwd)
    try:
        return fn()
    finally:
        _SESSION_CWD.reset(token)


class TestCronCwdProvenance:
    def test_cron_session_stamps_configured_workdir(self, tmp_path, monkeypatch):
        """A cron job with a configured workdir records it as the session cwd."""
        monkeypatch.setattr(os, "getcwd", lambda: "/unrelated/launch-dir")

        def _call():
            return run_agent._launch_cwd_for_session("cron")

        result = _with_session_cwd(str(tmp_path), _call)
        assert result == str(tmp_path)

    def test_cron_without_workdir_records_none(self, monkeypatch):
        """A cron job with no workdir records nothing (unchanged)."""

        def _call():
            return run_agent._launch_cwd_for_session("cron")

        result = _with_session_cwd("", _call)
        assert result is None

    def test_cron_missing_workdir_records_none(self, tmp_path, monkeypatch):
        """A configured workdir that no longer exists records nothing."""
        missing = str(tmp_path / "does-not-exist")

        def _call():
            return run_agent._launch_cwd_for_session("cron")

        result = _with_session_cwd(missing, _call)
        assert result is None

    def test_cli_still_records_launch_cwd(self, tmp_path, monkeypatch):
        """CLI behavior is unchanged: records os.getcwd()."""
        monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))
        assert run_agent._launch_cwd_for_session("cli") == str(tmp_path)

    def test_gateway_still_records_none(self, monkeypatch):
        """Gateway behavior is unchanged: records nothing."""
        assert run_agent._launch_cwd_for_session("telegram") is None
