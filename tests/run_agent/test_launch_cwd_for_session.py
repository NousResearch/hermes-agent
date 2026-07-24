"""Tests for run_agent._launch_cwd_for_session — see #50438.

The TUI (``hermes --tui``) is a local user-driven process exactly like the
CLI: it runs on the user's machine, is launched from a shell, and the
launch directory is meaningful for ``-c``/``--resume``. Before #50438, the
helper rejected any source other than ``"cli"``, so TUI sessions were
stored with ``cwd = NULL`` in ``state.db`` and the Desktop GUI could not
group them under the user's working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from run_agent import _launch_cwd_for_session


@pytest.fixture
def fake_cwd(tmp_path, monkeypatch):
    """``cd`` into tmp_path so ``os.getcwd()`` returns it."""
    monkeypatch.chdir(tmp_path)
    # The function checks TERMINAL_ENV; reset it for "local backend" tests.
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    return tmp_path


class TestLocalSourcesRecordCwd:
    """Local user-driven sources must record the launch directory."""

    @pytest.mark.parametrize("source", ["cli", "tui"])
    def test_local_sources_record_getcwd(self, source, fake_cwd):
        assert _launch_cwd_for_session(source) == str(fake_cwd)

    def test_cli_explicit(self, fake_cwd):
        """Smoke: the original "cli" path still works after the fix."""
        assert _launch_cwd_for_session("cli") == str(fake_cwd)

    def test_tui_records_cwd(self, fake_cwd):
        """The fix: TUI sessions now get a recorded cwd, not NULL."""
        # Before #50438: returned None.
        # After:          returns the current working directory.
        assert _launch_cwd_for_session("tui") == str(fake_cwd)


class TestRemoteSourcesDoNotRecordCwd:
    """Gateway and cron have no stable host cwd to restore."""

    @pytest.mark.parametrize("source", ["gateway", "cron", "desktop", "api"])
    def test_remote_sources_return_none(self, source, fake_cwd):
        assert _launch_cwd_for_session(source) is None


class TestNonLocalBackendSkipsCwd:
    """A non-local TERMINAL backend (docker/ssh/modal) means the host cwd
    is irrelevant — the function must return None even for local sources.
    """

    @pytest.mark.parametrize("source", ["cli", "tui"])
    @pytest.mark.parametrize("backend", ["docker", "ssh", "modal", "singularity", "daytona"])
    def test_nonlocal_backend_returns_none(self, source, backend, fake_cwd, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", backend)
        assert _launch_cwd_for_session(source) is None

    @pytest.mark.parametrize("source", ["cli", "tui"])
    def test_local_backend_keeps_cwd(self, source, fake_cwd, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "local")
        assert _launch_cwd_for_session(source) == str(fake_cwd)


class TestDeletedCwd:
    """If the cwd is unlinked out from under us, return None gracefully."""

    def test_oserror_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        # Replace os.getcwd with one that raises
        def _raise_oserror():
            raise OSError("cwd was deleted")
        monkeypatch.setattr(os, "getcwd", _raise_oserror)
        # Should not raise
        assert _launch_cwd_for_session("tui") is None
        assert _launch_cwd_for_session("cli") is None
