import builtins
import os
from subprocess import CompletedProcess

import pytest

from hermes_cli import version_info


@pytest.mark.skipif(os.name != "nt", reason="Windows managed-worker subprocess hang")
def test_run_git_skips_live_provenance_in_managed_worker(monkeypatch, tmp_path):
    """Managed Windows workers must not spawn Git before receiving their prompt."""
    monkeypatch.setattr("agent.runtime_session_store.is_worker_process", lambda: True)

    def forbidden(*args, **kwargs):
        raise AssertionError("managed workers must not spawn Git for version telemetry")

    monkeypatch.setattr(version_info.subprocess, "run", forbidden)

    assert version_info._run_git(tmp_path, "rev-parse", "HEAD") is None


def test_run_git_retains_live_provenance_outside_managed_worker(monkeypatch, tmp_path):
    monkeypatch.setattr("agent.runtime_session_store.is_worker_process", lambda: False)
    monkeypatch.setattr(
        version_info.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(args[0], 0, stdout="abc123\n", stderr=""),
    )

    assert version_info._run_git(tmp_path, "rev-parse", "HEAD") == "abc123"


@pytest.mark.skipif(os.name != "nt", reason="Windows managed-worker subprocess hang")
def test_run_git_keeps_provenance_for_windows_non_worker(monkeypatch, tmp_path):
    monkeypatch.setattr("agent.runtime_session_store.is_worker_process", lambda: False)
    monkeypatch.setattr(
        version_info.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(args[0], 0, stdout="abc123\n", stderr=""),
    )

    assert version_info._run_git(tmp_path, "rev-parse", "HEAD") == "abc123"


def test_run_git_falls_back_when_worker_check_import_fails(monkeypatch, tmp_path):
    real_import = builtins.__import__

    def import_without_worker_check(name, *args, **kwargs):
        if name == "agent.runtime_session_store":
            raise ImportError("worker helper unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(version_info.os, "name", "nt")
    monkeypatch.setattr(builtins, "__import__", import_without_worker_check)
    monkeypatch.setattr(
        version_info.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(args[0], 0, stdout="abc123\n", stderr=""),
    )

    assert version_info._run_git(tmp_path, "rev-parse", "HEAD") == "abc123"
