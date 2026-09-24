"""Tests for the network git timeout budget in hermes_cli.plugins_cmd (#121027).

Plugin install touches the network via clone / fetch / pull. Those verbs used a
hard-coded 60s timeout while `hermes update`'s own network git budget is 300s,
so slow-connection homes could self-update but never install a plugin (the
memory-provider migration of a provider that left core, e.g. hindsight, died on
the same wall). Network verbs now use a configurable budget with a 300s
default; local-only verbs keep their short budgets.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

import hermes_cli.plugins_cmd as pc


class TestNetworkGitTimeoutValue:
    def test_default_is_300(self, monkeypatch):
        monkeypatch.delenv("HERMES_GIT_TIMEOUT_SECONDS", raising=False)
        assert pc._network_git_timeout_seconds() == 300

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("HERMES_GIT_TIMEOUT_SECONDS", "900")
        assert pc._network_git_timeout_seconds() == 900

    def test_env_floor_at_30(self, monkeypatch):
        monkeypatch.setenv("HERMES_GIT_TIMEOUT_SECONDS", "30")
        assert pc._network_git_timeout_seconds() == 30

    def test_env_below_floor_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("HERMES_GIT_TIMEOUT_SECONDS", "5")
        assert pc._network_git_timeout_seconds() == 300

    def test_env_whitespace_is_tolerated(self, monkeypatch):
        monkeypatch.setenv("HERMES_GIT_TIMEOUT_SECONDS", " 120 ")
        assert pc._network_git_timeout_seconds() == 120

    def test_env_garbage_falls_back_to_default(self, monkeypatch):
        for bad in ("abc", "", "12.5"):
            monkeypatch.setenv("HERMES_GIT_TIMEOUT_SECONDS", bad)
            assert pc._network_git_timeout_seconds() == 300


def _capture_run(monkeypatch, captured: dict):
    """Replace pc._run_plugin_git with a recorder; returns the fake."""
    def fake(git_exe, target, *args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        raise AssertionError("test did not expect success")
    monkeypatch.setattr(pc, "_run_plugin_git", fake)
    return fake


class TestCloneTimeout:
    def test_clone_uses_network_budget_by_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HERMES_GIT_TIMEOUT_SECONDS", raising=False)
        captured = {}
        _capture_run(monkeypatch, captured)
        with pytest.raises(AssertionError, match="did not expect success"):
            pc._clone_plugin_repo(tmp_path / "clone", "https://example.test/repo.git", None)
        assert captured["kwargs"]["timeout"] == 300
        assert captured["args"][0] == "clone"

    def test_clone_uses_env_budget(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_GIT_TIMEOUT_SECONDS", "900")
        captured = {}
        _capture_run(monkeypatch, captured)
        with pytest.raises(AssertionError, match="did not expect success"):
            pc._clone_plugin_repo(tmp_path / "clone", "https://example.test/repo.git", None)
        assert captured["kwargs"]["timeout"] == 900

    def test_clone_timeout_message_reports_real_budget(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HERMES_GIT_TIMEOUT_SECONDS", raising=False)
        monkeypatch.setattr(pc, "_resolve_git_executable", lambda: "git")
        import subprocess as _sp
        with pytest.raises(pc.PluginOperationError) as excinfo:
            with _patch_run_git(monkeypatch, _sp.TimeoutExpired("git clone", 300)):
                pc._clone_plugin_repo(tmp_path / "clone", "https://example.test/repo.git", None)
        msg = str(excinfo.value)
        assert "timed out after 300 seconds" in msg
        assert "HERMES_GIT_TIMEOUT_SECONDS" in msg

    def test_clone_timeout_message_reports_custom_budget(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_GIT_TIMEOUT_SECONDS", "120")
        monkeypatch.setattr(pc, "_resolve_git_executable", lambda: "git")
        import subprocess as _sp
        with pytest.raises(pc.PluginOperationError) as excinfo:
            with _patch_run_git(monkeypatch, _sp.TimeoutExpired("git clone", 120)):
                pc._clone_plugin_repo(tmp_path / "clone", "https://example.test/repo.git", None)
        assert "timed out after 120 seconds" in str(excinfo.value)


def _patch_run_git(monkeypatch, exc_to_raise):
    """Context manager: make pc._run_plugin_git raise *exc_to_raise* (timeout path)."""
    from contextlib import contextmanager

    @contextmanager
    def _patch():
        def fake(git_exe, target, *args, **kwargs):
            raise exc_to_raise
        monkeypatch.setattr(pc, "_run_plugin_git", fake)
        try:
            yield
        finally:
            pass

    return _patch()


class TestCheckoutExactRevisionTimeout:
    def test_fetch_uses_network_budget_checkout_keeps_local(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HERMES_GIT_TIMEOUT_SECONDS", raising=False)
        monkeypatch.setenv("HERMES_GIT_TIMEOUT_SECONDS", "450")
        seen = {}

        def fake(git_exe, repo, *args, failure_prefix="", **kwargs):
            seen[args[0]] = kwargs.get("timeout", 60)
            if args[0] == "fetch":
                raise subprocess.TimeoutExpired("git fetch", 450)
            raise AssertionError("test did not expect success")

        monkeypatch.setattr(pc, "_run_plugin_git", fake)
        with pytest.raises(pc.PluginOperationError) as excinfo:
            pc._checkout_exact_revision(tmp_path, "git", "a" * 40)
        assert seen.get("fetch") == 450
        assert "timed out after 450 seconds" in str(excinfo.value)
        assert "HERMES_GIT_TIMEOUT_SECONDS" in str(excinfo.value)


class TestPullTimeout:
    def test_pull_uses_network_budget(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HERMES_GIT_TIMEOUT_SECONDS", raising=False)
        monkeypatch.setattr(pc, "_resolve_git_executable", lambda: "git")
        timeouts = []

        def fake(git_exe, target, *args, **kwargs):
            timeouts.append((args, kwargs.get("timeout")))
            if args and args[0] == "pull":
                raise subprocess.TimeoutExpired("git pull", 300)
            # empty stdout everywhere: `status --porcelain -z` sees a clean tree,
            # so _autostash_dirty_tree skips the stash path and we reach the pull.
            return subprocess.CompletedProcess(list(args), 0, stdout="")

        monkeypatch.setattr(pc, "_run_plugin_git", fake)
        ok, err = pc._git_pull_plugin_dir(tmp_path)
        assert ok is False
        pull = next((t for a, t in timeouts if a and a[0] == "pull"), None)
        assert pull == 300
        assert "timed out after 300 seconds" in err

    def test_pull_error_reports_env_budget(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_GIT_TIMEOUT_SECONDS", "180")
        monkeypatch.setattr(pc, "_resolve_git_executable", lambda: "git")

        def fake(git_exe, target, *args, **kwargs):
            if args and args[0] == "pull":
                raise subprocess.TimeoutExpired("git pull", 180)
            return subprocess.CompletedProcess(list(args), 0, stdout="")

        monkeypatch.setattr(pc, "_run_plugin_git", fake)
        ok, err = pc._git_pull_plugin_dir(tmp_path)
        assert ok is False
        assert "timed out after 180 seconds" in err
