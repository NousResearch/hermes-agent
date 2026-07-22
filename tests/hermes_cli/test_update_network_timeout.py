"""Tests for hermes update network timeout handling.

Regression for: subprocess.run calls in the update path lacked ``timeout``
arguments, so a stalled GitHub SSH connection (flaky WiFi, captive
portal, stateful firewall drop) caused ``hermes update`` to hang
indefinitely. KeyboardInterrupt produced an ugly traceback instead of
a clean error.

Each test mocks ``subprocess.run`` to raise ``TimeoutExpired`` and
verifies that the affected code path exits cleanly without leaking
the underlying exception to the user.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import hermes_cli.main as m


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_git_repo(tmp_path: Path) -> Path:
    """Create ``tmp_path/.git`` so ``_cmd_update_impl`` takes the git path
    instead of falling through to ``_cmd_update_pip`` (which would try to
    reach PyPI)."""
    (tmp_path / ".git").mkdir()
    return tmp_path


def _ok_result(**overrides) -> MagicMock:
    """Default success-shaped CompletedProcess for unrelated git commands."""
    result = MagicMock()
    result.returncode = 0
    result.stdout = overrides.get("stdout", "")
    result.stderr = overrides.get("stderr", "")
    return result


def _stub_unrelated_git_calls(monkeypatch) -> None:
    """Stub functions called by ``_cmd_update_impl`` *before* the fetch,
    so the test reaches the network call we want to exercise."""
    monkeypatch.setattr(m, "_run_pre_update_backup", lambda args: None)
    monkeypatch.setattr(m, "_pause_windows_gateways_for_update", lambda: None)
    monkeypatch.setattr(m, "_resume_windows_gateways_after_update", lambda *a, **k: None)
    monkeypatch.setattr(m, "_install_hangup_protection", lambda **k: None)
    monkeypatch.setattr(m, "_finalize_update_output", lambda _state: None)
    monkeypatch.setattr(m, "_invalidate_update_cache", lambda: None)
    monkeypatch.setattr(m, "_resolve_update_branch", lambda args: "main")
    monkeypatch.setattr(m, "_stash_local_changes_if_needed", lambda *a, **k: None)
    monkeypatch.setattr(m, "_capture_head_sha", lambda *a, **k: "deadbeef1234")
    monkeypatch.setattr("hermes_cli.backup.create_quick_snapshot", lambda *a, **k: None)


def _patched_exit(exit_calls: list):
    """Return a ``sys.exit`` replacement that records the code and actually
    raises ``SystemExit`` so the function-under-test unwinds cleanly. A
    plain record-and-return mock would let execution fall through to
    ``UnboundLocalError`` on ``pull_result`` after a timeout catch."""

    def _exit(code=0):
        exit_calls.append(code)
        raise SystemExit(code)

    return _exit


# ---------------------------------------------------------------------------
# _sync_with_upstream_if_needed (fork upstream sync)
# ---------------------------------------------------------------------------


def test_sync_with_upstream_handles_timeout_expired(monkeypatch, tmp_path, capsys):
    """When the upstream fetch hangs, the sync returns cleanly and
    prints a 'timed out' message instead of hanging or propagating."""
    monkeypatch.setattr(m, "_has_upstream_remote", lambda *a, **k: True)

    def fake_run(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        if "fetch" in cmd:
            raise subprocess.TimeoutExpired(cmd="git fetch", timeout=60)
        return _ok_result()

    monkeypatch.setattr(m.subprocess, "run", fake_run)

    m._sync_with_upstream_if_needed(["git"], tmp_path)

    captured = capsys.readouterr()
    assert "timed out" in captured.out, (
        f"expected 'timed out' in output, got: {captured.out!r}"
    )


def test_sync_with_upstream_handles_called_process_error(monkeypatch, tmp_path, capsys):
    """Pre-existing behavior: a non-zero exit on the upstream fetch
    also returns cleanly. Ensures the new timeout branch doesn't
    regress the CalledProcessError path."""
    monkeypatch.setattr(m, "_has_upstream_remote", lambda *a, **k: True)

    def fake_run(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        if "fetch" in cmd:
            err = subprocess.CalledProcessError(returncode=128, cmd="git fetch")
            err.stderr = "fatal: could not resolve host github.com"
            raise err
        return _ok_result()

    monkeypatch.setattr(m.subprocess, "run", fake_run)

    m._sync_with_upstream_if_needed(["git"], tmp_path)

    captured = capsys.readouterr()
    assert "Failed to fetch" in captured.out, (
        f"expected 'Failed to fetch' in output, got: {captured.out!r}"
    )


def test_sync_with_upstream_passes_timeout_to_fetch(monkeypatch, tmp_path):
    """The fetch call must include timeout= so subprocess.run doesn't
    block forever on a stalled connection. This is the property the
    user actually cares about."""
    monkeypatch.setattr(m, "_has_upstream_remote", lambda *a, **k: True)
    captured_kwargs: dict = {}

    def fake_run(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        if "fetch" in cmd:
            captured_kwargs.update(kwargs)
            raise subprocess.TimeoutExpired(cmd="git fetch", timeout=kwargs.get("timeout"))
        return _ok_result()

    monkeypatch.setattr(m.subprocess, "run", fake_run)
    m._sync_with_upstream_if_needed(["git"], tmp_path)

    assert "timeout" in captured_kwargs, (
        f"git fetch must be called with a timeout, got kwargs={captured_kwargs!r}"
    )
    assert captured_kwargs["timeout"] >= 10, (
        f"timeout must be a non-trivial duration, got {captured_kwargs['timeout']!r}"
    )


# ---------------------------------------------------------------------------
# _cmd_update_impl — origin fetch (line ~8670)
# ---------------------------------------------------------------------------


def test_update_origin_fetch_handles_timeout(monkeypatch, tmp_path, capsys):
    """_cmd_update_impl must catch TimeoutExpired on the origin fetch
    and exit cleanly with a Network timeout message."""
    _make_fake_git_repo(tmp_path)
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    _stub_unrelated_git_calls(monkeypatch)

    exit_calls: list = []
    monkeypatch.setattr(m.sys, "exit", _patched_exit(exit_calls))

    def fake_run(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        if "fetch" in cmd and "origin" in cmd:
            raise subprocess.TimeoutExpired(cmd="git fetch", timeout=kwargs.get("timeout"))
        return _ok_result(stdout="main")

    monkeypatch.setattr(m.subprocess, "run", fake_run)

    import argparse
    args = argparse.Namespace(
        target_branch=None,
        assume_yes=True,
        gateway=False,
        discard_local_changes=False,
        zip_update=False,
        yes=False,
        no=False,
        check=False,
    )

    with pytest.raises(SystemExit):
        m._cmd_update_impl(args, gateway_mode=False)

    captured = capsys.readouterr()
    assert "Network timeout" in captured.out, (
        f"expected 'Network timeout' in output, got: {captured.out!r}"
    )
    assert exit_calls == [1], (
        f"expected sys.exit(1) on network timeout, got exit_calls={exit_calls!r}"
    )


def test_update_origin_fetch_passes_timeout(monkeypatch, tmp_path):
    """The origin fetch call must include a timeout kwarg."""
    _make_fake_git_repo(tmp_path)
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    _stub_unrelated_git_calls(monkeypatch)
    monkeypatch.setattr(m.sys, "exit", _patched_exit([]))

    captured_kwargs: dict = {}

    def fake_run(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        if "fetch" in cmd and "origin" in cmd:
            captured_kwargs.update(kwargs)
            raise subprocess.TimeoutExpired(cmd="git fetch", timeout=kwargs.get("timeout"))
        return _ok_result(stdout="main")

    monkeypatch.setattr(m.subprocess, "run", fake_run)

    import argparse
    args = argparse.Namespace(
        target_branch=None,
        assume_yes=True,
        gateway=False,
        discard_local_changes=False,
        zip_update=False,
        yes=False,
        no=False,
        check=False,
    )

    with pytest.raises(SystemExit):
        m._cmd_update_impl(args, gateway_mode=False)

    assert "timeout" in captured_kwargs, (
        f"origin fetch must be called with a timeout, got kwargs={captured_kwargs!r}"
    )
    assert 10 <= captured_kwargs["timeout"] <= 120, (
        f"timeout must be a reasonable duration, got {captured_kwargs['timeout']!r}"
    )


# ---------------------------------------------------------------------------
# _cmd_update_impl — origin pull (line ~8830)
# ---------------------------------------------------------------------------


def test_update_origin_pull_handles_timeout(monkeypatch, tmp_path, capsys):
    """_cmd_update_impl must catch TimeoutExpired on the origin pull
    and exit cleanly without leaving the working tree in a partial state."""
    _make_fake_git_repo(tmp_path)
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    _stub_unrelated_git_calls(monkeypatch)

    exit_calls: list = []
    monkeypatch.setattr(m.sys, "exit", _patched_exit(exit_calls))

    def fake_run(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        if "fetch" in cmd and "origin" in cmd:
            return _ok_result()  # fetch succeeds
        if "rev-parse" in cmd and "--abbrev-ref" in cmd:
            return _ok_result(stdout="main")
        if "rev-list" in cmd:
            return _ok_result(stdout="1")  # 1 commit behind → enter pull
        if "pull" in cmd and "origin" in cmd:
            raise subprocess.TimeoutExpired(cmd="git pull", timeout=kwargs.get("timeout"))
        return _ok_result()

    monkeypatch.setattr(m.subprocess, "run", fake_run)

    import argparse
    args = argparse.Namespace(
        target_branch=None,
        assume_yes=True,
        gateway=False,
        discard_local_changes=False,
        zip_update=False,
        yes=False,
        no=False,
        check=False,
    )

    with pytest.raises(SystemExit):
        m._cmd_update_impl(args, gateway_mode=False)

    captured = capsys.readouterr()
    assert "Network timeout" in captured.out, (
        f"expected 'Network timeout' on pull failure, got: {captured.out!r}"
    )
    assert exit_calls == [1], (
        f"expected sys.exit(1) on pull timeout, got exit_calls={exit_calls!r}"
    )
