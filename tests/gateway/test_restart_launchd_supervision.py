"""Tests for launchd-aware /restart routing (incident 2026-06-11).

macOS launchd does not set INVOCATION_ID, so ``/restart`` historically took
the detached path under launchd: the supervised process exited 0,
``KeepAlive.SuccessfulExit=false`` treated that as deliberate, and the
detached replacement ran unsupervised — first crash left the gateway down
with ``last exit code = 0``.  ``_gateway_supervised_by_launchd`` closes the
detection gap; these tests cover the probe and the handler's path choice.
"""

import os
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import gateway.run as gateway_run
import gateway.slash_commands as slash_commands
import hermes_cli.gateway as hermes_cli_gateway
from gateway.platforms.base import MessageEvent, MessageType
from tests.gateway.restart_test_helpers import (
    make_restart_runner,
    make_restart_source,
)


def _fake_launchctl(stdout: str, returncode: int = 0):
    def _run(argv, capture_output=True, text=True, timeout=5):
        assert argv[0] == "launchctl" and argv[1] == "print"
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")

    return _run


def _patch_plist_path(monkeypatch, tmp_path):
    monkeypatch.setattr(
        hermes_cli_gateway,
        "get_launchd_plist_path",
        lambda: tmp_path / "ai.hermes.gateway.plist",
    )


# ── _gateway_supervised_by_launchd probe ─────────────────────────────────


def test_probe_true_when_launchd_holds_our_pid(tmp_path, monkeypatch):
    monkeypatch.setattr(slash_commands.sys, "platform", "darwin")
    _patch_plist_path(monkeypatch, tmp_path)
    monkeypatch.setattr(
        subprocess, "run", _fake_launchctl(f"\tstate = running\n\tpid = {os.getpid()}\n")
    )

    assert slash_commands._gateway_supervised_by_launchd() is True


def test_probe_false_when_launchd_holds_other_pid(tmp_path, monkeypatch):
    monkeypatch.setattr(slash_commands.sys, "platform", "darwin")
    _patch_plist_path(monkeypatch, tmp_path)
    monkeypatch.setattr(
        subprocess, "run", _fake_launchctl("\tstate = running\n\tpid = 99999999\n")
    )

    assert slash_commands._gateway_supervised_by_launchd() is False


def test_probe_false_when_service_not_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(slash_commands.sys, "platform", "darwin")
    _patch_plist_path(monkeypatch, tmp_path)
    monkeypatch.setattr(subprocess, "run", _fake_launchctl("", returncode=113))

    assert slash_commands._gateway_supervised_by_launchd() is False


def test_probe_false_when_no_pid_line(tmp_path, monkeypatch):
    """Loaded-but-stopped service prints no pid line — must not match."""
    monkeypatch.setattr(slash_commands.sys, "platform", "darwin")
    _patch_plist_path(monkeypatch, tmp_path)
    monkeypatch.setattr(
        subprocess, "run", _fake_launchctl("\tstate = not running\n\tlast exit code = 0\n")
    )

    assert slash_commands._gateway_supervised_by_launchd() is False


def test_probe_false_on_launchctl_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(slash_commands.sys, "platform", "darwin")
    _patch_plist_path(monkeypatch, tmp_path)

    def _boom(*args, **kwargs):
        raise FileNotFoundError("launchctl")

    monkeypatch.setattr(subprocess, "run", _boom)

    assert slash_commands._gateway_supervised_by_launchd() is False


def test_probe_false_on_non_darwin(monkeypatch):
    monkeypatch.setattr(slash_commands.sys, "platform", "linux")

    assert slash_commands._gateway_supervised_by_launchd() is False


# ── /restart handler path choice ─────────────────────────────────────────


def _make_restart_event():
    return MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="42"),
        message_id="m1",
    )


@pytest.mark.asyncio
async def test_restart_uses_service_path_when_launchd_supervised(tmp_path, monkeypatch):
    """launchd-supervised gateway must exit via the service path (75 → relaunch)."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.setattr(slash_commands, "_gateway_supervised_by_launchd", lambda: True)

    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)

    await runner._handle_restart_command(_make_restart_event())
    runner.request_restart.assert_called_once_with(detached=False, via_service=True)


@pytest.mark.asyncio
async def test_restart_uses_detached_path_when_unsupervised(tmp_path, monkeypatch):
    """No systemd, no launchd supervision → historical detached behavior."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.setattr(slash_commands, "_gateway_supervised_by_launchd", lambda: False)

    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)

    await runner._handle_restart_command(_make_restart_event())
    runner.request_restart.assert_called_once_with(detached=True, via_service=False)
