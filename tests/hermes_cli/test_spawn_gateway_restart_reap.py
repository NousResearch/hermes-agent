"""The gateway-restart flow refuses BEFORE it stops anything (#125394, #77276).

The Desktop backend's ``_spawn_gateway_restart`` used to reap "orphan" gateways
before spawning ``hermes gateway restart``; when that child then refused (a
named profile under the host multiplexer) the profile had already lost its
gateway. The reap belongs to the child, after its refusal guard.
"""
from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import gateway as gw


@pytest.fixture(autouse=True)
def reset_restart_cooldown():
    import hermes_cli.web_server as web_server

    web_server._LAST_GATEWAY_RESTART = None
    yield
    web_server._LAST_GATEWAY_RESTART = None


@patch("hermes_cli.web_server_gateway._gateway_subcommand", return_value=["gateway", "restart"])
@patch("hermes_cli.web_server_gateway._spawn_hermes_action")
@patch("hermes_cli.web_server_gateway._ACTION_PROCS", {})
def test_backend_spawns_restart_without_stopping_anything(mock_spawn, mock_subcmd):
    mock_proc = MagicMock(spec=subprocess.Popen)
    mock_proc.poll.return_value = None
    mock_spawn.return_value = mock_proc

    from hermes_cli.web_server import _spawn_gateway_restart

    with patch("hermes_cli.gateway._reap_unsupervised_gateway_orphans") as reap:
        proc, reused = _spawn_gateway_restart()

    reap.assert_not_called()
    mock_spawn.assert_called_once()
    assert proc is mock_proc and not reused


def test_cli_restart_reaps_after_the_refusal_guard(monkeypatch):
    order: list[str] = []
    monkeypatch.setattr(gw, "_refuse_from_inside_gateway", lambda *a, **k: None)
    monkeypatch.setattr(gw, "_guard_named_profile_under_multiplexer", lambda **k: order.append("guard"))
    monkeypatch.setattr(gw, "_dispatch_via_service_manager_if_s6", lambda *a, **k: False)
    monkeypatch.setattr(gw, "_installed_service_kind_for", lambda windows: "launchd")
    monkeypatch.setattr(gw, "_reap_unsupervised_gateway_orphans", lambda *a, **k: order.append("reap"))
    monkeypatch.setattr(gw, "_service_call", lambda *a, **k: order.append("restart"))

    gw._cmd_restart(SimpleNamespace(system=False, all=False, force=False))

    assert order == ["guard", "reap", "restart"]


def test_cli_restart_reap_failure_does_not_block_restart(monkeypatch):
    """The reap is best-effort: a scan error must not abort the restart it precedes."""
    order: list[str] = []
    monkeypatch.setattr(gw, "_refuse_from_inside_gateway", lambda *a, **k: None)
    monkeypatch.setattr(gw, "_guard_named_profile_under_multiplexer", lambda **k: None)
    monkeypatch.setattr(gw, "_dispatch_via_service_manager_if_s6", lambda *a, **k: False)
    monkeypatch.setattr(gw, "_installed_service_kind_for", lambda windows: "launchd")

    def _boom(*a, **k):
        raise OSError("permission denied")

    monkeypatch.setattr(gw, "_reap_unsupervised_gateway_orphans", _boom)
    monkeypatch.setattr(gw, "_service_call", lambda *a, **k: order.append("restart"))

    gw._cmd_restart(SimpleNamespace(system=False, all=False, force=False))

    assert order == ["restart"]
