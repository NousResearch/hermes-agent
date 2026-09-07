"""Regression for #104478 — _run_pending_fleet_restart discarded restart bookkeeping.

``_run_pending_fleet_restart()`` (pending fleet catch-up path) called
``_restart_macos_launchd_gateways([], failed, ...)`` with a literal empty list
for the ``restarted_services`` accumulator instead of a mutable list.
Every successful label was silently discarded, reconciliation saw ``unaccounted``,
and ``hermes update`` exited 1 even after a clean restart → Desktop hand-off
retry → double-restart loop.

The function mutates ``restarted_services`` in place (append on success).
This test pins that the pending-restart path MUST now pass a local accumulator
so reconciliation can read it after the call returns.
"""

from __future__ import annotations

import pytest

import hermes_cli.gateway as gw
from hermes_cli.update_cmd_fleet import _run_pending_fleet_restart


pytestmark = pytest.mark.skipif(
    __import__("sys").platform == "win32",
    reason="launchd fleet restart is macOS-only",
)


@pytest.fixture
def _macos_env(monkeypatch):
    """Minimal macOS environment stubs so the pending restart path doesn't exit early."""
    monkeypatch.setattr(gw, "is_macos", lambda: True)
    monkeypatch.setattr(gw, "is_windows", lambda: False)
    monkeypatch.setattr(gw, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(gw, "find_gateway_pids", lambda **kw: [999])
    monkeypatch.setattr(gw, "kill_gateway_processes", lambda **kw: None)
    monkeypatch.setattr(gw, "_wait_for_gateway_exit", lambda **kw: None)


@pytest.fixture(autouse=True)
def _no_stale_modules(monkeypatch):
    """update_cmd imports _purge_stale_hermes_modules — stub it to avoid module side effects."""
    import hermes_cli.update_cmd as update_cmd

    monkeypatch.setattr(update_cmd, "_m", lambda: __import__("types").SimpleNamespace(_purge_stale_hermes_modules=lambda: None))


def test_pending_restart_passes_mutable_restarted_list_to_launchd(monkeypatch, _macos_env):
    """The pending-restart path MUST collect restarted labels in a local list
    so future reconciliation paths (if any) can read them. The bug was passing
    a literal [] that discarded every append."""
    calls = []

    def fake_restart(restarted, failed, drain_budget):
        # Simulate a successful restart appending to the passed list.
        restarted.append("ai.hermes.gateway")
        restarted.append("ai.hermes.gateway-spock")
        calls.append({"restarted": restarted, "failed": failed})

    monkeypatch.setattr(
        "hermes_cli.update_cmd_fleet._restart_macos_launchd_gateways", fake_restart
    )
    # _run_pending_fleet_restart returns True on success (no failures).
    result = _run_pending_fleet_restart()
    assert result is True
    # The bug: _restart_macos_launchd_gateways was called with [], so the function
    # returned and the appended labels were lost. The fix ensures the list is kept.
    assert len(calls) == 1
    assert calls[0]["restarted"] == ["ai.hermes.gateway", "ai.hermes.gateway-spock"]
