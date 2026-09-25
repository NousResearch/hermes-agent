"""Tests for ``uninstall_gateway_service`` — the per-platform service dispatch.

The dispatch is table-driven (``_GATEWAY_SERVICE_REMOVERS``, keyed by
``platform.system()``): exactly the current host's remover runs, a remover
failure is reported with the table's warning label, and a platform with no
entry (tested here via an empty table) cleans up nothing. The tests run on the
actual host — no OS faking — and all removers plus the standalone-process kill
are stubbed, so no real service is stopped and no real gateway process is
killed.
"""
from __future__ import annotations

import platform

import pytest

import hermes_cli.gateway as gateway
import hermes_cli.uninstall as uninstall


@pytest.fixture
def no_live_gateways(monkeypatch: pytest.MonkeyPatch) -> None:
    """The standalone-process sweep finds nothing (and must not kill)."""
    monkeypatch.setattr(gateway, "find_gateway_pids", lambda *a, **k: [])
    monkeypatch.setattr(
        gateway, "kill_gateway_processes",
        lambda *a, **k: pytest.fail("kill_gateway_processes must not run without live pids"))
    # Neutral env in case the CI runner exports Termux variables; this host
    # always reaches the dispatch table.
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    monkeypatch.setenv("PREFIX", "")


def _dispatch_table(calls: list) -> dict:
    """A remover table that records which platform's remover ran."""
    def remover(name):
        def run():
            calls.append(name)
            return True
        return run
    return {
        "Linux": (remover("Linux"), "linux label"),
        "Darwin": (remover("Darwin"), "darwin label"),
        "Windows": (remover("Windows"), "windows label"),
    }


def test_current_platform_remover_runs_and_others_do_not(monkeypatch, no_live_gateways):
    calls: list = []
    monkeypatch.setattr(uninstall, "_GATEWAY_SERVICE_REMOVERS", _dispatch_table(calls))

    assert uninstall.uninstall_gateway_service() is True
    assert calls == [platform.system()]


def test_remover_failure_reports_the_table_label(monkeypatch, no_live_gateways, capsys):
    table = _dispatch_table([])
    def boom():
        raise RuntimeError("no service manager here")
    table[platform.system()] = (boom, "remover blew up")
    monkeypatch.setattr(uninstall, "_GATEWAY_SERVICE_REMOVERS", table)

    assert uninstall.uninstall_gateway_service() is False
    assert "remover blew up" in capsys.readouterr().out


def test_unknown_platform_runs_no_remover(monkeypatch, no_live_gateways):
    monkeypatch.setattr(uninstall, "_GATEWAY_SERVICE_REMOVERS", {})

    assert uninstall.uninstall_gateway_service() is False
