"""Gateway --replace must stop dashboard before the old gateway exits.

A live dashboard holding deleted state.db-wal/shm is the DeletedWalGenerationError
latch. These tests pin the quiesce helper and fail-closed wait.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import gateway.session_db_writers as writers


def test_hold_supervised_dashboard_down_uses_s6_svc_d(monkeypatch, tmp_path):
    service = tmp_path / "dashboard"
    (service / "supervise").mkdir(parents=True)
    (service / "run").write_text("#!/bin/sh\n")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        result = MagicMock()
        result.returncode = 0
        result.stderr = b""
        result.stdout = b""
        return result

    monkeypatch.setattr(writers, "_DASHBOARD_SERVICE_CANDIDATES", (service,))
    monkeypatch.setattr(writers.subprocess, "run", fake_run)

    held = writers.hold_supervised_dashboard_down()
    assert held == [service]
    assert calls == [["s6-svc", "-d", str(service)]]


def test_quiesce_signals_dashboard_after_s6_hold(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(writers, "hold_supervised_dashboard_down", lambda: [tmp_path / "svc"])
    monkeypatch.setattr(writers, "iter_same_home_dashboard_pids", lambda hermes_home, except_pids=(): [4242])
    signalled = []
    monkeypatch.setattr(writers, "terminate_pids", lambda pids, force=False: signalled.append((list(pids), force)))

    held = writers.quiesce_session_db_writers_for_replace(home)
    assert held == [tmp_path / "svc"]
    assert signalled == [([4242], False)]


def test_wait_deleted_sidecar_holders_gone_true_when_empty(monkeypatch, tmp_path):
    import sys
    fake = MagicMock()
    fake.iter_deleted_sqlite_sidecar_holders = lambda path: []
    monkeypatch.setitem(sys.modules, "hermes_state_dbfile", fake)
    assert writers.wait_deleted_sidecar_holders_gone(tmp_path / "state.db", timeout_s=0.2) is True


def test_wait_deleted_sidecar_holders_gone_false_when_stuck(monkeypatch, tmp_path):
    import sys
    fake = MagicMock()
    fake.iter_deleted_sqlite_sidecar_holders = lambda path: [(99, "state.db-wal (deleted)")]
    monkeypatch.setitem(sys.modules, "hermes_state_dbfile", fake)
    assert writers.wait_deleted_sidecar_holders_gone(tmp_path / "state.db", timeout_s=0.2) is False


def test_fail_closed_if_deleted_holders_aborts_replace(monkeypatch, tmp_path):
    monkeypatch.setattr(writers, "wait_deleted_sidecar_holders_gone", lambda db_path, timeout_s=10.0: False)
    assert writers.fail_closed_if_deleted_holders(tmp_path) is False


def test_fail_closed_if_deleted_holders_allows_open(monkeypatch, tmp_path):
    monkeypatch.setattr(writers, "wait_deleted_sidecar_holders_gone", lambda db_path, timeout_s=10.0: True)
    assert writers.fail_closed_if_deleted_holders(tmp_path) is True


def test_wait_deleted_sidecar_holders_gone_false_when_scanner_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(writers, "_deleted_sidecar_holder_iter", lambda: None)
    assert writers.wait_deleted_sidecar_holders_gone(tmp_path / "state.db") is False


def test_hold_supervised_dashboard_ignores_run_only_dir(monkeypatch, tmp_path):
    service = tmp_path / "dashboard"
    service.mkdir()
    (service / "run").write_text("#!/bin/sh\n")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        result = MagicMock()
        result.returncode = 0
        result.stderr = b""
        result.stdout = b""
        return result

    monkeypatch.setattr(writers, "_DASHBOARD_SERVICE_CANDIDATES", (service,))
    monkeypatch.setattr(writers.subprocess, "run", fake_run)
    assert writers.hold_supervised_dashboard_down() == []
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("holders_gone", [False, True])
async def test_replace_dashboard_release_requires_clean_holders(monkeypatch, tmp_path, holders_gone):
    from unittest.mock import AsyncMock
    from gateway import code_skew, run, run_bootstrap as bootstrap, status
    from gateway.config import GatewayConfig
    from hermes_cli import resource_limits

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(bootstrap, "_start_gateway_resolve_terminal_cwd", lambda: None)
    monkeypatch.setattr(bootstrap, "_start_gateway_early_environment_probe", lambda: None)
    monkeypatch.setattr(resource_limits, "apply_nofile_soft_limit", lambda: None)
    monkeypatch.setattr(code_skew, "record_boot_fingerprint", lambda: None)
    monkeypatch.setattr(run, "_host_attach_or_none", AsyncMock(return_value=None))
    monkeypatch.setattr(status, "get_running_pid", lambda: None)
    calls = []
    held = [tmp_path / "dashboard"]
    monkeypatch.setattr(writers, "quiesce_session_db_writers_for_replace",
                        lambda home: calls.append("held") or held)
    monkeypatch.setattr(writers, "fail_closed_if_deleted_holders",
                        lambda home: calls.append("checked") or holders_gone)
    monkeypatch.setattr(writers, "release_supervised_dashboard",
                        lambda services: calls.append(("released", services)))
    monkeypatch.setattr(bootstrap, "_start_gateway_reserve_profiles",
                        lambda homes: calls.append("reserved") or False)
    assert await bootstrap.start_gateway(GatewayConfig(), replace=True) is False
    expected = ["held", "checked"]
    if holders_gone:
        expected += ["reserved", ("released", held)]
    assert calls == expected
