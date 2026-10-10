"""Churn hooks at the refactored gateway startup boundary."""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway import churn, host_attach, run, status
from gateway.config import GatewayConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("takeover", ["none", "profile", "host"])
@pytest.mark.parametrize("write_result", [True, False, OSError("unwritable journal")])
async def test_churn_after_adapters_before_cron(
    tmp_path, monkeypatch, caplog, takeover, write_result
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv(churn.GATEWAY_CHURN_PATH_ENV, str(tmp_path / "churn.jsonl"))
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    events = []
    old_pid = os.getpid() + 10_000
    decision = SimpleNamespace(
        outcome=host_attach.REPLACE_HOST if takeover == "host" else host_attach.START,
        owner=SimpleNamespace(pid=old_pid),
    )
    monkeypatch.setattr(host_attach, "decide", lambda *a, **kw: decision)
    monkeypatch.setattr(status, "get_running_pid", lambda: old_pid if takeover == "profile" else None)
    monkeypatch.setattr(run, "_start_gateway_replace_existing_instance", AsyncMock(return_value=True))
    monkeypatch.setattr("hermes_cli.resource_limits.apply_nofile_soft_limit", lambda: None)
    monkeypatch.setattr("gateway.code_skew.record_boot_fingerprint", lambda: None)
    monkeypatch.setattr("gateway.run_startup.recover_left_core_at_gateway_start", lambda: None)
    for name in (
        "_start_gateway_configure_logging", "_enable_multiplex_log_routing",
        "_refresh_host_gateway_record", "_log_standalone_profiles_at_boot",
        "_ensure_windows_gateway_venv_imports", "_best_effort",
    ):
        monkeypatch.setattr(run, name, Mock())
    monkeypatch.setattr(run, "_start_gateway_claim_pid_file", lambda **kw: events.append("pid") or True)
    monkeypatch.setattr(run, "_start_gateway_start_control_socket", AsyncMock(return_value=None))
    monkeypatch.setattr(run, "_discover_gateway_mcp_tools", AsyncMock())
    monkeypatch.setattr(run, "_run_planned_stop_watcher", lambda *a: None)
    monkeypatch.setattr(asyncio.get_running_loop(), "add_signal_handler", Mock())

    async def start():
        events.append("adapters")
        return True

    runner = SimpleNamespace(
        config=GatewayConfig(), adapters={}, _running=True, should_exit_cleanly=False,
        start=start, wait_for_shutdown=AsyncMock(), _start_systemd_watchdog=Mock(),
    )
    monkeypatch.setattr(run, "GatewayRunner", lambda config: runner)
    monkeypatch.setattr(run, "_start_gateway_make_shutdown_signal_handler", lambda *a: Mock())
    monkeypatch.setattr(run, "_start_gateway_make_restart_signal_handler", lambda *a: Mock())
    monkeypatch.setattr(run, "_start_gateway_shutdown_tail", AsyncMock(return_value=True))

    def append(event_type, *, pid_old, pid_new):
        events.append((event_type, pid_old, pid_new))
        if isinstance(write_result, Exception):
            raise write_result
        return write_result

    monkeypatch.setattr(churn, "append_gateway_churn_event", append)
    monkeypatch.setattr(
        run, "_start_gateway_start_cron_and_housekeeping",
        lambda runner: events.append("cron") or (None, None, None, None),
    )

    assert await run.start_gateway(config=runner.config, replace=takeover != "none", verbosity=None)
    assert events == [
        "pid", "adapters",
        ("start" if takeover == "none" else "replace",
         None if takeover == "none" else old_pid, os.getpid()),
        "cron",
    ]
    warnings = [record.message for record in caplog.records if "Configured gateway churn" in record.message]
    if write_result is True:
        assert warnings == []
    elif isinstance(write_result, Exception):
        assert warnings == ["Configured gateway churn hook failed: unwritable journal"]
    else:
        assert warnings == ["Configured gateway churn event could not be recorded"]
