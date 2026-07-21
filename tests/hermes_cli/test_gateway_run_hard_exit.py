"""Regression tests for CLI gateway run exit behavior.

``hermes gateway run`` enters through hermes_cli.gateway, not gateway.run.main().
After graceful teardown it must use the same hard-exit backstop as gateway.run.main()
so Python finalization does not wait on non-daemon worker threads (for example
in-flight cron ThreadPoolExecutor jobs) and delay service-managed restarts.
"""

from __future__ import annotations

import types

import pytest


class _HardExitObserved(BaseException):
    def __init__(self, code: int):
        super().__init__(code)
        self.code = code


def _prepare(monkeypatch):
    import hermes_cli.gateway as gateway_cli
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_cli, "_guard_official_docker_root_gateway", lambda: None)
    monkeypatch.setattr(gateway_cli, "_guard_named_profile_under_multiplexer", lambda force=False: None)
    monkeypatch.setattr(gateway_cli, "_guard_supervised_gateway_conflict", lambda force=False: None)
    monkeypatch.setattr(gateway_cli, "_guard_existing_gateway_process_conflict", lambda replace=False: None)
    monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(gateway_cli.sys, "stdin", types.SimpleNamespace(isatty=lambda: False))
    monkeypatch.setenv("HERMES_GATEWAY_EXIT_DIAG", "0")

    async def _start_gateway(*args, **kwargs):  # pragma: no cover - never awaited by fake run
        return True

    def _hard_exit(code: int) -> None:
        raise _HardExitObserved(code)

    monkeypatch.setattr(gateway_run, "start_gateway", _start_gateway)
    monkeypatch.setattr(gateway_run, "_exit_after_graceful_shutdown", _hard_exit)
    return gateway_cli


def test_run_gateway_hard_exits_after_clean_return(monkeypatch):
    gateway_cli = _prepare(monkeypatch)

    def _fake_run(coro):
        coro.close()
        return True

    monkeypatch.setattr(gateway_cli.asyncio, "run", _fake_run)

    with pytest.raises(_HardExitObserved) as excinfo:
        gateway_cli.run_gateway()

    assert excinfo.value.code == 0


def test_run_gateway_hard_exits_after_service_restart_systemexit(monkeypatch):
    gateway_cli = _prepare(monkeypatch)

    def _fake_run(coro):
        coro.close()
        raise SystemExit(75)

    monkeypatch.setattr(gateway_cli.asyncio, "run", _fake_run)

    with pytest.raises(_HardExitObserved) as excinfo:
        gateway_cli.run_gateway()

    assert excinfo.value.code == 75


def test_run_gateway_hard_exits_after_failed_return(monkeypatch):
    gateway_cli = _prepare(monkeypatch)

    def _fake_run(coro):
        coro.close()
        return False

    monkeypatch.setattr(gateway_cli.asyncio, "run", _fake_run)

    with pytest.raises(_HardExitObserved) as excinfo:
        gateway_cli.run_gateway()

    assert excinfo.value.code == 1
