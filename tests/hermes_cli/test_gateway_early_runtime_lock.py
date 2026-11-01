"""Ordinary ``gateway run`` starts claim the runtime lock before booting the runner.

Concurrent Windows ``pythonw … gateway run`` attempts all passed the PID-file preflight and
reached ``start_gateway``'s lock only inside ``asyncio.run``; a loser could stall in Proactor
loop construction first and linger as an orphan with a gateway argv.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types

import pytest

import hermes_cli.gateway as gateway_cli
from gateway import status
from gateway.runtime_ownership import OwnershipConflict, ProfileOwnership, process_ownership


class _HardExitObserved(BaseException):
    pass


class _BootReached(BaseException):
    pass


@pytest.fixture
def release_lock():
    yield
    status.release_gateway_runtime_lock()


def _stub_preflight(monkeypatch):
    for guard in (
        "_guard_official_docker_root_gateway",
        "_attach_to_host_gateway_or_guard",
        "_guard_supervised_gateway_conflict",
        "_guard_existing_gateway_process_conflict",
    ):
        monkeypatch.setattr(gateway_cli, guard, lambda *_a, **_kw: None)


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _conflict(_homes):
    raise OwnershipConflict("Gateway runtime already owns profile")


def test_lock_loser_exits_before_importing_runner(monkeypatch):
    _stub_preflight(monkeypatch)
    monkeypatch.setattr(process_ownership, "reserve", _conflict)

    def _booted_past_the_claim():
        # First step after the claim: fail here rather than boot (and os._exit) a real runner.
        raise _BootReached

    monkeypatch.setattr(gateway_cli, "_apply_startup_watchdog_config", _booted_past_the_claim)
    monkeypatch.delitem(sys.modules, "gateway.run", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        gateway_cli.run_gateway()

    assert excinfo.value.code == 1
    assert "gateway.run" not in sys.modules


def test_lock_is_held_before_the_runner_boots(monkeypatch, release_lock):
    import gateway.run as gateway_run
    from hermes_cli.observability import shared_metrics_process

    _stub_preflight(monkeypatch)
    owned = []
    monkeypatch.setattr(
        gateway_cli, "_apply_startup_watchdog_config", lambda: owned.append(status.owns_gateway_runtime_lock()))
    monkeypatch.setattr(gateway_cli, "_windows_gateway_should_absorb_console_controls", lambda: False)
    monkeypatch.setattr(gateway_cli, "_respawn_storm_backoff", lambda: None)
    monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(shared_metrics_process, "begin_process", lambda *_a, **_kw: None)
    monkeypatch.setattr(gateway_cli.sys, "stdin", types.SimpleNamespace(isatty=lambda: False))
    monkeypatch.setenv("HERMES_GATEWAY_EXIT_DIAG", "0")

    async def _start_gateway(*_a, **_kw):  # pragma: no cover - closed unawaited by the fake run
        return True

    def _hard_exit(_code: int) -> None:
        raise _HardExitObserved

    monkeypatch.setattr(gateway_run, "start_gateway", _start_gateway)
    monkeypatch.setattr(gateway_run, "_exit_after_graceful_shutdown", _hard_exit)

    def _fake_run(coro):
        owned.append(status.owns_gateway_runtime_lock())
        coro.close()
        return True

    monkeypatch.setattr(gateway_cli.asyncio, "run", _fake_run)

    with pytest.raises(_HardExitObserved):
        gateway_cli.run_gateway()

    assert owned == [True, True]  # before the watchdog arms, and when the loop is built


def test_replace_leaves_the_claim_to_start_gateway(monkeypatch):
    calls = []
    monkeypatch.setattr(process_ownership, "reserve", calls.append)

    gateway_cli._claim_gateway_runtime_lock_before_boot(replace=True)

    assert calls == []


@pytest.mark.parametrize("standalone, claims", [(False, False), (True, True)])
def test_named_profile_claims_only_when_standalone(monkeypatch, standalone, claims):
    import hermes_constants
    from hermes_cli import profiles

    calls = []
    monkeypatch.setattr(hermes_constants, "profile_name_for_home", lambda _home: "ops")
    monkeypatch.setattr(profiles, "profile_is_standalone", lambda _home: standalone)
    monkeypatch.setattr(process_ownership, "reserve", calls.append)

    gateway_cli._claim_gateway_runtime_lock_before_boot()

    # A multiplexer member's home is reserved by the default multiplexer too: start_gateway decides.
    assert bool(calls) is claims


def test_scope_probe_failure_leaves_the_claim_to_start_gateway(monkeypatch):
    from hermes_cli import profiles

    def _broken(_home):
        raise RuntimeError("profiles unreadable")

    calls = []
    monkeypatch.setattr(profiles, "profile_is_standalone", _broken)
    monkeypatch.setattr("hermes_constants.profile_name_for_home", lambda _home: "ops")
    monkeypatch.setattr(process_ownership, "reserve", calls.append)

    gateway_cli._claim_gateway_runtime_lock_before_boot()

    assert calls == []


def test_claim_clears_stale_pid_file_so_the_pid_claim_still_wins(release_lock):
    pid_path = status._get_pid_path()
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(json.dumps({**status._build_pid_record(), "pid": _dead_pid()}), encoding="utf-8")

    gateway_cli._claim_gateway_runtime_lock_before_boot()

    assert status.owns_gateway_runtime_lock()
    assert not pid_path.exists()
    # start_gateway's own claim: idempotent reserve, then the O_EXCL PID file.
    assert status.acquire_gateway_runtime_lock()
    status.write_pid_file()
    assert status._read_pid_record()["pid"] == os.getpid()
    status.remove_pid_file()


def test_second_starter_loses_to_the_lock_holder():
    holder = ProfileOwnership()
    holder.reserve([status._get_process_hermes_home()])
    try:
        with pytest.raises(SystemExit) as excinfo:
            gateway_cli._claim_gateway_runtime_lock_before_boot()
    finally:
        holder.close()

    assert excinfo.value.code == 1
    assert not status.owns_gateway_runtime_lock()


def test_unusable_lock_reports_its_own_error(monkeypatch, capsys):
    def _foreign(_homes):
        raise PermissionError("foreign_stale_lock: run `sudo rm gateway.lock`")

    monkeypatch.setattr(process_ownership, "reserve", _foreign)

    with pytest.raises(SystemExit) as excinfo:
        gateway_cli._claim_gateway_runtime_lock_before_boot()

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "foreign_stale_lock" in out
    assert "gateway status" not in out  # the recovery text is the hint
