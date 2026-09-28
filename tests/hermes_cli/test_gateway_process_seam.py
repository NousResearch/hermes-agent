"""Keep process helpers reachable and patchable through the gateway facade."""

from types import FunctionType, SimpleNamespace

import pytest

from hermes_cli import gateway, gateway_process


def test_moved_process_functions_keep_facade_identity():
    moved = {
        name: value
        for name, value in vars(gateway_process).items()
        if isinstance(value, FunctionType) and value.__module__ == gateway_process.__name__
    }

    assert moved
    for name, value in moved.items():
        assert getattr(gateway, name) is value
        assert all(not isinstance(annotation, str) for annotation in value.__annotations__.values())

    for name in (
        "GATEWAY_LOOP_ALIVE",
        "GATEWAY_LOOP_WEDGED",
        "GATEWAY_LOOP_UNKNOWN",
        "DEFAULT_LOOP_LIVENESS_STALE_AFTER_S",
        "_LOOP_TICK_ABSENT",
    ):
        assert getattr(gateway, name) is getattr(gateway_process, name)


def test_moved_pid_finder_reads_facade_patches(monkeypatch):
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr(gateway, "_get_service_pids", lambda **_kw: {123})
    monkeypatch.setattr(gateway, "_scan_gateway_pids", lambda *_a, **_kw: [456])
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)

    assert gateway.find_gateway_pids() == [123, 456]


@pytest.mark.linux_only
def test_moved_process_scan_honors_facade_pid_admission(monkeypatch):
    admitted = []

    def unavailable_proc(_excluded):
        raise OSError("exercise the ps fallback")

    monkeypatch.setattr(gateway, "_get_ancestor_pids", lambda: set())
    monkeypatch.setattr(gateway, "_iter_proc_cmdlines", unavailable_proc)
    monkeypatch.setattr(
        gateway.subprocess,
        "run",
        lambda *_a, **_kw: SimpleNamespace(
            returncode=0, stdout="999999 python -m hermes_cli.main gateway run\n"
        ),
    )
    monkeypatch.setattr(
        gateway, "_append_unique_pid", lambda *args: admitted.append(args)
    )

    # The facade's admission seam rejects the candidate; moving the scanner
    # must not bypass that decision by reading the sibling's original helper.
    assert gateway._scan_gateway_pids(set(), all_profiles=True) == []
    assert len(admitted) == 1
    assert admitted[0][1:] == (999999, set())


def test_moved_loop_probe_reads_facade_witness(monkeypatch):
    monkeypatch.setattr(gateway, "_probe_loop_tick_socket", lambda *_a, **_kw: True)

    assert gateway._probe_loop_tick_socket_sustained(123, None) is True


def test_moved_restart_fallback_reads_facade_launchers(monkeypatch):
    calls = []
    argv = ["python", "-m", "hermes_cli.main", "gateway", "run"]
    monkeypatch.setattr(gateway, "_capture_gateway_argv", lambda _pid: argv)
    monkeypatch.setattr(gateway, "launch_detached_profile_gateway_restart", lambda *_a: False)
    monkeypatch.setattr(
        gateway,
        "launch_detached_gateway_restart_by_cmdline",
        lambda pid, args: calls.append((pid, args)) or True,
    )

    assert gateway._prepare_profile_gateway_update_restart("default", 123) == "detached-cmdline"
    assert calls == [(123, argv)]
