"""Regression tests for authoritative gateway PID discovery on Windows."""

from types import SimpleNamespace

import sys

import hermes_cli.gateway as gateway


def test_run_gateway_claims_runtime_lock_before_importing_runner(monkeypatch):
    monkeypatch.setattr(gateway, "_guard_official_docker_root_gateway", lambda: None)
    monkeypatch.setattr(
        gateway, "_guard_named_profile_under_multiplexer", lambda force=False: None
    )
    monkeypatch.setattr(
        gateway, "_guard_supervised_gateway_conflict", lambda force=False: None
    )
    monkeypatch.setattr(
        gateway,
        "_guard_existing_gateway_process_conflict",
        lambda replace=False: None,
    )
    monkeypatch.setattr(
        "gateway.status.acquire_gateway_runtime_lock", lambda: False
    )
    monkeypatch.delitem(sys.modules, "gateway.run", raising=False)

    assert gateway.run_gateway() is False
    assert "gateway.run" not in sys.modules


def test_run_gateway_releases_early_lock_on_clean_exit(monkeypatch):
    monkeypatch.setattr(gateway, "_guard_official_docker_root_gateway", lambda: None)
    monkeypatch.setattr(
        gateway, "_guard_named_profile_under_multiplexer", lambda force=False: None
    )
    monkeypatch.setattr(
        gateway, "_guard_supervised_gateway_conflict", lambda force=False: None
    )
    monkeypatch.setattr(
        gateway,
        "_guard_existing_gateway_process_conflict",
        lambda replace=False: None,
    )
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)
    monkeypatch.setattr("gateway.status.acquire_gateway_runtime_lock", lambda: True)
    released = []
    monkeypatch.setattr(
        "gateway.status.release_gateway_runtime_lock", lambda: released.append(True)
    )
    monkeypatch.setattr(gateway.asyncio, "run", lambda coro: (coro.close(), True)[1])
    monkeypatch.setattr(
        "gateway.run._exit_after_graceful_shutdown", lambda code: None
    )

    gateway.run_gateway()

    assert released == [True]


def test_find_gateway_pids_passes_profile_pid_as_authoritative_owner(monkeypatch):
    calls = []

    monkeypatch.setattr(gateway, "_get_service_pids", lambda: set())
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: 47688)
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)

    def fake_scan(
        exclude_pids,
        all_profiles=False,
        include_restart_managers=False,
        authoritative_pid=None,
    ):
        calls.append(authoritative_pid)
        return [authoritative_pid]

    monkeypatch.setattr(gateway, "_scan_gateway_pids", fake_scan)

    assert gateway.find_gateway_pids() == [47688]
    assert calls == [47688]


def test_scan_gateway_pids_windows_ignores_non_owner_when_pid_file_is_live(
    monkeypatch,
):
    monkeypatch.setattr(gateway, "is_windows", lambda: True)
    monkeypatch.setattr(gateway, "_get_ancestor_pids", lambda: set())
    monkeypatch.setattr(
        gateway.shutil,
        "which",
        lambda name: "wmic.exe" if name == "wmic" else None,
    )

    def fake_run(cmd, **kwargs):
        if cmd[:4] == ["wmic.exe", "process", "get", "ProcessId,CommandLine"]:
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "CommandLine=pythonw.exe -m hermes_cli.main gateway run\n"
                    "ProcessId=3856\n\n"
                    "CommandLine=pythonw.exe -m hermes_cli.main gateway run\n"
                    "ProcessId=47688\n\n"
                    "CommandLine=pythonw.exe -m hermes_cli.main gateway run\n"
                    "ProcessId=51960\n\n"
                ),
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(gateway.subprocess, "run", fake_run)

    assert gateway._scan_gateway_pids(
        set(), authoritative_pid=47688
    ) == [47688]


def test_scan_gateway_pids_all_profiles_remains_broad(monkeypatch):
    monkeypatch.setattr(gateway, "is_windows", lambda: True)
    monkeypatch.setattr(gateway, "_get_ancestor_pids", lambda: set())
    monkeypatch.setattr(
        gateway.shutil,
        "which",
        lambda name: "wmic.exe" if name == "wmic" else None,
    )
    monkeypatch.setattr(
        gateway.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                "CommandLine=pythonw.exe -m hermes_cli.main gateway run\n"
                "ProcessId=3856\n\n"
                "CommandLine=pythonw.exe -m hermes_cli.main gateway run\n"
                "ProcessId=47688\n\n"
            ),
            stderr="",
        ),
    )

    assert gateway._scan_gateway_pids(
        set(), all_profiles=True, authoritative_pid=47688
    ) == [3856, 47688]
