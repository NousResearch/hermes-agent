"""Service-wrapper gateway descendant discovery regressions (#66900, #105938).

The recursive descendant design and strict command matching are adapted from PR #66913 by
@Tranquil-Flow. This current-main variant covers the all-profiles service discovery contract used
by update/reaper sweeps while preserving default profile scoping.
"""

from types import SimpleNamespace
from unittest.mock import patch

import psutil

import hermes_cli.gateway as gateway


class _FakeProc:
    def __init__(self, pid, cmdline=(), children=()):
        self.pid = pid
        self._cmdline = list(cmdline)
        self._children = list(children)

    def children(self, recursive=False):
        if not recursive:
            return list(self._children)
        found, pending = [], list(self._children)
        while pending:
            child = pending.pop()
            found.append(child)
            pending.extend(child._children)
        return found

    def cmdline(self):
        return list(self._cmdline)


def _gateway_proc(pid=503):
    return _FakeProc(
        pid,
        ["python", "-m", "hermes_cli.main", "--profile", "default", "gateway", "run", "--external-supervisor"],
    )


def test_gateway_descendants_walks_nested_wrappers_and_uses_strict_matcher(monkeypatch):
    gateway_child = _gateway_proc()
    misleading = _FakeProc(504, ["sh", "-c", "echo gateway run"])
    shell = _FakeProc(502, ["sh", "-c", "exec hermes"], [gateway_child, misleading])
    wrapper = _FakeProc(501, ["python", "-m", "hermes_cli.stderr_timestamp"], [shell])
    monkeypatch.setattr(psutil, "Process", lambda pid: wrapper)

    assert gateway._gateway_descendants_of(501) == {503}


def test_gateway_descendants_tolerates_vanished_wrapper(monkeypatch):
    def missing(pid):
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(psutil, "Process", missing)
    assert gateway._gateway_descendants_of(501) == set()


def test_systemd_service_pid_includes_recursive_gateway_descendant(monkeypatch):
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(gateway, "is_macos", lambda: False)
    monkeypatch.setattr(gateway, "get_service_name", lambda: "hermes-gateway.service")
    monkeypatch.setattr(gateway, "_gateway_descendants_of", lambda pid: {503})

    def run(args, **kwargs):
        if "list-units" in args:
            return SimpleNamespace(returncode=0, stdout="hermes-gateway.service loaded active running\n", stderr="")
        if "show" in args:
            return SimpleNamespace(returncode=0, stdout="501\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gateway.subprocess, "run", run)
    assert gateway._get_service_pids() == {501, 503}


def test_launchd_default_scope_includes_recursive_gateway_descendant(monkeypatch):
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(gateway, "is_macos", lambda: True)
    monkeypatch.setattr(gateway, "get_launchd_label", lambda: "ai.hermes.gateway.default")
    monkeypatch.setattr(gateway, "_locate_launchd_gateway_service", lambda label: ("gui/501", 501))
    monkeypatch.setattr(gateway, "_gateway_descendants_of", lambda pid: {503})

    assert gateway._get_service_pids() == {501, 503}


def test_launchd_fleet_prefix_scan_expands_each_unmapped_wrapper(monkeypatch):
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(gateway, "is_macos", lambda: True)
    monkeypatch.setattr(gateway, "get_launchd_label", lambda: "ai.hermes.gateway.default")
    monkeypatch.setattr(gateway, "launchd_gateway_labels_for_install", lambda: [])
    monkeypatch.setattr(gateway, "_locate_launchd_gateway_service", lambda label: (None, None))
    monkeypatch.setattr(gateway, "_gateway_descendants_of", lambda pid: {pid + 1})
    monkeypatch.setattr(
        gateway.subprocess,
        "run",
        lambda args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="501\t-\tai.hermes.gateway.default\n601\t-\tai.hermes.gateway.other\n",
            stderr="",
        ),
    )

    assert gateway._get_service_pids(all_profiles=True) == {501, 502, 601, 602}


def test_wrapped_service_gateway_is_excluded_from_manual_sweep(monkeypatch):
    service_pids = {501, 503}
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr(gateway, "is_windows", lambda: False)
    monkeypatch.setattr(gateway, "_scan_gateway_pids", lambda *args, **kwargs: [503, 700])

    assert gateway.find_gateway_pids(
        exclude_pids=service_pids,
        all_profiles=True,
    ) == [700]
