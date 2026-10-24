"""Updater progress for optional default-tool installs."""

from __future__ import annotations

import threading

import pm
import pm.defaults
import pm.install
import pm.lock
import pm.paths
from hermes_cli import update_cmd_maint as update


def test_default_tool_install_emits_flushed_heartbeats(monkeypatch):
    heartbeat_seen = threading.Event()
    printed: list[tuple[str, bool]] = []

    monkeypatch.setattr(pm.install, "sealed", lambda: False)
    monkeypatch.setattr(pm.install, "lazy_installs_allowed", lambda: True)
    monkeypatch.setattr(pm.defaults, "default_packages", lambda names: ["agent-browser"])
    monkeypatch.setattr(pm.paths, "lockfile_path", lambda: None)
    monkeypatch.setattr(pm.lock, "Lockfile", lambda path: type("Lock", (), {"names": lambda self: []})())
    monkeypatch.setattr(pm, "installed_package", lambda name: None)
    monkeypatch.setattr(update, "_DEFAULT_TOOL_HEARTBEAT_SECONDS", 0.0, raising=False)

    def fake_print(message="", *, flush=False):
        printed.append((message, flush))
        heartbeats = [line for line, _ in printed if "Still installing agent-browser" in line]
        if len(heartbeats) >= 2:
            heartbeat_seen.set()

    def ensure(name, *, explicit):
        assert heartbeat_seen.wait(2), "default-tool install emitted no periodic heartbeat"

    monkeypatch.setattr("builtins.print", fake_print)
    monkeypatch.setattr(pm, "ensure", ensure)

    update._install_default_tools_after_update()

    heartbeats = [(line, flush) for line, flush in printed if "Still installing agent-browser" in line]
    assert len(heartbeats) >= 2
    assert all(flush for _, flush in heartbeats)


def test_default_tool_install_failure_remains_best_effort(monkeypatch, capsys):
    monkeypatch.setattr(pm.install, "sealed", lambda: False)
    monkeypatch.setattr(pm.install, "lazy_installs_allowed", lambda: True)
    monkeypatch.setattr(pm.defaults, "default_packages", lambda names: ["agent-browser"])
    monkeypatch.setattr(pm.paths, "lockfile_path", lambda: None)
    monkeypatch.setattr(pm.lock, "Lockfile", lambda path: type("Lock", (), {"names": lambda self: []})())
    monkeypatch.setattr(pm, "installed_package", lambda name: None)
    monkeypatch.setattr(
        pm,
        "ensure",
        lambda name, *, explicit: (_ for _ in ()).throw(pm.InstallError(name, "network down")),
    )

    update._install_default_tools_after_update()

    output = capsys.readouterr().out
    assert "agent-browser was not installed" in output
    assert "Retry with: hermes pm install agent-browser" in output
