"""A host gateway respawned by the restart watcher must stay the host under a sticky named profile.

``hermes update`` relaunches the multiplex host through the detached restart watcher. The respawn
carries no supervisor marker, so a selector-less ``gateway run`` follows the sticky ``active_profile``
(#22502): after ``hermes profile use <named>`` it re-homed into that profile and was refused ("Profile
'<named>' does not get a gateway of its own"). The update reported the restart as done and the host
stayed down (#132645).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import gateway

_HOST_ARGV = [sys.executable, "-m", "hermes_cli.main", "gateway", "run", "--replace"]
_OLD_PID = 4242


@pytest.fixture
def sticky_named_profile(tmp_path, monkeypatch):
    """``active_profile`` names ``worker``; the updater runs on that profile's home, unsupervised."""
    root = tmp_path / ".hermes"
    worker = root / "profiles" / "worker"
    worker.mkdir(parents=True)
    (worker / "config.yaml").write_text("{}\n", encoding="utf-8")
    (root / "active_profile").write_text("worker", encoding="utf-8")
    monkeypatch.setattr("hermes_constants._get_platform_default_hermes_home", lambda: root)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(worker))
    for var in ("HERMES_SUPERVISED_CHILD", "HERMES_S6_SUPERVISED_CHILD", "INVOCATION_ID",
                "HERMES_GATEWAY_EXTERNAL_SUPERVISOR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: False)
    return root


class _HostRecord:
    """The host's published rendezvous record; it disappears when the host dies."""

    def __init__(self, monkeypatch, profiles: tuple[str, ...]):
        from gateway import host_rendezvous as hr

        self.alive = True
        record = type("Record", (), {"profiles": profiles})()
        monkeypatch.setattr(hr, "read_record", lambda role, **_kw: record if self.alive else None)
        monkeypatch.setattr(hr, "liveness_is_proven", lambda _record: self.alive)


def _capture_spawns(monkeypatch) -> list[list[str]]:
    spawned: list[list[str]] = []
    monkeypatch.setattr(subprocess, "Popen", lambda argv, *a, **kw: spawned.append(list(argv)) or object())
    return spawned


def _respawned_home(monkeypatch, root: Path, watcher_argv: list[str]) -> Path:
    """Run the CLI profile pre-parse on the argv the watcher will respawn, in the host's env."""
    respawn = watcher_argv[watcher_argv.index(str(_OLD_PID)) + 1:]
    monkeypatch.setenv("HERMES_HOME", str(root))  # host_gateway_child_env targets the default root
    monkeypatch.setattr(sys, "argv", ["hermes", *respawn[respawn.index("hermes_cli.main") + 1:]])
    from hermes_cli.main import _apply_profile_override

    _apply_profile_override()
    return Path(os.environ["HERMES_HOME"]).resolve()


def test_live_single_profile_host_respawns_on_the_default_root(sticky_named_profile, monkeypatch):
    """Profile-derived / fleet path: identity inferred while the host is live (``host=None``)."""
    _HostRecord(monkeypatch, ("default",))
    spawned = _capture_spawns(monkeypatch)

    assert gateway.launch_detached_gateway_restart_by_cmdline(_OLD_PID, list(_HOST_ARGV))

    assert _respawned_home(monkeypatch, sticky_named_profile, spawned[0]) == sticky_named_profile.resolve()


def test_windows_update_relaunch_keeps_the_host_after_its_record_is_gone(sticky_named_profile, monkeypatch):
    """Windows ``hermes update``: pause (classify + kill) an unmapped host, then relaunch it."""
    from hermes_cli import main as cli_main
    import gateway.status as gateway_status
    import hermes_cli.update_cmd_windows as update_cmd_windows

    record = _HostRecord(monkeypatch, ("default", "worker"))
    monkeypatch.setattr(cli_main, "_is_windows", lambda: True)
    monkeypatch.setattr(update_cmd_windows, "_discover_windows_gateways", lambda: ({}, [], set(), [_OLD_PID]))
    monkeypatch.setattr(update_cmd_windows, "_request_socket_pauses", lambda *a: ({}, [], []))
    monkeypatch.setattr(update_cmd_windows, "_record_attested_cold_start_profiles", lambda token, running: None)
    monkeypatch.setattr(cli_main, "_venv_launcher_ancestors", lambda pids: [])
    monkeypatch.setattr(cli_main, "_wait_for_windows_update_gateway_exit", lambda pids, timeout: set())
    monkeypatch.setattr(gateway, "_capture_gateway_argv", lambda pid: list(_HOST_ARGV))
    monkeypatch.setattr(gateway_status, "get_process_start_time", lambda pid: None)
    monkeypatch.setattr(gateway_status, "terminate_pid", lambda pid, **_kw: None)

    token = update_cmd_windows._pause_windows_gateways_for_update()
    record.alive = False  # the update runs; the host and its rendezvous record are gone
    spawned = _capture_spawns(monkeypatch)
    update_cmd_windows._relaunch_paused_gateways(token, {}, token["unmapped"])

    assert _respawned_home(monkeypatch, sticky_named_profile, spawned[0]) == sticky_named_profile.resolve()
