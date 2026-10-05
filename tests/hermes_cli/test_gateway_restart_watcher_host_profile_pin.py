"""The detached host-gateway restart watcher must not follow the sticky ``active_profile``.

``hermes update`` respawns the multiplex host through ``_spawn_gateway_restart_watcher`` with a
selector-less ``gateway run --replace``. The respawn carries no supervisor marker, so
``_apply_profile_override`` honours ``active_profile`` (#22502): after ``hermes profile use
<named>`` the new process re-homed into that profile and was refused ("Profile '<named>' does not
get a gateway of its own"). The update still reported the restart as done and the host stayed
down. The watcher now names the host explicitly with ``--profile default``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import gateway


@pytest.mark.parametrize(
    "argv, expected",
    [
        (
            ["python", "-m", "hermes_cli.main", "gateway", "run", "--replace"],
            ["python", "-m", "hermes_cli.main", "--profile", "default", "gateway", "run", "--replace"],
        ),
        (
            ["hermes", "gateway", "run"],
            ["hermes", "--profile", "default", "gateway", "run"],
        ),
    ],
)
def test_selectorless_host_argv_is_pinned_to_default(argv, expected):
    assert gateway._pin_host_profile_selector(argv) == expected


@pytest.mark.parametrize(
    "argv",
    [
        ["python", "-m", "hermes_cli.main", "--profile", "default", "gateway", "run"],
        ["python", "-m", "hermes_cli.main", "-p", "worker", "gateway", "run"],
        ["python", "-m", "hermes_cli.main", "--profile=worker", "gateway", "run"],
        ["python", "-c", "print('not a gateway command')"],
    ],
)
def test_existing_selector_or_non_gateway_argv_is_left_alone(argv):
    assert gateway._pin_host_profile_selector(argv) == argv


def _capture_watcher_spawn(monkeypatch) -> list[list[str]]:
    spawned: list[list[str]] = []

    def _fake_popen(argv, *args, **kwargs):
        spawned.append(list(argv))
        return object()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(gateway, "_host_gateway_watcher_env", lambda: {})
    return spawned


def _respawn_argv(watcher_argv: list[str], old_pid: int) -> list[str]:
    """The command the watcher will respawn: everything after ``<old_pid>``."""
    return watcher_argv[watcher_argv.index(str(old_pid)) + 1:]


def test_host_restart_watcher_respawns_with_explicit_default_profile(monkeypatch):
    spawned = _capture_watcher_spawn(monkeypatch)

    assert gateway._spawn_gateway_restart_watcher(
        4242, [sys.executable, "-m", "hermes_cli.main", "gateway", "run", "--replace"], host=True,
    )

    respawn = _respawn_argv(spawned[0], 4242)
    gw = respawn.index("gateway")
    assert respawn[gw - 2:gw] == ["--profile", "default"], respawn
    assert respawn[gw:] == ["gateway", "run", "--replace"]


def test_named_profile_restart_watcher_argv_is_unchanged(monkeypatch):
    spawned = _capture_watcher_spawn(monkeypatch)

    assert gateway._spawn_gateway_restart_watcher(
        4242, [sys.executable, "-m", "hermes_cli.main", "--profile", "worker", "gateway", "run"],
        host=False,
    )

    respawn = _respawn_argv(spawned[0], 4242)
    assert respawn.count("--profile") == 1
    assert respawn[-4:] == ["--profile", "worker", "gateway", "run"]


@pytest.fixture
def _sticky_named_profile(tmp_path, monkeypatch):
    """A default root whose ``active_profile`` names ``worker`` and no supervisor marker."""
    root = tmp_path / ".hermes"
    worker = root / "profiles" / "worker"
    worker.mkdir(parents=True)
    (worker / "config.yaml").write_text("{}\n", encoding="utf-8")
    (root / "active_profile").write_text("worker", encoding="utf-8")
    monkeypatch.setattr("hermes_constants._get_platform_default_hermes_home", lambda: root)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    for var in ("HERMES_SUPERVISED_CHILD", "HERMES_S6_SUPERVISED_CHILD", "INVOCATION_ID",
                "HERMES_GATEWAY_EXTERNAL_SUPERVISOR"):
        monkeypatch.delenv(var, raising=False)
    return root, worker


def _home_after_override(monkeypatch, argv: list[str]) -> Path:
    monkeypatch.setattr(sys, "argv", argv)
    from hermes_cli.main import _apply_profile_override
    _apply_profile_override()
    return Path(os.environ["HERMES_HOME"]).resolve()


def test_premise_selectorless_respawn_follows_sticky_profile(_sticky_named_profile, monkeypatch):
    _root, worker = _sticky_named_profile
    assert _home_after_override(monkeypatch, ["hermes", "gateway", "run", "--replace"]) == worker.resolve()


def test_pinned_respawn_stays_on_the_default_root(_sticky_named_profile, monkeypatch):
    root, _worker = _sticky_named_profile
    pinned = gateway._pin_host_profile_selector(["hermes", "gateway", "run", "--replace"])
    assert _home_after_override(monkeypatch, pinned) == root.resolve()


# ---------------------------------------------------------------------------
# Host identity must be settled while the gateway is still live.
#
# A selector-less argv is recognised as the host only through live evidence: the
# published rendezvous record. ``hermes update`` on Windows replays unmapped gateways
# AFTER force-killing them, so inferring at spawn time (``host=None``) classified the
# host as a named-profile gateway, skipped the pin, and the respawn followed the sticky
# ``active_profile`` -- the failure this fix targets. The updater itself sits on the
# sticky named profile's home, so neither the default-root fallback nor its own
# multiplex flag can rescue it.
# ---------------------------------------------------------------------------

_HOST_ARGV = [sys.executable, "-m", "hermes_cli.main", "gateway", "run", "--replace"]


@pytest.fixture
def _updater_on_sticky_named_profile(tmp_path, monkeypatch):
    """The updater runs with ``HERMES_HOME`` on the sticky named profile, not the default root."""
    root = tmp_path / ".hermes"
    worker = root / "profiles" / "worker"
    worker.mkdir(parents=True)
    (root / "active_profile").write_text("worker", encoding="utf-8")
    monkeypatch.setattr("hermes_constants._get_platform_default_hermes_home", lambda: root)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(worker))
    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: False)
    return root, worker


class _HostRecord:
    """Controls what the gateway rendezvous record says, and whether its owner is alive."""

    def __init__(self, monkeypatch, profiles: tuple[str, ...]):
        from gateway import host_rendezvous as hr

        self.alive = True
        self.record = type("Record", (), {"profiles": profiles})()
        monkeypatch.setattr(hr, "read_record", lambda role, **_kw: self.record if self.alive else None)
        monkeypatch.setattr(hr, "liveness_is_proven", lambda record: self.alive)


@pytest.mark.parametrize(
    "served, is_host",
    [
        (("default", "worker"), True),   # multiplex roster
        (("default",), True),            # single-profile host: still the host
        (("worker",), False),            # a standalone named profile owns the host lock
        ((), False),                     # provisional claim-time record: nothing settled yet
    ],
)
def test_live_record_settles_selectorless_identity(_updater_on_sticky_named_profile, monkeypatch, served, is_host):
    _HostRecord(monkeypatch, served)
    assert gateway._restart_argv_is_host_gateway(_HOST_ARGV) is is_host


def test_inference_is_lost_once_the_gateway_is_gone(_updater_on_sticky_named_profile, monkeypatch):
    """Premise: why the replay cannot infer late. Same argv, same updater -- only the evidence died."""
    record = _HostRecord(monkeypatch, ("default", "worker"))
    assert gateway._restart_argv_is_host_gateway(_HOST_ARGV) is True
    record.alive = False
    assert gateway._restart_argv_is_host_gateway(_HOST_ARGV) is False


def test_inferred_host_pins_the_respawn_while_live(_updater_on_sticky_named_profile, monkeypatch):
    """``host=None`` -- the path every production caller took -- pins when live evidence exists."""
    _HostRecord(monkeypatch, ("default",))
    spawned = _capture_watcher_spawn(monkeypatch)

    assert gateway._spawn_gateway_restart_watcher(4242, list(_HOST_ARGV))

    respawn = _respawn_argv(spawned[0], 4242)
    gw = respawn.index("gateway")
    assert respawn[gw - 2:gw] == ["--profile", "default"], respawn


def test_cmdline_replay_forwards_the_settled_identity(_updater_on_sticky_named_profile, monkeypatch):
    """With the gateway already gone, the caller's settled ``host=True`` still pins the respawn."""
    _HostRecord(monkeypatch, ("default", "worker")).alive = False
    spawned = _capture_watcher_spawn(monkeypatch)

    assert gateway.launch_detached_gateway_restart_by_cmdline(4242, list(_HOST_ARGV), host=True)

    respawn = _respawn_argv(spawned[0], 4242)
    gw = respawn.index("gateway")
    assert respawn[gw - 2:gw] == ["--profile", "default"], respawn


def test_fleet_cmdline_fallback_pins_a_single_profile_host(_updater_on_sticky_named_profile, monkeypatch):
    """``_prepare_profile_gateway_update_restart`` arms its cmdline fallback before the kill, so the
    live record is still there; a single-profile host (served == ("default",)) must count as the host."""
    _HostRecord(monkeypatch, ("default",))
    spawned = _capture_watcher_spawn(monkeypatch)
    monkeypatch.setattr(gateway, "_capture_gateway_argv", lambda pid: list(_HOST_ARGV))
    monkeypatch.setattr(gateway, "launch_detached_profile_gateway_restart", lambda profile, pid: False)

    assert gateway._prepare_profile_gateway_update_restart("default", 4242) == "detached-cmdline"

    respawn = _respawn_argv(spawned[0], 4242)
    gw = respawn.index("gateway")
    assert respawn[gw - 2:gw] == ["--profile", "default"], respawn


def test_windows_unmapped_host_keeps_its_identity_across_pause_and_relaunch(
    _updater_on_sticky_named_profile, monkeypatch,
):
    """Regression (#132645): the Windows updater captures an unmapped selector-less host, kills it,
    and replays it after the update. The identity is settled at pause time and carried on the token,
    so the replay still pins ``--profile default`` once the rendezvous record is gone."""
    from hermes_cli import main as cli_main
    import gateway.status as gateway_status
    import hermes_cli.update_cmd_windows as update_cmd_windows

    record = _HostRecord(monkeypatch, ("default", "worker"))
    killed: list[int] = []

    monkeypatch.setattr(cli_main, "_is_windows", lambda: True)
    monkeypatch.setattr(update_cmd_windows, "_discover_windows_gateways", lambda: ({}, [], set(), [4242]))
    monkeypatch.setattr(update_cmd_windows, "_request_socket_pauses", lambda *a: ({}, [], []))
    monkeypatch.setattr(update_cmd_windows, "_record_attested_cold_start_profiles", lambda token, running: None)
    monkeypatch.setattr(cli_main, "_venv_launcher_ancestors", lambda pids: [])
    monkeypatch.setattr(cli_main, "_wait_for_windows_update_gateway_exit", lambda pids, timeout: set())
    monkeypatch.setattr(gateway, "_capture_gateway_argv", lambda pid: list(_HOST_ARGV))
    monkeypatch.setattr(gateway_status, "get_process_start_time", lambda pid: None)
    monkeypatch.setattr(gateway_status, "terminate_pid", lambda pid, **_kw: killed.append(pid))

    token = update_cmd_windows._pause_windows_gateways_for_update()

    assert killed == [4242]
    assert token["unmapped"] == [{"pid": 4242, "argv": _HOST_ARGV, "host": True}]

    # The update runs; the host is gone and so is its rendezvous record.
    record.alive = False
    spawned = _capture_watcher_spawn(monkeypatch)

    relaunched, unmapped_count = update_cmd_windows._relaunch_paused_gateways(token, {}, token["unmapped"])

    assert (relaunched, unmapped_count) == ([], 1)
    respawn = _respawn_argv(spawned[0], 4242)
    gw = respawn.index("gateway")
    assert respawn[gw - 2:gw] == ["--profile", "default"], respawn


def test_windows_relaunch_without_settled_identity_still_infers(_updater_on_sticky_named_profile, monkeypatch):
    """Tokens written before the ``host`` key (or a failed classification) keep the old inference."""
    import hermes_cli.update_cmd_windows as update_cmd_windows

    seen: list = []
    monkeypatch.setattr(
        gateway, "launch_detached_gateway_restart_by_cmdline",
        lambda old_pid, argv, *, host=None: seen.append(host) or True,
    )
    token: dict = {}
    update_cmd_windows._relaunch_paused_gateways(
        token, {}, [{"pid": 1, "argv": list(_HOST_ARGV)}, {"pid": 2, "argv": list(_HOST_ARGV), "host": "yes"}],
    )
    assert seen == [None, None]
