"""Behaviour contracts for the Omarchy VM realm kind.

Native-VM lifecycle needs KVM and a base image and is proven by the E2E rig,
not here. These pin the contracts a refactor could silently break: which kind a
conversation routes to, what a routed execution context declares about the far
side's filesystem, and that the host-escape guard blocks the escape it was
written for without blocking ordinary work.
"""
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "plugins/hermes-realms"


def load(module):
    return runpy.run_path(str(PLUGIN / "realms/_binding.py"))["load_runtime"](module)


@pytest.fixture
def guard():
    return load("host_guard")


@pytest.mark.linux_only
@pytest.mark.parametrize(
    "command",
    [
        "env WAYLAND_DISPLAY=wayland-0 hyprctl monitors",
        "HYPRLAND_INSTANCE_SIGNATURE=abc hyprctl dispatch exit",
        "export DISPLAY=:0; xdotool key Return",
        "XDG_RUNTIME_DIR=/run/user/1000 grim /tmp/host.png",
        "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus gdbus call --session",
        # shlex keeps a trailing operator attached to the assignment token, so
        # these are the shapes a naive scan silently lets through.
        "cd /tmp && DISPLAY=:0 xdotool key Return",
        "echo start; WAYLAND_DISPLAY=wayland-0 grim /tmp/host.png",
    ],
)
def test_guard_refuses_host_display_repointing(guard, command):
    """Re-pointing a realm's GUI handles at the host is refused, with a reason."""
    message = guard.host_escape(command)
    assert message is not None, command
    assert "/realm off" in message


@pytest.mark.linux_only
@pytest.mark.parametrize(
    "command",
    [
        "ls -la ~",
        "grep -rn 'DISPLAY=:0' /etc",
        "echo WAYLAND_DISPLAY=wayland-0 >> notes.txt",
        "WAYLAND_DISPLAY= grim /tmp/shot.png",
        "python -c 'print(1)'",
        "",
    ],
)
def test_guard_allows_ordinary_commands(guard, command):
    """A guard that blocks ordinary work would be turned off, so it must not."""
    assert guard.host_escape(command) is None, command


@pytest.mark.linux_only
def test_guard_ignores_unparseable_shell_text(guard):
    """Unbalanced quotes are reported as ordinary rather than guessed at."""
    assert guard.host_escape("echo 'unterminated") is None


@pytest.mark.linux_only
def test_vm_settings_round_trip_through_a_launch_spec(tmp_path):
    """A realm's frozen spec must rebuild the same config it was written from.

    ``asdict``/``Config(**data)`` is how a payload-only launcher attaches to an
    already-owned guest without importing host config; a nested dataclass that
    does not survive that round trip breaks every routed command.
    """
    config_module = load("config")
    from dataclasses import asdict

    original = config_module.Config(
        default_kind="omarchy-vm",
        vm=config_module.VmConfig(memory=4096, network=False),
    )
    assert config_module.Config(**asdict(original)) == original


@pytest.mark.linux_only
@pytest.mark.parametrize(
    "settings",
    [
        {"default_kind": "nonsense"},
        {"vm": {"memory": 64}},
        {"vm": {"memory": "3072"}},
        {"vm": {"network": "yes"}},
        {"vm": {"omarchy_vm_path": "relative/omarchy-vm"}},
    ],
)
def test_invalid_vm_settings_are_refused(settings):
    """Bad settings fail at load, not at the first routed command."""
    config_module = load("config")
    with pytest.raises(ValueError):
        config_module.Config(**settings)


@pytest.mark.linux_only
def test_vendored_script_matches_its_pin():
    """The executed copy is the reviewed one.

    An unpinned vendored script is an unreviewed one: this is what stops a
    tampered or silently upgraded copy from being run against a guest.
    """
    import hashlib

    vm_manager = load("vm_manager")
    digest = hashlib.sha256(vm_manager.VENDORED_SCRIPT.read_bytes()).hexdigest()
    assert digest == vm_manager.VENDORED_SHA256


@pytest.mark.linux_only
def test_vendored_script_is_headless_and_parameterised(tmp_path):
    """Upstream is single-VM and opens a window; the vendored copy must not.

    Both properties are behavioural, not cosmetic: an SDL window would appear on
    the user's own desktop, and a fixed unit name would make two conversations
    fight over one guest. Checked by running the script's own dispatcher rather
    than reading its text.
    """
    vm_manager = load("vm_manager")
    # ``source`` would run the script's own dispatcher (and its ``set -e``), so
    # the definitions are read with the body stripped of its final case block.
    body = vm_manager.VENDORED_SCRIPT.read_text(encoding="utf-8")
    definitions = body.split('\ncommand="${1:-}"', 1)[0]
    probe = tmp_path / "definitions.sh"
    probe.write_text(definitions, encoding="utf-8")
    result = subprocess.run(
        ["bash", "-c",
         f'source {probe}; graphics_args; echo "UNIT=$UNIT"; echo "VNC=$VNC_SOCKET"'],
        env={"PATH": os.environ["PATH"], "HOME": "/nonexistent", "USER": "probe",
             "OMARCHY_VM_UNIT": "hermes-vm-probe",
             "OMARCHY_VM_VNC_SOCKET": "/run/probe/vnc.sock"},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "-display none" in result.stdout, result.stdout
    assert "sdl" not in result.stdout.lower(), result.stdout
    assert "UNIT=hermes-vm-probe" in result.stdout
    assert "VNC=/run/probe/vnc.sock" in result.stdout


@pytest.mark.linux_only
def test_tool_arguments_and_slash_text_are_one_grammar():
    """The model tool and the slash command must not drift into separate parsers."""
    plugin = runpy.run_path(str(PLUGIN / "plugin.py"))
    render = plugin["_raw_command"]
    assert render({"action": "on", "kind": "omarchy-vm"}) == "on omarchy"
    assert render({"action": "on"}) == "on"
    assert render({"action": "size", "size": "1280x720"}) == "size 1280x720"
    assert render({"action": "push", "path": "/src", "destination": "/dst"}) == (
        "push /src /dst")
    assert render({"action": "pull", "path": "/g", "destination": "/h"}) == "pull /g /h"
    assert render({"action": "status"}) == "status"
    for action in plugin["SCHEMA"]["parameters"]["properties"]["action"]["enum"]:
        assert render({"action": action}).split()[0] == action


@pytest.mark.linux_only
def test_update_check_never_fabricates_a_verdict(tmp_path, monkeypatch):
    """An unreachable release API reports "unknown", not "up to date".

    The settings panel drives an Update button from this. Treating a failed
    network call as "no update" hides a stale base forever; treating it as
    "update available" nags for a 6 GB rebuild nobody needs. Both are worse
    than saying it could not be checked.
    """
    vm_manager = load("vm_manager")
    manager = vm_manager.VmManager.__new__(vm_manager.VmManager)
    monkeypatch.setattr(manager, "base_status",
                        lambda: {"present": True, "iso": "omarchy-4.0.3.iso"}, raising=False)
    monkeypatch.setattr(manager, "storage", lambda: {}, raising=False)
    manager.config = type("C", (), {"vm": type("V", (), {"memory": 3072, "network": True})()})()

    monkeypatch.setattr(manager, "latest_version", lambda **_: None, raising=False)
    unknown = manager.settings(check_updates=True)["update"]
    assert unknown["latest"] is None and unknown["available"] is None

    monkeypatch.setattr(manager, "latest_version", lambda **_: "4.1.0", raising=False)
    assert manager.settings(check_updates=True)["update"]["available"] is True

    monkeypatch.setattr(manager, "latest_version", lambda **_: "4.0.3", raising=False)
    assert manager.settings(check_updates=True)["update"]["available"] is False


@pytest.mark.linux_only
def test_settings_render_does_not_require_the_network(monkeypatch):
    """Rendering the panel must never depend on reaching GitHub."""
    vm_manager = load("vm_manager")
    manager = vm_manager.VmManager.__new__(vm_manager.VmManager)
    monkeypatch.setattr(manager, "base_status", lambda: {"present": False}, raising=False)
    monkeypatch.setattr(manager, "storage", lambda: {}, raising=False)
    manager.config = type("C", (), {"vm": type("V", (), {"memory": 3072, "network": True})()})()

    def _fail(**_):
        raise AssertionError("settings() checked updates without being asked")

    monkeypatch.setattr(manager, "latest_version", _fail, raising=False)
    assert manager.settings()["update"]["available"] is None


@pytest.mark.linux_only
@pytest.mark.parametrize("command", [
    "env WAYLAND_DISPLAY=wayland-0 hyprctl dispatch exit",
    "WAYLAND_DISPLAY=wayland-0 hyprctl dispatch exec kitty",
    "DISPLAY=:0 xdotool key super",
])
def test_escape_guard_does_not_trust_its_own_environment(guard, command, monkeypatch):
    """The guard must block a host handle even when the backend already has it.

    ``pre_tool`` runs in the Hermes backend process, which for a normal user
    sits on their own desktop session — so its environment holds the HOST's
    WAYLAND_DISPLAY/DISPLAY. An earlier version exempted "the value already in
    my environment" as self-evidently private, which exempted exactly the
    escape this guard exists to stop. A realm's private compositor is also
    ``wayland-0``, so the string can never distinguish them.
    """
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("HYPRLAND_INSTANCE_SIGNATURE", "abc123")
    assert guard.host_escape(command) is not None


@pytest.mark.linux_only
def test_guest_ssh_refuses_forwarding_the_users_agent(tmp_path):
    """Agent forwarding is a per-command opt-in, never inherited from ssh config.

    ``ssh`` merges ``~/.ssh/config``, and a ``Host *`` block with
    ``ForwardAgent yes`` is a common setup. Forwarding it into a disposable
    guest that has passwordless sudo and no disk encryption would let anything
    in there authenticate as the user; ``ForwardX11`` is the same hazard
    pointed back at the host's display. Asserted through ``ssh -G``, which
    resolves the real merged configuration rather than our argv.
    """
    vm_manager = load("vm_manager")
    manager = vm_manager.VmManager.__new__(vm_manager.VmManager)
    argv = manager.ssh_argv({"ssh_port": 2222})

    hostile = tmp_path / "config"
    hostile.write_text("Host *\n  ForwardAgent yes\n  ForwardX11 yes\n")
    resolved = subprocess.run(
        ["ssh", "-F", str(hostile), "-G", *argv[1:]],
        capture_output=True, text=True, timeout=30).stdout.lower()

    assert "forwardagent no" in resolved, resolved
    assert "forwardx11 no" in resolved, resolved
    assert "identitiesonly yes" in resolved, resolved


@pytest.mark.linux_only
def test_network_off_restricts_the_guest_netdev():
    """``vm.network: false`` must reach QEMU as ``restrict=on``.

    The setting is a safety posture, so it has to change the guest's actual
    netdev rather than only a status payload. Proven against a live guest in
    the E2E rig (on: HTTP 200, off: connection failure); this pins the wiring
    that carries it there.
    """
    import dataclasses

    vm_manager = load("vm_manager")
    manager = vm_manager.VmManager.__new__(vm_manager.VmManager)
    config = load("config")
    manager.config = config.Config(vm=config.VmConfig(network=False))
    manager.data = Path("/tmp/realms-data")
    off = manager._script_env(vm_home="/tmp/vm", ssh_port=2222,
                              unit="u.service", runtime=Path("/tmp/rt"))
    assert off["OMARCHY_VM_NETDEV_EXTRA"] == ",restrict=on"

    manager.config = dataclasses.replace(
        manager.config, vm=dataclasses.replace(manager.config.vm, network=True))
    on = manager._script_env(vm_home="/tmp/vm", ssh_port=2222,
                             unit="u.service", runtime=Path("/tmp/rt"))
    assert "OMARCHY_VM_NETDEV_EXTRA" not in on or not on["OMARCHY_VM_NETDEV_EXTRA"]


@pytest.mark.linux_only
def test_vm_vnc_peer_is_proven_by_the_realms_own_scope(tmp_path):
    """A VM realm's VNC listener is QEMU, not a process the plugin started.

    The labwc path proves socket ownership by matching the listener against the
    worker/VNC processes in the record's ``processes`` map. A VM record has no
    such map, so that check rejected QEMU outright and ``/realm watch`` could
    never open a VM realm. Cgroup membership is the equivalent proof, and it
    must still refuse a PID outside the scope — otherwise it is decorative.
    """
    bridge = load("bridge")
    realm = {"kind": "omarchy-vm", "cgroup": "/user.slice/hermes-vm-probe.service"}

    assert bridge._peer_in_realm_scope(os.getpid(), realm) is False
    # No scope recorded must fail closed, never fall open.
    assert bridge._peer_in_realm_scope(os.getpid(), {"kind": "omarchy-vm"}) is False
    assert bridge._peer_in_realm_scope(os.getpid(), {**realm, "cgroup": ""}) is False

    # This process IS inside its own cgroup, which is the accept path.
    with open(f"/proc/{os.getpid()}/cgroup", encoding="utf-8") as handle:
        mine = handle.read().splitlines()[0].split(":", 2)[-1].strip()
    assert bridge._peer_in_realm_scope(os.getpid(), {**realm, "cgroup": mine}) is True


def _reset_probe_service(tmp_path, kind):
    """A service whose realm managers are recorded rather than really started."""
    integration = load("integration")
    service = integration.RealmIntegration(str(tmp_path))
    owner = "probe-session"
    identity = {key: owner for key in
                ("session_id", "stored_session_id", "runtime_session_id", "task_id")}
    service.bind(**identity)
    service.owners.set_mode(owner, "realm")
    service.owners.set_kind(owner, kind)

    stopped = []
    service.stop = lambda who: stopped.append(who)
    return service, identity, stopped


@pytest.mark.parametrize("kind", ["realm", "omarchy-vm"])
def test_a_context_reset_does_not_tear_down_the_realm(tmp_path, kind):
    """``/clear`` means "forget the transcript", never "destroy my desktop".

    ``on_session_reset`` fires for ``/new``, ``/clear`` AND whenever the gateway
    builds this session's agent — which happens right after the realm is
    started. Routing it to teardown destroyed a realm the user had just asked
    for, and with it any work inside. Only ``finalize`` ends a realm.
    """
    service, identity, stopped = _reset_probe_service(tmp_path, kind)

    service.reset(**identity)
    assert stopped == []

    service.finalize(**identity)
    assert stopped == ["probe-session"], "finalize must still reclaim the realm"


def test_a_reset_never_constructs_a_vm_manager_on_a_labwc_host(tmp_path):
    """The ``vm`` property is deferred so a host with no QEMU never pays for it.

    Reconciliation on reset is worth doing, but it must not be the thing that
    finally builds a ``VmManager`` (a config load plus a registry mkdir) for a
    session that only ever used the labwc kind.
    """
    service, identity, _ = _reset_probe_service(tmp_path, "realm")
    assert service._vm is None

    service.reset(**identity)

    assert service._vm is None, "reset built a VmManager on a labwc-only session"
