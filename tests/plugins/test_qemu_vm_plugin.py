"""Behaviour tests for the bundled ``qemu_vm`` terminal backend plugin (``plugins/terminal_env/qemu_vm``).

No hypervisor is involved: the manager's host-touching calls (qemu, qemu-img, ssh, downloads)
are injected fakes, and discovery runs through the real ``PluginManager`` against a temp
``HERMES_HOME``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from plugins.terminal_env.qemu_vm import manager as vm
from plugins.terminal_env.qemu_vm.provider import QemuVmProvider, guest_cwd
from tools.environments.base import EnvironmentConnectionError


# --- discovery -------------------------------------------------------------------------------

def test_bundled_plugin_registers_backend_and_cli(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from hermes_cli.plugins import discover_plugins, get_plugin_manager
    import agent.terminal_env_registry as reg

    reg._reset_for_tests()
    discover_plugins(force=True)
    try:
        loaded = get_plugin_manager()._plugins["terminal_env/qemu_vm"]
        assert loaded.enabled and loaded.error is None
        provider = reg.get_provider("qemu_vm")
        assert provider is not None
        assert provider.is_remote is True
        assert provider.is_container is False
        assert provider.skip_container_guards is True
        assert provider.cache_path_base == "~/.hermes"
        assert get_plugin_manager()._cli_commands["vm"]["handler_fn"] is not None
    finally:
        reg._reset_for_tests()


def test_backend_is_classified_remote_and_not_container(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    import agent.terminal_env_registry as reg
    from agent.prompt_builder import _plugin_backend_is_remote
    from tools.terminal_tool_config import _is_container_backend

    reg._reset_for_tests()
    reg.register_provider(QemuVmProvider())
    try:
        assert _plugin_backend_is_remote("qemu_vm") is True
        assert _is_container_backend("qemu_vm") is False
    finally:
        reg._reset_for_tests()


# --- pure builders ---------------------------------------------------------------------------

def test_qemu_argv_forwards_ssh_on_loopback_only_and_runs_headless(tmp_path):
    paths = vm.VmPaths(tmp_path)
    argv = vm.build_qemu_argv("/usr/bin/qemu-system-x86_64", vm.VmSpec(cpus=3, memory_mb=1024), paths, 40022, label="p1")
    netdev = argv[argv.index("-netdev") + 1]
    assert netdev == "user,id=net0,hostfwd=tcp:127.0.0.1:40022-:22"
    assert "restrict=on" not in netdev
    assert "q35,accel=kvm" in argv and "-daemonize" in argv
    assert argv[argv.index("-smp") + 1] == "3" and argv[argv.index("-m") + 1] == "1024M"
    assert argv[argv.index("-pidfile") + 1] == str(paths.pidfile)
    assert argv[argv.index("-display") + 1] == "none"
    assert "-vga" not in argv  # GRUB in the Debian cloud image needs a VGA BIOS or it boot-loops
    assert str(paths.disk) in argv[argv.index("-drive") + 1]


def test_qemu_argv_network_none_restricts_guest(tmp_path):
    argv = vm.build_qemu_argv("qemu", vm.VmSpec(network="none"), vm.VmPaths(tmp_path), 1, label="x")
    netdev = argv[argv.index("-netdev") + 1]
    assert "restrict=on" in netdev and "hostfwd=tcp:127.0.0.1:1-:22" in netdev


def test_user_data_locks_down_login_and_installs_the_key():
    text = vm.build_user_data(vm.VmSpec(user="agent", passwordless_sudo=True), "ssh-ed25519 AAAA key\n")
    assert text.startswith("#cloud-config\n")
    assert "ssh-ed25519 AAAA key" in text
    assert "ssh_pwauth: false" in text
    assert "NOPASSWD" in text and "name: agent" in text
    assert "AcceptEnv *" in text


def test_user_data_without_passwordless_sudo():
    text = vm.build_user_data(vm.VmSpec(passwordless_sudo=False), "k")
    assert "NOPASSWD" not in text


@pytest.mark.parametrize("bad", [
    {"network": "bridge"}, {"user": "root"}, {"user": "no-dash"}, {"memory_mb": 64},
    {"cpus": 0}, {"disk_gb": 1}, {"distro": "arch"}, {"cpus": True},
])
def test_spec_validation_rejects(bad):
    with pytest.raises(ValueError):
        replace(vm.VmSpec(), **bad).validate()


def test_spec_from_mapping_ignores_unknown_keys():
    spec = vm.spec_from_mapping({"cpus": 8, "bogus": 1}, base=vm.VmSpec(memory_mb=512))
    assert spec.cpus == 8 and spec.memory_mb == 512


def test_ssh_options_use_per_vm_trust_store(tmp_path):
    target = vm.SshTarget("127.0.0.1", 2222, "hermes", str(tmp_path / "id"), str(tmp_path / "kh"))
    argv = vm.ssh_base_argv(target)
    assert f"UserKnownHostsFile={tmp_path / 'kh'}" in argv
    assert f"HostKeyAlias={vm.HOST_KEY_ALIAS}" in argv
    assert "IdentitiesOnly=yes" in argv and str(tmp_path / "id") in argv


@pytest.mark.parametrize("cwd, expected", [
    (None, "~"), ("", "~"), ("/root", "~"), ("/home/u", "~"), ("/home/u/proj", "~/proj"), ("/srv/data", "/srv/data"),
])
def test_guest_cwd_maps_host_defaults_to_guest_home(cwd, expected):
    assert guest_cwd(cwd, (Path("/home/u"),)) == expected


def test_guest_cwd_passes_tilde_paths_through_and_checks_every_host_home():
    homes = (Path("/home/u"), Path("/opt/data/home"))
    assert guest_cwd("~/x", homes) == "~/x"
    assert guest_cwd("/opt/data/home/proj", homes) == "~/proj"


def test_pinned_images_have_https_urls_and_full_sha512():
    for spec in vm.IMAGES.values():
        assert spec.url.startswith("https://")
        assert len(spec.sha512) == 128 and int(spec.sha512, 16)


# --- lifecycle with fakes --------------------------------------------------------------------

class _Host:
    """Records every host argv the manager would run and simulates their side effects."""

    def __init__(self, root: Path):
        self.root = root
        self.calls: list[list[str]] = []
        self.probes = 0

    def run(self, argv, *, timeout=0, what=""):
        self.calls.append(list(argv))
        if argv[0] == "ssh-keygen":
            Path(argv[-1]).write_text("PRIVATE")
            Path(argv[-1] + ".pub").write_text("ssh-ed25519 AAAA fake")
        elif argv[0] == "qemu-img" and argv[1] == "convert":
            Path(argv[-1]).write_bytes(b"disk")
        elif argv[0] == "qemu":
            Path(argv[argv.index("-pidfile") + 1]).write_text("4242")
        elif argv[0] == "iso":
            Path(argv[1]).write_bytes(b"iso")

    def download(self, url, dest, progress=None):
        self.calls.append(["download", url, str(dest)])
        dest.write_bytes(b"IMAGE")

    def ssh_probe(self, argv):
        self.probes += 1
        return True


@pytest.fixture
def fake_host(tmp_path, monkeypatch):
    host = _Host(tmp_path)
    image = vm.ImageSpec("debian", "amd64", "https://example.invalid/debian.qcow2", hashlib.sha512(b"IMAGE").hexdigest())
    monkeypatch.setattr(vm, "IMAGES", {("debian", "amd64"): image})
    monkeypatch.setattr(vm, "native_arch", lambda: "amd64")
    monkeypatch.setattr(vm, "host_requirements", lambda: [vm.HostRequirement(True, "all", "ok")])
    monkeypatch.setattr(vm, "require_tool", lambda name: {"qemu-system-x86_64": "qemu"}.get(name, name))
    monkeypatch.setattr(vm, "iso_tool", lambda: ["iso", "{iso}", "{user_data}", "{meta_data}"])
    monkeypatch.setattr(vm, "free_loopback_port", lambda: 45000)
    monkeypatch.setattr(vm, "_port_free", lambda port: True)
    # "running" = pidfile present; the monitor socket cannot exist without a real qemu.
    monkeypatch.setattr(vm.VmManager, "running", lambda self: self.paths.pidfile.is_file())
    return host


def _manager(tmp_path, host, **kw):
    return vm.VmManager(tmp_path / "vm", image_cache=tmp_path / "cache", label="t", runner=host.run,
                        downloader=host.download, ssh_probe=host.ssh_probe, sleep=lambda s: None, log=lambda m: None, **kw)


def test_create_provisions_disk_key_seed_and_boots(tmp_path, fake_host):
    manager = _manager(tmp_path, fake_host)
    manager.create(vm.VmSpec(disk_gb=30, cpus=4), boot_timeout=10)

    names = [c[0] for c in fake_host.calls]
    assert names == ["download", "qemu-img", "qemu-img", "ssh-keygen", "iso", "qemu"]
    resize = fake_host.calls[2]
    assert resize[1:3] == ["resize", str(manager.paths.disk)] and resize[3] == "30G"
    assert manager.exists() and manager.running()
    state = json.loads(manager.paths.state.read_text())
    assert state["port"] == 45000 and state["spec"]["cpus"] == 4 and state["spec"]["disk_gb"] == 30
    assert manager.ssh_target() == vm.SshTarget("127.0.0.1", 45000, "hermes", str(manager.paths.key),
                                                str(manager.paths.known_hosts))
    assert "cloud-init" in " ".join(fake_host.calls[-1]) or fake_host.probes >= 1
    user_data = (manager.paths.root / "user-data").read_text()
    assert "ssh-ed25519 AAAA fake" in user_data


def test_create_refuses_to_overwrite_without_force(tmp_path, fake_host):
    manager = _manager(tmp_path, fake_host)
    manager.create(vm.VmSpec(), boot_timeout=10)
    with pytest.raises(vm.VmError, match="--force"):
        manager.create(vm.VmSpec(), boot_timeout=10)


def test_image_cache_is_reused_when_checksum_matches(tmp_path, fake_host):
    manager = _manager(tmp_path, fake_host)
    manager.create(vm.VmSpec(), boot_timeout=10)
    fake_host.calls.clear()
    manager.create(vm.VmSpec(), force=True, boot_timeout=10)
    assert "download" not in [c[0] for c in fake_host.calls]


def test_checksum_mismatch_rejects_and_removes_download(tmp_path, fake_host, monkeypatch):
    bad = vm.ImageSpec("debian", "amd64", "https://example.invalid/x.qcow2", "0" * 128)
    monkeypatch.setattr(vm, "IMAGES", {("debian", "amd64"): bad})
    manager = _manager(tmp_path, fake_host)
    with pytest.raises(vm.VmError, match="Checksum mismatch"):
        manager.create(vm.VmSpec(), boot_timeout=10)
    assert not (tmp_path / "cache" / "x.qcow2").exists()
    assert not manager.exists()


def test_start_requires_a_provisioned_vm(tmp_path, fake_host):
    with pytest.raises(vm.VmError, match="hermes vm create"):
        _manager(tmp_path, fake_host).start()


def test_start_applies_runtime_knobs_but_keeps_creation_knobs(tmp_path, fake_host):
    manager = _manager(tmp_path, fake_host)
    manager.create(vm.VmSpec(disk_gb=25, user="dev"), boot_timeout=10)
    manager.paths.pidfile.unlink()  # simulate a stopped VM
    fake_host.calls.clear()
    manager.start(vm.VmSpec(cpus=6, memory_mb=4096, network="none", disk_gb=99, user="other"), boot_timeout=10)
    argv = fake_host.calls[-1]
    assert argv[0] == "qemu" and argv[argv.index("-smp") + 1] == "6" and argv[argv.index("-m") + 1] == "4096M"
    assert "restrict=on" in argv[argv.index("-netdev") + 1]
    spec = manager.spec()
    assert (spec.disk_gb, spec.user, spec.cpus) == (25, "dev", 6)
    assert [c[0] for c in fake_host.calls] == ["qemu"]  # no re-provisioning


def test_boot_failure_when_qemu_exits_early(tmp_path, fake_host, monkeypatch):
    manager = _manager(tmp_path, fake_host)
    monkeypatch.setattr(vm.VmManager, "running", lambda self: False)
    with pytest.raises(vm.VmError, match="exited during boot"):
        manager.create(vm.VmSpec(), boot_timeout=10)


def test_destroy_removes_everything_but_the_lock(tmp_path, fake_host, monkeypatch):
    manager = _manager(tmp_path, fake_host)
    manager.create(vm.VmSpec(), boot_timeout=10)
    monkeypatch.setattr(vm.VmManager, "running", lambda self: False)
    manager.destroy()
    assert not manager.exists()
    assert [p.name for p in manager.paths.root.iterdir()] == [".lock"]


# --- provider ---------------------------------------------------------------------------------

def test_create_environment_without_vm_is_a_connection_error(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    with pytest.raises(EnvironmentConnectionError) as exc:
        QemuVmProvider().create_environment(cwd="/root", timeout=30)
    assert "hermes vm create" in exc.value.retry_hint


def test_create_environment_respects_autostart_off(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(vm.VmManager, "exists", lambda self: True)
    monkeypatch.setattr(vm.VmManager, "running", lambda self: False)
    provider = QemuVmProvider(settings=lambda key, default=None: False if key == "autostart" else default)
    with pytest.raises(EnvironmentConnectionError, match="autostart is disabled"):
        provider.create_environment(cwd="/root", timeout=30)


def test_provider_spec_reads_plugin_settings():
    values = {"cpus": 5, "network": "none", "disk_gb": 50}
    provider = QemuVmProvider(settings=lambda key, default=None: values.get(key, default))
    spec = provider.spec()
    assert (spec.cpus, spec.network, spec.disk_gb, spec.memory_mb) == (5, "none", 50, 2048)
    assert provider.autostart() is True and provider.boot_timeout() == 180


def test_probe_reports_host_blocker_before_vm_state(monkeypatch):
    import plugins.terminal_env.qemu_vm.provider as prov
    monkeypatch.setattr(prov, "host_requirements", lambda: [vm.HostRequirement(False, "/dev/kvm", "missing")])
    status, detail = QemuVmProvider().probe()
    assert status == "unavailable" and "/dev/kvm" in detail
