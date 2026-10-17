"""``qemu_vm`` terminal backend: a persistent, per-profile Debian VM on local QEMU/KVM.

The VM is reached over SSH on a loopback port, so the environment object is the built-in
:class:`SSHEnvironment` with the VM's own key and ``known_hosts`` injected — file sync,
persistent-shell snapshots and passthrough env all come for free.
"""

from __future__ import annotations

import logging
import shlex
from pathlib import Path
from typing import Any, Callable, Optional

from agent.terminal_env_provider import TerminalEnvironmentProvider
from hermes_constants import get_hermes_home, get_real_home, get_subprocess_home, profile_name_for_home
from tools.environments.base import EnvironmentConnectionError
from tools.environments.ssh import SSHEnvironment

from plugins.terminal_env.qemu_vm.manager import (
    BACKEND_NAME, SshTarget, VmError, VmManager, VmSpec, default_vm_root, host_ready, host_requirements,
    spec_from_mapping, ssh_base_argv)

logger = logging.getLogger(__name__)

SETTING_KEYS = ("distro", "user", "disk_gb", "passwordless_sudo", "cpus", "memory_mb", "network")
_CREATE_HINT = "Run `hermes vm create` (or `hermes setup terminal` and pick QEMU VM) to provision it."


def host_homes() -> tuple[Path, ...]:
    """Directories that are "home" on the machine running Hermes: the OS user's home and, in the
    official Docker image, the subprocess home (``/opt/data/home``)."""
    homes = [Path(get_real_home())]
    sub = get_subprocess_home()
    if sub and Path(sub) != homes[0]:
        homes.append(Path(sub))
    return tuple(homes)


def guest_cwd(cwd: Optional[str], homes: tuple[Path, ...]) -> str:
    """Map a cwd hint from the core onto the guest. ``/root`` is the factory default for plugin
    backends (the guest user is not root); a path under a host home is a host workspace the
    core registered (``terminal.cwd``, a desktop/TUI surface) and lands under the guest home.
    Everything else (``~``-prefixed, guest-absolute) passes through."""
    if not cwd or cwd == "/root":
        return "~"
    for home in homes:
        try:
            rel = Path(cwd).relative_to(home)
        except ValueError:
            continue
        return "~" if str(rel) == "." else f"~/{rel.as_posix()}"
    return cwd


class QemuVmEnvironment(SSHEnvironment):
    """SSH to the guest with the VM's key pair and per-VM known_hosts (never the user's).

    The core only remaps host cwds for container backends and for the built-in ``ssh`` name, so
    this class maps every cwd it is handed — at creation, on ``env.cwd`` writes (task overrides)
    and per command — onto the guest; otherwise the session wrapper's ``cd <host path>`` exits 126.
    """

    def __init__(self, target: SshTarget, **kwargs: Any):
        self._target = target
        self._homes = host_homes()
        self._ensured_dirs: set[str] = set()
        super().__init__(host=target.host, user=target.user, port=target.port, key_path=target.key_path, **kwargs)

    @property
    def cwd(self) -> str:
        return self._guest_cwd

    @cwd.setter
    def cwd(self, value: str) -> None:
        self._guest_cwd = guest_cwd(value, self._homes)

    def init_session(self):
        self._ensure_guest_dir(self.cwd)
        return super().init_session()

    def execute(self, command: str, cwd: str = "", **kwargs: Any) -> dict:
        target = guest_cwd(cwd, self._homes) if cwd else cwd
        if target:
            self._ensure_guest_dir(target)
        return super().execute(command, target, **kwargs)

    def _ensure_guest_dir(self, cwd: str) -> None:
        """A mapped host workspace (``~/<rel>``) rarely exists in a fresh guest, and the session
        wrapper's ``cd`` exits 126 when it is missing — create it once per path."""
        if not cwd.startswith("~/") or cwd in self._ensured_dirs:
            return
        self._run_ssh(f"mkdir -p $HOME/{shlex.quote(cwd[2:])}", timeout=10)
        self._ensured_dirs.add(cwd)

    def _target_flags(self, port_flag: str) -> list:
        # Shared by ssh (-p) and scp (-P): every connection carries the VM's trust settings.
        return [port_flag, str(self.port), *ssh_base_argv(self._target)]


def manager_for_active_profile(**kwargs: Any) -> VmManager:
    """One VM per profile: paths and label follow the profile whose scope is bound right now."""
    label = profile_name_for_home(get_hermes_home()) or "default"
    return VmManager(default_vm_root(), label=label, **kwargs)


class QemuVmProvider(TerminalEnvironmentProvider):
    is_remote = True
    is_container = False  # SSH-style path semantics: the guest has a normal home, not /workspace
    session_isolated_when_nonpersistent = False  # one durable VM per profile by design

    def __init__(self, settings: Optional[Callable[[str, Any], Any]] = None):
        self._settings = settings or (lambda key, default=None: default)

    # -- classification --

    @property
    def name(self) -> str:
        return BACKEND_NAME

    @property
    def display_name(self) -> str:
        return "QEMU VM"

    @property
    def description(self) -> str:
        return "Persistent per-profile Debian VM on local QEMU/KVM, reached over SSH (kernel isolation, no cloud)."

    @property
    def env_description(self) -> str:
        return "a local QEMU/KVM virtual machine (Debian Linux)"

    @property
    def skip_container_guards(self) -> bool:
        # The host filesystem is unreachable from the guest; destructive commands only affect
        # the disposable VM disk, exactly like the Docker sandbox.
        return True

    @property
    def cache_path_base(self) -> Optional[str]:
        return "~/.hermes"

    # -- settings --

    def spec(self) -> VmSpec:
        values = {key: self._settings(key, None) for key in SETTING_KEYS}
        return spec_from_mapping({k: v for k, v in values.items() if v is not None})

    def autostart(self) -> bool:
        return bool(self._settings("autostart", True))

    def boot_timeout(self) -> int:
        return int(self._settings("boot_timeout", 180) or 180)

    # -- health --

    def is_available(self) -> bool:
        return host_ready()

    def probe(self) -> tuple[str, str]:
        failing = [r for r in host_requirements() if not r.ok]
        if failing:
            return "unavailable", f"{failing[0].label}: {failing[0].detail}"
        manager = manager_for_active_profile()
        if not manager.exists():
            return "needs_setup", f"No VM provisioned yet. {_CREATE_HINT}"
        return "ready", "VM running" if manager.running() else "VM stopped (boots on first command)"

    def setup_instructions(self) -> list[str]:
        return [
            "Requires Linux with KVM (/dev/kvm), qemu-system-x86, qemu-utils, genisoimage and openssh-client.",
            "A checksum-pinned Debian cloud image (~400 MB) is downloaded once and cached for all profiles.",
            "Settings: plugins.entries.terminal_env/qemu_vm.settings.{cpus,memory_mb,disk_gb,network,...}",
            "Manage the VM with `hermes vm create|start|stop|status|destroy|ssh`.",
        ]

    def post_setup(self) -> None:
        from hermes_cli.setup import print_info, print_success, print_warning

        failing = [f"{r.label}: {r.detail}" for r in host_requirements() if not r.ok]
        if failing:
            print_warning("Host is not ready for the QEMU VM backend:")
            for line in failing:
                print_info(f"  - {line}")
            return
        manager = manager_for_active_profile(log=print_info)
        if manager.exists():
            print_success("VM already provisioned.")
            return
        try:
            manager.create(self.spec(), boot_timeout=max(self.boot_timeout(), 600))
            print_success("VM provisioned and reachable over SSH.")
        except (VmError, ValueError) as exc:
            print_warning(f"VM provisioning failed: {exc}")

    def doctor_checks(self) -> list[tuple[bool, str, str]]:
        rows = [(r.ok, f"QEMU VM: {r.label}", r.detail) for r in host_requirements()]
        manager = manager_for_active_profile()
        if manager.exists():
            rows.append((True, "QEMU VM: instance", "running" if manager.running() else f"stopped ({manager.paths.root})"))
        else:
            rows.append((False, "QEMU VM: instance", f"not provisioned — {_CREATE_HINT}"))
        return rows

    # -- environments --

    def create_environment(self, *, cwd: str, timeout: int, task_id: str = "default",
                           image: Optional[str] = None, container_config: Optional[dict[str, Any]] = None,
                           **kwargs: Any):
        manager = manager_for_active_profile()
        if not manager.exists():
            raise EnvironmentConnectionError(f"No QEMU VM provisioned for this profile ({manager.paths.root}).",
                                             retry_hint=_CREATE_HINT)
        if not manager.running():
            if not self.autostart():
                raise EnvironmentConnectionError("The QEMU VM is stopped and autostart is disabled.",
                                                 retry_hint="Run `hermes vm start`, then retry.")
            try:
                manager.start(self.spec(), boot_timeout=self.boot_timeout())
            except (VmError, ValueError) as exc:
                raise EnvironmentConnectionError(f"QEMU VM failed to start: {exc}",
                                                 retry_hint="Check `hermes vm status` and the console log, then retry.") from exc
        return QemuVmEnvironment(manager.ssh_target(), cwd=cwd, timeout=timeout)
