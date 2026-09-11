"""Per-conversation disposable Omarchy guests in QEMU/KVM.

A second realm kind beside :mod:`realms.manager`. Where a labwc realm separates
GUIs inside this login session, a VM realm is a separate machine: its own
kernel, disk and desktop. That is what makes `omarchy-shell` (one Hyprland and
one quickshell per session) hostable at all, and it is why breaking the guest
cannot break the host.

The guest itself is not ours. It is installed by Omarchy's own signed ISO
through the vendored ``omarchy vm`` script, and every session boots a
copy-on-write clone of one installed base image, so a chat costs 196 KiB and a
few seconds rather than a download and an install.

Isolation honesty, same as the labwc realm's: project files are *copied* in and
out, never mounted; the guest reaches the network through QEMU user-mode
networking unless ``plugins.realms.vm.network`` is false; the disk has no
encryption and passwordless sudo, so nothing secret may be put in it.
"""

from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import uuid

from .config import Config, effective_home, vm_data_path
from .lifecycle import (
    RealmError,
    OwnershipError,
    alive,
    atomic_json,
    host_control_env,
    scope_info,
)


VENDORED_SCRIPT = Path(__file__).resolve().with_name("vendor") / "omarchy-vm"
# Upstream omacom/omarchy PR #10977 `bin/omarchy-vm` as vendored, before the
# local patches recorded in vendor/VENDOR.md. Checked on every launch so a
# tampered or silently upgraded copy cannot be executed.
UPSTREAM_SHA256 = "19a51517c2713e033b2f703bfa387a79a9880c1c7a918e327e1327cfde47c3b6"
# The vendored copy *with* those patches applied. Pinned in code rather than in
# a file beside the script: a checksum an attacker can rewrite pins nothing.
VENDORED_SHA256 = "e811cade90b9e9341e767a8bc9ab2771b82d056350739322ea4acd43f41971c4"

# Ports the per-session guests forward SSH on, loopback only. A realm picks the
# first free one and records it; a collision fails the launch rather than
# silently attaching to another guest.
SSH_PORT_RANGE = range(2300, 2400)

# Guest-bound environment. The host process environment holds this profile's
# API keys and desktop handles; forwarding it into a disposable guest with an
# unencrypted disk would put them on that disk. Only terminal presentation
# crosses the boundary.
FORWARDED_ENV = ("TERM", "LANG", "LC_ALL", "COLUMNS", "LINES")


class VmError(RealmError):
    pass


def _generation_paths(uid, generation):
    return Path(f"/run/user/{uid}/hv-{generation[:16]}")


class VmRegistry:
    """Crash-safe per-profile VM records, beside the labwc realm registry."""

    def __init__(self, home):
        self.root = Path(home) / "realms"
        if self.root.is_symlink():
            raise OwnershipError("registry must not be a symlink")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.stat().st_uid != os.getuid():
            raise OwnershipError("registry has a foreign owner")
        os.chmod(self.root, 0o700)

    def lock(self):
        from .lifecycle import Registry

        return Registry(self.root.parent).lock()

    def path(self, vm_id):
        if not isinstance(vm_id, str) or not re.fullmatch(r"v-[0-9a-f]{24}", vm_id):
            raise VmError("invalid realm id")
        return self.root / (vm_id + ".json")

    def get(self, vm_id):
        try:
            value = json.loads(self.path(vm_id).read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise VmError("realm not found: " + vm_id) from exc
        if (
            value.get("id") != vm_id
            or value.get("uid") != os.getuid()
            or value.get("home") != str(self.root.parent)
        ):
            raise OwnershipError("registry ownership mismatch")
        validate_vm_record(value)
        return value

    def put(self, record):
        atomic_json(self.path(record["id"]), record)

    def records(self):
        return [self.get(p.stem) for p in sorted(self.root.glob("v-*.json"))]

    def remove(self, vm_id):
        self.path(vm_id).unlink(missing_ok=True)


def validate_vm_record(record):
    """Every identity in the record must be derivable from its generation.

    A record is the only thing standing between ``stop`` and a systemd unit, so
    a record that names a unit, runtime directory or socket it could not have
    created is refused rather than acted on.
    """
    generation = record.get("generation", "")
    if (
        not re.fullmatch(r"[0-9a-f]{32}", generation)
        or record.get("id") != "v-" + generation[:24]
    ):
        raise OwnershipError("invalid realm generation")
    runtime = _generation_paths(os.getuid(), generation)
    if (
        record.get("runtime_dir") != str(runtime)
        or record.get("unit") != f"hermes-vm-{generation}.service"
        or record.get("guardian_unit") != f"hermes-vm-{generation}-guard.service"
    ):
        raise OwnershipError("realm runtime or unit ownership mismatch")
    if record.get("status") == "running" and record.get("vnc_socket") != str(
        runtime / "vnc.sock"
    ):
        raise OwnershipError("realm VNC endpoint identity changed")
    if runtime.is_symlink():
        raise OwnershipError("realm runtime is a symlink")
    if runtime.exists():
        info = runtime.stat()
        if info.st_uid != os.getuid() or info.st_mode & 0o777 != 0o700:
            raise OwnershipError("realm runtime permissions changed")


def unit_active(unit):
    try:
        return scope_info(unit).get("ActiveState") == "active"
    except RealmError:
        return False


class VmManager:
    KIND = "omarchy-vm"

    def __init__(self, home=None, *, vm_id=None):
        self.home = effective_home(home)
        self.registry = VmRegistry(self.home)
        self.data = vm_data_path(self.home)
        if vm_id is None:
            self.config = Config.load(self.home)
        else:
            # Payload-only launchers attach to an already owned generation. Its
            # validated launch spec, not the host's current config, governs that
            # guest; importing the host core here would resolve against whichever
            # Hermes is on sys.path rather than the one that started the realm.
            self.config = Config(**json.loads(
                (Path(self.registry.get(vm_id)["session_dir"]) / "spec.json")
                .read_text(encoding="utf-8")))

    # --- vendored script ---

    def script(self):
        """The ``omarchy vm`` implementation to drive.

        The vendored copy is the default even when a system ``omarchy vm``
        exists: upstream opens an SDL window on the user's own desktop, which
        is exactly what a realm exists to avoid. ``vm.omarchy_vm_path`` lets an
        operator point at their own already-headless copy.
        """
        configured = self.config.vm.omarchy_vm_path
        if configured:
            path = Path(configured)
            if not os.access(path, os.X_OK):
                raise VmError("configured vm.omarchy_vm_path is not executable")
            return path
        return VENDORED_SCRIPT

    def _script_env(self, *, vm_home, ssh_port=None, unit=None, runtime=None):
        """Host-side environment for one vendored-script invocation.

        Built from nothing rather than inherited: the script runs ssh, qemu and
        systemctl, and a realm-sanitized or secret-bearing environment reaching
        any of them is a bug either way.
        """
        uid = os.getuid()
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(Path.home()),
            "USER": os.environ.get("USER") or Path.home().name,
            "XDG_RUNTIME_DIR": f"/run/user/{uid}",
            "DBUS_SESSION_BUS_ADDRESS": f"unix:path=/run/user/{uid}/bus",
            "OMARCHY_VM_HOME": str(vm_home),
            "OMARCHY_VM_ISO_DIR": str(self.data / "iso"),
            "OMARCHY_VM_MEMORY": str(self.config.vm.memory),
            "OMARCHY_VM_DISK_SIZE": self.config.vm.disk_size,
        }
        if ssh_port is not None:
            env["OMARCHY_VM_SSH_PORT"] = str(ssh_port)
        if unit is not None:
            env["OMARCHY_VM_UNIT"] = unit.removesuffix(".service")
        if runtime is not None:
            env["OMARCHY_VM_VNC_SOCKET"] = str(runtime / "vnc.sock")
            env["OMARCHY_VM_QMP_SOCKET"] = str(runtime / "qmp.sock")
        if not self.config.vm.network:
            # QEMU user networking with every outbound route refused. The guest
            # keeps its SSH forward, which is how we reach it at all.
            env["OMARCHY_VM_NETDEV_EXTRA"] = ",restrict=on"
        return env

    def _run_script(self, arguments, *, vm_home, ssh_port=None, unit=None,
                    runtime=None, timeout=120, check=True, stdout=None):
        script = self.script()
        if script == VENDORED_SCRIPT:
            import hashlib

            digest = hashlib.sha256(script.read_bytes()).hexdigest()
            if digest != VENDORED_SHA256:
                raise VmError("vendored omarchy-vm script does not match its pin")
        return subprocess.run(
            [str(script), *arguments],
            env=self._script_env(
                vm_home=vm_home, ssh_port=ssh_port, unit=unit, runtime=runtime
            ),
            capture_output=stdout is None,
            stdout=stdout,
            stderr=subprocess.PIPE if stdout is not None else None,
            text=stdout is None,
            timeout=float(timeout),
            check=check,
        )

    # --- base image ---

    def base_home(self):
        return self.data / "base"

    def base_status(self):
        disk = self.base_home() / "disk.qcow2"
        info = {"present": disk.is_file(), "path": str(disk)}
        if info["present"]:
            info["bytes"] = disk.stat().st_size
            info["built_at"] = disk.stat().st_mtime
            metadata = self.base_home() / "base.json"
            if metadata.is_file():
                try:
                    info.update(json.loads(metadata.read_text(encoding="utf-8")))
                except ValueError:
                    pass
        return info

    def install_base(self, *, iso=None, timeout=5400, stdout=None):
        """Install the shared base guest. Explicit, consented, never implicit.

        Downloads and signature-verifies the official ISO through the vendored
        script, installs unattended, provisions autologin and passwordless sudo
        in the guest, then powers it off so session clones start from a
        consistent disk.
        """
        base = self.base_home()
        if (base / "disk.qcow2").is_file():
            raise VmError("a base image already exists; remove it before rebuilding")
        base.mkdir(mode=0o700, parents=True, exist_ok=True)
        (self.data / "iso").mkdir(mode=0o700, parents=True, exist_ok=True)
        unit = "hermes-vm-base.service"
        runtime = Path(f"/run/user/{os.getuid()}/hv-base")
        runtime.mkdir(mode=0o700, exist_ok=True)
        port = self._free_port(set())
        try:
            self._run_script(
                ["install", *(["--iso", str(iso)] if iso else [])],
                vm_home=base, ssh_port=port, unit=unit, runtime=runtime,
                timeout=timeout, stdout=stdout,
            )
            self._run_script(
                ["stop"], vm_home=base, ssh_port=port, unit=unit, runtime=runtime,
                timeout=120, stdout=stdout,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            self._force_stop(unit)
            raise VmError("base image install failed: " + _reason(exc)) from exc
        finally:
            shutil.rmtree(runtime, ignore_errors=True)
        atomic_json(
            base / "base.json",
            {"built_at": time.time(), "iso": _installed_iso(self.data / "iso")},
        )
        return self.base_status()

    def remove_base(self):
        if self.registry.records():
            raise VmError("stop every VM realm before removing the base image")
        shutil.rmtree(self.base_home(), ignore_errors=True)
        return True

    def storage(self):
        def total(path):
            root = Path(path)
            if not root.exists():
                return 0
            return sum(
                f.stat().st_size for f in root.rglob("*") if f.is_file()
            )

        sessions = self.registry.root / "vm"
        return {
            "iso_bytes": total(self.data / "iso"),
            "base_bytes": total(self.base_home()),
            "session_bytes": total(sessions),
        }

    def latest_version(self, *, timeout=5):
        """Newest published Omarchy release, or None when it cannot be read.

        The same source the vendored script installs from, so a base built by
        that script is comparable to it. Best effort on purpose: a settings
        panel must still render with no network, and "unknown" is an honest
        answer that never fabricates an update prompt.
        """
        import json as _json
        import urllib.request

        request = urllib.request.Request(
            "https://api.github.com/repos/omacom/omarchy/releases/latest",
            headers={"Accept": "application/vnd.github+json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                tag = _json.loads(response.read(1 << 20)).get("tag_name") or ""
        except (OSError, ValueError):
            return None
        return tag.lstrip("v") or None

    def settings(self, *, check_updates=False):
        """The Desktop settings block: base image, storage, and the VM knobs.

        ``check_updates`` is opt-in because it costs a network round trip;
        rendering the panel must never depend on reaching GitHub.
        """
        base = self.base_status()
        installed = _iso_version(base.get("iso"))
        latest = self.latest_version() if check_updates else None
        return {
            "base": {**base, "version": installed},
            "storage": self.storage(),
            "memory_mb": self.config.vm.memory,
            "network": self.config.vm.network,
            "update": {
                "installed": installed,
                "latest": latest,
                # None means "not checked / unreachable", never "up to date".
                "available": None if (latest is None or installed is None)
                else latest != installed,
            },
        }

    # --- lifecycle ---

    def list(self):
        with self.registry.lock():
            self._reconcile_locked()
            return self.registry.records()

    def _reconcile_locked(self):
        """Retire records whose guest is gone, and guests nobody is using.

        A VM realm costs its full ``-m`` figure in host RAM for as long as it
        runs — there is no balloon device and the guest touches all of it. A
        conversation that ends without finalizing would otherwise pin that
        memory until the user notices, so the labwc realm's ``idle_ttl`` governs
        this kind too. Reconciliation runs on every list/start, which is the
        same cadence the registry is read at.
        """
        idle_ttl = self.config.idle_ttl
        now = time.time()
        for record in self.registry.records():
            if record["status"] == "running" and not unit_active(record["unit"]):
                self._remove_locked(record)
            elif (record["status"] == "running"
                  and now - record.get("last_activity", now) > idle_ttl):
                self._stop_locked(record)
            elif record["status"] not in ("running", "starting"):
                self._remove_locked(record)

    def _stop_locked(self, record):
        """Power the guest down cleanly, then drop it. Caller holds the lock."""
        try:
            self._run_script(
                ["stop"],
                vm_home=record["session_dir"], ssh_port=record["ssh_port"],
                unit=record["unit"], runtime=Path(record["runtime_dir"]),
                timeout=90, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        self._remove_locked(record)

    def _remove_locked(self, record):
        validate_vm_record(record)
        self._force_stop(record["guardian_unit"])
        self._force_stop(record["unit"].removesuffix(".service") + "-owner.service")
        self._force_stop(record["unit"])
        shutil.rmtree(record["runtime_dir"], ignore_errors=True)
        shutil.rmtree(record["session_dir"], ignore_errors=True)
        self.registry.remove(record["id"])

    def _force_stop(self, unit):
        try:
            subprocess.run(
                ["systemctl", "--user", "stop", unit],
                env=host_control_env(), capture_output=True, timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            pass

    def _free_port(self, taken):
        for port in SSH_PORT_RANGE:
            if port in taken:
                continue
            with socket.socket() as probe:
                try:
                    probe.bind(("127.0.0.1", port))
                except OSError:
                    continue
            return port
        raise VmError("no free loopback SSH port for a VM realm")

    def start(self, session_id):
        if not isinstance(session_id, str) or not session_id or len(session_id) > 256:
            raise ValueError("session_id must contain 1 to 256 characters")
        base_disk = self.base_home() / "disk.qcow2"
        if not base_disk.is_file():
            raise VmError(
                "No Omarchy base image in this profile. Build one explicitly with "
                "hermes realms vm install (downloads the signed ISO, ~5 GB, then "
                "installs unattended); host fallback is disabled."
            )
        with self.registry.lock():
            self._reconcile_locked()
            existing = [
                r for r in self.registry.records() if r["session_id"] == session_id
            ]
            for record in existing:
                if record["status"] == "running" and unit_active(record["unit"]):
                    record["last_activity"] = time.time()
                    self.registry.put(record)
                    # Adopting a guest makes THIS process its owner. The
                    # watcher still names whichever process started it — often
                    # a backend from a previous app run — so without re-binding
                    # the guest's 3 GB would answer to a PID that may be gone,
                    # and nothing would reclaim it.
                    self._bind_to_owner(record)
                    return record
                self._remove_locked(record)

            generation = uuid.uuid4().hex
            vm_id = "v-" + generation[:24]
            runtime = _generation_paths(os.getuid(), generation)
            runtime.mkdir(mode=0o700)
            session_dir = self.registry.root / "vm" / generation
            session_dir.mkdir(mode=0o700, parents=True)
            port = self._free_port(
                {r.get("ssh_port") for r in self.registry.records()}
            )
            record = {
                "id": vm_id,
                "session_id": session_id,
                "generation": generation,
                "kind": self.KIND,
                "uid": os.getuid(),
                "home": str(self.home),
                "runtime_dir": str(runtime),
                "session_dir": str(session_dir),
                "unit": f"hermes-vm-{generation}.service",
                "guardian_unit": f"hermes-vm-{generation}-guard.service",
                "status": "starting",
                "created_at": time.time(),
                "last_activity": time.time(),
                "ssh_port": port,
                "memory": self.config.vm.memory,
                "network": self.config.vm.network,
                "base_disk": str(base_disk),
            }
            self.registry.put(record)
            try:
                atomic_json(session_dir / "spec.json", asdict(self.config))
                self._clone_base(session_dir, base_disk)
                self._launch(record)
                record.update(
                    status="running",
                    vnc_socket=str(runtime / "vnc.sock"),
                    qmp_socket=str(runtime / "qmp.sock"),
                    invocation_id=scope_info(record["unit"])["InvocationID"],
                    cgroup=scope_info(record["unit"])["ControlGroup"],
                    last_activity=time.time(),
                )
                self.registry.put(record)
                return record
            except BaseException:
                self._remove_locked(record)
                raise

    def _clone_base(self, session_dir, base_disk):
        """Copy-on-write clone. 196 KiB and hundredths of a second per chat."""
        subprocess.run(
            [
                "qemu-img", "create", "-f", "qcow2",
                "-b", str(base_disk), "-F", "qcow2",
                str(session_dir / "disk.qcow2"),
            ],
            capture_output=True, text=True, timeout=60, check=True,
        )
        shutil.copyfile(
            self.base_home() / "OVMF_VARS.4m.fd", session_dir / "OVMF_VARS.4m.fd"
        )
        for name in ("credentials", "cidata.img"):
            source = self.base_home() / name
            if source.exists():
                shutil.copyfile(source, session_dir / name)
                os.chmod(session_dir / name, 0o600)

    def _launch(self, record):
        runtime = Path(record["runtime_dir"])
        self._run_script(
            ["launch"],
            vm_home=record["session_dir"], ssh_port=record["ssh_port"],
            unit=record["unit"], runtime=runtime,
            timeout=self.config.vm.boot_timeout + 30,
        )
        if not unit_active(record["unit"]):
            raise VmError("VM realm unit is not active after launch")
        # The viewer bridge only accepts a mode-0600 socket directly inside the
        # realm's own mode-0700 runtime directory.
        socket_path = runtime / "vnc.sock"
        if not socket_path.is_socket():
            raise VmError("VM realm did not publish its viewer socket")
        os.chmod(socket_path, 0o600)
        self._guard(record)
        self._bind_to_owner(record)

    def _bind_to_owner(self, record):
        """Tie the guest's lifetime to the agent process that started it.

        Reconciliation only runs while the plugin is being called, and
        ``unload`` only runs on a clean shutdown. A backend that is killed or
        crashes would otherwise strand a guest holding its full ``-m`` figure
        in host RAM (and a sleep inhibitor) until someone notices. A watcher
        scope outlives this call but not this PID, so systemd stops the guest
        when the owning process goes away, whatever killed it.

        A guest may be adopted by a later process (a new backend resuming the
        conversation), so the watcher is replaced rather than added to: systemd
        refuses a transient unit whose name is already loaded, and a silent
        refusal there would leave the guest bound to a dead PID forever.
        """
        owner_unit = record["unit"].removesuffix(".service") + "-owner.service"
        env = host_control_env()
        subprocess.run(
            ["systemctl", "--user", "stop", owner_unit],
            env=env, capture_output=True, text=True, timeout=30, check=False,
        )
        result = subprocess.run(
            [
                "systemd-run", "--user", "--quiet", "--collect",
                "--unit=" + owner_unit,
                "--property=BindsTo=" + record["unit"],
                "--property=After=" + record["unit"],
                "/bin/sh", "-c",
                # No `kill -0` race: the guest is stopped as soon as the owner is
                # gone, and the scope exits so systemd collects it.
                f'while kill -0 {os.getpid()} 2>/dev/null; do sleep 5; done; '
                f'exec systemctl --user stop {shlex.quote(record["unit"])}',
            ],
            env=env, capture_output=True, text=True,
            timeout=30, check=False,
        )
        if result.returncode != 0:
            # Losing the watcher means a crash strands the guest's full memory
            # figure, so this is worth a warning rather than silence.
            import logging

            logging.getLogger(__name__).warning(
                "VM realm owner watcher not installed for %s: %s",
                record["id"], (result.stderr or "").strip()[:200])

    def _guard(self, record):
        """Hold a sleep-only inhibitor for exactly as long as the guest runs.

        ``BindsTo`` makes systemd retire the inhibitor with the VM unit, so a
        crashed or stopped guest cannot leave the machine unable to suspend.
        """
        subprocess.run(
            [
                "systemd-run", "--user", "--quiet", "--collect",
                "--unit=" + record["guardian_unit"],
                "--property=BindsTo=" + record["unit"],
                "--property=After=" + record["unit"],
                "systemd-inhibit", "--what=sleep", "--who=Hermes Realms",
                "--why=Omarchy VM realm", "--mode=block",
                "sleep", "infinity",
            ],
            env=host_control_env(), capture_output=True, text=True,
            timeout=30, check=True,
        )

    def stop(self, vm_id):
        with self.registry.lock():
            if not self.registry.path(vm_id).exists():
                return False
            record = self.registry.get(vm_id)
            validate_vm_record(record)
            self._stop_locked(record)
            return True

    # --- routing surface (mirrors Manager) ---

    def validate(self, vm_id):
        with self.registry.lock():
            record = self.registry.get(vm_id)
            if record["status"] != "running":
                raise VmError("realm is not running")
            info = scope_info(record["unit"])
            if (
                info.get("ActiveState") != "active"
                or info.get("InvocationID") != record.get("invocation_id")
            ):
                raise OwnershipError("VM realm unit is not the owned live invocation")
            record["last_activity"] = time.time()
            self.registry.put(record)
            return record

    def env(self, vm_id):
        """Environment the routed terminal runs with on the *host* side.

        The command itself executes in the guest; these values only mark the
        routing and are what ``pre_tool`` and the escape guard read back.
        """
        record = self.validate(vm_id)
        return {
            "HERMES_REALM_KIND": self.KIND,
            "HERMES_REALM_ID": record["id"],
        }

    def command_prefix(self, vm_id):
        self.validate(vm_id)
        return (
            sys.executable,
            str(Path(__file__).with_name("vm_launch.py").resolve()),
            str(self.home),
            vm_id,
            "--",
        )

    def guest_home(self, vm_id):
        """The guest desktop user's home, read from the guest itself.

        Where a routed session starts. Asking the guest beats assuming a path:
        the user name comes from the host at install time, so it is not a
        constant we get to hardcode.
        """
        result = self.guest_run(
            vm_id, ["sh", "-c", "getent passwd 1000 | cut -d: -f6"], timeout=30)
        home = result["stdout"].strip()
        if result["returncode"] != 0 or not home.startswith("/"):
            raise VmError("could not resolve the guest user's home directory")
        return home

    def ssh_argv(self, record, *, user="root", tty=False):
        """SSH into this realm's guest, with the host's ssh config overridden.

        Every forwarding option is pinned OFF rather than left to defaults.
        ``ssh`` merges ``~/.ssh/config``, and a ``Host *`` block with
        ``ForwardAgent yes`` is a common setup: it would hand the user's
        private-key agent to a disposable guest that has passwordless sudo and
        no disk encryption, so anything running in there could authenticate as
        them. ``ForwardX11 yes`` is the same hazard pointed the other way — a
        route from the guest back to the host's display. Agent forwarding is a
        per-command, user-approved opt-in, never an ambient default.
        ``IdentitiesOnly`` keeps ssh from offering unrelated host keys.
        """
        key = Path.home() / ".ssh" / "id_ed25519"
        return [
            "ssh", "-p", str(record["ssh_port"]),
            *(["-tt"] if tty else ["-T"]),
            "-o", "ConnectTimeout=5",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "BatchMode=yes",
            "-o", "LogLevel=ERROR",
            "-o", "ForwardAgent=no",
            "-o", "ForwardX11=no",
            "-o", "ForwardX11Trusted=no",
            "-o", "IdentitiesOnly=yes",
            *(["-i", str(key)] if key.is_file() else []),
            f"{user}@127.0.0.1",
        ]

    def guest_run(self, vm_id, command, *, timeout=60, user="root"):
        """One captured command in the guest, as an argv list."""
        record = self.validate(vm_id)
        if (
            not isinstance(command, (list, tuple))
            or not command
            or not all(isinstance(a, str) and "\0" not in a for a in command)
        ):
            raise ValueError("command must be a nonempty argv list")
        result = subprocess.run(
            [*self.ssh_argv(record, user=user), "--", shlex.join(command)],
            capture_output=True, text=True, timeout=timeout,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def shot(self, vm_id, path):
        """grim inside the guest, over SSH. Never a host capture.

        Driven directly rather than through the vendored script's ``shot``: that
        writes the PNG to a named file and then echoes the file name on the same
        stdout, which corrupts a streamed capture.
        """
        record = self.validate(vm_id)
        target = Path(path).expanduser().resolve()
        session = (
            "export XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-1; "
            "export HYPRLAND_INSTANCE_SIGNATURE="
            "$(ls /run/user/1000/hypr 2>/dev/null | head -1); "
        )
        remote = "/tmp/hermes-realm-shot.png"
        with open(target, "wb") as sink:
            subprocess.run(
                [*self.ssh_argv(record), "--",
                 f"{session} grim {remote} && cat {remote} && rm -f {remote}"],
                stdout=sink, stderr=subprocess.PIPE, timeout=90, check=True,
            )
        data = target.read_bytes()
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            target.unlink(missing_ok=True)
            raise VmError("guest capture did not return a PNG")
        return str(target)

    def push(self, vm_id, source, destination=None):
        """Copy a host path into the guest. Copies, never mounts."""
        record = self.validate(vm_id)
        local = Path(source).expanduser().resolve()
        if not local.exists():
            raise ValueError("push source does not exist: " + str(local))
        remote = str(destination or local)
        self.guest_run(vm_id, ["mkdir", "-p", str(Path(remote).parent)])
        subprocess.run(
            [
                "scp", "-r", "-P", str(record["ssh_port"]),
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "BatchMode=yes", "-o", "LogLevel=ERROR",
                str(local), f"root@127.0.0.1:{remote}",
            ],
            capture_output=True, text=True, timeout=600, check=True,
        )
        return {"source": str(local), "destination": remote}

    def pull(self, vm_id, source, destination):
        """Copy a guest path out. Only on an explicit request."""
        record = self.validate(vm_id)
        local = Path(destination).expanduser().resolve()
        local.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "scp", "-r", "-P", str(record["ssh_port"]),
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "BatchMode=yes", "-o", "LogLevel=ERROR",
                f"root@127.0.0.1:{source}", str(local),
            ],
            capture_output=True, text=True, timeout=600, check=True,
        )
        return {"source": str(source), "destination": str(local)}

    def stats(self, vm_id):
        """What the guest costs, read from its own cgroup and disk."""
        record = self.registry.get(vm_id)
        cgroup = Path("/sys/fs/cgroup") / record.get("cgroup", "").lstrip("/")
        result: dict[str, int | None] = {
            "memory_bytes": None, "cpu_usec": None, "disk_bytes": None}
        try:
            result["memory_bytes"] = int(
                (cgroup / "memory.current").read_text(encoding="utf-8").strip()
            )
        except (OSError, ValueError):
            pass
        try:
            for line in (cgroup / "cpu.stat").read_text(encoding="utf-8").splitlines():
                if line.startswith("usage_usec "):
                    result["cpu_usec"] = int(line.split()[1])
        except (OSError, ValueError):
            pass
        try:
            result["disk_bytes"] = (
                Path(record["session_dir"]) / "disk.qcow2"
            ).stat().st_size
        except OSError:
            pass
        return result

    def doctor(self, vm_id=None):
        tools = {
            name: shutil.which(name)
            for name in ("qemu-system-x86_64", "qemu-img", "ssh", "scp", "socat",
                         "jq", "mcopy", "mkfs.vfat", "openssl", "systemd-run",
                         "systemd-inhibit")
        }
        firmware = {
            path: os.access(path, os.R_OK)
            for path in ("/usr/share/edk2/x64/OVMF_CODE.4m.fd",
                         "/usr/share/edk2/x64/OVMF_VARS.4m.fd")
        }
        key = Path.home() / ".ssh" / "id_ed25519.pub"
        base = self.base_status()
        report = {
            "kind": self.KIND,
            "platform": sys.platform,
            "tools": tools,
            "firmware": firmware,
            "kvm": os.access("/dev/kvm", os.R_OK | os.W_OK),
            "ssh_key": key.is_file(),
            "base_image": base,
            "script": str(self.script()),
            "security_boundary": (
                "separate kernel, disk and desktop; files are copied, not mounted. "
                "The guest disk is unencrypted with passwordless sudo: never put a "
                "secret in it"
            ),
        }
        report["missing"] = [
            name for name, found in tools.items() if not found
        ] + [
            path for path, ok in firmware.items() if not ok
        ] + (
            [] if report["kvm"] else ["/dev/kvm access (add your user to the kvm group)"]
        ) + (
            [] if report["ssh_key"] else ["~/.ssh/id_ed25519.pub"]
        ) + (
            [] if base["present"] else ["Omarchy base image (hermes realms vm install)"]
        )
        report["ok"] = sys.platform == "linux" and not report["missing"]
        if vm_id is not None:
            try:
                report["realm"] = self.validate(vm_id)
                report["stats"] = self.stats(vm_id)
                report["desktop"] = (
                    self.guest_run(vm_id, ["ls", "/run/user/1000/hypr"])["returncode"]
                    == 0
                )
                report["ok"] = report["ok"] and report["desktop"]
            except (RealmError, OSError, ValueError, subprocess.SubprocessError) as exc:
                report.update(ok=False, error=str(exc))
        return report


def _installed_iso(iso_dir):
    isos = sorted(Path(iso_dir).glob("omarchy-*.iso"))
    return isos[-1].name if isos else None


def _iso_version(name):
    """``omarchy-4.0.3.iso`` -> ``4.0.3``; None when there is no ISO recorded."""
    if not isinstance(name, str):
        return None
    match = re.fullmatch(r"omarchy-(.+)\.iso", name)
    return match.group(1) if match else None


def _reason(exc):
    output = getattr(exc, "stderr", None) or getattr(exc, "stdout", None) or ""
    return (output.strip().splitlines() or [str(exc)])[-1][:400]
