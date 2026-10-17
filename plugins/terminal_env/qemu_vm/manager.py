"""QEMU/KVM virtual-machine lifecycle for the ``qemu_vm`` terminal backend.

One persistent VM per Hermes profile under ``<sandboxes>/qemu_vm/``: a self-contained qcow2
disk (converted from a checksum-pinned Debian cloud image, never a backing-file overlay so a
pruned image cache cannot brick the VM), a cloud-init NoCloud seed ISO, a dedicated ed25519 key
pair and a per-VM ``known_hosts``. The guest is reached only through an SSH port forwarded on
``127.0.0.1``; the VM is never exposed on the network.

Everything that touches the host (qemu, qemu-img, ISO tools, ssh, downloads) goes through
injectable callables so the lifecycle is testable without a hypervisor.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.request
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Iterator, Optional

import hermes_yaml as yaml
import psutil

from hermes_constants import get_default_hermes_root
from hermes_platform.host.facts import native_arch, os_family
from hermes_platform.resolver import locate_command

logger = logging.getLogger(__name__)

BACKEND_NAME = "qemu_vm"
NETWORK_MODES = ("egress", "none")
SUPPORTED_DISTROS = ("debian",)
HOST_KEY_ALIAS = "hermes-qemu-vm"
_STATE_FILE = "vm.json"
_CHUNK = 1 << 20
_PROGRESS_EVERY = 64 << 20


@dataclass(frozen=True)
class ImageSpec:
    """A checksum-pinned cloud image. URL and digest are code constants on purpose: no config
    value (which a model-authored command could edit) may redirect the download."""

    distro: str
    arch: str
    url: str
    sha512: str

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


# Dated Debian cloud builds, never ``latest/`` (its digest changes under us).
IMAGES: dict[tuple[str, str], ImageSpec] = {
    ("debian", "amd64"): ImageSpec(
        distro="debian", arch="amd64",
        url="https://cloud.debian.org/images/cloud/trixie/20261001-2618/"
            "debian-13-genericcloud-amd64-20261001-2618.qcow2",
        sha512="f46f0671a6e5bdec5291ab8972bae2f10e5408c2f64a74078f11efc2f06a436a"
               "9d0313ed50e0472542eeabf780e9f7c792ac0a314c6c20507fcd9fd81b468c3d"),
}

_QEMU_BINARY = {"amd64": "qemu-system-x86_64"}


@dataclass(frozen=True)
class VmSpec:
    """Creation-time knobs (fixed for the VM's life) plus runtime knobs re-read on every start."""

    distro: str = "debian"
    user: str = "hermes"
    disk_gb: int = 20
    passwordless_sudo: bool = True
    cpus: int = 2
    memory_mb: int = 2048
    network: str = "egress"

    def validate(self) -> None:
        if self.distro not in SUPPORTED_DISTROS:
            raise ValueError(f"Unsupported distro {self.distro!r}; supported: {', '.join(SUPPORTED_DISTROS)}")
        if self.network not in NETWORK_MODES:
            raise ValueError(f"Unsupported network mode {self.network!r}; use one of: {', '.join(NETWORK_MODES)}")
        if not self.user.isidentifier() or self.user == "root":
            raise ValueError(f"Invalid guest user {self.user!r}")
        for label, value, floor in (("cpus", self.cpus, 1), ("memory_mb", self.memory_mb, 256), ("disk_gb", self.disk_gb, 4)):
            if isinstance(value, bool) or not isinstance(value, int) or value < floor:
                raise ValueError(f"{label} must be an integer >= {floor} (got {value!r})")


def spec_from_mapping(values: dict, base: Optional[VmSpec] = None) -> VmSpec:
    """*base* overridden by the known keys of *values* (unknown keys ignored)."""
    known = {k: v for k, v in values.items() if k in VmSpec.__dataclass_fields__}
    return replace(base or VmSpec(), **known)


@dataclass(frozen=True)
class SshTarget:
    host: str
    port: int
    user: str
    key_path: str
    known_hosts: str


@dataclass(frozen=True)
class HostRequirement:
    ok: bool
    label: str
    detail: str


class VmError(RuntimeError):
    """A lifecycle step failed; the message is user-facing."""


def image_for(distro: str, arch: str) -> ImageSpec:
    try:
        return IMAGES[(distro, arch)]
    except KeyError:
        raise VmError(f"No pinned {distro} image for architecture {arch!r} "
                      f"(supported: {', '.join(f'{d}/{a}' for d, a in IMAGES)})") from None


def _tool(name: str) -> Optional[str]:
    cmd = locate_command(name).command
    return cmd[0] if cmd else None


def require_tool(name: str) -> str:
    path = _tool(name)
    if path is None:
        raise VmError(f"{name} not found on PATH")
    return path


def iso_tool() -> Optional[list[str]]:
    """Argv template that builds a NoCloud seed ISO, or None. ``genisoimage``/``xorrisofs``/``mkisofs``
    share the flag set; ``cloud-localds`` (cloud-image-utils) is the fallback with its own argv."""
    for name in ("genisoimage", "xorrisofs", "mkisofs"):
        if path := _tool(name):
            return [path, "-quiet", "-output", "{iso}", "-volid", "cidata", "-joliet", "-rock", "{user_data}", "{meta_data}"]
    if path := _tool("cloud-localds"):
        return [path, "{iso}", "{user_data}", "{meta_data}"]
    return None


def host_requirements() -> list[HostRequirement]:
    """Cheap, no-network checks for ``probe()``/``doctor``. KVM is Linux-only and the VM boots only
    with hardware acceleration: TCG would be unusably slow, so it is refused rather than degraded."""
    arch = native_arch()
    qemu_name = _QEMU_BINARY.get(arch)
    arch_ok = qemu_name is not None and ("debian", arch) in IMAGES
    kvm_ok = os.access("/dev/kvm", os.R_OK | os.W_OK)
    rows = [
        HostRequirement(os_family() == "linux", "Linux host", "KVM is only available on Linux"),
        HostRequirement(arch_ok, "Host architecture", f"{arch} ({'supported' if arch_ok else 'no pinned image'})"),
        HostRequirement(kvm_ok, "/dev/kvm", "accessible" if kvm_ok else
                        "missing or not writable (install qemu-kvm and add your user to the kvm group)"),
    ]
    for name, hint in ((qemu_name or "qemu-system-x86_64", "apt install qemu-system-x86"),
                       ("qemu-img", "apt install qemu-utils"),
                       ("ssh", "apt install openssh-client"), ("ssh-keygen", "apt install openssh-client")):
        path = _tool(name)
        rows.append(HostRequirement(path is not None, name, path or f"not found ({hint})"))
    iso = iso_tool()
    rows.append(HostRequirement(iso is not None, "ISO tool", iso[0] if iso else
                                "none of genisoimage/xorrisofs/mkisofs/cloud-localds found (apt install genisoimage)"))
    return rows


def host_ready() -> bool:
    return all(r.ok for r in host_requirements())


# --- pure builders (unit-tested without a hypervisor) ---------------------------------------

def build_user_data(spec: VmSpec, public_key: str) -> str:
    """cloud-init ``#cloud-config``: one sudo user with the VM's key, password auth off, sshd
    accepting the env names Hermes forwards (``terminal.env_passthrough`` rides ``SendEnv``)."""
    user: dict = {
        "name": spec.user, "shell": "/bin/bash", "groups": ["sudo"], "lock_passwd": True,
        "ssh_authorized_keys": [public_key.strip()],
    }
    if spec.passwordless_sudo:
        user["sudo"] = ["ALL=(ALL) NOPASSWD:ALL"]
    doc = {
        "hostname": "hermes-vm",
        "users": [user],
        "ssh_pwauth": False,
        "package_update": True,
        "packages": ["git", "curl", "ca-certificates", "python3"],
        "growpart": {"mode": "auto", "devices": ["/"]},
        "write_files": [{"path": "/etc/ssh/sshd_config.d/hermes.conf", "content": "AcceptEnv *\n"}],
        "runcmd": [["systemctl", "restart", "ssh"]],
    }
    return "#cloud-config\n" + yaml.safe_dump(doc, sort_keys=False)


def build_meta_data(instance_id: str) -> str:
    return f"instance-id: {instance_id}\nlocal-hostname: hermes-vm\n"


def build_qemu_argv(qemu: str, spec: VmSpec, paths: VmPaths, port: int, *, label: str) -> list[str]:
    """Headless q35/KVM guest. User-mode networking: NAT out (``egress``) or ``restrict=on``
    (``none``: no outbound at all); the SSH ``hostfwd`` on loopback works in both modes.

    ``-display none`` keeps the default VGA device: with ``-vga none`` the Debian cloud image's
    GRUB has no video BIOS to talk to and reboots before loading the kernel (observed boot loop)."""
    netdev = f"user,id=net0,hostfwd=tcp:127.0.0.1:{port}-:22"
    if spec.network == "none":
        netdev += ",restrict=on"
    return [
        qemu, "-name", f"hermes-vm-{label}", "-machine", "q35,accel=kvm", "-cpu", "host",
        "-smp", str(spec.cpus), "-m", f"{spec.memory_mb}M",
        "-drive", f"file={paths.disk},if=virtio,format=qcow2,discard=unmap",
        "-cdrom", str(paths.seed_iso),
        "-netdev", netdev, "-device", "virtio-net-pci,netdev=net0",
        "-device", "virtio-rng-pci",
        "-display", "none", "-serial", f"file:{paths.console_log}",
        "-monitor", f"unix:{paths.monitor_sock},server,nowait",
        "-pidfile", str(paths.pidfile), "-daemonize",
    ]


def ssh_base_argv(target: SshTarget) -> list[str]:
    """Options every connection to the guest needs. ``HostKeyAlias`` keeps the per-VM known_hosts
    entry stable across port reallocation; the user's own ``~/.ssh/known_hosts`` is never touched."""
    return ["-o", f"UserKnownHostsFile={target.known_hosts}", "-o", f"HostKeyAlias={HOST_KEY_ALIAS}",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "IdentitiesOnly=yes", "-i", target.key_path]


def sha512_of(path: Path) -> str:
    digest = hashlib.sha512()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def download_file(url: str, dest: Path, progress: Optional[Callable[[int, int], None]] = None) -> None:
    """Stream *url* (a pinned https URL from :data:`IMAGES`) to ``dest.part`` and rename at the end,
    so an interrupted download never looks complete."""
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        for chunk in iter(lambda: resp.read(_CHUNK), b""):
            out.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)
    os.replace(part, dest)


# --- paths & state --------------------------------------------------------------------------

@dataclass(frozen=True)
class VmPaths:
    root: Path

    @property
    def disk(self) -> Path:
        return self.root / "disk.qcow2"

    @property
    def seed_iso(self) -> Path:
        return self.root / "seed.iso"

    @property
    def key(self) -> Path:
        return self.root / "id_ed25519"

    @property
    def public_key(self) -> Path:
        return self.root / "id_ed25519.pub"

    @property
    def pidfile(self) -> Path:
        return self.root / "qemu.pid"

    @property
    def monitor_sock(self) -> Path:
        return self.root / "monitor.sock"

    @property
    def console_log(self) -> Path:
        return self.root / "console.log"

    @property
    def known_hosts(self) -> Path:
        return self.root / "known_hosts"

    @property
    def state(self) -> Path:
        return self.root / _STATE_FILE

    @property
    def lock(self) -> Path:
        return self.root / ".lock"


def default_vm_root() -> Path:
    from tools.environments.base import get_sandbox_dir
    return get_sandbox_dir() / BACKEND_NAME


def default_image_cache() -> Path:
    """Shared across profiles: the base image is identical for every VM and ~400 MB."""
    return get_default_hermes_root() / "sandboxes" / BACKEND_NAME / "images"


def run_checked(argv: list[str], *, timeout: int = 600, what: str) -> subprocess.CompletedProcess:
    result = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=timeout, stdin=subprocess.DEVNULL, check=False)
    if result.returncode != 0:
        raise VmError(f"{what} failed (rc={result.returncode}): {(result.stderr or result.stdout).strip()}")
    return result


def _ssh_probe(argv: list[str]) -> bool:
    try:
        return subprocess.run(argv, capture_output=True, timeout=900, stdin=subprocess.DEVNULL, check=False).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


class VmManager:
    """Lifecycle of the one VM under *root*. ``runner`` executes host argv with :func:`run_checked`
    semantics (raise :class:`VmError` on failure); ``downloader`` fetches the base image;
    ``ssh_probe`` returns whether a guest command succeeded."""

    def __init__(self, root: Path, *, image_cache: Optional[Path] = None, label: str = "default",
                 runner: Callable = run_checked, downloader: Callable = download_file,
                 ssh_probe: Callable[[list[str]], bool] = _ssh_probe, sleep: Callable = time.sleep,
                 log: Callable[[str], None] = logger.info):
        self.paths = VmPaths(root)
        self.image_cache = image_cache or default_image_cache()
        self.label = label
        self._run = runner
        self._download = downloader
        self._ssh_probe = ssh_probe
        self._sleep = sleep
        self._log = log

    # -- state --

    def exists(self) -> bool:
        return self.paths.state.is_file() and self.paths.disk.is_file()

    def read_state(self) -> dict:
        if not self.paths.state.is_file():
            return {}
        try:
            return json.loads(self.paths.state.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return {}

    def _write_state(self, state: dict) -> None:
        tmp = self.paths.state.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.paths.state)

    def spec(self) -> VmSpec:
        return spec_from_mapping(self.read_state().get("spec") or {})

    def pid(self) -> Optional[int]:
        try:
            return int(self.paths.pidfile.read_text(encoding="utf-8-sig").strip())
        except (OSError, ValueError):
            return None

    def running(self) -> bool:
        """Alive pid AND a monitor socket that accepts connections: a stale pidfile after a crash
        or a recycled pid never counts as running."""
        pid = self.pid()
        if pid is None or not psutil.pid_exists(pid):
            return False
        sock = self._monitor_connect()
        if sock is None:
            return False
        sock.close()
        return True

    def ssh_target(self) -> SshTarget:
        state = self.read_state()
        return SshTarget(host="127.0.0.1", port=int(state.get("port") or 0), user=self.spec().user,
                         key_path=str(self.paths.key), known_hosts=str(self.paths.known_hosts))

    def status(self) -> dict:
        state = self.read_state()
        return {"exists": self.exists(), "running": self.running(), "root": str(self.paths.root),
                "port": state.get("port"), "spec": asdict(self.spec()) if state else None,
                "image": state.get("image")}

    # -- lifecycle --

    def create(self, spec: VmSpec, *, force: bool = False, boot_timeout: int = 600) -> None:
        spec.validate()
        if self.exists() and not force:
            raise VmError(f"VM already exists at {self.paths.root}; use --force to recreate (destroys its disk)")
        self._require_host()
        with self._locked():
            if self.exists():
                self._stop_locked()
                self._wipe()
            image = self._ensure_image(image_for(spec.distro, native_arch()))
            qemu_img = require_tool("qemu-img")
            self._log("Preparing disk ...")
            self._run([qemu_img, "convert", "-O", "qcow2", str(image), str(self.paths.disk)], what="qemu-img convert")
            self._run([qemu_img, "resize", str(self.paths.disk), f"{spec.disk_gb}G"], what="qemu-img resize")
            self._make_key()
            self._make_seed(spec)
            self._write_state({"spec": asdict(spec), "image": image.name, "port": None,
                               "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            self._start_locked(spec, boot_timeout=boot_timeout, first_boot=True)

    def start(self, runtime: Optional[VmSpec] = None, *, boot_timeout: int = 180) -> None:
        """Boot if not running. *runtime* supplies the hot knobs (cpus/memory/network); creation
        knobs always come from the stored spec."""
        if not self.exists():
            raise VmError(f"No VM at {self.paths.root}. Run `hermes vm create` first.")
        if self.running():
            return
        self._require_host()
        spec = self.spec()
        if runtime is not None:
            spec = replace(spec, cpus=runtime.cpus, memory_mb=runtime.memory_mb, network=runtime.network)
        spec.validate()
        with self._locked():
            if not self.running():
                self._start_locked(spec, boot_timeout=boot_timeout, first_boot=False)

    def stop(self, *, timeout: int = 30) -> None:
        with self._locked():
            self._stop_locked(timeout=timeout)

    def destroy(self) -> None:
        with self._locked():
            self._stop_locked()
            self._wipe()

    # -- internals --

    def _require_host(self) -> None:
        missing = [f"{r.label}: {r.detail}" for r in host_requirements() if not r.ok]
        if missing:
            raise VmError("Host cannot run the QEMU VM backend:\n  - " + "\n  - ".join(missing))

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        """Cross-process exclusive lock: two sessions of one profile must not both boot the VM."""
        import fcntl  # Linux-only like KVM itself; lazy so discovery can import this module anywhere

        self.paths.root.mkdir(parents=True, exist_ok=True)
        with open(self.paths.lock, "w", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def _ensure_image(self, image: ImageSpec) -> Path:
        self.image_cache.mkdir(parents=True, exist_ok=True)
        dest = self.image_cache / image.filename
        if dest.is_file() and sha512_of(dest) == image.sha512:
            return dest
        self._log(f"Downloading {image.filename} ...")
        self._download(image.url, dest, self._progress)
        digest = sha512_of(dest)
        if digest != image.sha512:
            dest.unlink(missing_ok=True)
            raise VmError(f"Checksum mismatch for {image.filename}: expected {image.sha512[:16]}..., got {digest[:16]}...")
        return dest

    def _progress(self, done: int, total: int) -> None:
        if done == total or done // _PROGRESS_EVERY != (done - _CHUNK) // _PROGRESS_EVERY:
            self._log(f"  {done >> 20} / {total >> 20} MB")

    def _make_key(self) -> None:
        self.paths.key.unlink(missing_ok=True)
        self.paths.public_key.unlink(missing_ok=True)
        self._run([require_tool("ssh-keygen"), "-q", "-t", "ed25519", "-N", "", "-C",
                   f"hermes-{BACKEND_NAME}-{self.label}", "-f", str(self.paths.key)], what="ssh-keygen")

    def _make_seed(self, spec: VmSpec) -> None:
        tool = iso_tool()
        if tool is None:
            raise VmError("No ISO tool found (install genisoimage or cloud-image-utils)")
        user_data = self.paths.root / "user-data"
        meta_data = self.paths.root / "meta-data"
        public_key = self.paths.public_key.read_text(encoding="utf-8-sig")
        user_data.write_text(build_user_data(spec, public_key), encoding="utf-8")
        meta_data.write_text(build_meta_data(f"hermes-{self.label}-{int(time.time())}"), encoding="utf-8")
        fill = {"iso": str(self.paths.seed_iso), "user_data": str(user_data), "meta_data": str(meta_data)}
        self._run([arg.format(**fill) for arg in tool], what="seed ISO build")
        self.paths.known_hosts.unlink(missing_ok=True)  # new instance => new host key

    def _start_locked(self, spec: VmSpec, *, boot_timeout: int, first_boot: bool) -> None:
        state = self.read_state()
        port = int(state.get("port") or 0)
        if not port or not _port_free(port):
            port = free_loopback_port()
        qemu = require_tool(_QEMU_BINARY[native_arch()])
        self.paths.pidfile.unlink(missing_ok=True)
        self.paths.monitor_sock.unlink(missing_ok=True)
        self._log(f"Starting VM ({spec.cpus} vCPU, {spec.memory_mb} MB RAM, network={spec.network}) ...")
        self._run(build_qemu_argv(qemu, spec, self.paths, port, label=self.label), timeout=60, what="qemu-system")
        state.update({"port": port, "spec": asdict(spec)})
        self._write_state(state)
        self._wait_for_ssh(boot_timeout, first_boot=first_boot)

    def _wait_for_ssh(self, timeout: int, *, first_boot: bool) -> None:
        """Poll the forwarded port until the guest answers; on first boot also wait for cloud-init
        so packages are installed before the agent's first command."""
        target = self.ssh_target()
        base = [require_tool("ssh"), "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", *ssh_base_argv(target),
                "-p", str(target.port), f"{target.user}@{target.host}"]
        command = ["cloud-init", "status", "--wait"] if first_boot else ["true"]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.running():
                raise VmError(f"QEMU exited during boot; see {self.paths.console_log}")
            if self._ssh_probe(base + command):
                self._log("VM is up.")
                return
            self._sleep(2)
        raise VmError(f"VM did not answer on SSH within {timeout}s; see {self.paths.console_log}")

    def _monitor_connect(self) -> Optional[socket.socket]:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(2)
        try:
            sock.connect(str(self.paths.monitor_sock))
        except OSError:
            sock.close()
            return None
        return sock

    def _monitor_send(self, command: str) -> bool:
        sock = self._monitor_connect()
        if sock is None:
            return False
        try:
            sock.sendall((command + "\n").encode())
            return True
        except OSError:
            return False
        finally:
            sock.close()

    def _stop_locked(self, *, timeout: int = 30) -> None:
        pid = self.pid()
        if pid is None or not self.running():
            self.paths.pidfile.unlink(missing_ok=True)
            return
        self._log("Stopping VM ...")
        self._monitor_send("system_powerdown")
        if not self._wait_exit(pid, timeout):
            self._monitor_send("quit")
            if not self._wait_exit(pid, 5):
                os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
                self._wait_exit(pid, 5)
        self.paths.pidfile.unlink(missing_ok=True)
        self.paths.monitor_sock.unlink(missing_ok=True)

    def _wait_exit(self, pid: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not psutil.pid_exists(pid):
                return True
            self._sleep(0.5)
        return False

    def _wipe(self) -> None:
        if not self.paths.root.is_dir():
            return
        for child in self.paths.root.iterdir():
            if child.name == self.paths.lock.name:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)
