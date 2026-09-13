"""Read-only, subprocess-free proposals for explicitly consented setup."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import shutil
from types import SimpleNamespace

from .config import Config, KINDS, effective_home, vm_data_path
from .install_driver import profile_target

PACKAGES = {
    "labwc": "labwc", "Xwayland": "xorg-xwayland", "wayvnc": "wayvnc",
    "dbus-daemon": "dbus", "systemd-run": "systemd", "systemd-inhibit": "systemd",
    "grim": "grim", "wlr-randr": "wlr-randr", "gdbus": "glib2", "bwrap": "bubblewrap",
    "/usr/lib/at-spi-bus-launcher": "at-spi2-core",
    "/usr/lib/at-spi2-registryd": "at-spi2-core",
    "qemu-system-x86_64": "qemu-full", "qemu-img": "qemu-full",
    "ssh": "openssh", "scp": "openssh", "socat": "socat", "jq": "jq",
    "mcopy": "mtools", "mkfs.vfat": "dosfstools", "openssl": "openssl",
    "curl": "curl", "gpg": "gnupg", "pacman": "pacman",
    "qemu-full": "qemu-full", "edk2-ovmf": "edk2-ovmf", "mtools": "mtools",
}
NATIVE_TOOLS = tuple(PACKAGES)[:12]
VM_TOOLS = ("qemu-system-x86_64", "qemu-img", "ssh", "scp", "socat", "jq",
            "mcopy", "mkfs.vfat", "openssl", "systemd-run", "systemd-inhibit", "curl", "gpg", "pacman")


def confined(home, path):
    home, path = effective_home(home), Path(path)
    relative = path.relative_to(home)
    cursor = home
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("Realms setup paths must not contain symlink redirects")
        if cursor.exists() and cursor.stat().st_uid != os.getuid():  # windows-footgun: ok — Linux-only plugin
            raise PermissionError("Realms setup path has unsafe ownership")
    return path


def package_plan(missing, distribution):
    if missing and distribution != "arch":
        raise ValueError("Unsupported distribution: install prerequisites with your package manager")
    if any(name not in PACKAGES for name in missing):
        raise ValueError("Unsupported prerequisite")
    return sorted({PACKAGES[name] for name in missing})


def distribution():
    try:
        return "arch" if platform.freedesktop_os_release().get("ID") == "arch" else "unsupported"
    except OSError:
        return "unsupported"


def base_present(home):
    base = confined(home, vm_data_path(home) / "base")
    disk = confined(home, base / "disk.qcow2")
    receipt = confined(home, base / "base.json")
    try:
        metadata = json.loads(receipt.read_text(encoding="utf-8"))
        return disk.is_file() and disk.stat().st_size > 0 and isinstance(metadata.get("built_at"), (int, float))
    except (OSError, ValueError, AttributeError):
        return False


def setup_module():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "setup.py"))


def proposal_service():
    """Resolve persisted ownership without initializing runtime infrastructure."""
    from .integration import OwnerError, OwnershipStore

    home = effective_home(None)
    if not confined(home, home / "realms" / "sessions.sqlite3").is_file():
        raise OwnerError("An active conversation is required")
    owners = OwnershipStore(home, readonly=True)
    config = Config.load(home)
    return SimpleNamespace(home=home, owners=owners,
                           kind=lambda owner: owners.kind(owner, config.default_kind))


def build_plan(service, owner, kind):
    if kind not in KINDS:
        raise ValueError("Unsupported realm kind")
    home = effective_home(service.home)
    profile_target(home)
    confined(home, vm_data_path(home))
    config = Config.load(home)
    blockers = []
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        blockers.append("Realms requires Linux x86-64.")
    tools = NATIVE_TOOLS if kind == "realm" else VM_TOOLS
    missing = [name for name in tools if not (os.access(name, os.X_OK) if name.startswith("/") else shutil.which(name))]
    if kind == "realm":
        from . import install_driver
        description = setup_module()["describe"](home)
        ready = description["ready"]
        revision = description["revision"]
        details = [
            f"Pinned release: {install_driver.URL}",
            f"Driver destination: {profile_target(home)}",
            f"Archive SHA-256: {install_driver.ARCHIVE_SHA256}",
            f"Binary SHA-256: {install_driver.BINARY_SHA256}",
            "Re-verifies this profile's existing driver, or downloads the pinned release if needed.",
            "Starts this conversation's private desktop after verification.",
            "Does not change the global Cua driver, host desktop configuration, or tool approval settings.",
        ]
        action = "start" if ready else "repair"
    else:
        from .vm_manager import VENDORED_SHA256
        # Check installed package metadata without invoking pacman during prepare.
        database = Path("/var/lib/pacman/local")
        for package in ("qemu-full", "edk2-ovmf", "mtools"):
            if not any(database.glob(package + "-[0-9]*")):
                missing.append(package)
        key = Path.home() / ".ssh" / "id_ed25519"
        if not key.is_file() or not key.with_suffix(".pub").is_file():
            blockers.append("An existing SSH id_ed25519 key pair is required. Setup will not create or replace your keys.")
        if not os.access("/dev/kvm", os.R_OK | os.W_OK):
            blockers.append("KVM access is required; enable virtualization and grant your user access before retrying.")
        if config.vm.omarchy_vm_path:
            blockers.append("In-app installation requires the pinned vendored VM installer; remove the custom installer override first.")
        ready = base_present(home) and not missing and not blockers
        action = "start" if base_present(home) else "install"
        revision = VENDORED_SHA256
        details = ["Downloads the official Omarchy ISO (~6 GB) from https://iso.omarchy.org and verifies its release signature.",
                   "Signer: 40DFB630FF42BCFFB047046CF0134EE680CAC571",
                   f"Virtual disk: {config.vm.disk_size}; guest memory: {config.vm.memory} MiB. Base and disposable copies consume additional local storage.",
                   "Installs a profile-local base disk, then boots a disposable copy for this conversation.",
                   "The guest has network access when enabled, an unencrypted disk and passwordless sudo. Never put secrets in it.",
                   "Uses your existing SSH key for guest authentication; does not create or replace host keys."]
    try:
        packages = package_plan(missing, distribution())
    except ValueError:
        packages = []
        blockers.append("Automatic package installation supports Arch Linux only. Install missing prerequisites with your distribution's package manager.")
    if packages:
        details.append("Requests administrator consent to install: " + ", ".join(packages) + ".")
        if not shutil.which("pkexec"):
            blockers.append("Polkit pkexec is required for administrator package installation.")
    details.extend(blockers)
    fingerprint = hashlib.sha256(json.dumps({"version": 1, "setup": revision, "config": asdict(config)}, sort_keys=True).encode()).hexdigest()
    return {"home": str(home), "owner": owner, "kind": kind, "revision": fingerprint, "config": asdict(config),
            "ready": ready and not missing and not blockers, "action": action,
            "summary": {"repair": "Repair and start Realms", "install": "Install and start Omarchy VM", "start": "Start this conversation's realm"}[action],
            "details": details, "packages": packages, "blockers": blockers,
            "previous_kind": service.kind(owner)}
