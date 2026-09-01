"""Binary acquisition for the managed llama.cpp runtime.

llama.cpp publishes per-tag assets (rolling ``bNNNN`` tags, no semver).
Backends are dlopen'd plugins, so a runtime = CPU/base zip + backend zip
extracted into one directory, plus the cudart runtime zip on Windows CUDA
(end users have no CUDA toolkit). We pin the tag in config, sha256-verify
every download, and keep the previous tag for rollback (N-1).

Layout: ``$HERMES_HOME/runtimes/llamacpp/<tag>/<backend>/<binaries>``
with a ``manifest.json`` recording zips, sha256s, and the verified
llama-server version string.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import shutil
import subprocess
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

RELEASE_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{asset}"

# Windows CUDA zips ship per CUDA major; the runtime zip must be paired with
# its cudart zip so end users need no toolkit. 13.3 verified on 13.1 and
# 13.2 drivers.
_WIN_CUDA_VERSION = "13.3"
# arm64 Windows CUDA prebuilts landed upstream (~b1036x) on CUDA 13.4 —
# verified against live asset lists (b10362, b10630, b10679). Tags at or before
# b10290 don't have them; resolution succeeds and the download 404s
# honestly on such tags, which only arises if a user pins backward.
_WIN_CUDA_VERSION_ARM64 = "13.4"


# Fallback when the config section is missing entirely (deep-merge normally
# guarantees the key). Single source: DEFAULT_CONFIG owns the shipped tag.
def default_tag() -> str:
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    return DEFAULT_CONFIG["local_runtime"]["tag"]


class BinaryResolutionError(RuntimeError):
    """No usable asset combination for this platform/backend."""


@dataclass
class AssetPlan:
    """The exact zips one runtime install needs, in extraction order."""

    tag: str
    backend: str            # cuda | metal | vulkan | hip | cpu
    assets: list[str] = field(default_factory=list)

    @property
    def install_dir(self) -> Path:
        return runtimes_root() / self.tag / self.backend


def runtimes_root() -> Path:
    """Machine-scoped, deliberately NOT profile-scoped. Engine binaries,
    presets, and server state describe this machine's hardware and its one
    managed server (stable port) — a second profile re-downloading the
    engine or fighting over the port would be the bug. Profile-scoped
    things (which model is the default, enabled) live in each profile's
    config.yaml as ever."""
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root() / "runtimes" / "llamacpp"


def installed_tags() -> list[str]:
    """Tags with a verified install (manifest carries verified_version),
    newest first by release number. The boot ladder and the update check
    both read installed-ness from here — one resolver, every caller."""
    root = runtimes_root()
    if not root.exists():
        return []
    found: list[str] = []
    for entry in root.iterdir():
        if not entry.is_dir() or entry.name == "downloads":
            continue
        for manifest in entry.glob("*/manifest.json"):
            try:
                if json.loads(manifest.read_text(encoding="utf-8")).get("verified_version"):
                    found.append(entry.name)
                    break
            except (json.JSONDecodeError, OSError):
                continue

    def _release_number(tag: str) -> int:
        digits = "".join(ch for ch in tag if ch.isdigit())
        return int(digits) if digits else 0

    return sorted(set(found), key=_release_number, reverse=True)


def _host_os_arch() -> tuple[str, str]:
    """(os, arch) normalized to release-asset vocabulary.

    PITFALL: PROCESSOR_ARCHITECTURE lies under x64 emulation on
    ARM64 Windows. platform.machine() reads the same env on some Pythons, so
    on Windows prefer PROCESSOR_IDENTIFIER's text when present.
    """
    system = platform.system().lower()
    os_name = {"windows": "win", "darwin": "macos", "linux": "ubuntu"}.get(system, system)
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
    if os_name == "win":
        import os as _os
        ident = _os.environ.get("PROCESSOR_IDENTIFIER", "")
        if "armv8" in ident.lower() or "arm " in ident.lower():
            arch = "arm64"
    return os_name, arch


def select_backend(gpu_vendor: str | None, os_name: str | None = None) -> str:
    """Backend choice per design: CUDA if NVIDIA, Metal on macOS, Vulkan if
    a non-NVIDIA GPU is present, else CPU. ``--list-devices`` validates the
    choice post-install; the supervisor's touch generation is ground truth."""
    if os_name is None:
        os_name, _ = _host_os_arch()
    if os_name == "macos":
        return "metal"
    vendor = (gpu_vendor or "").lower()
    if "nvidia" in vendor:
        return "cuda"
    if vendor in ("amd", "intel") or "radeon" in vendor or "arc" in vendor:
        return "vulkan"
    return "cpu"


def resolve_assets(tag: str, backend: str, os_name: str | None = None,
                   arch: str | None = None) -> AssetPlan:
    """Compose the asset list for (tag, backend, platform).

    Raises BinaryResolutionError for combinations the release does not ship
    (a platform/backend pair upstream publishes no artifact for). Callers
    fall back down the backend ladder: cuda -> vulkan -> cpu.
    """
    host_os, host_arch = _host_os_arch()
    os_name = os_name or host_os
    arch = arch or host_arch
    plan = AssetPlan(tag=tag, backend=backend)

    if os_name == "macos":
        # macOS tarballs are unified (Metal built in).
        plan.assets = [f"llama-{tag}-bin-macos-{arch}.tar.gz"]
        return plan

    if os_name == "ubuntu":
        if backend == "cuda":
            # No prebuilt Linux CUDA zips at current tags — Linux CUDA users
            # build from source or use vulkan; resolver is honest about it.
            raise BinaryResolutionError(
                f"no prebuilt linux CUDA asset at {tag}; use vulkan/cpu or a source build")
        suffix = {"vulkan": f"vulkan-{arch}", "hip": f"rocm-7.2-{arch}",
                  "cpu": arch}.get(backend)
        if suffix is None:
            raise BinaryResolutionError(f"unsupported linux backend {backend}")
        plan.assets = [f"llama-{tag}-bin-ubuntu-{suffix}.tar.gz"]
        return plan

    if os_name == "win":
        if backend == "cuda":
            cuda_ver = _WIN_CUDA_VERSION_ARM64 if arch == "arm64" else _WIN_CUDA_VERSION
            plan.assets = [
                f"llama-{tag}-bin-win-cuda-{cuda_ver}-{arch}.zip",
                f"cudart-llama-bin-win-cuda-{cuda_ver}-{arch}.zip",
            ]
        elif backend == "vulkan":
            if arch == "arm64":
                raise BinaryResolutionError(f"no win-vulkan-arm64 asset at {tag}")
            plan.assets = [f"llama-{tag}-bin-win-vulkan-x64.zip"]
        elif backend == "hip":
            plan.assets = [f"llama-{tag}-bin-win-hip-radeon-x64.zip"]
        elif backend == "cpu":
            plan.assets = [f"llama-{tag}-bin-win-cpu-{arch}.zip"]
        else:
            raise BinaryResolutionError(f"unsupported windows backend {backend}")
        return plan

    raise BinaryResolutionError(f"unsupported platform {os_name}-{arch}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path,
              progress: "Callable[[int, int], None] | None" = None) -> None:
    """Stream url -> dest. ``progress(done_bytes, total_bytes)`` ticks per
    chunk (total 0 when the server sends no Content-Length) — a several-
    hundred-MB archive on a slow line must never look hung."""
    logger.info("downloading %s", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if progress is not None:
                progress(done, total)
    tmp.replace(dest)


def _extract(archive: Path, dest: Path,
             progress: "Callable[[int, int], None] | None" = None) -> None:
    """Extract member by member so ``progress(done, total)`` can tick in
    uncompressed bytes — big archives take real time on laptop disks."""
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            members = z.infolist()
            total = sum(m.file_size for m in members)
            done = 0
            for m in members:
                z.extract(m, dest)
                done += m.file_size
                if progress is not None:
                    progress(done, total)
    else:
        import tarfile
        with tarfile.open(archive) as t:
            members = t.getmembers()
            total = sum(m.size for m in members)
            done = 0
            for m in members:
                t.extract(m, dest, filter="data")
                done += m.size
                if progress is not None:
                    progress(done, total)


def server_binary(install_dir: Path) -> Path:
    """Locate llama-server within an extracted runtime (zips differ in
    whether they nest a build/bin directory)."""
    names = ("llama-server.exe", "llama-server")
    for name in names:
        direct = install_dir / name
        if direct.exists():
            return direct
    for name in names:
        hits = sorted(install_dir.rglob(name))
        if hits:
            return hits[0]
    raise BinaryResolutionError(f"llama-server not found under {install_dir}")


def verify_install(install_dir: Path, tag: str) -> str:
    """Run --version; require the tag's build number in the output.
    (The binary prints the tag WITHOUT the 'b' prefix.)"""
    exe = server_binary(install_dir)
    out = subprocess.run([str(exe), "--version"], capture_output=True,
                         text=True, encoding="utf-8", errors="replace",
                         timeout=60, cwd=str(exe.parent))
    text = (out.stdout + out.stderr).strip()
    if tag.lstrip("b") not in text:
        raise BinaryResolutionError(
            f"version check failed for {exe}: expected {tag}, got: {text[:120]}")
    return text.splitlines()[0] if text else ""


def prune_old_tags(keep: list[str]) -> None:
    """Retain only the tags in ``keep`` (current + previous — N-1 rollback).
    The shared ``downloads/`` archive cache is not a tag and always survives."""
    root = runtimes_root()
    if not root.exists():
        return
    for entry in root.iterdir():
        if entry.is_dir() and entry.name != "downloads" and entry.name not in keep:
            shutil.rmtree(entry, ignore_errors=True)
            logger.info("pruned old runtime %s", entry.name)


def ensure_runtime_installed(tag: str, backend: str,
                             expected_sha256: dict[str, str] | None = None,
                             progress: "Callable[[str, int, int, str], None] | None" = None) -> Path:
    """Idempotent: resolve, download, verify, extract, version-check.

    ``expected_sha256`` maps asset name -> hash when the catalog pins them;
    without pins the computed hash is recorded in the manifest (trust on
    first download, verified on every reinstall).
    ``progress(stage, done_bytes, total_bytes, label)`` ticks through the
    slow parts — stage is "download" | "extract" | "verify", label is the
    asset counter ("1/2") when the plan has several archives.
    Returns the install directory containing llama-server.
    """
    plan = resolve_assets(tag, backend)
    install_dir = plan.install_dir
    manifest_path = install_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("verified_version"):
                return install_dir
        except (json.JSONDecodeError, OSError):
            pass  # damaged manifest -> reinstall

    install_dir.mkdir(parents=True, exist_ok=True)
    downloads = runtimes_root() / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)

    recorded: dict[str, str] = {}
    n_assets = len(plan.assets)
    for i, asset in enumerate(plan.assets, 1):
        label = f"{i}/{n_assets}" if n_assets > 1 else ""
        archive = downloads / asset
        if not archive.exists():
            _download(RELEASE_URL.format(tag=tag, asset=asset), archive,
                      progress=(lambda d, t, _l=label: progress("download", d, t, _l))
                      if progress is not None else None)
        if progress is not None:
            progress("verify", 0, 0, label)
        digest = _sha256(archive)
        expected = (expected_sha256 or {}).get(asset)
        if expected and digest != expected:
            archive.unlink(missing_ok=True)
            raise BinaryResolutionError(
                f"sha256 mismatch for {asset}: expected {expected}, got {digest}")
        recorded[asset] = digest
        _extract(archive, install_dir,
                 progress=(lambda d, t, _l=label: progress("extract", d, t, _l))
                 if progress is not None else None)

    if progress is not None:
        progress("verify", 0, 0, "")
    version = verify_install(install_dir, tag)
    manifest_path.write_text(json.dumps({
        "tag": tag, "backend": plan.backend, "assets": recorded,
        "verified_version": version,
    }, indent=2), encoding="utf-8")
    logger.info("installed llama.cpp %s (%s): %s", tag, backend, version)
    return install_dir
