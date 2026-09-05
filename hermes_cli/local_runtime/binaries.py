"""Binary acquisition for the managed llama.cpp runtime."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


logger = logging.getLogger(__name__)

RELEASE_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{asset}"

# Windows CUDA zips ship per CUDA major; the runtime zip must be paired with its cudart zip so
# end users need no toolkit. 13.3 verified on 13.1 and 13.2 drivers.
_WIN_CUDA_VERSION = "13.3"
# arm64 Windows CUDA prebuilts landed upstream (~b1036x) on CUDA 13.4. Tags at or before b10290
# don't have them; resolution succeeds and the download 404s honestly if a user pins backward.
_WIN_CUDA_VERSION_ARM64 = "13.4"


def default_tag() -> str:
    """Fallback when the config section is missing entirely; DEFAULT_CONFIG owns the shipped tag."""
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
    """Machine-scoped, deliberately NOT profile-scoped: engine binaries, presets and server state
    describe this machine's hardware and its one managed server (stable port) — a second profile
    re-downloading the engine or fighting over the port would be the bug. Profile-scoped things
    (default model, enabled) live in each profile's config.yaml."""
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root() / "runtimes" / "llamacpp"


def manifest_verified(manifest: Path) -> bool:
    """True when an install manifest records a verified_version (missing/damaged -> False)."""
    try:
        return bool(json.loads(manifest.read_text(encoding="utf-8")).get("verified_version"))
    except (json.JSONDecodeError, OSError):
        return False


def _release_number(tag: str) -> int:
    digits = "".join(ch for ch in tag if ch.isdigit())
    return int(digits) if digits else 0


def installed_tags() -> list[str]:
    """Tags with a verified install, newest first by release number. The boot ladder and the
    update check both read installed-ness from here — one resolver, every caller."""
    root = runtimes_root()
    if not root.exists():
        return []
    found = {entry.name for entry in root.iterdir()
             if entry.is_dir() and entry.name != "downloads"
             and any(manifest_verified(m) for m in entry.glob("*/manifest.json"))}
    return sorted(found, key=_release_number, reverse=True)


def _host_os_arch() -> tuple[str, str]:
    """(os, arch) normalized to release-asset vocabulary. PITFALL: PROCESSOR_ARCHITECTURE lies
    under x64 emulation on ARM64 Windows, and platform.machine() reads the same env on some
    Pythons — so on Windows prefer PROCESSOR_IDENTIFIER's text when present."""
    system = platform.system().lower()
    os_name = {"windows": "win", "darwin": "macos", "linux": "ubuntu"}.get(system, system)
    arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"
    if os_name == "win":
        ident = os.environ.get("PROCESSOR_IDENTIFIER", "").lower()
        if "armv8" in ident or "arm " in ident:
            arch = "arm64"
    return os_name, arch


def select_backend(gpu_vendor: str | None, os_name: str | None = None) -> str:
    """CUDA if NVIDIA, Metal on macOS, Vulkan if a non-NVIDIA GPU is present, else CPU.
    ``--list-devices`` validates post-install; the supervisor's touch generation is ground truth."""
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


# Per-OS (human label, {backend: asset-name templates}). Windows CUDA pairs the runtime zip with
# its cudart zip; ubuntu ships tarballs, win ships zips.
_ASSET_TEMPLATES = {
    "ubuntu": ("linux", {
        "vulkan": ["llama-{tag}-bin-ubuntu-vulkan-{arch}.tar.gz"],
        "hip": ["llama-{tag}-bin-ubuntu-rocm-7.2-{arch}.tar.gz"],
        "cpu": ["llama-{tag}-bin-ubuntu-{arch}.tar.gz"],
    }),
    "win": ("windows", {
        "cuda": ["llama-{tag}-bin-win-cuda-{cuda_ver}-{arch}.zip",
                 "cudart-llama-bin-win-cuda-{cuda_ver}-{arch}.zip"],
        "vulkan": ["llama-{tag}-bin-win-vulkan-x64.zip"],
        "hip": ["llama-{tag}-bin-win-hip-radeon-x64.zip"],
        "cpu": ["llama-{tag}-bin-win-cpu-{arch}.zip"],
    }),
}


def resolve_assets(tag: str, backend: str, os_name: str | None = None,
                   arch: str | None = None) -> AssetPlan:
    """Compose the asset list for (tag, backend, platform). Raises BinaryResolutionError for pairs
    the release ships no artifact for; callers fall back down the ladder cuda -> vulkan -> cpu."""
    host_os, host_arch = _host_os_arch()
    os_name = os_name or host_os
    arch = arch or host_arch
    if os_name == "macos":
        # macOS tarballs are unified (Metal built in).
        return AssetPlan(tag, backend, [f"llama-{tag}-bin-macos-{arch}.tar.gz"])
    if os_name not in _ASSET_TEMPLATES:
        raise BinaryResolutionError(f"unsupported platform {os_name}-{arch}")
    if os_name == "ubuntu" and backend == "cuda":
        # No prebuilt Linux CUDA zips at current tags — Linux CUDA users build from source or
        # use vulkan; the resolver is honest about it.
        raise BinaryResolutionError(
            f"no prebuilt linux CUDA asset at {tag}; use vulkan/cpu or a source build")
    if os_name == "win" and backend == "vulkan" and arch == "arm64":
        raise BinaryResolutionError(f"no win-vulkan-arm64 asset at {tag}")
    label, templates = _ASSET_TEMPLATES[os_name]
    if backend not in templates:
        raise BinaryResolutionError(f"unsupported {label} backend {backend}")
    cuda_ver = _WIN_CUDA_VERSION_ARM64 if arch == "arm64" else _WIN_CUDA_VERSION
    return AssetPlan(tag, backend, [t.format(tag=tag, arch=arch, cuda_ver=cuda_ver)
                                    for t in templates[backend]])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path,
              progress: "Callable[[int, int], None] | None" = None) -> None:
    """Stream url -> dest. ``progress(done_bytes, total_bytes)`` ticks per chunk (total 0 when
    the server sends no Content-Length) — a several-hundred-MB archive must never look hung."""
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
    """Extract member by member so ``progress(done, total)`` can tick in uncompressed bytes."""
    if archive.name.endswith(".zip"):
        opener, list_members, size = zipfile.ZipFile, "infolist", "file_size"
        kwargs = {}
    else:
        import tarfile
        opener, list_members, size = tarfile.open, "getmembers", "size"
        kwargs = {"filter": "data"}
    with opener(archive) as ar:
        members = getattr(ar, list_members)()
        total = sum(getattr(m, size) for m in members)
        done = 0
        for m in members:
            ar.extract(m, dest, **kwargs)
            done += getattr(m, size)
            if progress is not None:
                progress(done, total)


def server_binary(install_dir: Path) -> Path:
    """Locate llama-server within an extracted runtime (zips differ in whether they nest a
    build/bin directory)."""
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
    """Run --version; require the tag's build number in the output (printed WITHOUT the 'b')."""
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
    """Retain only the tags in ``keep`` (current + previous — N-1 rollback). The shared
    ``downloads/`` archive cache is not a tag and always survives."""
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
    """Idempotent: resolve, download, verify, extract, version-check; returns the install dir.
    ``expected_sha256`` pins hashes per asset; without pins the computed hash is recorded in the
    manifest (trust on first download, verified on every reinstall). ``progress(stage, done,
    total, label)`` ticks through download/extract/verify."""
    plan = resolve_assets(tag, backend)
    install_dir = plan.install_dir
    manifest_path = install_dir / "manifest.json"
    if manifest_verified(manifest_path):
        return install_dir

    install_dir.mkdir(parents=True, exist_ok=True)
    downloads = runtimes_root() / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)

    def stage_progress(stage: str, label: str):
        if progress is None:
            return None
        tick = progress
        return lambda d, t: tick(stage, d, t, label)

    recorded: dict[str, str] = {}
    n_assets = len(plan.assets)
    for i, asset in enumerate(plan.assets, 1):
        label = f"{i}/{n_assets}" if n_assets > 1 else ""
        archive = downloads / asset
        if not archive.exists():
            _download(RELEASE_URL.format(tag=tag, asset=asset), archive,
                      progress=stage_progress("download", label))
        if progress is not None:
            progress("verify", 0, 0, label)
        digest = _sha256(archive)
        expected = (expected_sha256 or {}).get(asset)
        if expected and digest != expected:
            archive.unlink(missing_ok=True)
            raise BinaryResolutionError(
                f"sha256 mismatch for {asset}: expected {expected}, got {digest}")
        recorded[asset] = digest
        _extract(archive, install_dir, progress=stage_progress("extract", label))

    if progress is not None:
        progress("verify", 0, 0, "")
    version = verify_install(install_dir, tag)
    manifest_path.write_text(json.dumps({"tag": tag, "backend": plan.backend, "assets": recorded,
                                         "verified_version": version}, indent=2), encoding="utf-8")
    logger.info("installed llama.cpp %s (%s): %s", tag, backend, version)
    return install_dir


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.


_PLUGIN_COMPAT_LAZY = {
    'get_hermes_home': ('hermes_constants', 'get_hermes_home'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
