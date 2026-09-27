"""Hardware selection for PM-owned llama.cpp binaries; mutable runtime state stays separate."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import pm
from pm.downloader import ProgressFn

logger = logging.getLogger(__name__)

BACKEND_PACKAGES = {
    "cuda": "llamacpp-cuda",
    "vulkan": "llamacpp-vulkan",
    "metal": "llamacpp-metal",
    "hip": "llamacpp-hip",
    "cpu": "llamacpp-cpu",
}


class BinaryResolutionError(RuntimeError):
    """The requested backend has no pinned or installed engine."""


@dataclass(frozen=True)
class Engine:
    backend: str
    tag: str
    binary: Path


def runtimes_root() -> Path:
    """Machine-scoped presets and server state. Binaries belong to PM's store."""
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root() / "runtimes" / "llamacpp"


def pinned_tag(backend: str) -> str:
    version = pm.Lockfile(pm.paths.lockfile_path()).version(BACKEND_PACKAGES[backend])
    if version is None:
        raise BinaryResolutionError(f"llama.cpp {backend} has no PM version pin")
    return f"b{version}"


def select_backend(gpu_vendor: str | None, os_name: str | None = None) -> str:
    if os_name is None:
        os_name = "macos" if pm.current_target().startswith("darwin-") else "other"
    if os_name == "macos":
        return "metal"
    vendor = (gpu_vendor or "").lower()
    if "nvidia" in vendor:
        return "cuda"
    if any(name in vendor for name in ("amd", "intel", "radeon", "arc")):
        return "vulkan"
    return "cpu"


def _candidates(requested: str, gpu_vendor: str | None, target: str) -> tuple[str, ...]:
    if requested != "auto":
        if requested not in BACKEND_PACKAGES:
            raise BinaryResolutionError(f"unknown backend {requested}")
        return (requested,)
    preferred = select_backend(gpu_vendor, "macos" if target.startswith("darwin-") else "other")
    fallback = ("vulkan", "cpu") if gpu_vendor else ("cpu",)
    return tuple(dict.fromkeys((preferred, *fallback)))


def unavailable_reason(backend: str, target: str | None = None) -> str | None:
    name = BACKEND_PACKAGES.get(backend)
    if name is None:
        return f"unknown backend {backend}"
    target = target or pm.current_target()
    reason = pm.get_package(name).missing_reason(target)
    if reason:
        return reason
    lock = pm.Lockfile(pm.paths.lockfile_path())
    if not lock.version(name) or not lock.artifacts(name, target):
        return f"{name} is not pinned for {target}"
    return None


def resolve_backend(requested: str = "auto", *, gpu_vendor: str | None = None,
                    target: str | None = None) -> str:
    """Explicit choices are strict. Auto only falls back to compatible pinned builds."""
    if requested == "auto" and target is None and gpu_vendor is None:
        from hermes_cli.local_runtime.bootstrap import _detect_gpu_vendor

        gpu_vendor = _detect_gpu_vendor()
    target = target or pm.current_target()
    reasons = []
    for backend in _candidates(requested, gpu_vendor, target):
        reason = unavailable_reason(backend, target)
        if reason is None:
            return backend
        reasons.append(reason)
    raise BinaryResolutionError("; ".join(reasons))


def _installed(candidates: tuple[str, ...], allow_outdated: bool) -> Engine | None:
    for candidate in candidates:
        found = pm.installed_package(BACKEND_PACKAGES[candidate], allow_outdated=allow_outdated)
        if found is not None and found.binary is not None:
            return Engine(candidate, f"b{found.version}", found.binary)
    return None


def installed_engine(backend: str = "auto", *, allow_outdated: bool = True) -> Engine | None:
    """Boot may retain a prior PM pin and moves a pre-PM engine into PM's store once; it never downloads."""
    vendor = None
    if backend == "auto":
        from hermes_cli.local_runtime.bootstrap import _detect_gpu_vendor

        vendor = _detect_gpu_vendor()
    candidates = _candidates(backend, vendor, pm.current_target())
    engine = _installed(candidates, allow_outdated)
    if engine is None and any(adopt_legacy_engine(candidate) for candidate in candidates):
        engine = _installed(candidates, allow_outdated)
    return engine


# Before PM owned binaries, Hermes installed each engine to runtimes/llamacpp/b<tag>/<backend>/
# and wrote a manifest.json with the archive digests and the llama-server --version it saw.
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ADOPTION_LOCK = threading.Lock()
# Installs that failed to move stay put for this process; the pane polls status every few seconds.
_LEFT_IN_PLACE: set[Path] = set()


def _legacy_installs(backend: str) -> list[tuple[str, Path, dict]]:
    """Verified pre-PM installs of ``backend`` as (version, dir, manifest), newest first."""
    found = []
    for manifest_path in runtimes_root().glob(f"b*/{backend}/manifest.json"):
        tag = manifest_path.parent.parent.name
        try:
            number = int(tag.removeprefix("b"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if (isinstance(manifest, dict) and manifest.get("tag") == tag and manifest.get("backend") == backend
                and manifest.get("verified_version") and isinstance(manifest.get("assets"), dict)
                and manifest_path.parent not in _LEFT_IN_PLACE):
            found.append((number, str(number), manifest_path.parent, manifest))
    return [(version, path, manifest) for _, version, path, manifest in sorted(found, reverse=True)]


def _legacy_artifacts(package, version: str, target: str, assets: dict) -> list[str] | None:
    """The manifest's archive digests in PM's archive order, or None when one is missing."""
    shas = [assets.get(url.rsplit("/", 1)[-1]) for url in package.fetch_urls(version, target)]
    if not shas or not all(isinstance(sha, str) and _SHA256.fullmatch(sha) for sha in shas):
        return None
    return shas


@contextmanager
def _store_lock(root: Path) -> Iterator[bool]:
    """PM's store lock, or False when another PM operation holds it (a download can take minutes)."""
    from pm.filesystem import lock_fd

    root.mkdir(parents=True, exist_ok=True)
    fd = os.open(root / ".install.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        yield lock_fd(fd, wait=True, timeout=2)
    finally:
        os.close(fd)


def adopt_legacy_engine(backend: str) -> bool:
    """Move the newest pre-PM install of ``backend`` into PM's store and record it as installed.

    A machine that already ran a Hermes-installed engine keeps it across the update to PM. The
    manifest's archive digests become the PM identity, so a tag that matches the pin counts as
    current and an older tag counts as outdated, which offers the update. ``os.rename`` only:
    instant on one volume, and a store on another volume leaves the engine where it is. Never
    raises; returns whether an engine was moved.
    """
    try:
        candidates = _legacy_installs(backend)
        if not candidates:
            return False
        from pm import paths as pm_paths

        name = BACKEND_PACKAGES[backend]
        package = pm.get_package(name)
        target = pm.current_target()
        root = pm_paths.writable_store_root()
        facts_path = pm_paths.facts_path() if root == pm_paths.store_root() else root / "facts.json"
        with _ADOPTION_LOCK, _store_lock(root) as held:
            if not held or pm.installed_package(name, allow_outdated=True) is not None:
                return False
            for version, source, manifest in candidates:
                if _adopt(package, version, target, source, manifest, root, facts_path):
                    return True
                _LEFT_IN_PLACE.add(source)
    except Exception as exc:  # noqa: BLE001 - a failed move must not break the callers that ask
        logger.warning("could not move the pre-PM llama.cpp %s engine into the PM store: %s", backend, exc)
    return False


def _adopt(package, version: str, target: str, source: Path, manifest: dict, root: Path,
           facts_path: Path) -> bool:
    from pm.store import tree_digest

    artifacts = _legacy_artifacts(package, version, target, manifest["assets"])
    entry_name = package.store_entry(version, target)
    entry = root / entry_name
    if artifacts is None or entry.exists() or entry.is_symlink():
        return False
    reason = package.verify(source, target)
    if reason:
        logger.warning("pre-PM llama.cpp at %s left in place: %s", source, reason)
        return False
    try:
        os.rename(source, entry)
    except OSError as exc:
        logger.warning("pre-PM llama.cpp at %s left in place: %s", source, exc)
        return False
    try:
        pm.Facts(facts_path).record(package.name, version, entry_name, package.env(entry, target), root,
                                    target=target, artifacts=artifacts, digest=tree_digest(entry))
    except BaseException:
        with suppress(OSError):
            os.rename(entry, source)
        raise
    with suppress(OSError):
        source.parent.rmdir()
    logger.info("moved llama.cpp b%s (%s) from %s into the PM store", version, package.name, source)
    return True


def ensure_engine(backend: str, *, progress: Callable[[str, int, int, str], None] | None = None,
                  pause_event: threading.Event | None = None,
                  download_progress: ProgressFn | None = None) -> Engine:
    """Only deliberate install/update jobs call this. PM verifies every archive and publishes."""
    resolved = resolve_backend(backend)
    pm.ensure(BACKEND_PACKAGES[resolved], explicit=True, progress=progress,
              pause_event=pause_event, download_progress=download_progress)
    engine = installed_engine(resolved, allow_outdated=False)
    if engine is None:
        raise BinaryResolutionError(f"llama.cpp {resolved} install has no usable pinned binary")
    return engine


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
