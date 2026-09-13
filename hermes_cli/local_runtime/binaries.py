"""Hardware selection for PM-owned llama.cpp binaries; mutable runtime state stays separate."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pm
from pm.downloader import ProgressFn

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


def installed_engine(backend: str = "auto", *, allow_outdated: bool = True) -> Engine | None:
    """Boot may retain a prior PM pin, but never installs or adopts unmanaged bytes."""
    vendor = None
    if backend == "auto":
        from hermes_cli.local_runtime.bootstrap import _detect_gpu_vendor

        vendor = _detect_gpu_vendor()
    for candidate in _candidates(backend, vendor, pm.current_target()):
        found = pm.installed_package(BACKEND_PACKAGES[candidate], allow_outdated=allow_outdated)
        if found is not None and found.binary is not None:
            return Engine(candidate, f"b{found.version}", found.binary)
    return None


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
