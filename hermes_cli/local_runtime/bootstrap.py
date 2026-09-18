"""Bootstrap for the managed runtime: config -> installed binaries -> running supervised server.

One public call, ``ensure_local_runtime(config)``, safe at any session start: disabled or
already-running -> no-op; enabled -> serve the installed build under a supervisor. Kept
import-light: callers gate on config before importing so disabled sessions never pay the import.
"""

from __future__ import annotations

from contextlib import suppress
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from hermes_cli.local_runtime.binaries import runtimes_root
from hermes_cli.local_runtime.gguf import SPLIT_PART_RE, model_id_from_stem

logger = logging.getLogger(__name__)

_SUPERVISOR = None  # process-wide singleton; one router per Hermes process


def _detect_gpu_vendor() -> str | None:
    """Best-effort GPU vendor for backend selection. NVIDIA via nvidia-smi resolved by the hardware
    probe's PATH-independent ladder (a stripped service PATH must not demote an NVIDIA box to
    vulkan/cpu); AMD/Intel via the OS device lists so select_backend's vulkan ladder is reachable
    on non-NVIDIA machines; anything else defers to select_backend's fallback ladder. Software-only
    renderers never count as a GPU. Never raises — a probe miss means CPU, not a broken boot."""
    from hermes_cli.local_runtime.hardware import _nvidia_smi_path

    smi = _nvidia_smi_path()
    if smi is not None:
        with suppress(OSError, subprocess.TimeoutExpired):
            out = subprocess.run(
                [smi, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10)
            if out.returncode == 0 and out.stdout.strip():
                return "nvidia " + out.stdout.strip().splitlines()[0]
    # macOS has no lister below (select_backend is Metal regardless), so this is None there.
    return _vendor_from_names(_os_gpu_names())


def _os_gpu_names() -> list[str]:
    """OS device-list names for the vendor map above; one seam so tests pin the mapping without
    caring which host they run on."""
    if sys.platform == "darwin":
        return []
    if os.name == "nt":
        return _windows_gpu_names()
    return _linux_gpu_names()


# PCI vendor IDs as seen under /sys/class/drm/card*/device/vendor on Linux.
_DRM_VENDOR_NAMES = {
    "0x10de": "nvidia",
    "0x1002": "amd",
    "0x8086": "intel",
}

# (needle, vendor): first hit wins. NVIDIA needles come first so a hybrid laptop (NVIDIA dGPU +
# AMD/Intel iGPU) resolves to CUDA — the discrete card is the faster inference target.
_GPU_NAME_VENDORS = (
    ("nvidia", "nvidia"),
    ("geforce", "nvidia"),
    ("quadro", "nvidia"),
    ("tesla", "nvidia"),
    ("amd", "amd"),
    ("radeon", "amd"),
    ("arc", "intel"),
    ("intel", "intel"),
    ("iris", "intel"),
    ("uhd graphics", "intel"),
    ("xe graphics", "intel"),
)

# Names that describe a software/virtual display adapter, never a real GPU.
_SOFTWARE_RENDERERS = (
    "basic render",
    "llvmpipe",
    "softpipe",
    "swrast",
    "lavapipe",
    "remote desktop",
    "rdpdd",
    "virtualbox",
    "vmware",
    "hyper-v",
    "qxl",
    "virtio",
)


def _vendor_from_names(names: "list[str] | tuple[str, ...] | None") -> str | None:
    """Map OS-reported GPU names to select_backend's vendor vocabulary ("nvidia" | "amd" |
    "intel"), or None. Pure function of its input — the seam host-independent tests pin."""
    for raw in names or ():
        name = (raw or "").strip().lower()
        if not name or any(s in name for s in _SOFTWARE_RENDERERS):
            continue
        for needle, vendor in _GPU_NAME_VENDORS:
            if needle in name:
                return vendor
    return None


def _windows_gpu_names() -> list[str]:
    """Display-adapter names via Win32_VideoController. Absolute engine paths first: gateway and
    service sessions run under a stripped PATH that may not resolve powershell/wmic."""
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    powershells = [
        str(Path(system_root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"),
        # 32-bit Python on 64-bit Windows: System32 redirects, Sysnative reaches through.
        str(Path(system_root) / "Sysnative" / "WindowsPowerShell" / "v1.0" / "powershell.exe"),
        "powershell",
    ]
    ps_command = ("Get-CimInstance Win32_VideoController"
                  " | Select-Object -ExpandProperty Name")
    for ps in powershells:
        with suppress(OSError, subprocess.TimeoutExpired):
            out = subprocess.run(
                [ps, "-NoProfile", "-Command", ps_command],
                capture_output=True, text=True, timeout=15)
            if out.returncode == 0 and out.stdout.strip():
                return [line for line in
                        (l.strip() for l in out.stdout.splitlines()) if line]
    wmic = Path(system_root) / "System32" / "wbem" / "wmic.exe"
    if wmic.exists():
        with suppress(OSError, subprocess.TimeoutExpired):
            out = subprocess.run(
                [str(wmic), "path", "win32_VideoController", "get", "name", "/format:list"],
                capture_output=True, text=True, timeout=15)
            if out.returncode == 0 and out.stdout.strip():
                return [line.split("=", 1)[1].strip() for line in out.stdout.splitlines()
                        if line.strip().lower().startswith("name=")]
    return []


def _linux_gpu_names() -> list[str]:
    """GPU vendors from the DRM sysfs tree (no subprocess, no dependencies); lspci as fallback."""
    names: list[str] = []
    with suppress(OSError):
        vendor_files = sorted(Path("/sys/class/drm").glob("card*/device/vendor"))
        for vendor_file in vendor_files:
            with suppress(OSError, ValueError):
                vendor = _DRM_VENDOR_NAMES.get(vendor_file.read_text().strip().lower())
                if vendor is not None and vendor not in names:
                    names.append(vendor)
    if names:
        return names
    with suppress(OSError, subprocess.TimeoutExpired):
        lspci = subprocess.run(
            ["lspci", "-mm"], capture_output=True, text=True, timeout=10)
        if lspci.returncode != 0:
            return []
        out: list[str] = []
        for line in lspci.stdout.splitlines():
            lowered = line.lower()
            if ("vga" in lowered or "\"3d controller\"" in lowered
                    or "\"display controller\"" in lowered):
                quoted = [part for part in line.split("\"") if part.strip()]
                if quoted:
                    out.append(quoted[-1].strip())
        return out
    return []


def models_dir() -> Path:
    """Machine-scoped, deliberately NOT profile-scoped: a 20 GB GGUF is a machine asset, and every
    profile shares the one managed server that serves it (same rule as runtimes_root())."""
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root() / "models"


def assets_dir() -> Path:
    """Non-model companion files (mmproj projectors, spec-decode drafts). A subdirectory so the
    router's model listing — and staged_models() — never mistakes an asset for a servable model."""
    return models_dir() / "assets"


def staged_in(models_dir: Path, *, require_complete: bool = True) -> "list[Path]":
    """Servable GGUFs in a directory: single files, plus split GGUFs once by their first part.
    With ``require_complete`` a split counts only when EVERY part is on disk — a mid-download split
    is not servable and must not surface anywhere as a model."""
    files = sorted(models_dir.glob("*.gguf"))
    names = {p.name for p in files}
    out = []
    for p in files:
        m = SPLIT_PART_RE.search(p.name)
        if m is None:
            out.append(p)
            continue
        if m.group(1) != "00001":
            continue
        stem, total = p.name[: m.start()], int(m.group(2))
        if not require_complete or all(f"{stem}-{i:05d}-of-{m.group(2)}.gguf" in names
                                       for i in range(2, total + 1)):
            out.append(p)
    return out


def staged_models() -> "list[Path]":
    """Servable staged models (continuation parts, incomplete splits and assets/ never count)."""
    return staged_in(models_dir())


def staged_model_ids() -> "list[str]":
    return [model_id_from_stem(p.stem) for p in staged_models()]


def _presets_stale() -> bool:
    """True when a staged model has no section in the preset INI — it would autoload with stock
    fit instead of a policy decision."""
    with suppress(Exception):
        from hermes_cli.local_runtime.presets import read_preset_decisions

        known = read_preset_decisions()
        return any(mid not in known or (not known[mid].refusal and not (known[mid].keys or {}).get("model"))
                   for mid in staged_model_ids())
    return False


def _stop_state_server(state: dict) -> None:
    """Best-effort stop of the server the state file points at (an incumbent this process doesn't
    supervise). The state pid is ours by contract — the file only ever describes the managed
    server."""
    from hermes_cli.local_runtime.endpoint import _pid_alive

    try:
        pid = int(state.get("pid"))
        if pid <= 0:
            return
        os.kill(pid, signal.SIGTERM)
    except (TypeError, ValueError, OSError):
        return
    # Give it a moment to release the port and the GPU. Liveness via psutil — on Windows
    # os.kill(pid, 0) TERMINATES the process, it is not a probe.
    for _ in range(50):
        if not _pid_alive(pid):
            return
        time.sleep(0.1)


def refresh_local_runtime() -> bool:
    """Restart the managed server so it rescans the models directory. The router's model list is
    SPAWN-ONLY: a GGUF added after start is invisible to GET /models and 400s on completion, so
    anything that changes the staged set while the server runs must bounce it."""
    global _SUPERVISOR
    try:
        from hermes_cli.config import load_config

        if _SUPERVISOR is None:
            from hermes_cli.local_runtime.endpoint import _state_endpoint

            state = _state_endpoint()
            if state is None:
                return False
            logger.info("bouncing adopted llama-server (pid=%s) to rescan models", state.get("pid"))
            _stop_state_server(state)
        else:
            shutdown_local_runtime()
        return ensure_local_runtime(load_config(), force=True) is not None
    except Exception as exc:  # noqa: BLE001
        logger.warning("local runtime refresh failed: %s", exc)
        return False


def _generate_presets(mdir: Path, preset_path: Path) -> Path | None:
    """Write the launch-policy INI for every staged model; returns the path to hand the router.

    Priced against CAPACITY, not live free VRAM: this runs while the outgoing server instance may
    still hold the card (restart, refresh after a download), and its memory is freed before the new
    instance loads anything. Pricing against live-free once pinned a fitting model's weights to CPU.

    Degradation ladder on failure: a STALE policy still beats no policy — stock fit (f16 KV at max
    context, no placement) is the silent-busy-wait failure on Windows. Keep serving with the
    previous INI when one exists; only a first boot with no INI at all falls to stock fit.
    """
    from hermes_cli.local_runtime.hardware import probe_budget
    from hermes_cli.local_runtime.presets import generate_presets

    try:
        for entry in generate_presets(mdir, probe_budget(planning=True), preset_path):
            if entry.refusal:
                logger.warning("model refused by physics check: %s", entry.refusal)
        return preset_path
    except Exception as exc:  # noqa: BLE001 — policy failure must not block serving
        if preset_path.exists():
            logger.error("preset generation failed (%s); serving with the "
                         "PREVIOUS launch policies — models staged since "
                         "the last successful generation run unpoliced "
                         "until this is fixed", exc)
            return preset_path
        logger.error("preset generation failed (%s) and no previous "
                     "policy file exists; router runs stock fit", exc)
        return None


def ensure_local_runtime(config: dict, force: bool = False) -> "object | None":
    """Idempotent boot of the managed runtime. Returns the supervisor (or None when
    disabled/unavailable). Never raises into a session start — failures log and return None; chat
    falls back to configured providers."""
    global _SUPERVISOR
    section = (config or {}).get("local_runtime") or {}
    if not force and not section.get("enabled"):
        return None
    if _SUPERVISOR is not None:
        return _SUPERVISOR

    # Residency: no staged models means nothing to serve — don't boot an empty server (delete
    # your last model and boots stop). force boots as ever.
    if not force and not staged_models():
        logger.info("local runtime enabled but no models staged; not booting")
        return None

    # Another Hermes process may already be supervising — reuse via state, but ONLY while its
    # launch policy still covers every staged model. A server whose preset file predates a
    # download serves the new model with no policy at all (--models-autoload + stock fit). A stale
    # incumbent gets stopped and replaced by a fresh boot with regenerated presets; sessions ride
    # through like any other supervised restart (stable port + persisted key).
    from hermes_cli.local_runtime.endpoint import _state_endpoint

    state = _state_endpoint()
    if state is not None:
        if not _presets_stale():
            logger.info("managed llama-server already running (another process)")
            return None
        logger.info("running server's presets predate the staged models; "
                    "replacing it so every model launches with a policy")
        _stop_state_server(state)

    try:
        from hermes_cli.local_runtime.binaries import (
            default_tag, ensure_runtime_installed, installed_tags, select_backend)
        from hermes_cli.local_runtime.supervisor import LlamaServerSupervisor

        backend = section.get("backend", "auto")
        if backend == "auto":
            backend = select_backend(_detect_gpu_vendor())
        # Boot ladder: serve what is INSTALLED, never download here. The configured tag is
        # preferred; when it isn't installed yet, the newest installed tag serves and the status
        # endpoint reports the pending update — the download is a deliberate click in the pane,
        # not a boot-path surprise (a multi-minute inline download here is exactly how the
        # onboarding bounce returns).
        tag = section.get("tag") or default_tag()
        have = installed_tags()
        if tag not in have:
            if not have:
                logger.info("local runtime enabled but no build installed; "
                            "install happens in the Local Models pane")
                return None
            logger.info("configured tag %s not installed; serving %s "
                        "(update is a click in Local Models)", tag, have[0])
            tag = have[0]
        install_dir = ensure_runtime_installed(tag, backend)

        mdir = models_dir()
        mdir.mkdir(parents=True, exist_ok=True)
        preset_path = _generate_presets(mdir, runtimes_root() / "presets.ini")

        sup = LlamaServerSupervisor(install_dir, mdir, preset_path=preset_path,
                                    models_max=int(section.get("models_max", 4)),
                                    port=int(section.get("port", 0)) or None)
        try:
            sup.start()
        except Exception:
            # start() can fail after the router process exists (health timeout): leaving it
            # running unsupervised strands its VRAM behind a port nothing will clean up.
            with suppress(Exception):
                sup.stop()
            raise
        _SUPERVISOR = sup
        logger.info("managed llama-server up at %s (backend=%s tag=%s)", sup.base_url, backend, tag)
        _start_idle_sweeper(sup)
        return sup
    except Exception as exc:  # noqa: BLE001 — never break session start
        logger.warning("managed local runtime unavailable: %s", exc)
        return None


def shutdown_local_runtime() -> None:
    global _SUPERVISOR
    if _SUPERVISOR is not None:
        _SUPERVISOR.stop()
        _SUPERVISOR = None


def get_supervisor():
    """The process-local supervisor, or None (a server may still run under another process —
    check the state file)."""
    return _SUPERVISOR


def _start_idle_sweeper(sup) -> None:
    """Idle-residency loop: every couple of minutes, unload models idle past the supervisor's
    threshold. Daemon thread tied to the supervisor's lifetime — exits when the server stops."""
    import threading

    def _loop():
        while sup.proc is not None and sup.proc.poll() is None:
            time.sleep(120)
            try:
                sup.sweep_idle()
            except Exception as exc:  # noqa: BLE001
                logger.debug("idle sweep skipped: %s", exc)

    threading.Thread(target=_loop, daemon=True, name="local-runtime-idle-sweep").start()


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
