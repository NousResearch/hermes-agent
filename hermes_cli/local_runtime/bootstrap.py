"""Bootstrap for the managed runtime: config -> installed binaries ->
running supervised server.

One public call, ``ensure_local_runtime(config)``, safe to call at any
session start:
- disabled or already-running (state file answers /health) -> no-op
- enabled -> install binaries if missing (idempotent), spawn supervisor

Kept import-light: callers gate on config before importing this module so
sessions with local_runtime disabled never pay the import.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path

from hermes_constants import get_hermes_home  # noqa: F401 — config paths

from hermes_cli.local_runtime.binaries import runtimes_root

logger = logging.getLogger(__name__)

_SUPERVISOR = None  # process-wide singleton; one router per Hermes process


def _detect_gpu_vendor() -> str | None:
    """Best-effort GPU vendor for backend selection. NVIDIA via nvidia-smi
    (resolved by the hardware probe's PATH-independent ladder — a stripped
    service PATH must not demote an NVIDIA box to vulkan/cpu); anything
    else defers to select_backend's fallback ladder."""
    from hermes_cli.local_runtime.hardware import _nvidia_smi_path

    smi = _nvidia_smi_path()
    if smi is None:
        return None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return "nvidia " + out.stdout.strip().splitlines()[0]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def models_dir() -> Path:
    """Machine-scoped, deliberately NOT profile-scoped: a 20 GB GGUF is a
    machine asset, and every profile shares the one managed server that
    serves it. See runtimes_root() for the same rule on the engine."""
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root() / "models"


def assets_dir() -> Path:
    """Non-model companion files (mmproj vision projectors, spec-decode
    draft models). A subdirectory so the router's model listing — and our
    staged_models() — never mistakes an asset for a servable model."""
    return models_dir() / "assets"


def staged_models() -> "list[Path]":
    """Servable staged models: single-file GGUFs count when present; a
    split GGUF counts once, by its first part, and only when EVERY part
    is on disk — a mid-download split is not servable and must not
    surface anywhere as a model. Continuation parts and assets/ never
    count."""
    import re

    part = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")
    files = sorted(models_dir().glob("*.gguf"))
    names = {p.name for p in files}
    out = []
    for p in files:
        m = part.search(p.name)
        if m is None:
            out.append(p)
            continue
        if m.group(1) != "00001":
            continue
        stem = p.name[: m.start()]
        total = int(m.group(2))
        if all(f"{stem}-{i:05d}-of-{m.group(2)}.gguf" in names
               for i in range(2, total + 1)):
            out.append(p)
    return out


def staged_model_ids() -> "list[str]":
    import re

    return [re.sub(r"-\d{5}-of-\d{5}$", "", p.stem) for p in staged_models()]


def _presets_stale() -> bool:
    """True when a staged model has no section in the preset INI — it
    would autoload with stock fit instead of a policy decision."""
    try:
        from hermes_cli.local_runtime.presets import read_preset_decisions

        known = set(read_preset_decisions())
        return any(mid not in known for mid in staged_model_ids())
    except Exception:  # noqa: BLE001
        return False


def _stop_state_server(state: dict) -> None:
    """Best-effort stop of the server the state file points at (an
    incumbent this process doesn't supervise). The state pid is ours by
    contract — the file only ever describes the managed server."""
    from hermes_cli.local_runtime.endpoint import _pid_alive

    pid = state.get("pid")
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return
    if pid <= 0:
        return
    try:
        import signal

        os.kill(pid, signal.SIGTERM)
    except (OSError, ValueError):
        return
    # Give it a moment to release the port and the GPU. Liveness via
    # psutil — on Windows os.kill(pid, 0) TERMINATES the process, it is
    # not a probe (the endpoint.py pitfall note; #local-models review).
    for _ in range(50):
        if not _pid_alive(pid):
            return
        time.sleep(0.1)


def refresh_local_runtime() -> bool:
    """Restart the managed server so it rescans the models directory.

    The router's model list is SPAWN-ONLY: a GGUF added after start is
    invisible to GET /models and 400s on completion, so anything that
    changes the staged set while the server runs must bounce it. Covers
    both ownership shapes: a supervised server restarts in-process; an
    ADOPTED server (started by a previous backend session — the normal
    shape after any restart) is stopped via its state-file pid and
    replaced with a supervised boot. Without the adopted branch, every
    download/delete in a post-restart session silently no-ops the bounce
    and the router serves a stale catalog. Returns False when there is
    nothing to refresh (no server anywhere; next boot scans fresh).
    """
    global _SUPERVISOR
    try:
        from hermes_cli.config import load_config

        if _SUPERVISOR is None:
            from hermes_cli.local_runtime.endpoint import _state_endpoint

            state = _state_endpoint()
            if state is None:
                return False
            logger.info("bouncing adopted llama-server (pid=%s) to rescan models",
                        state.get("pid"))
            _stop_state_server(state)
        else:
            shutdown_local_runtime()
        return ensure_local_runtime(load_config(), force=True) is not None
    except Exception as exc:  # noqa: BLE001
        logger.warning("local runtime refresh failed: %s", exc)
        return False


def ensure_local_runtime(config: dict, force: bool = False) -> "object | None":
    """Idempotent boot of the managed runtime. Returns the supervisor (or
    None when disabled/unavailable). Never raises into a session start —
    failures log and return None; chat falls back to configured providers.

    ``force=True`` skips the enabled gate — used by the explicit "Use this
    model" action, where the click IS the opt-in (the caller records it in
    config so future boots auto-start).
    """
    global _SUPERVISOR
    section = (config or {}).get("local_runtime") or {}
    if not force and not section.get("enabled"):
        return None
    if _SUPERVISOR is not None:
        return _SUPERVISOR

    # Residency: no staged models means nothing to serve — don't boot an
    # empty server. The walked-away story handled with zero configuration
    # (delete your last model and boots stop); Use force-boots as ever.
    if not force and not staged_models():
        logger.info("local runtime enabled but no models staged; not booting")
        return None

    # Another Hermes process may already be supervising — reuse via state,
    # but ONLY while its launch policy still covers every staged model. A
    # server whose preset file predates a download serves the new model
    # with no policy at all (--models-autoload + stock fit: f16 KV at max
    # context, no placement — the silent-demotion busy-wait on WDDM). A
    # stale incumbent gets stopped and replaced by a fresh boot with
    # regenerated presets; sessions ride through exactly like any other
    # supervised restart (stable port + persisted key).
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
            ensure_runtime_installed,
            select_backend,
        )
        from hermes_cli.local_runtime.hardware import probe_budget
        from hermes_cli.local_runtime.presets import generate_presets
        from hermes_cli.local_runtime.supervisor import LlamaServerSupervisor

        backend = section.get("backend", "auto")
        if backend == "auto":
            backend = select_backend(_detect_gpu_vendor())
        # Boot ladder: serve what is INSTALLED, never download here. The
        # configured tag (config root-of-trust; deep-merge supplies the
        # Hermes-release default when unpinned) is preferred; when it isn't
        # installed yet, the newest installed tag serves and the status
        # endpoint reports the pending update — the download is a deliberate
        # button click in the pane, not a boot-path surprise (a multi-minute
        # inline download here is exactly how the onboarding bounce returns).
        from hermes_cli.local_runtime.binaries import default_tag, installed_tags

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

        # Context policy: one launch decision per staged model, carried to
        # the router via the preset INI. Priced against CAPACITY, not live
        # free VRAM: this runs while the outgoing server instance may still
        # hold the card (restart, refresh after a download), and its memory
        # is freed before the new instance loads anything. Pricing against
        # live-free here once pinned a fitting model's weights to CPU
        # because the probe saw the predecessor's VRAM as gone.
        preset_path = runtimes_root() / "presets.ini"
        try:
            entries = generate_presets(mdir, probe_budget(planning=True), preset_path)
            for entry in entries:
                if entry.refusal:
                    logger.warning("model refused by physics check: %s", entry.refusal)
        except Exception as exc:  # noqa: BLE001 — policy failure must not block serving
            # Degradation ladder: a STALE policy still beats no policy —
            # stock fit (f16 KV at max context, no placement) is the
            # silent-busy-wait failure on Windows. Keep serving with the
            # previous INI when one exists; only a first boot with no INI
            # at all falls to stock fit.
            if preset_path.exists():
                logger.error("preset generation failed (%s); serving with the "
                             "PREVIOUS launch policies — models staged since "
                             "the last successful generation run unpoliced "
                             "until this is fixed", exc)
            else:
                logger.error("preset generation failed (%s) and no previous "
                             "policy file exists; router runs stock fit", exc)
                preset_path = None

        sup = LlamaServerSupervisor(
            install_dir, mdir,
            models_max=int(section.get("models_max", 4)),
            port=int(section.get("port", 0)) or None,
            preset_path=preset_path,
        )
        try:
            sup.start()
        except Exception:
            # start() can fail after the router process exists (health
            # timeout, spawn error): leaving it running unsupervised
            # strands its VRAM behind a port nothing will clean up.
            try:
                sup.stop()
            except Exception:  # noqa: BLE001 — cleanup is best-effort
                pass
            raise
        _SUPERVISOR = sup
        logger.info("managed llama-server up at %s (backend=%s tag=%s)",
                    sup.base_url, backend, tag)
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
    """The process-local supervisor, or None (server may still be running
    under another process — check the state file)."""
    return _SUPERVISOR


def _start_idle_sweeper(sup) -> None:
    """Idle-residency loop: every couple of minutes, unload non-primary
    models idle past the supervisor's threshold. Daemon thread tied to the
    supervisor's lifetime — exits when the server stops."""
    import threading

    def _loop():
        while sup.proc is not None and sup.proc.poll() is None:
            time.sleep(120)
            try:
                sup.sweep_idle()
            except Exception as exc:  # noqa: BLE001
                logger.debug("idle sweep skipped: %s", exc)

    threading.Thread(target=_loop, daemon=True,
                     name="local-runtime-idle-sweep").start()
