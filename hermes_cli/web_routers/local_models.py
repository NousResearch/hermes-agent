"""Local-models dashboard routes — the desktop's window into the managed llama.cpp runtime.

Every payload carries plain-language, pre-formatted facts the UI shows verbatim
(what will this model do ON THIS MACHINE, how big is the download, what is the
runtime doing), never raw internals. Long jobs follow the repo's job pattern:
start-POST -> {job_id} -> GET poll with byte progress.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from starlette.concurrency import run_in_threadpool

from hermes_cli import config as config_mod, web_deps
from hermes_cli.local_runtime import (
    binaries, bootstrap, catalog, context_policy, estimator, growth, hardware, hf_browse,
    load_progress, presets, supervisor,
)
from pm.downloader import Download, DownloadPaused, Source

from hermes_cli.local_runtime.endpoint import _state_endpoint

logger = logging.getLogger(__name__)

router = APIRouter()

_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_RUNNING: Dict[str, Dict[str, Any]] = {}
# One quickstart at a time: the job sequences installs, downloads, a server bounce and a config write — two
# racing runs would interleave all four. Held for the job's lifetime, released in the worker.
_QUICKSTART_LOCK = threading.Lock()
_LLAMACPP_PROVIDERS = ("llamacpp", "llama.cpp", "llama-cpp")
_SPLIT_PART_RE = r"-\d{5}-of-\d{5}"
_DOWNLOAD_PHASES = frozenset({"starting", "installing-runtime", "downloading-runtime", "downloading"})
_SERVER_START_FAILED = "The local server could not start — check the runtime is installed"


class RuntimeInstallBody(BaseModel):
    backend: Optional[str] = None   # None/auto -> detect


class ModelDownloadBody(BaseModel):
    model_id: str


class JobIdBody(BaseModel):
    job_id: str


class QuickstartBody(BaseModel):
    model_id: str | None = None   # default: the catalog's recommended entry


class ServerActionBody(BaseModel):
    action: str                 # "stop" | "start"


class ModelEjectBody(BaseModel):
    model_id: str


class ModelActivateBody(BaseModel):
    model_id: str               # exact variant id (a staged .gguf stem)


class BrowsedDownloadBody(BaseModel):
    repo: str
    paths: list[str]            # one GGUF, or every part of a split, in order


class SideloadBody(BaseModel):
    path: str                   # absolute path to a .gguf on this machine


def _human_gb(n: int | float) -> str:
    return f"{n / (1 << 30):.1f} GB"


def _k_label(tokens: int) -> str:
    return f"{tokens // 1024}K"


@contextlib.contextmanager
def _http_error(status: int, prefix: str = ""):
    """Map any exception to ``HTTPException(status, f"{prefix}{exc}")``."""
    try:
        yield
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status, detail=f"{prefix}{exc}") from exc


def _quiet(fn: Callable[[], Any], default: Any, *, warn: str | None = None, debug: str | None = None) -> Any:
    """``fn()`` or ``default`` on any exception — for garnish that must never 500. ``warn`` logs a
    warning with the exception (%s), ``debug`` a debug line with traceback; silent otherwise."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        if warn:
            logger.warning(warn, exc)
        if debug:
            logger.debug(debug, exc_info=True)
        return default


# ── jobs ─────────────────────────────────────────────────────
def _job(kind: str, target: str, model_id: str | None = None) -> Dict[str, Any]:
    job = {
        "job_id": uuid.uuid4().hex[:12], "kind": kind, "target": target,
        "model_id": model_id,       # catalog id for downloads; None otherwise
        "status": "running",        # running | paused | done | error
        "phase": "starting",        # human-readable step name
        "detail": "", "total_bytes": None, "done_bytes": 0, "started_at": time.time(), "error": None,
    }
    with _JOBS_LOCK:
        _JOBS[job["job_id"]] = job
    return job


def _job_view(job: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(job)
    running = _RUNNING.get(job["job_id"], {})
    pause = running.get("pause")
    out["pause_requested"] = bool(pause is not None and pause.is_set() and out["status"] == "running")
    out["can_pause"] = bool(pause is not None and out["status"] == "running"
                            and out["phase"] in _DOWNLOAD_PHASES and not out["pause_requested"])
    out["can_resume"] = out["status"] == "paused" and "resume" in running
    if out["total_bytes"]:
        out["percent"] = min(100, round(out["done_bytes"] / out["total_bytes"] * 100))
    return out


def _check_job_pause(job: Dict[str, Any]) -> None:
    pause = _RUNNING.get(job["job_id"], {}).get("pause")
    if pause is not None and pause.is_set():
        raise DownloadPaused("download paused")


def _step(job: Dict[str, Any], phase: str, detail: str) -> None:
    with _JOBS_LOCK:
        if job.get("phase") in _DOWNLOAD_PHASES and phase not in _DOWNLOAD_PHASES:
            _check_job_pause(job)
        job["phase"] = phase
        job["detail"] = detail


def _finish(job: Dict[str, Any], detail: str) -> None:
    # The worker publishes terminal status after releasing its resources.
    _step(job, "done", detail)


def _spawn_job(job: Dict[str, Any], name: str, body: Callable[[], None], *, fail_msg: str | None = None,
               on_exit: Callable[[], None] | None = None, download_label: str | None = None,
               resumable: bool = False) -> None:
    """One worker per job. Pauses retain ownership; terminal outcomes release it."""
    guard = threading.Lock()
    pause = threading.Event()

    def _run():
        status = "done"
        try:
            body()
            if download_label is not None:
                _finish(job, f"{download_label} ready")
                _refresh_runtime("post-download runtime refresh skipped")
        except DownloadPaused:
            status = "paused"
        except Exception as exc:  # noqa: BLE001
            if fail_msg:
                logger.warning(fail_msg, exc)
            status = "error"
            job["error"] = str(exc)
        finally:
            try:
                if status != "paused" and on_exit is not None:
                    on_exit()
            finally:
                with _JOBS_LOCK:
                    if status != "paused":
                        _RUNNING.pop(job["job_id"], None)
                    job["status"] = status
                    guard.release()

    def start(*, initial: bool = False) -> bool:
        if not guard.acquire(blocking=False):
            return False
        with _JOBS_LOCK:
            if not initial and job["status"] != "paused":
                guard.release()
                return False
            pause.clear()
            job["status"] = "running"
            job["error"] = None
        try:
            threading.Thread(target=_run, daemon=True, name=name).start()
        except Exception:
            _RUNNING.pop(job["job_id"], None)
            if on_exit is not None:
                on_exit()
            job["status"] = "error"
            guard.release()
            raise
        return True

    if resumable:
        _RUNNING[job["job_id"]] = {"resume": start, "pause": pause}
    start(initial=True)


# ── runtime / router plumbing ────────────────────────────────
def _refresh_runtime(skip_msg: str) -> None:
    """Bounce a running router so it rescans the models dir (it only scans at spawn).
    Never raises — the file operation already succeeded."""
    _quiet(bootstrap.refresh_local_runtime, None, debug=skip_msg)


def _router_request(endpoint: Dict[str, Any], path: str, *, timeout: float, payload: dict | None = None) -> Any:
    """Call the local router (base_url minus ``/v1``) with its bearer key; GET (no payload) -> parsed JSON, POST -> None."""
    headers = {"Authorization": f"Bearer {endpoint.get('api_key', '')}"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()
    req = urllib.request.Request(endpoint["base_url"].rsplit("/v1", 1)[0] + path, data=data, headers=headers,
                                 method="POST" if payload is not None else None)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return None if payload is not None else json.loads(r.read())


def _load_config() -> dict:
    return _quiet(config_mod.load_config, {})


def _runtime_section() -> dict:
    return (_load_config() or {}).get("local_runtime") or {}


def _set_runtime_enabled(enabled: bool) -> dict:
    """Persist ``local_runtime.enabled`` and return the config written."""
    config = config_mod.load_config()
    config.setdefault("local_runtime", {})["enabled"] = enabled
    config_mod.save_config(config)
    return config


def _runtime_target(requested: str | None = None) -> tuple[str, str]:
    """Resolve only reviewed PM artifacts; hardware work stays off the event loop."""
    section = _runtime_section()
    backend = requested or section.get("backend", "auto")
    with _http_error(400):
        backend = binaries.resolve_backend(backend)
        return binaries.pinned_tag(backend), backend


def _engine_too_old(min_engine: str) -> bool:
    if not min_engine:
        return False
    requested = _runtime_section().get("backend", "auto")
    engine = binaries.installed_engine(requested)
    tag = engine.tag if engine is not None else _runtime_target()[0]
    return int(tag.removeprefix("b")) < int(min_engine.removeprefix("b"))


def _eligible_entries():
    """Catalog entries this engine can activate today (engine-gated ones can't be the recommendation either)."""
    return tuple(e for e in catalog.CATALOG if not _engine_too_old(e.min_engine))


def _entry_or_404(model_id: str):
    entry = catalog.catalog_by_id().get(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"unknown model {model_id}")
    return entry


def _start_local_server(config: dict, fail_detail: str):
    """Force-start the local server; raise ``fail_detail`` when neither we nor another process ended up serving."""
    sup = bootstrap.ensure_local_runtime(config, force=True)
    if sup is None and _state_endpoint() is None:
        raise RuntimeError(fail_detail)
    return sup


def _ensure_server(job: Dict[str, Any], config: dict, model_id: str, *, fail_detail: str, skip_msg: str) -> None:
    """Start the local server if needed and self-heal a stale router: the model list is spawn-only, so a
    server started before ``model_id`` finished downloading can't serve it — bounce it when it doesn't know it."""
    _step(job, "starting-server", "Starting the local server")
    sup = _start_local_server(config, fail_detail)

    def rescan_if_unknown() -> None:
        if model_id not in sup.models():
            job["detail"] = "Refreshing the local server"
            bootstrap.refresh_local_runtime()

    if sup is not None:
        _quiet(rescan_if_unknown, None, debug=skip_msg)


def _assign_default(job: Dict[str, Any], model_id: str) -> None:
    """Make ``model_id`` the main model via the same machinery as /api/model/set."""
    _step(job, "setting-default", "Making it your default")
    web_deps.late("_apply_model_assignment_sync", "hermes_cli.web_server_config")("main", "llamacpp", model_id, "", "", "")


# ── downloads: ranged parallel streams ───────────────────────
def _hf_url(repo: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{path}"


def _model_id_for(gguf: Path) -> str:
    """Variant model id for a staged file (strips split-part suffixes)."""
    return re.sub(_SPLIT_PART_RE + "$", "", gguf.stem)


def _variant_files_on_disk(model_id: str) -> "list[Path]":
    """Every local file of a staged model: all split parts plus catalog-declared assets (mmproj/draft) when present."""
    files = [p for p in bootstrap.models_dir().glob("*.gguf") if _model_id_for(p) == model_id]
    hit = catalog.find_entry_for_model(model_id)
    assets = (hit[0].mmproj, hit[0].draft) if hit is not None else ()
    files += [bootstrap.assets_dir() / a.local_name for a in assets
              if a is not None and (bootstrap.assets_dir() / a.local_name).exists()]
    return files


def _download_plan(entry, variant) -> list:
    """Everything a variant needs: split parts + mmproj/draft assets, as (url, dest, bytes) tuples."""
    plan = [(_hf_url(entry.repo, a.path), bootstrap.models_dir() / a.local_name, a.size_bytes) for a in variant.files]
    plan += [(_hf_url(entry.repo, a.path), bootstrap.assets_dir() / a.local_name, a.size_bytes)
             for a in (entry.mmproj, entry.draft) if a is not None]
    return plan


def _run_download_plan(job: Dict[str, Any], plan: list, label: str) -> None:
    """The downloader counts completed files and every part of this plan."""
    _step(job, "downloading", f"Downloading {label}")
    _download_job(job, plan)


def _download_progress_hook(job: Dict[str, Any]):
    def tick(done: int, total: int, ranges: dict) -> None:
        with _JOBS_LOCK:
            job.update(done_bytes=done, total_bytes=total or None, ranges=ranges)
    return tick


def _download_job(job: Dict[str, Any], plan) -> None:
    """Model bytes use the same resumable transfer as pinned PM archives."""
    running = _RUNNING.get(job["job_id"], {})
    dl = Download([Source(url, dest) for url, dest, _ in plan], pause_event=running.get("pause"))
    dl.run(progress=_download_progress_hook(job))


def _loaded_models(running: Dict[str, Any]) -> "tuple[Dict[str, str], Dict[str, Any]]":
    """Resident models right now, plus how each is placed (granted window from the child, spill facts from
    the preset decision) — the difference between 'fast' and 'why is my CPU busy', so it must be inspectable.
    'loading' is its own state (a 20-GB load in flight is the most important thing the pane can show)."""
    data = _router_request(running, "/models", timeout=3)
    loaded = {m["id"]: m.get("status", {}).get("value", "unknown") for m in data.get("data", [])
              if m.get("status", {}).get("value") in ("loaded", "ready", "loading")}
    placement: Dict[str, Any] = {}
    decisions = presets.read_preset_decisions()
    for model_id, state in loaded.items():
        facts: Dict[str, Any] = {}
        plan = decisions.get(model_id)
        if plan is not None:
            facts.update(window=plan.window, window_label=_k_label(plan.window), spilled=plan.spilled)
        n_ctx = state in ("loaded", "ready") and _quiet(
            lambda: _router_request(running, f"/props?model={model_id}", timeout=3)
            .get("default_generation_settings", {}).get("n_ctx"), None)
        if n_ctx:
            facts.update(granted_window=int(n_ctx), granted_window_label=_k_label(int(n_ctx)))
        if facts:
            placement[model_id] = facts
    return loaded, placement


def _staged_row(gguf: Path) -> Dict[str, Any]:
    model_id = _model_id_for(gguf)
    # Split models: report the whole variant's bytes, not one part's.
    hit = catalog.find_entry_for_model(model_id)
    size = hit[1].size_bytes if hit is not None else gguf.stat().st_size
    return {"id": model_id, "size_bytes": size, "size_label": _human_gb(size)}


def _active_llamacpp_model_id() -> str | None:
    """The active main model when it is one of ours (config authority: the model.provider + model.default
    that /api/model/set writes)."""
    def read() -> str | None:
        model_section = (_load_config() or {}).get("model") or {}
        if str(model_section.get("provider", "")).strip().lower() in _LLAMACPP_PROVIDERS:
            return str(model_section.get("default") or model_section.get("name") or "").strip() or None
        return None

    return _quiet(read, None)


@router.get("/api/local-models/status")
def local_models_status():
    """Cheap, immediate: config state + installed runtime + staged models + supervisor state (GPU facts live
    in /hardware). Sync def on purpose: blocking urlopen/scans run in the threadpool."""
    section = _runtime_section()
    engine = binaries.installed_engine(section.get("backend", "auto"))
    runtime_backend = engine.backend if engine is not None else None
    configured_tag = binaries.pinned_tag(runtime_backend) if runtime_backend else _runtime_target()[0]
    tag = engine.tag if engine is not None else configured_tag
    import pm

    current = pm.installed_package(binaries.BACKEND_PACKAGES[runtime_backend]) if runtime_backend else None
    mdir = bootstrap.models_dir()
    running = _state_endpoint()
    # Resident models from the live router ({} when down): Loaded pills + eject. A failed read is never
    # silent: an empty dict here renders as 'Not in memory' on a machine whose VRAM is visibly full.
    loaded, placement = ({}, {}) if running is None else _quiet(
        lambda: _loaded_models(running), ({}, {}), warn="loaded-models read failed: %r")
    return {
        "enabled": bool(section.get("enabled")), "tag": tag, "configured_tag": configured_tag,
        "update_available": bool(section.get("enabled") and engine is not None and current is None),
        "runtime_installed": runtime_backend is not None, "runtime_backend": runtime_backend,
        "server_running": running is not None, "server_base_url": (running or {}).get("base_url"),
        "active_model_id": _active_llamacpp_model_id(), "loaded_models": loaded,
        # Live load progress per model (SSE-fed): {model_id: {stage, value, percent}}.
        # The chat's loading bar and the picker rows poll this; garnish, never a 500.
        "loading": _quiet(load_progress.get_loading_progress, {}),
        "placement": placement,
        "models": [_staged_row(gguf) for gguf in bootstrap.staged_models()] if mdir.exists() else [],
        "models_dir": str(mdir),
    }


# ── hardware: what this machine can do ───────────────────────
def _nvidia_smi_facts() -> dict:
    """GPU identity + live utilization (NVIDIA only; other vendors degrade to {} and the UI hides those readouts)."""
    smi_exe = hardware._nvidia_smi_path()
    if not smi_exe:
        return {}
    smi = subprocess.run([smi_exe, "--query-gpu=name,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=5)
    if smi.returncode != 0 or not smi.stdout.strip():
        return {}
    name, util, used_mib = (x.strip() for x in smi.stdout.strip().splitlines()[0].split(","))
    return dict(gpu_name=name, gpu_util_percent=int(util), vram_used_bytes=int(used_mib) << 20)


@router.get("/api/local-models/hardware")
def local_models_hardware():
    """The budget as plain facts, polled by the pane and statusbar. Sync def: shells out to nvidia-smi — threadpool."""
    budget = hardware.probe_budget()
    ram_total, ram_avail = hardware._ram_bytes()
    out = {
        "uma": budget.uma, "vram_total_bytes": budget.total_device_bytes, "vram_usable_bytes": budget.usable_vram_bytes,
        "ram_total_bytes": ram_total, "ram_available_bytes": ram_avail, "vram_label": _human_gb(budget.total_device_bytes),
        "gpu_name": None, "gpu_util_percent": None, "vram_used_bytes": None,
    }
    out.update(_quiet(_nvidia_smi_facts, {}))
    return out


# ── catalog: priced for THIS machine before download ─────────
_QUANT_REASONS = {
    "best-large-window": ("Recommended build ({quant}) — the quant class this engine is optimized for; "
                          "runs fully on your GPU with a large context window"),
    "best-fits": ("Recommended build ({quant}) — the quant class this engine is optimized for; "
                  "runs fully on your GPU"),
}
_QUANT_REASON_COMPACT = "Compact build sized for this machine ({quant}) — larger than GPU memory, runs slower"


def _catalog_row(entry, budget, recommended, recommended_reason, staged_ids) -> Dict[str, Any]:
    choice = catalog.select_variant(entry, budget)
    # Any variant of this family on disk counts as downloaded.
    dl = next((v for v in entry.variants if v.model_id in staged_ids
               and all(dest.is_file() for _, dest, _ in _download_plan(entry, v))), None)
    row: Dict[str, Any] = {
        "id": entry.id, "display_name": entry.display_name, "description": entry.description,
        "native_context": entry.n_ctx_train, "native_context_label": _k_label(entry.n_ctx_train),
        "recommended": entry.id == recommended,
        "recommended_reason": recommended_reason if entry.id == recommended else None,
        "downloaded": dl is not None, "downloaded_model_id": dl.model_id if dl else None,
        "downloaded_quant": dl.quant if dl else None, "mtp": entry.mtp, "vision": entry.mmproj is not None,
        # Day-0 architectures need the llama.cpp release where their support landed: True gates
        # download/activate until the engine updates, but the row still renders (visible + explained beats hidden).
        "needs_engine": _engine_too_old(entry.min_engine),
        "min_engine": entry.min_engine or None,
    }
    if choice is None:
        smallest = min(entry.variants, key=lambda v: v.size_bytes)
        smallest_total = entry.download_bytes(smallest)
        row.update({
            "fits": False, "size_bytes": smallest_total, "size_label": _human_gb(smallest_total),
            "fit_summary": "Needs more memory than this machine has",
            "fit_detail": (f"even the most compact build ({smallest.quant}, {_human_gb(smallest_total)}) "
                           "exceeds GPU + system memory"),
        })
        return row

    variant = choice.variant
    decision = entry.launch_plan(variant, budget).decision
    download_total = entry.download_bytes(variant)
    row.update({
        "fits": True, "model_id": variant.model_id, "quant": variant.quant,
        "quant_validated": variant.validated, "size_bytes": download_total,
        "size_label": _human_gb(download_total), "variant_count": len(entry.variants),
        "quant_reason": _QUANT_REASONS.get(choice.reason_key, _QUANT_REASON_COMPACT).format(quant=variant.quant),
    })
    if isinstance(decision, estimator.PhysicsRefusal):
        row["fit_summary"] = row["quant_reason"]
        return row
    row.update(start_window=decision.window, start_window_label=_k_label(decision.window), spilled=decision.spilled)
    if decision.window >= entry.n_ctx_train:
        shape = f"runs at its full {row['native_context_label']} context"
    else:
        shape = f"starts at {row['start_window_label']} and grows toward {row['native_context_label']} as you use it"
    row["fit_summary"] = shape + (" (larger than your GPU memory — runs slower)" if decision.spilled else "")
    return row


@router.get("/api/local-models/catalog")
def local_models_catalog():
    """Every entry answers up front: how big is the download, will it fit, what context/speed shape will I
    get. The row advertises the BEST build for this machine (highest quality fully on GPU at the 64K floor;
    else the smallest that works, spilled and priced). No entry is hidden; unaffordable models show WHY.
    Sync def: blocking I/O -> threadpool."""
    # Serve the in-memory catalog; a TTL-gated background fetch lands new entries for the next call
    # (day-0 models without an app release).
    catalog.refresh_catalog_soon()
    # Planning budget: machine capacity, not live-free VRAM — a loaded model must not make every row unaffordable.
    budget = hardware.probe_budget(planning=True)
    # The reason key ships with the row so the Recommended badge's tooltip is the branch that actually
    # fired, not a re-derivation that can drift.
    recommended, recommended_reason = catalog.recommended_entry(budget, _eligible_entries()) or (None, None)
    recommended_id = recommended.id if recommended is not None else None
    # Completeness-checked staging (split parts all present) — same answer the picker and router see, so a
    # mid-download model never reads as downloaded.
    staged_ids = set(bootstrap.staged_model_ids())
    return {"models": [_catalog_row(e, budget, recommended_id, recommended_reason, staged_ids) for e in catalog.CATALOG]}


# ── runtime install (job) ────────────────────────────────────
def _runtime_progress_hook(job: Dict[str, Any]):
    """PM owns byte accounting; stages describe work without resetting it."""
    phases = {
        "download": ("downloading-runtime", "Downloading the local engine"),
        "unpack": ("unpacking-runtime", "Unpacking the local engine"),
        "verify": ("verifying-runtime", "Verifying the local engine"),
    }

    def hook(stage: str, done: int, total: int, label: str) -> None:
        phase, detail = phases[stage]
        _step(job, phase, detail + (f" ({label})" if label else ""))
    return hook


def _install_engine_job(job: Dict[str, Any], backend: str):
    _step(job, "installing-runtime", "Preparing the pinned local engine")
    return binaries.ensure_engine(
        backend, progress=_runtime_progress_hook(job),
        download_progress=_download_progress_hook(job),
        pause_event=_RUNNING[job["job_id"]]["pause"],
    )


@router.post("/api/local-models/runtime/install")
def local_models_runtime_install(body: RuntimeInstallBody):
    tag, backend = _runtime_target(body.backend)
    job = _job("runtime-install", f"llama.cpp {tag} ({backend})")

    def run():
        previous = binaries.installed_engine(backend)
        engine = _install_engine_job(job, backend)
        running = bootstrap.get_supervisor()
        if running is not None and previous != engine:
            _step(job, "restarting", "Switching the running server to the pinned build")
            bootstrap.refresh_local_runtime()
        _finish(job, f"llama.cpp {tag} ready ({backend})")

    _spawn_job(job, "lr-runtime-install", run, fail_msg="runtime install failed: %s", resumable=True)
    return {"job_id": job["job_id"], "backend": backend, "tag": tag}


# ── model download (job with byte progress) ──────────────────
def _download_target(model_id: str):
    """(entry, variant) for a family id (this machine's selected variant — the same planning budget as the
    catalog, so the user downloads exactly the build the row advertised) or an exact variant model_id."""
    entry = catalog.catalog_by_id().get(model_id)
    if entry is None:  # exact variant id, or nothing we know (404)
        return catalog.find_entry_for_model(model_id) or _entry_or_404(model_id)
    if _engine_too_old(entry.min_engine):
        raise HTTPException(status_code=409, detail=(
            f"{entry.display_name} needs llama.cpp {entry.min_engine} or newer — update the engine first"))
    choice = catalog.select_variant(entry, hardware.probe_budget(planning=True))
    if choice is None:
        raise HTTPException(status_code=409, detail=f"no variant of {entry.id} fits this machine")
    return entry, choice.variant


@router.post("/api/local-models/download")
def local_models_download(body: ModelDownloadBody):
    """Accepts either a family id (downloads this machine's selected variant) or an exact variant model_id."""
    entry, variant = _download_target(body.model_id)
    plan = _download_plan(entry, variant)
    if plan and all(dest.is_file() for _, dest, _ in plan):
        return {"job_id": None, "already_downloaded": True, "model_id": variant.model_id}
    job = _job("model-download", f"{entry.display_name} ({variant.quant})", model_id=entry.id)
    def _run():
        _run_download_plan(job, plan, entry.display_name)
        _finish(job, f"{entry.display_name} ready")
        _refresh_runtime("post-download runtime refresh skipped")

    _spawn_job(job, "lr-model-download", _run, fail_msg="model download failed: %s", resumable=True)
    return {"job_id": job["job_id"], "model_id": variant.model_id}


@router.post("/api/local-models/download/pause")
async def local_models_download_pause(body: JobIdBody):
    """Pause the download phase of a model, component or quickstart job."""
    job = _JOBS.get(body.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown download job")
    with _JOBS_LOCK:
        if not _job_view(job)["can_pause"]:
            return {"ok": True, "paused": False}
        _RUNNING[body.job_id]["pause"].set()
    return {"ok": True, "paused": True}


@router.post("/api/local-models/download/resume")
async def local_models_download_resume(body: JobIdBody):
    """Restart the same job; PM reuses verified files and durable ranges."""
    if body.job_id not in _JOBS:
        raise HTTPException(status_code=404, detail="unknown download job")
    resume = _RUNNING.get(body.job_id, {}).get("resume")
    return {"ok": True, "resumed": bool(resume and resume())}


@router.delete("/api/local-models/models/{model_id}")
async def local_models_delete(model_id: str):
    """Remove every split part plus private assets, then bounce the router off the request thread (deleting
    the active file mid-serve is exactly the stale state the refresh exists for)."""
    files = _variant_files_on_disk(model_id)
    if not files:
        raise HTTPException(status_code=404, detail="model not found")
    for path in files:
        path.unlink(missing_ok=True)
    # Growth state dies with the model: a re-download starts back at its zero-spill window, not a stale grown one.
    _quiet(lambda: growth.clear_window_override(model_id), None, debug="window-override clear skipped")
    threading.Thread(target=_refresh_runtime, args=("post-delete runtime refresh skipped",), daemon=True,
                     name="lr-post-delete").start()
    return {"ok": True}


# ── quickstart: one click from nothing to a working default ──


def _quickstart_target(body: QuickstartBody, budget):
    """Resolve an explicit model, or start with the machine's automatic recommendation.

    With no recommendation, require an explicit choice before starting setup.
    """
    if body.model_id:
        candidates = [_entry_or_404(body.model_id)]
    else:
        picked = catalog.recommended_entry(budget, _eligible_entries())
        if picked is None:
            raise HTTPException(
                status_code=409,
                detail="No automatic recommendation for this machine — open Local Models to browse or choose a model explicitly",
            )
        candidates = [picked[0]] + [e for e in catalog.CATALOG if e.id != picked[0].id]
    for candidate in candidates:
        choice = catalog.select_variant(candidate, budget)
        if choice is not None and not _engine_too_old(candidate.min_engine):
            return candidate, choice.variant
    raise HTTPException(status_code=409, detail=(
        "no catalog model fits this machine — open Local Models to browse for a smaller build"))


@router.post("/api/local-models/quickstart")
def local_models_quickstart(body: QuickstartBody):
    """One job: install the runtime (if missing), download this machine's build of the recommended model (if
    missing), make it the default. Each leg uses the same code as the individual setup routes.
    Preflight rejects (no automatic recommendation or no servable choice) fail the POST
    synchronously so the button can explain itself; everything slow runs in the job with phase/byte progress."""
    entry, variant = _quickstart_target(body, hardware.probe_budget(planning=True))
    tag, backend = _runtime_target()
    need_runtime = binaries.installed_engine(backend) is None
    download_plan = _download_plan(entry, variant)
    need_download = any(not dest.is_file() for _, dest, _ in download_plan)
    if not need_download:
        download_plan = []
    download_bytes = sum(p[2] for p in download_plan)
    if not _QUICKSTART_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Setup is already running")
    job = _job("quickstart", entry.display_name, model_id=entry.id)
    def _run():
        if need_runtime and binaries.installed_engine(backend) is None:
            _install_engine_job(job, backend)
        if need_download:
            # Each phase has its own complete download plan. Resume within a
            # phase retains counters until PM reports the durable bytes.
            if job["phase"] != "downloading":
                with _JOBS_LOCK:
                    job.update(done_bytes=0, total_bytes=None, ranges={})
            _run_download_plan(job, download_plan, entry.display_name)
        _step(job, "starting-server", "Starting the local server")
        _ensure_server(job, _set_runtime_enabled(True), variant.model_id,
                       fail_detail="The local server could not start — open Local Models for details",
                       skip_msg="quickstart rescan check skipped")
        _assign_default(job, variant.model_id)
        _finish(job, f"{entry.display_name} is ready — new chats use it")

    _spawn_job(job, "lr-quickstart", _run, fail_msg="quickstart failed: %s",
               on_exit=_QUICKSTART_LOCK.release, resumable=True)
    return {"job_id": job["job_id"], "model_id": entry.id, "display_name": entry.display_name,
            "needs_runtime": need_runtime, "needs_download": need_download, "download_bytes": download_bytes}


# ── server lifecycle: turn the engine on/off ─────────────────
def _terminate_state_pid() -> None:
    """Server owned by another process (or an orphan): terminate via the state file's pid, then clear the state."""
    import psutil  # type: ignore

    state = json.loads(supervisor.state_path().read_text(encoding="utf-8-sig"))
    pid = int(state.get("pid") or 0)
    if pid > 0 and psutil.pid_exists(pid):
        psutil.Process(pid).terminate()
    supervisor.state_path().unlink(missing_ok=True)


def _stop_server() -> None:
    if bootstrap.get_supervisor() is not None:
        bootstrap.shutdown_local_runtime()
    elif _state_endpoint() is not None:
        _quiet(_terminate_state_pid, None)  # best-effort
    _set_runtime_enabled(False)


def _start_server() -> None:
    _start_local_server(_set_runtime_enabled(True), _SERVER_START_FAILED)


_SERVER_ACTIONS = {"stop": _stop_server, "start": _start_server}


@router.post("/api/local-models/server")
async def local_models_server(body: ServerActionBody):
    """Turn the local engine off (stop the server, free ALL GPU memory, disable auto-start) or back on. Unlike
    per-model eject the off switch IS durable: the user said off, so boots stay off until they say on."""
    action = (body.action or "").strip().lower()
    if action not in _SERVER_ACTIONS:
        raise HTTPException(status_code=400, detail="action must be 'stop' or 'start'")
    with _http_error(502):
        await asyncio.to_thread(_SERVER_ACTIONS[action])
    return {"ok": True, "action": action}


# ── eject / activate ─────────────────────────────────────────
@router.post("/api/local-models/eject")
def local_models_eject(body: ModelEjectBody):
    """Free a loaded model's GPU memory now; only demand (the next message) reloads it — residency v2 has no
    automatic loading anywhere. Sync def: the fallback path blocks on a 120s urlopen — threadpool, never the loop."""
    sup = bootstrap.get_supervisor()
    if sup is not None:
        with _http_error(502):
            sup.unload_model(body.model_id)
        return {"ok": True}
    # Server owned by another process (or state-file only): drive the router directly with the persisted endpoint.
    endpoint = _state_endpoint()
    if endpoint is None:
        raise HTTPException(status_code=409, detail="local server is not running")
    with _http_error(502):
        _router_request(endpoint, "/models/unload", timeout=120, payload={"model": body.model_id})
    return {"ok": True}


@router.post("/api/local-models/activate")
async def local_models_activate(body: ModelActivateBody):
    """Make a downloaded model the default for new chats: a config write via the same machinery as
    /api/model/set plus making sure the server is up. NO model loading (residency v2: models load on first
    inference; an empty router costs nothing). Kept as a job for UI continuity."""
    # Split variants stage under their first part — resolve like the other routes.
    if body.model_id not in bootstrap.staged_model_ids():
        raise HTTPException(status_code=404, detail=f"{body.model_id} is not downloaded")
    job = _job("model-activate", body.model_id, model_id=body.model_id)

    def _run():
        _ensure_server(job, config_mod.load_config(), body.model_id,
                       fail_detail=_SERVER_START_FAILED, skip_msg="activate rescan check skipped")
        _step(job, "setting-default", "Making it your default")
        _set_runtime_enabled(True)
        _assign_default(job, body.model_id)
        _finish(job, f"{body.model_id} is the default for new chats")

    _spawn_job(job, "lr-model-activate", _run, fail_msg="model activate failed: %s")
    return {"job_id": job["job_id"]}


# ── job polling ──────────────────────────────────────────────
@router.get("/api/local-models/jobs")
async def local_models_jobs():
    """All recent jobs, running first — the pane and app-level poller rediscover in-flight work here after a remount."""
    with _JOBS_LOCK:
        jobs = sorted(_JOBS.values(), key=lambda job: -job["started_at"])
        active = [job for job in jobs if job["status"] in ("running", "paused")]
        recent = [job for job in jobs if job["status"] in ("done", "error")][:20]
        return {"jobs": [_job_view(job) for job in [*active, *recent]]}


@router.get("/api/local-models/jobs/{job_id}")
async def local_models_job(job_id: str):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return _job_view(job)


# ── Hugging Face browser: search, repo files, arbitrary download ─
@router.get("/api/local-models/search")
async def local_models_search(q: str, limit: int = 20):
    """Full-text HF search over GGUF models — the firehose behind the curated catalog; fit pills come from /search/files."""
    if not q.strip():
        return {"hits": []}
    with _http_error(502, "Hugging Face search unavailable: "):
        return {"hits": [h.__dict__ for h in await run_in_threadpool(hf_browse.search_models, q, limit)]}


@router.get("/api/local-models/search/files")
async def local_models_search_files(repo: str):
    """Servable GGUFs in one HF repo with a rough pre-download fit verdict per quant (file size + conservative
    fill-ins; the GGUF header refines it)."""
    with _http_error(502, f"Could not list {repo}: "):
        groups = await run_in_threadpool(hf_browse.priced_repo_files, repo, hardware.probe_budget(planning=True))
    return {"files": [dict(g.__dict__, paths=list(g.paths)) for g in groups]}


@router.post("/api/local-models/download-browsed")
async def local_models_download_browsed(body: BrowsedDownloadBody):
    """Download an arbitrary HF GGUF into the managed models dir. Once landed it is a normal staged model (the
    post-download bounce regenerates presets from its real header); with no catalog entry it serves
    'unverified', capabilities answered from the live server only."""
    paths = [p for p in (body.paths or []) if p.lower().endswith(".gguf")]
    if not paths:
        raise HTTPException(status_code=422, detail="no .gguf files given")
    model_id = re.sub(rf"(?:{_SPLIT_PART_RE})?\.gguf$", "", paths[0].rsplit("/", 1)[-1], flags=re.IGNORECASE)
    if model_id in bootstrap.staged_model_ids():
        return {"job_id": None, "already_downloaded": True, "model_id": model_id}
    job = _job("model-download", f"{model_id} (from {body.repo})", model_id=model_id)

    plan = [(_hf_url(body.repo, urllib.parse.quote(path)),
             bootstrap.models_dir() / path.rsplit("/", 1)[-1], 0) for path in paths]

    def _fetch():
        _run_download_plan(job, plan, model_id)

    _spawn_job(job, "lm-download-browsed", _fetch, download_label=model_id, resumable=True)
    return {"job_id": job["job_id"], "model_id": model_id}


@router.post("/api/local-models/sideload")
async def local_models_sideload(body: SideloadBody):
    """Register a GGUF already on this machine: link it into the managed models dir (copy only when linking is
    impossible) and bounce the router. The original stays put; delete-from-Hermes removes only our link."""
    src = Path(body.path)
    if not src.is_file() or src.suffix.lower() != ".gguf":
        raise HTTPException(status_code=422, detail="Pick a .gguf model file")
    dest = bootstrap.models_dir() / src.name
    if dest.exists():
        return {"ok": True, "model_id": dest.stem, "already_present": True}
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)          # hardlink: instant, no extra disk
    except OSError:
        try:
            os.symlink(src, dest)   # cross-volume fallback
        except OSError:
            await run_in_threadpool(shutil.copyfile, src, dest)
    _refresh_runtime("post-sideload runtime refresh skipped")
    return {"ok": True, "model_id": dest.stem}
