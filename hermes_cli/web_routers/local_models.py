"""Local-models dashboard routes — the desktop's window into the managed
llama.cpp runtime.

Everything here is designed for a first-run user on an RTX laptop: every
payload carries plain-language, pre-formatted facts the UI can show verbatim
(what will this model do ON THIS MACHINE, how big is the download, what is
the runtime doing right now), never raw internals the renderer would have to
interpret.

Long jobs (runtime install, model download) follow the repo's job pattern:
start-POST -> {job_id} -> GET poll with byte progress. Downloads are
byte-size checked against the catalog (no hash verification by design);
a short download deletes the file and reports it plainly.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from hermes_cli.local_runtime.endpoint import _state_endpoint

logger = logging.getLogger(__name__)

router = APIRouter()

_GIB = 1 << 30
_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


def _human_gb(n: int | float) -> str:
    return f"{n / _GIB:.1f} GB"


def _job(kind: str, target: str, model_id: str | None = None) -> Dict[str, Any]:
    job = {
        "job_id": uuid.uuid4().hex[:12],
        "kind": kind,               # "runtime-install" | "model-download"
        "target": target,
        "model_id": model_id,       # catalog id for downloads; None otherwise
        "status": "running",        # running | done | error
        "phase": "starting",        # human-readable step name
        "detail": "",
        "total_bytes": None,
        "done_bytes": 0,
        "started_at": time.time(),
        "error": None,
    }
    with _JOBS_LOCK:
        _JOBS[job["job_id"]] = job
    return job


# ── fast download: ranged parallel streams ───────────────────

# One TCP stream to a CDN rarely fills a fast line; 8 ranged connections
# writing into a preallocated file saturate consumer gigabit.
_DOWNLOAD_CONNECTIONS = 8
_CHUNK = 4 << 20


def _probe_range_support(url: str) -> int:
    """Total size when the server honors Range requests, else 0.

    Auth-shaped failures raise with a plain-language message — a 401/403
    from the CDN means the repo is gated or the catalog entry names a
    wrong repo, and the user deserves better than a bare status code.
    """
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            if r.status == 206:
                content_range = r.headers.get("Content-Range", "")
                if "/" in content_range:
                    return int(content_range.rsplit("/", 1)[1])
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise RuntimeError(
                "The model host refused the download (gated or moved). "
                "This is a catalog problem, not yours — please report it.") from exc
        raise
    except Exception:  # noqa: BLE001
        pass
    return 0


def _model_id_for(gguf: Path) -> str:
    """Variant model id for a staged file (strips split-part suffixes)."""
    import re

    return re.sub(r"-\d{5}-of-\d{5}$", "", gguf.stem)


def _variant_files_on_disk(model_id: str) -> "list[Path]":
    """Every local file belonging to a staged model: all split parts plus
    its catalog-declared assets (mmproj/draft) when present."""
    from hermes_cli.local_runtime.bootstrap import assets_dir
    from hermes_cli.local_runtime.catalog import find_entry_for_model

    mdir = _models_dir()
    files = [p for p in mdir.glob("*.gguf") if _model_id_for(p) == model_id]
    hit = find_entry_for_model(model_id)
    if hit is not None:
        entry, _variant = hit
        for asset in (entry.mmproj, entry.draft):
            if asset is not None:
                p = assets_dir() / asset.local_name
                if p.exists():
                    files.append(p)
    return files


def download_file(url: str, dest: Path, job: Dict[str, Any],
                  *,
                  base_done: int = 0, keep_totals: bool = False) -> None:
    """Download url -> dest with byte progress on ``job``.

    Ranged-parallel when the server supports it, single-stream fallback
    otherwise. There is no integrity check against the CATALOG by
    design: catalog sizes may lag an upstream re-upload, and a
    newer file than we know about must download fine. Completeness is
    checked only against what the SERVER declared for this transfer
    (range-probe total / Content-Length) — self-consistent and always
    current — so a dropped connection still errors instead of staging a
    truncated file. Never leaves a .part behind.

    Multi-file variants: ``base_done`` offsets the progress so this file's
    bytes accumulate onto the files before it, and ``keep_totals=True``
    stops the per-file size from overwriting the variant's total.
    """
    import shutil
    import threading as _threading

    tmp = dest.with_suffix(".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    file_done = [0]
    progress_lock = _threading.Lock()

    def bump(n: int) -> None:
        with progress_lock:
            file_done[0] += n
            job["done_bytes"] = base_done + file_done[0]

    try:
        # The probe and the preallocation both take real seconds on a
        # 20+ GB file — narrate them, or the pane shows a dead '— of X GB'
        # until the first ranged byte lands.
        job["detail"] = "Connecting"
        total = _probe_range_support(url)
        if total:
            if not keep_totals:
                job["total_bytes"] = total
            # Preallocate so each worker writes at its own offset.
            job["detail"] = f"Reserving {_human_gb(total)} of disk space"
            with open(tmp, "wb") as f:
                f.truncate(total)
            job["detail"] = ""
            errors: list[Exception] = []
            bounds = [(i * total // _DOWNLOAD_CONNECTIONS,
                       (i + 1) * total // _DOWNLOAD_CONNECTIONS - 1)
                      for i in range(_DOWNLOAD_CONNECTIONS)]

            def fetch_range(start: int, end: int) -> None:
                try:
                    req = urllib.request.Request(
                        url, headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(req, timeout=120) as r, \
                            open(tmp, "r+b") as f:
                        f.seek(start)
                        while True:
                            chunk = r.read(_CHUNK)
                            if not chunk:
                                break
                            f.write(chunk)
                            bump(len(chunk))
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [_threading.Thread(target=fetch_range, args=b, daemon=True,
                                         name=f"lm-dl-{i}")
                       for i, b in enumerate(bounds)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            if errors:
                raise errors[0]
            if file_done[0] != total:
                raise RuntimeError(
                    f"download incomplete ({file_done[0]} of {total} bytes)")
        else:
            # No range support: single stream, large chunks. Completeness
            # is judged by the server's own Content-Length when it sent
            # one — never by the catalog, which may lag a re-upload.
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                length = int(r.headers.get("Content-Length") or 0)
                if length and not keep_totals:
                    job["total_bytes"] = length
                while True:
                    chunk = r.read(_CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    bump(len(chunk))
            if length and file_done[0] != length:
                raise RuntimeError(
                    f"Download ended at {file_done[0]:,} bytes but the server "
                    f"said {length:,} — connection dropped? Removed; try again")

        shutil.move(str(tmp), str(dest))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _models_dir() -> Path:
    from hermes_cli.local_runtime.bootstrap import models_dir

    return models_dir()


def _engine_too_old(min_engine: str) -> bool:
    """True when the installed llama.cpp predates a model's requirement.
    Tags are release numbers (b10362); no engine installed compares as
    too old only when the model states a requirement."""
    if not min_engine:
        return False
    try:
        from hermes_cli.local_runtime.binaries import default_tag, installed_tags

        tags = installed_tags() or [default_tag()]
        newest = max(int(t.lstrip("b")) for t in tags if t.lstrip("b").isdigit())
        return newest < int(min_engine.lstrip("b"))
    except Exception:  # noqa: BLE001
        return False


def _load_config() -> dict:
    from hermes_cli.config import load_config

    try:
        return load_config()
    except Exception:  # noqa: BLE001
        return {}


def _runtime_section() -> dict:
    return (_load_config() or {}).get("local_runtime") or {}


# ── status: the one call the pane opens with ─────────────────


@router.get("/api/local-models/status")
def local_models_status():
    """Cheap, immediate, never blocks on probes (responsiveness standard):
    config state + installed runtime + staged models + supervisor state.
    GPU facts come from /api/local-models/hardware (slower, polled).

    Sync def on purpose: the body does blocking urlopen/scans, so it runs
    in FastAPI's threadpool instead of stalling the event loop."""
    from hermes_cli.local_runtime.binaries import (
        default_tag,
        installed_tags,
        runtimes_root,
        server_binary,
    )

    section = _runtime_section()
    configured_tag = section.get("tag") or default_tag()
    have = installed_tags()

    # The tag actually serving (boot ladder: configured if installed, else
    # newest installed). Present tense for the pane header.
    tag = configured_tag if configured_tag in have else (have[0] if have else configured_tag)

    # A pending engine update exists when the user runs the local engine
    # (enabled + something installed) and the configured tag — pinned or
    # the Hermes-release default — is newer than anything on disk. The
    # download is a button click, never automatic.
    update_available = bool(
        section.get("enabled") and have and configured_tag not in have)

    runtime_installed = False
    runtime_backend = None
    root = runtimes_root() / tag
    if root.exists():
        for backend_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            try:
                server_binary(backend_dir)
                runtime_installed = True
                runtime_backend = backend_dir.name
                break
            except Exception:  # noqa: BLE001
                continue

    staged = []
    mdir = _models_dir()
    if mdir.exists():
        from hermes_cli.local_runtime.bootstrap import staged_models

        # Split models: report the whole variant's bytes, not one part's.
        from hermes_cli.local_runtime.catalog import find_entry_for_model

        for gguf in staged_models():
            model_id = _model_id_for(gguf)
            size = gguf.stat().st_size
            hit = find_entry_for_model(model_id)
            if hit is not None:
                size = hit[1].size_bytes
            staged.append({
                "id": model_id,
                "size_bytes": size,
                "size_label": _human_gb(size),
            })

    running = _state_endpoint()

    # Which staged models are resident right now (loaded in VRAM). Read
    # from the live router when it's up; {} when down. Feeds the pane's
    # Loaded pills and eject buttons.
    loaded: Dict[str, str] = {}
    placement: Dict[str, Any] = {}
    if running is not None:
        try:
            import urllib.request as _url

            req = _url.Request(
                running["base_url"].rsplit("/v1", 1)[0] + "/models",
                headers={"Authorization": f"Bearer {running.get('api_key', '')}"})
            with _url.urlopen(req, timeout=3) as r:
                data = json.loads(r.read())
            loaded = {
                m["id"]: m.get("status", {}).get("value", "unknown")
                for m in data.get("data", [])
                # Everything resident or becoming resident: 'loading' renders
                # as its own state in the pane (a 20-GB load in flight is the
                # single most important thing the pane can show).
                if m.get("status", {}).get("value") in ("loaded", "ready", "loading")
            }
            # How each loaded model is actually running: the granted window
            # from the child itself, and the plan's spill facts from the
            # preset decision. The pane shows this verbatim — placement is
            # the difference between 'fast' and 'why is my CPU busy', so it
            # must be inspectable, not inferred from Task Manager.
            from hermes_cli.local_runtime.presets import read_preset_decisions

            decisions = read_preset_decisions()
            for model_id in loaded:
                entry_facts: Dict[str, Any] = {}
                plan = decisions.get(model_id)
                if plan is not None:
                    entry_facts["window"] = plan.window
                    entry_facts["window_label"] = f"{plan.window // 1024}K"
                    entry_facts["spilled"] = plan.spilled
                if loaded[model_id] in ("loaded", "ready"):
                    try:
                        preq = _url.Request(
                            running["base_url"].rsplit("/v1", 1)[0]
                            + f"/props?model={model_id}",
                            headers={"Authorization":
                                     f"Bearer {running.get('api_key', '')}"})
                        with _url.urlopen(preq, timeout=3) as pr:
                            props = json.loads(pr.read())
                        n_ctx = (props.get("default_generation_settings", {})
                                 .get("n_ctx"))
                        if n_ctx:
                            entry_facts["granted_window"] = int(n_ctx)
                            entry_facts["granted_window_label"] = f"{int(n_ctx) // 1024}K"
                    except Exception:  # noqa: BLE001
                        pass
                if entry_facts:
                    placement[model_id] = entry_facts
        except Exception as exc:  # noqa: BLE001
            # Never silent: an empty dict here renders as 'Not in memory'
            # on a machine whose VRAM is visibly full.
            logger.warning("loaded-models read failed: %r", exc)
            loaded = {}

    # The active main model, when it is one of ours (config authority: the
    # same model.provider + model.default that /api/model/set writes).
    active_model_id = None
    try:
        config = _load_config()
        model_section = (config or {}).get("model") or {}
        if str(model_section.get("provider", "")).strip().lower() in (
                "llamacpp", "llama.cpp", "llama-cpp"):
            active_model_id = str(
                model_section.get("default") or model_section.get("name") or ""
            ).strip() or None
    except Exception:  # noqa: BLE001
        pass

    return {
        "enabled": bool(section.get("enabled")),
        "tag": tag,
        "configured_tag": configured_tag,
        "update_available": update_available,
        "runtime_installed": runtime_installed,
        "runtime_backend": runtime_backend,
        "server_running": running is not None,
        "server_base_url": (running or {}).get("base_url"),
        "active_model_id": active_model_id,
        "loaded_models": loaded,
        # Live load progress per model (SSE-fed): {model_id: {stage, value,
        # percent}}. The chat's loading bar and the picker rows poll this.
        "loading": _loading_progress(),
        "placement": placement,
        "models": staged,
        "models_dir": str(mdir),
    }


def _loading_progress() -> Dict[str, Any]:
    try:
        from hermes_cli.local_runtime.load_progress import get_loading_progress

        return get_loading_progress()
    except Exception:  # noqa: BLE001 — progress is garnish, never a 500
        return {}


# ── hardware: what this machine can do ───────────────────────


@router.get("/api/local-models/hardware")
def local_models_hardware():
    """The budget as plain facts. Polled by the pane and the statusbar
    resource item (throttled client-side). Sync def on purpose: the body
    shells out to nvidia-smi and probes budgets — threadpool, not loop."""
    from hermes_cli.local_runtime.hardware import probe_budget, _nvidia_vram, _ram_bytes

    budget = probe_budget()
    ram_total, ram_avail = _ram_bytes()
    out = {
        "uma": budget.uma,
        "vram_total_bytes": budget.total_device_bytes,
        "vram_usable_bytes": budget.usable_vram_bytes,
        "ram_total_bytes": ram_total,
        "ram_available_bytes": ram_avail,
        "vram_label": _human_gb(budget.total_device_bytes),
        "gpu_name": None,
        "gpu_util_percent": None,
        "vram_used_bytes": None,
    }
    # GPU identity + live utilization (NVIDIA; other vendors degrade to None
    # and the UI hides those readouts).
    try:
        import subprocess

        from hermes_cli.local_runtime.hardware import _nvidia_smi_path

        smi_exe = _nvidia_smi_path()
        smi = subprocess.run(
            [smi_exe, "--query-gpu=name,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5) if smi_exe else None
        if smi and smi.returncode == 0 and smi.stdout.strip():
            name, util, used_mib = (x.strip() for x in smi.stdout.strip().splitlines()[0].split(","))
            out["gpu_name"] = name
            out["gpu_util_percent"] = int(util)
            out["vram_used_bytes"] = int(used_mib) << 20
    except Exception:  # noqa: BLE001
        pass
    return out


# ── catalog: priced for THIS machine before download ─────────


@router.get("/api/local-models/catalog")
def local_models_catalog():
    """Every entry answers the user's three questions up front: how big is
    the download, will it fit, and what context/speed shape will I get —
    computed from the catalog's measured numbers + this machine's
    budget. Hardware-aware quant selection: the row advertises the BEST
    build for this machine (highest quality that runs fully on the GPU at
    the 64K floor; else the smallest that works, spilled and priced). No
    entry is hidden; unaffordable models show WHY. Sync def on purpose:
    probe_budget + catalog I/O block — threadpool, not loop."""
    from hermes_cli.local_runtime.catalog import (
        CATALOG,
        recommended_entry,
        refresh_catalog_soon,
        select_variant,
    )
    from hermes_cli.local_runtime.context_policy import (
        RUNTIME_OVERHEAD_BYTES,
        initial_window,
        ub_logits_bytes,
    )
    from hermes_cli.local_runtime.estimator import PhysicsRefusal
    from hermes_cli.local_runtime.hardware import probe_budget

    # This request serves the catalog already in memory; a TTL-gated
    # background fetch from the repo lands new entries for the next one
    # (day-0 models reach the pane without an app release).
    refresh_catalog_soon()

    # Planning budget: price against machine capacity, not live-free VRAM.
    # A loaded model must not make the catalog call every row unaffordable.
    budget = probe_budget(planning=True)
    # The default pick for THIS machine: quality-ranked, fit- and
    # speed-gated (recommended_entry). Engine-gated entries can't be
    # activated today, so they can't be the recommendation either. The
    # reason key ships with the row — the Recommended badge's tooltip is
    # the branch that actually fired, not a re-derivation that can drift.
    eligible = tuple(e for e in CATALOG if not _engine_too_old(e.min_engine))
    picked = recommended_entry(budget, eligible)
    recommended = picked[0].id if picked is not None else None
    recommended_reason = picked[1] if picked is not None else None
    # Completeness-checked staging (split parts all present) — the same
    # answer the picker and the router see, so a mid-download model never
    # reads as downloaded here.
    from hermes_cli.local_runtime.bootstrap import staged_model_ids

    staged_ids = set(staged_model_ids())
    entries = []
    for entry in CATALOG:
        choice = select_variant(entry, budget)
        # Any variant of this family already on disk counts as downloaded
        # (split variants stage under their first part).
        downloaded_variant = next(
            (v for v in entry.variants if v.model_id in staged_ids), None)
        row: Dict[str, Any] = {
            "id": entry.id,
            "display_name": entry.display_name,
            "description": entry.description,
            "native_context": entry.n_ctx_train,
            "native_context_label": f"{entry.n_ctx_train // 1024}K",
            "recommended": entry.id == recommended,
            "recommended_reason": recommended_reason if entry.id == recommended else None,
            "downloaded": downloaded_variant is not None,
            "downloaded_model_id": downloaded_variant.model_id if downloaded_variant else None,
            "downloaded_quant": downloaded_variant.quant if downloaded_variant else None,
            "mtp": entry.mtp,
            "vision": entry.mmproj is not None,
            # Day-0 architectures need the llama.cpp release where their
            # support landed. True gates download/activate in the pane
            # until the engine updates; the row still renders (visible +
            # explained beats hidden).
            "needs_engine": _engine_too_old(entry.min_engine),
            "min_engine": entry.min_engine or None,
        }
        if choice is None:
            smallest = min(entry.variants, key=lambda v: v.size_bytes)
            smallest_total = entry.download_bytes(smallest)
            row.update({
                "fits": False,
                "size_bytes": smallest_total,
                "size_label": _human_gb(smallest_total),
                "fit_summary": "Needs more memory than this machine has",
                "fit_detail": (f"even the most compact build ({smallest.quant}, "
                               f"{_human_gb(smallest_total)}) exceeds GPU + system memory"),
            })
            entries.append(row)
            continue

        variant = choice.variant
        profile = entry.profile(variant)
        # Same overhead the launch decision prices (runtime buffers +
        # vision projector + the microbatch/MTP logits buffers): the row
        # must advertise the window the model will actually get, not a
        # paper number the server's own fit then shaves down.
        overhead = (RUNTIME_OVERHEAD_BYTES
                    + (entry.mmproj.size_bytes if entry.mmproj else 0)
                    + ub_logits_bytes(entry.n_vocab, mtp_capable=entry.mtp))
        decision = initial_window(profile, budget, overhead_bytes=overhead)
        download_total = entry.download_bytes(variant)
        row.update({
            "fits": True,
            "model_id": variant.model_id,
            "quant": variant.quant,
            "quant_validated": variant.validated,
            "size_bytes": download_total,
            "size_label": _human_gb(download_total),
            "variant_count": len(entry.variants),
        })
        if choice.reason_key == "best-large-window":
            row["quant_reason"] = (
                f"Recommended build ({variant.quant}) — the quant class this "
                "engine is optimized for; runs fully on your GPU with a "
                "large context window")
        elif choice.reason_key == "best-fits":
            row["quant_reason"] = (
                f"Recommended build ({variant.quant}) — the quant class this "
                "engine is optimized for; runs fully on your GPU")
        else:
            row["quant_reason"] = (
                f"Compact build sized for this machine ({variant.quant}) — "
                "larger than GPU memory, runs slower")
        if not isinstance(decision, PhysicsRefusal):
            row["start_window"] = decision.window
            row["start_window_label"] = f"{decision.window // 1024}K"
            row["spilled"] = decision.spilled
            if decision.window >= entry.n_ctx_train:
                shape = f"runs at its full {row['native_context_label']} context"
            else:
                shape = (f"starts at {row['start_window_label']} and grows toward "
                         f"{row['native_context_label']} as you use it")
            if decision.spilled:
                shape += " (larger than your GPU memory — runs slower)"
            row["fit_summary"] = shape
        else:
            row["fit_summary"] = row["quant_reason"]
        entries.append(row)
    return {"models": entries}


# ── runtime install (job) ────────────────────────────────────


class RuntimeInstallBody(BaseModel):
    backend: Optional[str] = None   # None/auto -> detect


def _runtime_progress_hook(job: Dict[str, Any]):
    """Adapter: ensure_runtime_installed's progress stream -> job fields.

    Throttled to ~4 updates/s. The byte counters are CUMULATIVE across the
    plan: a multi-asset engine (CUDA zip + cudart zip) reads as one growing
    download, not a bar that restarts at zero per asset. The total grows as
    each asset's size becomes known (sizes arrive with the response, not
    the plan). Unpack/verify keep the download's counters in place — the
    stage text says what's happening, and a bar that bounces back to zero
    after the bytes finished reads as a failure."""
    state = {"last": 0.0, "banked": 0, "asset": None, "asset_total": 0}

    def hook(stage: str, done: int, total: int, label: str) -> None:
        now = time.monotonic()
        if now - state["last"] < 0.25 and done < total:
            return
        state["last"] = now
        suffix = f" ({label})" if label else ""
        if stage == "download":
            if label != state["asset"]:
                # Previous asset finished: bank its bytes so the counters
                # keep climbing instead of restarting for the next asset.
                state["banked"] += state["asset_total"]
                state["asset"] = label
            state["asset_total"] = total or done
            plan_done = state["banked"] + done
            plan_total = state["banked"] + (total or 0)
            job["phase"] = "downloading-runtime"
            if total:
                job["detail"] = (f"Downloading the local engine{suffix} — "
                                 f"{_human_gb(plan_done)} of {_human_gb(plan_total)}")
            else:
                job["detail"] = (f"Downloading the local engine{suffix} — "
                                 f"{_human_gb(plan_done)}")
            job["done_bytes"] = plan_done
            job["total_bytes"] = plan_total or None
        elif stage == "extract":
            job["phase"] = "unpacking-runtime"
            pct = f" — {min(100, round(done / total * 100))}%" if total else ""
            job["detail"] = f"Unpacking the engine{suffix}{pct}"
        else:  # verify
            job["phase"] = "verifying-runtime"
            job["detail"] = f"Verifying the engine{suffix}"

    return hook


@router.post("/api/local-models/runtime/install")
async def local_models_runtime_install(body: RuntimeInstallBody):
    from hermes_cli.local_runtime.binaries import (
        default_tag,
        resolve_assets,
        select_backend,
    )
    from hermes_cli.local_runtime.bootstrap import _detect_gpu_vendor

    section = _runtime_section()
    tag = section.get("tag") or default_tag()
    backend = body.backend or section.get("backend", "auto")
    if backend == "auto":
        backend = select_backend(_detect_gpu_vendor())
    # Resolve first so an impossible combination fails the POST, not the job.
    try:
        plan = resolve_assets(tag, backend)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))

    job = _job("runtime-install", f"llama.cpp {tag} ({backend})")

    def _run():
        try:
            from hermes_cli.local_runtime.binaries import (
                ensure_runtime_installed,
                installed_tags,
                prune_old_tags,
            )

            previous = installed_tags()
            job["phase"] = "downloading"
            job["detail"] = f"Fetching {len(plan.assets)} package(s) for {backend}"
            ensure_runtime_installed(tag, backend,
                                     progress=_runtime_progress_hook(job))

            # Engine update path: a server already running on an older tag
            # moves to the new one now — the click was the consent. Fresh
            # installs (no server) skip this; Use/boot handles their start.
            restarted = False
            try:
                from hermes_cli.local_runtime.bootstrap import (
                    ensure_local_runtime,
                    get_supervisor,
                    shutdown_local_runtime,
                )

                sup = get_supervisor()
                if sup is not None and previous and tag not in previous:
                    job["phase"] = "restarting"
                    job["detail"] = "Switching the running server to the new build"
                    shutdown_local_runtime()
                    ensure_local_runtime(_load_config(), force=True)
                    restarted = True
            except Exception as exc:  # noqa: BLE001
                # The new build is installed either way; the next boot serves
                # it. Never fail the job on the restart nicety.
                logger.warning("post-update restart skipped: %s", exc)

            # N-1 retention, only after the new tag verified: keep it and the
            # newest previous build as the rollback pin target.
            try:
                keep = [tag] + [t for t in previous if t != tag][:1]
                prune_old_tags(keep)
            except Exception as exc:  # noqa: BLE001
                logger.warning("runtime prune skipped: %s", exc)

            job["phase"] = "done"
            job["status"] = "done"
            job["detail"] = (f"llama.cpp {tag} ready ({backend})"
                             + (" — server restarted on the new build" if restarted else ""))
        except Exception as exc:  # noqa: BLE001
            logger.warning("runtime install failed: %s", exc)
            job["status"] = "error"
            job["error"] = str(exc)

    threading.Thread(target=_run, daemon=True, name="lr-runtime-install").start()
    return {"job_id": job["job_id"], "backend": backend, "tag": tag}


# ── model download (job with byte progress) ──────────────────


class ModelDownloadBody(BaseModel):
    model_id: str


@router.post("/api/local-models/download")
async def local_models_download(body: ModelDownloadBody):
    """Accepts either a family id (downloads this machine's selected
    variant) or an exact variant model_id."""
    from hermes_cli.local_runtime.catalog import (
        CATALOG,
        catalog_by_id,
        select_variant,
    )
    from hermes_cli.local_runtime.hardware import probe_budget

    entry = catalog_by_id().get(body.model_id)
    variant = None
    if entry is not None:
        if _engine_too_old(entry.min_engine):
            raise HTTPException(
                status_code=409,
                detail=(f"{entry.display_name} needs llama.cpp {entry.min_engine} "
                        f"or newer — update the engine first"))
        # Same planning budget as the catalog — the user downloads exactly
        # the build the row advertised.
        choice = select_variant(entry, probe_budget(planning=True))
        if choice is None:
            raise HTTPException(status_code=409,
                                detail=f"no variant of {entry.id} fits this machine")
        variant = choice.variant
    else:
        for candidate in CATALOG:
            for v in candidate.variants:
                if v.model_id == body.model_id:
                    entry, variant = candidate, v
                    break
            if variant:
                break
    if entry is None or variant is None:
        raise HTTPException(status_code=404, detail=f"unknown model {body.model_id}")

    from hermes_cli.local_runtime.bootstrap import assets_dir, staged_model_ids

    if variant.model_id in staged_model_ids():
        return {"job_id": None, "already_downloaded": True, "model_id": variant.model_id}

    # Everything this variant needs: split parts + mmproj/draft assets.
    plan = []  # (url, dest, bytes)
    for asset in variant.files:
        plan.append((f"https://huggingface.co/{entry.repo}/resolve/main/{asset.path}",
                     _models_dir() / asset.local_name, asset.size_bytes))
    for asset in (entry.mmproj, entry.draft):
        if asset is not None:
            plan.append((f"https://huggingface.co/{entry.repo}/resolve/main/{asset.path}",
                         assets_dir() / asset.local_name, asset.size_bytes))

    total = sum(p[2] for p in plan)
    job = _job("model-download", f"{entry.display_name} ({variant.quant})",
               model_id=entry.id)
    job["total_bytes"] = total

    def _run():
        try:
            job["phase"] = "downloading"
            job["detail"] = f"{entry.display_name} — {_human_gb(total)}"
            done_before = 0
            for url, dest, size in plan:
                if dest.exists():
                    done_before += size
                    job["done_bytes"] = done_before
                    continue
                download_file(url, dest, job,
                              base_done=done_before, keep_totals=True)
                job["phase"] = "downloading"
                done_before += size
                job["done_bytes"] = done_before
            job["phase"] = "done"
            job["status"] = "done"
            job["detail"] = f"{entry.display_name} ready"
            # A running router only scans models at spawn —
            # bounce it so the new model is servable
            # immediately instead of 400ing until the next app restart.
            try:
                from hermes_cli.local_runtime.bootstrap import refresh_local_runtime

                refresh_local_runtime()
            except Exception:  # noqa: BLE001
                logger.debug("post-download runtime refresh skipped", exc_info=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("model download failed: %s", exc)
            job["status"] = "error"
            job["error"] = str(exc)

    threading.Thread(target=_run, daemon=True, name="lr-model-download").start()
    return {"job_id": job["job_id"], "model_id": variant.model_id}


@router.delete("/api/local-models/models/{model_id}")
async def local_models_delete(model_id: str):
    """Remove a staged model: every split part plus its private assets.
    A running router keeps serving from its spawn-time scan, so bounce it
    off the request thread — deleting the active file mid-serve is the
    kind of stale state the refresh exists for."""
    files = _variant_files_on_disk(model_id)
    if not files:
        raise HTTPException(status_code=404, detail="model not found")
    for path in files:
        path.unlink(missing_ok=True)
    # Growth state dies with the model: a re-download starts back at its
    # zero-spill window instead of inheriting a stale grown one.
    try:
        from hermes_cli.local_runtime.growth import clear_window_override

        clear_window_override(model_id)
    except Exception:  # noqa: BLE001
        logger.debug("window-override clear skipped", exc_info=True)

    def _refresh():
        try:
            from hermes_cli.local_runtime.bootstrap import refresh_local_runtime

            refresh_local_runtime()
        except Exception:  # noqa: BLE001
            logger.debug("post-delete runtime refresh skipped", exc_info=True)

    threading.Thread(target=_refresh, daemon=True, name="lr-post-delete").start()
    return {"ok": True}


# ── server lifecycle: turn the engine on/off ─────────────────


class ServerActionBody(BaseModel):
    action: str                 # "stop" | "start"


# ── quickstart: one click from nothing to a working default ──


class QuickstartBody(BaseModel):
    model_id: str | None = None   # default: the catalog's recommended entry


# One quickstart at a time: the job sequences installs, downloads, a
# server bounce, and a config write — two racing runs would interleave
# all four. Held for the job's lifetime, released in the worker.
_QUICKSTART_LOCK = threading.Lock()


@router.post("/api/local-models/quickstart")
async def local_models_quickstart(body: QuickstartBody):
    """The dummy-proof path: one job that installs the runtime (if
    missing), downloads this machine's build of the recommended model
    (if missing), and makes it the default for new chats. Each leg is
    the same code the individual routes run — this route only sequences
    them, so 'Configure' (the existing pane) and quickstart can never
    disagree about what gets installed.

    Preflight rejects (no servable entry, engine too old) fail the POST
    synchronously so the button can explain itself; everything slow runs
    in the job with the usual phase/byte progress.
    """
    from hermes_cli.local_runtime.binaries import (
        default_tag,
        installed_tags,
        resolve_assets,
        select_backend,
    )
    from hermes_cli.local_runtime.bootstrap import (
        _detect_gpu_vendor,
        assets_dir,
        staged_model_ids,
    )
    from hermes_cli.local_runtime.catalog import (
        CATALOG,
        catalog_by_id,
        recommended_entry,
        select_variant,
    )
    from hermes_cli.local_runtime.hardware import probe_budget

    # Resolve the target entry: explicit id, else this machine's
    # recommendation (quality-ranked, fit- and speed-gated), else the
    # first catalog entry this machine can serve.
    budget = probe_budget(planning=True)
    entry = None
    if body.model_id:
        entry = catalog_by_id().get(body.model_id)
        if entry is None:
            raise HTTPException(status_code=404,
                                detail=f"unknown model {body.model_id}")
        candidates = [entry]
    else:
        eligible = tuple(e for e in CATALOG if not _engine_too_old(e.min_engine))
        picked = recommended_entry(budget, eligible)
        best = picked[0] if picked is not None else None
        candidates = ([best] if best is not None else []) + [
            e for e in CATALOG if best is None or e.id != best.id]
    chosen = None
    for candidate in candidates:
        choice = select_variant(candidate, budget)
        if choice is not None and not _engine_too_old(candidate.min_engine):
            chosen = (candidate, choice.variant)
            break
    if chosen is None:
        raise HTTPException(
            status_code=409,
            detail="no catalog model fits this machine — open Local Models "
                   "to browse for a smaller build")
    entry, variant = chosen

    section = _runtime_section()
    tag = section.get("tag") or default_tag()
    backend = section.get("backend", "auto")
    if backend == "auto":
        backend = select_backend(_detect_gpu_vendor())
    need_runtime = not installed_tags()
    if need_runtime:
        # Same preflight as /runtime/install: impossible combos fail the POST.
        try:
            resolve_assets(tag, backend)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc))

    need_download = variant.model_id not in staged_model_ids()
    download_plan = []  # (url, dest, bytes)
    if need_download:
        for asset in variant.files:
            download_plan.append(
                (f"https://huggingface.co/{entry.repo}/resolve/main/{asset.path}",
                 _models_dir() / asset.local_name, asset.size_bytes))
        for asset in (entry.mmproj, entry.draft):
            if asset is not None:
                download_plan.append(
                    (f"https://huggingface.co/{entry.repo}/resolve/main/{asset.path}",
                     assets_dir() / asset.local_name, asset.size_bytes))

    if not _QUICKSTART_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409,
                            detail="Setup is already running")

    job = _job("quickstart", entry.display_name, model_id=entry.id)
    job["total_bytes"] = sum(p[2] for p in download_plan) or None

    def _run():
        try:
            if need_runtime:
                from hermes_cli.local_runtime.binaries import ensure_runtime_installed

                job["phase"] = "installing-runtime"
                job["detail"] = "Installing the local engine"
                ensure_runtime_installed(tag, backend,
                                         progress=_runtime_progress_hook(job))

            if need_download:
                job["phase"] = "downloading"
                total = sum(p[2] for p in download_plan)
                # The runtime leg repurposed the byte counters for its own
                # stages — reset them to the model plan before download.
                job["done_bytes"] = 0
                job["total_bytes"] = total
                job["detail"] = f"{entry.display_name} — {_human_gb(total)}"
                done_before = 0
                for url, dest, size in download_plan:
                    if dest.exists():
                        done_before += size
                        job["done_bytes"] = done_before
                        continue
                    download_file(url, dest, job,
                                  base_done=done_before, keep_totals=True)
                    job["phase"] = "downloading"
                    done_before += size
                    job["done_bytes"] = done_before

            # Activate: same sequence as /activate's job body.
            from hermes_cli.config import load_config, save_config
            from hermes_cli.local_runtime.bootstrap import (
                ensure_local_runtime,
                refresh_local_runtime,
            )

            job["phase"] = "starting-server"
            job["detail"] = "Starting the local server"
            config = load_config()
            config.setdefault("local_runtime", {})["enabled"] = True
            save_config(config)
            sup = ensure_local_runtime(config, force=True)
            if sup is None and _state_endpoint() is None:
                raise RuntimeError(
                    "The local server could not start — open Local Models for details")
            if sup is not None:
                try:
                    if variant.model_id not in sup.models():
                        job["detail"] = "Refreshing the local server"
                        refresh_local_runtime()
                except Exception:  # noqa: BLE001
                    logger.debug("quickstart rescan check skipped", exc_info=True)

            job["phase"] = "setting-default"
            job["detail"] = "Making it your default"
            from hermes_cli.web_deps import late

            late("_apply_model_assignment_sync")(
                "main", "llamacpp", variant.model_id, "", "", "")

            job["phase"] = "done"
            job["status"] = "done"
            job["detail"] = f"{entry.display_name} is ready — new chats use it"
        except Exception as exc:  # noqa: BLE001
            logger.warning("quickstart failed: %s", exc)
            job["status"] = "error"
            job["error"] = str(exc)
        finally:
            _QUICKSTART_LOCK.release()

    threading.Thread(target=_run, daemon=True, name="lr-quickstart").start()
    return {
        "job_id": job["job_id"],
        "model_id": entry.id,
        "display_name": entry.display_name,
        "needs_runtime": need_runtime,
        "needs_download": need_download,
        "download_bytes": sum(p[2] for p in download_plan),
    }


@router.post("/api/local-models/server")
async def local_models_server(body: ServerActionBody):
    """Turn the local engine off (stop the server, free ALL GPU memory,
    and disable auto-start) or back on. The off switch is the whole-engine
    counterpart of per-model eject — and unlike eject it IS durable: the
    user said off, so boots stay off until they say on."""
    import asyncio

    from hermes_cli.config import load_config, save_config

    action = (body.action or "").strip().lower()
    if action not in ("stop", "start"):
        raise HTTPException(status_code=400, detail="action must be 'stop' or 'start'")

    def _stop():
        from hermes_cli.local_runtime.bootstrap import (
            get_supervisor,
            shutdown_local_runtime,
        )

        sup = get_supervisor()
        if sup is not None:
            shutdown_local_runtime()
        else:
            # Server owned by another process (or an orphan): best-effort
            # terminate via the state file's pid, then clear the state.
            endpoint = _state_endpoint()
            if endpoint is not None:
                try:
                    import psutil  # type: ignore

                    from hermes_cli.local_runtime.supervisor import state_path

                    state = json.loads(state_path().read_text(encoding="utf-8"))
                    pid = int(state.get("pid") or 0)
                    if pid > 0 and psutil.pid_exists(pid):
                        psutil.Process(pid).terminate()
                    state_path().unlink(missing_ok=True)
                except Exception:  # noqa: BLE001
                    pass
        config = load_config()
        config.setdefault("local_runtime", {})["enabled"] = False
        save_config(config)

    def _start():
        from hermes_cli.local_runtime.bootstrap import ensure_local_runtime

        config = load_config()
        config.setdefault("local_runtime", {})["enabled"] = True
        save_config(config)
        sup = ensure_local_runtime(config, force=True)
        if sup is None and _state_endpoint() is None:
            raise RuntimeError("The local server could not start — check the "
                               "runtime is installed")

    try:
        await asyncio.to_thread(_stop if action == "stop" else _start)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"ok": True, "action": action}


# ── activate: make a downloaded model THE model ──────────────


class ModelEjectBody(BaseModel):
    model_id: str


@router.post("/api/local-models/eject")
def local_models_eject(body: ModelEjectBody):
    """Free a loaded model's GPU memory now. Nothing reloads it except
    demand — the next message to it (residency v2: no automatic loading
    exists anywhere). Sync def on purpose: the fallback path blocks on a
    urlopen with a 120s timeout — threadpool, never the event loop."""
    from hermes_cli.local_runtime.bootstrap import get_supervisor

    sup = get_supervisor()
    if sup is not None:
        try:
            sup.unload_model(body.model_id)
            return {"ok": True}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    # Server owned by another process (or state-file only): drive the
    # router directly with the persisted endpoint.
    endpoint = _state_endpoint()
    if endpoint is None:
        raise HTTPException(status_code=409, detail="local server is not running")
    try:
        import urllib.request as _url

        req = _url.Request(
            endpoint["base_url"].rsplit("/v1", 1)[0] + "/models/unload",
            data=json.dumps({"model": body.model_id}).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {endpoint.get('api_key', '')}"},
            method="POST")
        with _url.urlopen(req, timeout=120):
            pass
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc


class ModelActivateBody(BaseModel):
    model_id: str               # exact variant id (a staged .gguf stem)


@router.post("/api/local-models/activate")
async def local_models_activate(body: ModelActivateBody):
    """Make a downloaded model the default for new chats. Pure selection
    (residency v2): a config write through the same machinery as
    /api/model/set, plus making sure the server is up. NO model loading —
    models load on first inference, always; an empty router costs nothing.
    Fast enough to be synchronous-feeling, but kept as a job for UI
    continuity."""
    # Split variants stage under their first part — resolve like the rest
    # of the routes instead of assuming a single flat file.
    from hermes_cli.local_runtime.bootstrap import staged_model_ids

    if body.model_id not in staged_model_ids():
        raise HTTPException(status_code=404, detail=f"{body.model_id} is not downloaded")

    job = _job("model-activate", body.model_id, model_id=body.model_id)

    def _run():
        try:
            from hermes_cli.config import load_config, save_config
            from hermes_cli.local_runtime.bootstrap import (
                ensure_local_runtime,
                refresh_local_runtime,
            )

            job["phase"] = "starting-server"
            job["detail"] = "Starting the local server"
            config = load_config()
            sup = ensure_local_runtime(config, force=True)
            if sup is None:
                if _state_endpoint() is None:
                    raise RuntimeError(
                        "The local server could not start — check the runtime is installed")

            # Self-heal a stale router: the model list is spawn-only, so a
            # server started before this model finished downloading can't
            # serve it. If the router doesn't know the model, bounce it.
            if sup is not None:
                try:
                    if body.model_id not in sup.models():
                        job["detail"] = "Refreshing the local server"
                        refresh_local_runtime()
                except Exception:  # noqa: BLE001
                    logger.debug("activate rescan check skipped", exc_info=True)

            job["phase"] = "setting-default"
            job["detail"] = "Making it your default"
            config = load_config()
            config.setdefault("local_runtime", {})["enabled"] = True
            save_config(config)
            from hermes_cli.web_deps import late

            late("_apply_model_assignment_sync")(
                "main", "llamacpp", body.model_id, "", "", "")

            job["phase"] = "done"
            job["status"] = "done"
            job["detail"] = f"{body.model_id} is the default for new chats"
        except Exception as exc:  # noqa: BLE001
            logger.warning("model activate failed: %s", exc)
            job["status"] = "error"
            job["error"] = str(exc)

    threading.Thread(target=_run, daemon=True, name="lr-model-activate").start()
    return {"job_id": job["job_id"]}


# ── job polling ──────────────────────────────────────────────


@router.get("/api/local-models/jobs")
async def local_models_jobs():
    """All recent jobs, running first — the pane and the app-level poller
    rediscover in-flight work here after a remount or app restart."""
    with _JOBS_LOCK:
        jobs = sorted(_JOBS.values(),
                      key=lambda j: (j["status"] != "running", -j["started_at"]))
    out = []
    for job in jobs[:20]:
        entry = dict(job)
        if entry["total_bytes"]:
            entry["percent"] = min(100, round(entry["done_bytes"] / entry["total_bytes"] * 100))
        out.append(entry)
    return {"jobs": out}


@router.get("/api/local-models/jobs/{job_id}")
async def local_models_job(job_id: str):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    out = dict(job)
    if out["total_bytes"]:
        out["percent"] = min(100, round(out["done_bytes"] / out["total_bytes"] * 100))
    return out


# ── Hugging Face browser: search, repo files, arbitrary download ─


@router.get("/api/local-models/search")
async def local_models_search(q: str, limit: int = 20):
    """Full-text HF search over GGUF models — the firehose behind the
    curated catalog. The pane's per-quant fit pills come from the
    repo-files call once the user opens a hit."""
    from starlette.concurrency import run_in_threadpool

    from hermes_cli.local_runtime.hf_browse import search_models

    if not q.strip():
        return {"hits": []}
    try:
        hits = await run_in_threadpool(search_models, q, limit)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502,
                            detail=f"Hugging Face search unavailable: {exc}") from exc
    return {"hits": [h.__dict__ for h in hits]}


@router.get("/api/local-models/search/files")
async def local_models_search_files(repo: str):
    """The servable GGUFs in one HF repo with a rough pre-download fit
    verdict per quant (file size + conservative fill-ins — the GGUF
    header refines it after download)."""
    from starlette.concurrency import run_in_threadpool

    from hermes_cli.local_runtime.hardware import probe_budget
    from hermes_cli.local_runtime.hf_browse import priced_repo_files

    try:
        groups = await run_in_threadpool(
            priced_repo_files, repo, probe_budget(planning=True))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502,
                            detail=f"Could not list {repo}: {exc}") from exc
    return {"files": [dict(g.__dict__, paths=list(g.paths)) for g in groups]}


class BrowsedDownloadBody(BaseModel):
    repo: str
    paths: list[str]            # one GGUF, or every part of a split, in order


@router.post("/api/local-models/download-browsed")
async def local_models_download_browsed(body: BrowsedDownloadBody):
    """Download an arbitrary HF GGUF (browsed or pasted) into the managed
    models dir. From the moment it lands it is a normal staged model: the
    post-download bounce regenerates presets from its real header and the
    fit policy owns its launch. No catalog entry — it serves 'unverified',
    capabilities answered from the live server only."""
    import re as _re

    from hermes_cli.local_runtime.bootstrap import staged_model_ids

    paths = [p for p in (body.paths or []) if p.lower().endswith(".gguf")]
    if not paths:
        raise HTTPException(status_code=422, detail="no .gguf files given")
    first = paths[0].rsplit("/", 1)[-1]
    model_id = _re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", first, flags=_re.IGNORECASE)
    model_id = model_id[:-5] if model_id.lower().endswith(".gguf") else model_id
    if model_id in staged_model_ids():
        return {"job_id": None, "already_downloaded": True, "model_id": model_id}

    job = _job("model-download", f"{model_id} (from {body.repo})",
               model_id=model_id)

    def _run():
        try:
            job["phase"] = "downloading"
            for p in paths:
                url = (f"https://huggingface.co/{body.repo}"
                       f"/resolve/main/{urllib.parse.quote(p)}")
                dest = _models_dir() / p.rsplit("/", 1)[-1]
                if dest.exists():
                    continue
                download_file(url, dest, job,
                              base_done=int(job.get("done_bytes") or 0),
                              keep_totals=bool(job.get("total_bytes")))
                job["phase"] = "downloading"
            job["phase"] = "done"
            job["status"] = "done"
            job["detail"] = f"{model_id} ready"
            try:
                from hermes_cli.local_runtime.bootstrap import refresh_local_runtime

                refresh_local_runtime()
            except Exception:  # noqa: BLE001
                logger.debug("post-download runtime refresh skipped", exc_info=True)
        except Exception as exc:  # noqa: BLE001
            job["status"] = "error"
            job["error"] = str(exc)

    threading.Thread(target=_run, daemon=True, name="lm-download-browsed").start()
    return {"job_id": job["job_id"], "model_id": model_id}


class SideloadBody(BaseModel):
    path: str                   # absolute path to a .gguf on this machine


@router.post("/api/local-models/sideload")
async def local_models_sideload(body: SideloadBody):
    """Register a GGUF that already exists on this machine: link it into
    the managed models dir (copy only when linking is impossible) and
    bounce the router so it serves immediately. The original stays where
    it is; delete-from-Hermes removes only our link."""
    import os
    import shutil

    from starlette.concurrency import run_in_threadpool

    src = Path(body.path)
    if not src.is_file() or src.suffix.lower() != ".gguf":
        raise HTTPException(status_code=422, detail="Pick a .gguf model file")
    dest = _models_dir() / src.name
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
    try:
        from hermes_cli.local_runtime.bootstrap import refresh_local_runtime

        refresh_local_runtime()
    except Exception:  # noqa: BLE001
        logger.debug("post-sideload runtime refresh skipped", exc_info=True)
    return {"ok": True, "model_id": dest.stem}
