"""Local-models dashboard routes — the desktop's window into the managed llama.cpp runtime.

Every payload carries plain-language, pre-formatted facts the UI shows verbatim
(what will this model do ON THIS MACHINE, how big is the download, what is the
runtime doing), never raw internals. Long jobs follow the repo's job pattern:
start-POST -> {job_id} -> GET poll with byte progress.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
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
from hermes_cli.local_runtime.endpoint import _state_endpoint

logger = logging.getLogger(__name__)

router = APIRouter()

_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
# A browser click can arrive twice before the desktop's job poll disables its button. Keep one
# worker per model identity, and serialize same-process destination work so quickstart/catalog
# jobs do not waste a second transfer. Every transfer also gets its own temporary file and
# atomically claims the final name, covering independent Hermes processes that share a models dir.
_MODEL_DOWNLOADS_LOCK = threading.Lock()
_INFLIGHT_MODEL_DOWNLOADS: Dict[str, str] = {}
_DOWNLOAD_DESTINATION_LOCKS: Dict[str, threading.Lock] = {}
_DOWNLOAD_DESTINATION_LOCKS_LOCK = threading.Lock()
# One quickstart at a time: the job sequences installs, downloads, a server bounce and a config write — two
# racing runs would interleave all four. Held for the job's lifetime, released in the worker.
_QUICKSTART_LOCK = threading.Lock()
# Applying an advanced plan regenerates the shared router preset file and can
# restart the one managed server. Reject a second click while that transaction
# is in flight rather than letting two config snapshots race each other.
_ADVANCED_APPLY_LOCK = threading.Lock()
_LLAMACPP_PROVIDERS = ("llamacpp", "llama.cpp", "llama-cpp")
_SPLIT_PART_RE = r"-\d{5}-of-\d{5}"
_SPLIT_GGUF_RE = re.compile(
    r"^(?P<stem>.+)-(?P<part>\d{5})-of-(?P<total>\d{5})\.gguf$", re.IGNORECASE)
_WINDOWS_DRIVE_PART_RE = re.compile(r"^[A-Za-z]:")
_HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_GGUF_SUFFIX = ".gguf"
_ENGINE_TAG_RE = re.compile(r"^b[0-9]+$")
# One TCP stream to a CDN rarely fills a fast line; 8 ranged connections into a preallocated file saturate gigabit.
_DOWNLOAD_CONNECTIONS = 8
_CHUNK = 4 << 20
_SERVER_START_FAILED = "The local server could not start — check the runtime is installed"
# Header inspection is fast, but the pane polls status every few seconds and a multi-shard model's
# metadata can still be sizeable. Cache only immutable (path, size, mtime, active engine) snapshots.
_LOOKUP_INSPECTION_CACHE: Dict[tuple, Dict[str, Any]] = {}


class RuntimeInstallBody(BaseModel):
    backend: Optional[str] = None   # None/auto -> detect
    # A model card may request the minimum compatible build explicitly. This
    # is still a user-clicked install; Hermes never upgrades an engine merely
    # because a model exists on disk.
    tag: Optional[str] = None


class ModelDownloadBody(BaseModel):
    model_id: str


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


class AdvancedPlanBody(BaseModel):
    """Typed launch preferences for one staged model; never accepts raw llama.cpp flags."""
    model_id: str
    context_tokens: int | None = None
    slots: int = 1
    kv_cache: str = "q8_0"
    speculation: str = "auto"
    mtp_draft_depth: int | None = None

    def request_mapping(self) -> dict[str, object]:
        return {
            "context_tokens": self.context_tokens, "slots": self.slots,
            "kv_cache": self.kv_cache, "speculation": self.speculation,
            "mtp_draft_depth": self.mtp_draft_depth,
        }


class GatewayPublishBody(BaseModel):
    alias: str
    model_id: str
    mode: str = "agent"  # agent | raw


def _human_gb(n: int | float) -> str:
    return f"{n / (1 << 30):.1f} GB"


def _k_label(tokens: int) -> str:
    return f"{tokens // 1024}K"


@contextlib.contextmanager
def _http_error(status: int, prefix: str = ""):
    """Map any exception to ``HTTPException(status, f"{prefix}{exc}")``."""
    try:
        yield
    except HTTPException:
        raise
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
        "status": "running",        # running | done | error
        "phase": "starting",        # human-readable step name
        "detail": "", "total_bytes": None, "done_bytes": 0, "started_at": time.time(), "error": None,
    }
    with _JOBS_LOCK:
        _JOBS[job["job_id"]] = job
    return job


def _claim_model_download(download_key: str, target: str, model_id: str) -> "tuple[Dict[str, Any], bool]":
    """Return one active model-download job for a model identity, creating it exactly once."""
    with _MODEL_DOWNLOADS_LOCK:
        previous_id = _INFLIGHT_MODEL_DOWNLOADS.get(download_key)
        if previous_id:
            with _JOBS_LOCK:
                previous = _JOBS.get(previous_id)
            if previous is not None and previous.get("status") == "running":
                return previous, False
            _INFLIGHT_MODEL_DOWNLOADS.pop(download_key, None)
        job = _job("model-download", target, model_id=model_id)
        _INFLIGHT_MODEL_DOWNLOADS[download_key] = job["job_id"]
        return job, True


def _release_model_download(download_key: str, job_id: str) -> None:
    with _MODEL_DOWNLOADS_LOCK:
        if _INFLIGHT_MODEL_DOWNLOADS.get(download_key) == job_id:
            _INFLIGHT_MODEL_DOWNLOADS.pop(download_key, None)


def _job_view(job: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(job)
    if out["total_bytes"]:
        out["percent"] = min(100, round(out["done_bytes"] / out["total_bytes"] * 100))
    return out


def _step(job: Dict[str, Any], phase: str, detail: str) -> None:
    job["phase"] = phase
    job["detail"] = detail


def _finish(job: Dict[str, Any], detail: str) -> None:
    _step(job, "done", detail)
    job["status"] = "done"


def _spawn_job(job: Dict[str, Any], name: str, body: Callable[[], None], *, fail_msg: str | None = None,
               on_exit: Callable[[], None] | None = None, download_label: str | None = None,
               download_detail: str | None = None) -> None:
    """Run ``body`` on a daemon thread; an exception marks the job errored (warning ``fail_msg`` when
    given); ``on_exit`` always runs last. ``download_label`` identifies a finished download and
    bounces the router to pick the file up. ``download_detail`` can deliberately say ``downloaded``
    when readiness still depends on a model-specific engine requirement."""
    def _run():
        try:
            body()
            if download_label is not None:
                _finish(job, download_detail or f"{download_label} ready")
                _refresh_runtime("post-download runtime refresh skipped")
        except Exception as exc:  # noqa: BLE001
            if fail_msg:
                logger.warning(fail_msg, exc)
            job["status"] = "error"
            job["error"] = str(exc)
        finally:
            if on_exit is not None:
                on_exit()

    threading.Thread(target=_run, daemon=True, name=name).start()


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


def _runtime_target(requested_backend: str | None = None, requested_tag: str | None = None) -> "tuple[str, str]":
    """(tag, backend) for an explicit runtime install.

    ``requested_tag`` comes only from a compatible-model card. It must be a
    normal llama.cpp release tag; resolving its platform asset remains the
    second preflight before a job is created.
    """
    section = _runtime_section()
    tag = str(requested_tag or section.get("tag") or binaries.default_tag())
    if not _ENGINE_TAG_RE.fullmatch(tag):
        raise HTTPException(status_code=422, detail="engine tag must look like b10679")
    backend = requested_backend or section.get("backend", "auto")
    if backend == "auto":
        backend = binaries.select_backend(bootstrap._detect_gpu_vendor())
    return tag, backend


def _serving_engine_tag() -> str | None:
    """The engine actually serving, or the next boot's engine when none is running.

    A legacy state file has no engine identity. Returning ``None`` there is
    deliberate: claiming the configured/current install would falsely promise
    mmap-backed PLE placement for an old adopted server.
    """
    running = _state_endpoint()
    if running is not None:
        tag = running.get("engine_tag")
        return str(tag) if isinstance(tag, str) and tag else None
    # ``_state_endpoint`` is built from the validated private state reader.  Atomic state writes
    # mean a dead/corrupt leftover is not evidence of a running old router, so use the next boot's
    # installed tag and let bootstrap heal it.  A live legacy state does return above, but without
    # ``engine_tag`` and therefore remains deliberately unknown.
    return binaries.active_tag(_runtime_section())


def _persist_runtime_tag(tag: str) -> None:
    """Make a user-selected compatible build the next boot target after it verifies."""
    config = config_mod.load_config()
    section = config.setdefault("local_runtime", {})
    if section.get("tag") != tag:
        section["tag"] = tag
        config_mod.save_config(config)


def _resolve_assets_or_400(tag: str, backend: str):
    """Resolve first so an impossible combination fails the POST, not the job."""
    with _http_error(400):
        return binaries.resolve_assets(tag, backend)


def _engine_too_old(min_engine: str) -> bool:
    """True when the boot engine cannot be proven to meet a model's release requirement.

    ``active_tag`` deliberately falls back to the newest installed build while an update is pending.
    Its value can still come from a hand-edited config or a stale runtime directory, so malformed and
    unreadable tags fail closed rather than accidentally approving a PLE-capable catalog card.
    """
    return not catalog.engine_meets_minimum(
        _quiet(lambda: binaries.active_tag(_runtime_section()), None), min_engine)


def _eligible_entries():
    """Catalog entries this engine can activate today (engine-gated ones can't be the recommendation either)."""
    return tuple(e for e in catalog.CATALOG if not _engine_too_old(e.min_engine))


def _entry_or_404(model_id: str):
    entry = catalog.catalog_by_id().get(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"unknown model {model_id}")
    return entry


def _advanced_plan(model_id: str, value: object):
    """Resolve a staged model and calculate a side-effect-free launch plan."""
    from hermes_cli.local_runtime.advanced import LaunchRequest, plan_launch
    from hermes_cli.local_runtime.estimator import profile_from_gguf
    from hermes_cli.local_runtime.gguf import read_gguf_header

    gguf = _staged_gguf(model_id)
    if gguf is None:
        raise HTTPException(status_code=404, detail=f"{model_id} is not downloaded")
    _require_model_engine(model_id, gguf)
    with _http_error(422, "unable to inspect model: "):
        profile = profile_from_gguf(read_gguf_header(gguf), engine_tag=_serving_engine_tag())
    entry = catalog.entry_for_model(model_id)
    mtp_supported = bool(entry and entry.mtp)
    default_depth = entry.mtp_draft_depth if entry is not None else 3
    default = context_policy.initial_window(profile, hardware.probe_budget(planning=True))
    if isinstance(default, estimator.PhysicsRefusal):
        raise HTTPException(status_code=422, detail=default.message)
    request = LaunchRequest.from_mapping(value)
    return plan_launch(profile, hardware.probe_budget(planning=True), request,
                       default_context_tokens=default.window, mtp_supported=mtp_supported,
                       default_mtp_depth=default_depth)


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
    """Make ``model_id`` the main model via the same machinery as /api/model/set (late-bound: tests stub web_deps.late)."""
    _step(job, "setting-default", "Making it your default")
    web_deps.late("_apply_model_assignment_sync")("main", "llamacpp", model_id, "", "", "")


# ── downloads: ranged parallel streams ───────────────────────
def _hf_url(repo: str, path: str) -> str:
    # Quote each external component here rather than asking individual callers to remember it.
    # Preserve slash only because both values have been validated as repository-relative paths.
    return (f"https://huggingface.co/{urllib.parse.quote(repo, safe='/')}/resolve/main/"
            f"{urllib.parse.quote(path, safe='/')}")


def _safe_repo_relative_gguf_path(path: str) -> bool:
    """Whether ``path`` is a literal, safe Hugging Face repository-relative GGUF path.

    ``Path`` on the current host does not recognize a foreign Windows drive path, so inspect
    POSIX components ourselves.  The result is used only as a remote URL path and is never joined
    directly to local storage, but rejecting escape-shaped values avoids both ambiguous requests
    and future unsafe refactors.
    """
    if not isinstance(path, str) or not path or not path.casefold().endswith(_GGUF_SUFFIX):
        return False
    if path.startswith(("/", "\\")) or "\\" in path:
        return False
    parts = path.split("/")
    return all(part not in ("", ".", "..") and not _WINDOWS_DRIVE_PART_RE.match(part)
               for part in parts)


def _safe_hf_repo(repo: str) -> bool:
    """Hugging Face model repo IDs are exactly ``owner/name``; do not let a POST reshape a URL."""
    return isinstance(repo, str) and bool(_HF_REPO_RE.fullmatch(repo))


def _complete_browsed_paths(raw_paths: list[str]) -> list[str]:
    """Validate one GGUF or every ordered shard of one split before any transfer begins."""
    paths = list(raw_paths or [])
    if not paths or any(not isinstance(path, str) or not path.casefold().endswith(_GGUF_SUFFIX)
                        for path in paths):
        raise HTTPException(status_code=422, detail="Pick one GGUF quant, including every split shard")
    if any(not _safe_repo_relative_gguf_path(path) for path in paths):
        raise HTTPException(status_code=422, detail="GGUF paths must be safe repository-relative paths")
    if len(set(paths)) != len(paths):
        raise HTTPException(status_code=422, detail="A GGUF split cannot contain duplicate shards")

    matches = [_SPLIT_GGUF_RE.fullmatch(path) for path in paths]
    if not any(matches):
        if len(paths) != 1:
            raise HTTPException(status_code=422, detail="Choose one GGUF quant at a time")
        return paths
    if any(match is None for match in matches):
        raise HTTPException(status_code=422, detail="A split GGUF download must include every shard")

    split_matches = [match for match in matches if match is not None]
    stems = {match.group("stem") for match in split_matches}
    totals = {int(match.group("total")) for match in split_matches}
    expected = next(iter(totals), 0)
    indexes = {int(match.group("part")) for match in split_matches}
    if (len(stems) != 1 or len(totals) != 1 or len(paths) != expected
            or indexes != set(range(1, expected + 1))):
        raise HTTPException(status_code=422, detail=(
            f"Select all {expected} shards of this split GGUF before downloading"))
    return [path for _, path in sorted((int(match.group("part")), path)
                                       for match, path in zip(split_matches, paths))]


def _browsed_group_stem(path: str) -> str:
    """Repository-relative source stem, shared by every part of a split GGUF."""
    match = _SPLIT_GGUF_RE.fullmatch(path)
    return match.group("stem") if match is not None else path[:-len(_GGUF_SUFFIX)]


def _safe_browsed_stem(path: str) -> str:
    """Readable, Windows-safe local filename stem derived from a repository-relative source path."""
    source_name = _browsed_group_stem(path).rsplit("/", 1)[-1]
    # Keep the useful quant/model words, while the digest below preserves the exact source identity.
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", source_name).strip(" ._-")
    return (stem or "model")[:96]


def _browsed_local_name(repo: str, path: str) -> str:
    """Collision-proof canonical local filename for a browsed GGUF.

    The managed model directory is intentionally flat for llama.cpp.  Never flatten an upstream
    ``Q4/model.gguf`` into just ``model.gguf``: two directory-contained quants (or two repos)
    would silently become one model.  A deterministic digest carries the full source identity;
    the lower-case extension is required because the staged-model scanner is case-sensitive on
    Linux while Windows treats extensions case-insensitively.
    """
    source_stem = _browsed_group_stem(path)
    digest = hashlib.sha256(f"{repo}\0{source_stem}".encode("utf-8")).hexdigest()[:12]
    match = _SPLIT_GGUF_RE.fullmatch(path)
    if match is not None:
        return (f"{_safe_browsed_stem(path)}--{digest}-{int(match.group('part')):05d}"
                f"-of-{int(match.group('total')):05d}{_GGUF_SUFFIX}")
    return f"{_safe_browsed_stem(path)}--{digest}{_GGUF_SUFFIX}"


def _browsed_model_id(repo: str, path: str) -> str:
    """Managed model id corresponding to a validated browsed GGUF path."""
    return _model_id_for(Path(_browsed_local_name(repo, path)))


def _model_id_for(gguf: Path) -> str:
    """Variant model id for a staged file (strips split-part suffixes)."""
    return re.sub(_SPLIT_PART_RE + "$", "", gguf.stem)


def _inspection_parts(gguf: Path) -> tuple[Path, ...]:
    """Expected shards for a staged GGUF; mirrors the header reader without opening the files."""
    match = _SPLIT_GGUF_RE.fullmatch(gguf.name)
    if match is None:
        return (gguf,)
    stem = match.group("stem")
    total = int(match.group("total"))
    return tuple(gguf.with_name(f"{stem}-{index:05d}-of-{total:05d}.gguf")
                 for index in range(1, total + 1))


def _lookup_inspection(gguf: Path) -> Dict[str, Any]:
    """Placement truth for a completed staged GGUF, independent of whether it is loaded.

    The raw header records the table even when a whole-model quant has taken it below llama.cpp's
    strict automatic-lazy threshold.  Only an eligible table on the *active* engine is disk-backed;
    this function never turns it into an ordinary spill or emits launch flags.
    """
    from hermes_cli.local_runtime.gguf import (
        LAZY_LOOKUP_MIN_ENGINE_BUILD, read_gguf_header, supports_automatic_lazy_lookup)

    tag = _serving_engine_tag()
    try:
        signature = tuple((str(part), part.stat().st_size, part.stat().st_mtime_ns)
                          for part in _inspection_parts(gguf))
    except OSError:
        return {"lookup_placement": "unknown"}
    key = (tag, signature)
    cached = _LOOKUP_INSPECTION_CACHE.get(key)
    if cached is not None:
        return dict(cached)

    try:
        header = read_gguf_header(gguf)
    except Exception as exc:  # noqa: BLE001 - malformed third-party GGUFs must not break status
        logger.debug("lookup inspection skipped for %s: %s", gguf.name, exc)
        facts: Dict[str, Any] = {"lookup_placement": "unknown"}
    else:
        lookup_bytes = int(header.lookup_table_bytes)
        if not lookup_bytes:
            facts = {"lookup_placement": "none"}
        else:
            facts = {
                "lookup_table_bytes": lookup_bytes,
                "lookup_table_label": _human_gb(lookup_bytes),
            }
            if header.lazy_table_bytes:
                if supports_automatic_lazy_lookup(tag):
                    facts.update(
                        lookup_placement="disk-backed",
                        disk_backed_lookup_bytes=header.lazy_table_bytes,
                        disk_backed_lookup_label=_human_gb(header.lazy_table_bytes),
                    )
                else:
                    facts.update(
                        lookup_placement="requires-engine-update",
                        required_engine=f"b{LAZY_LOOKUP_MIN_ENGINE_BUILD}",
                    )
            else:
                # The file has a PLE/Engram lookup table, but its whole-model quant made the
                # table <= 4 GiB. llama.cpp deliberately treats that as ordinary resident memory.
                facts["lookup_placement"] = "resident"

    if len(_LOOKUP_INSPECTION_CACHE) >= 256:
        _LOOKUP_INSPECTION_CACHE.clear()
    _LOOKUP_INSPECTION_CACHE[key] = dict(facts)
    return facts


def _staged_gguf(model_id: str) -> Path | None:
    return next((path for path in bootstrap.staged_models() if _model_id_for(path) == model_id), None)


def _require_catalog_engine(model_id: str) -> None:
    """Keep known catalog architecture floors enforced on every API surface.

    Sideloaded/browser models are intentionally absent from the catalog and remain governed only by
    their parsed lookup placement. A catalog model's ``min_engine`` is broader than PLE support,
    so a successful header inspection never waives it.
    """
    entry = catalog.entry_for_model(model_id)
    if entry is not None and not catalog.engine_meets_minimum(_serving_engine_tag(), entry.min_engine):
        raise HTTPException(status_code=409, detail=(
            f"{entry.display_name} needs llama.cpp {entry.min_engine} or newer — update the engine first"))


def _require_lookup_engine(model_id: str, gguf: Path | None = None) -> None:
    """Require a verified lookup placement before a model can be activated or routed."""
    gguf = gguf or _staged_gguf(model_id)
    if gguf is None:
        return
    inspection = _lookup_inspection(gguf)
    if inspection.get("lookup_placement") == "unknown":
        raise HTTPException(status_code=422, detail=(
            f"{model_id} could not be inspected as a valid GGUF — verify the file before using it"))
    required = inspection.get("required_engine")
    if inspection.get("lookup_placement") == "requires-engine-update" and required:
        raise HTTPException(status_code=409, detail=(
            f"{model_id} needs llama.cpp {required} or newer to keep its large lookup table "
            "disk-backed — update the engine first"))


def _require_model_engine(model_id: str, gguf: Path | None = None) -> None:
    """Apply catalog architecture and parsed lookup gates together for a staged model."""
    _require_catalog_engine(model_id)
    _require_lookup_engine(model_id, gguf)


def _variant_files_on_disk(model_id: str) -> "list[Path]":
    """Every local file of a staged model: all split parts plus catalog-declared assets (mmproj/draft) when present."""
    files = [p for p in bootstrap.models_dir().glob("*.gguf") if _model_id_for(p) == model_id]
    hit = catalog.find_entry_for_model(model_id)
    assets = (hit[0].mmproj, hit[0].draft) if hit is not None else ()
    files += [bootstrap.assets_dir() / a.local_name for a in assets
              if a is not None and (bootstrap.assets_dir() / a.local_name).exists()]
    return files


def _probe_range_support(url: str) -> int:
    """Total size when the server honors Range requests, else 0. 401/403 = gated repo or wrong catalog
    repo — raise a plain-language message, not a bare status."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            content_range = r.headers.get("Content-Range", "") if r.status == 206 else ""
            if "/" in content_range:
                return int(content_range.rsplit("/", 1)[1])
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise RuntimeError("The model host refused the download (gated or moved). "
                               "This is a catalog problem, not yours — please report it.") from exc
        raise
    except Exception:  # noqa: BLE001
        pass
    return 0


def _download_destination_lock(dest: Path) -> threading.Lock:
    """One process-local lock per final model file to avoid redundant same-process transfers."""
    key = str(dest.absolute())
    with _DOWNLOAD_DESTINATION_LOCKS_LOCK:
        lock = _DOWNLOAD_DESTINATION_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _DOWNLOAD_DESTINATION_LOCKS[key] = lock
        return lock


def download_file(url: str, dest: Path, job: Dict[str, Any], *, base_done: int = 0, keep_totals: bool = False) -> None:
    """Download one final path safely even when independent jobs chose the same model."""
    with _download_destination_lock(dest):
        # The caller may have checked before it waited for this lock.  A previous job can finish
        # in that gap; never reopen its final GGUF or its shared ``.part`` path.
        if dest.exists():
            return
        _download_file_unlocked(url, dest, job, base_done=base_done, keep_totals=keep_totals)


def _publish_download(tmp: Path, dest: Path) -> bool:
    """Atomically expose ``tmp`` as ``dest`` without replacing a competing completed download."""
    try:
        # link(2) creates the final path only if it did not already exist. Unlike replace/rename,
        # it cannot overwrite a good model another Hermes process completed while this one was
        # downloading. Both paths are in the managed models directory, so this is always one
        # filesystem.
        os.link(tmp, dest)
    except FileExistsError:
        return False
    finally:
        # Drop our temporary link whether we won or another process got there first.
        tmp.unlink(missing_ok=True)
    return True


def _download_file_unlocked(url: str, dest: Path, job: Dict[str, Any], *, base_done: int = 0,
                            keep_totals: bool = False) -> None:
    """Download url -> dest with byte progress on ``job``; ranged-parallel when the server supports it,
    single-stream otherwise. Each worker uses a private temporary path, then atomically claims the final
    filename without replacing another process's completed file. Never leaves a .part. Completeness is
    checked only against what the SERVER declared (range-probe total / Content-Length), never the CATALOG
    (its sizes may lag a re-upload), so a dropped connection still errors instead of staging a truncated
    file. Multi-file variants: ``base_done`` offsets progress onto earlier files; ``keep_totals=True`` keeps
    the per-file size from overwriting the variant's total."""
    tmp = dest.with_name(f"{dest.name}.{uuid.uuid4().hex}.part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    file_done = [0]
    progress_lock = threading.Lock()
    errors: list[Exception] = []

    def pump(r, f) -> None:
        for chunk in iter(lambda: r.read(_CHUNK), b""):
            f.write(chunk)
            with progress_lock:
                file_done[0] += len(chunk)
                job["done_bytes"] = base_done + file_done[0]

    def fetch_range(start: int, end: int) -> None:
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "r+b") as f:
                f.seek(start)
                pump(r, f)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    try:
        # Probe and preallocation take real seconds on a 20+ GB file — narrate them, or the pane shows a dead '— of X GB'.
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
            n = _DOWNLOAD_CONNECTIONS
            threads = [threading.Thread(target=fetch_range, daemon=True, name=f"lm-dl-{i}",
                                        args=(i * total // n, (i + 1) * total // n - 1)) for i in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            if errors:
                raise errors[0]
            if file_done[0] != total:
                raise RuntimeError(f"download incomplete ({file_done[0]} of {total} bytes)")
        else:
            # No range support: single stream; completeness judged by the server's
            # own Content-Length when it sent one — never the catalog.
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                length = int(r.headers.get("Content-Length") or 0)
                if length and not keep_totals:
                    job["total_bytes"] = length
                pump(r, f)
            if length and file_done[0] != length:
                raise RuntimeError(f"Download ended at {file_done[0]:,} bytes but the server "
                                   f"said {length:,} — connection dropped? Removed; try again")
        _publish_download(tmp, dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _download_plan(entry, variant) -> list:
    """Everything a variant needs: split parts + mmproj/draft assets, as (url, dest, bytes) tuples."""
    plan = [(_hf_url(entry.repo, a.path), bootstrap.models_dir() / a.local_name, a.size_bytes) for a in variant.files]
    plan += [(_hf_url(entry.repo, a.path), bootstrap.assets_dir() / a.local_name, a.size_bytes)
             for a in (entry.mmproj, entry.draft) if a is not None]
    return plan


def _run_download_plan(job: Dict[str, Any], plan: list, label: str) -> None:
    """Download every missing file in ``plan``; already-present files count toward progress without a transfer."""
    _step(job, "downloading", f"{label} — {_human_gb(sum(p[2] for p in plan))}")
    done_before = 0
    for url, dest, size in plan:
        if not dest.exists():
            download_file(url, dest, job, base_done=done_before, keep_totals=True)
            job["phase"] = "downloading"
        done_before += size
        job["done_bytes"] = done_before


# ── status: the one call the pane opens with ─────────────────
def _loaded_models(running: Dict[str, Any]) -> "tuple[Dict[str, str], Dict[str, Any]]":
    """Resident models right now, plus how each is placed (granted window from the child, spill facts from
    the preset decision) — the difference between 'fast' and 'why is my CPU busy', so it must be inspectable.
    'loading' is its own state (a 20-GB load in flight is the most important thing the pane can show)."""
    from hermes_cli.local_runtime.gguf import supports_automatic_lazy_lookup

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
            # Presets describe the physics selected at generation time, but
            # the state file proves the build that is actually serving this
            # live model. Do not report a disk-backed table for a legacy or
            # unknown engine identity.
            if plan.lazy_table_bytes and supports_automatic_lazy_lookup(running.get("engine_tag")):
                facts.update(
                    disk_backed_lookup_bytes=plan.lazy_table_bytes,
                    disk_backed_lookup_label=_human_gb(plan.lazy_table_bytes),
                )
        n_ctx = state in ("loaded", "ready") and _quiet(
            lambda: _router_request(running, f"/props?model={model_id}", timeout=3)
            .get("default_generation_settings", {}).get("n_ctx"), None)
        if n_ctx:
            facts.update(granted_window=int(n_ctx), granted_window_label=_k_label(int(n_ctx)))
        if facts:
            placement[model_id] = facts
    return loaded, placement


def _installed_backend(tag: str) -> str | None:
    """Name of the first backend dir under ``tag`` with a working server binary."""
    root = binaries.runtimes_root() / tag
    dirs = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
    return next((d.name for d in dirs if _quiet(lambda: binaries.server_binary(d), None) is not None), None)


def _staged_row(gguf: Path) -> Dict[str, Any]:
    model_id = _model_id_for(gguf)
    # Split models: report the whole variant's bytes, not one part's.
    hit = catalog.find_entry_for_model(model_id)
    size = (hit[1].size_bytes if hit is not None else
            sum(path.stat().st_size for path in _inspection_parts(gguf)))
    row: Dict[str, Any] = {"id": model_id, "size_bytes": size, "size_label": _human_gb(size)}
    # This is the finished file's placement fact, available before a model is loaded. Live
    # placement remains in the separate status.placement map below.
    row.update(_lookup_inspection(gguf))
    return row


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
    configured_tag = section.get("tag") or binaries.default_tag()
    have = binaries.installed_tags()
    running = _state_endpoint()
    # The tag actually serving when state proves it; otherwise the boot ladder
    # target. Inspection itself stays conservative for an old state file with
    # no engine_tag (see _serving_engine_tag).
    boot_tag = binaries.active_tag(section)
    serving_tag = (running or {}).get("engine_tag")
    tag = str(serving_tag) if isinstance(serving_tag, str) and serving_tag else boot_tag
    runtime_backend = _installed_backend(tag)
    mdir = bootstrap.models_dir()
    # Resident models from the live router ({} when down): Loaded pills + eject. A failed read is never
    # silent: an empty dict here renders as 'Not in memory' on a machine whose VRAM is visibly full.
    loaded, placement = ({}, {}) if running is None else _quiet(
        lambda: _loaded_models(running), ({}, {}), warn="loaded-models read failed: %r")
    return {
        "enabled": bool(section.get("enabled")), "tag": tag, "configured_tag": configured_tag,
        "serving_tag": serving_tag if isinstance(serving_tag, str) and serving_tag else None,
        # Update pending = engine in use (enabled + something installed) and the configured tag
        # (pinned or release default) isn't on disk. The download is a button click, never automatic.
        "update_available": bool(section.get("enabled") and have and configured_tag not in have),
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


def _catalog_row(entry, budget, recommended, recommended_reason, staged_ids, *,
                 engine_tag: str | None = None) -> Dict[str, Any]:
    choice = catalog.select_variant(entry, budget, engine_tag=engine_tag)
    # Any variant of this family on disk counts as downloaded.
    dl = next((v for v in entry.variants if v.model_id in staged_ids), None)
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
    # Same overhead the launch decision prices (runtime buffers + vision projector + microbatch/MTP
    # logits): the row must advertise the window the model will actually get, not a paper number.
    overhead = (context_policy.RUNTIME_OVERHEAD_BYTES
                + (entry.mmproj.size_bytes if entry.mmproj else 0)
                + context_policy.ub_logits_bytes(entry.n_vocab, mtp_capable=entry.mtp))
    profile = entry.profile(variant, engine_tag=engine_tag)
    decision = context_policy.initial_window(profile, budget, overhead_bytes=overhead)
    download_total = entry.download_bytes(variant)
    if profile.lazy_table_bytes:
        if choice.zero_spill:
            quant_reason = (
                f"Recommended build ({variant.quant}) — core weights fit your GPU; a "
                f"{_human_gb(profile.lazy_table_bytes)} disk-backed lookup table is read on demand")
        else:
            quant_reason = (
                f"Compact build sized for this machine ({variant.quant}) — "
                f"{_human_gb(profile.lazy_table_bytes)} uses a disk-backed lookup table; "
                "ordinary weights still exceed GPU memory")
    else:
        quant_reason = _QUANT_REASONS.get(choice.reason_key, _QUANT_REASON_COMPACT).format(quant=variant.quant)
    row.update({
        "fits": True, "model_id": variant.model_id, "quant": variant.quant,
        "quant_validated": variant.validated, "size_bytes": download_total,
        "size_label": _human_gb(download_total), "variant_count": len(entry.variants),
        "quant_reason": quant_reason,
    })
    if isinstance(decision, estimator.PhysicsRefusal):
        row["fit_summary"] = row["quant_reason"]
        return row
    row.update(start_window=decision.window, start_window_label=_k_label(decision.window), spilled=decision.spilled)
    if profile.lazy_table_bytes:
        row.update(
            disk_backed_lookup_bytes=profile.lazy_table_bytes,
            disk_backed_lookup_label=_human_gb(profile.lazy_table_bytes),
        )
    if decision.window >= entry.n_ctx_train:
        shape = f"runs at its full {row['native_context_label']} context"
    else:
        shape = f"starts at {row['start_window_label']} and grows toward {row['native_context_label']} as you use it"
    notes = []
    if decision.spilled:
        notes.append("larger than your GPU memory — runs slower")
    if profile.lazy_table_bytes:
        notes.append(f"uses a {_human_gb(profile.lazy_table_bytes)} disk-backed lookup table")
    row["fit_summary"] = shape + (f" ({'; '.join(notes)})" if notes else "")
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
    engine_tag = binaries.active_tag(_runtime_section())
    # The reason key ships with the row so the Recommended badge's tooltip is the branch that actually
    # fired, not a re-derivation that can drift.
    recommended, recommended_reason = (
        catalog.recommended_entry(budget, _eligible_entries(), engine_tag=engine_tag) or (None, None))
    recommended_id = recommended.id if recommended is not None else None
    # Completeness-checked staging (split parts all present) — same answer the picker and router see, so a
    # mid-download model never reads as downloaded.
    staged_ids = set(bootstrap.staged_model_ids())
    return {"models": [
        _catalog_row(e, budget, recommended_id, recommended_reason, staged_ids, engine_tag=engine_tag)
        for e in catalog.CATALOG
    ]}


@router.post("/api/local-models/advanced/plan")
def local_models_advanced_plan(body: AdvancedPlanBody):
    """Preview advanced settings without changing configuration or the running server."""
    with _http_error(422):
        return _advanced_plan(body.model_id, body.request_mapping()).to_dict()


@router.post("/api/local-models/advanced/apply")
async def local_models_advanced_apply(body: AdvancedPlanBody):
    """Persist a validated per-model launch request and regenerate managed presets.

    A busy server is deliberately not bounced underneath live work. The user can retry after its
    request finishes, which preserves streams and session consistency rather than treating apply
    as an unsafe best-effort update.
    """
    if not _ADVANCED_APPLY_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="another advanced local-model update is in progress")
    try:
        plan = _advanced_plan(body.model_id, body.request_mapping())
        if not plan.fits:
            raise HTTPException(status_code=422, detail="; ".join(plan.reasons))
        sup = bootstrap.get_supervisor()
        # A refresh replaces the one router process, so checking only the
        # selected model could still cut off work running on another resident
        # model. The supervisor's no-argument form checks every child.
        if sup is not None and not _quiet(sup.is_idle, False):
            raise HTTPException(status_code=409, detail="the local runtime is serving a request; wait for it to become idle before applying")
        config = config_mod.load_config()
        section = config.setdefault("local_runtime", {})
        overrides = section.setdefault("launch_overrides", {})
        overrides[body.model_id] = plan.request.to_mapping()
        config_mod.save_config(config)
        restarted = await asyncio.to_thread(bootstrap.refresh_local_runtime) if sup is not None else False
        return {"ok": True, "model_id": body.model_id, "plan": plan.to_dict(),
                "restarted": restarted}
    finally:
        _ADVANCED_APPLY_LOCK.release()


def _api_server_config(config: dict) -> dict:
    """Return the profile-scoped API-server config block without assuming a running gateway."""
    return config.setdefault("gateway", {}).setdefault("platforms", {}).setdefault("api_server", {})


@router.get("/api/local-models/gateway-routes")
def local_models_gateway_routes():
    """List only local-model routes owned by this surface; credentials are never returned."""
    section = _api_server_config(_load_config())
    rows = []
    for mode, key in (("agent", "model_routes"), ("raw", "raw_model_routes")):
        for alias, route in (section.get(key) or {}).items():
            if not isinstance(route, dict) or str(route.get("provider", "")).lower() not in _LLAMACPP_PROVIDERS:
                continue
            rows.append({"alias": str(alias), "model_id": str(route.get("model") or ""), "mode": mode})
    return {"routes": sorted(rows, key=lambda row: (row["alias"], row["mode"]))}


@router.post("/api/local-models/gateway-routes")
def local_models_gateway_publish(body: GatewayPublishBody):
    """Register a stable, authenticated gateway alias. Gateway restart is deliberately explicit:
    changing a config file must not interrupt a live multi-profile gateway process."""
    alias = (body.alias or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", alias):
        raise HTTPException(status_code=422, detail="alias must contain letters, numbers, dots, underscores, or hyphens")
    if body.model_id not in bootstrap.staged_model_ids():
        raise HTTPException(status_code=404, detail=f"{body.model_id} is not downloaded")
    _require_model_engine(body.model_id)
    mode = (body.mode or "").strip().lower()
    if mode not in {"agent", "raw"}:
        raise HTTPException(status_code=422, detail="mode must be agent or raw")
    config = config_mod.load_config()
    api = _api_server_config(config)
    other_key = "raw_model_routes" if mode == "agent" else "model_routes"
    if alias in (api.get(other_key) or {}):
        raise HTTPException(status_code=409, detail="alias is already used by the other gateway route type")
    key = "model_routes" if mode == "agent" else "raw_model_routes"
    api.setdefault(key, {})[alias] = {"model": body.model_id, "provider": "llamacpp"}
    config_mod.save_config(config)
    return {"ok": True, "alias": alias, "model_id": body.model_id, "mode": mode,
            "restart_required": True}


@router.delete("/api/local-models/gateway-routes/{alias}")
def local_models_gateway_unpublish(alias: str):
    """Remove only this surface's local alias, leaving all unrelated gateway routes intact."""
    config = config_mod.load_config()
    api = _api_server_config(config)
    removed = False
    for key in ("model_routes", "raw_model_routes"):
        routes = api.get(key)
        route = routes.get(alias) if isinstance(routes, dict) else None
        if isinstance(route, dict) and str(route.get("provider", "")).lower() in _LLAMACPP_PROVIDERS:
            del routes[alias]
            removed = True
    if not removed:
        raise HTTPException(status_code=404, detail="local gateway route not found")
    config_mod.save_config(config)
    return {"ok": True, "alias": alias, "restart_required": True}


# ── runtime install (job) ────────────────────────────────────
def _runtime_progress_hook(job: Dict[str, Any]):
    """Adapter: ensure_runtime_installed's progress stream -> job fields, throttled to ~4 updates/s. Byte
    counters are CUMULATIVE across the plan (a multi-asset engine reads as one growing download, total
    growing as each asset's size becomes known); unpack/verify keep the counters — a bar bouncing back to
    zero after the bytes finished reads as failure."""
    state = {"last": 0.0, "banked": 0, "asset": None, "asset_total": 0}

    def hook(stage: str, done: int, total: int, label: str) -> None:
        now = time.monotonic()
        if now - state["last"] < 0.25 and done < total:
            return
        state["last"] = now
        suffix = f" ({label})" if label else ""
        if stage == "download":
            if label != state["asset"]:
                # Previous asset finished: bank its bytes so the counters keep climbing instead of restarting.
                state["banked"] += state["asset_total"]
                state["asset"] = label
            state["asset_total"] = total or done
            plan_done = state["banked"] + done
            plan_total = state["banked"] + (total or 0)
            _step(job, "downloading-runtime", f"Downloading the local engine{suffix} — {_human_gb(plan_done)}"
                  + (f" of {_human_gb(plan_total)}" if total else ""))
            job["done_bytes"] = plan_done
            job["total_bytes"] = plan_total or None
        elif stage == "extract":
            pct = f" — {min(100, round(done / total * 100))}%" if total else ""
            _step(job, "unpacking-runtime", f"Unpacking the engine{suffix}{pct}")
        else:  # verify
            _step(job, "verifying-runtime", f"Verifying the engine{suffix}")

    return hook


def _restart_on_new_tag(job: Dict[str, Any], tag: str) -> bool:
    """Engine update path: a server already running on an older tag moves to the new one now — the click was
    the consent. Fresh installs (no server) skip this; Use/boot handles their start."""
    sup = bootstrap.get_supervisor()
    if sup is None:
        # A backend restart can leave an adopted managed server behind. The
        # persisted state proves it is ours; an explicit engine-update click
        # is consent to bounce that server too, so the new compatible build
        # takes effect immediately rather than waiting for a later restart.
        running = _state_endpoint()
        if running is None or running.get("engine_tag") == tag:
            return False
        _step(job, "restarting", "Switching the running server to the new build")
        return bootstrap.refresh_local_runtime()
    if getattr(sup, "engine_tag", None) == tag:
        return False
    _step(job, "restarting", "Switching the running server to the new build")
    bootstrap.shutdown_local_runtime()
    bootstrap.ensure_local_runtime(_load_config(), force=True)
    return True


@router.post("/api/local-models/runtime/install")
async def local_models_runtime_install(body: RuntimeInstallBody):
    tag, backend = _runtime_target(body.backend, body.tag)
    plan = _resolve_assets_or_400(tag, backend)
    job = _job("runtime-install", f"llama.cpp {tag} ({backend})")

    def _run():
        previous = binaries.installed_tags()
        _step(job, "downloading", f"Fetching {len(plan.assets)} package(s) for {backend}")
        binaries.ensure_runtime_installed(tag, backend, progress=_runtime_progress_hook(job))
        # A compatible-build click is also consent to use that build for the
        # next boot. Persist only after its archive passed verification.
        _persist_runtime_tag(tag)
        # Restart failure is logged only: the new build is installed either way and the next boot serves it.
        restarted = _quiet(lambda: _restart_on_new_tag(job, tag), False, warn="post-update restart skipped: %s")
        # N-1 retention, only after the new tag verified: keep it + the newest previous build as the rollback pin target.
        _quiet(lambda: binaries.prune_old_tags([tag] + [t for t in previous if t != tag][:1]), None,
               warn="runtime prune skipped: %s")
        _finish(job, f"llama.cpp {tag} ready ({backend})" + (" — server restarted on the new build" if restarted else ""))

    _spawn_job(job, "lr-runtime-install", _run, fail_msg="runtime install failed: %s")
    return {"job_id": job["job_id"], "backend": backend, "tag": tag}


# ── model download (job with byte progress) ──────────────────
def _download_target(model_id: str):
    """(entry, variant) for a family id (this machine's selected variant — the same planning budget as the
    catalog, so the user downloads exactly the build the row advertised) or an exact variant model_id."""
    entry = catalog.catalog_by_id().get(model_id)
    if entry is None:  # exact variant id, or nothing we know (404)
        found = catalog.find_entry_for_model(model_id)
        if found is None:
            _entry_or_404(model_id)
        entry, variant = found
    else:
        variant = None
    if _engine_too_old(entry.min_engine):
        raise HTTPException(status_code=409, detail=(
            f"{entry.display_name} needs llama.cpp {entry.min_engine} or newer — update the engine first"))
    if variant is not None:
        return entry, variant
    choice = catalog.select_variant(
        entry, hardware.probe_budget(planning=True),
        engine_tag=binaries.active_tag(_runtime_section()))
    if choice is None:
        raise HTTPException(status_code=409, detail=f"no variant of {entry.id} fits this machine")
    return entry, choice.variant


@router.post("/api/local-models/download")
async def local_models_download(body: ModelDownloadBody):
    """Accepts either a family id (downloads this machine's selected variant) or an exact variant model_id."""
    entry, variant = _download_target(body.model_id)
    if variant.model_id in bootstrap.staged_model_ids():
        return {"job_id": None, "already_downloaded": True, "model_id": variant.model_id}
    plan = _download_plan(entry, variant)
    job, created = _claim_model_download(
        variant.model_id, f"{entry.display_name} ({variant.quant})", entry.id)
    if not created:
        return {"job_id": job["job_id"], "already_downloading": True, "model_id": variant.model_id}
    job["total_bytes"] = sum(p[2] for p in plan)
    _spawn_job(job, "lr-model-download", lambda: _run_download_plan(job, plan, entry.display_name),
               fail_msg="model download failed: %s", download_label=entry.display_name,
               on_exit=lambda: _release_model_download(variant.model_id, job["job_id"]))
    return {"job_id": job["job_id"], "model_id": variant.model_id}


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
    """(entry, variant) to set up: explicit id, else this machine's recommendation, else the first servable entry."""
    engine_tag = binaries.active_tag(_runtime_section())
    if body.model_id:
        candidates = [_entry_or_404(body.model_id)]
    else:
        picked = catalog.recommended_entry(budget, _eligible_entries(), engine_tag=engine_tag)
        candidates = ([picked[0]] if picked else []) + [e for e in catalog.CATALOG if not picked or e.id != picked[0].id]
    for candidate in candidates:
        choice = catalog.select_variant(candidate, budget, engine_tag=engine_tag)
        if choice is not None and not _engine_too_old(candidate.min_engine):
            return candidate, choice.variant
    raise HTTPException(status_code=409, detail=(
        "no catalog model fits this machine — open Local Models to browse for a smaller build"))


@router.post("/api/local-models/quickstart")
async def local_models_quickstart(body: QuickstartBody):
    """One job: install the runtime (if missing), download this machine's build of the recommended model (if
    missing), make it the default. Each leg is the same code the individual routes run, so 'Configure' and
    quickstart can never disagree. Preflight rejects (no servable entry, engine too old) fail the POST
    synchronously so the button can explain itself; everything slow runs in the job with phase/byte progress."""
    entry, variant = _quickstart_target(body, hardware.probe_budget(planning=True))
    tag, backend = _runtime_target()
    need_runtime = not binaries.installed_tags()
    if need_runtime:
        _resolve_assets_or_400(tag, backend)
    need_download = variant.model_id not in bootstrap.staged_model_ids()
    download_plan = _download_plan(entry, variant) if need_download else []
    download_bytes = sum(p[2] for p in download_plan)
    if not _QUICKSTART_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Setup is already running")
    job = _job("quickstart", entry.display_name, model_id=entry.id)
    job["total_bytes"] = download_bytes or None

    def _run():
        if need_runtime:
            _step(job, "installing-runtime", "Installing the local engine")
            binaries.ensure_runtime_installed(tag, backend, progress=_runtime_progress_hook(job))
        if need_download:
            # The runtime leg repurposed the byte counters for its own stages — reset them to the model plan.
            job["done_bytes"] = 0
            job["total_bytes"] = download_bytes
            _run_download_plan(job, download_plan, entry.display_name)
        # Activate: same sequence as /activate's job body.
        _ensure_server(job, _set_runtime_enabled(True), variant.model_id,
                       fail_detail="The local server could not start — open Local Models for details",
                       skip_msg="quickstart rescan check skipped")
        _assign_default(job, variant.model_id)
        _finish(job, f"{entry.display_name} is ready — new chats use it")

    _spawn_job(job, "lr-quickstart", _run, fail_msg="quickstart failed: %s", on_exit=_QUICKSTART_LOCK.release)
    return {"job_id": job["job_id"], "model_id": entry.id, "display_name": entry.display_name,
            "needs_runtime": need_runtime, "needs_download": need_download, "download_bytes": download_bytes}


# ── server lifecycle: turn the engine on/off ─────────────────
def _terminate_state_pid() -> None:
    """Server owned by another process (or an orphan): terminate via the state file's pid, then clear the state."""
    import psutil  # type: ignore

    state = json.loads(supervisor.state_path().read_text(encoding="utf-8"))
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
    active_model_id = _active_llamacpp_model_id()
    if active_model_id and active_model_id in bootstrap.staged_model_ids():
        _require_model_engine(active_model_id)
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
    _require_model_engine(body.model_id)
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
        jobs = sorted(_JOBS.values(), key=lambda j: (j["status"] != "running", -j["started_at"]))
    return {"jobs": [_job_view(job) for job in jobs[:20]]}


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
    return {"files": [
        dict(g.__dict__, paths=list(g.paths), download_model_id=_browsed_model_id(repo, g.paths[0]))
        for g in groups
    ]}


@router.post("/api/local-models/download-browsed")
async def local_models_download_browsed(body: BrowsedDownloadBody):
    """Download a publisher-provided GGUF quant into the managed models dir.

    Hermes does not rewrite/re-quantize it. Once every shard lands, status reads the real header
    and reports whether its lookup table is disk-backed, resident, or needs an engine update.
    """
    if not _safe_hf_repo(body.repo):
        raise HTTPException(status_code=422, detail="Pick a Hugging Face model repository (owner/name)")
    paths = _complete_browsed_paths(body.paths)
    model_id = _browsed_model_id(body.repo, paths[0])
    if model_id in bootstrap.staged_model_ids():
        return {"job_id": None, "already_downloaded": True, "model_id": model_id}
    job, created = _claim_model_download(model_id, f"{model_id} (from {body.repo})", model_id)
    if not created:
        return {"job_id": job["job_id"], "already_downloading": True, "model_id": model_id}

    def _fetch():
        job["phase"] = "downloading"
        urls = [_hf_url(body.repo, p) for p in paths]
        # Browser listings are advisory.  Ask the publisher for every shard's size so split
        # progress is a real whole-model total, never the first part's size.  If a host does not
        # support ranges, leave the denominator unknown instead of showing a bogus percentage.
        source_sizes = [_probe_range_support(url) for url in urls]
        known_total = bool(source_sizes) and all(size > 0 for size in source_sizes)
        if known_total:
            job["total_bytes"] = sum(source_sizes)
        done_before = 0
        for p, url in zip(paths, urls):
            dest = bootstrap.models_dir() / _browsed_local_name(body.repo, p)
            if dest.exists():
                done_before += dest.stat().st_size
            else:
                # Preserve the verified group total when available.  For an unknown-size split,
                # also preserve ``None``: a per-shard denominator would lie about completion.
                download_file(url, dest, job, base_done=done_before,
                              keep_totals=known_total or len(paths) > 1)
                done_before += dest.stat().st_size
            job["done_bytes"] = done_before
            job["phase"] = "downloading"

    _spawn_job(job, "lm-download-browsed", _fetch, download_label=model_id,
               download_detail=f"{model_id} downloaded",
               on_exit=lambda: _release_model_download(model_id, job["job_id"]))
    return {"job_id": job["job_id"], "model_id": model_id}


@router.post("/api/local-models/sideload")
async def local_models_sideload(body: SideloadBody):
    """Register a GGUF already on this machine: link it into the managed models dir (copy only when linking is
    impossible) and bounce the router. The original stays put; delete-from-Hermes removes only our link."""
    src = Path(body.path)
    if not src.is_file() or src.suffix.lower() != ".gguf":
        raise HTTPException(status_code=422, detail="Pick a .gguf model file")
    # The staged-model scanner and split reader use canonical lower-case
    # `.gguf` names on Linux. Preserve a user's stem but normalize only the
    # extension so a valid `MODEL.GGUF` sideload is actually discoverable.
    dest = bootstrap.models_dir() / f"{src.stem}{_GGUF_SUFFIX}"
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
