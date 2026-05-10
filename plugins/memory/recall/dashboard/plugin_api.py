"""Recall dashboard plugin API.

Mounted by Hermes dashboard at /api/plugins/recall/.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter()


def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")


def _plugin_root() -> Path:
    # dashboard/plugin_api.py -> plugin root
    return Path(__file__).resolve().parents[1]


def _load_store_class():
    root = _plugin_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from store import RecallStore
    return RecallStore


def _db_path() -> Path:
    return _hermes_home() / "recall_memory.sqlite"


def _with_store(fn):
    Store = _load_store_class()
    store = Store(_db_path())
    try:
        return fn(store)
    finally:
        store.close()


def _load_provider_class():
    root = _plugin_root()
    module_name = "hermes_dashboard_recall_provider"
    existing = sys.modules.get(module_name)
    if existing is not None and hasattr(existing, "RecallMemoryProvider"):
        return existing.RecallMemoryProvider
    spec = importlib.util.spec_from_file_location(module_name, root / "__init__.py", submodule_search_locations=[str(root)])
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load Recall provider module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.RecallMemoryProvider


def _provider(cwd: str = ""):
    Provider = _load_provider_class()
    home = _hermes_home()
    provider = Provider({"db_path": str(_db_path())})
    provider.initialize("dashboard", hermes_home=home, cwd=cwd)
    return provider


def _call(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    provider = _provider()
    try:
        raw = provider.handle_tool_call(tool_name, args)
    finally:
        provider.shutdown()
    try:
        return json.loads(raw)
    except Exception:
        return {"success": False, "error": raw}


class MarkBody(BaseModel):
    status: str
    reason: str = ""


class PromoteBody(BaseModel):
    target: str = "memory"
    content: Optional[str] = None
    confirm: bool = False
    allow_low_quality: bool = False
    allow_rejected: bool = False
    reason: str = ""


class ConsolidationApplyBody(BaseModel):
    canonical_id: str
    duplicate_ids: list[str]
    confirm: bool = False
    reason: str = ""


def _build_info(store) -> dict[str, Any]:
    try:
        Provider = _load_provider_class()
        version = getattr(sys.modules.get(Provider.__module__), "__version__", "")
        capabilities = getattr(sys.modules.get(Provider.__module__), "PROVIDER_BUILD", {}).get("capabilities", [])
    except Exception:
        version = ""
        capabilities = []
    try:
        schema_version = store.conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()["value"]
    except Exception:
        schema_version = ""
    return {
        "name": "recall",
        "version": version,
        "schema_version": str(schema_version),
        "db_path": str(_db_path()),
        "capabilities": capabilities,
    }


@router.get("/overview")
async def overview() -> dict[str, Any]:
    def read(store):
        from audit import verify_audit_chain
        return {
            "stats": store.archive_stats(),
            "diagnose": store.diagnose(),
            "audit": verify_audit_chain(store.conn),
            "build_info": _build_info(store),
        }
    return _with_store(read)


@router.get("/observations")
async def observations(
    status: str = Query("candidate"),
    scope: Optional[str] = None,
    type: Optional[str] = None,
    q: Optional[str] = None,
    recommended_action: Optional[str] = None,
    min_quality_score: float = Query(0.0, ge=0.0, le=1.0),
    exclude_episode: bool = False,
    limit: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    statuses = ["candidate", "active", "promoted", "rejected"] if status == "all" else [status]
    def read(store):
        if q and q.strip():
            results = store.search_observations(q, limit=limit * 3, scope=scope, project_path=None)
            allowed = set(statuses)
            results = [store._quality_rank_item(row) for row in results if row.get("status") in allowed]
        else:
            results = store.rank_observations(limit=limit * max(len(statuses), 1), include_statuses=statuses, scope=scope, project_path=None)
        if type:
            results = [row for row in results if row.get("type") == type]
        if exclude_episode:
            results = [row for row in results if row.get("type") != "episode"]
        if recommended_action:
            results = [row for row in results if row.get("recommended_action") == recommended_action]
        if min_quality_score:
            results = [row for row in results if float(row.get("quality_score") or 0.0) >= min_quality_score]
        return {
            "results": results[:limit],
            "query": q or "",
            "filters": {
                "status": status,
                "scope": scope,
                "type": type,
                "recommended_action": recommended_action,
                "min_quality_score": min_quality_score,
                "exclude_episode": exclude_episode,
            },
            "trust": "lower-trust Recall archive; review before applying",
        }
    return _with_store(read)


@router.get("/observations/{observation_id}")
async def observation_detail(observation_id: str) -> dict[str, Any]:
    def read(store):
        row = store.get_observation(observation_id)
        if not row:
            raise HTTPException(status_code=404, detail="Observation not found")
        return store._quality_rank_item(row)
    return _with_store(read)


@router.get("/consolidations")
async def consolidations(
    limit: int = Query(20, ge=1, le=100),
    include_low_quality: bool = False,
    min_quality_score: float = 0.45,
) -> dict[str, Any]:
    def read(store):
        return {
            "results": store.suggest_consolidations(limit=limit, include_low_quality=include_low_quality, min_quality_score=min_quality_score),
            "filters": {"include_low_quality": include_low_quality, "min_quality_score": min_quality_score},
            "trust": "suggestions only; no archive rows were mutated",
        }
    return _with_store(read)


@router.post("/consolidations/apply")
async def apply_consolidation(body: ConsolidationApplyBody) -> dict[str, Any]:
    result = _call(
        "memory_consolidation_apply",
        {
            "canonical_id": body.canonical_id,
            "duplicate_ids": body.duplicate_ids,
            "confirm": body.confirm,
            "reason": body.reason,
        },
    )
    if result.get("success") is False and not result.get("requires_confirm"):
        raise HTTPException(status_code=400, detail=result.get("error") or "Consolidation apply failed")
    return result


@router.post("/observations/{observation_id}/mark")
async def mark(observation_id: str, body: MarkBody) -> dict[str, Any]:
    if body.status not in {"candidate", "active", "rejected", "promoted"}:
        raise HTTPException(status_code=400, detail="Invalid status")
    def write(store):
        ok = store.mark_observation_status(observation_id, body.status)
        if ok:
            store.append_audit_event("result", "candidate_mark", "observation", observation_id, {"status": body.status, "reason": body.reason})
        return {"success": ok, "id": observation_id, "status": body.status}
    result = _with_store(write)
    if result.get("success") is False:
        raise HTTPException(status_code=400, detail=result.get("error") or "Mark failed")
    return result


@router.post("/observations/{observation_id}/promote")
async def promote(observation_id: str, body: PromoteBody) -> dict[str, Any]:
    args = body.model_dump()
    args["id"] = observation_id
    result = _call("memory_promote_candidate", args)
    if result.get("success") is False and not result.get("requires_confirm"):
        raise HTTPException(status_code=400, detail=result.get("error") or "Promotion failed")
    return result


@router.get("/audit")
async def audit(limit: int = Query(20, ge=1, le=100)) -> dict[str, Any]:
    return _with_store(lambda store: {"events": store.audit_events(limit=limit)})
