from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query

router = APIRouter()


def _load_core():
    path = Path(__file__).resolve().parents[1] / "roster_core.py"
    spec = importlib.util.spec_from_file_location("agent_roster_dashboard_core", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@router.get("/roster")
def roster():
    return _load_core().build_roster()


@router.get("/profiles")
def profiles():
    data = _load_core().build_roster()
    return {
        "profiles": data["profiles"],
        "summary": data["summary"],
        "count": len(data["profiles"]),
        "generated_at": data["generated_at"],
    }


@router.get("/boards")
def boards():
    data = _load_core().build_roster()
    return {
        "boards": data["boards"],
        "summary": data["summary"],
        "count": len(data["boards"]),
        "generated_at": data["generated_at"],
    }


@router.get("/violations")
def violations(severity: Optional[str] = Query(None)):
    data = _load_core().build_roster()
    items = data["violations"]
    if severity:
        wanted = severity.strip().lower()
        items = [v for v in items if str(v.get("severity", "")).lower() == wanted]
    return {
        "violations": items,
        "count": len(items),
        "summary": data["summary"],
        "generated_at": data["generated_at"],
    }


@router.get("/pipeline")
def pipeline():
    data = _load_core().build_roster()
    payload = data["pipeline"]
    return {
        **payload,
        "summary": data["summary"],
        "generated_at": data["generated_at"],
    }


@router.get("/audit")
def audit(limit: int = Query(200, ge=1, le=1000)):
    rows = _load_core().read_audit(limit=limit)
    return {"events": rows, "count": len(rows)}
