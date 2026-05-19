"""
Workflow Engine plugin — FastAPI router.

Phase 1: /health returns real data; all other endpoints return 501.
Phase 2a+ will fill in the bodies.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)

router = APIRouter()

_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Health — the only real endpoint in Phase 1
# ---------------------------------------------------------------------------


@router.get("/health")
async def health() -> dict:
    return {"ok": True, "version": _VERSION}


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


def _not_implemented() -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={"error": "not_implemented", "detail": "Phase 2a — coming soon"},
    )


# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------


@router.get("/definitions")
async def list_definitions():
    return _not_implemented()


@router.post("/definitions")
async def create_definition():
    return _not_implemented()


@router.get("/definitions/{id}")
async def get_definition(id: str):  # noqa: A002
    return _not_implemented()


@router.get("/definitions/{id}/parsed")
async def get_definition_parsed(id: str):  # noqa: A002
    return _not_implemented()


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


@router.get("/runs")
async def list_runs():
    return _not_implemented()


@router.post("/runs")
async def create_run():
    return _not_implemented()


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    return _not_implemented()


@router.post("/runs/{run_id}/approve")
async def approve_run(run_id: str):
    return _not_implemented()


# ---------------------------------------------------------------------------
# Events (SSE)
# ---------------------------------------------------------------------------


@router.get("/events")
async def events():
    return _not_implemented()
