"""
Meta-Router API Server v2.0
FastAPI wrapper around the shared meta-router runtime on port 3120.
"""
from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from gateway.meta_router_executor import run_outcome_only
from gateway.meta_router_runtime import make_route_decision

app = FastAPI(title="Meta-Router", version="2.0.0")


class ClassifyRequest(BaseModel):
    text: str
    source: str = "api"
    surface: str = "http"
    session_id: str | None = None


class ClassifyResponse(BaseModel):
    request_id: str
    type: str
    mode: str
    confidence: float
    directive: str
    prepend_text: str
    text_with_directive: str
    primary: str
    secondary: str | None = None
    budget_multiplier: float
    routing_artifact_version: str
    bypassed: bool
    bypass_reason: str = ""
    memory_need: str
    memory_authority: list[str]
    required_tools: list[str]
    optional_tools: list[str]
    skip_tools: list[str]
    max_memory_steps: int
    memory_policy_version: str


class OutcomeRequest(BaseModel):
    request_id: str
    task_type: str
    session_id: str | None = None
    source: str = "api"
    surface: str = "http"
    routing_artifact_version: str = "static-default"
    success: bool = True
    error: str | None = None
    duration_ms: float | None = None
    notes: list[str] | None = None


class OutcomeResponse(BaseModel):
    status: str


@app.post("/classify", response_model=ClassifyResponse)
def classify_message(req: ClassifyRequest) -> ClassifyResponse:
    if not req.text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    decision = make_route_decision(
        req.text,
        source=req.source or "api",
        surface=req.surface or "http",
        session_id=req.session_id,
    )
    prepend_text = getattr(decision, "prepend_text", decision.directive) if not decision.bypassed else ""
    text_with_directive = req.text if decision.bypassed or not prepend_text else f"{prepend_text}\n\n{req.text}"

    return ClassifyResponse(
        request_id=decision.request_id,
        type=decision.type,
        mode=decision.mode,
        confidence=decision.confidence,
        directive=decision.directive,
        prepend_text=prepend_text,
        text_with_directive=text_with_directive,
        primary=decision.primary,
        secondary=decision.secondary,
        budget_multiplier=decision.budget_multiplier,
        routing_artifact_version=decision.routing_artifact_version,
        bypassed=decision.bypassed,
        bypass_reason=decision.bypass_reason,
        memory_need=decision.memory_need,
        memory_authority=list(decision.memory_authority or []),
        required_tools=list(decision.required_tools or []),
        optional_tools=list(decision.optional_tools or []),
        skip_tools=list(decision.skip_tools or []),
        max_memory_steps=decision.max_memory_steps,
        memory_policy_version=decision.memory_policy_version,
    )


@app.post("/outcome", response_model=OutcomeResponse)
def log_terminal_outcome(req: OutcomeRequest) -> OutcomeResponse:
    if not req.request_id.strip():
        raise HTTPException(status_code=400, detail="request_id must not be empty")
    if not req.task_type.strip():
        raise HTTPException(status_code=400, detail="task_type must not be empty")

    duration_ms = max(float(req.duration_ms or 0.0), 0.0)
    t0 = time.time() - (duration_ms / 1000.0)
    notes_extra = list(req.notes or [])
    notes_extra.append(f"success={str(bool(req.success)).lower()}")

    run_outcome_only(
        request_id=req.request_id,
        task_type=req.task_type,
        t0=t0,
        routing_artifact_version=req.routing_artifact_version,
        session_id=req.session_id,
        source=req.source or "api",
        surface=req.surface or "http",
        error=req.error or None,
        notes_extra=notes_extra,
    )
    return OutcomeResponse(status="ok")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "2.0.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("gateway.meta_router_server:app", host="127.0.0.1", port=3120, reload=False)
