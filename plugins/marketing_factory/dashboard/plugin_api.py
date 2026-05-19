"""Dashboard API routes for the Marketing Agent Factory plugin.

Mounted by the Hermes dashboard at ``/api/plugins/marketing_factory``.
All operations remain dry-run-first: the only publishing endpoint delegates to
``PublisherAgent.dry_run_publish_scheduled`` / ``MarketingFactoryStore.dry_run_publish``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
from plugins.marketing_factory.store import MarketingFactoryStore

router = APIRouter()


class GenerateBody(BaseModel):
    app_slug: str
    days: int = Field(default=7, ge=1, le=31)


class DraftActionBody(BaseModel):
    reviewer: str = "dashboard"
    reason: Optional[str] = None


class ScheduleBody(BaseModel):
    scheduled_for: Optional[str] = None


def _store() -> MarketingFactoryStore:
    return MarketingFactoryStore()


def _pipe(store: MarketingFactoryStore) -> MarketingFactoryPipeline:
    return MarketingFactoryPipeline(store)


def _status_counts(drafts: list[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for draft in drafts:
        status = str(draft.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _overview(store: MarketingFactoryStore) -> Dict[str, Any]:
    store.initialize()
    apps = store.list_apps()
    drafts = store.list_drafts()
    campaigns = store.list_campaigns()
    schedules = store.list_schedules()
    audit = store.list_audit(limit=25)
    state = store.load()
    return {
        "summary": store.summary(),
        "apps": apps,
        "campaigns": campaigns[-12:],
        "drafts": drafts[-50:],
        "pending_approvals": [draft for draft in drafts if draft.get("status") == "needs_review"],
        "schedules": schedules[:50],
        "publish_events": sorted(state["publish_events"].values(), key=lambda item: item.get("created_at", ""))[-25:],
        "analytics": sorted(state["analytics"].values(), key=lambda item: item.get("created_at", ""))[-20:],
        "brand_memories": state.get("brand_memories", {}),
        "draft_status_counts": _status_counts(drafts),
        "audit": audit,
        "next_action": _next_action(store.summary(), drafts, schedules),
    }


def _next_action(summary: Dict[str, Any], drafts: list[Dict[str, Any]], schedules: list[Dict[str, Any]]) -> Dict[str, str]:
    if not summary.get("apps"):
        return {"title": "Initialize brand profiles", "detail": "Seed Pupular and SetVenue sample profiles before generating campaigns."}
    pending = [draft for draft in drafts if draft.get("status") == "needs_review"]
    if pending:
        return {"title": "Review pending drafts", "detail": f"{len(pending)} draft(s) need approval before scheduling."}
    approved = [draft for draft in drafts if draft.get("status") == "approved"]
    if approved:
        return {"title": "Schedule approved drafts", "detail": f"{len(approved)} approved draft(s) are ready for the calendar."}
    scheduled = [draft for draft in drafts if draft.get("status") == "scheduled"]
    if scheduled:
        return {"title": "Run dry-run publisher", "detail": f"{len(scheduled)} scheduled draft(s) can be dry-run published. No public posting occurs."}
    if not schedules:
        return {"title": "Generate a campaign", "detail": "Create dry-run draft queues for Pupular or SetVenue."}
    return {"title": "Inspect audit trail", "detail": "Dry-run system is idle. Review logs, learnings, and channel performance notes."}


@router.get("/overview")
async def overview():
    return _overview(_store())


@router.post("/init")
async def initialize_samples():
    store = _store()
    result = _pipe(store).initialize_samples()
    return {"result": result, "overview": _overview(store)}


@router.post("/campaigns/generate")
async def generate_campaign(body: GenerateBody):
    store = _store()
    try:
        result = _pipe(store).generate_campaign(body.app_slug, days=body.days)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"result": {"campaign": result["campaign"], "drafts": result["drafts"]}, "overview": _overview(store)}


@router.post("/drafts/{draft_id}/approve")
async def approve_draft(draft_id: str, body: DraftActionBody):
    store = _store()
    try:
        result = store.set_approval(draft_id, "approved", reviewer=body.reviewer, reason=body.reason)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"result": result, "overview": _overview(store)}


@router.post("/drafts/{draft_id}/reject")
async def reject_draft(draft_id: str, body: DraftActionBody):
    store = _store()
    try:
        result = store.set_approval(draft_id, "rejected", reviewer=body.reviewer, reason=body.reason)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"result": result, "overview": _overview(store)}


@router.post("/drafts/{draft_id}/schedule")
async def schedule_draft(draft_id: str, body: ScheduleBody):
    store = _store()
    try:
        draft = store.get_draft(draft_id)
        scheduled_for = body.scheduled_for or draft.get("scheduled_for")
        if not scheduled_for:
            raise ValueError("scheduled_for is required when the draft has no suggested schedule")
        result = store.schedule_draft(draft_id, str(scheduled_for))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"result": result, "overview": _overview(store)}


@router.post("/drafts/{draft_id}/publish-dry-run")
async def publish_draft_dry_run(draft_id: str):
    store = _store()
    try:
        result = store.dry_run_publish(draft_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"result": result, "overview": _overview(store)}


@router.post("/schedule")
async def schedule_approved(app_slug: Optional[str] = None):
    store = _store()
    result = _pipe(store).scheduler.schedule_approved(store, app_slug=app_slug)
    return {"result": result, "overview": _overview(store)}


@router.post("/publish-dry-run")
async def publish_scheduled_dry_run(app_slug: Optional[str] = None):
    store = _store()
    result = _pipe(store).publisher.dry_run_publish_scheduled(store, app_slug=app_slug)
    return {"result": result, "overview": _overview(store)}
