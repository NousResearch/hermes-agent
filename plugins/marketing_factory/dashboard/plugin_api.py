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


class ChannelModeBody(BaseModel):
    mode: str = Field(..., pattern="^(dry_run|live)$")
    reviewer: str = "dashboard"


class AppCreateBody(BaseModel):
    slug: str
    name: str
    positioning: str = ""
    icp: str = ""
    tone: str = ""
    cta: str = ""
    channels: list[str] = []
    content_pillars: list[str] = []
    claims: list[str] = []
    forbidden_claims: list[str] = []
    links: list[str] = []
    competitors: list[str] = []
    assets: list[str] = []


class AppPatchBody(BaseModel):
    name: Optional[str] = None
    positioning: Optional[str] = None
    icp: Optional[str] = None
    tone: Optional[str] = None
    cta: Optional[str] = None
    channels: Optional[list[str]] = None
    content_pillars: Optional[list[str]] = None
    claims: Optional[list[str]] = None
    forbidden_claims: Optional[list[str]] = None
    links: Optional[list[str]] = None
    competitors: Optional[list[str]] = None
    assets: Optional[list[str]] = None


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
    advisor = _pipe(store).advise()
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
        "advisor": advisor,
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


@router.post("/drafts/approve-all")
async def approve_all_pending(app_slug: Optional[str] = None, reviewer: str = "dashboard"):
    store = _store()
    approved: list[Dict[str, Any]] = []
    for draft in store.list_drafts(app_slug=app_slug, status="needs_review"):
        if draft.get("safety", {}).get("passed"):
            approved.append(store.set_approval(draft["id"], "approved", reviewer=reviewer, reason="bulk-approved from dashboard"))
    return {"result": approved, "overview": _overview(store)}


@router.post("/drafts/reject-all")
async def reject_all_pending(app_slug: Optional[str] = None, reviewer: str = "dashboard"):
    store = _store()
    rejected: list[Dict[str, Any]] = []
    for draft in store.list_drafts(app_slug=app_slug, status="needs_review"):
        rejected.append(store.set_approval(draft["id"], "rejected", reviewer=reviewer, reason="bulk-rejected from dashboard"))
    return {"result": rejected, "overview": _overview(store)}


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


@router.post("/publish")
async def publish_scheduled_with_channel_modes(app_slug: Optional[str] = None):
    """Channel-mode-aware publish. Drafts on channels marked `live` go through
    the registered connector if available; everything else falls back to dry_run.
    Always safe to call — no connectors are registered by default, so this
    behaves identically to /publish-dry-run until a human wires one in."""
    store = _store()
    result = _pipe(store).publisher.publish_scheduled(store, app_slug=app_slug)
    return {"result": result, "overview": _overview(store)}


@router.post("/apps")
async def create_app(body: AppCreateBody):
    if not body.name.strip() or not body.slug.strip():
        raise HTTPException(status_code=400, detail="slug and name are required")
    if not body.channels:
        raise HTTPException(status_code=400, detail="at least one channel is required")
    store = _store()
    store.initialize()
    try:
        result = store.upsert_app(body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"result": result, "overview": _overview(store)}


@router.patch("/apps/{app_slug}")
async def patch_app(app_slug: str, body: AppPatchBody):
    store = _store()
    try:
        existing = store.require_app(app_slug)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    merged = {**existing, **updates, "slug": app_slug}
    result = store.upsert_app(merged)
    return {"result": result, "overview": _overview(store)}


@router.delete("/apps/{app_slug}")
async def delete_app(app_slug: str, cascade: bool = True):
    store = _store()
    try:
        result = store.remove_app(app_slug, cascade=cascade)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"result": result, "overview": _overview(store)}


@router.post("/drafts/{draft_id}/regenerate")
async def regenerate_draft(draft_id: str):
    store = _store()
    try:
        result = _pipe(store).regenerate_draft(draft_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"result": result, "overview": _overview(store)}


@router.get("/advise")
async def advise():
    store = _store()
    result = _pipe(store).advise()
    return result


@router.post("/poll")
async def run_poll():
    """One scheduled-poller tick across all apps. Safe to call any time.
    Idempotent for not-yet-due drafts (skipped) and for already-published drafts
    (skipped). Designed for cron:
        hermes cron create --schedule "every 5m" --command "hermes marketing-factory poll"
    """
    store = _store()
    result = _pipe(store).poll()
    return {"result": result, "overview": _overview(store)}


@router.post("/apps/{app_slug}/channels/{channel}/mode")
async def set_channel_mode(app_slug: str, channel: str, body: ChannelModeBody):
    store = _store()
    try:
        result = store.set_channel_mode(app_slug, channel, body.mode, reviewer=body.reviewer)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"result": result, "overview": _overview(store)}
