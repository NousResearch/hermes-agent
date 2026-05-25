"""Dashboard API routes for the Marketing Agent Factory plugin.

Mounted by the Hermes dashboard at ``/api/plugins/marketing_factory``.
All operations remain dry-run-first: the only publishing endpoint delegates to
``PublisherAgent.dry_run_publish_scheduled`` / ``MarketingFactoryStore.dry_run_publish``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
from plugins.marketing_factory import progress_bus
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


class DraftEditBody(BaseModel):
    body: str
    editor: str = "dashboard"


class DraftRescheduleBody(BaseModel):
    scheduled_for: str


class AutoGenerateBody(BaseModel):
    enabled: bool
    threshold: Optional[int] = None
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
    from plugins.marketing_factory.pipeline import draft_checklist
    store.initialize()
    apps = store.list_apps()
    drafts = store.list_drafts()
    campaigns = store.list_campaigns()
    schedules = store.list_schedules()
    audit = store.list_audit(limit=25)
    state = store.load()
    advisor = _pipe(store).advise()
    # Cheap per-draft checklist; computed once per overview rather than fetched
    # per-draft from the client.
    apps_by_slug = {app["slug"]: app for app in apps}
    enriched_drafts = []
    for draft in drafts[-50:]:
        enriched = dict(draft)
        owning_app = apps_by_slug.get(draft.get("app_slug"))
        if owning_app:
            enriched["_checklist"] = draft_checklist(draft, owning_app)
        enriched_drafts.append(enriched)
    drafts_window = enriched_drafts
    return {
        "summary": store.summary(),
        "apps": apps,
        "campaigns": campaigns[-12:],
        "drafts": drafts_window,
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
    pipe = _pipe(store)
    # generate_campaign is CPU+LLM-bound and takes 60-90s. Run it on a worker
    # thread so the event loop stays free to stream progress events to the
    # dashboard via the SSE endpoint while it runs.
    try:
        result = await asyncio.to_thread(pipe.generate_campaign, body.app_slug, body.days)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"result": {"campaign": result["campaign"], "drafts": result["drafts"]}, "overview": _overview(store)}


@router.get("/progress/stream")
async def progress_stream():
    """Server-Sent Events stream of live agent activity.

    Each event is published by the pipeline at agent boundaries
    (campaign.start, agent.start, agent.end, campaign.end). On connect we
    backfill the last ~30 events so the dashboard immediately reflects
    what happened seconds ago.
    """
    loop = asyncio.get_running_loop()
    queue = progress_bus.subscribe(loop=loop)

    async def event_generator():
        try:
            # Backfill so a fresh SSE subscriber instantly sees recent state.
            for event in progress_bus.recent(limit=30):
                yield f"data: {json.dumps(event)}\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=20.0)
                except asyncio.TimeoutError:
                    # SSE heartbeat — keeps proxies / browsers from closing idle streams
                    yield ": ping\n\n"
                    continue
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            progress_bus.unsubscribe(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/drafts/{draft_id}/approve")
async def approve_draft(draft_id: str, body: DraftActionBody, resolve_variants: bool = True):
    """Approve one draft. With resolve_variants=True (default), any sibling
    drafts (same regenerated_from) that are still needs_review get
    auto-rejected as 'lost A/B comparison' so the brand memory loop sees
    the choice as a comparison, not isolated yes/no signals."""
    store = _store()
    pipe = _pipe(store)
    try:
        if resolve_variants:
            result = pipe.resolve_variant_winner(draft_id, reviewer=body.reviewer, reason=body.reason)
        else:
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


@router.patch("/drafts/{draft_id}/scheduled-for")
async def reschedule_draft(draft_id: str, body: DraftRescheduleBody):
    store = _store()
    try:
        result = store.update_draft_scheduled_for(draft_id, body.scheduled_for)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"result": result, "overview": _overview(store)}


@router.patch("/drafts/{draft_id}")
async def edit_draft(draft_id: str, body: DraftEditBody):
    store = _store()
    try:
        result = _pipe(store).edit_draft(draft_id, body.body, editor=body.editor)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"result": result, "overview": _overview(store)}


@router.post("/drafts/{draft_id}/variants")
async def make_variants(draft_id: str, count: int = 3):
    store = _store()
    try:
        result = _pipe(store).generate_variants(draft_id, count=count)
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


@router.get("/apps/{app_slug}/analytics")
async def app_analytics(app_slug: str, days: int = 30):
    store = _store()
    try:
        result = _pipe(store).app_analytics(app_slug, days=days)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return result


@router.get("/apps/{app_slug}/digest")
async def app_digest(app_slug: str, days: int = 7):
    store = _store()
    try:
        markdown = _pipe(store).weekly_digest(app_slug, days=days)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"app_slug": app_slug, "days": days, "markdown": markdown}


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


@router.post("/apps/{app_slug}/auto-generate")
async def set_auto_generate(app_slug: str, body: AutoGenerateBody):
    store = _store()
    try:
        result = store.set_auto_generate(app_slug, body.enabled, threshold=body.threshold, reviewer=body.reviewer)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
