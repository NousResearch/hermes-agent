"""Agent pipeline for Marketing Agent Factory.

Live LLM dispatch (cheap=local Ollama qwen2.5:14b, mid=qwen3:30b, premium=Claude
CLI/OAuth) is wired behind a feature gate. When LLM dispatch is disabled (tests,
or `MF_USE_LLM=0`), all agents fall through to the deterministic template path
they originally shipped with — so the pipeline remains testable and never breaks
when models are unreachable.

Real public posting is still hard-gated by the PublisherAgent's dry-run-only path.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from plugins.marketing_factory.connectors import (
    BaseChannelConnector,
    get_dry_run_connector,
    get_live_connector,
)
from plugins.marketing_factory.connectors.base import ConnectorError
from plugins.marketing_factory.model_dispatcher import dispatch, dispatch_json
from plugins.marketing_factory.store import MarketingFactoryStore, utc_now

logger = logging.getLogger(__name__)


def _should_use_llm() -> bool:
    """Live LLM dispatch is opt-out via `MF_USE_LLM=0`, and auto-off inside pytest."""
    if os.environ.get("MF_USE_LLM") == "0":
        return False
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    return True


PUPULAR_PROFILE: Dict[str, Any] = {
    "slug": "pupular",
    "name": "Pupular",
    "positioning": "A cute, modern adopt-a-pet app that helps people discover real adoptable pets.",
    "icp": "Pet lovers, adopters, rescue supporters, and families looking for a new companion.",
    "tone": "Cute, warm, optimistic, brand-safe, playful but never flippant about adoption.",
    "cta": "Download Pupular and meet adoptable pets near you.",
    "links": ["https://apps.apple.com/us/app/pupular-adopt-a-pet/id6761799693"],
    "channels": ["x", "instagram", "tiktok", "app_store"],
    "content_pillars": ["real adoptable pets", "adoption joy", "shelter support", "cute animal moments"],
    "claims": ["Helps people discover adoptable pets", "Available on the App Store"],
    "forbidden_claims": ["Guaranteed adoption", "medical or behavioral guarantees", "shelter affiliation unless verified"],
    "assets": ["real pet photos", "app screenshots", "short cute captions"],
    "competitors": ["Petfinder", "Adopt a Pet"],
    "current_campaigns": ["Four cute posts per day cadence"],
}

SETVENUE_PROFILE: Dict[str, Any] = {
    "slug": "setvenue",
    "name": "SetVenue",
    "positioning": "A marketplace for booking unique homes and spaces for productions, events, and creative work.",
    "icp": "Producers, photographers, creators, event planners, and hosts with bookable spaces.",
    "tone": "Trustworthy, practical, premium, clear, founder-led, conversion-focused.",
    "cta": "List your space or book a unique venue on SetVenue.",
    "links": ["https://setvenue.com"],
    "channels": ["linkedin", "x", "blog", "email"],
    "content_pillars": ["unique venues", "host earnings", "production logistics", "trust and booking flow"],
    "claims": ["Hosts can submit homes", "Bookings are finalized after host approval"],
    "forbidden_claims": ["Guaranteed income", "instant booking when host approval is required", "unverified customer counts"],
    "assets": ["venue screenshots", "listing examples", "booking flow visuals"],
    "competitors": ["Peerspace", "Giggster", "Airbnb for events"],
    "current_campaigns": ["Host acquisition and booker trust"],
}

SAMPLE_PROFILES = {"pupular": PUPULAR_PROFILE, "setvenue": SETVENUE_PROFILE}


class BrandBrainAgent:
    def __init__(self, store: MarketingFactoryStore):
        self.store = store

    def seed_samples(self) -> List[Dict[str, Any]]:
        return [self.store.upsert_app(profile) for profile in SAMPLE_PROFILES.values()]


class ResearchAgent:
    def research(
        self,
        app: Dict[str, Any],
        token_ledger: Optional[List[Dict[str, Any]]] = None,
        steering: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        pillars = app.get("content_pillars") or []
        channels = app.get("channels") or []

        base: Dict[str, Any] = {
            "generated_at": utc_now(),
            "agent": "research",
            "model_route": "cheap",
            "competitors": app.get("competitors", []),
            "trends": [f"{pillar} conversation hooks" for pillar in pillars[:3]],
            "pain_points": _pain_points(app["slug"]),
            "channel_opportunities": {channel: _channel_opportunity(app["slug"], channel) for channel in channels},
            "llm_used": False,
            "steering_applied": bool(steering),
        }

        if not _should_use_llm():
            return base

        steering_block = _format_steering_for_prompt(steering)
        prompt = (
            f"Brand: {app['name']} ({app['slug']})\n"
            f"Positioning: {app.get('positioning')}\n"
            f"ICP: {app.get('icp')}\n"
            f"Tone: {app.get('tone')}\n"
            f"Channels: {', '.join(channels)}\n"
            f"Content pillars: {', '.join(pillars)}\n"
            f"Forbidden claims: {', '.join(app.get('forbidden_claims', []))}\n"
            + (f"\n{steering_block}\n" if steering_block else "")
            + "\nProduce a single JSON object with EXACTLY these keys:\n"
            '  "trends": array of 3 short strings (current conversational hooks relevant to this brand)\n'
            '  "pain_points": array of 3 short strings (real audience pains this brand can address)\n'
            f'  "channel_opportunities": object mapping each of these channel keys exactly — {channels} — to a one-sentence opportunity for that channel\n'
            "Constraints: each string under 140 chars; avoid forbidden claims; no markdown; output ONLY valid JSON."
        )
        env = dispatch_json(
            "cheap",
            prompt,
            system="You are a marketing research analyst. You produce concise, brand-safe JSON.",
            max_tokens=600,
            temperature=0.3,
        )
        if token_ledger is not None and env.get("tokens_used"):
            token_ledger.append({
                "route": env["route"],
                "model": env["model"],
                "tokens": env["tokens_used"],
                "agent": "research",
                "channel": None,
                "elapsed_ms": env.get("elapsed_ms"),
            })
        if env.get("fallback_used") or not isinstance(env.get("parsed"), dict):
            base["llm_error"] = env.get("error")
            return base

        parsed = env["parsed"]
        if isinstance(parsed.get("trends"), list) and parsed["trends"]:
            base["trends"] = [str(t)[:200] for t in parsed["trends"][:5]]
        if isinstance(parsed.get("pain_points"), list) and parsed["pain_points"]:
            base["pain_points"] = [str(p)[:200] for p in parsed["pain_points"][:5]]
        if isinstance(parsed.get("channel_opportunities"), dict) and parsed["channel_opportunities"]:
            merged = dict(base["channel_opportunities"])
            for channel, value in parsed["channel_opportunities"].items():
                if channel in merged and isinstance(value, str):
                    merged[channel] = value[:300]
            base["channel_opportunities"] = merged
        base["llm_used"] = True
        base["llm_model"] = env.get("model")
        return base


class StrategyAgent:
    def plan_campaign(self, app: Dict[str, Any], research: Dict[str, Any], days: int = 7) -> Dict[str, Any]:
        objective = "app adoption and brand awareness" if app["slug"] == "pupular" else "host/booker acquisition and trust"
        plan = []
        base = datetime.now(timezone.utc).replace(hour=15, minute=0, second=0, microsecond=0)
        channels = app.get("channels") or ["x"]
        for idx in range(days):
            channel = channels[idx % len(channels)]
            pillar = (app.get("content_pillars") or ["brand story"])[idx % len(app.get("content_pillars") or ["brand story"])]
            plan.append({
                "day": idx + 1,
                "date": (base + timedelta(days=idx)).date().isoformat(),
                "channel": channel,
                "pillar": pillar,
                "angle": _angle_for(app["slug"], pillar),
                "scheduled_for": (base + timedelta(days=idx)).isoformat(),
            })
        return {
            "name": f"{app['name']} MVP Dry-Run Campaign",
            "objective": objective,
            "channels": channels,
            "research_summary": research,
            "plan": plan,
            "model_route": "premium",
        }


class CopyAgent:
    PREMIUM_CHANNELS = {"blog", "email", "linkedin"}
    MID_CHANNELS = {"x", "instagram", "tiktok", "app_store"}

    def _route_for(self, channel: str) -> str:
        if channel in self.PREMIUM_CHANNELS:
            return "premium"
        if channel in self.MID_CHANNELS:
            return "mid"
        return "cheap"

    def draft_for_item(
        self,
        app: Dict[str, Any],
        campaign_id: str,
        item: Dict[str, Any],
        token_ledger: Optional[List[Dict[str, Any]]] = None,
        steering: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        channel = item["channel"]
        route = self._route_for(channel)
        template_body = _draft_body(app, item)
        body = template_body
        llm_used = False
        llm_model = None
        llm_error = None

        if _should_use_llm():
            prompt = _copy_prompt(app, item, channel, steering=steering)
            env = dispatch(
                route,
                prompt,
                system=(
                    "You are a senior brand copywriter. Write channel-native copy that strictly "
                    "respects the brand voice, channel character limits, and forbidden-claim list. "
                    "Output ONLY the post body — no markdown headers, no preamble, no quotes around the post."
                ),
                max_tokens=_max_tokens_for(channel),
                temperature=0.55,
            )
            if token_ledger is not None and env.get("tokens_used"):
                token_ledger.append({
                    "route": env["route"],
                    "model": env["model"],
                    "tokens": env["tokens_used"],
                    "agent": "copy",
                    "channel": channel,
                    "elapsed_ms": env.get("elapsed_ms"),
                })
            if env.get("fallback_used") or not env.get("text"):
                llm_error = env.get("error") or "empty response"
            else:
                candidate = env["text"].strip()
                candidate = _strip_wrapping_quotes(candidate)
                if _within_channel_constraints(channel, candidate):
                    body = candidate
                    llm_used = True
                    llm_model = env.get("model")
                else:
                    llm_error = f"response violated channel constraints (len={len(candidate)})"

        draft: Dict[str, Any] = {
            "campaign_id": campaign_id,
            "channel": channel,
            "content_type": _content_type(channel),
            "body": body,
            "cta": app.get("cta"),
            "assets": _asset_concepts(app, item),
            "model_route": route,
            "llm_used": llm_used,
            "llm_model": llm_model,
        }
        if llm_error:
            draft["llm_error"] = llm_error
        return draft


class CreativeAgent:
    def add_concepts(self, app: Dict[str, Any], draft: Dict[str, Any]) -> Dict[str, Any]:
        draft = dict(draft)
        draft["creative_concepts"] = _asset_concepts(app, {"pillar": draft.get("content_type", "post"), "channel": draft["channel"]})
        return draft


class ReviewSafetyAgent:
    def review(self, app: Dict[str, Any], draft: Dict[str, Any]) -> Dict[str, Any]:
        body_lower = draft["body"].lower()
        forbidden_hits = [claim for claim in app.get("forbidden_claims", []) if claim.lower() in body_lower]
        channel_ok = _within_channel_constraints(draft["channel"], draft["body"])
        useful = len(draft["body"].strip()) >= 40
        passed = not forbidden_hits and channel_ok and useful
        return {
            "agent": "review_safety",
            "model_route": "premium",
            "passed": passed,
            "checks": {
                "brand_fit": True,
                "forbidden_claims": forbidden_hits,
                "channel_constraints": channel_ok,
                "spam_risk": "low",
                "duplicate_risk": "low",
                "hallucinated_claims_risk": "low" if not forbidden_hits else "medium",
                "useful": useful,
            },
            "recommendation": "needs_human_approval" if passed else "reject_or_rewrite",
        }


class SchedulerAgent:
    def schedule_approved(self, store: MarketingFactoryStore, app_slug: Optional[str] = None) -> List[Dict[str, Any]]:
        scheduled = []
        drafts = store.list_drafts(app_slug=app_slug, status="approved")
        base = datetime.now(timezone.utc).replace(hour=15, minute=0, second=0, microsecond=0)
        for idx, draft in enumerate(drafts):
            scheduled_for = draft.get("scheduled_for") or (base + timedelta(days=idx)).isoformat()
            scheduled.append(store.schedule_draft(draft["id"], scheduled_for))
        return scheduled


class PublisherAgent:
    def dry_run_publish_scheduled(self, store: MarketingFactoryStore, app_slug: Optional[str] = None) -> List[Dict[str, Any]]:
        """Force a dry-run publish regardless of brand-profile `channel_modes`.

        Kept for backward-compat with tests/dashboard/CLI and as the "preview"
        primitive — never posts publicly. For the channel-mode-aware path,
        callers should use `publish_scheduled`.
        """
        events = []
        for schedule in store.list_schedules(app_slug=app_slug):
            draft = store.get_draft(schedule["draft_id"], app_slug=schedule["app_slug"])
            if draft["status"] == "scheduled":
                events.append(store.dry_run_publish(draft["id"]))
        return events

    def publish_scheduled(
        self,
        store: MarketingFactoryStore,
        app_slug: Optional[str] = None,
        due_only: bool = False,
        now: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Respect each draft's `channel_modes[channel]`:
          - mode == "live" + a real connector registered → call connector.publish(draft)
          - mode == "live" + no connector → dry_run + audit fallback("no_live_connector")
          - mode == "live" + connector raises ConnectorError → dry_run + audit fallback(error)
          - mode == "dry_run" (default) → DryRunConnector

        Idempotent: drafts already in `posted` or `dry_run_posted` are skipped.

        When `due_only=True`, only schedules whose `scheduled_for` is <= `now`
        (default `datetime.now(utc)`) are published. This is the cron-poller path.
        """
        events: List[Dict[str, Any]] = []
        cutoff = now or datetime.now(timezone.utc)
        for schedule in store.list_schedules(app_slug=app_slug):
            draft = store.get_draft(schedule["draft_id"], app_slug=schedule["app_slug"])
            if draft["status"] != "scheduled":
                continue
            if due_only:
                scheduled_for_str = schedule.get("scheduled_for") or draft.get("scheduled_for")
                if not scheduled_for_str:
                    continue
                try:
                    scheduled_dt = datetime.fromisoformat(scheduled_for_str.replace("Z", "+00:00"))
                except (TypeError, ValueError):
                    continue
                if scheduled_dt.tzinfo is None:
                    scheduled_dt = scheduled_dt.replace(tzinfo=timezone.utc)
                if scheduled_dt > cutoff:
                    continue
            app = store.require_app(draft["app_slug"])
            channel = draft["channel"]
            mode = (app.get("channel_modes") or {}).get(channel, "dry_run")
            connector: Optional[BaseChannelConnector] = None
            fallback_reason: Optional[str] = None
            if mode == "live":
                connector = get_live_connector(channel)
                if connector is None:
                    fallback_reason = "no_live_connector"
            if connector is not None:
                try:
                    result = connector.publish(draft)
                except ConnectorError as exc:
                    fallback_reason = f"connector_error: {exc}"
                    connector = None
            if connector is None:
                result = get_dry_run_connector().publish(draft)
            events.append(store.record_publish_event(draft["id"], result, fallback_reason=fallback_reason))
        return events


class AnalyticsAgent:
    def feed_learning(self, store: MarketingFactoryStore, app_slug: str, summary: str) -> Dict[str, Any]:
        return store.record_analytics(app_slug, {"source": "dry_run", "summary": summary, "learning": summary})


class BrandMemoryAgent:
    """Distills approve/reject history into a steering blob the other agents read.

    Cached on `brand_memories[slug].steering`; the store invalidates it on every
    new approval/rejection so the next campaign generation re-summarizes.
    """

    MIN_LEARNINGS_TO_SUMMARIZE = 2

    def get_steering(
        self,
        store: MarketingFactoryStore,
        app_slug: str,
        token_ledger: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        state = store.load()
        memories = (state.get("brand_memories") or {}).get(app_slug) or {}
        cached = memories.get("steering")
        if cached:
            return cached

        learnings = [
            entry for entry in (memories.get("learnings") or [])
            if isinstance(entry, dict) and entry.get("kind") in {"draft_approved", "draft_rejected"}
        ]
        if len(learnings) < self.MIN_LEARNINGS_TO_SUMMARIZE:
            return None

        steering = self._fallback_steering(learnings)
        if _should_use_llm():
            llm_steering = self._llm_steering(app_slug, learnings, token_ledger=token_ledger)
            if llm_steering is not None:
                steering = llm_steering

        return store.write_steering(app_slug, steering)

    def _fallback_steering(self, learnings: List[Dict[str, Any]]) -> Dict[str, Any]:
        approved = [e for e in learnings if e.get("kind") == "draft_approved"]
        rejected = [e for e in learnings if e.get("kind") == "draft_rejected"]
        return {
            "generated_at": utc_now(),
            "method": "fallback",
            "what_works": [f"{e.get('channel')}: {e.get('reason') or 'approved'}" for e in approved[-3:]],
            "what_to_avoid": [f"{e.get('channel')}: {e.get('reason') or 'rejected'}" for e in rejected[-3:]],
            "tone_notes": [],
            "approved_count": len(approved),
            "rejected_count": len(rejected),
        }

    def _llm_steering(
        self,
        app_slug: str,
        learnings: List[Dict[str, Any]],
        token_ledger: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        recent = learnings[-25:]
        bullets = "\n".join(
            f"- [{e.get('kind','?').upper()}] channel={e.get('channel')} reason={e.get('reason')!r} excerpt={e.get('excerpt')!r}"
            for e in recent
        )
        prompt = (
            f"You are reviewing the prior approve/reject decisions for brand `{app_slug}` to "
            "extract steering signal for future copy generation.\n\n"
            f"Recent decisions ({len(recent)}):\n{bullets}\n\n"
            "Produce a single JSON object with these keys ONLY:\n"
            '  "what_works": array of up to 3 short strings — patterns the reviewer approves\n'
            '  "what_to_avoid": array of up to 3 short strings — patterns the reviewer rejects\n'
            '  "tone_notes": array of up to 2 short strings — voice/tone observations\n'
            "Each string under 140 chars. No markdown. JSON only."
        )
        env = dispatch_json(
            "cheap",
            prompt,
            system="You distill reviewer feedback into concise steering hints for marketing copy.",
            max_tokens=500,
            temperature=0.2,
        )
        if token_ledger is not None and env.get("tokens_used"):
            token_ledger.append({
                "route": env["route"],
                "model": env["model"],
                "tokens": env["tokens_used"],
                "agent": "brand_memory",
                "channel": None,
                "elapsed_ms": env.get("elapsed_ms"),
            })
        parsed = env.get("parsed")
        if env.get("fallback_used") or not isinstance(parsed, dict):
            return None
        return {
            "generated_at": utc_now(),
            "method": "llm",
            "model": env.get("model"),
            "what_works": [str(x)[:200] for x in (parsed.get("what_works") or [])[:5]],
            "what_to_avoid": [str(x)[:200] for x in (parsed.get("what_to_avoid") or [])[:5]],
            "tone_notes": [str(x)[:200] for x in (parsed.get("tone_notes") or [])[:5]],
            "approved_count": sum(1 for e in learnings if e.get("kind") == "draft_approved"),
            "rejected_count": sum(1 for e in learnings if e.get("kind") == "draft_rejected"),
        }


def _format_steering_for_prompt(steering: Optional[Dict[str, Any]]) -> str:
    if not steering:
        return ""
    works = steering.get("what_works") or []
    avoid = steering.get("what_to_avoid") or []
    tone = steering.get("tone_notes") or []
    if not (works or avoid or tone):
        return ""
    lines = ["Prior reviewer feedback to honor:"]
    if works:
        lines.append("  WHAT WORKS (do more of this):")
        lines.extend(f"    - {w}" for w in works)
    if avoid:
        lines.append("  WHAT TO AVOID (reviewer rejected these patterns):")
        lines.extend(f"    - {a}" for a in avoid)
    if tone:
        lines.append("  TONE NOTES:")
        lines.extend(f"    - {t}" for t in tone)
    return "\n".join(lines)


class MarketingFactoryPipeline:
    def __init__(self, store: Optional[MarketingFactoryStore] = None):
        self.store = store or MarketingFactoryStore()
        self.brand = BrandBrainAgent(self.store)
        self.research = ResearchAgent()
        self.strategy = StrategyAgent()
        self.copy = CopyAgent()
        self.creative = CreativeAgent()
        self.review = ReviewSafetyAgent()
        self.scheduler = SchedulerAgent()
        self.publisher = PublisherAgent()
        self.analytics = AnalyticsAgent()
        self.brand_memory = BrandMemoryAgent()

    def initialize_samples(self) -> Dict[str, Any]:
        self.store.initialize()
        apps = self.brand.seed_samples()
        return {"apps": apps, "summary": self.store.summary()}

    def generate_campaign(self, app_slug: str, days: int = 7, auto_approve: bool = False) -> Dict[str, Any]:
        app = self.store.require_app(app_slug)
        token_ledger: List[Dict[str, Any]] = []
        steering = self.brand_memory.get_steering(self.store, app["slug"], token_ledger=token_ledger)
        research = self.research.research(app, token_ledger=token_ledger, steering=steering)
        campaign_plan = self.strategy.plan_campaign(app, research, days=days)
        campaign = self.store.create_campaign(app["slug"], campaign_plan)
        drafts = []
        for item in campaign["plan"]:
            raw_draft = self.copy.draft_for_item(app, campaign["id"], item, token_ledger=token_ledger, steering=steering)
            raw_draft["scheduled_for"] = item["scheduled_for"]
            enriched = self.creative.add_concepts(app, raw_draft)
            safety = self.review.review(app, enriched)
            enriched["safety"] = safety
            enriched["status"] = "needs_review" if safety["passed"] else "rejected"
            draft = self.store.create_draft(app["slug"], enriched)
            if auto_approve and safety["passed"]:
                self.store.set_approval(draft["id"], "approved", reviewer="auto-test", reason="auto approval for dry-run verification only")
                draft = self.store.get_draft(draft["id"])
            drafts.append(draft)
        token_summary = self.store.record_token_usage(app["slug"], campaign["id"], token_ledger) if token_ledger else None
        self.store.audit(
            "campaign.generated",
            app["slug"],
            {
                "campaign_id": campaign["id"],
                "draft_count": len(drafts),
                "auto_approve": auto_approve,
                "tokens_used": (token_summary or {}).get("tokens_used", 0),
                "llm_calls": len(token_ledger),
            },
        )
        return {"app": app, "campaign": campaign, "drafts": drafts, "token_summary": token_summary}

    def approve_all_for_app(self, app_slug: str, reviewer: str = "human") -> List[Dict[str, Any]]:
        approvals = []
        for draft in self.store.list_drafts(app_slug=app_slug, status="needs_review"):
            if draft.get("safety", {}).get("passed"):
                approvals.append(self.store.set_approval(draft["id"], "approved", reviewer=reviewer, reason="approved for dry-run scheduling"))
        return approvals

    def advise(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        """Run a set of health checks and return actionable items.

        Each item: {severity, app_slug?, message, action}. Severity is
        "warning" (likely a misconfiguration the user should fix) or "info"
        (background observation that's not urgent but worth knowing).

        Healthy state → items=[], healthy=True.
        """
        from plugins.marketing_factory import connectors as _connectors

        now_dt = now or datetime.now(timezone.utc)
        state = self.store.load()
        items: List[Dict[str, Any]] = []

        # Check 1: channel_modes says "live" but no real connector registered.
        for app in state.get("apps", {}).values():
            modes = app.get("channel_modes") or {}
            for channel, mode in modes.items():
                if mode != "live":
                    continue
                if _connectors.get_live_connector(channel) is None:
                    items.append({
                        "severity": "warning",
                        "app_slug": app["slug"],
                        "message": f"{app['slug']}.{channel} is set to live but no connector is registered — publish_scheduled will silently fall back to dry_run.",
                        "action": f"Wire connectors/{channel}_stub.py and register it in connectors/__init__.py, or flip {channel} back to dry_run.",
                    })

        # Check 2: app with no campaign in 7+ days.
        seven_days_ago = now_dt - timedelta(days=7)
        campaigns_by_app: Dict[str, datetime] = {}
        for camp in state.get("campaigns", {}).values():
            try:
                created = datetime.fromisoformat(str(camp.get("created_at") or "").replace("Z", "+00:00"))
            except ValueError:
                continue
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            existing = campaigns_by_app.get(camp.get("app_slug"))
            if existing is None or created > existing:
                campaigns_by_app[camp.get("app_slug")] = created
        for slug, app in state.get("apps", {}).items():
            last_camp = campaigns_by_app.get(slug)
            if last_camp is None:
                items.append({
                    "severity": "info",
                    "app_slug": slug,
                    "message": f"{slug} has never had a campaign generated.",
                    "action": f"`hermes marketing-factory generate --app {slug} --days 7` to seed the dry-run queue.",
                })
            elif last_camp < seven_days_ago:
                items.append({
                    "severity": "info",
                    "app_slug": slug,
                    "message": f"{slug}'s most recent campaign is {(now_dt - last_camp).days} day(s) old.",
                    "action": f"`hermes marketing-factory generate --app {slug} --days 7` to refresh the dry-run queue.",
                })

        # Check 3: pending-approval queue backing up.
        pending_by_app: Dict[str, int] = {}
        for draft in state.get("drafts", {}).values():
            if draft.get("status") == "needs_review":
                pending_by_app[draft["app_slug"]] = pending_by_app.get(draft["app_slug"], 0) + 1
        for slug, count in pending_by_app.items():
            if count >= 14:
                items.append({
                    "severity": "warning",
                    "app_slug": slug,
                    "message": f"{slug} has {count} drafts pending review.",
                    "action": "Open /marketing-factory dashboard and clear the queue (bulk approve, or reject with reasons for steering).",
                })

        # Check 4: token spend approaching daily cap.
        budgets = state.get("budgets") or {}
        daily_cap = budgets.get("daily_tokens") or 0
        spent = budgets.get("spent_tokens_today") or 0
        if daily_cap and spent >= int(daily_cap * 0.8):
            pct = round(100 * spent / daily_cap)
            items.append({
                "severity": "warning" if pct >= 90 else "info",
                "app_slug": None,
                "message": f"Daily token spend at {pct}% of cap ({spent:,} / {daily_cap:,}).",
                "action": "Bump daily_tokens in state.budgets or pause new generation until tomorrow (UTC).",
            })

        # Check 5: scheduled poller hasn't ticked recently.
        poll_state = state.get("poll") or {}
        last_poll_str = poll_state.get("last_poll_at")
        any_schedules = bool(state.get("schedules"))
        if last_poll_str:
            try:
                last_poll = datetime.fromisoformat(str(last_poll_str).replace("Z", "+00:00"))
                if last_poll.tzinfo is None:
                    last_poll = last_poll.replace(tzinfo=timezone.utc)
                if any_schedules and (now_dt - last_poll) > timedelta(hours=2):
                    minutes_since = int((now_dt - last_poll).total_seconds() / 60)
                    items.append({
                        "severity": "warning",
                        "app_slug": None,
                        "message": f"Scheduled poller has not ticked in {minutes_since} minutes (last: {last_poll_str}).",
                        "action": "Check `hermes cron list` for the marketing-factory-poll job, or run `hermes marketing-factory enable-poller` to (re)install it.",
                    })
            except ValueError:
                pass
        elif any_schedules:
            items.append({
                "severity": "info",
                "app_slug": None,
                "message": "There are scheduled drafts but the poller has never ticked.",
                "action": "Run `hermes marketing-factory enable-poller` to install the autonomous loop.",
            })

        return {"items": items, "healthy": not items, "checked_at": now_dt.isoformat()}

    def poll(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        """One tick of the scheduled poller. Walks every app, fires
        `publish_scheduled(due_only=True)`, records the result on
        `state.poll`. Designed to be called from a cron job:
            hermes cron create --schedule "every 5m" --command "hermes marketing-factory poll"

        Returns: {polled_apps, due_count, fired_count, events, last_poll}
        """
        cutoff = now or datetime.now(timezone.utc)
        apps = self.store.list_apps()
        all_events: List[Dict[str, Any]] = []
        # Count drafts that are due across all apps (for reporting, includes
        # ones we did not fire because they were already non-scheduled).
        due_count = 0
        for app in apps:
            schedules = self.store.list_schedules(app_slug=app["slug"])
            for schedule in schedules:
                scheduled_for_str = schedule.get("scheduled_for") or ""
                try:
                    sched_dt = datetime.fromisoformat(scheduled_for_str.replace("Z", "+00:00"))
                except (TypeError, ValueError):
                    continue
                if sched_dt.tzinfo is None:
                    sched_dt = sched_dt.replace(tzinfo=timezone.utc)
                if sched_dt <= cutoff:
                    due_count += 1
            events = self.publisher.publish_scheduled(self.store, app_slug=app["slug"], due_only=True, now=cutoff)
            all_events.extend(events)
        last_poll = self.store.record_poll(fired=len(all_events), due=due_count, polled_apps=len(apps))
        return {
            "polled_apps": len(apps),
            "due_count": due_count,
            "fired_count": len(all_events),
            "events": all_events,
            "last_poll": last_poll,
        }

    def run_full_dry_run(self, app_slug: str, days: int = 7, reviewer: str = "human") -> Dict[str, Any]:
        generated = self.generate_campaign(app_slug, days=days, auto_approve=False)
        approvals = self.approve_all_for_app(app_slug, reviewer=reviewer)
        schedules = self.scheduler.schedule_approved(self.store, app_slug=app_slug)
        publish_events = self.publisher.dry_run_publish_scheduled(self.store, app_slug=app_slug)
        learning = self.analytics.feed_learning(
            self.store,
            app_slug,
            f"Dry-run campaign generated {len(generated['drafts'])} drafts, scheduled {len(schedules)}, and produced {len(publish_events)} dry-run publish events.",
        )
        return {"generated": generated, "approvals": approvals, "schedules": schedules, "publish_events": publish_events, "analytics": learning}


def _pain_points(slug: str) -> List[str]:
    if slug == "pupular":
        return ["Finding adoptable pets feels fragmented", "People want cute, trustworthy pet discovery", "Shelters need more attention for real pets"]
    if slug == "setvenue":
        return ["Producers need unique spaces without endless back-and-forth", "Hosts need trust and clear booking expectations", "Bookers want to know approval and payment timing"]
    return ["Audience needs clearer education", "Brand needs consistent channel-native content"]


def _channel_opportunity(slug: str, channel: str) -> str:
    if slug == "pupular":
        return f"Use {channel} for cute, real-pet discovery hooks with App Store CTA."
    if slug == "setvenue":
        return f"Use {channel} for trust-building venue education and host/booker conversion."
    return f"Use {channel} for brand-safe useful content."


def _angle_for(slug: str, pillar: str) -> str:
    if slug == "pupular":
        return f"Make {pillar} feel immediate, cute, and adoption-positive."
    if slug == "setvenue":
        return f"Explain {pillar} with practical trust and booking clarity."
    return f"Turn {pillar} into a clear audience benefit."


def _content_type(channel: str) -> str:
    return {"blog": "seo_outline", "email": "email", "linkedin": "linkedin_post", "x": "short_social", "instagram": "visual_caption", "tiktok": "short_script", "app_store": "app_store_copy"}.get(channel, "post")


_CHANNEL_GUIDANCE: Dict[str, str] = {
    "x": "Single X/Twitter post, max 280 chars. Hook in first 8 words. Lowercase ok. No hashtag spam. At most ONE link.",
    "instagram": "Instagram caption, 1-3 short paragraphs, max 2200 chars. Optional 3-5 niche hashtags on a final line.",
    "tiktok": "TikTok short script — 6 to 12 spoken seconds. Format as: HOOK | LINE | LINE | CTA. No emojis.",
    "linkedin": "LinkedIn post, conversational founder voice, 3-6 short paragraphs, max 1300 chars. No hashtag spam.",
    "blog": "SEO blog OUTLINE: H1 + 3-5 H2 sections + one-sentence intro for each. Max 1500 chars total.",
    "email": "Plain-text email body, conversational, max 1200 chars. Subject line ON ITS OWN FIRST LINE prefixed 'Subject:'.",
    "app_store": "App Store promotional text, max 170 chars, conversion-focused.",
}


_CHANNEL_MAX_TOKENS: Dict[str, int] = {
    "x": 200,
    "instagram": 700,
    "tiktok": 250,
    "linkedin": 600,
    "blog": 900,
    "email": 600,
    "app_store": 140,
}


def _copy_prompt(app: Dict[str, Any], item: Dict[str, Any], channel: str, steering: Optional[Dict[str, Any]] = None) -> str:
    link = (app.get("links") or [""])[0]
    pillar = item.get("pillar") or "brand story"
    angle = item.get("angle") or _angle_for(app["slug"], pillar)
    forbidden = app.get("forbidden_claims") or []
    claims_ok = app.get("claims") or []
    steering_block = _format_steering_for_prompt(steering)
    return (
        f"Brand: {app['name']}\n"
        f"Positioning: {app.get('positioning')}\n"
        f"Audience (ICP): {app.get('icp')}\n"
        f"Tone: {app.get('tone')}\n"
        f"Channel: {channel}\n"
        f"Content pillar today: {pillar}\n"
        f"Angle: {angle}\n"
        f"CTA: {app.get('cta')}\n"
        f"Primary link: {link}\n"
        f"Approved claims (only use these or paraphrases): {claims_ok}\n"
        f"Forbidden claims (do NOT make or imply these): {forbidden}\n"
        + (f"\n{steering_block}\n" if steering_block else "")
        + f"\nChannel rules: {_CHANNEL_GUIDANCE.get(channel, 'Brand-safe post.')}\n\n"
        "Write the post body NOW. Output only the body text — no quotes, no preamble, no commentary."
    )


def _max_tokens_for(channel: str) -> int:
    return _CHANNEL_MAX_TOKENS.get(channel, 400)


def _strip_wrapping_quotes(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] in {'"', "'", "“", "‘"} and text[-1] in {'"', "'", "”", "’"}:
        return text[1:-1].strip()
    return text


def _draft_body(app: Dict[str, Any], item: Dict[str, Any]) -> str:
    link = app.get("links", [""])[0]
    if app["slug"] == "pupular":
        return (
            f"Tiny paws, huge main-character energy. Today’s {item['pillar']} reminder: adoptable pets are out there waiting to be discovered. "
            f"Open Pupular, meet real pets, and maybe find your next best friend. {link}"
        )
    if app["slug"] == "setvenue":
        return (
            f"Great shoots and events start with the right space. SetVenue helps teams discover unique homes and venues while keeping booking expectations clear: "
            f"requests go to hosts, and payment is finalized after host approval. {link}"
        )
    return f"{app['name']} helps {app.get('icp', 'its audience')} with {item['pillar']}. {app.get('cta', '')} {link}".strip()


def _asset_concepts(app: Dict[str, Any], item: Dict[str, Any]) -> List[str]:
    if app["slug"] == "pupular":
        return ["Real adoptable pet photo with soft rounded caption card", "Short vertical clip prompt: happy pet close-up + App Store CTA"]
    if app["slug"] == "setvenue":
        return ["Premium venue screenshot carousel: space, use case, booking clarity", "Short explainer graphic: request → host approval → finalized booking"]
    return [f"Brand-safe visual concept for {item.get('pillar', 'campaign')}"]


def _within_channel_constraints(channel: str, body: str) -> bool:
    if channel == "x":
        return len(body) <= 280
    if channel in {"linkedin", "email", "blog"}:
        return len(body) <= 5000
    return len(body) <= 2200
