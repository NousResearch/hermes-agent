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
from plugins.marketing_factory.progress_bus import publish as _publish_progress
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
    VISUAL_CHANNELS = {"instagram", "tiktok", "app_store"}

    def __init__(self, image_gen: Optional["ImageGenAgent"] = None) -> None:
        self.image_gen = image_gen or ImageGenAgent()

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
        # Auto-generate an image for visual channels.
        if channel in self.VISUAL_CHANNELS and self.image_gen.is_enabled():
            image_result = self.image_gen.generate(app, item, body, token_ledger=token_ledger)
            if image_result.get("url"):
                draft["images"] = [image_result]
        return draft


class CreativeAgent:
    def add_concepts(self, app: Dict[str, Any], draft: Dict[str, Any]) -> Dict[str, Any]:
        draft = dict(draft)
        draft["creative_concepts"] = _asset_concepts(app, {"pillar": draft.get("content_type", "post"), "channel": draft["channel"]})
        return draft


class ImageGenAgent:
    """Generate an image URL for visual-channel drafts.

    Default backend is Pollinations.ai (free, no auth, wraps Flux/SDXL). Set
    `MF_IMAGE_BACKEND=disabled` to skip image generation entirely; set to
    other future-supported backends ("dalle", "local-sdxl") to swap.

    Returns: {kind: "image_prompt", url, prompt, model, fallback_used, error,
              backend, elapsed_ms}.
    """

    POLLINATIONS_BASE = "https://image.pollinations.ai/prompt/"

    def __init__(self) -> None:
        self.backend = os.environ.get("MF_IMAGE_BACKEND", "pollinations").lower()

    def is_enabled(self) -> bool:
        # Default OFF as of 2026-05: Pollinations free tier now caps at 1
        # concurrent request per IP and returns HTTP 402 on overflow, which
        # breaks the campaign-time burst of 6-7 parallel image requests.
        # User opts back in via MF_AUTO_IMAGES=1 once they have either a
        # paid Pollinations API key OR we plug in a local SDXL backend.
        return self.backend != "disabled" and os.environ.get("MF_AUTO_IMAGES", "0") == "1"

    def generate(
        self,
        app: Dict[str, Any],
        item: Dict[str, Any],
        body: str,
        *,
        token_ledger: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not self.is_enabled():
            return {"kind": "image_prompt", "url": None, "prompt": None, "fallback_used": True, "error": "image gen disabled", "backend": self.backend}
        prompt = self._build_prompt(app, item, body, token_ledger=token_ledger)
        if self.backend == "pollinations":
            url = self._pollinations_url(prompt)
            return {
                "kind": "image_prompt",
                "url": url,
                "prompt": prompt,
                "model": "flux",
                "fallback_used": False,
                "error": None,
                "backend": "pollinations",
            }
        return {"kind": "image_prompt", "url": None, "prompt": prompt, "fallback_used": True, "error": f"unsupported backend: {self.backend}", "backend": self.backend}

    def _build_prompt(
        self,
        app: Dict[str, Any],
        item: Dict[str, Any],
        body: str,
        *,
        token_ledger: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        # Fast deterministic fallback. Used when LLM is off OR when LLM fails.
        fallback = _default_image_prompt(app, item, body)
        if not _should_use_llm():
            return fallback
        synth_prompt = (
            f"Brand: {app['name']} ({app.get('positioning')})\n"
            f"Channel: {item.get('channel')}\n"
            f"Caption being illustrated: {body[:400]}\n\n"
            "Write a SINGLE image generation prompt (under 60 words) that would make a great visual for this post. "
            "Be visual: subject, framing, mood, lighting, color palette. NO references to text overlays or logos. "
            "No quotes, no preamble — output ONLY the prompt."
        )
        env = dispatch(
            "cheap",
            synth_prompt,
            system="You are an art director who writes vivid, concrete image generation prompts.",
            max_tokens=180,
            temperature=0.6,
        )
        if token_ledger is not None and env.get("tokens_used"):
            token_ledger.append({
                "route": env["route"],
                "model": env["model"],
                "tokens": env["tokens_used"],
                "agent": "image_gen",
                "channel": item.get("channel"),
                "elapsed_ms": env.get("elapsed_ms"),
            })
        if env.get("fallback_used") or not env.get("text"):
            return fallback
        return env["text"].strip().strip('"').strip("'")[:400]

    def _pollinations_url(self, prompt: str) -> str:
        from urllib.parse import quote
        encoded = quote(prompt[:400], safe="")
        # Pollinations changed their access tiers — `flux` now returns HTTP 402
        # (paid model). `sana` is the only currently-free model and renders in
        # ~1s. Overridable via MF_POLLINATIONS_MODEL for users with a paid key.
        import os as _os
        model = _os.environ.get("MF_POLLINATIONS_MODEL", "sana")
        # 768x768 is fast and renders well at dashboard preview sizes.
        url = f"{self.POLLINATIONS_BASE}{encoded}?model={model}&width=768&height=768&nologo=true"
        # Fire-and-forget prewarm: trigger Pollinations to render+cache this
        # image in a background thread so by the time the user's browser asks
        # for it, the response is a fast Cloudflare cache hit. Without this,
        # the browser's first GET often times out during Pollinations
        # cold-start (30-90s of Flux compute) while a patient curl eventually
        # succeeds — producing "image unavailable" in the dashboard for
        # otherwise-valid URLs.
        self._prewarm(url)
        return url

    def _prewarm(self, url: str) -> None:
        import threading
        import httpx as _httpx
        def _ping():
            try:
                with _httpx.Client(timeout=180.0) as c:
                    c.get(url)
            except Exception:
                pass
        threading.Thread(target=_ping, daemon=True).start()


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
        _publish_progress("campaign.start", app_slug=app["slug"], days=days)
        token_ledger: List[Dict[str, Any]] = []

        _publish_progress("agent.start", agent="brand_memory", detail=f"Reading prior feedback for {app['slug']}")
        steering = self.brand_memory.get_steering(self.store, app["slug"], token_ledger=token_ledger)
        _publish_progress("agent.end", agent="brand_memory", detail=("Steering loaded · " + steering.get("method", "n/a") if steering else "No steering yet"))

        _publish_progress("agent.start", agent="research", detail=f"Extracting trends for {app['name']}")
        research = self.research.research(app, token_ledger=token_ledger, steering=steering)
        _publish_progress("agent.end", agent="research", detail=f"{len(research.get('trends', []))} trends · {len(research.get('pain_points', []))} pain points")

        _publish_progress("agent.start", agent="strategy", detail=f"Planning {days}-day campaign")
        campaign_plan = self.strategy.plan_campaign(app, research, days=days)
        _publish_progress("agent.end", agent="strategy", detail=f"{len(campaign_plan.get('plan', []))} slots across {len(campaign_plan.get('channels', []))} channels")

        campaign = self.store.create_campaign(app["slug"], campaign_plan)
        drafts = []
        for idx, item in enumerate(campaign["plan"], start=1):
            channel = item.get("channel")
            slot_label = f"day {idx}/{len(campaign['plan'])} · {channel}"

            _publish_progress("agent.start", agent="copy", channel=channel, detail=f"Writing {channel} post — {slot_label}")
            raw_draft = self.copy.draft_for_item(app, campaign["id"], item, token_ledger=token_ledger, steering=steering)
            raw_draft["scheduled_for"] = item["scheduled_for"]
            _publish_progress("agent.end", agent="copy", channel=channel, detail=f"{len(raw_draft.get('body') or '')} chars · {raw_draft.get('llm_model') or 'template'}")

            # ImageGen ran inside copy (only for visual channels); publish a snapshot event
            if raw_draft.get("images"):
                img = raw_draft["images"][0]
                _publish_progress("agent.start", agent="image_gen", channel=channel, detail=f"Rendering image for {slot_label}")
                _publish_progress("agent.end", agent="image_gen", channel=channel, detail=f"{img.get('backend', '?')} · {(img.get('prompt') or '')[:80]}")

            enriched = self.creative.add_concepts(app, raw_draft)

            _publish_progress("agent.start", agent="safety", channel=channel, detail=f"Checking {channel} body against forbidden claims")
            safety = self.review.review(app, enriched)
            _publish_progress("agent.end", agent="safety", channel=channel, detail="passed" if safety["passed"] else f"FAIL · {safety.get('recommendation')}")

            enriched["safety"] = safety
            enriched["status"] = "needs_review" if safety["passed"] else "rejected"
            freshness = _compute_freshness(self.store, app["slug"], item["channel"], enriched["body"])
            enriched["freshness_score"] = freshness["score"]
            enriched["freshness_compared_against"] = freshness["compared_against"]
            enriched["freshness_most_similar_id"] = freshness["most_similar_id"]
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
        _publish_progress(
            "campaign.end",
            app_slug=app["slug"],
            draft_count=len(drafts),
            tokens_used=(token_summary or {}).get("tokens_used", 0),
            llm_calls=len(token_ledger),
        )
        return {"app": app, "campaign": campaign, "drafts": drafts, "token_summary": token_summary}

    def approve_all_for_app(self, app_slug: str, reviewer: str = "human") -> List[Dict[str, Any]]:
        approvals = []
        for draft in self.store.list_drafts(app_slug=app_slug, status="needs_review"):
            if draft.get("safety", {}).get("passed"):
                approvals.append(self.store.set_approval(draft["id"], "approved", reviewer=reviewer, reason="approved for dry-run scheduling"))
        return approvals

    def app_analytics(self, app_slug: str, days: int = 30, now: Optional[datetime] = None) -> Dict[str, Any]:
        """Per-app performance rollup over the last `days` days.

        Returns approval_rate, by-channel stats, average freshness, average
        body length, draft status counts, tokens spent on this app. Counts
        only drafts created within the window so old activity doesn't drown
        out current trends.
        """
        app = self.store.require_app(app_slug)
        cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=days)
        state = self.store.load()

        def _within(record: Dict[str, Any]) -> bool:
            try:
                dt = datetime.fromisoformat(str(record.get("created_at") or "").replace("Z", "+00:00"))
            except ValueError:
                return False
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt >= cutoff

        drafts = [d for d in state.get("drafts", {}).values() if d.get("app_slug") == app["slug"] and _within(d)]
        approved = [d for d in drafts if d.get("status") == "approved"]
        rejected = [d for d in drafts if d.get("status") == "rejected"]
        posted = [d for d in drafts if d.get("status") in {"posted", "dry_run_posted"}]
        needs_review = [d for d in drafts if d.get("status") == "needs_review"]
        scheduled = [d for d in drafts if d.get("status") == "scheduled"]

        decided = len(approved) + len(rejected)
        approval_rate = round(len(approved) / decided, 3) if decided else None

        freshness_scores = [d["freshness_score"] for d in drafts if isinstance(d.get("freshness_score"), (int, float))]
        avg_freshness = round(sum(freshness_scores) / len(freshness_scores), 3) if freshness_scores else None
        body_lengths = [len(d.get("body") or "") for d in drafts]
        avg_body_length = round(sum(body_lengths) / len(body_lengths), 1) if body_lengths else None

        by_channel: Dict[str, Dict[str, Any]] = {}
        for channel in app.get("channels") or []:
            ch_drafts = [d for d in drafts if d.get("channel") == channel]
            ch_approved = [d for d in ch_drafts if d.get("status") == "approved"]
            ch_rejected = [d for d in ch_drafts if d.get("status") == "rejected"]
            ch_decided = len(ch_approved) + len(ch_rejected)
            ch_fresh = [d["freshness_score"] for d in ch_drafts if isinstance(d.get("freshness_score"), (int, float))]
            by_channel[channel] = {
                "total": len(ch_drafts),
                "approved": len(ch_approved),
                "rejected": len(ch_rejected),
                "needs_review": sum(1 for d in ch_drafts if d.get("status") == "needs_review"),
                "posted": sum(1 for d in ch_drafts if d.get("status") in {"posted", "dry_run_posted"}),
                "approval_rate": round(len(ch_approved) / ch_decided, 3) if ch_decided else None,
                "avg_freshness": round(sum(ch_fresh) / len(ch_fresh), 3) if ch_fresh else None,
                "mode": (app.get("channel_modes") or {}).get(channel, "dry_run"),
            }

        tokens_for_app = (state.get("budgets") or {}).get("per_app_tokens", {}).get(app["slug"], 0)

        return {
            "app_slug": app["slug"],
            "period_days": days,
            "checked_at": (now or datetime.now(timezone.utc)).isoformat(),
            "total_drafts": len(drafts),
            "by_status": {
                "needs_review": len(needs_review),
                "approved": len(approved),
                "rejected": len(rejected),
                "scheduled": len(scheduled),
                "posted_or_dry_run": len(posted),
            },
            "approval_rate": approval_rate,
            "avg_freshness": avg_freshness,
            "avg_body_length": avg_body_length,
            "by_channel": by_channel,
            "tokens_spent_per_app_today": tokens_for_app,
            "auto_generate": bool(app.get("auto_generate")),
            "auto_generate_threshold": int(app.get("auto_generate_threshold") or 3),
        }

    def weekly_digest(self, app_slug: str, days: int = 7, now: Optional[datetime] = None) -> str:
        """Markdown digest of one app's activity over the last `days` days.

        Includes campaign + draft counts by status, approved-drafts list
        with body previews + channel, rejected-drafts list with reasons
        (so the user can see what the factory's learning), the current
        steering snapshot, and per-channel/per-route token spend.
        """
        app = self.store.require_app(app_slug)
        cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=days)
        state = self.store.load()

        def _belongs(record: Dict[str, Any]) -> bool:
            return record.get("app_slug") == app["slug"]

        def _within_window(record: Dict[str, Any]) -> bool:
            try:
                created = datetime.fromisoformat(str(record.get("created_at") or "").replace("Z", "+00:00"))
            except ValueError:
                return False
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            return created >= cutoff

        campaigns = [c for c in state.get("campaigns", {}).values() if _belongs(c) and _within_window(c)]
        drafts = [d for d in state.get("drafts", {}).values() if _belongs(d) and _within_window(d)]

        status_counts: Dict[str, int] = {}
        for d in drafts:
            status_counts[d.get("status", "unknown")] = status_counts.get(d.get("status", "unknown"), 0) + 1

        approved = sorted([d for d in drafts if d.get("status") == "approved"], key=lambda d: d.get("created_at") or "")
        rejected = sorted([d for d in drafts if d.get("status") == "rejected"], key=lambda d: d.get("created_at") or "")
        posted = sorted([d for d in drafts if d.get("status") in {"posted", "dry_run_posted"}], key=lambda d: d.get("created_at") or "")

        steering = (state.get("brand_memories", {}).get(app["slug"]) or {}).get("steering") or {}
        budgets = state.get("budgets") or {}
        per_app = (budgets.get("per_app_tokens") or {}).get(app["slug"], 0)

        def _truncate(text: str, n: int = 200) -> str:
            text = (text or "").strip()
            return text if len(text) <= n else text[:n] + "…"

        lines: List[str] = []
        lines.append(f"# {app.get('name') or app['slug']} — weekly digest")
        lines.append("")
        lines.append(f"_Period: last {days} day(s) (since {cutoff.date().isoformat()} UTC)_")
        lines.append("")
        lines.append(f"**Positioning**: {app.get('positioning') or '—'}")
        lines.append(f"**ICP**: {app.get('icp') or '—'}")
        lines.append(f"**Channels**: {', '.join(app.get('channels') or []) or '—'}")
        lines.append("")
        lines.append("## Activity")
        lines.append(f"- Campaigns generated: **{len(campaigns)}**")
        lines.append(f"- Drafts created: **{len(drafts)}**")
        if status_counts:
            lines.append("- Status breakdown: " + ", ".join(f"`{status}`: {count}" for status, count in sorted(status_counts.items())))
        lines.append(f"- Tokens spent on this brand (all-time per current daily window): **{per_app:,}**")
        lines.append("")
        if posted:
            lines.append(f"## Published / dry-run published ({len(posted)})")
            for d in posted:
                marker = "🟢 LIVE" if d.get("status") == "posted" else "⚪ dry-run"
                lines.append(f"- {marker} `{d.get('channel')}` — {_truncate(d.get('body') or '', 220)}")
            lines.append("")
        if approved:
            lines.append(f"## Approved, awaiting schedule/publish ({len(approved)})")
            for d in approved:
                lines.append(f"- `{d.get('channel')}` — {_truncate(d.get('body') or '', 220)}")
            lines.append("")
        if rejected:
            lines.append(f"## Rejected — the factory's learning signal ({len(rejected)})")
            for d in rejected:
                reason = (state.get("approvals", {}).get(d["id"], {}) or {}).get("reason") or "no reason recorded"
                lines.append(f"- `{d.get('channel')}` rejected — reason: _{reason}_")
                lines.append(f"  - Body excerpt: {_truncate(d.get('body') or '', 160)}")
            lines.append("")
        if steering:
            lines.append("## Current brand steering (applied to future generations)")
            works = steering.get("what_works") or []
            avoid = steering.get("what_to_avoid") or []
            tone = steering.get("tone_notes") or []
            if works:
                lines.append("**What works:**")
                lines.extend(f"- {w}" for w in works)
                lines.append("")
            if avoid:
                lines.append("**What to avoid:**")
                lines.extend(f"- {a}" for a in avoid)
                lines.append("")
            if tone:
                lines.append("**Tone notes:**")
                lines.extend(f"- {t}" for t in tone)
                lines.append("")
        else:
            lines.append("## Brand steering")
            lines.append("_No steering yet — approve/reject some drafts (with reasons) to start the learning loop._")
            lines.append("")

        lines.append("---")
        lines.append(f"_Generated by Hermes marketing_factory at {(now or datetime.now(timezone.utc)).isoformat()}_")
        return "\n".join(lines)

    def resolve_variant_winner(self, draft_id: str, reviewer: str = "human", reason: Optional[str] = None) -> Dict[str, Any]:
        """Approve a draft AND auto-reject its needs_review siblings.

        Two drafts are "siblings" when they share a `regenerated_from` value
        (i.e., they were generated as variants of the same source). When the
        reviewer picks one, the others are implicitly losers — we record that
        as a structured rejection so the brand memory steering loop sees the
        comparison, not just isolated yes/no decisions.

        Returns: {approved: <record>, auto_rejected: [<record>, ...]}.
        Idempotent: if no siblings exist, behaves identically to plain approve.
        """
        winner = self.store.get_draft(draft_id)
        approved = self.store.set_approval(draft_id, "approved", reviewer=reviewer, reason=reason or "approved")
        source_id = winner.get("regenerated_from")
        auto_rejected: List[Dict[str, Any]] = []
        if source_id:
            for candidate in self.store.list_drafts(app_slug=winner["app_slug"], status="needs_review"):
                if candidate["id"] == draft_id:
                    continue
                if candidate.get("regenerated_from") != source_id:
                    continue
                rejection = self.store.set_approval(
                    candidate["id"], "rejected",
                    reviewer=reviewer,
                    reason=f"lost A/B comparison — reviewer chose draft {draft_id}",
                )
                auto_rejected.append(rejection)
        if auto_rejected:
            self.store.audit("variants.resolved", winner["app_slug"], {
                "winner_id": draft_id,
                "source_id": source_id,
                "losers": [r["draft_id"] for r in auto_rejected],
                "reviewer": reviewer,
            })
        return {"approved": approved, "auto_rejected": auto_rejected, "loser_count": len(auto_rejected)}

    def edit_draft(self, draft_id: str, body: str, *, editor: str = "human") -> Dict[str, Any]:
        """Edit a draft's body in place. Re-runs ReviewSafetyAgent on the new body
        and persists everything via store.update_draft_body."""
        draft = self.store.get_draft(draft_id)
        app = self.store.require_app(draft["app_slug"])
        # Re-run safety on a hypothetical draft with the new body
        candidate = {**draft, "body": body}
        safety = self.review.review(app, candidate)
        return self.store.update_draft_body(draft_id, body, editor=editor, safety=safety)

    def generate_variants(self, draft_id: str, count: int = 3) -> Dict[str, Any]:
        """Produce `count` alternative drafts for the same plan item.

        Each call is a fresh `regenerate_draft`, so steering + LLM stochasticity
        gives genuinely distinct options. The reviewer can then pick the best
        of N without manual re-roll loops.
        """
        if count < 1:
            raise ValueError("count must be >= 1")
        if count > 5:
            raise ValueError("count > 5 not allowed (cost / quality cliff)")
        variants: List[Dict[str, Any]] = []
        for _ in range(count):
            result = self.regenerate_draft(draft_id)
            variants.append(result["new_draft"])
        self.store.audit("variants.generated", self.store.get_draft(draft_id)["app_slug"], {
            "source_draft_id": draft_id,
            "count": len(variants),
            "variant_ids": [v["id"] for v in variants],
        })
        return {
            "source_draft_id": draft_id,
            "variants": variants,
            "count_generated": len(variants),
        }

    def regenerate_draft(self, draft_id: str) -> Dict[str, Any]:
        """Re-run CopyAgent on the same plan item using the latest steering.

        Creates a NEW draft (keeps the old one — its status is preserved so
        you can compare). New draft carries `regenerated_from=<old_id>` for
        lineage. Token-tracked. Re-runs safety review on the new body.
        """
        old_draft = self.store.get_draft(draft_id)
        app = self.store.require_app(old_draft["app_slug"])
        # Rebuild a plan item from the stored draft so CopyAgent has the right shape.
        channels = app.get("channels") or [old_draft["channel"]]
        pillar_guess = (app.get("content_pillars") or ["brand story"])[0]
        item = {
            "channel": old_draft["channel"],
            "pillar": pillar_guess,
            "angle": _angle_for(app["slug"], pillar_guess),
            "scheduled_for": old_draft.get("scheduled_for"),
        }
        token_ledger: List[Dict[str, Any]] = []
        steering = self.brand_memory.get_steering(self.store, app["slug"], token_ledger=token_ledger)
        raw_draft = self.copy.draft_for_item(
            app,
            old_draft["campaign_id"],
            item,
            token_ledger=token_ledger,
            steering=steering,
        )
        if old_draft.get("scheduled_for"):
            raw_draft["scheduled_for"] = old_draft["scheduled_for"]
        enriched = self.creative.add_concepts(app, raw_draft)
        safety = self.review.review(app, enriched)
        enriched["safety"] = safety
        enriched["status"] = "needs_review" if safety["passed"] else "rejected"
        enriched["regenerated_from"] = old_draft["id"]
        freshness = _compute_freshness(self.store, app["slug"], old_draft["channel"], enriched["body"])
        enriched["freshness_score"] = freshness["score"]
        enriched["freshness_compared_against"] = freshness["compared_against"]
        enriched["freshness_most_similar_id"] = freshness["most_similar_id"]
        new_draft = self.store.create_draft(app["slug"], enriched)
        if token_ledger:
            self.store.record_token_usage(app["slug"], old_draft.get("campaign_id"), token_ledger)
        self.store.audit("draft.regenerated", app["slug"], {
            "from_draft_id": old_draft["id"],
            "to_draft_id": new_draft["id"],
            "channel": old_draft["channel"],
            "steering_applied": bool(steering),
        })
        return {"old_draft_id": old_draft["id"], "new_draft": new_draft, "steering_applied": bool(steering)}

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

    AUTO_GENERATE_COOLDOWN_HOURS = 24
    AUTO_GENERATE_DEFAULT_DAYS = 7

    def poll(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        """One tick of the scheduled poller. Walks every app, fires
        `publish_scheduled(due_only=True)`, AND triggers auto-generation
        for any opt-in app whose pending queue is below threshold. Records
        the result on `state.poll`. Designed to be called from a cron job:
            hermes cron create --schedule "every 5m" --command "hermes marketing-factory poll"

        Returns: {polled_apps, due_count, fired_count, events,
                  auto_generated_apps, last_poll}
        """
        cutoff = now or datetime.now(timezone.utc)
        apps = self.store.list_apps()
        all_events: List[Dict[str, Any]] = []
        auto_generated_apps: List[Dict[str, Any]] = []
        # Count drafts that are due across all apps (for reporting, includes
        # ones we did not fire because they were already non-scheduled).
        due_count = 0
        state = self.store.load()
        cooldown = timedelta(hours=self.AUTO_GENERATE_COOLDOWN_HOURS)

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

            # Auto-generation: opt-in per app. Only if queue is below threshold
            # AND no campaign was generated in the cooldown window — prevents
            # runaway loops when the user isn't approving.
            if not app.get("auto_generate"):
                continue
            threshold = int(app.get("auto_generate_threshold") or 3)
            pending = sum(
                1 for d in state.get("drafts", {}).values()
                if d.get("app_slug") == app["slug"] and d.get("status") in {"needs_review", "approved", "scheduled"}
            )
            if pending >= threshold:
                continue
            recent_campaign = False
            for camp in state.get("campaigns", {}).values():
                if camp.get("app_slug") != app["slug"]:
                    continue
                try:
                    camp_dt = datetime.fromisoformat(str(camp.get("created_at") or "").replace("Z", "+00:00"))
                except ValueError:
                    continue
                if camp_dt.tzinfo is None:
                    camp_dt = camp_dt.replace(tzinfo=timezone.utc)
                if (cutoff - camp_dt) < cooldown:
                    recent_campaign = True
                    break
            if recent_campaign:
                continue
            try:
                result = self.generate_campaign(app["slug"], days=self.AUTO_GENERATE_DEFAULT_DAYS)
                auto_generated_apps.append({
                    "app_slug": app["slug"],
                    "campaign_id": result["campaign"]["id"],
                    "draft_count": len(result["drafts"]),
                    "pending_before": pending,
                    "threshold": threshold,
                })
                # Refresh local state snapshot since we just wrote campaigns/drafts
                state = self.store.load()
            except Exception as exc:  # noqa: BLE001 — auto-gen failure must not break the poll
                logger.warning("auto_generate failed for %s: %s", app["slug"], exc)

        last_poll = self.store.record_poll(fired=len(all_events), due=due_count, polled_apps=len(apps))
        if auto_generated_apps:
            self.store.audit("poll.auto_generated", None, {"count": len(auto_generated_apps), "apps": [a["app_slug"] for a in auto_generated_apps]})
        return {
            "polled_apps": len(apps),
            "due_count": due_count,
            "fired_count": len(all_events),
            "events": all_events,
            "auto_generated_apps": auto_generated_apps,
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


_FRESHNESS_COMPARE_AGAINST = 20

_EMOJI_PATTERN = None
try:
    import re as _re_emoji
    _EMOJI_PATTERN = _re_emoji.compile(
        "["
        "\U0001F300-\U0001F5FF"
        "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF"
        "\U0001F900-\U0001F9FF"
        "☀-⛿"
        "✀-➿"
        "]"
    )
except Exception:
    _EMOJI_PATTERN = None


def _count_hashtags(body: str) -> int:
    return sum(1 for word in (body or "").split() if word.startswith("#") and len(word) > 1)


def _count_emoji(body: str) -> int:
    if _EMOJI_PATTERN is None or not body:
        return 0
    return len(_EMOJI_PATTERN.findall(body))


def _has_link(body: str, app: Dict[str, Any]) -> bool:
    """True if body contains a brand-owned link or any http URL."""
    if not body:
        return False
    if "http://" in body or "https://" in body:
        return True
    for link in app.get("links") or []:
        if link and link in body:
            return True
    return False


def draft_checklist(draft: Dict[str, Any], app: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Mechanically-computed per-channel quality signals.

    Each item: {label, passed: bool, severity: "info"|"warn"|"error", note}.
    Severity "warn" = brand best-practice; "error" = platform constraint
    violation (already caught by safety, but surfaced here for clarity).
    """
    body = draft.get("body") or ""
    channel = draft.get("channel") or ""
    length = len(body)
    items: List[Dict[str, Any]] = []
    has_link = _has_link(body, app)
    hashtags = _count_hashtags(body)
    emoji = _count_emoji(body)

    if channel == "x":
        items.append({"label": "≤280 chars", "passed": length <= 280, "severity": "error", "note": f"{length}/280"})
        items.append({"label": "has link", "passed": has_link, "severity": "warn", "note": "X posts benefit from a single CTA link" if not has_link else "ok"})
        items.append({"label": "hashtags ≤2", "passed": hashtags <= 2, "severity": "warn", "note": f"{hashtags} hashtags — X reads spammy past 2"})
    elif channel == "instagram":
        items.append({"label": "has image", "passed": bool(draft.get("images")), "severity": "error", "note": "IG posts require a visual" if not draft.get("images") else "ok"})
        items.append({"label": "100-2200 chars", "passed": 100 <= length <= 2200, "severity": "warn", "note": f"{length} chars (IG sweet spot is 100-2200)"})
        items.append({"label": "hashtags 3-10", "passed": 3 <= hashtags <= 10, "severity": "warn", "note": f"{hashtags} hashtags (3-10 is IG sweet spot)"})
        items.append({"label": "has CTA", "passed": has_link or "download" in body.lower() or "shop" in body.lower(), "severity": "warn", "note": "missing call-to-action" if not has_link else "ok"})
    elif channel == "tiktok":
        items.append({"label": "≤2200 chars", "passed": length <= 2200, "severity": "error", "note": f"{length}/2200"})
        items.append({"label": "has hook", "passed": length > 0 and len(body.split()[0]) >= 3, "severity": "warn", "note": "short hook in first line is critical"})
    elif channel == "linkedin":
        items.append({"label": "≤3000 chars", "passed": length <= 3000, "severity": "error", "note": f"{length}/3000"})
        items.append({"label": "multi-paragraph", "passed": body.count("\n") >= 1, "severity": "warn", "note": "LinkedIn rewards 2-5 short paragraphs"})
    elif channel == "blog":
        items.append({"label": "has H2 sections", "passed": "##" in body, "severity": "warn", "note": "outline should include H2s"})
        items.append({"label": "≤5000 chars", "passed": length <= 5000, "severity": "error", "note": f"{length}/5000"})
    elif channel == "email":
        items.append({"label": "has Subject:", "passed": body.lstrip().lower().startswith("subject:"), "severity": "warn", "note": "first line should be Subject: ..."})
        items.append({"label": "100-1200 chars", "passed": 100 <= length <= 1200, "severity": "warn", "note": f"{length} chars (email sweet spot 100-1200)"})
    elif channel == "app_store":
        items.append({"label": "≤170 chars", "passed": length <= 170, "severity": "error", "note": f"{length}/170"})
        items.append({"label": "conversion-focused", "passed": any(k in body.lower() for k in ("download", "get", "free", "today", "now")), "severity": "warn", "note": "should contain action verb"})

    # Brand-tone signals: cute brands benefit from emoji presence
    tone = (app.get("tone") or "").lower()
    if "cute" in tone or "playful" in tone or "warm" in tone:
        items.append({"label": "has emoji", "passed": emoji >= 1, "severity": "info", "note": f"{emoji} emoji (brand tone calls for warmth)"})

    return items


def _char_trigrams(text: str) -> set:
    """Lowercased, whitespace-collapsed character trigrams. Cheap signature."""
    text = " ".join((text or "").lower().split())
    if len(text) < 3:
        return {text}
    return {text[i:i + 3] for i in range(len(text) - 2)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _compute_freshness(store: MarketingFactoryStore, app_slug: str, channel: str, body: str) -> Dict[str, Any]:
    """Score 0-1: 1.0 = totally novel vs recent same-channel drafts, 0.0 = perfect duplicate.

    Compared against last `_FRESHNESS_COMPARE_AGAINST` drafts on the same channel
    for the same app, regardless of status (so rejected drafts also count —
    they're still patterns the LLM produced and we want to diverge from).
    Returns {score, compared_against, most_similar_id}.
    """
    state = store.load()
    candidates = [
        d for d in state.get("drafts", {}).values()
        if d.get("app_slug") == app_slug and d.get("channel") == channel and d.get("body")
    ]
    candidates.sort(key=lambda d: d.get("created_at") or "", reverse=True)
    candidates = candidates[:_FRESHNESS_COMPARE_AGAINST]
    if not candidates:
        return {"score": 1.0, "compared_against": 0, "most_similar_id": None}
    body_trigrams = _char_trigrams(body)
    most_similar_id = None
    max_similarity = 0.0
    for cand in candidates:
        sim = _jaccard(body_trigrams, _char_trigrams(cand.get("body") or ""))
        if sim > max_similarity:
            max_similarity = sim
            most_similar_id = cand.get("id")
    score = round(1.0 - max_similarity, 3)
    return {"score": score, "compared_against": len(candidates), "most_similar_id": most_similar_id}


def _default_image_prompt(app: Dict[str, Any], item: Dict[str, Any], body: str) -> str:
    """Deterministic fallback used when LLM is off or the LLM image-prompt synth fails."""
    slug = app.get("slug", "")
    channel = item.get("channel", "post")
    pillar = item.get("pillar") or "brand story"
    if slug == "pupular":
        return (
            f"A warm, inviting photograph of an adoptable {channel} subject: a real-looking small dog or cat with soft expression, "
            f"natural daylight, shallow depth of field, hopeful mood. {pillar}. Modern, friendly, brand-safe."
        )
    if slug == "setvenue":
        return (
            f"A premium architectural photograph of a unique production-friendly venue: warm light, considered composition, "
            f"sense of space and possibility. {pillar}. Editorial, trustworthy, photo-real."
        )
    return f"Brand-safe visual for {app.get('name') or slug} highlighting {pillar} on {channel}; modern, clean, editorial."


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
