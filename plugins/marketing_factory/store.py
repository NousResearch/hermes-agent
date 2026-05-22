"""Durable store for the Marketing Agent Factory plugin."""

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from hermes_constants import get_hermes_home


SCHEMA_VERSION = 1
STATUSES = {"draft", "needs_review", "approved", "rejected", "scheduled", "dry_run_posted"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def default_store_path() -> Path:
    return Path(get_hermes_home()) / "marketing_factory"


DEFAULT_STATE: Dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "apps": {},
    "campaigns": {},
    "drafts": {},
    "approvals": {},
    "schedules": {},
    "publish_events": {},
    "analytics": {},
    "brand_memories": {},
    "model_routing_policy": {
        "cheap": {"provider": "local", "model": "qwen-or-llama-small", "tasks": ["classification", "summarization", "duplicate_check", "scraping_cleanup"]},
        "mid": {"provider": "cloud", "model": "mid-tier-router", "tasks": ["rewrites", "repurposing", "channel_variants"]},
        "premium": {"provider": "claude-cli-oauth", "model": "claude", "tasks": ["strategy", "final_review", "high_value_copy"]},
    },
    "budgets": {
        "daily_tokens": 250000,
        "per_app_tokens": {},
        "per_channel_tokens": {},
        "spent_tokens_today": 0,
    },
}


@dataclass(frozen=True)
class MarketingFactoryPaths:
    root: Path

    @property
    def state_file(self) -> Path:
        return self.root / "state.json"

    @property
    def audit_file(self) -> Path:
        return self.root / "audit.jsonl"


class MarketingFactoryStore:
    """JSON-backed MVP store with app-scoped operations and JSONL audit log."""

    def __init__(self, root: Optional[str | Path] = None):
        self.paths = MarketingFactoryPaths(Path(root).expanduser() if root else default_store_path())

    def initialize(self) -> Dict[str, Any]:
        self.paths.root.mkdir(parents=True, exist_ok=True)
        if not self.paths.state_file.exists():
            self._write_state(deepcopy(DEFAULT_STATE))
        self.paths.audit_file.touch(exist_ok=True)
        self.audit("store.initialized", "system", {"store_path": str(self.paths.root)})
        return self.load()

    def load(self) -> Dict[str, Any]:
        if not self.paths.state_file.exists():
            return self.initialize()
        with self.paths.state_file.open("r", encoding="utf-8") as fh:
            state = json.load(fh)
        changed = False
        for key, value in DEFAULT_STATE.items():
            if key not in state:
                state[key] = deepcopy(value)
                changed = True
        if state.get("schema_version") != SCHEMA_VERSION:
            state["schema_version"] = SCHEMA_VERSION
            changed = True
        if changed:
            self._write_state(state)
        return state

    def _write_state(self, state: Dict[str, Any]) -> None:
        self.paths.root.mkdir(parents=True, exist_ok=True)
        tmp = self.paths.state_file.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, sort_keys=True)
            fh.write("\n")
        tmp.replace(self.paths.state_file)

    def audit(self, action: str, app_slug: Optional[str], payload: Dict[str, Any]) -> Dict[str, Any]:
        self.paths.root.mkdir(parents=True, exist_ok=True)
        event = {
            "id": new_id("audit"),
            "timestamp": utc_now(),
            "action": action,
            "app_slug": app_slug,
            "payload": payload,
        }
        with self.paths.audit_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, sort_keys=True) + "\n")
        return event

    def list_audit(self, app_slug: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        if not self.paths.audit_file.exists():
            return []
        events = []
        with self.paths.audit_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                event = json.loads(line)
                if app_slug is None or event.get("app_slug") == app_slug:
                    events.append(event)
        return events[-limit:]

    def upsert_app(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        slug = _require_slug(profile.get("slug"))
        state = self.load()
        existing = state["apps"].get(slug, {})
        now = utc_now()
        merged = {**existing, **deepcopy(profile), "slug": slug, "updated_at": now}
        merged.setdefault("created_at", now)
        merged.setdefault("channels", [])
        merged.setdefault("claims", [])
        merged.setdefault("forbidden_claims", [])
        merged.setdefault("assets", [])
        merged.setdefault("content_pillars", [])
        state["apps"][slug] = merged
        state["brand_memories"].setdefault(slug, {"learnings": [], "summaries": []})
        self._write_state(state)
        self.audit("brand_profile.upserted", slug, {"name": merged.get("name"), "channels": merged.get("channels", [])})
        return merged

    def require_app(self, app_slug: str) -> Dict[str, Any]:
        slug = _require_slug(app_slug)
        app = self.load()["apps"].get(slug)
        if not app:
            raise KeyError(f"Unknown app slug: {slug}")
        return app

    def list_apps(self) -> List[Dict[str, Any]]:
        return sorted(self.load()["apps"].values(), key=lambda item: item["slug"])

    def create_campaign(self, app_slug: str, campaign: Dict[str, Any]) -> Dict[str, Any]:
        app = self.require_app(app_slug)
        state = self.load()
        cid = campaign.get("id") or new_id("camp")
        record = {
            "id": cid,
            "app_slug": app["slug"],
            "name": campaign["name"],
            "objective": campaign.get("objective", "awareness"),
            "channels": list(campaign.get("channels") or app.get("channels") or []),
            "status": campaign.get("status", "planned"),
            "plan": campaign.get("plan", []),
            "model_route": campaign.get("model_route"),
            "research_summary": campaign.get("research_summary"),
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        state["campaigns"][cid] = record
        self._write_state(state)
        self.audit("campaign.created", app["slug"], {"campaign_id": cid, "name": record["name"]})
        return record

    def list_campaigns(self, app_slug: Optional[str] = None) -> List[Dict[str, Any]]:
        state = self.load()
        campaigns = state["campaigns"].values()
        if app_slug:
            slug = _require_slug(app_slug)
            campaigns = [c for c in campaigns if c["app_slug"] == slug]
        return sorted(campaigns, key=lambda item: item["created_at"])

    def create_draft(self, app_slug: str, draft: Dict[str, Any]) -> Dict[str, Any]:
        self.require_app(app_slug)
        state = self.load()
        did = draft.get("id") or new_id("draft")
        record = {
            "id": did,
            "app_slug": _require_slug(app_slug),
            "campaign_id": draft["campaign_id"],
            "channel": draft["channel"],
            "content_type": draft.get("content_type", "post"),
            "body": draft["body"],
            "cta": draft.get("cta"),
            "assets": list(draft.get("assets") or []),
            "safety": draft.get("safety", {}),
            "model_route": draft.get("model_route", "cheap"),
            "status": draft.get("status", "needs_review"),
            "scheduled_for": draft.get("scheduled_for"),
            "llm_used": bool(draft.get("llm_used")),
            "llm_model": draft.get("llm_model"),
            "llm_error": draft.get("llm_error"),
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        if record["status"] not in STATUSES:
            raise ValueError(f"Invalid draft status: {record['status']}")
        state["drafts"][did] = record
        state["approvals"][did] = {
            "draft_id": did,
            "app_slug": record["app_slug"],
            "status": "pending" if record["status"] == "needs_review" else record["status"],
            "reviewer": None,
            "reason": None,
            "updated_at": utc_now(),
        }
        self._write_state(state)
        self.audit("draft.created", record["app_slug"], {"draft_id": did, "channel": record["channel"], "status": record["status"]})
        return record

    def list_drafts(self, app_slug: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        drafts: Iterable[Dict[str, Any]] = self.load()["drafts"].values()
        if app_slug:
            slug = _require_slug(app_slug)
            drafts = [d for d in drafts if d["app_slug"] == slug]
        if status:
            drafts = [d for d in drafts if d["status"] == status]
        return sorted(drafts, key=lambda item: item["created_at"])

    def get_draft(self, draft_id: str, app_slug: Optional[str] = None) -> Dict[str, Any]:
        draft = self.load()["drafts"].get(draft_id)
        if not draft:
            raise KeyError(f"Unknown draft id: {draft_id}")
        if app_slug and draft["app_slug"] != _require_slug(app_slug):
            raise PermissionError(f"Draft {draft_id} does not belong to app {app_slug}")
        return draft

    def set_approval(self, draft_id: str, status: str, reviewer: str = "human", reason: Optional[str] = None) -> Dict[str, Any]:
        if status not in {"approved", "rejected"}:
            raise ValueError("approval status must be approved or rejected")
        state = self.load()
        draft = state["drafts"].get(draft_id)
        if not draft:
            raise KeyError(f"Unknown draft id: {draft_id}")
        draft["status"] = status
        draft["updated_at"] = utc_now()
        approval = state["approvals"].setdefault(draft_id, {"draft_id": draft_id, "app_slug": draft["app_slug"]})
        approval.update({"status": status, "reviewer": reviewer, "reason": reason, "updated_at": utc_now()})

        # Phase 3: capture every approve/reject as a structured brand-memory entry so
        # the steering loop has real signal to summarize. Generic learnings like
        # "campaign generated N drafts" are NOT useful — these are.
        body = draft.get("body") or ""
        excerpt = (body[:300] + ("…" if len(body) > 300 else ""))
        memory_entry = {
            "created_at": utc_now(),
            "kind": f"draft_{status}",
            "draft_id": draft_id,
            "channel": draft.get("channel"),
            "model_route": draft.get("model_route"),
            "llm_used": bool(draft.get("llm_used")),
            "llm_model": draft.get("llm_model"),
            "reason": reason or ("approved" if status == "approved" else "rejected"),
            "reviewer": reviewer,
            "excerpt": excerpt,
            "text": f"{status.upper()} ({draft.get('channel')}): {reason or 'no reason given'} — excerpt: {excerpt}",
        }
        memories = state["brand_memories"].setdefault(draft["app_slug"], {"learnings": [], "summaries": []})
        memories.setdefault("learnings", []).append(memory_entry)
        # New learning landed; invalidate the cached steering summary so the next
        # campaign generation triggers a re-summarize.
        memories.pop("steering", None)

        self._write_state(state)
        self.audit(f"draft.{status}", draft["app_slug"], {"draft_id": draft_id, "reviewer": reviewer, "reason": reason})
        return approval

    def schedule_draft(self, draft_id: str, scheduled_for: str) -> Dict[str, Any]:
        state = self.load()
        draft = state["drafts"].get(draft_id)
        if not draft:
            raise KeyError(f"Unknown draft id: {draft_id}")
        if draft["status"] not in {"approved", "scheduled", "dry_run_posted"}:
            raise ValueError("draft must be approved before scheduling")
        sid = state["schedules"].get(draft_id, {}).get("id") or new_id("sched")
        schedule = {
            "id": sid,
            "draft_id": draft_id,
            "app_slug": draft["app_slug"],
            "channel": draft["channel"],
            "scheduled_for": scheduled_for,
            "status": "scheduled",
            "created_at": state["schedules"].get(draft_id, {}).get("created_at", utc_now()),
            "updated_at": utc_now(),
        }
        state["schedules"][draft_id] = schedule
        draft["status"] = "scheduled"
        draft["scheduled_for"] = scheduled_for
        draft["updated_at"] = utc_now()
        self._write_state(state)
        self.audit("draft.scheduled", draft["app_slug"], {"draft_id": draft_id, "scheduled_for": scheduled_for})
        return schedule

    def list_schedules(self, app_slug: Optional[str] = None) -> List[Dict[str, Any]]:
        schedules: Iterable[Dict[str, Any]] = self.load()["schedules"].values()
        if app_slug:
            slug = _require_slug(app_slug)
            schedules = [s for s in schedules if s["app_slug"] == slug]
        return sorted(schedules, key=lambda item: item["scheduled_for"])

    def dry_run_publish(self, draft_id: str) -> Dict[str, Any]:
        state = self.load()
        draft = state["drafts"].get(draft_id)
        if not draft:
            raise KeyError(f"Unknown draft id: {draft_id}")
        if draft["status"] not in {"scheduled", "dry_run_posted"}:
            raise ValueError("draft must be scheduled before dry-run publish")
        event_id = new_id("pub")
        event = {
            "id": event_id,
            "draft_id": draft_id,
            "app_slug": draft["app_slug"],
            "channel": draft["channel"],
            "mode": "dry_run",
            "would_post": True,
            "posted": False,
            "body": draft["body"],
            "created_at": utc_now(),
        }
        state["publish_events"][event_id] = event
        draft["status"] = "dry_run_posted"
        draft["updated_at"] = utc_now()
        self._write_state(state)
        self.audit("publish.dry_run", draft["app_slug"], {"draft_id": draft_id, "event_id": event_id, "channel": draft["channel"]})
        return event

    def write_steering(self, app_slug: str, steering: Dict[str, Any]) -> Dict[str, Any]:
        """Persist a per-app steering blob into `brand_memories[slug].steering`."""
        self.require_app(app_slug)
        slug = _require_slug(app_slug)
        state = self.load()
        memories = state.setdefault("brand_memories", {}).setdefault(slug, {"learnings": [], "summaries": []})
        memories["steering"] = deepcopy(steering)
        self._write_state(state)
        self.audit("steering.summarized", slug, {
            "method": steering.get("method"),
            "approved_count": steering.get("approved_count"),
            "rejected_count": steering.get("rejected_count"),
        })
        return memories["steering"]

    def record_token_usage(
        self,
        app_slug: str,
        campaign_id: Optional[str],
        ledger_entries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Accumulate per-app / per-channel / per-route token spend into `budgets`.

        `ledger_entries` is a list of `{route, model, tokens, agent, channel, elapsed_ms}` dicts
        produced by `model_dispatcher.dispatch()`. The store resets daily counters whenever
        `last_reset_date` does not match today (UTC), so spend tracking is rolling-day.
        """
        self.require_app(app_slug)
        if not ledger_entries:
            return {"tokens_used": 0, "calls": 0}

        slug = _require_slug(app_slug)
        state = self.load()
        budgets = state["budgets"]
        today = datetime.now(timezone.utc).date().isoformat()
        if budgets.get("last_reset_date") != today:
            budgets["last_reset_date"] = today
            budgets["spent_tokens_today"] = 0
            budgets["spent_by_route"] = {"cheap": 0, "mid": 0, "premium": 0}
            budgets["per_app_tokens"] = {}
            budgets["per_channel_tokens"] = {}

        spent_by_route = budgets.setdefault("spent_by_route", {"cheap": 0, "mid": 0, "premium": 0})
        per_app = budgets.setdefault("per_app_tokens", {})
        per_channel = budgets.setdefault("per_channel_tokens", {})

        total = 0
        for entry in ledger_entries:
            tokens = int(entry.get("tokens") or 0)
            if tokens <= 0:
                continue
            total += tokens
            route = entry.get("route") or "unknown"
            channel = entry.get("channel") or "_none"
            spent_by_route[route] = spent_by_route.get(route, 0) + tokens
            per_app[slug] = per_app.get(slug, 0) + tokens
            per_channel[channel] = per_channel.get(channel, 0) + tokens

        budgets["spent_tokens_today"] = budgets.get("spent_tokens_today", 0) + total
        self._write_state(state)
        self.audit(
            "tokens.recorded",
            slug,
            {
                "campaign_id": campaign_id,
                "tokens_used": total,
                "calls": len(ledger_entries),
                "by_route": {r: sum(e.get("tokens", 0) for e in ledger_entries if e.get("route") == r) for r in {e.get("route") for e in ledger_entries if e.get("route")}},
            },
        )
        return {"tokens_used": total, "calls": len(ledger_entries), "daily_total": budgets["spent_tokens_today"]}

    def record_analytics(self, app_slug: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.require_app(app_slug)
        state = self.load()
        aid = payload.get("id") or new_id("ana")
        record = {"id": aid, "app_slug": _require_slug(app_slug), "created_at": utc_now(), **deepcopy(payload)}
        state["analytics"][aid] = record
        learning = payload.get("learning") or payload.get("summary")
        if learning:
            state["brand_memories"].setdefault(_require_slug(app_slug), {"learnings": [], "summaries": []})["learnings"].append({"created_at": utc_now(), "text": learning})
        self._write_state(state)
        self.audit("analytics.recorded", _require_slug(app_slug), {"analytics_id": aid, "learning": bool(learning)})
        return record

    def summary(self) -> Dict[str, Any]:
        state = self.load()
        return {
            "store_path": str(self.paths.root),
            "apps": len(state["apps"]),
            "campaigns": len(state["campaigns"]),
            "drafts": len(state["drafts"]),
            "pending_approvals": len([a for a in state["approvals"].values() if a.get("status") == "pending"]),
            "scheduled": len(state["schedules"]),
            "dry_run_publish_events": len(state["publish_events"]),
            "budgets": state["budgets"],
            "model_routing_policy": state["model_routing_policy"],
        }


def _require_slug(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("app slug is required")
    slug = value.strip().lower()
    if not all(ch.isalnum() or ch in {"-", "_"} for ch in slug):
        raise ValueError(f"Invalid app slug: {value!r}")
    return slug
