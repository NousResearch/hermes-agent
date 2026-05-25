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
STATUSES = {"draft", "needs_review", "approved", "rejected", "scheduled", "dry_run_posted", "posted"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def default_store_path() -> Path:
    return Path(get_hermes_home()) / "marketing_factory"


# Rough USD per 1M tokens for cost estimates. Local models (cheap/mid) are
# zero — electricity only. Premium routes through Claude CLI (the user's
# Code subscription) so the *marginal* cost is $0; we surface an
# API-equivalent estimate (blended Sonnet input/output) so the user can
# gauge how much they're "saving" by running through the subscription.
# Overridable per route via env: MF_PRICE_USD_PER_M_<ROUTE>.
DEFAULT_PRICES_USD_PER_M = {
    "cheap": 0.0,
    "mid": 0.0,
    "premium": 6.0,  # Sonnet 4.6 blended input+output ish
}


def _price_for(route: str) -> float:
    env_key = f"MF_PRICE_USD_PER_M_{route.upper()}"
    import os as _os
    raw = _os.environ.get(env_key)
    if raw is not None:
        try:
            return float(raw)
        except ValueError:
            pass
    return DEFAULT_PRICES_USD_PER_M.get(route, 0.0)


def estimate_costs(spent_by_route: Dict[str, Any]) -> Dict[str, Any]:
    """Convert per-route token counts to estimated USD."""
    breakdown: Dict[str, float] = {}
    total = 0.0
    for route, tokens in (spent_by_route or {}).items():
        rate = _price_for(route)
        usd = (int(tokens or 0) / 1_000_000.0) * rate
        breakdown[route] = round(usd, 4)
        total += usd
    return {
        "by_route_usd": breakdown,
        "total_usd": round(total, 4),
        "rates_usd_per_m": {r: _price_for(r) for r in ("cheap", "mid", "premium")},
        "note": "premium tokens flow through Claude Code subscription (marginal cost $0); shown rate is API-equivalent for budgeting only",
    }


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
    "poll": {"last_poll_at": None, "last_poll_fired": 0, "last_poll_due": 0, "total_polls": 0},
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
        # Phase 4: every channel starts in dry_run. `live` requires both a registered
        # real connector AND an explicit set_channel_mode call.
        existing_modes = dict(existing.get("channel_modes") or {})
        for channel in merged.get("channels", []):
            existing_modes.setdefault(channel, "dry_run")
        merged["channel_modes"] = existing_modes
        # Phase 15: auto_generate is opt-in (off by default) so the LLM doesn't burn
        # tokens on apps the user isn't actively running. Threshold defaults to 3.
        merged.setdefault("auto_generate", existing.get("auto_generate", False))
        merged.setdefault("auto_generate_threshold", existing.get("auto_generate_threshold", 3))
        # Per-brand visual identity. image_library = list of brand-approved URLs
        # rotated through per draft (zero cost, always works, brand-authentic).
        # image_style_guide = free-form text the ImageGenAgent weaves into the
        # LLM image-prompt synth when library is empty and MF_AUTO_IMAGES=1.
        merged.setdefault("image_library", existing.get("image_library", []))
        merged.setdefault("image_style_guide", existing.get("image_style_guide", ""))
        state["apps"][slug] = merged
        state["brand_memories"].setdefault(slug, {"learnings": [], "summaries": []})
        self._write_state(state)
        self.audit("brand_profile.upserted", slug, {"name": merged.get("name"), "channels": merged.get("channels", [])})
        return merged

    def remove_app(self, app_slug: str, *, cascade: bool = True) -> Dict[str, Any]:
        """Remove a brand profile. With cascade=True, also drops every record
        scoped to that slug. With cascade=False, refuses if any dependent record
        exists.
        """
        slug = _require_slug(app_slug)
        state = self.load()
        if slug not in state["apps"]:
            raise KeyError(f"Unknown app slug: {slug}")

        def _belongs(record: Dict[str, Any]) -> bool:
            return record.get("app_slug") == slug

        # Collect what would be removed for the audit summary.
        campaigns = [cid for cid, c in state["campaigns"].items() if _belongs(c)]
        drafts = [did for did, d in state["drafts"].items() if _belongs(d)]
        approvals = [aid for aid, a in state["approvals"].items() if a.get("app_slug") == slug]
        schedules = [sid for sid, s in state["schedules"].items() if _belongs(s)]
        publish_events = [pid for pid, p in state["publish_events"].items() if _belongs(p)]
        analytics_ids = [aid for aid, a in state["analytics"].items() if _belongs(a)]

        deps = {
            "campaigns": len(campaigns),
            "drafts": len(drafts),
            "approvals": len(approvals),
            "schedules": len(schedules),
            "publish_events": len(publish_events),
            "analytics": len(analytics_ids),
        }
        if not cascade and any(deps.values()):
            raise ValueError(f"Cannot remove {slug} with cascade=False; dependents exist: {deps}")

        for cid in campaigns:
            state["campaigns"].pop(cid, None)
        for did in drafts:
            state["drafts"].pop(did, None)
            state["approvals"].pop(did, None)  # approvals are keyed by draft_id
        for sid in schedules:
            state["schedules"].pop(sid, None)
        for pid in publish_events:
            state["publish_events"].pop(pid, None)
        for aid in analytics_ids:
            state["analytics"].pop(aid, None)
        state["apps"].pop(slug, None)
        state.get("brand_memories", {}).pop(slug, None)

        # Strip per-app rows from budgets if any survive.
        budgets = state.get("budgets") or {}
        for table_name in ("per_app_tokens",):
            table = budgets.get(table_name) or {}
            table.pop(slug, None)

        self._write_state(state)
        self.audit("brand_profile.removed", slug, {"removed": deps})
        return {"app_slug": slug, "removed": deps}

    def add_image_to_library(self, app_slug: str, url: str, reviewer: str = "human") -> Dict[str, Any]:
        """Append a brand-approved image URL to the brand's image_library."""
        slug = _require_slug(app_slug)
        if not url or not url.strip():
            raise ValueError("url is required")
        state = self.load()
        app = state["apps"].get(slug)
        if not app:
            raise KeyError(f"Unknown app slug: {slug}")
        library = list(app.get("image_library") or [])
        clean = url.strip()
        if clean not in library:
            library.append(clean)
        app["image_library"] = library
        app["updated_at"] = utc_now()
        state["apps"][slug] = app
        self._write_state(state)
        self.audit("image_library.added", slug, {"url": clean, "library_size": len(library), "reviewer": reviewer})
        return {"app_slug": slug, "library_size": len(library), "added": clean}

    def remove_image_from_library(self, app_slug: str, url: str, reviewer: str = "human") -> Dict[str, Any]:
        slug = _require_slug(app_slug)
        state = self.load()
        app = state["apps"].get(slug)
        if not app:
            raise KeyError(f"Unknown app slug: {slug}")
        library = [u for u in (app.get("image_library") or []) if u != url]
        app["image_library"] = library
        app["updated_at"] = utc_now()
        state["apps"][slug] = app
        self._write_state(state)
        self.audit("image_library.removed", slug, {"url": url, "library_size": len(library), "reviewer": reviewer})
        return {"app_slug": slug, "library_size": len(library), "removed": url}

    def set_image_style_guide(self, app_slug: str, style_guide: str, reviewer: str = "human") -> Dict[str, Any]:
        slug = _require_slug(app_slug)
        state = self.load()
        app = state["apps"].get(slug)
        if not app:
            raise KeyError(f"Unknown app slug: {slug}")
        app["image_style_guide"] = (style_guide or "").strip()
        app["updated_at"] = utc_now()
        state["apps"][slug] = app
        self._write_state(state)
        self.audit("image_style_guide.updated", slug, {"length": len(app["image_style_guide"]), "reviewer": reviewer})
        return {"app_slug": slug, "image_style_guide": app["image_style_guide"]}

    def set_auto_generate(self, app_slug: str, enabled: bool, *, threshold: Optional[int] = None, reviewer: str = "human") -> Dict[str, Any]:
        """Toggle the per-app auto-generation flag and optionally update its threshold."""
        slug = _require_slug(app_slug)
        state = self.load()
        app = state["apps"].get(slug)
        if not app:
            raise KeyError(f"Unknown app slug: {slug}")
        previous_enabled = bool(app.get("auto_generate", False))
        previous_threshold = int(app.get("auto_generate_threshold") or 3)
        app["auto_generate"] = bool(enabled)
        if threshold is not None:
            if int(threshold) < 1:
                raise ValueError("threshold must be >= 1")
            app["auto_generate_threshold"] = int(threshold)
        app["updated_at"] = utc_now()
        state["apps"][slug] = app
        self._write_state(state)
        self.audit("auto_generate.changed", slug, {
            "previous_enabled": previous_enabled,
            "next_enabled": bool(enabled),
            "previous_threshold": previous_threshold,
            "next_threshold": int(app["auto_generate_threshold"]),
            "reviewer": reviewer,
        })
        return {"app_slug": slug, "auto_generate": bool(enabled), "auto_generate_threshold": int(app["auto_generate_threshold"])}

    def set_channel_mode(self, app_slug: str, channel: str, mode: str, reviewer: str = "human") -> Dict[str, Any]:
        """Switch a channel between `dry_run` and `live`.

        Switching to `live` is *only* honored at publish time when a real connector
        is registered for that channel. Until then, the Publisher logs an audit
        warning and falls back to dry_run automatically.
        """
        if mode not in {"dry_run", "live"}:
            raise ValueError("mode must be 'dry_run' or 'live'")
        slug = _require_slug(app_slug)
        state = self.load()
        app = state["apps"].get(slug)
        if not app:
            raise KeyError(f"Unknown app slug: {slug}")
        if channel not in (app.get("channels") or []):
            raise ValueError(f"Channel {channel!r} is not declared on {slug}")
        modes = dict(app.get("channel_modes") or {})
        previous = modes.get(channel, "dry_run")
        modes[channel] = mode
        app["channel_modes"] = modes
        app["updated_at"] = utc_now()
        state["apps"][slug] = app
        self._write_state(state)
        self.audit("channel_mode.changed", slug, {
            "channel": channel,
            "previous": previous,
            "next": mode,
            "reviewer": reviewer,
        })
        return {"app_slug": slug, "channel": channel, "previous": previous, "mode": mode}

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
            "regenerated_from": draft.get("regenerated_from"),
            "images": list(draft.get("images") or []),
            "freshness_score": draft.get("freshness_score"),
            "freshness_compared_against": draft.get("freshness_compared_against"),
            "freshness_most_similar_id": draft.get("freshness_most_similar_id"),
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

    def update_draft_body(
        self,
        draft_id: str,
        body: str,
        *,
        editor: str = "human",
        safety: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update a draft's body in place — used for the inline-edit-before-approve flow.

        Records edited_at / edited_by, length deltas, optionally a re-run safety dict
        if the caller provided one (PublisherAgent / dashboard typically pass the
        recomputed safety result). Refuses empty bodies. Refuses to edit drafts that
        have already been published.
        """
        if not body or not body.strip():
            raise ValueError("body cannot be empty")
        state = self.load()
        draft = state["drafts"].get(draft_id)
        if not draft:
            raise KeyError(f"Unknown draft id: {draft_id}")
        if draft["status"] in {"posted", "dry_run_posted"}:
            raise ValueError("Cannot edit a draft that has already been published")
        previous_body = draft.get("body") or ""
        new_body = body.strip()
        draft["body"] = new_body
        draft["edited_at"] = utc_now()
        draft["edited_by"] = editor
        draft["updated_at"] = utc_now()
        if safety is not None:
            draft["safety"] = safety
        self._write_state(state)
        self.audit("draft.edited", draft["app_slug"], {
            "draft_id": draft_id,
            "editor": editor,
            "from_length": len(previous_body),
            "to_length": len(new_body),
        })
        return draft

    def update_draft_scheduled_for(self, draft_id: str, scheduled_for: str) -> Dict[str, Any]:
        """Move a draft's scheduled time without re-running the approval flow.

        For drafts still in needs_review/rejected: updates the advisory
        `draft.scheduled_for` only. For approved/scheduled drafts:
        additionally updates the matching `schedules[draft_id]` record so the
        next poll respects the new time. Refuses dry_run_posted / posted.

        Audits `draft.rescheduled` with previous + next ISO.
        """
        from datetime import datetime as _dt

        if not scheduled_for or not scheduled_for.strip():
            raise ValueError("scheduled_for is required")
        try:
            _dt.fromisoformat(scheduled_for.replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"scheduled_for must be ISO 8601: {exc}") from exc

        state = self.load()
        draft = state["drafts"].get(draft_id)
        if not draft:
            raise KeyError(f"Unknown draft id: {draft_id}")
        if draft["status"] in {"dry_run_posted", "posted"}:
            raise ValueError("Cannot reschedule a draft that has already been published")
        previous = draft.get("scheduled_for")
        draft["scheduled_for"] = scheduled_for
        draft["updated_at"] = utc_now()

        # If the draft already has a real schedule record (approved/scheduled),
        # update it too so the poller picks up the new time on its next tick.
        schedule = state["schedules"].get(draft_id)
        if schedule is not None:
            schedule["scheduled_for"] = scheduled_for
            schedule["updated_at"] = utc_now()

        self._write_state(state)
        self.audit("draft.rescheduled", draft["app_slug"], {
            "draft_id": draft_id,
            "previous": previous,
            "next": scheduled_for,
            "had_schedule_record": schedule is not None,
        })
        return draft

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
        if draft["status"] not in {"scheduled", "dry_run_posted", "posted"}:
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

    def record_publish_event(
        self,
        draft_id: str,
        connector_result: Dict[str, Any],
        *,
        fallback_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist a publish event produced by any connector and update draft status.

        `connector_result` must include `mode`, `would_post`, `posted` (per
        `BaseChannelConnector` contract). `fallback_reason` is set by
        `PublisherAgent` when a live publish was downgraded to dry_run (no
        connector registered, or connector raised). Audits the event with a
        mode-specific action so dashboards can distinguish real posts.
        """
        state = self.load()
        draft = state["drafts"].get(draft_id)
        if not draft:
            raise KeyError(f"Unknown draft id: {draft_id}")
        if draft["status"] not in {"scheduled", "dry_run_posted", "posted"}:
            raise ValueError("draft must be scheduled before publish")
        event_id = new_id("pub")
        mode = str(connector_result.get("mode") or "dry_run")
        posted = bool(connector_result.get("posted"))
        would_post = bool(connector_result.get("would_post"))
        event = {
            "id": event_id,
            "draft_id": draft_id,
            "app_slug": draft["app_slug"],
            "channel": draft["channel"],
            "mode": mode,
            "would_post": would_post,
            "posted": posted,
            "body": draft["body"],
            "payload": deepcopy(connector_result.get("payload") or {}),
            "fallback_reason": fallback_reason,
            "created_at": utc_now(),
        }
        state["publish_events"][event_id] = event
        if posted:
            draft["status"] = "posted"
        elif mode == "dry_run":
            draft["status"] = "dry_run_posted"
        draft["updated_at"] = utc_now()
        self._write_state(state)
        action = f"publish.{mode}" + (".fallback" if fallback_reason else "")
        self.audit(action, draft["app_slug"], {
            "draft_id": draft_id,
            "event_id": event_id,
            "channel": draft["channel"],
            "mode": mode,
            "posted": posted,
            "fallback_reason": fallback_reason,
        })
        return event

    def record_poll(self, *, fired: int, due: int, polled_apps: int) -> Dict[str, Any]:
        """Record one tick of the scheduled poller. Used by the dashboard's
        "last poll" indicator and audited as `poll.tick`."""
        state = self.load()
        poll = state.setdefault("poll", {"last_poll_at": None, "last_poll_fired": 0, "last_poll_due": 0, "total_polls": 0})
        poll["last_poll_at"] = utc_now()
        poll["last_poll_fired"] = fired
        poll["last_poll_due"] = due
        poll["last_polled_apps"] = polled_apps
        poll["total_polls"] = (poll.get("total_polls") or 0) + 1
        self._write_state(state)
        self.audit("poll.tick", None, {"fired": fired, "due": due, "polled_apps": polled_apps})
        return poll

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
            "poll": state.get("poll", {}),
            "cost_estimate": estimate_costs(state["budgets"].get("spent_by_route") or {}),
        }


def _require_slug(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("app slug is required")
    slug = value.strip().lower()
    if not all(ch.isalnum() or ch in {"-", "_"} for ch in slug):
        raise ValueError(f"Invalid app slug: {value!r}")
    return slug
