"""Agent-callable tools for Marketing Agent Factory."""

from __future__ import annotations

import json
from typing import Any, Dict

from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
from plugins.marketing_factory.store import MarketingFactoryStore


MARKETING_FACTORY_SCHEMA: Dict[str, Any] = {
    "name": "marketing_factory",
    "description": (
        "Operate the dry-run-first Marketing Agent Factory: initialize brand profiles, "
        "generate campaigns, approve/schedule/edit drafts, regenerate or fan out variants, "
        "dry-run publish, run scheduled poller, run advisor, export digests, inspect state "
        "and audit logs. Never performs real public posting unless an explicit channel "
        "connector is registered AND channel_modes[channel]=='live'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    # Core lifecycle
                    "init", "status", "generate", "full_dry_run", "audit",
                    # Browse
                    "list_apps", "list_drafts",
                    # Per-draft actions
                    "approve", "reject", "edit", "regenerate", "variants", "reschedule",
                    "schedule", "publish_dry_run",
                    # Brand profile CRUD
                    "add_app", "update_app", "remove_app",
                    # Channel + automation
                    "set_channel_mode", "set_auto_generate",
                    # Operations
                    "poll", "advise", "digest",
                ],
            },
            "app_slug": {"type": "string", "description": "Brand/app slug such as pupular or setvenue"},
            "draft_id": {"type": "string"},
            "days": {"type": "integer", "default": 7},
            "count": {"type": "integer", "description": "Variant count for `variants`", "default": 3},
            "reviewer": {"type": "string", "default": "human"},
            "editor": {"type": "string", "default": "agent"},
            "reason": {"type": "string"},
            "body": {"type": "string", "description": "New body text for `edit`"},
            "scheduled_for": {"type": "string", "description": "ISO timestamp for scheduling/rescheduling one draft"},
            "channel": {"type": "string", "description": "Channel name for `set_channel_mode`"},
            "mode": {"type": "string", "enum": ["dry_run", "live"], "description": "Channel mode value"},
            "enabled": {"type": "boolean", "description": "Toggle value for `set_auto_generate`"},
            "threshold": {"type": "integer", "description": "Queue threshold for auto-generate"},
            "profile": {
                "type": "object",
                "description": "Brand profile fields for `add_app` / `update_app` (slug, name, positioning, icp, tone, cta, channels, content_pillars, claims, forbidden_claims, links, competitors).",
                "additionalProperties": True,
            },
            "store_path": {"type": "string", "description": "Optional test/store override path"},
        },
        "required": ["action"],
    },
}


def handle_marketing_factory(args: Dict[str, Any], **_: Any) -> str:
    store = MarketingFactoryStore(args.get("store_path"))
    pipe = MarketingFactoryPipeline(store)
    action = args.get("action")
    try:
        if action == "init":
            result: Any = pipe.initialize_samples()
        elif action == "status":
            store.initialize()
            result = store.summary()
        elif action == "generate":
            full = pipe.generate_campaign(_require_arg(args, "app_slug"), days=int(args.get("days") or 7))
            result = {"campaign": full["campaign"], "drafts": full["drafts"], "token_summary": full.get("token_summary")}
        elif action == "full_dry_run":
            result = pipe.run_full_dry_run(_require_arg(args, "app_slug"), days=int(args.get("days") or 7), reviewer=args.get("reviewer") or "human")
        elif action == "list_apps":
            result = store.list_apps()
        elif action == "list_drafts":
            result = store.list_drafts(app_slug=args.get("app_slug"))
        elif action == "approve":
            result = store.set_approval(_require_arg(args, "draft_id"), "approved", reviewer=args.get("reviewer") or "human", reason=args.get("reason"))
        elif action == "reject":
            result = store.set_approval(_require_arg(args, "draft_id"), "rejected", reviewer=args.get("reviewer") or "human", reason=args.get("reason"))
        elif action == "edit":
            result = pipe.edit_draft(_require_arg(args, "draft_id"), _require_arg(args, "body"), editor=args.get("editor") or "agent")
        elif action == "regenerate":
            result = pipe.regenerate_draft(_require_arg(args, "draft_id"))
        elif action == "variants":
            result = pipe.generate_variants(_require_arg(args, "draft_id"), count=int(args.get("count") or 3))
        elif action == "reschedule":
            result = store.update_draft_scheduled_for(_require_arg(args, "draft_id"), _require_arg(args, "scheduled_for"))
        elif action == "schedule":
            draft_id = args.get("draft_id")
            if draft_id:
                result = store.schedule_draft(draft_id, _require_arg(args, "scheduled_for"))
            else:
                result = pipe.scheduler.schedule_approved(store, app_slug=args.get("app_slug"))
        elif action == "publish_dry_run":
            draft_id = args.get("draft_id")
            if draft_id:
                result = store.dry_run_publish(draft_id)
            else:
                result = pipe.publisher.dry_run_publish_scheduled(store, app_slug=args.get("app_slug"))
        elif action == "add_app":
            profile = args.get("profile") or {}
            if not profile.get("slug") or not profile.get("name"):
                raise ValueError("profile.slug and profile.name are required")
            store.initialize()
            result = store.upsert_app(profile)
        elif action == "update_app":
            slug = _require_arg(args, "app_slug")
            updates = args.get("profile") or {}
            existing = store.require_app(slug)
            merged = {**existing, **updates, "slug": slug}
            result = store.upsert_app(merged)
        elif action == "remove_app":
            result = store.remove_app(_require_arg(args, "app_slug"))
        elif action == "set_channel_mode":
            result = store.set_channel_mode(
                _require_arg(args, "app_slug"),
                _require_arg(args, "channel"),
                _require_arg(args, "mode"),
                reviewer=args.get("reviewer") or "agent",
            )
        elif action == "set_auto_generate":
            enabled = args.get("enabled")
            if enabled is None:
                raise ValueError("`enabled` is required for set_auto_generate")
            result = store.set_auto_generate(
                _require_arg(args, "app_slug"),
                bool(enabled),
                threshold=args.get("threshold"),
                reviewer=args.get("reviewer") or "agent",
            )
        elif action == "poll":
            result = pipe.poll()
        elif action == "advise":
            result = pipe.advise()
        elif action == "digest":
            result = {
                "app_slug": _require_arg(args, "app_slug"),
                "days": int(args.get("days") or 7),
                "markdown": pipe.weekly_digest(_require_arg(args, "app_slug"), days=int(args.get("days") or 7)),
            }
        elif action == "audit":
            result = store.list_audit(app_slug=args.get("app_slug"))
        else:
            raise ValueError(f"Unknown action: {action}")
        return json.dumps({"success": True, "result": result}, sort_keys=True, default=str)
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)}, sort_keys=True)


def _require_arg(args: Dict[str, Any], name: str) -> str:
    value = args.get(name)
    if not value:
        raise ValueError(f"{name} is required")
    return str(value)
