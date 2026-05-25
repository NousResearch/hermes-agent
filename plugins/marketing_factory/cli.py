"""CLI for the Marketing Agent Factory plugin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from plugins.marketing_factory.pipeline import MarketingFactoryPipeline
from plugins.marketing_factory.store import MarketingFactoryStore


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--store-path", default=None, help="Override store path (defaults to $HERMES_HOME/marketing_factory)")
    subs = subparser.add_subparsers(dest="marketing_command")

    subs.add_parser("init", help="Initialize store and seed Pupular/SetVenue brand profiles")
    subs.add_parser("status", help="Show factory status")
    bootstrap = subs.add_parser("bootstrap", help="Guided first-run setup: init + add your brand + enable autonomy")
    bootstrap.add_argument("--non-interactive", action="store_true", help="Skip prompts and just run init + print next steps")

    gen = subs.add_parser("generate", help="Generate a dry-run campaign and approval drafts for one app")
    gen.add_argument("--app", required=True, help="App slug, e.g. pupular or setvenue")
    gen.add_argument("--days", type=int, default=7)
    gen.add_argument("--auto-approve", action="store_true", help="Approve generated drafts for verification only")

    full = subs.add_parser("full-dry-run", help="Generate, approve, schedule, dry-run publish, and record analytics for one app")
    full.add_argument("--app", required=True)
    full.add_argument("--days", type=int, default=7)
    full.add_argument("--reviewer", default="human")

    apps = subs.add_parser("apps", help="List brand profiles")
    apps.add_argument("--json", action="store_true")

    campaigns = subs.add_parser("campaigns", help="List campaigns")
    campaigns.add_argument("--app", default=None)
    campaigns.add_argument("--json", action="store_true")

    drafts = subs.add_parser("drafts", help="List drafts")
    drafts.add_argument("--app", default=None)
    drafts.add_argument("--status", default=None)
    drafts.add_argument("--json", action="store_true")

    approvals = subs.add_parser("approvals", help="Show approval queue")
    approvals.add_argument("--app", default=None)
    approvals.add_argument("--json", action="store_true")

    approve = subs.add_parser("approve", help="Approve a draft for scheduling")
    approve.add_argument("draft_id")
    approve.add_argument("--reviewer", default="human")
    approve.add_argument("--reason", default="approved for dry-run")

    reject = subs.add_parser("reject", help="Reject a draft")
    reject.add_argument("draft_id")
    reject.add_argument("--reviewer", default="human")
    reject.add_argument("--reason", default="rejected by reviewer")

    schedule = subs.add_parser("schedule", help="Schedule approved drafts")
    schedule.add_argument("--app", default=None)
    schedule.add_argument("--draft-id", default=None)
    schedule.add_argument("--when", default=None, help="ISO timestamp when scheduling one draft")
    schedule.add_argument("--json", action="store_true")

    publish = subs.add_parser("publish-dry-run", help="Dry-run publish scheduled drafts only; never posts publicly")
    publish.add_argument("--app", default=None)
    publish.add_argument("--draft-id", default=None)
    publish.add_argument("--json", action="store_true")

    poll = subs.add_parser("poll", help="One scheduled-poller tick: fire publish on all due drafts across all apps")
    poll.add_argument("--json", action="store_true")

    digest = subs.add_parser("digest", help="Markdown digest of one app's last-N-days activity — share with stakeholders")
    digest.add_argument("--app", required=True)
    digest.add_argument("--days", type=int, default=7)

    analytics = subs.add_parser("analytics", help="Per-app performance rollup over the last N days (approval rate, by-channel, freshness)")
    analytics.add_argument("--app", required=True)
    analytics.add_argument("--days", type=int, default=30)

    regen = subs.add_parser("regenerate", help="Re-roll a single draft with the latest steering applied (keeps the old draft)")
    regen.add_argument("draft_id")

    edit = subs.add_parser("edit", help="Edit a draft's body in place; safety check re-runs automatically")
    edit.add_argument("draft_id")
    edit.add_argument("--body", required=True)
    edit.add_argument("--editor", default="human")

    reschedule = subs.add_parser("reschedule", help="Move a draft's scheduled time (ISO 8601)")
    reschedule.add_argument("draft_id")
    reschedule.add_argument("--at", required=True, dest="scheduled_for", help="New scheduled_for, e.g. 2026-05-22T18:00:00-07:00")

    auto_on = subs.add_parser("enable-auto-generate", help="Turn ON per-app auto-generation when queue is below threshold (cooldown 24h)")
    auto_on.add_argument("--app", required=True)
    auto_on.add_argument("--threshold", type=int, default=3)

    auto_off = subs.add_parser("disable-auto-generate", help="Turn OFF per-app auto-generation")
    auto_off.add_argument("--app", required=True)

    variants = subs.add_parser("variants", help="Generate N alternative drafts for the same plan item — pick the best of N")
    variants.add_argument("draft_id")
    variants.add_argument("--count", type=int, default=3)

    advise = subs.add_parser("advise", help="Run health checks against the factory and print actionable items")
    advise.add_argument("--json", action="store_true")

    enable_poller = subs.add_parser("enable-poller", help="Register a Hermes cron job that runs `marketing-factory poll` on a recurring interval")
    enable_poller.add_argument("--interval", default="5m", help="How often to poll (e.g. 1m, 5m, 1h). Default: 5m.")
    enable_poller.add_argument("--name", default="marketing-factory-poll", help="Cron job display name")

    disable_poller = subs.add_parser("disable-poller", help="Remove the scheduled-poller cron job and its wrapper script")
    disable_poller.add_argument("--name", default="marketing-factory-poll")

    add_app = subs.add_parser("add-app", help="Add or upsert a brand profile without editing Python")
    add_app.add_argument("--slug", required=True)
    add_app.add_argument("--name", required=True)
    add_app.add_argument("--positioning", default="")
    add_app.add_argument("--icp", default="")
    add_app.add_argument("--tone", default="")
    add_app.add_argument("--cta", default="")
    add_app.add_argument("--channels", default="", help="Comma-separated, e.g. x,instagram,tiktok")
    add_app.add_argument("--content-pillars", default="")
    add_app.add_argument("--claims", default="")
    add_app.add_argument("--forbidden-claims", default="")
    add_app.add_argument("--links", default="")
    add_app.add_argument("--competitors", default="")

    update_app = subs.add_parser("update-app", help="Partial-update an existing brand profile (only changes the fields you pass)")
    update_app.add_argument("--slug", required=True)
    update_app.add_argument("--name", default=None)
    update_app.add_argument("--positioning", default=None)
    update_app.add_argument("--icp", default=None)
    update_app.add_argument("--tone", default=None)
    update_app.add_argument("--cta", default=None)
    update_app.add_argument("--channels", default=None, help="Comma-separated; replaces full list")
    update_app.add_argument("--content-pillars", default=None)
    update_app.add_argument("--claims", default=None)
    update_app.add_argument("--forbidden-claims", default=None)
    update_app.add_argument("--links", default=None)
    update_app.add_argument("--competitors", default=None)

    remove_app = subs.add_parser("remove-app", help="Remove a brand profile (cascades drafts/campaigns/etc)")
    remove_app.add_argument("--slug", required=True)
    remove_app.add_argument("--no-cascade", action="store_true", help="Refuse if dependent records exist")

    audit = subs.add_parser("audit", help="Show audit trail")
    audit.add_argument("--app", default=None)
    audit.add_argument("--limit", type=int, default=20)
    audit.add_argument("--json", action="store_true")

    export = subs.add_parser("export", help="Export full state JSON")
    export.add_argument("--output", default=None)

    subparser.set_defaults(func=marketing_command)


def marketing_command(args: argparse.Namespace) -> int:
    sub = getattr(args, "marketing_command", None)
    if not sub:
        print("usage: hermes marketing-factory {init,bootstrap,status,apps,add-app,update-app,remove-app,enable-auto-generate,disable-auto-generate,campaigns,drafts,approvals,approve,reject,regenerate,variants,edit,reschedule,schedule,publish-dry-run,poll,enable-poller,disable-poller,advise,digest,analytics,audit,export,generate,full-dry-run}")
        return 2
    store = MarketingFactoryStore(getattr(args, "store_path", None))
    pipe = MarketingFactoryPipeline(store)
    try:
        if sub == "init":
            result = pipe.initialize_samples()
            _print_json(result)
            return 0
        if sub == "bootstrap":
            return _run_bootstrap(pipe, store, non_interactive=bool(getattr(args, "non_interactive", False)))
        if sub == "status":
            store.initialize()
            _print_json(store.summary())
            return 0
        if sub == "apps":
            _print_records(store.list_apps(), as_json=args.json, title="Apps")
            return 0
        if sub == "campaigns":
            _print_records(store.list_campaigns(app_slug=args.app), as_json=args.json, title="Campaigns")
            return 0
        if sub == "drafts":
            _print_records(store.list_drafts(app_slug=args.app, status=args.status), as_json=args.json, title="Drafts")
            return 0
        if sub == "approvals":
            state = store.load()
            records = list(state["approvals"].values())
            if args.app:
                records = [r for r in records if r.get("app_slug") == args.app]
            _print_records(records, as_json=args.json, title="Approvals")
            return 0
        if sub == "approve":
            _print_json(store.set_approval(args.draft_id, "approved", reviewer=args.reviewer, reason=args.reason))
            return 0
        if sub == "reject":
            _print_json(store.set_approval(args.draft_id, "rejected", reviewer=args.reviewer, reason=args.reason))
            return 0
        if sub == "schedule":
            if args.draft_id:
                if not args.when:
                    raise ValueError("--when is required with --draft-id")
                result = [store.schedule_draft(args.draft_id, args.when)]
            else:
                result = pipe.scheduler.schedule_approved(store, app_slug=args.app)
            _print_records(result, as_json=args.json, title="Scheduled")
            return 0
        if sub == "publish-dry-run":
            if args.draft_id:
                result = [store.dry_run_publish(args.draft_id)]
            else:
                result = pipe.publisher.dry_run_publish_scheduled(store, app_slug=args.app)
            _print_records(result, as_json=args.json, title="Dry-run publish events")
            return 0
        if sub == "add-app":
            store.initialize()
            profile = _profile_from_args(args, slug=args.slug, name=args.name)
            result = store.upsert_app(profile)
            _print_json({"slug": result["slug"], "channels": result.get("channels", []), "channel_modes": result.get("channel_modes", {})})
            return 0
        if sub == "update-app":
            existing = store.require_app(args.slug)
            updates = _profile_updates_from_args(args)
            if not updates:
                print("update-app: nothing to update (no fields provided)")
                return 2
            merged = {**existing, **updates, "slug": args.slug}
            result = store.upsert_app(merged)
            _print_json({"slug": result["slug"], "updated_fields": sorted(updates.keys())})
            return 0
        if sub == "remove-app":
            result = store.remove_app(args.slug, cascade=not args.no_cascade)
            _print_json(result)
            return 0
        if sub == "analytics":
            _print_json(pipe.app_analytics(args.app, days=args.days))
            return 0
        if sub == "digest":
            print(pipe.weekly_digest(args.app, days=args.days))
            return 0
        if sub == "enable-auto-generate":
            result = store.set_auto_generate(args.app, True, threshold=args.threshold, reviewer="cli")
            _print_json(result)
            return 0
        if sub == "disable-auto-generate":
            result = store.set_auto_generate(args.app, False, reviewer="cli")
            _print_json(result)
            return 0
        if sub == "reschedule":
            result = store.update_draft_scheduled_for(args.draft_id, args.scheduled_for)
            _print_json({"draft_id": result["id"], "scheduled_for": result["scheduled_for"], "status": result["status"]})
            return 0
        if sub == "edit":
            result = pipe.edit_draft(args.draft_id, args.body, editor=args.editor)
            _print_json({
                "draft_id": result["id"],
                "edited_by": result.get("edited_by"),
                "edited_at": result.get("edited_at"),
                "new_length": len(result.get("body") or ""),
                "safety_passed": (result.get("safety") or {}).get("passed"),
            })
            return 0
        if sub == "variants":
            result = pipe.generate_variants(args.draft_id, count=args.count)
            _print_json({
                "source_draft_id": result["source_draft_id"],
                "count_generated": result["count_generated"],
                "variant_ids": [v["id"] for v in result["variants"]],
            })
            return 0
        if sub == "regenerate":
            result = pipe.regenerate_draft(args.draft_id)
            _print_json({
                "old_draft_id": result["old_draft_id"],
                "new_draft_id": result["new_draft"]["id"],
                "channel": result["new_draft"]["channel"],
                "steering_applied": result["steering_applied"],
                "llm_used": result["new_draft"].get("llm_used"),
                "safety_passed": (result["new_draft"].get("safety") or {}).get("passed"),
            })
            return 0
        if sub == "advise":
            result = pipe.advise()
            if args.json:
                _print_json(result)
            else:
                if result["healthy"]:
                    print("OK: no advisor items.")
                else:
                    print(f"{len(result['items'])} advisor item(s):")
                    for item in result["items"]:
                        prefix = f"[{item['severity'].upper()}]"
                        scope = f" {item['app_slug']}" if item.get("app_slug") else ""
                        print(f"  {prefix}{scope}: {item['message']}")
                        print(f"      → {item['action']}")
            return 0
        if sub == "enable-poller":
            from hermes_constants import get_hermes_home
            from cron.jobs import create_job, list_jobs, remove_job
            scripts_dir = get_hermes_home() / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            script_name = "marketing-factory-poll.sh"
            script_path = scripts_dir / script_name
            script_path.write_text("#!/usr/bin/env bash\nexec hermes marketing-factory poll \"$@\"\n", encoding="utf-8")
            script_path.chmod(0o755)
            # Remove any existing job with the same name so this is idempotent.
            for existing_job in list_jobs(include_disabled=True):
                if existing_job.get("name") == args.name:
                    remove_job(existing_job["id"])
            job = create_job(
                prompt=None,
                schedule=f"every {args.interval}",
                name=args.name,
                script=script_name,
                no_agent=True,
            )
            _print_json({"job_id": job["id"], "name": job.get("name"), "schedule": job.get("schedule"), "script": script_name, "scripts_dir": str(scripts_dir)})
            return 0
        if sub == "disable-poller":
            from hermes_constants import get_hermes_home
            from cron.jobs import list_jobs, remove_job
            removed = []
            for existing_job in list_jobs(include_disabled=True):
                if existing_job.get("name") == args.name:
                    remove_job(existing_job["id"])
                    removed.append(existing_job["id"])
            script_path = get_hermes_home() / "scripts" / "marketing-factory-poll.sh"
            script_removed = False
            if script_path.exists():
                script_path.unlink()
                script_removed = True
            _print_json({"removed_jobs": removed, "script_removed": script_removed})
            return 0
        if sub == "poll":
            result = pipe.poll()
            summary = {
                "polled_apps": result["polled_apps"],
                "due_count": result["due_count"],
                "fired_count": result["fired_count"],
                "last_poll_at": result["last_poll"].get("last_poll_at"),
            }
            if args.json:
                _print_json(result)
            else:
                _print_json(summary)
            return 0
        if sub == "audit":
            _print_records(store.list_audit(app_slug=args.app, limit=args.limit), as_json=args.json, title="Audit")
            return 0
        if sub == "export":
            state = store.load()
            if args.output:
                Path(args.output).write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                print(args.output)
            else:
                _print_json(state)
            return 0
        if sub == "generate":
            result = pipe.generate_campaign(args.app, days=args.days, auto_approve=args.auto_approve)
            _print_json({"campaign": result["campaign"], "draft_count": len(result["drafts"]), "draft_ids": [d["id"] for d in result["drafts"]]})
            return 0
        if sub == "full-dry-run":
            result = pipe.run_full_dry_run(args.app, days=args.days, reviewer=args.reviewer)
            _print_json({
                "campaign_id": result["generated"]["campaign"]["id"],
                "draft_count": len(result["generated"]["drafts"]),
                "approvals": len(result["approvals"]),
                "scheduled": len(result["schedules"]),
                "dry_run_publish_events": len(result["publish_events"]),
                "analytics_id": result["analytics"]["id"],
            })
            return 0
    except Exception as exc:
        print(f"marketing-factory error: {exc}")
        return 1
    print(f"Unknown command: {sub}")
    return 2


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _ask(prompt: str, default: str = "") -> str:
    """Single interactive prompt with default. Returns empty string on EOF/Ctrl-D."""
    suffix = f" [{default}]" if default else ""
    try:
        raw = input(f"  {prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""
    return raw or default


def _ask_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        raw = input(f"  {prompt} {suffix}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if not raw:
        return default
    return raw in {"y", "yes", "1", "true"}


def _run_bootstrap(pipe, store, *, non_interactive: bool = False) -> int:
    """First-run wizard. Initializes samples, optionally adds a new brand,
    optionally turns on auto-generate, optionally installs the cron poller."""
    print("\n========================================")
    print("  Marketing Factory — bootstrap wizard")
    print("========================================\n")

    # Step 1 — always initialize the sample profiles (idempotent)
    print("→ Initializing store + seeding sample brand profiles (pupular, setvenue)…")
    init_result = pipe.initialize_samples()
    apps_seeded = [a["slug"] for a in init_result.get("apps", [])]
    print(f"  ok · {len(apps_seeded)} sample brand(s): {', '.join(apps_seeded)}\n")

    if non_interactive:
        _print_next_steps(store)
        return 0

    # Step 2 — optional: add user's own brand
    if _ask_yes_no("Do you want to add your own brand profile right now?", default=False):
        slug = _ask("Brand slug (lowercase, e.g. wingman or hardline)")
        if not slug:
            print("  (skipped — no slug given)\n")
        else:
            name = _ask("Display name", default=slug.capitalize())
            positioning = _ask("Positioning (one sentence about the brand)")
            icp = _ask("Ideal customer (who uses this?)")
            tone = _ask("Tone (e.g. 'cute warm playful' or 'trustworthy clear')", default="clear, helpful")
            cta = _ask("Default call-to-action", default=f"Try {name} today.")
            channels_raw = _ask("Channels (comma-separated: x,instagram,tiktok,linkedin,blog,email,app_store)", default="x,instagram")
            channels = [c.strip() for c in channels_raw.split(",") if c.strip()]
            pillars_raw = _ask("Content pillars (recurring themes, comma-separated)", default="")
            pillars = [p.strip() for p in pillars_raw.split(",") if p.strip()]
            forbidden_raw = _ask("Forbidden claims (things you must NEVER promise, comma-separated)", default="")
            forbidden = [f.strip() for f in forbidden_raw.split(",") if f.strip()]
            link = _ask("Primary link (URL)", default="")
            try:
                store.upsert_app({
                    "slug": slug,
                    "name": name,
                    "positioning": positioning,
                    "icp": icp,
                    "tone": tone,
                    "cta": cta,
                    "channels": channels,
                    "content_pillars": pillars,
                    "forbidden_claims": forbidden,
                    "links": [link] if link else [],
                    "claims": [],
                    "competitors": [],
                    "assets": [],
                })
                print(f"  ok · added {slug}\n")

                # Step 3 — optional: turn on auto-generate for the new brand
                if _ask_yes_no(f"Enable auto-generate for {slug}? (When queue dips below threshold, the cron poller will auto-fire a new 7-day campaign, max once every 24h)", default=False):
                    store.set_auto_generate(slug, True, threshold=3, reviewer="bootstrap")
                    print(f"  ok · auto-generate ON for {slug} (threshold 3)\n")
            except Exception as exc:
                print(f"  ! failed to add brand: {exc}\n")

    # Step 4 — optional: install the cron poller
    if _ask_yes_no("Install the Hermes cron poller? (Runs `marketing-factory poll` every 5 minutes — fires publish on due drafts and triggers auto-generate on opt-in brands)", default=False):
        try:
            from hermes_constants import get_hermes_home
            from cron.jobs import create_job, list_jobs, remove_job
            scripts_dir = get_hermes_home() / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            script_path = scripts_dir / "marketing-factory-poll.sh"
            script_path.write_text("#!/usr/bin/env bash\nexec hermes marketing-factory poll \"$@\"\n", encoding="utf-8")
            script_path.chmod(0o755)
            for existing_job in list_jobs(include_disabled=True):
                if existing_job.get("name") == "marketing-factory-poll":
                    remove_job(existing_job["id"])
            job = create_job(prompt=None, schedule="every 5m", name="marketing-factory-poll", script="marketing-factory-poll.sh", no_agent=True)
            print(f"  ok · poller installed (job id {job['id']}, every 5 minutes)\n")
        except Exception as exc:
            print(f"  ! could not install poller: {exc}\n")

    _print_next_steps(store)
    return 0


def _print_next_steps(store) -> None:
    print("\n----------------------------------------")
    print("  You're set up. What to do next:\n")
    apps = [a["slug"] for a in store.list_apps()]
    print(f"  • Dashboard:   http://127.0.0.1:9119/marketing-factory")
    print(f"  • Apps ready:  {', '.join(apps) if apps else '(none yet)'}")
    print(f"\n  Generate your first real campaign:")
    primary = apps[0] if apps else "pupular"
    print(f"      hermes marketing-factory generate --app {primary} --days 7\n")
    print(f"  Or click the big 'Make new content' button in the dashboard.")
    print(f"\n  Open Settings (⚙) in the dashboard to:")
    print(f"      • toggle auto-generate per brand")
    print(f"      • see cost / token spend today")
    print(f"      • copy a weekly digest")
    print("----------------------------------------\n")


def _split_csv(value: Optional[str]) -> Optional[list]:
    if value is None:
        return None
    items = [piece.strip() for piece in value.split(",") if piece.strip()]
    return items


def _profile_from_args(args, *, slug: str, name: str) -> Dict[str, Any]:
    return {
        "slug": slug,
        "name": name,
        "positioning": args.positioning,
        "icp": args.icp,
        "tone": args.tone,
        "cta": args.cta,
        "channels": _split_csv(args.channels) or [],
        "content_pillars": _split_csv(args.content_pillars) or [],
        "claims": _split_csv(args.claims) or [],
        "forbidden_claims": _split_csv(args.forbidden_claims) or [],
        "links": _split_csv(args.links) or [],
        "competitors": _split_csv(args.competitors) or [],
    }


def _profile_updates_from_args(args) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}
    if args.name is not None:
        updates["name"] = args.name
    if args.positioning is not None:
        updates["positioning"] = args.positioning
    if args.icp is not None:
        updates["icp"] = args.icp
    if args.tone is not None:
        updates["tone"] = args.tone
    if args.cta is not None:
        updates["cta"] = args.cta
    for field_name, csv_value in (
        ("channels", args.channels),
        ("content_pillars", args.content_pillars),
        ("claims", args.claims),
        ("forbidden_claims", args.forbidden_claims),
        ("links", args.links),
        ("competitors", args.competitors),
    ):
        parsed = _split_csv(csv_value)
        if parsed is not None:
            updates[field_name] = parsed
    return updates


def _print_records(records: Any, *, as_json: bool, title: str) -> None:
    if as_json:
        _print_json(records)
        return
    records = list(records)
    print(f"{title}: {len(records)}")
    for record in records:
        bits = [record.get("id") or record.get("slug") or "<unknown>"]
        for key in ("app_slug", "name", "channel", "status", "scheduled_for"):
            if record.get(key):
                bits.append(f"{key}={record[key]}")
        print("- " + " | ".join(bits))
