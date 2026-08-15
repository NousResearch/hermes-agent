#!/usr/bin/env python3
"""Denji profile review execution engine - scheduled weekly/monthly/quarterly scans.

Reads profile configs, session stats, ledger activity, and file timestamps to produce
structured review entries in the central activity ledger as ``profile.review.*`` events.

Each review cycle has a dedicated event type:
  - ``profile.review.weekly``   - lightweight activity snapshot
  - ``profile.review.monthly``  - mid-weight: usage + config drift + auto-promotions
  - ``profile.review.quarterly`` - full audit: session history + config + identity + trends

All findings are tamper-evident: append-only ledger events with per-profile payloads.
Exportable via any ``query_events(event_types=['profile.review.*'])`` call.

Usage:
  python3 denji-review-cycle.py --cycle weekly    # lightweight
  python3 denji-review-cycle.py --cycle monthly   # mid-weight
  python3 denji-review-cycle.py --cycle quarterly # full audit
"""

import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_cli.profile_activity_ledger import append_event, query_events

# ── Paths ────────────────────────────────────────────────────────────────────

HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
PROFILES_DIR = HERMES_HOME / "profiles"
LEDGER_DB = HERMES_HOME / "governance" / "profile-activity-ledger.sqlite"
LOGBOARD = HERMES_HOME / "governance" / "logboard"

# ── Profile discovery ────────────────────────────────────────────────────────

def _all_profiles() -> list[str]:
    names = []
    if (HERMES_HOME / "config.yaml").exists():
        names.append("default")
    if PROFILES_DIR.exists():
        for d in sorted(PROFILES_DIR.iterdir()):
            if d.is_dir() and (d / "config.yaml").exists():
                names.append(d.name)
    return names


def _config_path(profile: str) -> Path | None:
    if profile == "default":
        p = HERMES_HOME / "config.yaml"
    else:
        p = PROFILES_DIR / profile / "config.yaml"
    return p if p.exists() else None


# ── YAML reading ─────────────────────────────────────────────────────────────

def _safe_read_yaml(path: Path) -> dict:
    import yaml as _pyyaml
    try:
        return _pyyaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


# ── Profile file stats ───────────────────────────────────────────────────────

def _file_stats(profile: str) -> dict[str, dict]:
    home = HERMES_HOME if profile == "default" else PROFILES_DIR / profile
    files = ["SOUL.md", "USER.md", "config.yaml"]
    result = {}
    for fn in files:
        f = home / fn if fn != "config.yaml" else _config_path(profile)
        if not f or not f.exists():
            result[fn] = {"exists": False}
            continue
        try:
            st = f.stat()
            result[fn] = {
                "exists": True,
                "size": st.st_size,
                "mtime_epoch": int(st.st_mtime),
                "mtime_iso": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
            }
        except OSError:
            result[fn] = {"exists": True, "error": "stat failed"}
    return result


# ── Ledger queries ────────────────────────────────────────────────────────────

def _ledger_counts(profile: str, since: int | None = None) -> dict:
    """Return activity counts for a profile in the window."""
    db = LEDGER_DB
    if not db.exists():
        return {}
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        params = [profile]
        where = "WHERE (actor_profile = ? OR target_profile = ?)"
        if since:
            where += " AND occurred_at >= ?"
            params.extend([profile, str(since)])

        rows = con.execute(
            f"SELECT event_type, COUNT(*) as cnt FROM activity_events {where} GROUP BY event_type",
            params,
        ).fetchall()
        con.close()
        return {r["event_type"]: r["cnt"] for r in rows}
    except Exception:
        return {}


# ── Review generators ─────────────────────────────────────────────────────────

def _week_review(profile: str, since: int) -> dict:
    """Lightweight: activity snapshot + file changes."""
    counts = _ledger_counts(profile, since)
    files = _file_stats(profile)

    total_activity = sum(counts.values())
    skills_related = sum(
        v for k, v in counts.items()
        if k.startswith("skill.")
    )

    return {
        "cycle": "weekly",
        "profile": profile,
        "since_epoch": since,
        "total_events": total_activity,
        "skills_events": skills_related,
        "top_event_types": sorted(counts.items(), key=lambda kv: -kv[1])[:5],
        "files": files,
        "recommendation": _assess_activity(total_activity, profile),
    }


def _month_review(profile: str, since: int) -> dict:
    """Mid-weight: usage + config drift + auto-promotions."""
    counts = _ledger_counts(profile, since)
    files = _file_stats(profile)
    cfg_path = _config_path(profile)
    cfg = _safe_read_yaml(cfg_path) if cfg_path else {}
    skills = (cfg.get("skills") or {})
    enabled = skills.get("enabled_skills") or []

    total_activity = sum(counts.values())
    auto_promotions = counts.get("skill.enabled_auto", 0)

    return {
        "cycle": "monthly",
        "profile": profile,
        "since_epoch": since,
        "total_events": total_activity,
        "enabled_skills_count": len(enabled),
        "always_skills_count": len(skills.get("always_skills") or []),
        "auto_promotions": auto_promotions,
        "files": files,
        "recommendation": _assess_activity(total_activity, profile),
    }


def _quarter_review(profile: str, since: int) -> dict:
    """Full audit: session history + config + identity + trends."""
    counts = _ledger_counts(profile, since)
    files = _file_stats(profile)
    cfg_path = _config_path(profile)
    cfg = _safe_read_yaml(cfg_path) if cfg_path else {}
    skills = (cfg.get("skills") or {})

    # Also check SOUL.md and USER.md content
    home = HERMES_HOME if profile == "default" else PROFILES_DIR / profile
    soul_exists = (home / "SOUL.md").exists()
    user_exists = (home / "USER.md").exists()

    total_activity = sum(counts.values())
    borrow_count = counts.get("skill.borrowed", 0)
    load_count = counts.get("skill.loaded", 0)
    deny_count = counts.get("skill.denied", 0) + counts.get("skill.access.blocked", 0)

    return {
        "cycle": "quarterly",
        "profile": profile,
        "since_epoch": since,
        "total_events": total_activity,
        "enabled_skills_count": len(skills.get("enabled_skills") or []),
        "always_skills_count": len(skills.get("always_skills") or []),
        "borrow_count": borrow_count,
        "load_count": load_count,
        "deny_count": deny_count,
        "soul_exists": soul_exists,
        "user_exists": user_exists,
        "files": files,
        "recommendation": _assess_activity(total_activity),
    }


def _assess_activity(total: int, profile: str = "") -> str:
    # Gateway/Tier-1 override: profiles that run an active systemd gateway or
    # are listed Tier 1 in the registry are NEVER dormant, even with zero
    # ledger events (they work through the gateway ticker, not the ledger).
    if profile and _is_active_gateway(profile):
        return "active - gateway lead (tier-1 override)"
    if total >= 100:
        return "active - profile is in regular use"
    elif total >= 10:
        return "low activity - monitor for obsolescence"
    else:
        return "dormant - consider archival or removal"


def _is_active_gateway(profile: str) -> bool:
    """True if the profile runs an active systemd gateway unit or is Tier 1 in the registry."""
    try:
        # 1) systemd gateway unit check (profiles run hermes-gateway-<name>.service)
        unit = f"hermes-gateway-{profile}.service"
        r = subprocess.run(
            ["systemctl", "is-active", unit],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip() == "active":
            return True
    except Exception:
        pass
    try:
        # 2) profile-tier-registry.md Tier-1 check (fallback when unit name differs)
        reg = HERMES_HOME / "governance" / "profile-tier-registry.md"
        if reg.exists():
            for line in reg.read_text(errors="ignore").splitlines():
                if profile.lower() in line.lower() and "tier" in line.lower() and "1" in line:
                    # Require an explicit tier-1 designation rather than a random "1" digit
                    low = line.lower()
                    if "tier 1" in low or "tier-1" in low or "tier1" in low:
                        return True
    except Exception:
        pass
    return False


# ── Main ──────────────────────────────────────────────────────────────────────

def _run_cycle(cycle: str) -> str:
    now = int(time.time())
    if cycle == "weekly":
        since = now - 7 * 86400
        review_fn = _week_review
    elif cycle == "monthly":
        since = now - 30 * 86400
        review_fn = _month_review
    elif cycle == "quarterly":
        since = now - 90 * 86400
        review_fn = _quarter_review
    else:
        print(f"Unknown cycle: {cycle}")
        sys.exit(1)

    event_type = f"profile.review.{cycle}"
    profiles = _all_profiles()

    results = []
    for profile in sorted(profiles):
        findings = review_fn(profile, since)
        event_id = f"review-{cycle}-{profile}-{now}"
        payload = json.dumps(findings)

        append_event(
            source="denji-review-cycle",
            event_type=event_type,
            event_id=event_id,
            actor_profile="denji",
            target_profile=profile,
            object_type="profile.review",
            summary=f"{cycle.capitalize()} review for {profile}: {findings['recommendation']}",
            payload=findings,
            occurred_at=now,
        )
        results.append((profile, findings["recommendation"]))

    # Summary — only print when there are changes from last cycle
    active = sum(1 for _, r in results if "active" in r)
    low = sum(1 for _, r in results if "low activity" in r)
    dormant = sum(1 for _, r in results if "dormant" in r)

    # Compare against previous artifact to detect changes
    prev_artifacts = sorted(LOGBOARD.glob(f"profile-review-{cycle}-*.json"), reverse=True) if LOGBOARD.exists() else []
    prev_recs = {}
    if prev_artifacts:
        try:
            prev = json.loads(prev_artifacts[0].read_text())
            for entry in prev.get("results", []):
                prev_recs[entry["profile"]] = entry["recommendation"]
        except Exception:
            pass

    # Count changes
    changed = []
    for profile, rec in results:
        prev_rec = prev_recs.get(profile)
        if prev_rec != rec:
            changed.append((profile, prev_rec, rec))

    # Write JSON artifact to logboard for reference (always)
    LOGBOARD.mkdir(parents=True, exist_ok=True)
    artifact = {
        "cycle": cycle,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timestamp_epoch": now,
        "profiles_reviewed": len(results),
        "results": [
            {
                "profile": p,
                "recommendation": r,
            }
            for p, r in results
        ],
    }
    artifact_path = LOGBOARD / f"profile-review-{cycle}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    artifact_path.write_text(json.dumps(artifact, indent=2))

    # Only deliver to Discord when there are changes
    if not changed:
        return event_type  # silent — no changes from last cycle

    print(f"Denji Review Cycle - {cycle} - {len(profiles)} profiles")
    print(f"Changes: {len(changed)} (of {len(results)} reviewed)")
    print(f"Summary: {active} active, {low} low, {dormant} dormant")
    print(f"Events recorded to ledger: {len(results)} × {event_type}")
    for profile, prev_rec, new_rec in changed[:10]:
        prev_short = (prev_rec or "new")[:40]
        print(f"  {profile}: {prev_short} → {new_rec[:40]}")
    print(f"Artifact: {artifact_path}")

    return event_type


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Denji profile review cycle executor")
    ap.add_argument("--cycle", required=True, choices=["weekly", "monthly", "quarterly"])
    args = ap.parse_args()
    _run_cycle(args.cycle)
