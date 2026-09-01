#!/usr/bin/env python3
"""Denji profile review execution engine - scheduled weekly/monthly/quarterly scans.

P3.2 — four-dimensional evidence review.  Activity-only "dormant" conclusions
are replaced by four separate evidence dimensions:

  1. ``runtime``    — effective gateway/heartbeat/service state
  2. ``workload``   — Kanban assignments/runs plus direct delegation events
  3. ``quality``    — failures, rework, review outcomes, recurring findings
  4. ``capability`` — config validity, skill/tool denials, missing dependencies

Allowed verdicts: HEALTHY | OBSERVE | ACTION REQUIRED | COLD/STANDBY.
Dimensions are never collapsed into a synthetic score.  Each dimension
records its observation window, source refs, evidence counts and reason.

Key invariants:
  * Active gateways are NEVER labelled dormant solely because ledger volume
    is low (gateway activity does not flow through the ledger).
  * Direct delegation events count as workload and preserve specialist
    identity (Phase 2 telemetry).
  * Registry lifecycle (organisational state) and runtime evidence stay
    distinct: ``standby`` is not "inactive service".
  * Monthly scope = active + changed profiles; quarterly = full roster.
  * Structured event output stays append-only; no profile mutation.

Reads profile configs, session stats, ledger activity, and file timestamps to
produce structured review entries in the central activity ledger as
``profile.review.*`` events.

Usage:
  python3 denji-review-cycle.py --cycle weekly
  python3 denji-review-cycle.py --cycle monthly
  python3 denji-review-cycle.py --cycle quarterly
"""

import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_cli.profile_activity_ledger import append_event, query_events

# ── Paths ────────────────────────────────────────────────────────────────────

HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
PROFILES_DIR = HERMES_HOME / "profiles"
LEDGER_DB = HERMES_HOME / "governance" / "profile-activity-ledger.sqlite"
LOGBOARD = HERMES_HOME / "governance" / "logboard"
REGISTRY_YAML = HERMES_HOME / "governance" / "profile-registry.yaml"

ALLOWED_VERDICTS = ("HEALTHY", "OBSERVE", "ACTION REQUIRED", "COLD/STANDBY")

REVIEW_VERSION = "4dim-1"

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


# ── Registry lifecycle (organisational state) ───────────────────────────────

def _registry_lifecycle() -> dict[str, str]:
    """Profile name -> registry lifecycle, from the deployed registry.

    Read-only; missing/unreadable registry returns {} (organisational state
    is then simply absent from the review — never inferred).
    """
    try:
        import yaml as _pyyaml
        raw = _pyyaml.safe_load(REGISTRY_YAML.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    if raw.get("schema_version") != 1 or not isinstance(raw.get("profiles"), list):
        return {}
    return {
        e["name"]: e.get("lifecycle") or "active"
        for e in raw["profiles"]
        if isinstance(e, dict) and isinstance(e.get("name"), str)
    }


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

def _ledger_counts(profile: str, since: int | None = None,
                   actor_only: bool = False) -> dict:
    """Return activity counts for a profile in the window."""
    db = LEDGER_DB
    if not db.exists():
        return {}
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        params: list = []
        clauses = []
        if actor_only:
            clauses.append("actor_profile = ?")
        else:
            clauses.append("(actor_profile = ? OR target_profile = ?)")
            params.append(profile)
        params.insert(0, profile)
        if since:
            clauses.append("occurred_at >= ?")
            params.append(str(since))
        where = "WHERE " + " AND ".join(clauses)
        rows = con.execute(
            f"SELECT event_type, COUNT(*) as cnt FROM activity_events {where} GROUP BY event_type",
            params,
        ).fetchall()
        con.close()
        return {r["event_type"]: r["cnt"] for r in rows}
    except Exception:
        return {}


# ── Dimension 1: runtime ─────────────────────────────────────────────────────

# C8: explicit default/root identity mapping — the default profile IS the
# base gateway, not a "default" named unit.
_DEFAULT_PROFILE_GATEWAY_UNIT = "hermes-gateway.service"


def _registry_gateway_unit(profile: str, registry_index: Optional[dict] = None) -> Optional[str]:
    """Resolve the registry-declared gateway unit for a profile.

    C8: consumes the registry ``gateway_unit`` mapping (the authority),
    with the explicit default/root identity mapping applied first.  The
    legacy ``hermes-gateway-{profile}.service`` convention is only a
    fallback when the registry does not declare a unit.
    """
    if profile == "default":
        return _DEFAULT_PROFILE_GATEWAY_UNIT
    if registry_index is None:
        registry_index = _registry_lifecycle_index()
    entry = (registry_index or {}).get(profile)
    if isinstance(entry, dict) and entry.get("gateway_unit"):
        unit = str(entry["gateway_unit"])
        return unit if unit.endswith(".service") else f"{unit}.service"
    return f"hermes-gateway-{profile}.service"


def _registry_lifecycle_index() -> dict[str, dict]:
    """Profile name -> registry entry (full mapping incl. gateway_unit)."""
    try:
        import yaml as _pyyaml
        raw = _pyyaml.safe_load(REGISTRY_YAML.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    if raw.get("schema_version") != 1 or not isinstance(raw.get("profiles"), list):
        return {}
    return {
        e["name"]: e
        for e in raw["profiles"]
        if isinstance(e, dict) and isinstance(e.get("name"), str)
    }


def _runtime_dimension(profile: str, since: int) -> dict:
    """Effective gateway/heartbeat/service state.

    C8: unit identity comes from the registry (or the default/root
    mapping); systemd-uncheckable evidence is UNKNOWN, never inactive.
    """
    window_days = (int(time.time()) - since) // 86400
    registry_index = _registry_lifecycle_index()
    unit = _registry_gateway_unit(profile, registry_index)
    evidence = {
        "window_days": window_days,
        "gateway_unit": unit,
        "unit_active": None,
        "source_refs": ["systemctl is-active"],
    }
    unit_active = None
    try:
        r = subprocess.run(
            ["systemctl", "is-active", unit],
            capture_output=True, text=True, timeout=5,
        )
        unit_active = (r.returncode == 0 and r.stdout.strip() == "active")
    except Exception:
        unit_active = None  # UNKNOWN — cannot verify
    evidence["unit_active"] = unit_active
    # Ledger heartbeat evidence (kanban.heartbeat events are runtime proxies)
    counts = _ledger_counts(profile, since, actor_only=True)
    heartbeats = counts.get("kanban.heartbeat", 0)
    evidence["heartbeat_events"] = heartbeats
    evidence["source_refs"].append("profile-activity-ledger:kanban.heartbeat")

    if unit_active is True:
        verdict = "ACTIVE"
    elif unit_active is None:
        verdict = "UNKNOWN"
    elif unit_active is False and heartbeats > 0:
        verdict = "ACTIVE"  # runtime proof without a unit (on-demand profile)
    else:
        verdict = "INACTIVE"
    return {"dimension": "runtime", "verdict": verdict, "evidence": evidence,
            "method_version": REVIEW_VERSION}


# ── Dimension 2: workload ────────────────────────────────────────────────────

def _workload_dimension(profile: str, since: int) -> dict:
    """Kanban assignments/runs plus direct delegation events.

    Direct delegation events preserve specialist identity (Phase 2): they
    count here as workload even when no kanban row exists.
    """
    counts = _ledger_counts(profile, since)
    kanban_assignments = sum(
        v for k, v in counts.items() if k.startswith("kanban.assigned")
    )
    kanban_claims = sum(
        v for k, v in counts.items() if k.startswith("kanban.claimed")
    )
    # Direct delegation — actor or target identity preserved
    delegation_counts = _delegation_counts(profile, since)
    evidence = {
        "window_days": (int(time.time()) - since) // 86400,
        "kanban_assignments": kanban_assignments,
        "kanban_claims": kanban_claims,
        "direct_delegations": delegation_counts.get("delegation.started", 0),
        "delegations_received": delegation_counts.get("delegation.received", 0),
        "source_refs": [
            "profile-activity-ledger:kanban.*",
            "profile-activity-ledger:delegation.started",
        ],
        "method_version": REVIEW_VERSION,
    }
    total = (kanban_assignments + kanban_claims
             + delegation_counts.get("delegation.started", 0)
             + delegation_counts.get("delegation.received", 0))
    verdict = "EVIDENT" if total > 0 else "ABSENT"
    return {"dimension": "workload", "verdict": verdict, "evidence": evidence}


def _delegation_counts(profile: str, since: int) -> dict:
    """Count direct delegation events, preserving specialist identity.

    ``delegation.started`` events with the profile as actor (delegate_tool
    Phase 2 telemetry).  Read-only aggregate query; no payload bodies read.
    """
    db = LEDGER_DB
    if not db.exists():
        return {}
    # Live event taxonomy: only ``delegation.started`` exists; direction is
    # expressed by which profile is actor vs target.  Both counts read the
    # same event type from the appropriate identity column.
    out = {"delegation.started": 0, "delegation.received": 0}
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        started_as_actor = con.execute(
            "SELECT COUNT(*) FROM activity_events "
            "WHERE event_type = 'delegation.started' AND actor_profile = ? "
            "AND occurred_at >= ?",
            (profile, int(since)),
        ).fetchone()
        received_as_target = con.execute(
            "SELECT COUNT(*) FROM activity_events "
            "WHERE event_type = 'delegation.started' AND target_profile = ? "
            "AND occurred_at >= ?",
            (profile, int(since)),
        ).fetchone()
        out["delegation.started"] = int(started_as_actor[0]) if started_as_actor else 0
        out["delegation.received"] = int(received_as_target[0]) if received_as_target else 0
        con.close()
    except Exception:
        pass
    return out


# ── Dimension 3: quality ─────────────────────────────────────────────────────

def _open_finding_ids(profile: str, since: int) -> tuple[list[str], dict[str, int]]:
    """R2-10: compute currently-open findings by IDENTITY and LATEST state.

    Reads (object_id, event_type, occurred_at) for governance.finding.*
    events of the profile and reduces each finding identity to its latest
    state: open when the latest event is opened/updated, closed when
    resolved/dismissed.  ``since`` bounds which findings are reported
    (findings last touched before the window are not reported), but state
    history extends before the window so opened-before/resolved-inside is
    handled correctly.

    Returns (open_ids, counts) — safe IDs/counts only, never payloads.
    counts includes ``resolved_transitions``: resolved/dismissed events in
    the identity history of currently-open findings (the reopened signal).
    """
    db = LEDGER_DB
    open_ids: list[str] = []
    counts = {"opened": 0, "resolved": 0, "open": 0, "resolved_transitions": 0}
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        rows = con.execute(
            "SELECT object_id, event_type, occurred_at FROM activity_events "
            "WHERE event_type IN ('governance.finding.opened', "
            "'governance.finding.updated', 'governance.finding.resolved', "
            "'governance.finding.dismissed') "
            "AND (actor_profile = ? OR target_profile = ?) "
            "AND object_id IS NOT NULL AND object_id != '' "
            "ORDER BY object_id, occurred_at ASC, id ASC",
            (profile, profile),
        ).fetchall()
        con.close()
    except Exception:
        return open_ids, counts
    history: dict[str, list[tuple[str, int]]] = {}
    for object_id, event_type, occurred_at in rows:
        if occurred_at is None:
            continue
        history.setdefault(object_id, []).append((event_type, int(occurred_at)))
    resolved_transitions = 0
    for object_id, events in history.items():
        last_occurred = events[-1][1]
        if last_occurred < since:
            continue  # last activity predates the window
        latest_type = max(events, key=lambda e: e[1])[0]
        is_open = latest_type in (
            "governance.finding.opened", "governance.finding.updated")
        if is_open:
            open_ids.append(object_id)
            # recurrence evidence: this identity was closed at some point
            resolved_transitions += sum(
                1 for t, _ in events
                if t in ("governance.finding.resolved", "governance.finding.dismissed"))
        if latest_type in ("governance.finding.opened", "governance.finding.updated"):
            counts["opened"] += 1
        else:
            counts["resolved"] += 1
    counts["open"] = len(open_ids)
    counts["resolved_transitions"] = resolved_transitions
    return open_ids, counts


def _quality_dimension(profile: str, since: int) -> dict:
    counts = _ledger_counts(profile, since)
    failures = sum(
        v for k, v in counts.items()
        if k in ("kanban.crashed", "kanban.gave_up", "job_run_error",
                 "kanban.completion_blocked_hallucination")
    )
    rework = sum(
        v for k, v in counts.items() if k in ("kanban.council_revise", "kanban.audit_revise")
    )
    review_outcomes = sum(
        v for k, v in counts.items() if k.startswith("kanban.operator_")
    )
    # R2-10: identity/state-based open findings (not raw event arithmetic).
    open_ids, fcounts = _open_finding_ids(profile, since)
    open_findings = len(open_ids)
    # Recurrence is per finding identity: a currently-open finding whose
    # history contains a resolved/dismissed transition was reopened.
    reopened = fcounts["resolved_transitions"] > 0 and open_findings > 0
    recurring = reopened or fcounts["opened"] >= 2 and open_findings > 0
    evidence = {
        "window_days": (int(time.time()) - since) // 86400,
        "failures": failures,
        "rework": rework,
        "review_outcomes": review_outcomes,
        "governance_findings_open": open_findings,
        "governance_findings_open_ids": open_ids,
        "governance_findings_opened_events": fcounts["opened"],
        "governance_findings_resolved_events": fcounts["resolved"],
        "recurring_findings": recurring,
        "source_refs": [
            "profile-activity-ledger:kanban.crashed|gave_up",
            "profile-activity-ledger:kanban.council_revise|audit_revise",
            "profile-activity-ledger:governance.finding.opened|updated|resolved|dismissed (identity+latest-state)",
        ],
        "method_version": REVIEW_VERSION,
    }
    if failures >= 3 or open_findings >= 3 or recurring or reopened:
        verdict = "ATTENTION"
    elif failures + rework > 0 or open_findings > 0:
        verdict = "WATCH"
    else:
        verdict = "CLEAN"
    return {"dimension": "quality", "verdict": verdict, "evidence": evidence}


# ── Dimension 4: capability ──────────────────────────────────────────────────

def _capability_dimension(profile: str, since: int) -> dict:
    cfg_path = _config_path(profile)
    cfg = _safe_read_yaml(cfg_path) if cfg_path else {}
    config_valid = bool(cfg)
    counts = _ledger_counts(profile, since)
    skill_denials = counts.get("skill.denied", 0) + counts.get("skill.access.blocked", 0)
    tool_denials = counts.get("tool.denied", 0) + counts.get("tool.access.would_block", 0)
    # SOUL.md is the constitution-critical identity file; USER.md is
    # optional context and its absence is not a capability defect.
    stats = _file_stats(profile)
    missing_files = [
        fn for fn, st in stats.items()
        if not st.get("exists") and fn == "SOUL.md"
    ] if profile != "default" else []
    evidence = {
        "window_days": (int(time.time()) - since) // 86400,
        "config_valid": config_valid,
        "skill_denials": skill_denials,
        "tool_denials": tool_denials,
        "missing_identity_files": missing_files,
        "source_refs": [
            "profile config.yaml parse",
            "profile-activity-ledger:skill.denied|tool.denied",
        ],
        "method_version": REVIEW_VERSION,
    }
    if not config_valid or missing_files:
        verdict = "DEGRADED"
    elif skill_denials + tool_denials >= 5:
        verdict = "WATCH"
    else:
        verdict = "OK"
    return {"dimension": "capability", "verdict": verdict, "evidence": evidence}


# ── Legacy helpers kept for backward-compatible payloads ─────────────────────

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
    }


def _quarter_review(profile: str, since: int) -> dict:
    """Full audit: session history + config + identity + trends."""
    counts = _ledger_counts(profile, since)
    files = _file_stats(profile)
    cfg_path = _config_path(profile)
    cfg = _safe_read_yaml(cfg_path) if cfg_path else {}
    skills = (cfg.get("skills") or {})

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
    }


# ── Four-dimension verdict assembly ─────────────────────────────────────────

def _profile_changed_since(profile: str, since: int) -> bool:
    """True when the profile's identity/config files changed in the window."""
    base = HERMES_HOME if profile == "default" else PROFILES_DIR / profile
    for fn in ("config.yaml", "SOUL.md", "USER.md"):
        path = base / fn if fn != "config.yaml" or profile != "default" else HERMES_HOME / fn
        try:
            if path.exists() and path.stat().st_mtime >= since:
                return True
        except OSError:
            continue
    return False


def _monthly_scope(
    profiles: list[str],
    since: int,
    *,
    lifecycle_map: Optional[dict[str, str]] = None,
) -> list[str]:
    """C8: monthly scope = active profiles PLUS changed profiles.

    A frozen/retired profile with in-window identity/config changes IS in
    scope (material change triggers review).  Unchanged frozen/retired
    profiles are excluded.  Profiles with unknown lifecycle default to
    active scope.
    """
    lifecycle_map = lifecycle_map if lifecycle_map is not None else _registry_lifecycle()
    scope: list[str] = []
    for profile in profiles:
        lifecycle = lifecycle_map.get(profile, "active")
        if lifecycle in ("active", "standby"):
            scope.append(profile)
        elif _profile_changed_since(profile, since):
            scope.append(profile)  # changed frozen/retired → reviewed
    return scope


def _four_dimension_verdict(dimensions: dict[str, dict],
                            registry_lifecycle: str | None) -> tuple[str, list[str]]:
    """Map dimension evidence to one of the four allowed verdicts.

    Rules:
      * Dimensions are never collapsed into a score.
      * An active gateway can never be COLD/STANDBY merely because ledger
        volume is low (runtime is its own dimension).
      * ``standby`` registry lifecycle is organisational COLD/STANDBY only
        when runtime evidence agrees; inactive service alone is runtime
        evidence, not a lifecycle change.
    """
    reasons: list[str] = []
    runtime = dimensions.get("runtime", {}).get("verdict")
    workload = dimensions.get("workload", {}).get("verdict")
    quality = dimensions.get("quality", {}).get("verdict")
    capability = dimensions.get("capability", {}).get("verdict")

    if quality == "ATTENTION":
        reasons.append("quality dimension ATTENTION (repeated failures or recurring findings)")
        return "ACTION REQUIRED", reasons
    if capability == "DEGRADED":
        reasons.append("capability dimension DEGRADED (config invalid or identity files missing)")
        return "ACTION REQUIRED", reasons
    if capability == "WATCH":
        reasons.append("capability dimension WATCH (skill/tool denials above threshold)")
        return "ACTION REQUIRED", reasons
    if quality == "WATCH":
        reasons.append("quality dimension WATCH (failures or rework in window)")
        return "OBSERVE", reasons

    # Standby: organisational lifecycle with corroborating runtime evidence.
    if registry_lifecycle == "standby" and runtime != "ACTIVE":
        reasons.append("registry lifecycle standby with non-active runtime")
        return "COLD/STANDBY", reasons

    if runtime == "ACTIVE":
        reasons.append("runtime dimension ACTIVE (gateway/heartbeat evidence)")
        return "HEALTHY", reasons

    # No active runtime: only then does absent workload suggest standby.
    if workload == "ABSENT":
        reasons.append("runtime non-active and workload ABSENT in window")
        return "COLD/STANDBY", reasons
    reasons.append("evidence mixed — no defect threshold crossed")
    return "OBSERVE", reasons


def _build_review(profile: str, since: int, cycle: str) -> dict:
    dimensions = {
        "runtime": _runtime_dimension(profile, since),
        "workload": _workload_dimension(profile, since),
        "quality": _quality_dimension(profile, since),
        "capability": _capability_dimension(profile, since),
    }
    lifecycle = _registry_lifecycle().get(profile)
    verdict, reasons = _four_dimension_verdict(dimensions, lifecycle)
    review = {
        "review_version": REVIEW_VERSION,
        "cycle": cycle,
        "profile": profile,
        "since_epoch": since,
        "window_days": (int(time.time()) - since) // 86400,
        "dimensions": dimensions,
        "registry_lifecycle": lifecycle,
        "verdict": verdict,
        "reasons": reasons,
        "recommendation": verdict,  # legacy field name, now the 4-dim verdict
    }
    base = {"weekly": _week_review, "monthly": _month_review, "quarterly": _quarter_review}[cycle]
    review.update(base(profile, since))
    return review


# ── Main ──────────────────────────────────────────────────────────────────────

def _profiles_for_cycle(cycle: str, profiles: list[str]) -> tuple[list[str], str]:
    """Monthly scope is active and changed profiles; quarterly is full roster."""
    if cycle in ("weekly", "monthly"):
        return profiles, "full_roster_fallback"  # caller refines with changed-set below
    return profiles, "full_roster"


def _run_cycle(cycle: str) -> str:
    now = int(time.time())
    if cycle == "weekly":
        since = now - 7 * 86400
    elif cycle == "monthly":
        since = now - 30 * 86400
    elif cycle == "quarterly":
        since = now - 90 * 86400
    else:
        print(f"Unknown cycle: {cycle}")
        sys.exit(1)

    event_type = f"profile.review.{cycle}"
    profiles = _all_profiles()
    lifecycle_map = _registry_lifecycle()

    # C8: monthly scope = active + changed (changed frozen/retired included);
    # quarterly = full roster/architecture.
    if cycle == "monthly" and lifecycle_map:
        in_scope = _monthly_scope(profiles, since, lifecycle_map=lifecycle_map)
        scope_note = "active+changed (registry lifecycle + change detection)"
    else:
        in_scope = profiles
        scope_note = "full roster/architecture"

    results = []
    for profile in sorted(in_scope):
        findings = _build_review(profile, since, cycle)
        findings["scope_note"] = scope_note if cycle != "weekly" else "weekly snapshot"
        event_id = f"review-{cycle}-{profile}-{now}"
        append_event(
            source="denji-review-cycle",
            event_type=event_type,
            event_id=event_id,
            actor_profile="denji",
            target_profile=profile,
            object_type="profile.review",
            summary=f"{cycle.capitalize()} review for {profile}: {findings['verdict']}",
            payload=findings,
            occurred_at=now,
        )
        results.append((profile, findings["verdict"]))

    prev_artifacts = sorted(LOGBOARD.glob(f"profile-review-{cycle}-*.json"), reverse=True) if LOGBOARD.exists() else []
    prev_recs = {}
    if prev_artifacts:
        try:
            prev = json.loads(prev_artifacts[0].read_text())
            for entry in prev.get("results", []):
                prev_recs[entry["profile"]] = entry["recommendation"]
        except Exception:
            pass

    changed = []
    for profile, rec in results:
        prev_rec = prev_recs.get(profile)
        if prev_rec != rec:
            changed.append((profile, prev_rec, rec))

    LOGBOARD.mkdir(parents=True, exist_ok=True)
    artifact = {
        "cycle": cycle,
        "review_version": REVIEW_VERSION,
        "scope_note": scope_note if cycle == "monthly" else ("full roster/architecture" if cycle == "quarterly" else "weekly snapshot"),
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

    if not changed:
        return event_type  # silent — no changes from last cycle

    counts: dict[str, int] = {}
    for _, r in results:
        counts[r] = counts.get(r, 0) + 1
    print(f"Denji Review Cycle - {cycle} - {len(profiles)} profiles ({scope_note})")
    print(f"Changes: {len(changed)} (of {len(results)} reviewed)")
    print(f"Verdicts: {counts}")
    print(f"Events recorded to ledger: {len(results)} × {event_type}")
    for profile, prev_rec, new_rec in changed[:10]:
        prev_short = (prev_rec or "new")[:40]
        print(f"  {profile}: {prev_short} → {new_rec}")
    print(f"Artifact: {artifact_path}")

    return event_type


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Denji profile review cycle executor")
    ap.add_argument("--cycle", required=True, choices=["weekly", "monthly", "quarterly"])
    args = ap.parse_args()
    _run_cycle(args.cycle)