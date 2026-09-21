#!/usr/bin/env python3
"""
Skill Borrow Ledger Management Script

Backed by the central append-only profile activity ledger
(~/.hermes/governance/profile-activity-ledger.sqlite, mirrored to JSONL) — the
single source of truth for skill borrows/grants. Revocation is a separate
skill.revoked event referencing the borrow (the ledger forbids UPDATE/DELETE).

Usage:
  python3 skill-broker-ledger.py borrow <profile> <skill> <task_id> [board]
  python3 skill-broker-ledger.py revoke <event_id> [task_result]
  python3 skill-broker-ledger.py count <profile> <skill> [months]
  python3 skill-broker-ledger.py deny <profile> <skill> <task_id> <reason> [board]
  python3 skill-broker-ledger.py list [--profile <p>] [--skill <s>] [--limit <n>]
  python3 skill-broker-ledger.py review [--months <n>]
  python3 skill-broker-ledger.py status        # Quick summary of recent activity
  python3 skill-broker-ledger.py import-legacy  # One-time import of the old JSONL
"""

import json
import os
import sys
import time
from pathlib import Path

# Unified ledger: the skill broker is now backed by the central append-only
# profile activity ledger (one source of truth), not a private JSONL.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hermes_cli.profile_activity_ledger import (  # noqa: E402
    append_event,
    query_events,
    ledger_db_path,
)
# The grant engine is the single source of grant/deny/revoke logic.
from tools.skill_grants import (  # noqa: E402
    EV_BORROW,
    EV_DENY,
    EV_REVOKE,
    NEVER_GRANT,
    borrow_count,
    grant_skill,
    record_deny,
    revoke_by_event_id,
    revoked_borrow_ids,
)

BROKER_SOURCE = "skill.broker"

# Legacy private ledger — retained only for one-time import (see `import-legacy`).
LEGACY_JSONL = Path("/home/kensei/.hermes/governance/skill-broker-ledger.jsonl")
LEDGER_PATH = ledger_db_path()

# Safety lists (NEVER_GRANT) and all grant/deny/revoke recording live in the
# grant engine (tools.skill_grants) — this CLI is a thin wrapper over it.

# ---- Ledger projection helpers ----

def _now_ts():
    return int(time.time())


def _to_entry(ev):
    """Project a unified ledger event back into the broker's legacy entry shape,
    so CLI consumers (skill-broker-core skill) see a stable schema."""
    p = ev.get("payload") or {}
    et = ev.get("event_type")
    return {
        "event_id": ev.get("event_id"),
        "ts": ev.get("occurred_at"),
        "task_id": p.get("task_id"),
        "board": ev.get("board"),
        "worker_profile": ev.get("target_profile"),
        "skill_borrowed": ev.get("object_id"),
        "grant_type": p.get("grant_type", "temporary"),
        "grant_expiry": p.get("grant_expiry", "task_completion"),
        "success": et == EV_BORROW,
        "task_result": p.get("task_result"),
        "revoked_at": p.get("revoked_at"),
        "recommendation": p.get("recommendation"),
        "event_type": et,
    }


def cmd_borrow(profile, skill, task_id, board="ops"):
    """Grant via the engine (applies NEVER_GRANT + frequency safety)."""
    decision = grant_skill(profile, skill, task_id, board=board)
    print(json.dumps(decision))
    return decision.get("event_id")


def cmd_revoke(event_id, task_result="completed"):
    """Revoke a borrow by id via the engine."""
    result = revoke_by_event_id(event_id, task_result)
    print(json.dumps(result))
    return 1 if result.get("error") else 0


def cmd_count(profile, skill, months=1):
    """Count temporary borrows of a skill by a profile in the last N months."""
    n = borrow_count(profile, skill, days=months * 30)
    print(json.dumps({"profile": profile, "skill": skill, "months": months, "borrows": n}))
    return 0


def cmd_deny(profile, skill, task_id, reason, board="ops"):
    """Log a denied borrow request via the engine."""
    decision = record_deny(profile, skill, task_id, reason, board)
    print(json.dumps({"event_id": decision.get("event_id"), "status": "denied"}))
    return 0


def cmd_list(profile=None, skill=None, limit=20):
    """List recent broker events (borrow/deny/revoke)."""
    rows = query_events(
        event_types=[EV_BORROW, EV_DENY, EV_REVOKE],
        target_profile=profile,
        object_id=skill,
        limit=limit,
    )
    print(json.dumps([_to_entry(e) for e in rows], indent=2))
    return 0


def cmd_review(months=3):
    """Group borrows by (profile, skill) and flag promotion candidates."""
    from collections import defaultdict

    cutoff = int(time.time() - (months * 30 * 24 * 3600))
    recent = query_events(event_types=[EV_BORROW, EV_DENY], since=cutoff)
    groups = defaultdict(list)
    for ev in recent:
        groups[(ev.get("target_profile"), ev.get("object_id"))].append(ev)

    review = []
    for (profile, skill), group in sorted(groups.items(), key=lambda x: len(x[1]), reverse=True):
        count = len(group)
        success_count = sum(1 for e in group if e.get("event_type") == EV_BORROW)
        action = "monitor"
        if count >= 6:
            action = "ESCALATE: recommend permanent add"
        elif count >= 4:
            action = "flag Denji for review"
        review.append({
            "profile": profile,
            "skill": skill,
            "borrows": count,
            "successes": success_count,
            "action": action,
        })
    print(json.dumps({"period_months": months, "entries_analysed": len(recent), "findings": review}, indent=2))
    return 0


def cmd_status():
    """Quick summary from the unified ledger."""
    borrows = query_events(event_types=[EV_BORROW])
    denies = query_events(event_types=[EV_DENY])
    revoked = revoked_borrow_ids()
    active = sum(1 for e in borrows if e.get("event_id") not in revoked)
    month_ago = _now_ts() - (30 * 24 * 3600)
    recent = sum(1 for e in (borrows + denies) if (e.get("occurred_at") or 0) >= month_ago)

    top_borrowers = {}
    for e in borrows:
        p = e.get("target_profile", "?")
        top_borrowers[p] = top_borrowers.get(p, 0) + 1
    sorted_top = sorted(top_borrowers.items(), key=lambda x: x[1], reverse=True)[:5]

    print(json.dumps({
        "total_entries": len(borrows) + len(denies),
        "active_borrows": active,
        "denied": len(denies),
        "last_30_days": recent,
        "top_borrower_profiles": [{"profile": p, "count": c} for p, c in sorted_top],
        "ledger_path": str(LEDGER_PATH),
    }, indent=2))
    return 0


def cmd_import_legacy():
    """One-time: import legacy skill-broker-ledger.jsonl into the unified ledger."""
    if not LEGACY_JSONL.exists():
        print(json.dumps({"imported": 0, "note": "no legacy jsonl"}))
        return 0
    imported = 0
    skipped = 0
    with open(LEGACY_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            eid = e.get("event_id")
            skill = e.get("skill_borrowed")
            profile = e.get("worker_profile")
            board = e.get("board")
            ts = e.get("ts")
            if e.get("success"):
                append_event(
                    source=BROKER_SOURCE, event_type=EV_BORROW, event_id=eid,
                    actor_profile="skill-broker", target_profile=profile,
                    object_type="skill", object_id=skill, board=board, occurred_at=ts,
                    summary=f"[import] {profile} borrowed {skill}",
                    payload={"task_id": e.get("task_id"), "grant_type": e.get("grant_type", "temporary"),
                             "grant_expiry": e.get("grant_expiry", "task_completion")},
                )
                imported += 1
                if e.get("revoked_at"):
                    append_event(
                        source=BROKER_SOURCE, event_type=EV_REVOKE, event_id=f"{eid}-revoke",
                        actor_profile="skill-broker", target_profile=profile,
                        object_type="skill", object_id=skill, board=board, occurred_at=e.get("revoked_at"),
                        summary=f"[import] revoked {skill} grant {eid}",
                        payload={"borrow_event_id": eid, "task_result": e.get("task_result"),
                                 "revoked_at": e.get("revoked_at")},
                    )
            else:
                append_event(
                    source=BROKER_SOURCE, event_type=EV_DENY, event_id=eid,
                    actor_profile="skill-broker", target_profile=profile,
                    object_type="skill", object_id=skill, board=board, occurred_at=ts,
                    summary=f"[import] denied {skill} for {profile}",
                    payload={"task_id": e.get("task_id"), "task_result": "denied",
                             "recommendation": e.get("recommendation")},
                )
                imported += 1
    print(json.dumps({"imported": imported, "skipped": skipped, "from": str(LEGACY_JSONL)}))
    return 0


# ---- Entry Point ----

def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        return 1
    
    cmd = sys.argv[1]
    
    if cmd == "borrow":
        if len(sys.argv) < 5:
            print("Usage: ... borrow <profile> <skill> <task_id> [board]")
            return 1
        profile = sys.argv[2]
        skill = sys.argv[3]
        task_id = sys.argv[4]
        board = sys.argv[5] if len(sys.argv) > 5 else "ops"
        # The engine applies NEVER_GRANT + frequency safety and records the
        # grant or denial; the broker CLI is a thin wrapper.
        cmd_borrow(profile, skill, task_id, board)
        return 0
    
    elif cmd == "revoke":
        if len(sys.argv) < 3:
            print("Usage: ... revoke <event_id> [task_result]")
            return 1
        event_id = sys.argv[2]
        task_result = sys.argv[3] if len(sys.argv) > 3 else "completed"
        return cmd_revoke(event_id, task_result)
    
    elif cmd == "count":
        if len(sys.argv) < 4:
            print("Usage: ... count <profile> <skill> [months]")
            return 1
        profile = sys.argv[2]
        skill = sys.argv[3]
        months = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        return cmd_count(profile, skill, months)
    
    elif cmd == "deny":
        if len(sys.argv) < 6:
            print("Usage: ... deny <profile> <skill> <task_id> <reason> [board]")
            return 1
        profile = sys.argv[2]
        skill = sys.argv[3]
        task_id = sys.argv[4]
        reason = sys.argv[5]
        board = sys.argv[6] if len(sys.argv) > 6 else "ops"
        return cmd_deny(profile, skill, task_id, reason, board)
    
    elif cmd == "list":
        profile = None
        skill = None
        limit = 20
        args = sys.argv[2:]
        for i, a in enumerate(args):
            if a == "--profile" and i + 1 < len(args):
                profile = args[i + 1]
            elif a == "--skill" and i + 1 < len(args):
                skill = args[i + 1]
            elif a == "--limit" and i + 1 < len(args):
                limit = int(args[i + 1])
        return cmd_list(profile, skill, limit)
    
    elif cmd == "review":
        months = 3
        args = sys.argv[2:]
        for i, a in enumerate(args):
            if a == "--months" and i + 1 < len(args):
                months = int(args[i + 1])
        return cmd_review(months)
    
    elif cmd == "status":
        return cmd_status()

    elif cmd == "import-legacy":
        return cmd_import_legacy()

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__.strip())
        return 1


if __name__ == "__main__":
    sys.exit(main())
