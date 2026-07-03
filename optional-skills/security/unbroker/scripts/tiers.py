"""Automation-tier selection and per-subject action planning.

Tiers:
  T0  fully automated, no verification loop
  T1  automated submit + automated verification (email mode B/C, or backend-cleared captcha)
  T2  automated submit, verification needs a human (hard captcha / phone callback / account)
  T3  human-required end-to-end (gov ID, fax, mail, voice-only phone)
"""
from __future__ import annotations

import dossier as dossier_mod
import vectors as vectors_mod

HARD_HUMAN = ("gov_id", "fax", "mail", "phone_voice")


def select_tier(broker: dict, email_mode: str = "draft_only",
                browser_clears_captcha: bool = False) -> str:
    req = ((broker.get("optout") or {}).get("requires")) or {}

    if any(req.get(k) for k in HARD_HUMAN):
        return "T3"
    if req.get("account"):
        return "T2"

    captcha = bool(req.get("captcha"))
    if (captcha and not browser_clears_captcha) or req.get("phone_callback"):
        return "T2"

    if req.get("email_verification"):
        return "T1" if email_mode in ("programmatic", "alias") else "T2"

    if captcha and browser_clears_captcha:
        return "T1"
    return "T0"


def plan(subject_dossier: dict, brokers_list: list[dict], cfg: dict,
         browser_clears_captcha: bool = False) -> list[dict]:
    email_mode = (subject_dossier.get("preferences") or {}).get("email_mode") \
        or cfg.get("email_mode", "draft_only")
    actions: list[dict] = []
    for b in brokers_list:
        opt = b.get("optout") or {}
        search = b.get("search") or {}
        tier = select_tier(b, email_mode, browser_clears_captcha)
        disclosure = dossier_mod.select_disclosure(subject_dossier, opt.get("inputs", []))
        svectors = vectors_mod.search_vectors(subject_dossier, b)
        actions.append({
            "broker_id": b.get("id"),
            "broker_name": b.get("name"),
            "priority": b.get("priority"),
            "method": opt.get("method"),
            "tier": tier,
            "human_required": tier == "T3",
            "search_url": search.get("url"),
            "fetch": search.get("fetch", "web_extract"),
            "antibot": search.get("antibot"),
            "search_by": vectors_mod.supported_by(b),
            "search_vectors": svectors,
            "optout_url": opt.get("url"),
            "optout_email": opt.get("email"),
            "disclosure_fields": sorted(disclosure.keys()),
            "owns": b.get("owns") or [],
            "notes": opt.get("notes", ""),
            "optout_quirks": opt.get("quirks") or [],
            "optout_requires": opt.get("requires") or {},
            # The DELETION lane (right-to-delete), distinct from listing suppression. Structured so
            # the autopilot can route to it: {via: email|in_flow|web_form, email?, url?, kinds?, notes?}
            "deletion": opt.get("deletion") or {},
            # Exact ordered opt-out steps maintained IN the broker record (field-verified knowledge
            # lives with the data, not in code).
            "optout_playbook": opt.get("playbook") or [],
        })
    return actions


def fanout(brokers_list: list[dict], batch_size: int = 8) -> dict:
    """Group brokers into batches for parallel `delegate_task` scan subagents.

    Scanning many brokers serially is slow and burns context; above `batch_size`
    the agent is expected to spawn one subagent per batch (see SKILL.md).
    """
    ids = [b.get("id") for b in brokers_list if b.get("id")]
    batches = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
    return {
        "broker_count": len(ids),
        "batch_size": batch_size,
        "should_fanout": len(ids) > batch_size,
        "batches": batches,
    }


# States that mean "the crawl reached a verdict for this broker".
_SCANNED_STATES = {"found", "not_found", "indirect_exposure", "blocked", "submitted",
                   "verification_pending", "awaiting_processing", "confirmed_removed", "reappeared",
                   "action_selected", "human_task_queued"}
# States that still need a deletion action taken.
_ACTIONABLE_STATES = {"found", "indirect_exposure", "reappeared", "action_selected"}


def batch_plan(subject_dossier: dict, brokers_list: list[dict], cfg: dict,
               ledger: dict | None = None, browser_clears_captcha: bool = False) -> dict:
    """Reduce the per-broker plan into a phase-oriented batch view.

    Overlays the current ledger state on each broker, groups by what the operator
    should DO next, and collapses ownership clusters so a parent removal that clears
    children is ONE action, not N. Read-only: computes, never mutates the ledger.
    """
    ledger = ledger or {}
    actions = plan(subject_dossier, brokers_list, cfg, browser_clears_captcha)

    # child id -> parent id (only for parents present in this plan set)
    child_to_parent: dict[str, str] = {}
    for a in actions:
        for child in a.get("owns") or []:
            child_to_parent[child] = a["broker_id"]

    def state_of(bid: str) -> str:
        return (ledger.get(bid) or {}).get("state", "new")

    groups: dict[str, list[dict]] = {
        "unscanned": [],        # no verdict yet -> Phase 1 crawl
        "found": [],            # direct removable listing -> Phase 2 opt-out (incl. reappeared/action_selected)
        "indirect_exposure": [],# PII on a third party's record -> CCPA/GDPR delete email
        "blocked": [],          # anti-bot / needs stealth browser -> requeue
        "in_progress": [],      # submitted / verification_pending / awaiting_processing
        "human": [],            # human_task_queued -> the end-of-run digest, NOT re-scanning
        "done": [],             # confirmed_removed
        "not_found": [],
    }
    covered_by_parent: dict[str, list[str]] = {}

    for a in actions:
        bid = a["broker_id"]
        st = state_of(bid)
        # cluster collapse: if a parent in this set is already actioned, the child is covered
        parent = child_to_parent.get(bid)
        if parent and state_of(parent) in ("found", "reappeared", "action_selected", "submitted",
                                           "verification_pending", "awaiting_processing",
                                           "confirmed_removed", "human_task_queued"):
            covered_by_parent.setdefault(parent, []).append(bid)
            continue

        row = {"broker_id": bid, "broker_name": a["broker_name"], "priority": a["priority"],
               "tier": a["tier"], "method": a["method"], "state": st,
               "optout_url": a["optout_url"], "optout_email": a.get("optout_email"),
               "clears_children": a.get("owns") or [],
               "optout_requires": a.get("optout_requires") or {},
               "optout_quirks": a.get("optout_quirks") or [],
               "deletion": a.get("deletion") or {},
               "optout_playbook": a.get("optout_playbook") or [],
               "notes": a.get("notes", "")}
        if st in ("submitted", "verification_pending", "awaiting_processing"):
            groups["in_progress"].append(row)
        elif st == "confirmed_removed":
            groups["done"].append(row)
        elif st in ("reappeared", "action_selected"):
            groups["found"].append(row)   # still needs the opt-out action
        elif st == "human_task_queued":
            groups["human"].append(row)   # parked for the digest; never re-queued as work
        elif st in groups:
            groups[st].append(row)
        elif st not in _SCANNED_STATES:
            groups["unscanned"].append(row)
        else:
            groups.setdefault(st, []).append(row)

    # PARENTS FIRST: within the actionable 'found' group, order cluster parents (a removal
    # that clears children) ahead of standalone listings, most-children first. Working a
    # parent before its children is what makes the cluster dedup real -- do them in this order.
    groups["found"].sort(key=lambda r: (-len(r.get("clears_children") or []),
                                        {"T0": 0, "T1": 1, "T2": 2, "T3": 3}.get(r.get("tier") or "", 9),
                                        r["broker_id"]))

    return {
        "subject": subject_dossier.get("subject_id"),
        "phase": "discover" if groups["unscanned"] else "delete",
        "counts": {k: len(v) for k, v in groups.items()},
        "groups": groups,
        "cluster_savings": {p: kids for p, kids in covered_by_parent.items()},
        "parent_playbook": _parent_playbook(groups["found"]),
        "next_actions": _batch_next(groups, covered_by_parent),
    }


def synthesize_steps(r: dict) -> list[str]:
    """Generic ordered opt-out steps derived from an optout record's structured fields.

    Used for any broker without a hand-verified `optout.playbook`. Bespoke, field-verified
    step lists live IN the broker JSON (`optout.playbook`) - single source of truth that
    accrues knowledge as live runs discover mechanics (see methods.md logging rule).
    """
    steps = [f"Opt out at {r.get('optout_url') or r.get('optout_email') or '(see broker record)'}"
             + (f" -- clears {', '.join(r['clears_children'])}." if r.get("clears_children") else ".")]
    req = r.get("optout_requires") or {}
    if req.get("profile_url"):
        steps.append("Needs the confirmed profile_url (paste the listing URL you recorded).")
    if req.get("email_verification"):
        steps.append("Email verification: the same browser/inbox must open the confirmation link.")
    if req.get("phone_callback"):
        steps.append("Phone-callback code required; queue a human task if no operator is available.")
    if req.get("gov_id"):
        steps.append("Government ID demanded (T3): human task; never send SSN or a full ID number.")
    d = r.get("deletion") or {}
    if d.get("email"):
        steps.append(f"DELETION lane: a right-to-delete request can be emailed to {d['email']}"
                     + (f" ({d['notes']})" if d.get("notes") else "")
                     + " -- prefer deletion over suppression.")
    if r.get("notes"):
        steps.append(str(r["notes"]))
    for q in (r.get("optout_quirks") or [])[:3]:
        steps.append(str(q))
    return steps


def _parent_playbook(found_rows: list[dict]) -> list[dict]:
    """Tailored, ordered opt-out instructions for each cluster PARENT in the found group.

    Steps come from the broker record's own `optout.playbook` (field-verified, maintained with
    the data) with a synthesised fallback so the guidance is never empty. Standalone listings
    are intentionally omitted -- the playbook exists to make the parents-first order concrete.
    """
    playbook: list[dict] = []
    for i, r in enumerate([x for x in found_rows if x.get("clears_children")], start=1):
        steps = list(r.get("optout_playbook") or []) or synthesize_steps(r)
        playbook.append({
            "order": i,
            "broker_id": r["broker_id"],
            "broker_name": r["broker_name"],
            "tier": r["tier"],
            "clears_children": r["clears_children"],
            "optout_url": r.get("optout_url"),
            "optout_email": r.get("optout_email"),
            "deletion": r.get("deletion") or {},
            "steps": steps,
        })
    return playbook


def _batch_next(groups: dict, covered: dict) -> list[str]:
    tips: list[str] = []
    if groups["unscanned"]:
        tips.append(f"PHASE 1 (crawl): {len(groups['unscanned'])} broker(s) unscanned -- run `fanout` and "
                    "scan read-only before any deletion.")
    if groups["found"]:
        parents = [r for r in groups["found"] if r.get("clears_children")]
        if parents:
            order = " -> ".join(r["broker_id"] for r in parents)
            tips.append(f"PHASE 2 (opt-out): {len(groups['found'])} direct listing(s). DO CLUSTER PARENTS "
                        f"FIRST, in this order: {order} (see `parent_playbook` for tailored per-parent "
                        "steps), then the standalone listings.")
        else:
            tips.append(f"PHASE 2 (opt-out): {len(groups['found'])} direct listing(s) to remove.")
    if groups["indirect_exposure"]:
        tips.append(f"{len(groups['indirect_exposure'])} indirect-exposure case(s): send a targeted "
                    "CCPA/GDPR delete-my-PII email (render-email --kind ccpa_indirect), do NOT use the opt-out form.")
    if groups["blocked"]:
        tips.append(f"{len(groups['blocked'])} blocked (anti-bot): requeue for a stealth/cloud browser "
                    "pass; don't burn subagent time fighting CAPTCHAs.")
    if covered:
        n = sum(len(v) for v in covered.values())
        tips.append(f"Cluster dedup: {n} child site(s) covered by parent removals -- skip separate opt-outs.")
    if groups["in_progress"]:
        tips.append(f"{len(groups['in_progress'])} in progress: resolve verification links, then confirm removal.")
    if groups.get("human"):
        tips.append(f"{len(groups['human'])} parked human task(s): present via `tasks` at end of run "
                    "(do not re-scan or re-queue them).")
    return tips
