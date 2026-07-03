"""Autonomous action queue: what should the agent do RIGHT NOW for this subject?

`next_actions` turns (dossier, broker DB, config, ledger) into an ordered queue of
concrete agent actions plus a human digest. The agent's whole run becomes a loop:

    while True:
        q = pdd.py next <subject>
        if not q["actions"]: break
        execute each action, record outcomes
    present q["human_digest"] once; schedule cron at q["next_wake_at"]

Policy (cfg["autonomy"]):
  full     - intake consent is standing authorization; T0-T2 agent actions are
             executed without pausing. Humans appear only in the digest.
  assisted - same queue, but every submission action carries confirm_first=True.

The queue is deterministic and side-effect free: it never mutates the ledger, it
only reads. Executing + recording stays with the agent (and the record command).
"""
from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path

import brokers as brokers_mod
import emailer
import ledger as ledger_mod
import paths
import registry
import tiers

CACHE_STALE_DAYS = 7          # refresh the live broker list after this
FANOUT_THRESHOLD = 8          # above this many unscanned brokers, use delegate_task fan-out

# States with nothing left to do (absent a due recheck).
_TERMINAL = {"not_found", "confirmed_removed"}
_IN_FLIGHT = {"submitted", "verification_pending", "awaiting_processing"}


def cache_age_days(now: float | None = None) -> float | None:
    """Age of the live BADBOOL cache in days, or None if never pulled."""
    p: Path = paths.brokers_cache_path()
    if not p.exists():
        return None
    now = now if now is not None else _dt.datetime.now().timestamp()
    return max(0.0, (now - p.stat().st_mtime) / 86400.0)


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _min_future_recheck(ledger: dict, at: str) -> str | None:
    future = [c.get("next_recheck_at") for c in ledger.values()
              if c.get("next_recheck_at") and c["next_recheck_at"] > at]
    return min(future) if future else None


def _digest(broker_row: dict, reason: str, steps: list[str], prep: list[str] | None = None) -> dict:
    return {
        "broker_id": broker_row.get("broker_id"),
        "broker_name": broker_row.get("broker_name"),
        "reason": reason,
        "agent_prep": prep or [],     # commands the agent runs BEFORE handing this to the human
        "steps": steps,               # what the human actually does
        "withhold": ["SSN", "full driver's-license / passport numbers"],
    }


def request_kind(dossier: dict, allowed: list[str] | None = None) -> str:
    """Pick the honest legal basis for a deletion request from the subject's residency.

    ccpa only for California residents, gdpr only for EU/UK residents, generic otherwise.
    `allowed` (from the broker's deletion.kinds) can restrict DOWN to generic but never
    upgrades to a law the subject can't truthfully claim.
    """
    res = (dossier.get("residency_jurisdiction") or "US").upper()
    if res.startswith("US-CA"):
        kind = "ccpa"
    elif res.startswith(("EU", "UK", "GB")):
        kind = "gdpr"
    else:
        kind = "generic"
    if allowed and kind not in allowed and "generic" in allowed:
        kind = "generic"
    return kind


_HUMAN_GATES = ("gov_id", "fax", "mail", "phone_voice", "phone_callback", "account")


def _email_lane(row: dict) -> tuple[str | None, str]:
    """(address, why) for the autonomous email lane of this broker, if one exists.

    Lane rules:
      1. the broker's primary opt-out method IS email;
      2. the record marks its deletion lane email-preferred (deletion.via == "email");
      3. RESCUE: the primary flow is human-gated (gov ID / fax / phone / account) but a
         right-to-delete email exists - the email lane restores full autonomy (this is the
         verified Whitepages pattern: privacyrequest@ accepts requests precisely so people
         don't have to do the phone-callback tool).
    """
    deletion = row.get("deletion") or {}
    req = row.get("optout_requires") or {}
    if row.get("method") == "email":
        addr = row.get("optout_email") or deletion.get("email")
        return (addr, "primary opt-out method is email") if addr else (None, "")
    if deletion.get("via") == "email" and deletion.get("email"):
        return deletion["email"], "record prefers the right-to-delete email lane"
    if (row.get("tier") == "T3" or any(req.get(k) for k in _HUMAN_GATES)) and deletion.get("email"):
        return deletion["email"], "rescue: primary flow is human-gated; deletion email restores autonomy"
    return None, ""


def _optout_action(row: dict, playbook: dict[str, dict], subject_id: str, dossier: dict,
                   email_mode: str, smtp_ok: bool, confirm_first: bool) -> tuple[dict | None, dict | None]:
    """Map one actionable `found` row to (agent_action, human_digest_entry).

    Routing order maximizes autonomy: (1) the email lane (primary email method, preferred
    right-to-delete email, or rescue from a human-gated form) beats everything when SMTP is
    up; (2) genuinely human-only flows go to the digest; (3) web forms are driven with the
    record's own field-verified playbook steps.
    """
    bid = row["broker_id"]
    req = row.get("optout_requires") or {}
    tier = row.get("tier")
    deletion = row.get("deletion") or {}

    # 1) The autonomous EMAIL LANE (right-to-delete by email + confirm the reply).
    # Autonomous when SMTP is configured (programmatic/alias) OR in browser mode (agent sends via
    # the operator's logged-in webmail; no password needed).
    email_addr, lane_why = _email_lane(row)
    can_email = (email_mode in ("programmatic", "alias") and smtp_ok) or email_mode == "browser"
    if email_addr and can_email:
        kind = request_kind(dossier, deletion.get("kinds"))
        via = "browser" if email_mode == "browser" else "smtp"
        then = ("send-email records it + returns a recipient-locked payload; compose and send it in "
                "the operator's webmail via browser_*, then `verify-link` on the reply and open the link"
                if via == "browser" else
                "state auto-records as submitted; poll-verification picks up their verification reply, "
                "open its link, then record")
        return {
            "type": "optout_email_send",
            "broker_id": bid, "broker_name": row.get("broker_name"), "tier": tier,
            "confirm_first": confirm_first, "send_via": via,
            "to": email_addr, "kind": kind, "why": lane_why,
            "command": f"python3 scripts/pdd.py send-email {subject_id} {bid} --kind {kind} "
                       f"--to {email_addr} --listing <confirmed-url>",
            "then": then,
        }, None
    if row.get("method") == "email":
        return None, _digest(row, "email opt-out (draft mode: a human must hit send)",
                             ["Send the rendered draft from your own mail client",
                              f"Then: python3 scripts/pdd.py record {subject_id} {bid} submitted "
                              f"--disclosed contact_email --channel email"],
                             prep=[f"python3 scripts/pdd.py render-email {subject_id} {bid} --listing <confirmed-url>"])

    # 2) Genuinely human-only work goes to the digest (no email lane could rescue it).
    if tier == "T3":
        return None, _digest(row, "human-only opt-out (gov ID / fax / mail / voice phone)",
                             [f"Follow the broker's process at {row.get('optout_url') or row.get('optout_email')}",
                              "Provide only the fields the listing already shows; cross out ID numbers on any document"])
    if req.get("phone_callback"):
        return None, _digest(row, "phone-callback verification (operator must be on the phone)",
                             [f"Open {row.get('optout_url')} and submit with only the planned fields",
                              "Answer the automated call and enter the 4-digit code to finish"],
                             prep=[f"python3 scripts/pdd.py plan {subject_id} --batch  # confirm fields first"])
    if req.get("account"):
        return None, _digest(row, "requires creating/holding an account with the broker",
                             [f"Create/log in at {row.get('optout_url')} and submit the opt-out",
                              "Use the subject's contact email; no extra PII beyond the planned fields"])

    # 3) web_form: drive the browser with the record's own playbook steps.
    steps = (playbook.get(bid) or {}).get("steps") or list(row.get("optout_playbook") or []) \
        or tiers.synthesize_steps(row)
    action = {
        "type": "optout_web_form",
        "broker_id": bid, "broker_name": row.get("broker_name"), "tier": tier,
        "confirm_first": confirm_first,
        "optout_url": row.get("optout_url"),
        "clears_children": row.get("clears_children") or [],
        "steps": steps,
        "after": f"python3 scripts/pdd.py record {subject_id} {bid} submitted "
                 f"--disclosed <field>... --channel web_form",
    }
    if deletion:
        action["prefer_deletion"] = ("this record has a right-to-delete lane -- complete the DELETION "
                                     "flow, not just suppression" + (f" ({deletion.get('notes')})"
                                                                     if deletion.get("notes") else ""))
    if req.get("captcha"):
        action["note"] = ("CAPTCHA-gated: attempt with the configured browser backend once; if it "
                          "does not clear, record blocked (do NOT retry-loop or bypass)")
    return action, None


def next_actions(dossier: dict, brokers_list: list[dict], cfg: dict,
                 ledger: dict | None = None, env: dict | None = None) -> dict:
    env = os.environ if env is None else env
    ledger = ledger or {}
    subject_id = dossier.get("subject_id", "")
    autonomy = cfg.get("autonomy", "full")
    confirm_first = autonomy == "assisted"
    email_mode = cfg.get("email_mode", "draft_only")
    mail = emailer.available(env)
    at = _now_iso()

    batch = tiers.batch_plan(dossier, brokers_list, cfg, ledger,
                             browser_clears_captcha=cfg.get("browser_backend") == "browserbase"
                             or bool(env.get("BROWSERBASE_API_KEY")))
    groups = batch["groups"]
    playbook = {p["broker_id"]: p for p in batch.get("parent_playbook") or []}
    by_id = {b.get("id"): b for b in brokers_list}

    actions: list[dict] = []
    digest: list[dict] = []

    # 0) keep the broker DB fresh (autonomously)
    age = cache_age_days()
    if age is None or age > CACHE_STALE_DAYS:
        actions.append({
            "type": "refresh_brokers",
            "why": "live broker cache missing" if age is None else f"cache is {age:.0f} days old",
            "command": "python3 scripts/pdd.py refresh-brokers",
        })

    # 0b) DROP one-shot: for a CA resident, ONE request deletes from every registered
    # broker (the whole CA Data Broker Registry) -- the highest-leverage removal there is.
    registry_recs = brokers_mod.load_registry_cache()
    residency = (dossier.get("residency_jurisdiction") or "US").upper()
    drop_filed = bool((dossier.get("preferences") or {}).get("drop_filed_at"))
    if registry_recs and residency.startswith("US-CA") and not drop_filed:
        actions.append({
            "type": "drop_submit",
            "one_shot": True,
            "registry_count": len(registry_recs),
            "url": registry.DROP_URL,
            "command": f"python3 scripts/pdd.py drop {subject_id}",
            "why": f"CA resident: one DROP request deletes from all {len(registry_recs)} registered "
                   "data brokers at once (superset of what commercial services cover).",
            "after": f"python3 scripts/pdd.py drop {subject_id} --filed",
        })

    # 1) Phase 1 crawl: everything unscanned (read-only, parallel-safe)
    unscanned = groups.get("unscanned") or []
    if unscanned:
        ids = [r["broker_id"] for r in unscanned]
        if len(ids) > FANOUT_THRESHOLD:
            actions.append({
                "type": "fanout_scan",
                "broker_ids": ids,
                "command": f"python3 scripts/pdd.py fanout {subject_id}",
                "how": "spawn ONE delegate_task subagent per batch IN PARALLEL with each batch's brief; "
                       "parent re-verifies key `found` claims before trusting them",
            })
        else:
            actions.append({
                "type": "scan_inline",
                "broker_ids": ids,
                "command": f"python3 scripts/pdd.py plan {subject_id}",
                "how": "run every search_vector per broker via the methods.md ladder "
                       "(web_extract -> site: probe -> browser), record a verdict per broker",
            })

    # 2) in-flight email verifications: poll the inbox (or hand to the human in draft mode)
    for st in ("submitted", "verification_pending"):
        for bid, case in sorted(ledger.items()):
            if case.get("state") != st:
                continue
            broker = by_id.get(bid) or {}
            if not ((broker.get("optout") or {}).get("requires") or {}).get("email_verification"):
                continue
            if mail["imap"]:
                actions.append({
                    "type": "poll_verification", "via": "imap",
                    "broker_id": bid,
                    "command": f"python3 scripts/pdd.py poll-verification {subject_id} --broker {bid}",
                    "then": "browser_navigate the returned link IN THE SAME AGENT BROWSER (sessions are "
                            f"browser-bound), complete the flow, then record: awaiting_processing",
                })
            elif email_mode == "browser":
                actions.append({
                    "type": "poll_verification", "via": "browser", "broker_id": bid,
                    "how": "open the broker's confirmation email in the operator's logged-in webmail "
                           f"(browser_*), then `python3 scripts/pdd.py verify-link {subject_id} {bid} "
                           "--text '<email body>'` to score the link, browser_navigate it in the SAME "
                           "browser, then record awaiting_processing",
                })
            else:
                digest.append(_digest(
                    {"broker_id": bid, "broker_name": (broker.get("name") or bid)},
                    "verification email must be opened by a human (draft mode, no inbox access)",
                    ["Open the broker's verification email in the subject's inbox and click the link",
                     f"Then: python3 scripts/pdd.py record {subject_id} {bid} awaiting_processing"]))

    # 3) due rechecks: processing windows elapsed / reappearance sweeps
    for case in ledger_mod.due(subject_id, at=at, ledger=ledger):
        bid = case.get("broker_id")
        st = case.get("state")
        if st in ("awaiting_processing", "confirmed_removed"):
            actions.append({
                "type": "verify_removal",
                "broker_id": bid,
                "why": "processing window elapsed" if st == "awaiting_processing" else "periodic reappearance re-scan",
                "how": "re-run this broker's search_vectors; if gone record confirmed_removed; "
                       "if still listed record reappeared and requeue the opt-out",
            })
        elif st in ("submitted", "verification_pending") and not mail["imap"]:
            pass  # already covered by the digest entry above

    # 4) Phase 2 opt-outs: parents first (batch_plan already ordered them)
    for row in groups.get("found") or []:
        action, task = _optout_action(row, playbook, subject_id, dossier,
                                      email_mode, mail["smtp"], confirm_first)
        if action:
            actions.append(action)
        if task:
            digest.append(task)

    # 5) indirect exposure: targeted delete-my-PII requests
    for row in groups.get("indirect_exposure") or []:
        bid = row["broker_id"]
        if (email_mode in ("programmatic", "alias") and mail["smtp"]) or email_mode == "browser":
            actions.append({
                "type": "indirect_email_send",
                "broker_id": bid, "confirm_first": confirm_first,
                "send_via": "browser" if email_mode == "browser" else "smtp",
                "command": f"python3 scripts/pdd.py send-email {subject_id} {bid} --kind ccpa_indirect "
                           f"--listing <third-party-listing-url>",
            })
        else:
            digest.append(_digest(row, "indirect-exposure request (draft mode: a human must hit send)",
                                  ["Send the rendered ccpa_indirect draft",
                                   f"Then: python3 scripts/pdd.py record {subject_id} {bid} submitted "
                                   f"--disclosed contact_email --channel email"],
                                  prep=[f"python3 scripts/pdd.py render-email {subject_id} {bid} "
                                        f"--kind ccpa_indirect --listing <url>"]))

    # 6) blocked sites: stealth pass if we have one, else the operator-browser path
    blocked = groups.get("blocked") or []
    if blocked:
        ids = [r["broker_id"] for r in blocked]
        if bool(env.get("BROWSERBASE_API_KEY")):
            actions.append({
                "type": "stealth_rescan",
                "broker_ids": ids,
                "how": "retry these with the cloud/stealth browser backend, then record real verdicts",
            })
        else:
            for r in blocked:
                digest.append(_digest(r, "site blocks automated access (anti-bot); a human browser gets through",
                                      ["Open the paste-ready search URL from `plan` in your everyday browser",
                                       "Report the verdict (or a screenshot) back to the agent",
                                       f"Agent records: python3 scripts/pdd.py record {subject_id} "
                                       f"{r['broker_id']} <found|not_found|indirect_exposure>"]))

    # 7) anything already parked as a human task
    for bid, case in sorted(ledger.items()):
        if case.get("state") == "human_task_queued":
            broker = by_id.get(bid) or {}
            digest.append(_digest({"broker_id": bid, "broker_name": broker.get("name") or bid},
                                  case.get("human_task_reason") or "queued manual step",
                                  ["See `pdd.py tasks` for the exact steps recorded with this case"]))

    # registry coverage summary (breadth beyond the scannable people-search sites)
    coverage = None
    if registry_recs:
        coverage = {
            "people_search_sites": len(brokers_list),
            "registered_data_brokers": len(registry_recs),
            "worked_via": "CA DROP one-shot" if residency.startswith("US-CA") else "targeted CCPA/GDPR email",
        }
        if not residency.startswith("US-CA"):
            coverage["note"] = ("DROP is CA-only; for this subject the registry is covered by targeted "
                                "CCPA/GDPR deletion emails (`registry --search` then `send-email`), "
                                "not a single portal request.")
        elif drop_filed:
            coverage["note"] = "DROP already filed; registry deletions are in the brokers' hands."

    next_wake = _min_future_recheck(ledger, at)
    return {
        "subject": subject_id,
        "autonomy": autonomy,
        "phase": batch.get("phase"),
        "counts": batch.get("counts"),
        "actions": actions,
        "human_digest": digest,
        "coverage": coverage,
        "done_for_now": not actions,
        "fully_done": not actions and not digest and not next_wake,
        "next_wake_at": next_wake,
        "note": ("assisted mode: pause for operator confirmation on every action with confirm_first=true"
                 if confirm_first else
                 "full autonomy: recorded intake consent authorizes these submissions; do not pause. "
                 "Present human_digest ONCE at the end of the run, not per item."),
    }
