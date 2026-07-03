"""Case ledger: opt-out state machine + append-only audit log.

A "case" is one (subject x broker) record. State changes are validated against
TRANSITIONS and mirrored into audit.jsonl so every action is auditable.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import paths
import storage

STATES = [
    "new", "searching", "not_found", "found", "indirect_exposure", "action_selected", "submitted",
    "verification_pending", "awaiting_processing", "confirmed_removed", "reappeared",
    "human_task_queued", "blocked",
]

TRANSITIONS: dict[str, set[str]] = {
    "new": {"searching", "found", "not_found", "indirect_exposure", "blocked"},
    "searching": {"not_found", "found", "indirect_exposure", "blocked"},
    "not_found": {"searching", "found", "indirect_exposure", "blocked"},
    "found": {"action_selected", "submitted", "human_task_queued", "indirect_exposure", "blocked"},
    # indirect_exposure: subject's PII (email/phone/name) sits on a THIRD PARTY's record. The
    # self-service opt-out form does not apply; the lever is a targeted CCPA/GDPR delete-my-PII
    # request (-> submitted) or a human task. Re-scan can clear it (-> not_found) or upgrade it to a
    # direct listing (-> found).
    "indirect_exposure": {"submitted", "human_task_queued", "not_found", "found", "blocked"},
    "action_selected": {"submitted", "human_task_queued", "blocked"},
    "submitted": {"verification_pending", "awaiting_processing", "human_task_queued", "blocked"},
    # verification_pending -> awaiting_processing: the verify link was opened/acknowledged and the
    # broker is now processing the removal (their stated window). confirmed_removed still requires a
    # verifying re-scan, never the submission flow's own say-so.
    "verification_pending": {"awaiting_processing", "confirmed_removed", "human_task_queued", "blocked"},
    "awaiting_processing": {"confirmed_removed", "human_task_queued", "blocked"},
    "confirmed_removed": {"reappeared", "confirmed_removed"},
    "reappeared": {"found", "indirect_exposure"},
    "human_task_queued": {
        "found", "indirect_exposure", "action_selected", "submitted", "verification_pending",
        "awaiting_processing", "confirmed_removed", "blocked",
    },
    # blocked: automated tools (web_extract/proxyless browser) couldn't read the site. A later pass
    # -- a stealth/cloud browser OR guiding the operator's own (residential) browser -- can resolve it
    # to any real scan verdict, so blocked reaches not_found / indirect_exposure too, not just found.
    "blocked": {"searching", "found", "not_found", "indirect_exposure", "action_selected"},
}


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load(subject_id: str) -> dict:
    return storage.read_json(paths.ledger_path(subject_id), {}) or {}


def save(subject_id: str, ledger: dict) -> Path:
    return storage.write_json(paths.ledger_path(subject_id), ledger)


def new_case(subject_id: str, broker_id: str) -> dict:
    return {
        "case_id": f"case_{subject_id}_{broker_id}",
        "subject_id": subject_id,
        "broker_id": broker_id,
        "state": "new",
        "found": None,
        "evidence": {},
        "disclosure_log": [],
        "history": [],
    }


def get_case(subject_id: str, broker_id: str) -> dict:
    return load(subject_id).get(broker_id) or new_case(subject_id, broker_id)


def can_transition(old: str, new: str) -> bool:
    return new == old or new in TRANSITIONS.get(old, set())


def transition(subject_id: str, broker_id: str, new_state: str, **fields) -> dict:
    if new_state not in STATES:
        raise ValueError(f"unknown state {new_state!r}")
    # Lock the whole load-modify-save so a concurrent cron re-scan / other tenant
    # can't read a stale ledger and clobber this transition.
    with storage.locked(paths.ledger_path(subject_id)):
        ledger = load(subject_id)
        case = ledger.get(broker_id) or new_case(subject_id, broker_id)
        old = case.get("state", "new")
        if not can_transition(old, new_state):
            raise ValueError(f"illegal transition {old!r} -> {new_state!r} for broker {broker_id!r}")
        case["state"] = new_state
        for key, value in fields.items():
            case[key] = value
        stamp = now()
        case.setdefault("history", []).append({"at": stamp, "from": old, "to": new_state})
        ledger[broker_id] = case
        save(subject_id, ledger)
        storage.append_jsonl(
            paths.audit_path(subject_id),
            {"at": stamp, "broker_id": broker_id, "event": "transition", "from": old, "to": new_state},
        )
        return case


DEFAULT_PROCESSING_DAYS = 14   # when a broker record doesn't state est_processing_days
VERIFICATION_POLL_DAYS = 1     # how soon to re-poll for an unarrived verification email


def _plus_days(days: int, start: str | None = None) -> str:
    base = _dt.datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_dt.timezone.utc) \
        if start else _dt.datetime.now(_dt.timezone.utc)
    return (base + _dt.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def followup_fields(new_state: str, broker: dict | None = None,
                    dossier: dict | None = None) -> dict:
    """Auto-scheduling stamps for a transition, so nobody has to remember follow-ups.

    submitted / awaiting_processing -> recheck after the broker's stated processing window;
    verification_pending            -> re-poll the inbox quickly;
    confirmed_removed               -> periodic reappearance re-scan per subject preference.
    """
    if new_state in ("submitted", "awaiting_processing"):
        days = ((broker or {}).get("optout") or {}).get("est_processing_days") or DEFAULT_PROCESSING_DAYS
        return {"next_recheck_at": _plus_days(int(days))}
    if new_state == "verification_pending":
        return {"next_recheck_at": _plus_days(VERIFICATION_POLL_DAYS)}
    if new_state == "confirmed_removed":
        interval = ((dossier or {}).get("preferences") or {}).get("rescan_interval_days") or 120
        return {"removal_confirmed_at": now(), "next_recheck_at": _plus_days(int(interval))}
    return {}


def due(subject_id: str, at: str | None = None, ledger: dict | None = None) -> list[dict]:
    """Cases whose next_recheck_at has arrived - the autonomous follow-up queue."""
    stamp = at or now()
    out = []
    for case in (ledger if ledger is not None else load(subject_id)).values():
        when = case.get("next_recheck_at")
        if when and when <= stamp:
            out.append(case)
    out.sort(key=lambda c: c.get("next_recheck_at") or "")
    return out


def log_disclosure(subject_id: str, broker_id: str, fields: list[str], channel: str) -> dict:
    """Record exactly which PII field *names* were disclosed to a broker."""
    with storage.locked(paths.ledger_path(subject_id)):
        ledger = load(subject_id)
        case = ledger.get(broker_id) or new_case(subject_id, broker_id)
        stamp = now()
        record = {"at": stamp, "fields": sorted(fields), "channel": channel}
        case.setdefault("disclosure_log", []).append(record)
        ledger[broker_id] = case
        save(subject_id, ledger)
        storage.append_jsonl(
            paths.audit_path(subject_id),
            {"at": stamp, "broker_id": broker_id, "event": "disclosure",
             "fields": record["fields"], "channel": channel},
        )
        return record
