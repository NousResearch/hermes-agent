"""Detect removal requests a broker has ignored past its statutory deadline, and
build a ready-to-file regulator-complaint DRAFT for each.

Filing a complaint with a regulator is a consequential, outward-facing legal action,
so it is NEVER auto-submitted: this module produces DRAFTS the operator reviews and
files, surfaced through the human-task lane - the same hand-off pattern as
`render-email`. Everything here is pure (dossier + ledger in, drafts out); no network
and no filesystem, so it is hermetically testable.

Statutory response windows:
  CCPA/CPRA        - 45 days for a business to respond to a deletion request
                     (Cal. Civ. Code 1798.130). Past this, non-response is a violation.
  GDPR Article 12(3) - one month (30 days) to act on an erasure request (Article 17).

Honesty gate: a complaint is only generated for a jurisdiction the subject can
truthfully invoke - CCPA for California residents, GDPR for EU/UK residents. A
"generic" subject (neither) has no single statutory deadline or regulator, so no
complaint is emitted for them. We never route (say) a Texan's complaint to the
California Attorney General. Regime detection reuses autopilot.request_kind, the same
logic the email lane already uses, so the two can never disagree.
"""
from __future__ import annotations

import datetime as _dt

import autopilot
import dossier as dossier_mod

# States in which a request has been made but the broker has not confirmed deletion.
PENDING_STATES = ("submitted", "verification_pending", "awaiting_processing")

CCPA_WINDOW_DAYS = 45
GDPR_WINDOW_DAYS = 30
_WINDOW = {"ccpa": CCPA_WINDOW_DAYS, "gdpr": GDPR_WINDOW_DAYS}

# Where each regime's complaint is filed. The GDPR authority is the subject's own
# national DPA (the EDPB members page lists them); we cannot know which one from
# residency alone, so we name the lane and link the directory rather than guess.
AGENCY = {
    "ccpa": {
        "name": "California Attorney General / California Privacy Protection Agency",
        "portal": "https://oag.ca.gov/report (CA AG) or https://cppa.ca.gov (CPPA)",
    },
    "gdpr": {
        "name": "your national Data Protection Authority",
        "portal": "https://edpb.europa.eu/about-edpb/about-edpb/members_en",
    },
}


def _parse(ts: str | None) -> _dt.datetime | None:
    try:
        return _dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_dt.timezone.utc)
    except (TypeError, ValueError):
        return None


def submitted_at(case: dict) -> str | None:
    """ISO timestamp of the FIRST transition into a submitted/pending state.

    The statutory clock starts when the removal request was made, so we take the
    earliest history entry whose target state is one a request produces. A case that
    never reached a pending state (e.g. only `found`) has no clock start -> None.
    """
    for h in case.get("history") or []:
        if h.get("to") in PENDING_STATES:
            return h.get("at")
    return None


def overdue_cases(subject_id: str, dossier: dict, ledger: dict, now: _dt.datetime | None = None,
                  ccpa_days: int = CCPA_WINDOW_DAYS, gdpr_days: int = GDPR_WINDOW_DAYS) -> list[dict]:
    """Cases whose statutory response window has elapsed with no confirmed removal.

    Pure over (dossier, ledger). Returns rows sorted most-overdue-first:
      {broker_id, regime, window_days, submitted_at, days_overdue}
    Only CCPA (CA) and GDPR (EU/UK) subjects yield rows; generic subjects get none
    (see the module honesty gate). `now` is injectable for tests.
    """
    regime = autopilot.request_kind(dossier)
    if regime not in _WINDOW:
        return []
    window = {"ccpa": ccpa_days, "gdpr": gdpr_days}[regime]
    now = now or _dt.datetime.now(_dt.timezone.utc)

    rows: list[dict] = []
    for bid, case in (ledger or {}).items():
        if case.get("state") not in PENDING_STATES:
            continue
        if case.get("removal_confirmed_at"):
            continue  # broker complied; nothing to complain about
        started = _parse(submitted_at(case))
        if not started:
            continue
        days_overdue = (now - started).days - window
        if days_overdue <= 0:
            continue
        rows.append({
            "broker_id": bid,
            "regime": regime,
            "window_days": window,
            "submitted_at": submitted_at(case),
            "days_overdue": days_overdue,
        })
    rows.sort(key=lambda r: r["days_overdue"], reverse=True)
    return rows


def _broker_contact(broker: dict) -> str:
    opt = (broker or {}).get("optout") or {}
    return opt.get("url") or opt.get("email") or broker.get("email") \
        or "(no public opt-out contact on file)"


def complaint_context(dossier: dict, broker: dict, row: dict) -> dict:
    """Least-disclosure {field} context for a complaint template.

    Names ONLY the subject's own identity (name + contact email) plus the
    already-filed request - no PII beyond what the removal request itself disclosed.
    """
    ident = dossier.get("identity", {})
    agency = AGENCY[row["regime"]]
    return {
        "agency_name": agency["name"],
        "agency_portal": agency["portal"],
        "full_name": ident.get("full_name") or "[your name]",
        "contact_email": dossier_mod.contact_email(dossier) or "[your email]",
        "broker_name": (broker or {}).get("name") or row["broker_id"],
        "broker_contact": _broker_contact(broker or {}),
        "submitted_date": (row.get("submitted_at") or "")[:10],
        "window_days": row["window_days"],
        "days_overdue": row["days_overdue"],
    }
