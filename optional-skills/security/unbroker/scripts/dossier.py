"""Subject dossier management + consent gate + least-disclosure field selection."""
from __future__ import annotations

import datetime as _dt
import hashlib
import os
from pathlib import Path

import paths
import storage

# Identifiers we never volunteer in an opt-out (would expand exposure, not reduce it).
NEVER_VOLUNTEER = {"ssn", "social_security_number", "passport", "drivers_license"}

VALID_CONSENT_METHODS = {"self", "written_authorization", "poa"}


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_subject_id(full_name: str = "") -> str:
    # Opaque id: derives NOTHING from the name, so PII never leaks into directory names,
    # case ids, drafts, or the audit log. full_name kept only for call compatibility.
    return "sub_" + hashlib.sha1(os.urandom(8)).hexdigest()[:10]


def create(identity: dict, consent: dict, residency: str = "US", prefs: dict | None = None) -> dict:
    dossier = {
        "subject_id": new_subject_id(identity.get("full_name", "subject")),
        "consent": consent,
        "identity": identity,
        "residency_jurisdiction": residency,
        "preferences": prefs or {"email_mode": "draft_only", "rescan_interval_days": 120},
        "created_at": now(),
    }
    save(dossier)
    return dossier


def load(subject_id: str) -> dict | None:
    return storage.read_json(paths.dossier_path(subject_id), None)


def save(dossier: dict) -> Path:
    return storage.write_json(paths.dossier_path(dossier["subject_id"]), dossier)


def is_authorized(dossier: dict) -> bool:
    c = dossier.get("consent") or {}
    return bool(c.get("authorized")) and c.get("method") in VALID_CONSENT_METHODS


def require_authorized(dossier: dict) -> None:
    if not is_authorized(dossier):
        raise PermissionError(
            f"subject {dossier.get('subject_id')!r} has no recorded authorization; refusing to act"
        )


def all_names(dossier: dict) -> list[str]:
    """Primary name + aliases (maiden/married/nicknames), deduped, in priority order."""
    ident = dossier.get("identity", {})
    out: list[str] = []
    seen: set[str] = set()
    for n in [ident.get("full_name"), *(ident.get("also_known_as") or [])]:
        if n and n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out


def all_addresses(dossier: dict) -> list[dict]:
    """Current + prior addresses, each tagged with `kind` (current|prior)."""
    ident = dossier.get("identity", {})
    out: list[dict] = []
    cur = ident.get("current_address")
    if cur:
        out.append({**cur, "kind": cur.get("kind", "current")})
    for a in ident.get("prior_addresses") or []:
        out.append({**a, "kind": a.get("kind", "prior")})
    return out


def all_locations(dossier: dict) -> list[dict]:
    """Distinct city/state pairs across all addresses (the vectors for name searches)."""
    out: list[dict] = []
    seen: set[tuple] = set()
    for a in all_addresses(dossier):
        city = a.get("city")
        key = ((city or "").lower(), (a.get("state") or "").lower())
        if city and key not in seen:
            seen.add(key)
            out.append({"city": city, "state": a.get("state")})
    return out


def contact_email(dossier: dict) -> str | None:
    """The single email used for opt-out correspondence (designated, else the first)."""
    ident = dossier.get("identity", {})
    prefs = dossier.get("preferences", {})
    emails = ident.get("emails") or []
    return prefs.get("contact_email_for_optouts") or (emails[0] if emails else None)


def select_disclosure(dossier: dict, inputs: list[str], override_email: str | None = None) -> dict:
    """Return ONLY the dossier fields a broker's opt-out actually requires.

    Enforces least-disclosure: skips anything in NEVER_VOLUNTEER, and skips
    `profile_url` (that is captured per-listing at submit time, not from the dossier).
    A single contact email is used for correspondence even when the subject has several
    (see all_names / all_addresses / search vectors for using every alternate to *find* listings).
    """
    ident = dossier.get("identity", {})
    addr = ident.get("current_address") or {}
    phones = ident.get("phones") or []
    available = {
        "full_name": ident.get("full_name"),
        "first_name": (ident.get("full_name") or "").split(" ")[0] or None,
        "contact_email": override_email or contact_email(dossier),
        "current_address": addr or None,
        "street": addr.get("line1"),
        "city": addr.get("city"),
        "state": addr.get("state"),
        "postal": addr.get("postal"),
        "date_of_birth": ident.get("date_of_birth"),
        "phone": phones[0] if phones else None,
    }
    out: dict = {}
    for key in inputs:
        if key in NEVER_VOLUNTEER or key == "profile_url":
            continue
        if available.get(key) is not None:
            out[key] = available[key]
    return out
