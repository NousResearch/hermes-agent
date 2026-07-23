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

# Residency -> legal framework. US codes map to CCPA/CPRA state variants (existing behaviour);
# EU-*, EEA-* and UK codes map to GDPR/UK-GDPR. Anything else falls back to a generic
# right-to-delete request (no specific legal cite — the broker may or may not honour it).
RESIDENCY_LEGAL_FRAMEWORK = {
    # United States
    "US":     {"framework": "ccpa",      "default_request_kind": "ccpa",       "dpa": None},
    "US-CA":  {"framework": "ccpa",      "default_request_kind": "ccpa",       "dpa": None},
    "US-NY":  {"framework": "ccpa_ny",   "default_request_kind": "ccpa",       "dpa": None},
    "US-VT":  {"framework": "ccpa_vt",   "default_request_kind": "ccpa",       "dpa": None},
    "US-OR":  {"framework": "ccpa_or",   "default_request_kind": "ccpa",       "dpa": None},
    "US-TX":  {"framework": "ccpa_tx",   "default_request_kind": "ccpa",       "dpa": None},
    # European Union + EEA (one code per supported member state — extend as new EU members join).
    # Country-specific codes route to a national DPA adapter. The catch-all EU / EU-EEA codes
    # intentionally stay generic because they do not identify the subject's competent authority.
    "EU":     {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": None},
    "EU-IT":  {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": "garante"},
    "EU-FR":  {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": "cnil"},
    "EU-DE":  {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": "bfdi"},
    "EU-AT":  {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": "dsb_at"},
    "EU-BE":  {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": "apd_gba"},
    "EU-BG":  {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": "cpdp_bg"},
    "EU-HR":  {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": "azop_hr"},
    "EU-CY":  {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": "cpdp_cy"},
    "EU-CZ":  {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": "uoou_cz"},
    "EU-EE":  {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": "aki_ee"},
    "EU-ES":  {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": "aepd"},
    "EU-FI":  {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": "tietosuoja"},
    "EU-GR":  {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": "hdpa_gr"},
    "EU-HU":  {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": "naih_hu"},
    "EU-IE":  {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": "dpc_ie"},
    "EU-LT":  {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": "vdai_lt"},
    "EU-LU":  {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": "cnpd_lu"},
    "EU-LV":  {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": "dvi_lv"},
    "EU-MT":  {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": "idpc_mt"},
    "EU-NL":  {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": "ap_nl"},
    "EU-PL":  {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": "uodo"},
    "EU-PT":  {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": "cnpd_pt"},
    "EU-RO":  {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": "anspdcp_ro"},
    "EU-DK":  {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": "datatilsynet_dk"},
    "EU-SE":  {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": "imy"},
    "EU-SI":  {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": "ip_rs"},
    "EU-SK":  {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": "uoou_sk"},
    "EU-EEA": {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": None},  # EEA but non-EU (Norway, Iceland, Liechtenstein)
    "EEA-IS": {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": "personuvernd_is"},
    "EEA-LI": {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": "datenschutzstelle_li"},
    "EEA-NO": {"framework": "gdpr",      "default_request_kind": "gdpr",       "dpa": "datatilsynet_no"},
    # United Kingdom (post-Brexit UK GDPR, enforced by ICO)
    "UK":     {"framework": "uk_gdpr",   "default_request_kind": "gdpr",       "dpa": "ico"},
}


def legal_framework(residency: str) -> dict:
    """Return the legal-framework metadata for a residency code; fallback for unknown codes.

    The fallback is deliberately permissive: an unknown residency code yields a generic
    right-to-delete request rather than refusing — a subject can still try, just without
    a specific GDPR/CCPA citation. Better than locking them out.
    """
    return RESIDENCY_LEGAL_FRAMEWORK.get(
        residency,
        {"framework": "generic", "default_request_kind": "generic", "dpa": None},
    )


def is_eu_residency(residency: str) -> bool:
    """True if the residency is an EU/EEA/UK code that maps to GDPR/UK-GDPR."""
    meta = RESIDENCY_LEGAL_FRAMEWORK.get(residency)
    return bool(meta and meta["framework"] in ("gdpr", "uk_gdpr"))


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
