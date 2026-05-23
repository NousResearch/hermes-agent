"""Safety checks for routing Hermes prompts into jcode."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    risk_types: list[str]
    confirmation_fields: list[str]
    details: dict[str, Any]


_OUTBOUND_ACTION_TERMS = (
    "send",
    "dm",
    "direct message",
    "message",
    "email",
    "text",
    "sms",
    "reply",
    "post",
    "comment",
    "connect",
    "invite",
    "call",
)

_OUTBOUND_DESTINATION_TERMS = (
    "linkedin",
    "twitter",
    "x.com",
    "slack",
    "discord",
    "gmail",
    "email",
    "phone",
    "sms",
    "whatsapp",
    "telegram",
    "signal",
    "instagram",
    "facebook",
    "person",
    "friend",
    "recruiter",
    "client",
    "customer",
)

_PERSON_DATA_ACTION_TERMS = (
    "find",
    "lookup",
    "look up",
    "search",
    "get",
    "discover",
    "track down",
    "scrape",
    "identify",
)

_SENSITIVE_PERSON_DATA_TERMS = (
    "phone number",
    "cell number",
    "mobile number",
    "home address",
    "personal address",
    "personal email",
    "private email",
    "ssn",
    "social security",
    "date of birth",
    "dob",
)

_PERSON_REFERENCE_TERMS = (
    "friend",
    "classmate",
    "coworker",
    "co-worker",
    "colleague",
    "person",
    "someone",
    "individual",
    "recruiter",
    "client",
    "customer",
)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _normalize(text: str) -> str:
    lowered = text.lower()
    return re.sub(r"\s+", " ", lowered).strip()


def _has_term(text: str, term: str) -> bool:
    if " " in term or "." in term or "-" in term:
        return term in text
    return re.search(rf"\b{re.escape(term)}\b", text) is not None


def _matched_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if _has_term(text, term)]


def evaluate_jcode_bridge_safety(message: str, args: dict[str, Any]) -> SafetyDecision:
    """Return whether Hermes may route this prompt to jcode unattended.

    The bridge intentionally errs on the side of requiring a caller-visible
    confirmation for account actions and private personal-data discovery. The
    confirmation flags are route/tool controls, not a blanket permission system;
    Hermes still owns any higher-level policy or approval UX around the call.
    """
    text = _normalize(message)
    risks: list[str] = []
    confirmation_fields: list[str] = []
    details: dict[str, Any] = {}

    outbound_actions = _matched_terms(text, _OUTBOUND_ACTION_TERMS)
    outbound_destinations = _matched_terms(text, _OUTBOUND_DESTINATION_TERMS)
    if outbound_actions and outbound_destinations:
        details["outbound_human_contact"] = {
            "action_terms": outbound_actions,
            "destination_terms": outbound_destinations,
        }
        if not _as_bool(args.get("confirm_outbound_human_contact")):
            risks.append("outbound_human_contact")
            confirmation_fields.append("confirm_outbound_human_contact")

    person_data_actions = _matched_terms(text, _PERSON_DATA_ACTION_TERMS)
    sensitive_data_terms = _matched_terms(text, _SENSITIVE_PERSON_DATA_TERMS)
    person_references = _matched_terms(text, _PERSON_REFERENCE_TERMS)
    if sensitive_data_terms and (person_data_actions or person_references):
        details["sensitive_person_data"] = {
            "action_terms": person_data_actions,
            "data_terms": sensitive_data_terms,
            "person_reference_terms": person_references,
        }
        if not _as_bool(args.get("confirm_sensitive_person_data")):
            risks.append("sensitive_person_data")
            confirmation_fields.append("confirm_sensitive_person_data")

    return SafetyDecision(
        allowed=not risks,
        risk_types=risks,
        confirmation_fields=confirmation_fields,
        details=details,
    )
