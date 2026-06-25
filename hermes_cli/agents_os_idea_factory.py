"""Deterministic Idea Factory draft classifier for local Agents OS planning.

This module is intentionally local-only and side-effect free. It turns a raw idea
into a structured execution draft; actual task creation/execution stays in the
Agents OS command/API layer and must respect approval gates.
"""

from __future__ import annotations

from hashlib import sha1
from typing import Any


CLASSIFICATIONS = [
    "web_seo_offer",
    "agent_os_build",
    "coding_task",
    "research_intake",
    "memory_skill_ops",
    "media_asset",
    "security_gated",
    "finance_gated",
    "public_outbound_gated",
    "unclear_needs_context",
]

RISK_CLASSES = [
    "safe_local",
    "approval_local_write",
    "credential_gated",
    "public_gated",
    "security_gated",
    "finance_gated",
    "destructive_gated",
]

INPUT_FIELDS = ["idea_text", "context", "desired_output", "urgency"]
OUTPUT_FIELDS = [
    "idea_id",
    "classification",
    "risk_class",
    "recommended_lane",
    "plan_steps",
    "approval_required",
    "suggested_agent",
    "expected_artifacts",
    "acceptance_criteria",
    "source_links",
]


def idea_factory_schema() -> dict[str, Any]:
    return {
        "input_fields": list(INPUT_FIELDS),
        "output_fields": list(OUTPUT_FIELDS),
        "classifications": list(CLASSIFICATIONS),
        "risk_classes": list(RISK_CLASSES),
        "local_only": True,
        "side_effects": False,
    }


def _contains(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


def _classify(text: str) -> tuple[str, str, str]:
    normalized = text.lower()

    if _contains(normalized, "kraken", "trading", "trade", "invest", "portfolio", "burza", "crypto"):
        return "finance_gated", "finance_gated", "finance-approval-gate"
    if _contains(normalized, "skeniraj", "ranjiv", "pentest", "exploit", "hak", "hack", "security test"):
        return "security_gated", "security_gated", "security-scope-gate"
    if _contains(normalized, "pošalji", "posalji", "email", "objavi", "postaj", "klijentu", "outbound"):
        return "public_outbound_gated", "public_gated", "public-action-approval"
    if _contains(normalized, "credential", "api ključ", "api key", "token", "oauth", "lozink"):
        return "unclear_needs_context", "credential_gated", "credential-approval-gate"
    if _contains(normalized, "obriši", "obrisi", "delete", "izbriši", "izbrisi", "wipe", "remove"):
        return "memory_skill_ops", "destructive_gated", "destructive-action-gate"
    if _contains(normalized, "youtube", "video", "transcript", "vault", "sažmi", "sazmi", "obradi"):
        return "research_intake", "safe_local", "youtube-content-intake"
    if _contains(normalized, "memory galaxy", "mission control", "agent os", "operator dashboard", "dashboard", "jarvis", "voice"):
        return "agent_os_build", "approval_local_write", "mission-control-build"
    if _contains(normalized, "landing", "web", "seo", "revenue audit", "ponud", "offer", "stranic"):
        return "web_seo_offer", "safe_local", "web-seo-offer"
    if _contains(normalized, "kod", "code", "bug", "repo", "test", "implement"):
        return "coding_task", "approval_local_write", "code-build"
    if _contains(normalized, "slika", "audio", "glas", "video asset", "media", "screenshot"):
        return "media_asset", "safe_local", "media-asset-draft"

    return "unclear_needs_context", "safe_local", "clarify-or-research"


def _plan_for(classification: str, risk_class: str) -> list[str]:
    if risk_class.endswith("gated") or risk_class in {"public_gated", "security_gated", "finance_gated", "destructive_gated", "credential_gated"}:
        return [
            "Sažeti namjeru i izdvojiti rizične radnje.",
            "Napraviti approval draft bez izvršavanja rizične akcije.",
            "Čekati eksplicitno operator odobrenje prije bilo kakvog side-effecta.",
        ]
    if classification == "agent_os_build":
        return [
            "Zaključati capability contract i acceptance kriterije.",
            "Napisati test-first lokalni slice bez gateway restarta.",
            "Verificirati kroz focused test i local smoke prije spajanja u UI.",
        ]
    if classification == "research_intake":
        return [
            "Dohvatiti ili pročitati source artefakt.",
            "Spremiti transcript/note u canonical vault.",
            "Izvući routing odluku i sljedeći konkretan korak.",
        ]
    return [
        "Pretvoriti ideju u mali lokalni plan.",
        "Spremiti očekivane artefakte i acceptance kriterije.",
        "Kreirati safe local task ako nema approval-gated radnji.",
    ]


def draft_idea(
    idea_text: str,
    *,
    context: str | None = None,
    desired_output: str | None = None,
    urgency: str = "normal",
    source_links: list[str] | None = None,
) -> dict[str, Any]:
    if not idea_text or not idea_text.strip():
        raise ValueError("idea_text is required")

    classification, risk_class, lane = _classify(" ".join(x for x in [idea_text, context or "", desired_output or ""] if x))
    approval_required = risk_class != "safe_local"
    digest = sha1(idea_text.strip().encode("utf-8")).hexdigest()[:10]

    return {
        "idea_id": f"idea-{digest}",
        "classification": classification,
        "risk_class": risk_class,
        "recommended_lane": lane,
        "plan_steps": _plan_for(classification, risk_class),
        "approval_required": approval_required,
        "suggested_agent": "local-agent",
        "expected_artifacts": ["idea_draft", "plan", "verification_note"],
        "acceptance_criteria": [
            "classification and risk class are explicit",
            "approval gate is respected",
            "local artifact/task can be audited",
        ],
        "source_links": list(source_links or []),
        "urgency": urgency,
    }
