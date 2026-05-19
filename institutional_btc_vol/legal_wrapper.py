from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EVIDENCE_STATUS = "SCREEN-ONLY · LEGAL WRAPPER DRAFT · NOT EXECUTABLE"
REQUIRED_APPROVAL_FIELDS = [
    "approved_by_counsel",
    "approved_scope",
    "approval_date",
    "allowed_activities",
    "prohibited_activities",
]

BOUNDARY_MATRIX = [
    {
        "activity": "Analytics / research",
        "current_status": "internal-only",
        "allowed_now": True,
        "external_requires": "counsel-approved disclaimer and distribution controls",
    },
    {
        "activity": "Advisory / structuring",
        "current_status": "blocked-pre-wrapper",
        "allowed_now": False,
        "external_requires": "RIA/CTA/CPO/BD perimeter analysis and written engagement terms",
    },
    {
        "activity": "Execution / RFQ support",
        "current_status": "blocked-pre-wrapper",
        "allowed_now": False,
        "external_requires": "broker/FCM/OTC partner model, no agency ambiguity, counterparty onboarding controls",
    },
    {
        "activity": "Principal risk-taking",
        "current_status": "blocked-pre-wrapper",
        "allowed_now": False,
        "external_requires": "entity/risk capital approval, mandate limits, conflicts policy, trade surveillance",
    },
    {
        "activity": "Structured products / fund sleeve",
        "current_status": "blocked-pre-wrapper",
        "allowed_now": False,
        "external_requires": "securities/private fund counsel, offering docs, investor eligibility, marketing review",
    },
]


def build_legal_wrapper_package(*, approved_by_counsel: bool = False) -> dict[str, Any]:
    blockers = [] if approved_by_counsel else [
        "counsel approval missing",
        "external advisory/client/RFQ/fund use prohibited until wrapper approved",
        "quote evidence remains internal diligence only",
    ]
    return {
        "evidence_status": EVIDENCE_STATUS,
        "approved_by_counsel": approved_by_counsel,
        "status": "approved" if approved_by_counsel else "draft-blocked",
        "label": "COUNSEL APPROVED" if approved_by_counsel else "DRAFT — NOT APPROVED FOR EXTERNAL USE",
        "control_note": "Legal/business wrapper draft only. This artifact is not legal advice and does not authorize external client, investor, RFQ, execution, advisory, or fund activity.",
        "blockers": blockers,
        "required_approval_fields": REQUIRED_APPROVAL_FIELDS,
        "boundary_matrix": BOUNDARY_MATRIX,
        "counsel_questions": [
            "Can analytics/research be distributed externally, and under what disclaimers?",
            "Does treasury/miner hedge structuring create investment adviser, CTA, CPO, broker-dealer, FCM, or swap-adviser issues?",
            "What engagement letter, suitability, KYC/KYB, and conflicts controls are required before any client mandate?",
            "Can RFQ support be handled through a partner broker/FCM/OTC desk without agency/execution ambiguity?",
            "What marketing/investor deck language is prohibited before a fund or structured-product wrapper exists?",
            "What records, surveillance, and approvals are required before principal risk-taking?",
        ],
        "hard_gates": [
            "No external investor/client use until counsel-approved wrapper exists",
            "No RFQ submission by automation",
            "No executable quote labels without two distinct external indicative quote records",
            "No trade-verified label without execution record and post-trade evidence",
            "No offering/fund/product language without securities/private-fund counsel approval",
        ],
    }


def write_legal_wrapper_package(output_json: str | Path, output_md: str | Path | None = None) -> dict[str, Any]:
    package = build_legal_wrapper_package(approved_by_counsel=False)
    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = Path(output_md) if output_md else json_path.with_suffix(".md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    boundary = "\n".join(
        f"- **{row['activity']}** — {row['current_status']} — external requires: {row['external_requires']}"
        for row in package["boundary_matrix"]
    )
    blockers = "\n".join(f"- {item}" for item in package["blockers"]) or "- none"
    questions = "\n".join(f"- {item}" for item in package["counsel_questions"])
    gates = "\n".join(f"- {item}" for item in package["hard_gates"])
    md_path.write_text(
        "# BTC Vol Desk Legal / Business Wrapper Draft v1\n\n"
        f"**Evidence status:** `{package['evidence_status']}`\n\n"
        f"**Status:** `{package['label']}`\n\n"
        f"> {package['control_note']}\n\n"
        "## Blockers\n\n"
        f"{blockers}\n\n"
        "## Business Boundary Matrix\n\n"
        f"{boundary}\n\n"
        "## Counsel Questions\n\n"
        f"{questions}\n\n"
        "## Hard Gates\n\n"
        f"{gates}\n",
        encoding="utf-8",
    )
    return {"ok": True, "json_path": str(json_path), "markdown_path": str(md_path), **package}


def load_legal_wrapper_package(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return build_legal_wrapper_package(approved_by_counsel=False) | {"path": str(target), "missing": True}
    data = json.loads(target.read_text(encoding="utf-8"))
    return {"path": str(target), **data}
