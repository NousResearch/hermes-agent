"""Side-effect-free specialist dispatch gate policy."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

SIDE_EFFECTS_PERFORMED = {
    "kanban_task_created": False,
    "worker_launched": False,
    "cron_created": False,
    "external_send": False,
}

STANDARD_SUBAGENT_OUTPUT_SECTIONS = (
    "status",
    "result",
    "evidence",
    "limits_or_uncertainties",
    "risks",
    "recommended_next_action",
)

ROLE_CARDS: dict[str, dict[str, Any]] = {
    "hermespm": {
        "boards": {"sviluppo-hermes"},
        "actions": {"development_task", "dashboard_quality", "policy_refactor", "test_task"},
        "external_allowed": False,
        "purpose": "Coordina sviluppo Hermes, policy leggere, qualità dashboard e test senza creare lavoro nascosto.",
        "does_not": [
            "Non sostituisce Daniele nelle decisioni di priorità prodotto.",
            "Non avvia worker, cron o dispatch senza gate esplicito.",
            "Non propone automazioni nuove se non mitigano una frizione osservata.",
        ],
        "required_output_sections": STANDARD_SUBAGENT_OUTPUT_SECTIONS,
        "evidence_required": True,
    },
    "reliability": {
        "boards": {"sviluppo-hermes"},
        "actions": {"failure_triage", "regression_test", "watchdog_review", "incident_review"},
        "external_allowed": False,
        "purpose": "Isola failure mode, regressioni e rischi operativi con evidenze riproducibili.",
        "does_not": [
            "Non riscrive architetture sane senza un failure osservato.",
            "Non lancia watchdog o recovery automatici come default.",
            "Non considera completata un'analisi senza log, test o fonte verificabile.",
        ],
        "required_output_sections": STANDARD_SUBAGENT_OUTPUT_SECTIONS,
        "evidence_required": True,
    },
    "research": {
        "boards": {"team-operativo"},
        "actions": {"source_extraction", "literature_review", "transcript_analysis", "evidence_pack"},
        "external_allowed": False,
        "purpose": "Raccoglie fonti, transcript, estratti e pacchetti evidenza; prepara materiale per il Chief.",
        "does_not": [
            "Non produce decisioni finali senza verifica del Chief.",
            "Non inventa fonti mancanti o conclusioni non supportate.",
            "Non trasforma ricerca in invii esterni o task ready.",
        ],
        "required_output_sections": STANDARD_SUBAGENT_OUTPUT_SECTIONS,
        "evidence_required": True,
    },
    "co2farm-mrv": {
        "boards": {"team-operativo"},
        "actions": {"claim_qa", "evidence_register", "mrv_review", "methodology_check"},
        "external_allowed": False,
        "purpose": "Verifica claim, registri evidenza, MRV e metodologia CO2Farm/Kania con tracciabilità.",
        "does_not": [
            "Non approva claim esterni o marketing senza review Daniele.",
            "Non colma buchi RCI/metodologici con assunzioni non marcate.",
            "Non modifica documenti finali o registri ufficiali senza gate.",
        ],
        "required_output_sections": STANDARD_SUBAGENT_OUTPUT_SECTIONS,
        "evidence_required": True,
    },
    "docs-delivery": {
        "boards": {"team-operativo"},
        "actions": {"docx_task", "xlsx_task", "pdf_review", "drive_delivery"},
        "external_allowed": False,
        "purpose": "Prepara e revisiona deliverable Word/Excel/PDF mantenendo template, allegati e review gate.",
        "does_not": [
            "Non invia documenti a terzi senza conferma esterna separata.",
            "Non sovrascrive template o versioni finali senza backup/contesto.",
            "Non dichiara consegna completata senza percorso/file verificabile.",
        ],
        "required_output_sections": STANDARD_SUBAGENT_OUTPUT_SECTIONS,
        "evidence_required": True,
    },
}

ALL_DECISIONS = ("ready_to_launch", "preview_only", "blocked_for_input", "denied")


def get_role_contract(role: str, role_cards: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Return the lean specialist contract for a subagent role.

    This is intentionally a small, side-effect-free projection of ROLE_CARDS:
    enough for prompts/UI/gates to agree on purpose, limits, output shape and
    evidence policy without introducing a second orchestration system.
    """

    cards = role_cards or ROLE_CARDS
    role_key = _lower(role)
    card = cards.get(role_key)
    if not card:
        return {
            "role": role_key,
            "known": False,
            "purpose": "Unknown specialist role.",
            "does_not": [],
            "allowed_boards": [],
            "allowed_actions": [],
            "external_allowed": False,
            "evidence_required": True,
            "required_output_sections": list(STANDARD_SUBAGENT_OUTPUT_SECTIONS),
        }
    return {
        "role": role_key,
        "known": True,
        "purpose": _text(card.get("purpose")),
        "does_not": [str(item) for item in card.get("does_not", [])],
        "allowed_boards": sorted(str(item) for item in card.get("boards", set())),
        "allowed_actions": sorted(str(item) for item in card.get("actions", set())),
        "external_allowed": bool(card.get("external_allowed")),
        "evidence_required": bool(card.get("evidence_required", True)),
        "required_output_sections": list(card.get("required_output_sections") or STANDARD_SUBAGENT_OUTPUT_SECTIONS),
    }


def build_standard_subagent_output_contract() -> str:
    """Markdown handoff shape every subagent should use in its final answer."""

    return (
        "## Subagent handoff contract\n"
        "Return your final answer with these sections:\n"
        "- `Status`: completed | partial | blocked | uncertain\n"
        "- `Result`: concise finding/accomplishment, scoped to your role\n"
        "- `Evidence`: files, URLs, commands, logs, line refs, or tool outputs used\n"
        "- `Limits / uncertainties`: missing context, assumptions, weak evidence\n"
        "- `Risks`: concrete risks or over-engineering concerns, if any\n"
        "- `Recommended next action`: one practical next step for the parent/Chief\n"
        "If evidence is required and you cannot provide it, use status `uncertain` or `blocked`, not `completed`."
    )


def _text(value: Any) -> str:
    return str(value or "").strip()


def _lower(value: Any) -> str:
    return _text(value).lower()


def _has_evidence(value: Any) -> bool:
    if isinstance(value, (list, tuple, set)):
        return any(_text(v) for v in value)
    return bool(_text(value))


def evaluate_dispatch_request(request: Mapping[str, Any], role_cards: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Evaluate one dispatch request without launching anything."""

    cards = role_cards or ROLE_CARDS
    role = _lower(request.get("role"))
    action = _lower(request.get("action_type"))
    board = _lower(request.get("board"))
    evidence_present = _has_evidence(request.get("evidence") or request.get("evidence_refs"))
    explicit_confirmation = bool(request.get("explicit_confirmation"))
    input_complete = bool(request.get("input_complete", True))
    external_effect = bool(request.get("external_effect"))
    reasons: list[str] = []

    card = cards.get(role)
    if not card:
        reasons.append(f"unknown role/profile: {role or 'missing'}")
        return _result(request, "denied", "none", False, reasons, role_cards=cards)

    allowed_boards = set(card.get("boards", set()))
    allowed_actions = set(card.get("actions", set()))
    if board not in allowed_boards:
        reasons.append(f"board '{board or 'missing'}' not allowed for {role}")
    if action not in allowed_actions:
        reasons.append(f"action '{action or 'missing'}' not allowed for {role}")
    if external_effect and not bool(card.get("external_allowed")):
        reasons.append("external effect requires separate external confirmation")

    # Role/board/action/external policy violations deny the dispatch outright.
    if reasons:
        return _result(request, "denied", "none", False, reasons, role_cards=cards)

    input_reasons: list[str] = []
    if not input_complete:
        input_reasons.append("input insufficient; use blocked only for real missing human input")
    if not evidence_present:
        input_reasons.append("missing required evidence/source")
    if input_reasons:
        return _result(request, "blocked_for_input", "blocked", False, input_reasons, role_cards=cards)

    if not explicit_confirmation:
        return _result(request, "preview_only", "none", False, ["missing explicit launch confirmation"], role_cards=cards)

    return _result(request, "ready_to_launch", "ready", True, [], role_cards=cards)


def _result(
    request: Mapping[str, Any],
    decision: str,
    recommended_status: str,
    can_launch: bool,
    reasons: list[str],
    *,
    role_cards: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    role = _lower(request.get("role"))
    board = _lower(request.get("board"))
    if decision == "ready_to_launch":
        ui_copy = (
            f"Se confermi nel sistema live, il lavoro per {role} su {board} può essere creato in ready. "
            "Nessun invio esterno, cron o dispatch fuori dai cap dichiarati."
        )
    elif decision == "preview_only":
        ui_copy = "Preview-only: nessun task Kanban, worker, cron o invio esterno viene creato."
    elif decision == "blocked_for_input":
        ui_copy = f"Il lavoro per {role} non è pronto: manca input umano/evidenza. Stato consigliato: blocked."
    else:
        ui_copy = "Dispatch negato: ruolo, board, azione o gate esterno non valido."

    return {
        "request_id": request.get("request_id") or request.get("id"),
        "title": request.get("title", ""),
        "role": role,
        "role_contract": get_role_contract(role, role_cards=role_cards),
        "action_type": _lower(request.get("action_type")),
        "board": board,
        "decision": decision,
        "recommended_kanban_status": recommended_status,
        "can_launch_if_live_system": can_launch,
        "reasons": reasons,
        "ui_copy": ui_copy,
    }


def evaluate_dispatch_requests(requests: Iterable[Mapping[str, Any]], role_cards: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Evaluate many dispatch requests and summarize the gate result."""

    results = [evaluate_dispatch_request(request, role_cards=role_cards) for request in requests]
    counts = {decision: 0 for decision in ALL_DECISIONS}
    for result in results:
        counts[result["decision"]] += 1
    return {
        "ok": True,
        "input_count": len(results),
        "counts": counts,
        "would_launch_count": counts["ready_to_launch"],
        "results": results,
        "side_effects_performed": dict(SIDE_EFFECTS_PERFORMED),
    }
