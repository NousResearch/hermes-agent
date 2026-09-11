from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, fields

from .meeting import MeetingPreflightResult
from .meeting_orchestrator import RoundAssignment


def _normalize_text(value: object) -> str:
    return " ".join(str(value).split())


def _normalize_items(
    values: Sequence[object],
    *,
    field_name: str,
) -> tuple[str, ...]:
    normalized = tuple(
        item
        for value in values
        if (item := _normalize_text(value))
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicates")
    return normalized


def _deduplicate(values: Sequence[object]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            item
            for value in values
            if (item := _normalize_text(value))
        )
    )


@dataclass(frozen=True)
class CommonMeetingContext:
    """Immutable facts and constraints shared identically with every Staff."""

    objective: str
    shared_facts: tuple[str, ...] = ()
    supplied_materials: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    customer_user_information: tuple[str, ...] = ()
    unresolved_questions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        objective = _normalize_text(self.objective)
        if not objective:
            raise ValueError("meeting objective is required")
        object.__setattr__(self, "objective", objective)

        for definition in fields(self):
            if definition.name == "objective":
                continue
            values = getattr(self, definition.name)
            normalized = _normalize_items(
                values,
                field_name=definition.name,
            )
            object.__setattr__(self, definition.name, normalized)

    def to_dict(self) -> dict[str, object]:
        return {
            "objective": self.objective,
            "shared_facts": list(self.shared_facts),
            "supplied_materials": list(self.supplied_materials),
            "constraints": list(self.constraints),
            "customer_user_information": list(
                self.customer_user_information
            ),
            "unresolved_questions": list(self.unresolved_questions),
        }


def common_context_from_preflight(
    preflight: MeetingPreflightResult,
) -> CommonMeetingContext:
    """Translate validated preflight evidence into one shared context."""

    shared_facts = _deduplicate(
        tuple(
            fact
            for source in preflight.sources
            if source.status == "available"
            for fact in source.facts
        )
    )
    supplied_materials = _deduplicate(
        tuple(
            evidence
            for source in preflight.sources
            for evidence in source.inspection_evidence
        )
    )
    constraints = _deduplicate(
        tuple(
            premise.requirement
            for premise in preflight.premise_checks
            if premise.status != "satisfied"
        )
    )
    unresolved_questions = _deduplicate(
        preflight.request_to_sinclair or ()
    )

    return CommonMeetingContext(
        objective=preflight.agenda,
        shared_facts=shared_facts,
        supplied_materials=supplied_materials,
        constraints=constraints,
        unresolved_questions=unresolved_questions,
    )


def build_candidate_packet(
    *,
    meeting_id: str,
    generation: int,
    assignment: RoundAssignment,
    agenda: str,
    facts: Sequence[str],
    common_context: CommonMeetingContext | None = None,
) -> str:
    meeting_id = meeting_id.strip()
    if not meeting_id:
        raise ValueError("meeting_id is required")
    if generation < 1:
        raise ValueError("generation must be >= 1")
    if assignment.round_id < 1:
        raise ValueError("round_id must be >= 1")

    normalized_agenda = _normalize_text(agenda)
    normalized_facts = _normalize_items(
        facts,
        field_name="facts",
    )
    shared_context = common_context or CommonMeetingContext(
        objective=normalized_agenda,
        shared_facts=normalized_facts,
    )

    payload = {
        "schema_version": 1,
        "meeting_id": meeting_id,
        "round_id": assignment.round_id,
        "generation": generation,
        "participant": assignment.profile,
        "agenda": normalized_agenda,
        "facts": list(normalized_facts),
        "common_context": shared_context.to_dict(),
        "round_context": assignment.context,
        "independent": assignment.independent,
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )
