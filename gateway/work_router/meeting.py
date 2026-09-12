"""Strict, fail-closed meeting preflight and participant schema."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Mapping

PREFLIGHT_SCHEMA_VERSION = 3
PREFLIGHT_SOURCE_ORDER = ("notion", "kanban", "shared_files", "public_search")
PREMISE_KIND_ORDER = (
    "contract",
    "legal",
    "prerequisite",
    "schedule",
    "resource",
    "operational",
)
_INTERNAL_SOURCE_STATUSES = frozenset({"available", "missing", "unavailable"})
_PUBLIC_SEARCH_STATUS = "skipped_stage_2c"
_PREMISE_STATUSES = frozenset({"satisfied", "missing", "unavailable", "not_applicable"})
_SENSITIVE_LOOKUP_SOURCES = frozenset({"sales", "goldmine"})
_MAX_RESPONSE_LENGTH = 32_768
_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "agenda",
        "sources",
        "premise_checks",
        "missing_materials",
        "request_to_sinclair",
        "participants",
        "sensitive_lookup_request",
    }
)
_SOURCE_KEYS = frozenset({"source", "status", "facts", "inspection_evidence"})
_PREMISE_KEYS = frozenset(
    {"kind", "requirement", "status", "evidence", "blocking_reason", "required_materials"}
)
_SENSITIVE_LOOKUP_KEYS = frozenset({"sources", "targets", "items", "reason"})
_CANDIDATE_KEYS = frozenset(
    {"schema_version", "reason_to_speak", "reason_class", "statement"}
)
_CANDIDATE_REASON_CLASSES = frozenset(
    {"rebuttal", "fact_support", "missed_perspective", "risk_warning", "none"}
)
_SOURCE_LABELS = {
    "notion": "Notion",
    "kanban": "Kanban",
    "shared_files": "공유파일",
    "public_search": "외부 공개 검색",
}
_SOURCE_STATUS_LABELS = {
    "available": "확인됨",
    "missing": "관련 자료 없음",
    "unavailable": "조회 불가",
    "skipped_stage_2c": "이번 단계에서 조회하지 않음",
}
_PREMISE_LABELS = {
    "contract": "계약",
    "legal": "법률",
    "prerequisite": "선행 절차",
    "schedule": "일정",
    "resource": "자원",
    "operational": "운영 제약",
}
_PREMISE_STATUS_LABELS = {
    "satisfied": "충족",
    "missing": "자료 부족",
    "unavailable": "확인 불가",
    "not_applicable": "해당 없음",
}

BLOCKED_PREFLIGHT_ERROR_MESSAGE = (
    "자료 확인 결과를 안전하게 처리하지 못해 회의를 시작하지 않았습니다.\n"
    "자동으로 다시 시도하지 않았습니다.\n"
    "이 thread는 Work Mode로 복귀했습니다."
)
BLOCKED_ALL_ERROR_MESSAGE = (
    "참석자 의견을 정상적으로 수집하지 못해 회의를 진행하지 않았습니다.\n"
    "자동으로 다시 시도하지 않았습니다.\n"
    "이 thread는 Work Mode로 복귀했습니다."
)


class PreflightParseError(ValueError):
    """A safe, classified preflight parse failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class CandidateParseError(ValueError):
    """A safe, classified candidate parse failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PreflightSourceResult:
    source: str
    status: str
    facts: tuple[str, ...] = ()
    inspection_evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "status": self.status,
            "facts": list(self.facts),
            "inspection_evidence": list(self.inspection_evidence),
        }


@dataclass(frozen=True)
class PremiseCheckResult:
    kind: str
    requirement: str
    status: str
    evidence: tuple[str, ...]
    blocking_reason: str | None
    required_materials: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "requirement": self.requirement,
            "status": self.status,
            "evidence": list(self.evidence),
            "blocking_reason": self.blocking_reason,
            "required_materials": list(self.required_materials),
        }


@dataclass(frozen=True)
class SensitiveLookupRequest:
    sources: tuple[str, ...]
    targets: tuple[str, ...]
    items: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "sources": list(self.sources),
            "targets": list(self.targets),
            "items": list(self.items),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class MeetingPreflightResult:
    schema_version: int
    status: str
    agenda: str
    sources: tuple[PreflightSourceResult, ...]
    premise_checks: tuple[PremiseCheckResult, ...]
    missing_materials: tuple[str, ...]
    request_to_sinclair: tuple[str, ...] | None
    participants: tuple[str, ...]
    sensitive_lookup_request: SensitiveLookupRequest | None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "agenda": self.agenda,
            "sources": [source.to_dict() for source in self.sources],
            "premise_checks": [premise.to_dict() for premise in self.premise_checks],
            "missing_materials": list(self.missing_materials),
            "request_to_sinclair": (
                list(self.request_to_sinclair) if self.request_to_sinclair is not None else None
            ),
            "participants": list(self.participants),
            "sensitive_lookup_request": (
                self.sensitive_lookup_request.to_dict()
                if self.sensitive_lookup_request is not None
                else None
            ),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


def _source_contract_example(
    source: str,
    status: str,
    *,
    facts: tuple[str, ...] = (),
    inspection_evidence: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "source": source,
        "status": status,
        "facts": list(facts),
        "inspection_evidence": list(inspection_evidence),
    }


def _premise_contract_example(
    kind: str,
    requirement: str,
    status: str,
    *,
    evidence: tuple[str, ...] = (),
    blocking_reason: str | None = None,
    required_materials: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "kind": kind,
        "requirement": requirement,
        "status": status,
        "evidence": list(evidence),
        "blocking_reason": blocking_reason,
        "required_materials": list(required_materials),
    }


_READY_SCOPE_FACT = "계약·견적·고객 약속을 확정하지 않는 내부 전략 검토다."
_READY_FACTS = {
    "project": "프로젝트 범위가 확인됐다.",
    "schedule": "검토 일정이 확인됐다.",
    "resource": "담당 역할이 확인됐다.",
    "operational": "운영 제약이 확인됐다.",
}

_PREFLIGHT_RESPONSE_CONTRACT_EXAMPLES: tuple[dict[str, object], ...] = (
    {
        "schema_version": 3,
        "status": "ready",
        "agenda": "SK그룹 활성화 전략 내부 검토",
        "sources": [
            _source_contract_example(
                "notion",
                "available",
                facts=(_READY_FACTS["project"], _READY_SCOPE_FACT),
                inspection_evidence=("Notion 프로젝트 DB의 agenda 관련 항목을 확인했다.",),
            ),
            _source_contract_example(
                "kanban",
                "available",
                facts=(_READY_FACTS["schedule"], _READY_FACTS["resource"]),
                inspection_evidence=("Kanban의 agenda 관련 task와 event를 확인했다.",),
            ),
            _source_contract_example(
                "shared_files",
                "available",
                facts=(_READY_FACTS["operational"],),
                inspection_evidence=("공유파일의 승인된 내부 자료를 확인했다.",),
            ),
            _source_contract_example("public_search", "skipped_stage_2c"),
        ],
        "premise_checks": [
            _premise_contract_example(
                "contract",
                "계약·견적·고객 약속을 확정하는 안건인가",
                "not_applicable",
                evidence=(_READY_SCOPE_FACT,),
            ),
            _premise_contract_example(
                "legal",
                "법적 확정 또는 외부 의무가 발생하는 안건인가",
                "not_applicable",
                evidence=(_READY_SCOPE_FACT,),
            ),
            _premise_contract_example(
                "prerequisite",
                "검토 범위가 확인됐는가",
                "satisfied",
                evidence=(_READY_FACTS["project"],),
            ),
            _premise_contract_example(
                "schedule",
                "검토 일정이 확인됐는가",
                "satisfied",
                evidence=(_READY_FACTS["schedule"],),
            ),
            _premise_contract_example(
                "resource",
                "담당 역할이 확인됐는가",
                "satisfied",
                evidence=(_READY_FACTS["resource"],),
            ),
            _premise_contract_example(
                "operational",
                "운영 제약이 확인됐는가",
                "satisfied",
                evidence=(_READY_FACTS["operational"],),
            ),
        ],
        "missing_materials": [],
        "request_to_sinclair": None,
        "participants": ["Hans", "Mason", "Watson"],
        "sensitive_lookup_request": None,
    },
    {
        "schema_version": 3,
        "status": "blocked_materials",
        "agenda": "고객 제안 진행 조건 검토",
        "sources": [
            _source_contract_example(
                "notion",
                "available",
                facts=(
                    "고객과 제품 범위가 확인됐다.",
                    "법적 확정 또는 외부 의무를 결정하지 않는 내부 검토다.",
                    "일정·자원·운영 변경을 결정하지 않는 자료 확인 안건이다.",
                ),
                inspection_evidence=("Notion 프로젝트 DB의 고객 항목을 확인했다.",),
            ),
            _source_contract_example(
                "kanban",
                "missing",
                inspection_evidence=("Kanban의 관련 task를 확인했지만 최신 진행 상태가 없었다.",),
            ),
            _source_contract_example(
                "shared_files",
                "unavailable",
                inspection_evidence=("공유파일 connector를 사용할 수 없어 조회하지 못했다.",),
            ),
            _source_contract_example("public_search", "skipped_stage_2c"),
        ],
        "premise_checks": [
            _premise_contract_example(
                "contract",
                "고객에게 사용할 승인된 제안 조건이 확인됐는가",
                "missing",
                evidence=("공유파일 connector를 사용할 수 없어 조회하지 못했다.",),
                blocking_reason="승인된 제안 조건 없이 고객 제안을 확정할 수 없다.",
                required_materials=("승인된 제안서 원본",),
            ),
            _premise_contract_example(
                "legal",
                "법적 확정 또는 외부 의무가 발생하는 안건인가",
                "not_applicable",
                evidence=("법적 확정 또는 외부 의무를 결정하지 않는 내부 검토다.",),
            ),
            _premise_contract_example(
                "prerequisite",
                "최근 고객별 진행 상태가 확인됐는가",
                "missing",
                evidence=("Kanban의 관련 task를 확인했지만 최신 진행 상태가 없었다.",),
                blocking_reason="현재 진행 상태 없이 다음 제안 행동을 정할 수 없다.",
                required_materials=("최근 고객별 진행 상태",),
            ),
            _premise_contract_example(
                "schedule",
                "일정 판단이 현재 안건에 필요한가",
                "not_applicable",
                evidence=("일정·자원·운영 변경을 결정하지 않는 자료 확인 안건이다.",),
            ),
            _premise_contract_example(
                "resource",
                "자원 배정 판단이 현재 안건에 필요한가",
                "not_applicable",
                evidence=("일정·자원·운영 변경을 결정하지 않는 자료 확인 안건이다.",),
            ),
            _premise_contract_example(
                "operational",
                "운영 변경 판단이 현재 안건에 필요한가",
                "not_applicable",
                evidence=("일정·자원·운영 변경을 결정하지 않는 자료 확인 안건이다.",),
            ),
        ],
        "missing_materials": ["승인된 제안서 원본", "최근 고객별 진행 상태"],
        "request_to_sinclair": ["승인된 제안서 원본", "최근 고객별 진행 상태"],
        "participants": [],
        "sensitive_lookup_request": None,
    },
    {
        "schema_version": 3,
        "status": "awaiting_sensitive_approval",
        "agenda": "SK네트웍스와 SKAX의 재접촉 여부 검토",
        "sources": [
            _source_contract_example(
                "notion",
                "available",
                facts=(
                    "재접촉 여부를 구분해야 하는 안건이다.",
                    "외부 연락이나 법적·운영 변경을 실행하지 않는 내부 검토다.",
                ),
                inspection_evidence=("Notion 프로젝트 DB의 고객 항목을 확인했다.",),
            ),
            _source_contract_example(
                "kanban",
                "available",
                facts=("검토 일정과 담당 역할이 확인됐다.",),
                inspection_evidence=("Kanban의 관련 task와 event를 확인했다.",),
            ),
            _source_contract_example(
                "shared_files",
                "missing",
                inspection_evidence=("공유파일을 확인했지만 영업 이력 자료가 없었다.",),
            ),
            _source_contract_example("public_search", "skipped_stage_2c"),
        ],
        "premise_checks": [
            _premise_contract_example(
                "contract",
                "현재 영업 단계와 이전 사업 이력이 확인됐는가",
                "unavailable",
                evidence=("공유파일을 확인했지만 영업 이력 자료가 없었다.",),
                blocking_reason="Sales·Goldmine 조회는 Sinclair 승인이 필요하다.",
                required_materials=("현재 영업 단계", "이전 사업 이력"),
            ),
            _premise_contract_example(
                "legal",
                "법적 확정 또는 외부 의무가 발생하는 안건인가",
                "not_applicable",
                evidence=("외부 연락이나 법적·운영 변경을 실행하지 않는 내부 검토다.",),
            ),
            _premise_contract_example(
                "prerequisite",
                "검토 대상 고객이 확인됐는가",
                "satisfied",
                evidence=("재접촉 여부를 구분해야 하는 안건이다.",),
            ),
            _premise_contract_example(
                "schedule",
                "검토 일정이 확인됐는가",
                "satisfied",
                evidence=("검토 일정과 담당 역할이 확인됐다.",),
            ),
            _premise_contract_example(
                "resource",
                "담당 역할이 확인됐는가",
                "satisfied",
                evidence=("검토 일정과 담당 역할이 확인됐다.",),
            ),
            _premise_contract_example(
                "operational",
                "운영 변경 판단이 현재 안건에 필요한가",
                "not_applicable",
                evidence=("외부 연락이나 법적·운영 변경을 실행하지 않는 내부 검토다.",),
            ),
        ],
        "missing_materials": [],
        "request_to_sinclair": None,
        "participants": [],
        "sensitive_lookup_request": {
            "sources": ["sales", "goldmine"],
            "targets": ["SK네트웍스", "SKAX"],
            "items": ["현재 영업 단계", "이전 사업 이력"],
            "reason": "신규 제안과 재접촉을 구분하기 위해 필요",
        },
    },
)


def preflight_response_contract_examples() -> tuple[dict[str, object], ...]:
    """Return isolated copies of the complete examples embedded in Demian's prompt."""

    return copy.deepcopy(_PREFLIGHT_RESPONSE_CONTRACT_EXAMPLES)


def render_meeting_preflight_result(result: MeetingPreflightResult) -> str:
    """Render one validated result without adding facts or model-generated prose."""

    if result.status == "blocked_materials":
        missing = "\n".join(f"· {item}" for item in result.missing_materials)
        blocking = "\n".join(
            f"· {_PREMISE_LABELS[item.kind]}: {item.requirement} — {item.blocking_reason}"
            for item in result.premise_checks
            if item.status in {"missing", "unavailable"}
        )
        return (
            "자료가 부족해 회의를 시작하지 않았습니다.\n\n"
            f"확인이 필요한 전제:\n{blocking}\n\n"
            f"필요한 자료:\n{missing}\n\n"
            f"{_render_preflight_sources(result.sources)}"
        )
    if result.status == "ready":
        participants = "\n".join(f"· {participant}" for participant in result.participants)
        premises = "\n".join(
            f"· {_PREMISE_LABELS[item.kind]}: {_PREMISE_STATUS_LABELS[item.status]}"
            for item in result.premise_checks
        )
        return (
            "자료 확인을 마쳤습니다.\n\n"
            f"{_render_preflight_sources(result.sources)}\n\n"
            f"전제 확인:\n{premises}\n\n"
            f"참석자:\n{participants}\n\n"
            "참석자를 확정하고 회의 준비를 진행합니다."
        )
    if result.status == "awaiting_sensitive_approval":
        request = result.sensitive_lookup_request
        if request is None:
            raise ValueError("validated sensitive approval result lost its lookup request")
        sources = "\n".join(f"· {source}" for source in request.sources)
        targets = "\n".join(f"· {target}" for target in request.targets)
        items = "\n".join(f"· {item}" for item in request.items)
        return (
            "승인이 필요한 자료 조회가 있습니다.\n\n"
            f"조회 위치:\n{sources}\n\n"
            f"조회 대상:\n{targets}\n\n"
            f"확인 항목:\n{items}\n\n"
            f"필요한 이유:\n{request.reason}\n\n"
            "승인 전에는 조회하거나 회의를 시작하지 않습니다."
        )
    raise ValueError("unsupported validated preflight status")


def _render_preflight_sources(sources: tuple[PreflightSourceResult, ...]) -> str:
    lines = ["확인한 곳:"]
    for source in sources:
        label = _SOURCE_LABELS[source.source]
        status = _SOURCE_STATUS_LABELS[source.status]
        details = list(source.facts)
        if source.inspection_evidence:
            details.append(f"조회 근거: {'; '.join(source.inspection_evidence)}")
        suffix = f" — {'; '.join(details)}" if details else ""
        lines.append(f"· {label}: {status}{suffix}")
    return "\n".join(lines)


@dataclass(frozen=True)
class MeetingCandidateResult:
    reason_to_speak: bool
    reason_class: str
    statement: str


def parse_meeting_candidate_result(response_text: object) -> MeetingCandidateResult:
    """Parse one exact JSON-only private candidate response."""

    if not isinstance(response_text, str) or not response_text.strip():
        raise CandidateParseError("empty_response", "meeting candidate response is empty")
    text = response_text.strip()
    if len(text) > _MAX_RESPONSE_LENGTH:
        raise CandidateParseError("schema_violation", "meeting candidate response is too large")
    if text.startswith("```"):
        try:
            text = _unwrap_json_fence(text)
        except PreflightParseError as exc:
            raise CandidateParseError(exc.code, str(exc)) from exc
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CandidateParseError("malformed_json", "meeting candidate response is not valid JSON") from exc
    if not isinstance(value, Mapping) or frozenset(value) != _CANDIDATE_KEYS:
        raise CandidateParseError("schema_violation", "meeting candidate result keys do not match schema")
    version = value["schema_version"]
    if isinstance(version, bool) or version != 1:
        raise CandidateParseError("schema_violation", "unsupported meeting candidate schema version")
    reason_to_speak = value["reason_to_speak"]
    if not isinstance(reason_to_speak, bool):
        raise CandidateParseError("schema_violation", "reason_to_speak must be a boolean")
    reason_class = value["reason_class"]
    if reason_class not in _CANDIDATE_REASON_CLASSES:
        raise CandidateParseError("schema_violation", "meeting candidate reason_class is invalid")
    statement_value = value["statement"]
    if not isinstance(statement_value, str):
        raise CandidateParseError("schema_violation", "meeting candidate statement must be text")
    statement = " ".join(statement_value.split())
    if reason_to_speak:
        if reason_class == "none" or not statement:
            raise CandidateParseError(
                "schema_violation",
                "speaking candidate requires a classified non-empty statement",
            )
    elif reason_class != "none" or statement:
        raise CandidateParseError(
            "schema_violation",
            "silent candidate requires reason_class none and an empty statement",
        )
    return MeetingCandidateResult(reason_to_speak, str(reason_class), statement)


def parse_meeting_preflight_result(response_text: object) -> MeetingPreflightResult:
    """Parse one exact JSON result; prose, missing fields, and mock web claims fail closed."""

    if not isinstance(response_text, str) or not response_text.strip():
        raise PreflightParseError("empty_response", "meeting preflight response is empty")
    text = response_text.strip()
    if len(text) > _MAX_RESPONSE_LENGTH:
        raise PreflightParseError(
            "schema_violation:response_too_large",
            "meeting preflight response is too large",
        )
    if text.startswith("```"):
        text = _unwrap_json_fence(text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PreflightParseError("malformed_json", "meeting preflight response is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise PreflightParseError(
            "schema_violation:result_type",
            "meeting preflight result must be a JSON object",
        )
    version = value.get("schema_version")
    if isinstance(version, bool) or version != PREFLIGHT_SCHEMA_VERSION:
        raise PreflightParseError(
            "schema_violation:unsupported_version",
            "unsupported meeting preflight schema version",
        )
    if frozenset(value) != _RESULT_KEYS:
        raise PreflightParseError(
            "schema_violation:result_keys",
            "meeting preflight result keys do not match schema",
        )
    status = value["status"]
    if status not in {"ready", "blocked_materials", "awaiting_sensitive_approval"}:
        raise PreflightParseError(
            "schema_violation:status",
            "meeting preflight status is invalid",
        )
    agenda = _required_text(value["agenda"], "agenda")
    sources = _parse_sources(value["sources"])
    premise_checks = _parse_premise_checks(value["premise_checks"])
    missing_materials = _text_list(value["missing_materials"], "missing_materials")
    request_value = value["request_to_sinclair"]
    if request_value is None:
        request_to_sinclair = None
    else:
        request_to_sinclair = _text_list(request_value, "request_to_sinclair")
    participants = _text_list(value["participants"], "participants")
    sensitive_lookup_request = _parse_sensitive_lookup_request(
        value["sensitive_lookup_request"]
    )

    evidence_pool = {agenda}
    for source in sources:
        evidence_pool.update(source.facts)
        evidence_pool.update(source.inspection_evidence)
    for premise in premise_checks:
        if any(item not in evidence_pool for item in premise.evidence):
            raise PreflightParseError(
                f"schema_violation:premise_evidence:{premise.kind}",
                f"{premise.kind} premise evidence must copy agenda or validated source evidence",
            )

    internal_fact_count = sum(
        len(source.facts) for source in sources[:-1] if source.status == "available"
    )
    blocking_premises = tuple(
        premise
        for premise in premise_checks
        if premise.status in {"missing", "unavailable"}
    )
    premise_materials = _ordered_unique(
        item for premise in blocking_premises for item in premise.required_materials
    )
    if status == "ready":
        if missing_materials or request_to_sinclair is not None:
            raise PreflightParseError(
                "schema_violation:ready_materials",
                "ready preflight cannot contain missing materials or a material request",
            )
        if internal_fact_count == 0:
            raise PreflightParseError(
                "schema_violation:ready_facts",
                "ready preflight requires at least one available internal fact",
            )
        if sensitive_lookup_request is not None:
            raise PreflightParseError(
                "schema_violation:ready_sensitive_lookup",
                "ready preflight cannot contain a sensitive lookup request",
            )
        if not 3 <= len(participants) <= 4:
            raise PreflightParseError(
                "schema_violation:ready_participants",
                "ready preflight requires three or four unique participants",
            )
        if blocking_premises:
            raise PreflightParseError(
                "schema_violation:ready_blocking_premise",
                "ready preflight cannot contain a missing or unavailable premise",
            )
    elif status == "blocked_materials":
        if not blocking_premises:
            raise PreflightParseError(
                "schema_violation:blocked_without_premise",
                "blocked preflight requires a missing or unavailable premise",
            )
        if not missing_materials or request_to_sinclair is None:
            raise PreflightParseError(
                "schema_violation:blocked_materials",
                "blocked preflight requires missing materials and an ordered request array",
            )
        if participants or sensitive_lookup_request is not None:
            raise PreflightParseError(
                "schema_violation:blocked_execution",
                "blocked preflight cannot contain participants or a sensitive lookup request",
            )
        if missing_materials != premise_materials:
            raise PreflightParseError(
                "schema_violation:premise_materials_mismatch",
                "missing_materials must equal the ordered blocking-premise material union",
            )
        if request_to_sinclair != missing_materials:
            raise PreflightParseError(
                "schema_violation:request_order_mismatch",
                "request_to_sinclair must equal missing_materials as an ordered array",
            )
    else:
        if missing_materials or request_to_sinclair is not None or participants:
            raise PreflightParseError(
                "schema_violation:sensitive_execution",
                "sensitive approval preflight cannot contain material requests or participants",
            )
        if sensitive_lookup_request is None:
            raise PreflightParseError(
                "schema_violation:sensitive_lookup_missing",
                "sensitive approval preflight requires one lookup request",
            )
        if not blocking_premises:
            raise PreflightParseError(
                "schema_violation:sensitive_without_premise",
                "sensitive approval preflight requires a blocked premise",
            )
        if sensitive_lookup_request.items != premise_materials:
            raise PreflightParseError(
                "schema_violation:sensitive_items_mismatch",
                "sensitive lookup items must equal the ordered blocking-premise material union",
            )

    return MeetingPreflightResult(
        schema_version=PREFLIGHT_SCHEMA_VERSION,
        status=str(status),
        agenda=agenda,
        sources=sources,
        premise_checks=premise_checks,
        missing_materials=missing_materials,
        request_to_sinclair=request_to_sinclair,
        participants=participants,
        sensitive_lookup_request=sensitive_lookup_request,
    )


def preflight_response_schema_instruction() -> str:
    """Return the exact JSON-only response contract embedded in Demian's prompt."""

    ready, blocked, sensitive = preflight_response_contract_examples()
    return f"""응답은 설명문 없이 schema v3 JSON 하나만 반환하세요. JSON code fence는 허용됩니다.
최상위 key는 schema_version, status, agenda, sources, premise_checks, missing_materials, request_to_sinclair, participants, sensitive_lookup_request만 허용됩니다.
sources는 notion, kanban, shared_files, public_search 순서입니다. 내부 source는 inspection_evidence가 반드시 있어야 합니다.
available은 facts와 inspection_evidence가 모두 non-empty여야 합니다. missing 또는 unavailable은 facts=[]이고 inspection_evidence가 non-empty여야 합니다.
public_search는 단계 2c 범위 밖이므로 status=skipped_stage_2c, facts=[], inspection_evidence=[]로 기록하세요.
premise_checks는 contract, legal, prerequisite, schedule, resource, operational 순서로 종류별 정확히 1개씩 반환하세요.
satisfied와 not_applicable은 evidence가 있어야 하고 blocking_reason=null, required_materials=[]여야 합니다.
missing과 unavailable은 blocking_reason과 required_materials가 있어야 하며 ready를 반환할 수 없습니다.
premise evidence는 agenda 또는 sources의 facts/inspection_evidence 문자열을 그대로 복사하세요. 새로운 사실을 만들지 마세요.
blocked_materials에서는 blocking premise의 required_materials ordered union, missing_materials, request_to_sinclair 세 배열이 정확히 같아야 합니다.
ready는 available 내부 facts가 하나 이상이고 blocking premise가 없으며 participants가 canonical Staff 3~4명이어야 합니다. Demian은 participants에 넣지 마세요.
awaiting_sensitive_approval은 participants=[], missing_materials=[], request_to_sinclair=null이고 sensitive_lookup_request.items가 blocking premise의 required_materials ordered union과 정확히 같아야 합니다.
Sales·Goldmine은 승인 전에 조회하지 말고 sources, targets, items, reason만 반환하세요. 자격 증명, 내부 오류 원문, 추정 사실은 반환하지 마세요.

ready 완성 예시:
{json.dumps(ready, ensure_ascii=False, indent=2)}

blocked_materials 완성 예시:
{json.dumps(blocked, ensure_ascii=False, indent=2)}

awaiting_sensitive_approval 완성 예시:
{json.dumps(sensitive, ensure_ascii=False, indent=2)}"""


def _unwrap_json_fence(text: str) -> str:
    lines = text.splitlines()
    if len(lines) < 3 or lines[0] not in {"```", "```json"} or lines[-1].strip() != "```":
        raise PreflightParseError("malformed_json", "meeting preflight JSON fence is malformed")
    body = "\n".join(lines[1:-1]).strip()
    if not body or "```" in body:
        raise PreflightParseError("malformed_json", "meeting preflight JSON fence is malformed")
    return body


def _parse_sources(value: Any) -> tuple[PreflightSourceResult, ...]:
    if not isinstance(value, list) or len(value) != len(PREFLIGHT_SOURCE_ORDER):
        raise PreflightParseError(
            "schema_violation:sources_count",
            "preflight sources must contain four ordered entries",
        )
    parsed: list[PreflightSourceResult] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or frozenset(item) != _SOURCE_KEYS:
            raise PreflightParseError(
                f"schema_violation:source_keys:{index}",
                "preflight source entry does not match schema",
            )
        source = item["source"]
        if source != PREFLIGHT_SOURCE_ORDER[index]:
            raise PreflightParseError(
                f"schema_violation:source_order:{index}",
                "preflight source order is invalid",
            )
        status = item["status"]
        facts = _text_list(item["facts"], f"{source}.facts")
        inspection_evidence = _text_list(
            item["inspection_evidence"],
            f"{source}.inspection_evidence",
        )
        if source == "public_search":
            if status != _PUBLIC_SEARCH_STATUS or facts or inspection_evidence:
                raise PreflightParseError(
                    "schema_violation:public_search_contract",
                    "public search is excluded from stage 2c and must be recorded as skipped",
                )
        else:
            if status not in _INTERNAL_SOURCE_STATUSES:
                raise PreflightParseError(
                    f"schema_violation:source_status:{source}",
                    f"{source} status is invalid",
                )
            if not inspection_evidence:
                raise PreflightParseError(
                    f"schema_violation:source_inspection:{source}",
                    f"{source} requires inspection evidence",
                )
            if status == "available" and not facts:
                raise PreflightParseError(
                    f"schema_violation:source_available_facts:{source}",
                    f"{source} available status requires facts",
                )
            if status != "available" and facts:
                raise PreflightParseError(
                    f"schema_violation:source_facts:{source}",
                    f"{source} non-available status cannot claim facts",
                )
        parsed.append(
            PreflightSourceResult(
                str(source),
                str(status),
                facts,
                inspection_evidence,
            )
        )
    return tuple(parsed)


def _parse_premise_checks(value: Any) -> tuple[PremiseCheckResult, ...]:
    if not isinstance(value, list) or len(value) != len(PREMISE_KIND_ORDER):
        raise PreflightParseError(
            "schema_violation:premise_count",
            "premise_checks must contain six ordered entries",
        )
    parsed: list[PremiseCheckResult] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or frozenset(item) != _PREMISE_KEYS:
            raise PreflightParseError(
                f"schema_violation:premise_keys:{index}",
                "premise entry does not match schema",
            )
        kind = item["kind"]
        if kind != PREMISE_KIND_ORDER[index]:
            raise PreflightParseError(
                f"schema_violation:premise_order:{index}",
                "premise kind order is invalid",
            )
        requirement = _required_text(item["requirement"], f"{kind}.requirement")
        status = item["status"]
        if status not in _PREMISE_STATUSES:
            raise PreflightParseError(
                f"schema_violation:premise_status:{kind}",
                f"{kind} premise status is invalid",
            )
        evidence = _text_list(item["evidence"], f"{kind}.evidence")
        blocking_value = item["blocking_reason"]
        blocking_reason = (
            None
            if blocking_value is None
            else _required_text(blocking_value, f"{kind}.blocking_reason")
        )
        required_materials = _text_list(
            item["required_materials"],
            f"{kind}.required_materials",
        )
        if status in {"satisfied", "not_applicable"}:
            if not evidence or blocking_reason is not None or required_materials:
                raise PreflightParseError(
                    f"schema_violation:premise_complete:{kind}",
                    f"{kind} completed premise requires evidence and no blocking fields",
                )
        elif blocking_reason is None or not required_materials:
            raise PreflightParseError(
                f"schema_violation:premise_blocking:{kind}",
                f"{kind} blocking premise requires a reason and required materials",
            )
        parsed.append(
            PremiseCheckResult(
                kind=str(kind),
                requirement=requirement,
                status=str(status),
                evidence=evidence,
                blocking_reason=blocking_reason,
                required_materials=required_materials,
            )
        )
    return tuple(parsed)


def _parse_sensitive_lookup_request(value: Any) -> SensitiveLookupRequest | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or frozenset(value) != _SENSITIVE_LOOKUP_KEYS:
        raise PreflightParseError(
            "schema_violation:sensitive_lookup_keys",
            "sensitive lookup request does not match schema",
        )
    sources = _text_list(value["sources"], "sensitive_lookup_request.sources")
    if not sources or any(source not in _SENSITIVE_LOOKUP_SOURCES for source in sources):
        raise PreflightParseError(
            "schema_violation:sensitive_lookup_sources",
            "sensitive lookup sources must contain sales or goldmine",
        )
    targets = _text_list(value["targets"], "sensitive_lookup_request.targets")
    items = _text_list(value["items"], "sensitive_lookup_request.items")
    if not targets or not items:
        raise PreflightParseError(
            "schema_violation:sensitive_lookup_items",
            "sensitive lookup targets and items must not be empty",
        )
    return SensitiveLookupRequest(
        sources=sources,
        targets=targets,
        items=items,
        reason=_required_text(value["reason"], "sensitive_lookup_request.reason"),
    )


def _text_list(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise PreflightParseError(
            f"schema_violation:list_type:{field_name}",
            f"{field_name} must be an array",
        )
    items = tuple(_required_text(item, field_name) for item in value)
    if len(set(items)) != len(items):
        raise PreflightParseError(
            f"schema_violation:list_duplicates:{field_name}",
            f"{field_name} must not contain duplicates",
        )
    return items


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PreflightParseError(
            f"schema_violation:text:{field_name}",
            f"{field_name} must be non-empty text",
        )
    return " ".join(value.split())


def _ordered_unique(values: Any) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))
