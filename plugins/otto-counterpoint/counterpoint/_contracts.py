# -*- coding: utf-8 -*-
"""Typed artifacts exchanged by the counterpoint workflow."""
from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ._policy import RouteIdentity, RoutePolicyError

_HASH = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_STATUSES = frozenset({"produced", "needs_context", "blocked"})
_CRITIQUE_STATUSES = frozenset(
    {"no_material_finding", "changes_requested", "blocked", "insufficient_evidence"}
)
_GATE_STATUSES = frozenset({"pass", "fail", "unknown"})
_VERDICTS = frozenset({"accept_local", "revise", "block", "human_review"})
_OUTCOMES = frozenset({"sustained", "refuted", "unresolved"})


def _non_empty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _records(value: Sequence[Mapping[str, Any]], field: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list")
    output: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"{field} must contain objects")
        output.append(copy.deepcopy(dict(item)))
    return tuple(output)


def _strings(value: Sequence[str], field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list")
    output = tuple(_non_empty(item, field) for item in value)
    return output


def _hash(value: str, field: str = "content_sha256") -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True)
class Artifact:
    """Immutable metadata for the generated artifact; content stays external."""

    run_id: str
    task_id: str
    artifact_id: str
    version: int
    status: str
    content_ref: str
    content_sha256: str
    claims: Sequence[Mapping[str, Any]] = ()
    evidence_refs: Sequence[str] = ()
    assumptions: Sequence[str] = ()
    uncertainties: Sequence[str] = ()
    tests: Sequence[Mapping[str, Any]] = ()
    schema_version: str = "artifact.v1"

    def __post_init__(self) -> None:
        for field in ("run_id", "task_id", "artifact_id", "content_ref", "schema_version"):
            _non_empty(getattr(self, field), field)
        if self.schema_version != "artifact.v1":
            raise ValueError("unsupported artifact schema")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ValueError("version must be an integer >= 1")
        if self.status not in _ARTIFACT_STATUSES:
            raise ValueError(f"unsupported artifact status: {self.status!r}")
        _hash(self.content_sha256)
        object.__setattr__(self, "claims", _records(self.claims, "claims"))
        object.__setattr__(self, "tests", _records(self.tests, "tests"))
        object.__setattr__(self, "evidence_refs", _strings(self.evidence_refs, "evidence_refs"))
        object.__setattr__(self, "assumptions", _strings(self.assumptions, "assumptions"))
        object.__setattr__(self, "uncertainties", _strings(self.uncertainties, "uncertainties"))
        for test in self.tests:
            result = test.get("result")
            if result is not None and result not in {"pass", "fail", "not_run"}:
                raise ValueError("tests.result must be pass, fail or not_run")

    @classmethod
    def from_content(
        cls,
        *,
        run_id: str,
        task_id: str,
        artifact_id: str,
        version: int,
        content: str | bytes,
        content_ref: str,
        status: str = "produced",
        claims: Sequence[Mapping[str, Any]] = (),
        evidence_refs: Sequence[str] = (),
        assumptions: Sequence[str] = (),
        uncertainties: Sequence[str] = (),
        tests: Sequence[Mapping[str, Any]] = (),
    ) -> "Artifact":
        raw = content.encode("utf-8") if isinstance(content, str) else content
        if not isinstance(raw, bytes):
            raise ValueError("content must be text or bytes")
        return cls(
            run_id=run_id,
            task_id=task_id,
            artifact_id=artifact_id,
            version=version,
            status=status,
            content_ref=content_ref,
            content_sha256=hashlib.sha256(raw).hexdigest(),
            claims=tuple(dict(item) for item in claims),
            evidence_refs=tuple(evidence_refs),
            assumptions=tuple(assumptions),
            uncertainties=tuple(uncertainties),
            tests=tuple(dict(item) for item in tests),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "artifact_id": self.artifact_id,
            "version": self.version,
            "status": self.status,
            "content_ref": self.content_ref,
            "content_sha256": self.content_sha256,
            "claims": copy.deepcopy(list(self.claims)),
            "evidence_refs": list(self.evidence_refs),
            "assumptions": list(self.assumptions),
            "uncertainties": list(self.uncertainties),
            "tests": copy.deepcopy(list(self.tests)),
        }


@dataclass(frozen=True)
class GateResult:
    """One deterministic gate result; unknown is never treated as pass."""

    gate: str
    status: str
    evidence_id: str

    def __post_init__(self) -> None:
        _non_empty(self.gate, "gate")
        if self.status not in _GATE_STATUSES:
            raise ValueError(f"unsupported gate status: {self.status!r}")
        _non_empty(self.evidence_id, "evidence_id")

    def to_dict(self) -> dict[str, str]:
        return {"gate": self.gate, "status": self.status, "evidence_id": self.evidence_id}


@dataclass(frozen=True)
class Critique:
    """Adversarial review of one immutable artifact version."""

    critique_id: str
    run_id: str
    artifact_id: str
    artifact_sha256: str
    status: str
    findings: Sequence[Mapping[str, Any]]
    coverage: Mapping[str, Any]
    evidence_refs: Sequence[str] = ()
    route: RouteIdentity | None = None
    schema_version: str = "critique.v1"

    def __post_init__(self) -> None:
        for field in ("critique_id", "run_id", "artifact_id", "schema_version"):
            _non_empty(getattr(self, field), field)
        if self.schema_version != "critique.v1":
            raise ValueError("unsupported critique schema")
        _hash(self.artifact_sha256, "artifact_sha256")
        if self.status not in _CRITIQUE_STATUSES:
            raise ValueError(f"unsupported critique status: {self.status!r}")
        object.__setattr__(self, "findings", _records(self.findings, "findings"))
        if not isinstance(self.coverage, Mapping):
            raise ValueError("coverage must be an object")
        object.__setattr__(self, "coverage", copy.deepcopy(dict(self.coverage)))
        object.__setattr__(self, "evidence_refs", _strings(self.evidence_refs, "evidence_refs"))
        if self.route is not None and not isinstance(self.route, RouteIdentity):
            raise RoutePolicyError("critique route must be a RouteIdentity")
        if self.status == "no_material_finding" and self.findings:
            raise ValueError("no_material_finding cannot contain findings")
        if self.status == "changes_requested" and not self.findings:
            raise ValueError("changes_requested requires findings")

    def validate_for(self, artifact: Artifact) -> None:
        if self.run_id != artifact.run_id or self.artifact_id != artifact.artifact_id:
            raise ValueError("critique does not identify the artifact run")
        if self.artifact_sha256 != artifact.content_sha256:
            raise ValueError("critique hash does not match artifact")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "critique_id": self.critique_id,
            "run_id": self.run_id,
            "artifact_id": self.artifact_id,
            "artifact_sha256": self.artifact_sha256,
            "status": self.status,
            "findings": copy.deepcopy(list(self.findings)),
            "coverage": copy.deepcopy(self.coverage),
            "evidence_refs": list(self.evidence_refs),
            "route": _route_dict(self.route),
        }


def _route_dict(route: RouteIdentity | None) -> dict[str, Any] | None:
    if route is None:
        return None
    return {
        "vendor": route.vendor,
        "family": route.family,
        "provider": route.provider,
        "model": route.model,
        "reasoning_effort": route.reasoning_effort,
        "route_key": route.route_key,
        "authenticated": route.authenticated,
        "accessible": route.accessible,
        "smoke_tested": route.smoke_tested,
        "relative_load": route.relative_load,
    }


@dataclass(frozen=True)
class JudgeVerdict:
    """Structured adjudicator output; it has no authority over hard gates."""

    verdict: str
    dispositions: Sequence[Mapping[str, Any]] = ()
    evidence_refs: Sequence[str] = ()

    def __post_init__(self) -> None:
        if self.verdict not in _VERDICTS:
            raise ValueError(f"unsupported judge verdict: {self.verdict!r}")
        object.__setattr__(self, "dispositions", _records(self.dispositions, "dispositions"))
        object.__setattr__(self, "evidence_refs", _strings(self.evidence_refs, "evidence_refs"))
        for item in self.dispositions:
            if item.get("outcome") not in _OUTCOMES:
                raise ValueError("disposition outcome is invalid")


@dataclass(frozen=True)
class Decision:
    """Final local decision after deterministic gates and optional critique."""

    run_id: str
    artifact_sha256: str
    critique_id: str | None
    gate_results: Sequence[GateResult]
    verdict: str
    dispositions: Sequence[Mapping[str, Any]]
    route_record: Mapping[str, Any]
    next_actor: str
    evidence_refs: Sequence[str] = ()
    schema_version: str = "decision.v1"

    def __post_init__(self) -> None:
        _non_empty(self.run_id, "run_id")
        _hash(self.artifact_sha256, "artifact_sha256")
        if self.critique_id is not None:
            _non_empty(self.critique_id, "critique_id")
        if self.verdict not in _VERDICTS:
            raise ValueError(f"unsupported decision verdict: {self.verdict!r}")
        if self.schema_version != "decision.v1":
            raise ValueError("unsupported decision schema")
        if not isinstance(self.gate_results, (list, tuple)):
            raise ValueError("gate_results must be a list")
        object.__setattr__(self, "gate_results", tuple(self.gate_results))
        if any(not isinstance(item, GateResult) for item in self.gate_results):
            raise ValueError("gate_results must contain GateResult values")
        object.__setattr__(self, "dispositions", _records(self.dispositions, "dispositions"))
        if not isinstance(self.route_record, Mapping):
            raise ValueError("route_record must be an object")
        object.__setattr__(self, "route_record", copy.deepcopy(dict(self.route_record)))
        _non_empty(self.next_actor, "next_actor")
        object.__setattr__(self, "evidence_refs", _strings(self.evidence_refs, "evidence_refs"))
        if self.verdict == "accept_local":
            if any(gate.status != "pass" for gate in self.gate_results):
                raise ValueError("accept_local requires every gate to pass")
            if any(item.get("outcome") == "unresolved" for item in self.dispositions):
                raise ValueError("accept_local cannot contain unresolved findings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "artifact_sha256": self.artifact_sha256,
            "critique_ref": self.critique_id,
            "gate_results": [item.to_dict() for item in self.gate_results],
            "verdict": self.verdict,
            "dispositions": copy.deepcopy(list(self.dispositions)),
            "route_record": copy.deepcopy(self.route_record),
            "evidence_refs": list(self.evidence_refs),
            "next_actor": self.next_actor,
        }


__all__ = [
    "Artifact",
    "Critique",
    "Decision",
    "GateResult",
    "JudgeVerdict",
    "_route_dict",
]
