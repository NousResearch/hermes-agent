# -*- coding: utf-8 -*-
"""Bounded generator -> validator -> counterpoint -> adjudication workflow."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .ledger import Ledger, RunLifecycle

from ._contracts import (
    Artifact,
    Critique,
    Decision,
    GateResult,
    JudgeVerdict,
    _route_dict,
)
from ._policy import (
    RouteIdentity,
    RoutePolicyError,
    NoIndependentRoute,
    decide_counterpoint,
    eligible_routes,
)


class WorkflowContractError(ValueError):
    """A callback returned an artifact, gate, critique or verdict out of contract."""


@dataclass(frozen=True)
class WorkflowSpec:
    """Immutable scope and routing contract for one bounded workflow run."""

    project_id: str
    task_id: str
    run_id: str
    agent_id: str
    risk: str
    complexity: str
    generator_route: RouteIdentity
    counterpoint_route: RouteIdentity | None = None
    adjudicator_route: RouteIdentity | None = None
    criteria: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    external_effect: bool = False
    evidence_conflict: bool = False
    uncertainty: bool = False
    sampling_selected: bool = False
    max_corrections: int = 1

    def __post_init__(self) -> None:
        for field in ("project_id", "task_id", "run_id", "agent_id", "risk", "complexity"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise WorkflowContractError(f"{field} must be a non-empty string")
        if not isinstance(self.generator_route, RouteIdentity):
            raise WorkflowContractError("generator_route must be a RouteIdentity")
        if self.counterpoint_route is not None and not isinstance(self.counterpoint_route, RouteIdentity):
            raise WorkflowContractError("counterpoint_route must be a RouteIdentity")
        if self.adjudicator_route is not None and not isinstance(self.adjudicator_route, RouteIdentity):
            raise WorkflowContractError("adjudicator_route must be a RouteIdentity")
        if isinstance(self.max_corrections, bool) or not isinstance(self.max_corrections, int):
            raise WorkflowContractError("max_corrections must be an integer")
        if not 0 <= self.max_corrections <= 1:
            raise WorkflowContractError("max_corrections must be between zero and one")
        if not isinstance(self.criteria, (list, tuple)) or any(
            not isinstance(item, str) or not item.strip() for item in self.criteria
        ):
            raise WorkflowContractError("criteria must contain non-empty strings")
        if not isinstance(self.evidence_refs, (list, tuple)) or any(
            not isinstance(item, str) or not item.strip() for item in self.evidence_refs
        ):
            raise WorkflowContractError("evidence_refs must contain non-empty strings")
        decide_counterpoint(
            risk=self.risk,
            complexity=self.complexity,
            external_effect=self.external_effect,
            evidence_conflict=self.evidence_conflict,
            uncertainty=self.uncertainty,
            sampling_selected=self.sampling_selected,
        )


@dataclass(frozen=True)
class WorkflowResult:
    """Sanitized result that can be persisted or handed to a supervisor."""

    status: str
    decision: Decision
    artifact: Artifact | None
    critique: Critique | None
    correction_count: int
    blocked_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "decision": self.decision.to_dict(),
            "artifact": None if self.artifact is None else self.artifact.to_dict(),
            "critique": None if self.critique is None else self.critique.to_dict(),
            "correction_count": self.correction_count,
            "blocked_reason": self.blocked_reason,
        }


Generator = Callable[[Mapping[str, Any], Artifact | None, Critique | None], Artifact]
Validator = Callable[[Artifact], Sequence[GateResult]]
Critic = Callable[[Artifact, Sequence[str]], Critique]
Adjudicator = Callable[[Artifact, Critique, Sequence[GateResult]], JudgeVerdict]


class CounterpointWorkflow:
    """Run the bounded flow and persist only sanitized lifecycle metadata."""

    def __init__(
        self,
        *,
        ledger: Ledger,
        generator: Generator,
        validator: Validator,
        critic: Critic,
        adjudicator: Adjudicator | None = None,
    ) -> None:
        self.ledger = ledger
        self.generator = generator
        self.validator = validator
        self.critic = critic
        self.adjudicator = adjudicator

    def run(self, spec: WorkflowSpec, *, context: Mapping[str, Any]) -> WorkflowResult:
        lifecycle = RunLifecycle(self.ledger)
        policy = decide_counterpoint(
            risk=spec.risk,
            complexity=spec.complexity,
            external_effect=spec.external_effect,
            evidence_conflict=spec.evidence_conflict,
            uncertainty=spec.uncertainty,
            sampling_selected=spec.sampling_selected,
        )
        route_record = self._route_record(spec)
        lifecycle.create(
            project_id=spec.project_id,
            linear_issue_id=spec.task_id,
            run_id=spec.run_id,
            agent_id=spec.agent_id,
            evidence_refs=spec.evidence_refs,
            payload={
                "workflow": "counterpoint.v1",
                "policy_reason": policy.reason,
                "risk": spec.risk,
                "complexity": spec.complexity,
                "route_record": route_record,
            },
        )
        lifecycle.start(spec.run_id)

        artifact: Artifact | None = None
        critique: Critique | None = None
        correction_count = 0
        try:
            artifact = self._generate(spec, context, None, None)
            gates = self._validate(artifact)
            if not self._gates_pass(gates):
                return self._blocked(
                    lifecycle,
                    spec,
                    artifact,
                    critique,
                    gates,
                    correction_count,
                    route_record,
                    reason="deterministic_gate_failed",
                    verdict="block",
                    next_actor="generator",
                )

            if not policy.required:
                decision = self._decision(
                    spec,
                    artifact,
                    critique,
                    gates,
                    route_record,
                    verdict="accept_local",
                    next_actor="otto",
                )
                self._finish(
                    lifecycle,
                    spec,
                    decision,
                    correction_count,
                    artifact=artifact,
                    critique=critique,
                    gates=gates,
                )
                return WorkflowResult("succeeded", decision, artifact, critique, correction_count)

            self._require_counterpoint_route(spec)
            critique = self._critic(artifact, spec)
            if critique.status == "no_material_finding":
                decision = self._decision(
                    spec,
                    artifact,
                    critique,
                    gates,
                    route_record,
                    verdict="accept_local",
                    next_actor="otto",
                )
                self._finish(
                    lifecycle,
                    spec,
                    decision,
                    correction_count,
                    artifact=artifact,
                    critique=critique,
                    gates=gates,
                )
                return WorkflowResult("succeeded", decision, artifact, critique, correction_count)

            if critique.status == "changes_requested" and correction_count < spec.max_corrections:
                correction_count += 1
                artifact = self._generate(spec, context, artifact, critique)
                gates = self._validate(artifact)
                if self._gates_pass(gates):
                    critique = self._critic(artifact, spec)
                    if critique.status == "no_material_finding":
                        decision = self._decision(
                            spec,
                            artifact,
                            critique,
                            gates,
                            route_record,
                            verdict="accept_local",
                            next_actor="otto",
                        )
                        self._finish(
                            lifecycle,
                            spec,
                            decision,
                            correction_count,
                            artifact=artifact,
                            critique=critique,
                            gates=gates,
                        )
                        return WorkflowResult("succeeded", decision, artifact, critique, correction_count)

            return self._adjudicate_or_block(
                lifecycle,
                spec,
                artifact,
                critique,
                gates,
                correction_count,
                route_record,
            )
        except (NoIndependentRoute, RoutePolicyError):
            return self._blocked(
                lifecycle,
                spec,
                artifact,
                critique,
                locals().get("gates", ()),
                correction_count,
                route_record,
                reason="human_review_required",
                verdict="human_review",
                next_actor="human",
                dispositions=self._unresolved(critique),
            )
        except (WorkflowContractError, ValueError):
            return self._blocked(
                lifecycle,
                spec,
                artifact,
                critique,
                locals().get("gates", ()),
                correction_count,
                route_record,
                reason="workflow_contract_error",
                verdict="block",
                next_actor="otto",
            )
        except Exception:  # health: allow BLE001 -- workflow callback failures are sanitized to a blocked result
            return self._blocked(
                lifecycle,
                spec,
                artifact,
                critique,
                locals().get("gates", ()),
                correction_count,
                route_record,
                reason="workflow_callback_failed",
                verdict="block",
                next_actor="otto",
            )

    def _generate(
        self,
        spec: WorkflowSpec,
        context: Mapping[str, Any],
        previous: Artifact | None,
        critique: Critique | None,
    ) -> Artifact:
        artifact = self.generator(context, previous, critique)
        if not isinstance(artifact, Artifact):
            raise WorkflowContractError("generator must return Artifact")
        if (artifact.run_id, artifact.task_id) != (spec.run_id, spec.task_id):
            raise WorkflowContractError("artifact identity does not match workflow")
        return artifact

    def _validate(self, artifact: Artifact) -> tuple[GateResult, ...]:
        results = self.validator(artifact)
        if not isinstance(results, (list, tuple)) or not results:
            raise WorkflowContractError("validator must return at least one GateResult")
        normalized = tuple(results)
        if any(not isinstance(item, GateResult) for item in normalized):
            raise WorkflowContractError("validator returned an invalid gate")
        return normalized

    def _critic(self, artifact: Artifact, spec: WorkflowSpec) -> Critique:
        critique = self.critic(artifact, tuple(spec.criteria))
        if not isinstance(critique, Critique):
            raise WorkflowContractError("critic must return Critique")
        critique.validate_for(artifact)
        if critique.route != spec.counterpoint_route:
            raise WorkflowContractError("critique route does not match selected counterpoint route")
        return critique

    def _require_counterpoint_route(self, spec: WorkflowSpec) -> None:
        if spec.counterpoint_route is None:
            raise RoutePolicyError("counterpoint route is required by policy")
        eligible_routes(
            [spec.counterpoint_route],
            role="counterpoint",
            generator_route=spec.generator_route,
            require_independence=True,
        )

    def _adjudicate_or_block(
        self,
        lifecycle: RunLifecycle,
        spec: WorkflowSpec,
        artifact: Artifact,
        critique: Critique,
        gates: Sequence[GateResult],
        correction_count: int,
        route_record: dict[str, Any],
    ) -> WorkflowResult:
        if critique.status not in {"blocked", "insufficient_evidence"} and self.adjudicator is not None and spec.adjudicator_route is not None:
            try:
                eligible_routes(
                    [spec.adjudicator_route],
                    role="adjudicator",
                    generator_route=spec.generator_route,
                    counterpoint_route=spec.counterpoint_route,
                    require_independence=True,
                )
                judge = self.adjudicator(artifact, critique, tuple(gates))
                if not isinstance(judge, JudgeVerdict):
                    raise WorkflowContractError("adjudicator must return JudgeVerdict")
                if (
                    judge.verdict == "accept_local"
                    and self._gates_pass(gates)
                    and self._judge_evidence_is_grounded(judge, critique, gates, spec)
                ):
                    decision = self._decision(
                        spec,
                        artifact,
                        critique,
                        gates,
                        route_record,
                        verdict="accept_local",
                        next_actor="otto",
                        dispositions=judge.dispositions,
                        evidence_refs=judge.evidence_refs,
                    )
                    self._finish(
                        lifecycle,
                        spec,
                        decision,
                        correction_count,
                        artifact=artifact,
                        critique=critique,
                        gates=gates,
                    )
                    return WorkflowResult("succeeded", decision, artifact, critique, correction_count)
            except (WorkflowContractError, RoutePolicyError, ValueError):
                pass

        return self._blocked(
            lifecycle,
            spec,
            artifact,
            critique,
            gates,
            correction_count,
            route_record,
            reason="human_review_required",
            verdict="human_review",
            next_actor="human",
            dispositions=self._unresolved(critique),
        )

    @staticmethod
    def _judge_evidence_is_grounded(
        judge: JudgeVerdict,
        critique: Critique,
        gates: Sequence[GateResult],
        spec: WorkflowSpec,
    ) -> bool:
        """Require adjudication evidence and a disposition for every finding."""
        known = set(spec.evidence_refs)
        known.update(item.evidence_id for item in gates)
        known.update(critique.evidence_refs)
        finding_ids: set[str] = set()
        for finding in critique.findings:
            finding_id = finding.get("finding_id")
            if isinstance(finding_id, str) and finding_id:
                finding_ids.add(finding_id)
            known.update(
                evidence_id
                for evidence_id in finding.get("evidence_ids", ())
                if isinstance(evidence_id, str)
            )
        disposition_ids = {
            str(item.get("finding_id"))
            for item in judge.dispositions
            if item.get("finding_id")
        }
        return bool(judge.evidence_refs) and set(judge.evidence_refs) <= known and finding_ids <= disposition_ids

    @staticmethod
    def _gates_pass(gates: Sequence[GateResult]) -> bool:
        return bool(gates) and all(item.status == "pass" for item in gates)

    @staticmethod
    def _unresolved(critique: Critique | None) -> tuple[dict[str, Any], ...]:
        if critique is None:
            return ()
        return tuple(
            {
                "finding_id": item.get("finding_id", "unknown-finding"),
                "outcome": "unresolved",
                "evidence_ids": list(item.get("evidence_ids", [])),
            }
            for item in critique.findings
        )

    @staticmethod
    def _route_record(spec: WorkflowSpec) -> dict[str, Any]:
        return {
            "generator": _route_dict(spec.generator_route),
            "counterpoint": _route_dict(spec.counterpoint_route),
            "adjudicator": _route_dict(spec.adjudicator_route),
        }

    @staticmethod
    def _decision(
        spec: WorkflowSpec,
        artifact: Artifact,
        critique: Critique | None,
        gates: Sequence[GateResult],
        route_record: Mapping[str, Any],
        *,
        verdict: str,
        next_actor: str,
        dispositions: Sequence[Mapping[str, Any]] = (),
        evidence_refs: Sequence[str] = (),
    ) -> Decision:
        return Decision(
            run_id=spec.run_id,
            artifact_sha256=artifact.content_sha256,
            critique_id=None if critique is None else critique.critique_id,
            gate_results=tuple(gates),
            verdict=verdict,
            dispositions=tuple(dict(item) for item in dispositions),
            route_record=copy.deepcopy(dict(route_record)),
            next_actor=next_actor,
            evidence_refs=tuple(evidence_refs),
        )

    def _finish(
        self,
        lifecycle: RunLifecycle,
        spec: WorkflowSpec,
        decision: Decision,
        correction_count: int,
        *,
        artifact: Artifact,
        critique: Critique | None,
        gates: Sequence[GateResult],
    ) -> None:
        lifecycle.finish(
            spec.run_id,
            "succeeded",
            evidence_refs=decision.evidence_refs,
            payload=self._audit_payload(
                decision=decision,
                artifact=artifact,
                critique=critique,
                gates=gates,
                correction_count=correction_count,
            ),
        )

    @staticmethod
    def _audit_payload(
        *,
        decision: Decision,
        artifact: Artifact,
        critique: Critique | None,
        gates: Sequence[GateResult],
        correction_count: int,
    ) -> dict[str, Any]:
        """Persist audit metadata without raw artifact or model text."""
        artifact_record = {
            "artifact_id": artifact.artifact_id,
            "version": artifact.version,
            "status": artifact.status,
            "content_ref": artifact.content_ref,
            "content_sha256": artifact.content_sha256,
            "claims_count": len(artifact.claims),
            "evidence_refs": list(artifact.evidence_refs),
            "assumptions_count": len(artifact.assumptions),
            "uncertainties_count": len(artifact.uncertainties),
            "tests": [
                {
                    key: test[key]
                    for key in ("test_id", "name", "result", "evidence_id")
                    if key in test
                }
                for test in artifact.tests
            ],
        }
        critique_record = None
        if critique is not None:
            critique_record = {
                "critique_id": critique.critique_id,
                "artifact_id": critique.artifact_id,
                "artifact_sha256": critique.artifact_sha256,
                "status": critique.status,
                "route": _route_dict(critique.route),
                "evidence_refs": list(critique.evidence_refs),
                "finding_count": len(critique.findings),
                "findings": [
                    {
                        key: finding[key]
                        for key in ("finding_id", "severity", "category", "evidence_ids")
                        if key in finding
                    }
                    for finding in critique.findings
                ],
                "coverage_fields": sorted(str(key) for key in critique.coverage),
            }
        return {
            "workflow": "counterpoint.v1",
            "artifact": artifact_record,
            "critique": critique_record,
            "gate_results": [item.to_dict() for item in gates],
            "decision": decision.to_dict(),
            "correction_count": correction_count,
        }

    def _blocked(
        self,
        lifecycle: RunLifecycle,
        spec: WorkflowSpec,
        artifact: Artifact | None,
        critique: Critique | None,
        gates: Sequence[GateResult],
        correction_count: int,
        route_record: Mapping[str, Any],
        *,
        reason: str,
        verdict: str,
        next_actor: str,
        dispositions: Sequence[Mapping[str, Any]] = (),
    ) -> WorkflowResult:
        if artifact is None:
            artifact = Artifact.from_content(
                run_id=spec.run_id,
                task_id=spec.task_id,
                artifact_id=f"{spec.run_id}-blocked",
                version=1,
                content=b"",
                content_ref="internal://blocked",
                status="blocked",
            )
        decision = self._decision(
            spec,
            artifact,
            critique,
            tuple(gates),
            route_record,
            verdict=verdict,
            next_actor=next_actor,
            dispositions=dispositions,
        )
        lifecycle.block(
            spec.run_id,
            reason,
            evidence_refs=tuple(item.evidence_id for item in gates),
            payload=self._audit_payload(
                decision=decision,
                artifact=artifact,
                critique=critique,
                gates=gates,
                correction_count=correction_count,
            ),
        )
        return WorkflowResult("blocked", decision, artifact, critique, correction_count, reason)


__all__ = [
    "CounterpointWorkflow",
    "WorkflowContractError",
    "WorkflowResult",
    "WorkflowSpec",
]
