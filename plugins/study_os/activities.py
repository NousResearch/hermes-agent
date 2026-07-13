"""Domain adapters that turn a learning objective into grounded activities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


APPLIED_EVIDENCE_TARGETS = {"execution", "near_transfer", "far_transfer"}


@dataclass(frozen=True)
class ActivityContext:
    """The caller-owned facts an adapter needs to shape one activity."""

    project: dict[str, Any]
    contract: dict[str, Any]
    evidence_target: str
    recommendation: dict[str, Any] | None
    success_criteria: list[str]
    source_anchors: list[dict[str, Any]]


@dataclass(frozen=True)
class EvidenceIssue:
    code: str
    message: str


class ActivityAdapter(ABC):
    """A domain boundary for activity shape and acceptable evidence."""

    name = "general.v1"

    @abstractmethod
    def build(self, context: ActivityContext) -> dict[str, Any]:
        """Return domain-specific ActivitySpec fields."""

    def validate_observation(
        self,
        activity: dict[str, Any],
        observation: dict[str, Any],
    ) -> list[EvidenceIssue]:
        """Return unmet domain evidence requirements before persistence."""

        return []


class GeneralActivityAdapter(ActivityAdapter):
    """Default adapter for exam study and domain-neutral learning."""

    name = "general.v1"

    def build(self, context: ActivityContext) -> dict[str, Any]:
        target = context.evidence_target
        return {
            "activity_adapter": self.name,
            "kind": str((context.recommendation or {}).get("intervention") or "evidence_probe"),
            "instructions": f"Produce learner-authored {target} evidence for: {context.contract['objective']}",
            "response_policy": "Collect the learner's response before feedback or evaluator judgment.",
            "rubric_requirements": context.success_criteria
            or ["valid result", "reasoning made explicit", "independent contribution identified"],
            "evidence_requirements": ["evaluator"],
        }


class EngineeringActivityAdapter(ActivityAdapter):
    """Ground engineering learning in real source, commands, and artifacts."""

    name = "engineering.v1"
    _KINDS = {
        "recall": "engineering_retrieval",
        "recognition": "engineering_source_trace",
        "execution": "engineering_execution",
        "explanation": "engineering_invariant_explanation",
        "near_transfer": "engineering_near_transfer",
        "far_transfer": "engineering_design_transfer",
    }

    def build(self, context: ActivityContext) -> dict[str, Any]:
        target = context.evidence_target
        instructions = {
            "execution": (
                "Work in the real engineering workspace: run, trace, reproduce, or implement the smallest "
                f"observable task that demonstrates {context.contract['objective']}."
            ),
            "explanation": (
                "Inspect the anchored implementation or command output, then explain the controlling invariant, "
                f"boundary, and one failure mode for {context.contract['objective']}."
            ),
            "near_transfer": (
                "Change one implementation condition while preserving the core invariant; predict the result, "
                "execute it, and compare prediction with observed output."
            ),
            "far_transfer": (
                "Apply the demonstrated engineering principle in a materially different component or design, "
                "and justify which constraints still hold."
            ),
        }.get(
            target,
            "Retrieve the engineering concept from the actual source and identify where it controls runtime behavior.",
        )
        requirements = ["evaluator", "source_anchors"]
        if target in APPLIED_EVIDENCE_TARGETS:
            requirements.append("artifact_refs")
        return {
            "activity_adapter": self.name,
            "kind": self._KINDS.get(target, "engineering_source_trace"),
            "instructions": instructions,
            "response_policy": (
                "Require a prediction or explanation before feedback; distinguish inspected facts from inference."
            ),
            "rubric_requirements": context.success_criteria
            or [
                "source file, symbol, command, or benchmark identified",
                "observable result recorded",
                "controlling invariant explained",
                "verification or failure condition stated",
            ],
            "evidence_requirements": requirements,
        }

    def validate_observation(
        self,
        activity: dict[str, Any],
        observation: dict[str, Any],
    ) -> list[EvidenceIssue]:
        return _grounding_issues(activity, observation)


class ResearchActivityAdapter(ActivityAdapter):
    """Ground research learning in claims, source locations, and replication artifacts."""

    name = "research.v1"
    _KINDS = {
        "recall": "research_claim_retrieval",
        "recognition": "research_source_grounding",
        "execution": "research_replication",
        "explanation": "research_mechanism_explanation",
        "near_transfer": "research_boundary_replication",
        "far_transfer": "research_hypothesis_transfer",
    }

    def build(self, context: ActivityContext) -> dict[str, Any]:
        target = context.evidence_target
        instructions = {
            "execution": (
                "Reproduce the smallest source-anchored result relevant to the objective. Record method, "
                "environment, observed result, and any divergence from the cited claim."
            ),
            "explanation": (
                "Explain the source-anchored claim in your own causal or mathematical terms, then state one "
                "assumption and one limitation that the evidence does not remove."
            ),
            "near_transfer": (
                "Vary one assumption, dataset slice, seed, or parameter from the source method; predict the "
                "effect and compare the replicated result with that prediction."
            ),
            "far_transfer": (
                "Form a falsifiable extension of the source claim in a different setting and specify the "
                "experiment and evidence that would reject it."
            ),
        }.get(
            target,
            "Locate the exact source claim and retrieve its method, evidence, and stated boundary without cues.",
        )
        requirements = ["evaluator", "source_anchors"]
        if target in APPLIED_EVIDENCE_TARGETS:
            requirements.append("artifact_refs")
        return {
            "activity_adapter": self.name,
            "kind": self._KINDS.get(target, "research_source_grounding"),
            "instructions": instructions,
            "response_policy": (
                "Separate the source's claim, the learner's inference, and the observed replication result."
            ),
            "rubric_requirements": context.success_criteria
            or [
                "claim and exact source location identified",
                "method and environment recorded",
                "observed result distinguished from interpretation",
                "uncertainty, assumption, or limitation stated",
            ],
            "evidence_requirements": requirements,
        }

    def validate_observation(
        self,
        activity: dict[str, Any],
        observation: dict[str, Any],
    ) -> list[EvidenceIssue]:
        return _grounding_issues(activity, observation)


def _grounding_issues(
    activity: dict[str, Any],
    observation: dict[str, Any],
) -> list[EvidenceIssue]:
    issues: list[EvidenceIssue] = []
    anchors = observation.get("source_anchors", activity.get("source_anchors", []))
    if not isinstance(anchors, list) or not any(
        isinstance(anchor, dict) and str(anchor.get("ref") or "").strip() for anchor in anchors
    ):
        issues.append(
            EvidenceIssue(
                "SOURCE_ANCHOR_REQUIRED",
                "This activity requires a file, command, paper, dataset, or other source anchor.",
            )
        )
    if activity.get("evidence_target") in APPLIED_EVIDENCE_TARGETS:
        artifacts = observation.get("artifact_refs")
        if not isinstance(artifacts, list) or not artifacts or any(
            not isinstance(item, str) or not item.strip() for item in artifacts
        ):
            issues.append(
                EvidenceIssue(
                    "ARTIFACT_REFERENCE_REQUIRED",
                    "Applied evidence requires artifact_refs naming reproducible commands, outputs, files, or results.",
                )
            )
    return issues


def activity_adapter_for(project: dict[str, Any]) -> ActivityAdapter:
    """Select the narrowest domain adapter supported by a project manifest."""

    domain_pack = str(project.get("domain_pack") or "").casefold()
    domain = str(project.get("domain") or "").casefold()
    if domain_pack.startswith("engineering.") or domain == "engineering":
        return EngineeringActivityAdapter()
    if domain_pack.startswith("research.") or domain == "research":
        return ResearchActivityAdapter()
    return GeneralActivityAdapter()
