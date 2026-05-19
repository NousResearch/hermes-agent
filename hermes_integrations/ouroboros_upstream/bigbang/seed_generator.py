"""Vendored upstream SeedGenerator extraction/build subset for Hermes gateway mode.

Copied/adapted from Q00/ouroboros `src/ouroboros/bigbang/seed_generator.py`
at the commit recorded in VENDORED_UPSTREAM.md.

Hermes does not call an LLM from the gateway path here. Instead, the gateway
adapter builds the same structured extraction text from already-confirmed
interview values, then this vendored parser/build path constructs the upstream
Seed model. That moves Seed construction through the upstream generator shape
without introducing secrets, provider calls, or execution authority.
"""

from __future__ import annotations

from typing import Any

from hermes_integrations.ouroboros_upstream.bigbang.interview import InterviewState
from hermes_integrations.ouroboros_upstream.core.seed import (
    BrownfieldContext,
    ContextReference,
    EvaluationPrinciple,
    ExitCondition,
    OntologyField,
    OntologySchema,
    Seed,
    SeedMetadata,
)


class SeedGenerator:
    """Gateway-safe subset of upstream SeedGenerator.

    Retains upstream structured-response preprocessing, parsing, and `_build_seed`
    semantics. Provider-driven `_extract_requirements` is intentionally not
    wired in Hermes gateway mode because it would require runtime model/provider
    access and can mutate cost/side-effect boundaries without explicit approval.
    """

    _KNOWN_PREFIXES = (
        "GOAL:",
        "CONSTRAINTS:",
        "ACCEPTANCE_CRITERIA:",
        "ONTOLOGY_NAME:",
        "ONTOLOGY_DESCRIPTION:",
        "ONTOLOGY_FIELDS:",
        "EVALUATION_PRINCIPLES:",
        "EXIT_CONDITIONS:",
        "PROJECT_TYPE:",
        "CONTEXT_REFERENCES:",
        "EXISTING_PATTERNS:",
        "EXISTING_DEPENDENCIES:",
    )

    def _preprocess_response(self, response: str) -> str:
        import re

        text = response.strip()
        code_block_match = re.search(r"```(?:\w*)\n(.*?)```", text, re.DOTALL)
        if code_block_match:
            text = code_block_match.group(1).strip()
        lines = text.split("\n")
        start_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if any(stripped.startswith(p) for p in self._KNOWN_PREFIXES):
                start_idx = i
                break
        return "\n".join(lines[start_idx:])

    def _parse_extraction_response(self, response: str) -> dict[str, Any]:
        cleaned = self._preprocess_response(response)
        lines = cleaned.strip().split("\n")
        requirements: dict[str, Any] = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            for prefix in self._KNOWN_PREFIXES:
                if line.startswith(prefix):
                    key = prefix[:-1].lower()
                    requirements[key] = line[len(prefix) :].strip()
                    break
        required_fields = ["goal", "ontology_name", "ontology_description"]
        for field_name in required_fields:
            if field_name not in requirements:
                raise ValueError(
                    f"Missing required field: {field_name}. Found: {list(requirements.keys())}. "
                    f"Response preview: {response[:200]}"
                )
        return requirements

    def _build_seed(self, requirements: dict[str, Any], metadata: SeedMetadata) -> Seed:
        constraints: tuple[str, ...] = ()
        if requirements.get("constraints"):
            constraints = tuple(c.strip() for c in requirements["constraints"].split("|") if c.strip())

        acceptance_criteria: tuple[str, ...] = ()
        if requirements.get("acceptance_criteria"):
            acceptance_criteria = tuple(
                c.strip() for c in requirements["acceptance_criteria"].split("|") if c.strip()
            )

        ontology_fields: list[OntologyField] = []
        if requirements.get("ontology_fields"):
            for field_str in requirements["ontology_fields"].split("|"):
                field_str = field_str.strip()
                if not field_str:
                    continue
                parts = field_str.split(":")
                if len(parts) >= 3:
                    ontology_fields.append(
                        OntologyField(
                            name=parts[0].strip(),
                            type=parts[1].strip(),
                            description=":".join(parts[2:]).strip(),
                        )
                    )

        ontology_schema = OntologySchema(
            name=requirements["ontology_name"],
            description=requirements["ontology_description"],
            fields=tuple(ontology_fields),
        )

        evaluation_principles: list[EvaluationPrinciple] = []
        if requirements.get("evaluation_principles"):
            for principle_str in requirements["evaluation_principles"].split("|"):
                principle_str = principle_str.strip()
                if not principle_str:
                    continue
                parts = principle_str.split(":")
                if len(parts) >= 2:
                    weight = 1.0
                    if len(parts) >= 3:
                        try:
                            weight = float(parts[2].strip())
                        except ValueError:
                            weight = 1.0
                    evaluation_principles.append(
                        EvaluationPrinciple(
                            name=parts[0].strip(),
                            description=parts[1].strip(),
                            weight=min(1.0, max(0.0, weight)),
                        )
                    )

        exit_conditions: list[ExitCondition] = []
        if requirements.get("exit_conditions"):
            for condition_str in requirements["exit_conditions"].split("|"):
                condition_str = condition_str.strip()
                if not condition_str:
                    continue
                parts = condition_str.split(":")
                if len(parts) >= 3:
                    exit_conditions.append(
                        ExitCondition(
                            name=parts[0].strip(),
                            description=parts[1].strip(),
                            criteria=":".join(parts[2:]).strip(),
                        )
                    )

        context_references: list[ContextReference] = []
        if requirements.get("context_references"):
            for ref_str in requirements["context_references"].split("|"):
                ref_str = ref_str.strip()
                if not ref_str:
                    continue
                parts = ref_str.split(":")
                if len(parts) >= 3:
                    context_references.append(
                        ContextReference(
                            path=parts[0].strip(),
                            role=parts[1].strip(),
                            summary=":".join(parts[2:]).strip(),
                        )
                    )

        existing_patterns: tuple[str, ...] = ()
        if requirements.get("existing_patterns"):
            existing_patterns = tuple(
                p.strip() for p in requirements["existing_patterns"].split("|") if p.strip()
            )
        existing_dependencies: tuple[str, ...] = ()
        if requirements.get("existing_dependencies"):
            existing_dependencies = tuple(
                d.strip() for d in requirements["existing_dependencies"].split("|") if d.strip()
            )

        return Seed(
            goal=requirements["goal"],
            task_type="code",
            brownfield_context=BrownfieldContext(
                project_type=requirements.get("project_type", "greenfield"),
                context_references=tuple(context_references),
                existing_patterns=existing_patterns,
                existing_dependencies=existing_dependencies,
            ),
            constraints=constraints,
            acceptance_criteria=acceptance_criteria,
            ontology_schema=ontology_schema,
            evaluation_principles=tuple(evaluation_principles),
            exit_conditions=tuple(exit_conditions),
            metadata=metadata,
        )

    def build_from_structured_response(
        self,
        response: str,
        *,
        state: InterviewState | None = None,
        ambiguity_score: float = 0.15,
    ) -> Seed:
        requirements = self._parse_extraction_response(response)
        metadata = SeedMetadata(
            ambiguity_score=ambiguity_score,
            interview_id=state.interview_id if state is not None else None,
        )
        return self._build_seed(requirements, metadata)
