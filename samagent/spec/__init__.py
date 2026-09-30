"""SamAgent specification package: models, adaptive interview, critique linter, and red-first acceptance tests."""
from __future__ import annotations

from samagent.spec.acceptance import (
    RedFirstCheckResult,
    generate_acceptance_suite,
    verify_red_first,
)
from samagent.spec.critique import CritiqueReport, SpecIssue, critique_spec
from samagent.spec.interview import (
    InterviewOption,
    InterviewQuestion,
    generate_interview_questions,
    synthesize_spec_from_brief,
)
from samagent.spec.models import (
    AssumptionEntry,
    AutonomyLevel,
    BudgetSpec,
    ModuleSpec,
    SpecDocument,
    UserStory,
)

__all__ = [
    "AssumptionEntry",
    "AutonomyLevel",
    "BudgetSpec",
    "CritiqueReport",
    "InterviewOption",
    "InterviewQuestion",
    "ModuleSpec",
    "RedFirstCheckResult",
    "SpecDocument",
    "SpecIssue",
    "UserStory",
    "critique_spec",
    "generate_acceptance_suite",
    "generate_interview_questions",
    "synthesize_spec_from_brief",
    "verify_red_first",
]
