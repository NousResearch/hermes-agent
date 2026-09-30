"""SamAgent — Spec-driven, contract-gated multi-agent builder on top of Hermes Agent.

Architecture D+ (docs/samagent/05-final-plan.md):
- Hermes Agent remains the unmodified runtime engine.
- All SamAgent domain logic lives in this ``samagent`` library and ``plugins/samagent``.
"""
from __future__ import annotations

__version__ = "0.1.0"

from samagent.spec.models import (
    AssumptionEntry,
    AutonomyLevel,
    BudgetSpec,
    ModuleSpec,
    SpecDocument,
    UserStory,
)

__all__ = [
    "__version__",
    "AssumptionEntry",
    "AutonomyLevel",
    "BudgetSpec",
    "ModuleSpec",
    "SpecDocument",
    "UserStory",
]
