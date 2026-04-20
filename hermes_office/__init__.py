"""
Hermes Digital Office — web-based, game-style management UI for digital employees.

This package is intentionally **additive**: importing it never alters Hermes core
behavior. The CLI hooks (`hermes office …`) are wired in `hermes_cli/office_cmd.py`.

Public surface (kept tiny on purpose; full types live in `models.py`):

    from hermes_office import (
        Employee, Department, Task, ActivityEvent,
        ResolvedRole, CapacityReport,
        Store, SkillResolver, EventBus,
        compute_capacity,
        make_runtime,
    )
    from hermes_office.server import build_app   # FastAPI factory

Spec:  ``.kiro/specs/digital-office-ui/{requirements,design,tasks}.md``
"""

from __future__ import annotations

# Re-exports — keep alphabetical
from .capacity import compute as compute_capacity
from .eventbus import EventBus
from .models import (
    Activity,
    ActivityEvent,
    AvatarStyle,
    CapacityReport,
    Department,
    Employee,
    HostProfile,
    ModelProfile,
    ResolvedRole,
    Task,
    Zone,
)
from .skill_resolver import SkillResolver
from .store import Store

__all__ = [
    "Activity",
    "ActivityEvent",
    "AvatarStyle",
    "CapacityReport",
    "Department",
    "Employee",
    "EventBus",
    "HostProfile",
    "ModelProfile",
    "ResolvedRole",
    "SkillResolver",
    "Store",
    "Task",
    "Zone",
    "compute_capacity",
    "make_runtime",
]


def make_runtime(kind: str = "simulated"):
    """Lazy-import factory so importing the package never imports the runtime
    bridge (which may pull in the heavy ``run_agent`` module)."""
    from .runtime import make_runtime as _make_runtime

    return _make_runtime(kind)


__version__ = "0.1.0"
