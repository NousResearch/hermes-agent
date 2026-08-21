"""
Autonomous initiation path (V1).

Public surface:

  * attempt_autonomous_initiation(spec, ctx) -> InitiationResult
        in  agent.autonomy.initiator
  * state.{enable, disable, fire_kill_switch, clear_kill_switch,
          is_enabled, is_kill_switch_active, get_active_objective_id,
          get_active_task_id, get_active_initiated_at,
          get_policy_version, get_profile, reset}
        in  agent.autonomy.state
  * policy.evaluate(spec, ctx) -> Verdict
        in  agent.autonomy.policy

This package is the smallest possible bridge that lets the existing
Executive/Kanban pipeline admit an autonomous objective without
requiring the operator to invoke `hermes kanban create`. It does NOT
create a second Executive, a second Scheduler, a second Kanban, or
a second Worker Dispatcher; it uses the canonical
hermes_cli.kanban_db.create_task API and the standard claim/complete/
comment lifecycle the dispatcher already drives.
"""

from __future__ import annotations

from .initiator import (
    InitiationResult,
    attempt_autonomous_initiation,
    summarize,
)
from .policy import (
    ADMITTED_RISK_CLASS,
    Verdict,
    evaluate,
)
from . import state


__all__ = [
    # initiator
    "InitiationResult",
    "attempt_autonomous_initiation",
    "summarize",
    # policy
    "ADMITTED_RISK_CLASS",
    "Verdict",
    "evaluate",
    # state (re-export submodule)
    "state",
]
