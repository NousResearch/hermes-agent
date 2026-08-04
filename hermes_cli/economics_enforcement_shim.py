#!/usr/bin/env python3
"""Production enforcement shim for the Economics Gate + Stall Guard.

MCL-HERMES-ECONOMICS-GATE-002 Finalization. This is the ONLY file added to
`hermes-agent` for runtime enforcement. It is a thin, FAIL-CLOSED bridge from
the dispatcher's spawn loop to the canonical adapter
(`ECONOMICS_TEAM/enforcement/runtime_adapter.py`), which wraps the two pure
engines (`economics_gate`, `stall_detector`).

Fail-closed contract (the whole point):
  - If the adapter cannot be imported -> block (reason ECONOMICS_ENFORCEMENT_UNAVAILABLE).
  - If the adapter raises -> block (reason ECONOMICS_ENFORCEMENT_ERROR).
  - If the gate returns dispatch_allowed=False -> block.
A blocked task is NEVER spawned. This satisfies "ninguna ruta puede convertir
un BLOCK en ejecución" and "BLOCKED BEFORE MODEL INVOCATION".

The shim never invents prices, never reads secrets, never calls a provider.
It only maps the claimed task row into a MissionProposal and asks the adapter.
"""

from __future__ import annotations

import os
import sys
import time

# ── locate the canonical adapter ─────────────────────────────────────────────
# The Economics Team module lives in a SEPARATE repo (BusinessOS), not under
# hermes-agent. The canonical, operator-documented location is absolute below.
# Override with HERMES_ECONOMICS_ENFORCEMENT_DIR if your layout differs.
_DEFAULT_DIR = os.path.normpath(
    r"C:\BusinessOS\BusinessOS\2. BUSINESS\0_BUSINESS_OS"
    r"\AI_Organization\ECONOMICS_TEAM\enforcement"
)


def _adapter_module():
    """Import the canonical adapter once; None on any failure."""
    env = os.environ.get("HERMES_ECONOMICS_ENFORCEMENT_DIR")
    base = env if env else _DEFAULT_DIR
    if base not in sys.path:
        sys.path.insert(0, base)
    try:
        import runtime_adapter  # noqa: F401  (imported lazily, only when wired)
        return runtime_adapter
    except Exception:
        return None


# resolved once at module load (import errors are caught per-call too)
_ADAPTER = _adapter_module()


def _row_models(claimed) -> str:
    # claimed is a task row / namedtuple with optional model fields.
    for attr in ("models", "model", "model_id", "llm_model"):
        v = getattr(claimed, attr, None)
        if v:
            return v if isinstance(v, str) else ",".join(str(x) for x in v)
    # Fall back to a board/env default if the task carries no model hint.
    return os.environ.get("HERMES_ECONOMICS_DEFAULT_MODEL", "")


def _row_roles(claimed) -> str:
    for attr in ("roles", "role", "agent_roles"):
        v = getattr(claimed, attr, None)
        if v:
            return v if isinstance(v, str) else ",".join(str(x) for x in v)
    return ""


def _is_attended(claimed, *, default_attended: bool = False) -> bool:
    # A kanban task spawned by the dispatcher is, by definition, autonomous /
    # unattended. Attended work runs interactively through the chat surface,
    # not through dispatch_once. We therefore treat dispatch as UNATTENDED
    # unless an explicit attended flag is present on the row.
    for attr in ("attended", "is_attended"):
        v = getattr(claimed, attr, None)
        if v is not None:
            return bool(v)
    return default_attended


def economics_check(claimed, *, now: float | None = None) -> dict:
    """Run the gate on a claimed ready task. Returns a decision dict.

    Always safe to call. Returns dispatch_allowed=False on any failure.
    """
    now = now if now is not None else time.time()
    adapter = _ADAPTER
    if adapter is None:
        return {
            "dispatch_allowed": False,
            "decision": "BLOCK_MISSING_DATA",
            "block_reason": "ECONOMICS_ENFORCEMENT_UNAVAILABLE",
            "mission_id": getattr(claimed, "id", "?"),
            "event": None,
        }
    try:
        reg = adapter.load_default_registry()
        prop = adapter.row_to_proposal(
            mission_id=str(getattr(claimed, "id", "?")),
            klass=getattr(claimed, "mission_class", "") or getattr(claimed, "klass", "") or "simple",
            models=_row_models(claimed),
            team_size=int(getattr(claimed, "team_size", 1) or 1),
            roles=_row_roles(claimed),
            attended=_is_attended(claimed),
        )
        res = adapter.enforce_before_dispatch(prop, reg)
        return {
            "dispatch_allowed": res.dispatch_allowed,
            "decision": res.decision,
            "block_reason": res.event.block_reason,
            "mission_id": res.event.mission_id,
            "event": res.event.as_record(),
        }
    except Exception as exc:  # never let a gate bug spawn a blocked task
        return {
            "dispatch_allowed": False,
            "decision": "BLOCK_MISSING_DATA",
            "block_reason": f"ECONOMICS_ENFORCEMENT_ERROR: {exc}",
            "mission_id": getattr(claimed, "id", "?"),
            "event": None,
        }


def stall_tick(signals, now: float | None = None):
    """Thin stall check for the agent loop. Returns the canonical decision dict."""
    adapter = _ADAPTER
    if adapter is None:
        return {"status": "UNKNOWN", "dispatch_allowed": False}
    try:
        dec = adapter.stall_guard_tick(signals, now=now)
        return {
            "status": dec.status,
            "since_progress_s": dec.since_progress_s,
            "action": dec.action,
            "block_new_subtasks": dec.block_new_subtasks,
            "block_additional_models": dec.block_additional_models,
            "block_fanout": dec.block_fanout,
        }
    except Exception:
        return {"status": "UNKNOWN", "dispatch_allowed": False}


def fanout_constrain(children, mission_class: str, *, dedupe_roles: bool = True) -> dict:
    """Seam #4: constrain a decomposer's child list to the Economics fan-out limit.

    Fail-closed: missing adapter -> blocked=True, runnable=[], deferred=all.
    The caller creates ONLY `runnable` and persists `deferred` as a record
    (event), never as executable tasks.
    """
    adapter = _ADAPTER
    if adapter is None:
        return {
            "blocked": True,
            "within_limit": False,
            "limit": 0,
            "runnable": [],
            "deferred": list(children),
            "runnable_count": 0,
            "deferred_count": len(children),
        }
    try:
        return adapter.constrain_fanout_children(
            children, mission_class, dedupe_roles=dedupe_roles
        )
    except Exception as exc:
        return {
            "blocked": True,
            "within_limit": False,
            "limit": 0,
            "runnable": [],
            "deferred": list(children),
            "runnable_count": 0,
            "deferred_count": len(children),
            "error": f"ECONOMICS_FANOUT_ERROR: {exc}",
        }
