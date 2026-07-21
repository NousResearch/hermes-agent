"""Phase 4B Kanban Mapping — pure TaskSpec -> create_task kwargs.

This module is the **only** place that translates a Phase 3
`TaskSpec` dict into the kwargs that ``hermes_cli.kanban_db.create_task``
expects. The translation is **pure** (no side effects, no I/O, no
LLM calls) so it can be unit-tested without a real Kanban DB.

Design constraints:

* **Reuse only** the existing ``kb.create_task`` API.
* **Prohibit** argparse wrappers (``kanban_command``, ``_cmd_create``)
  and LLM-driven helpers (``kanban_decompose``, ``kanban_specify``,
  ``kanban_swarm``, ``create_swarm``).
* **Linear parent linkage only**: task N has task N-1 as parent.
  No DAG. No ``link_tasks`` for cross-links.
* **Idempotency**: each spec has a deterministic
  ``idempotency_key`` of the form ``exec-v2-phase4b:<oid>:<idx>``.
  ``kb.create_task`` already handles dedup on this key.

See ``.hermes/reports/hermes_executive_v2_phase4b_kanban_apply_design/``
for the full design rationale.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from .types import (
    ACTION_CREATE_KANBAN_TASK,
    ACTION_WRITE_KANBAN_METADATA,
)


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

IDEMPOTENCY_KEY_PREFIX = "exec-v2-phase4b"
DEFAULT_WORKSPACE_KIND = "scratch"
DEFAULT_INITIAL_STATUS = "ready"  # never auto-dispatch (Phase 5+)
DEFAULT_CREATED_BY = "executive_v2_phase4b"
RISK_LEVEL_TO_PRIORITY: dict[str, int] = {
    "low": 0,
    "medium": 2,
    "high": 5,
}
REQUIRES_USER_INPUT_PRIORITY_BOOST = 3
MAX_PRIORITY = 10


# ──────────────────────────────────────────────────────────────────────
# Pure helpers
# ──────────────────────────────────────────────────────────────────────

def _priority_from_spec(spec_dict: dict) -> int:
    """Map TaskSpec fields to a Kanban priority (0-10).

    * risk_level=high -> +5
    * risk_level=medium -> +2
    * requires_user_input=True -> +3
    Clamped to [0, MAX_PRIORITY].
    """
    priority = 0
    risk_level = str(spec_dict.get("risk_level", "low") or "low").lower()
    priority += RISK_LEVEL_TO_PRIORITY.get(risk_level, 0)
    if spec_dict.get("requires_user_input", False):
        priority += REQUIRES_USER_INPUT_PRIORITY_BOOST
    if priority < 0:
        return 0
    if priority > MAX_PRIORITY:
        return MAX_PRIORITY
    return priority


def _clamp_timeout_s(timeout_s: Any) -> int:
    """Clamp timeout_s to [1, 86400] (1 second to 1 day)."""
    try:
        v = int(timeout_s)
    except (TypeError, ValueError):
        return 60
    if v < 1:
        return 1
    if v > 86400:
        return 86400
    return v


# ──────────────────────────────────────────────────────────────────────
# Public mapping API
# ──────────────────────────────────────────────────────────────────────

def compute_idempotency_key(
    objective_id: str,
    spec_index: int,
) -> str:
    """Deterministic idempotency key for one (objective, spec_index) pair.

    ``kb.create_task`` with this key returns the existing task_id on retry.
    """
    return f"{IDEMPOTENCY_KEY_PREFIX}:{objective_id}:{int(spec_index)}"


def map_taskspec_to_kanban_payload(
    spec_dict: dict,
    *,
    spec_index: int,
    objective_id: str,
    created_by: str = DEFAULT_CREATED_BY,
    board: Optional[str] = None,
) -> dict:
    """Pure: TaskSpec dict -> create_task kwargs (no side effects).

    Parameters
    ----------
    spec_dict : dict
        A single TaskSpec as serialized by ``OrchestratorPlanPreview.task_specs``.
        Expected keys: ``description``, ``assigned_profile``, ``inputs``,
        ``expected_outputs``, ``dependencies``, ``timeout_s``,
        ``requires_user_input``, ``approval_id``. Unknown keys are ignored.
    spec_index : int
        Position of this spec in the parent's task_specs list (0-based).
        Used to compute the idempotency key.
    objective_id : str
        Phase 1 objective_id (used for the idempotency key and for
        ``session_id`` linkage).
    created_by : str
        Author tag stored in ``tasks.created_by``.
    board : Optional[str]
        Kanban board name. ``None`` means the current default board.

    Returns
    -------
    dict
        A kwargs dict suitable for ``kb.create_task(conn, **kwargs)``.
        The ``parents`` field is set to an empty tuple; the apply loop
        resolves the actual parent linkage as it creates tasks
        sequentially.
    """
    description = str(spec_dict.get("description", "") or "").strip()
    if not description:
        description = "(no description)"
    title = description
    assigned_profile = str(spec_dict.get("assigned_profile", "") or "").strip()
    timeout_s = _clamp_timeout_s(spec_dict.get("timeout_s", 60))
    requires_user_input = bool(spec_dict.get("requires_user_input", False))
    inputs = dict(spec_dict.get("inputs", {}) or {})
    expected_outputs = list(spec_dict.get("expected_outputs", []) or [])
    risk_level = str(spec_dict.get("risk_level", "low") or "low").lower()

    # Body: compact JSON of the spec's input/output/risk metadata.
    body = json.dumps(
        {
            "inputs": inputs,
            "expected_outputs": expected_outputs,
            "timeout_s": timeout_s,
            "requires_user_input": requires_user_input,
            "risk_level": risk_level,
            "phase": "executive_v2_phase4b",
        },
        sort_keys=True,
        ensure_ascii=False,
    )

    idempotency_key = compute_idempotency_key(objective_id, spec_index)

    return {
        "title": title,
        "body": body,
        "assignee": assigned_profile or None,  # Phase 5+ will actually dispatch
        "created_by": created_by,
        "workspace_kind": DEFAULT_WORKSPACE_KIND,
        "workspace_path": None,
        "branch_name": None,
        "tenant": None,
        "priority": _priority_from_spec(spec_dict),
        "parents": (),  # resolved at apply time (linear)
        "triage": False,
        "idempotency_key": idempotency_key,
        "max_runtime_seconds": timeout_s,
        "skills": None,
        "max_retries": None,
        "goal_mode": False,
        "goal_max_turns": None,
        "initial_status": DEFAULT_INITIAL_STATUS,  # never auto-dispatch
        "session_id": objective_id,  # linkage to state_meta
        "board": board,
        "project_id": None,
    }


def build_kanban_apply_preview(
    task_specs: list[dict],
    *,
    objective_id: str,
    created_by: str = DEFAULT_CREATED_BY,
    board: Optional[str] = None,
) -> list[dict]:
    """Pure: list[TaskSpec dict] -> list[create_task kwargs] (no side effects).

    Each item in the returned list has ``parents=()`` (placeholder).
    The actual parent linkage (linear: task N-1 -> task N) is
    resolved at apply time by ``KanbanApplyEngine.apply``.
    """
    out: list[dict] = []
    for idx, spec in enumerate(task_specs):
        kwargs = map_taskspec_to_kanban_payload(
            spec,
            spec_index=idx,
            objective_id=objective_id,
            created_by=created_by,
            board=board,
        )
        out.append(kwargs)
    return out


# ──────────────────────────────────────────────────────────────────────
# Action catalogue presence (used by tests; non-duplication gate)
# ──────────────────────────────────────────────────────────────────────

PHASE4B_REQUIRED_ACTIONS: tuple[str, ...] = (
    ACTION_WRITE_KANBAN_METADATA,
    ACTION_CREATE_KANBAN_TASK,
)


__all__ = [
    "IDEMPOTENCY_KEY_PREFIX",
    "DEFAULT_WORKSPACE_KIND",
    "DEFAULT_INITIAL_STATUS",
    "DEFAULT_CREATED_BY",
    "MAX_PRIORITY",
    "compute_idempotency_key",
    "map_taskspec_to_kanban_payload",
    "build_kanban_apply_preview",
    "_priority_from_spec",  # exported for test introspection
    "_clamp_timeout_s",  # exported for test introspection
    "PHASE4B_REQUIRED_ACTIONS",
]