"""Kanban board policy plugin.

Registers lifecycle hooks for board-level policy enforcement:
  - ``pre_tool_call`` — interim evidence gate: vetoes kanban_complete
    calls with empty result when ``kanban.require_result_for_verify``
    is enabled. This is the worker-side stopgap that ships before the
    permanent DB-backstop gate lands — it runs in the worker process
    and blocks the tool call before it reaches ``complete_task``.

Hook registration happens via the standard entry-point pattern:
import and call ``kanban_plugin_register(context)`` at module level,
as expected by the Hermes plugin loader.
"""

from __future__ import annotations


def _require_result_for_verify_plugin() -> bool:
    """Check the board config for kanban.require_result_for_verify.

    Best-effort: returns False when config is unavailable.
    """
    try:
        from hermes_cli.config import load_config, cfg_get
        cfg = load_config()
        return bool(cfg_get(cfg, "kanban", "require_result_for_verify"))
    except Exception:
        return False


def _pre_tool_call_veto_evidence_less_complete(
    tool_name: str,
    args: dict,
    task_id: str = "",
    **kwargs,
) -> dict | None:
    """Veto kanban_complete calls with empty/missing ``result`` field.

    This is the INTERIM stopgap (M0 harness physics): when the board
    policy ``kanban.require_result_for_verify`` is enabled, any
    ``kanban_complete`` call that lacks a non-empty ``result`` field
    is rejected before it reaches the DB layer. The model receives a
    clear tool_error explaining what's required.

    Returns:
        ``{'action': 'block', 'message': '...'}`` to veto the call,
        or ``None`` to allow it to proceed.
    """
    if tool_name != "kanban_complete":
        return None
    if not _require_result_for_verify_plugin():
        return None
    result = args.get("result") if isinstance(args, dict) else None
    if result is not None and (not isinstance(result, str) or result.strip()):
        # Has a non-empty result — allow.
        return None
    return {
        "action": "block",
        "message": (
            "kanban_complete blocked by interim evidence gate: "
            "kanban.require_result_for_verify is enabled. "
            "The result field is required — provide evidence of what was "
            "actually done. You can still pass summary= for the human-readable "
            "handoff, but result= must carry the verifiable output."
        ),
    }


# ---------------------------------------------------------------------------
# Plugin registration — called by the Hermes plugin loader at import time.
# ---------------------------------------------------------------------------

def register(context):
    """Register this plugin's hooks with the Hermes plugin system."""
    context.register_hook("pre_tool_call", _pre_tool_call_veto_evidence_less_complete)
