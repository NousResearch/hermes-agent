"""One common execution-budget gate for every agent conversation entry path."""

from functools import wraps
import uuid


def acquire_conversation_permit(*, execution_id, surface, profile_home=None, vouch=None):
    from pathlib import Path
    from hermes_constants import get_hermes_home
    from hermes_cli.execution_budget import (
        borrow_execution_permit, current_execution_permit, try_acquire_execution_permit,
    )
    current = current_execution_permit()
    if vouch is not None:
        if not isinstance(vouch, dict) or str(Path(profile_home or get_hermes_home()).resolve()) != vouch.get("profile_home"):
            from hermes_cli.execution_budget import _unavailable
            return _unavailable()
        return borrow_execution_permit(vouch)
    if (current is not None and not current.lease.released
            and current.profile_home == str(Path(profile_home or get_hermes_home()).resolve())
            and current.bind_root(execution_id)):
        return borrow_execution_permit(current.vouch())
    permit, refusal = try_acquire_execution_permit(profile_home=profile_home, execution_id=execution_id, surface=surface)
    if permit is not None:
        permit.bind_root(execution_id)
    return permit, refusal


def release_conversation_permit(permit):
    try:
        permit.release()
    except (OSError, RuntimeError, ValueError):
        from hermes_cli.execution_retirement import retry_settlement

        retry_settlement(permit)


def execution_budgeted(method):
    @wraps(method)
    def run(agent, *args, **kwargs):
        from agent.background_review import cancel_background_review_for_live_turn
        from hermes_cli.execution_budget import bind_execution_permit, reset_execution_permit

        cancel_background_review_for_live_turn(agent)
        execution_id = agent._conversation_root_id() or str(getattr(agent, "session_id", None) or uuid.uuid4().hex)
        permit, refusal = acquire_conversation_permit(
            execution_id=execution_id, surface=str(getattr(agent, "platform", None) or "agent"),
            vouch=getattr(agent, "_execution_permit_vouch", None))
        if permit is not None:
            permit.bind_root(execution_id)
        if refusal is not None:
            history = kwargs.get("conversation_history", args[2] if len(args) > 2 else None)
            return {"final_response": str(refusal), "messages": list(history or []), "api_calls": 0,
                    "failed": True, "completed": False, "error": str(refusal),
                    "error_code": refusal.reason}
        token = bind_execution_permit(permit)
        try:
            return method(agent, *args, **kwargs)
        finally:
            reset_execution_permit(token)
            release_conversation_permit(permit)
    return run


def managed_execution_budgeted(method):
    @wraps(method)
    async def run(authority, ref, row, policy):
        from hermes_cli.execution_budget import bind_execution_permit, reset_execution_permit, try_acquire_execution_permit

        permit, refusal = try_acquire_execution_permit(
            execution_id=ref.session_id, surface="managed", profile_home=authority.profile_id)
        if refusal is not None:
            result = {"final_response": str(refusal), "messages": [], "failed": True,
                      "completed": False, "error": str(refusal), "error_code": refusal.reason}
            authority.pending_results[row["admission_id"]] = {"result": result, "usage": {}}
            return str(refusal)
        token = bind_execution_permit(permit)
        try:
            return await method(authority, ref, row, policy)
        finally:
            reset_execution_permit(token)
            release_conversation_permit(permit)
    return run
