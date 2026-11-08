"""The claimed gateway turn's host permit and generation-qualified settlement."""

import logging

from agent.i18n import t
from agent.execution_admission import release_conversation_permit
from gateway.turn_lease import TurnLeaseTimeoutError

logger = logging.getLogger("gateway.run_inbound")


async def run_claimed_message(runner, event, source, session_key, generation, is_internal):
    from hermes_cli.execution_budget import (
        bind_execution_permit, reset_execution_permit, try_acquire_execution_permit,
    )
    permit = token = None
    try:
        permit, refusal = try_acquire_execution_permit(
            profile_home=runner._resolve_profile_home_for_source(source),
            execution_id=f"{session_key}:run:{generation}", surface=f"gateway:{source.platform.value}")
        if refusal is not None:
            return str(refusal)
        token = bind_execution_permit(permit)
        if not is_internal:
            consumed, reply = await runner._hm_post_admission_consume(event, source, session_key)
            if consumed:
                return reply
        try:
            result = await runner._handle_message_with_agent(event, source, session_key, generation)
        except TurnLeaseTimeoutError:
            logger.error("Rejecting turn after stored-session lease timeout; transcript was not loaded")
            return t("gateway.busy.another_turn_running")
        try:
            await runner._run_post_turn_hooks(agent_result=result, source=source,
                                             is_internal=is_internal, event=event)
        # health: allow BLE001 -- arbitrary extension failures preserve the completed turn; exception content may contain secrets
        except Exception:
            logger.debug("Post-turn extension hook failed; completed turn result preserved")
        return result
    finally:
        try:
            runner._restore_pending_one_turn_model_override(session_key, generation)
            if not getattr(event, "_turn_marker_handoff", False):
                await runner._clear_durable_active_turn(event)
            runner._release_running_agent_state(session_key, run_generation=generation)
            runner._release_turn_lease(session_key, generation)
        finally:
            if token is not None:
                reset_execution_permit(token)
            if permit is not None:
                release_conversation_permit(permit)
