"""Whole-turn AgentRuntime selection and host-owned finalization.

The conversation loop resolves the descriptor before prologue compaction and
dispatches after normal Hermes session/prompt setup. Provider-specific policy
remains inside the registered plugin.
"""

from collections.abc import Mapping
import logging
from typing import Any, Callable, Dict, List, Optional

from agent.runtime_api import RuntimeFailurePhase, RuntimeSelection, resolve_runtime_registration
from agent.runtime_dispatch import (
    RuntimeExecutionError, build_runtime_tool_inventory, build_runtime_turn_request,
    close_runtime_session, get_runtime_session, make_builtin_codex_registration,
)
from agent.turn_context import build_effective_prompt_messages, compose_effective_system_prompt
from agent.turn_finalizer import finalize_turn

logger = logging.getLogger(__name__)


def resolve_turn_runtime(agent: Any, builtin_runner: Callable[[], Any]) -> Any:
    """Resolve metadata before any compression, factory, auth, or provider call."""
    from hermes_cli.plugins import discover_plugins, get_plugin_manager

    discover_plugins()
    manager = get_plugin_manager()
    registration = resolve_runtime_registration(
        RuntimeSelection(provider=agent.provider, model=agent.model, api_mode=agent.api_mode),
        (make_builtin_codex_registration(builtin_runner), *manager.iter_agent_runtime_registrations()),
    )
    agent._runtime_descriptor = registration.descriptor if registration is not None else None
    agent._runtime_compaction_ownership = (
        registration.descriptor.compaction_ownership if registration is not None else None
    )
    if registration is None:
        close_runtime_session(agent)
        if agent.api_mode == "agent_runtime":
            raise RuntimeExecutionError("api_mode=agent_runtime has no registered runtime")
    return registration


def _materialize_runtime_value(value: Any) -> Any:
    """Convert an immutable runtime payload into JSON-compatible containers."""
    if isinstance(value, Mapping):
        materialized = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError("runtime message mapping keys must be strings")
            materialized[key] = _materialize_runtime_value(nested)
        return materialized
    if isinstance(value, (list, tuple)):
        return [_materialize_runtime_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("runtime message contains a non-serializable value")


def _materialize_runtime_messages(runtime_messages: Any) -> Optional[List[Dict[str, Any]]]:
    """Materialize a runtime message snapshot, or reject an invalid payload."""
    if not isinstance(runtime_messages, (list, tuple)):
        return None
    try:
        materialized = _materialize_runtime_value(runtime_messages)
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid external runtime message snapshot")
        return None
    if not isinstance(materialized, list) or not all(
        isinstance(message, dict) for message in materialized
    ):
        logger.warning("Ignoring invalid external runtime message snapshot")
        return None
    for message in materialized:
        # Persistence markers are host-owned. A runtime must not be able to
        # make a newly returned row look durable before the host flushes it.
        message.pop("_db_persisted", None)
    return materialized


def _runtime_messages_match(left: Any, right: Any) -> bool:
    """Compare rows at the shared effective-prompt boundary.

    Runtime requests intentionally omit host bookkeeping (persistence markers,
    display metadata, row IDs, and API sidecars). Normalize both sides through
    the same provider-neutral projection before deciding whether a returned
    transcript is already present in the live host list.
    """
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return False
    try:
        left_projection = build_effective_prompt_messages((left,))
        right_projection = build_effective_prompt_messages((right,))
    except (TypeError, ValueError):
        return False
    return (
        bool(left_projection and right_projection)
        and left_projection[0] == right_projection[0]
    )


def _merge_external_runtime_messages(
    messages: List[Dict[str, Any]], runtime_messages: Any
) -> None:
    """Merge a runtime transcript delta into the host-owned turn list.

    Runtime requests are immutable snapshots. A runtime may return that full
    snapshot, a current-turn snapshot, or only newly produced rows. Preserve
    the host's original row objects (and their idempotence markers) while
    appending only an unambiguous delta.
    """
    materialized = _materialize_runtime_messages(runtime_messages)
    if materialized is None:
        return
    common_prefix = 0
    while (
        common_prefix < len(messages)
        and common_prefix < len(materialized)
        and _runtime_messages_match(messages[common_prefix], materialized[common_prefix])
    ):
        common_prefix += 1
    if common_prefix == len(materialized):
        return
    if common_prefix:
        messages.extend(materialized[common_prefix:])
        return
    if materialized and _runtime_messages_match(messages[-1], materialized[0]):
        messages.extend(materialized[1:])
        return
    # A delta containing no user row is safe to append; a user-containing
    # snapshot with no stable anchor would risk replaying a prior turn.
    if not any(message.get("role") == "user" for message in materialized):
        messages.extend(materialized)
        return
    logger.warning("Ignoring unanchored external runtime message snapshot")


def _runtime_persistence_succeeded(agent: Any, result: Mapping[str, Any]) -> bool:
    """Report whether host finalization completed its durable DB flush."""
    if getattr(agent, "_persist_disabled", False):
        return False
    if not getattr(agent, "_session_db", None):
        return False
    if getattr(agent, "_last_persistence_error_cause", None) is not None:
        return False
    cleanup_errors = result.get("cleanup_errors") or ()
    if any(
        isinstance(error, str) and error.startswith("persist_session:")
        for error in cleanup_errors
    ):
        return False
    final_response = result.get("final_response")
    if final_response:
        messages = result.get("messages") or ()
        for message in reversed(messages):
            if isinstance(message, Mapping) and message.get("role") == "assistant":
                return message.get("_db_persisted") is True
        return False
    return True


def run_registered_runtime(agent: Any, runtime_registration: Any, context: Any) -> Dict[str, Any]:
    """Dispatch the resolved runtime using the completed Hermes turn context."""
    messages = context.messages
    user_message = context.user_message
    original_user_message = context.original_user_message
    conversation_history = context.conversation_history
    effective_task_id = context.effective_task_id
    turn_id = context.turn_id
    _should_review_memory = context._should_review_memory
    active_system_prompt = context.active_system_prompt
    current_turn_user_idx = context.current_turn_user_idx
    _ext_prefetch_cache = context._ext_prefetch_cache
    _plugin_user_context = context._plugin_user_context
    runtime_session_state = None
    runtime_database = getattr(agent, "_session_db", None)
    runtime_session_id = getattr(agent, "session_id", None)
    if runtime_database is not None and runtime_session_id:
        runtime_session_state = runtime_database.get_runtime_state(
            runtime_session_id,
            runtime_registration.descriptor.runtime_id,
        )
    runtime_tool_schemas = getattr(agent, "tools", ()) or ()
    runtime_prompt_snapshot = compose_effective_system_prompt(
        active_system_prompt,
        getattr(agent, "ephemeral_system_prompt", None),
    )
    runtime_prompt_messages = build_effective_prompt_messages(
        messages,
        current_turn_user_idx=current_turn_user_idx,
        ext_prefetch_cache=_ext_prefetch_cache,
        plugin_user_context=_plugin_user_context,
    )
    request = build_runtime_turn_request(
        provider=agent.provider,
        model=agent.model,
        api_mode=agent.api_mode,
        messages=runtime_prompt_messages,
        prompt_snapshot=runtime_prompt_snapshot,
        tool_schemas=runtime_tool_schemas,
        tool_inventory=build_runtime_tool_inventory(runtime_tool_schemas),
        session_state=runtime_session_state,
        correlation_id=turn_id,
    )
    runtime_session = get_runtime_session(
        agent,
        runtime_registration,
        task_id=effective_task_id,
        turn_messages=messages,
        correlation_id=request.correlation_id,
    )
    agent._last_effective_prompt_hash = request.effective_prompt_hash
    dispatched = runtime_session.run_turn(request)
    # The built-in Codex adapter owns its projected persistence and must
    # retain its existing short-circuit. External runtimes only return an
    # immutable result envelope; host finalization owns the durable turn.
    if runtime_registration.plugin_id != "hermes-core":
        runtime_response = dict(dispatched.response or {})
        _merge_external_runtime_messages(
            messages, runtime_response.get("messages")
        )
        runtime_api_calls = runtime_response.get("api_calls", 0)
        if not isinstance(runtime_api_calls, int) or isinstance(
            runtime_api_calls, bool
        ):
            runtime_api_calls = 0
        runtime_failure = dispatched.failure
        runtime_cancelled = dispatched.cancelled
        runtime_final_response = (
            None
            if runtime_failure is not None or runtime_cancelled
            else agent._strip_think_blocks(
                runtime_response.get("final_response")
            )
        )
        result = finalize_turn(
            agent,
            final_response=runtime_final_response,
            api_call_count=max(0, runtime_api_calls),
            interrupted=runtime_cancelled,
            failed=runtime_failure is not None,
            messages=messages,
            conversation_history=conversation_history,
            effective_task_id=effective_task_id,
            turn_id=turn_id,
            user_message=user_message,
            original_user_message=original_user_message,
            _should_review_memory=_should_review_memory,
            _turn_exit_reason=(
                "runtime_failure"
                if runtime_failure is not None
                else "runtime_cancelled"
                if runtime_cancelled
                else "runtime_completed"
            ),
        )
        result["agent_persisted"] = _runtime_persistence_succeeded(agent, result)
        if runtime_failure is not None:
            result.update(
                {
                    "error": runtime_failure.message,
                    "partial": runtime_failure.phase
                    in {
                        RuntimeFailurePhase.AFTER_VISIBLE_OUTPUT,
                        RuntimeFailurePhase.AFTER_SIDE_EFFECTS,
                    },
                    "failed": True,
                    "failure": runtime_failure,
                    "replay_safe": runtime_failure.replay_safe,
                }
            )
        elif runtime_cancelled:
            result.update(
                {
                    "error": (
                        dispatched.terminal.reason
                        if dispatched.terminal is not None
                        and hasattr(dispatched.terminal, "reason")
                        else "cancelled"
                    ),
                    "partial": True,
                    "interrupted": True,
                }
            )
        return result

    if dispatched.failure is not None:
        _runtime_failure = dispatched.failure
        return {
            "final_response": "",
            "messages": messages,
            "completed": False,
            "api_calls": 0,
            "error": _runtime_failure.message,
            "partial": _runtime_failure.phase
            in {
                RuntimeFailurePhase.AFTER_VISIBLE_OUTPUT,
                RuntimeFailurePhase.AFTER_SIDE_EFFECTS,
            },
            "failed": True,
            "failure": _runtime_failure,
            "replay_safe": _runtime_failure.replay_safe,
            "session_id": getattr(agent, "session_id", None),
        }
    if dispatched.cancelled:
        reason = (
            dispatched.terminal.reason
            if dispatched.terminal is not None
            and hasattr(dispatched.terminal, "reason")
            else "cancelled"
        )
        return {
            "final_response": "",
            "messages": messages,
            "completed": False,
            "api_calls": 0,
            "error": reason,
            "partial": True,
            "failed": False,
            "interrupted": True,
            "session_id": getattr(agent, "session_id", None),
        }
    return dict(dispatched.response)
