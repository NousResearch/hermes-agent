from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger('agent.codex_runtime')


def certify_codex_turn(agent, turn, *, original_user_message, messages, effective_task_id):
    """Repair a private provider turn at most four times, before transcript projection."""
    from agent import codex_runtime
    from agent.clinical_trace import emit
    from agent.turn_finalizer import _prepare_output_for_delivery, _pre_delivery_reason_codes
    from agent.turn_stop_gates import (
        _build_pre_delivery_repair_directive, _MAX_PRE_DELIVERY_REPAIR_ATTEMPTS,
        _PRE_DELIVERY_TERMINAL_RESPONSE,
    )
    retained: list[dict[str, Any]] = []
    api_calls = 1
    repair_count = 0
    while not turn.interrupted and turn.error is None:
        decision = _prepare_output_for_delivery(
            agent, turn.final_text, logger,
            platform=getattr(agent, 'platform', '') or '',
            effective_task_id=effective_task_id,
            turn_id=getattr(agent, '_current_turn_id', '') or effective_task_id,
            original_user_message=original_user_message, messages=messages,
        )
        projected = list(getattr(turn, 'projected_messages', None) or [])
        submitted = getattr(turn, 'submitted_user_text', None)
        if (submitted and projected and projected[0].get('role') == 'user'
                and projected[0].get('content') == submitted):
            projected = projected[1:]
        if decision.action != 'block':
            turn.final_text = decision.response_text
            if projected and projected[-1].get('role') == 'assistant':
                projected[-1] = {**projected[-1], 'content': decision.response_text}
            turn.projected_messages = retained + projected
            if getattr(agent, '_pre_delivery_gate_active', False) is True:
                from agent.pre_delivery_projection import seal_protected_messages
                seal_protected_messages(turn.projected_messages, decision.response_text)
                agent._release_pre_delivery_text(decision.response_text)
            return turn, api_calls
        # All model-authored commentary from the blocked provider turn stays private.
        for row in projected:
            if row.get('role') == 'assistant' and row.get('tool_calls'):
                retained.append({key: value for key, value in {**row, 'content': ''}.items()
                                 if key not in {'reasoning', 'reasoning_content'}})
            elif row.get('role') == 'tool':
                retained.append(row)
        if repair_count >= _MAX_PRE_DELIVERY_REPAIR_ATTEMPTS:
            turn.final_text = _PRE_DELIVERY_TERMINAL_RESPONSE
            turn.error = 'pre_delivery_repair_exhausted'
            turn.projected_messages = retained + [{'role': 'assistant', 'content': turn.final_text}]
            emit(agent, 'repair', outcome='failed', repair_count=repair_count)
            return turn, api_calls
        agent._llm_output_transform = None
        repair_count += 1
        agent._pre_delivery_repair_attempts = repair_count
        emit(agent, 'repair', outcome='block', repair_count=repair_count)
        agent._iters_since_skill = getattr(agent, '_iters_since_skill', 0) + turn.tool_iterations
        codex_runtime._record_codex_app_server_compaction(agent, turn)
        codex_runtime._record_codex_app_server_usage(agent, turn, messages=messages)
        directive = _build_pre_delivery_repair_directive(
            attempt=repair_count, reason=_pre_delivery_reason_codes(decision.reason)[0],
        )
        try:
            turn = agent._codex_session.run_turn(user_input=directive)
            api_calls += 1
        # health: allow BLE001 -- Provider repair errors must fail closed; tracebacks may expose protected clinical output.
        except Exception as exc:
            logger.warning('codex repair failed (%s)', type(exc).__name__)
            codex_runtime._close_codex_session(agent)
            turn.error = 'pre_delivery_repair_provider_error'
            break
    if turn.error and getattr(agent, '_pre_delivery_gate_active', False) is True:
        turn.error = 'pre_delivery_provider_error'
    turn.projected_messages = retained
    turn.final_text = None
    return turn, api_calls
