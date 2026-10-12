"""Finalizer fallback and protected certification before cleanup/persistence."""


def resolve_pending_tool_reply(agent, final_response, interrupted, messages, failed, turn_exit_reason):
    # A non-interrupted turn that fell out of the loop after a tool result, with no
    # follow-up assistant text, is the Desktop/TUI "silent stop" (#55316, #54756): the
    # composer returns to ready (or keeps spinning) while the durable transcript ends
    # at a raw ``tool`` row — the user never learns the turn stopped, and the next user
    # message lands as ``tool → user``. Interrupted tails keep
    # ``close_interrupted_tool_sequence``; this is the non-interrupt sibling. Mint the
    # exit reason, fail the turn, and synthesize the visible close so the tail close in
    # ``_persist_step`` persists an assistant row. A turn that already streamed text is
    # left alone: ``_recover_final_from_stream`` owns that recovery (#95514).
    if (
        not final_response
        and not interrupted
        and messages
        and isinstance(messages[-1], dict)
        and messages[-1].get("role") == "tool"
        and not (getattr(agent, "_current_streamed_assistant_text", "") or "").strip()
    ):
        turn_exit_reason = "pending_tool_result"
        failed = True
        final_response = ""
        try:
            if agent._turn_completion_explainer_enabled():
                final_response = (
                    agent._format_turn_completion_explanation("pending_tool_result", None) or ""
                )
        # health: allow BLE001 -- Explainer errors must preserve the pending-tool fallback; exception text may expose clinical context.
        except Exception as exc:
            from agent.conversation_loop import logger
            logger.debug("pending-tool explanation failed (%s)", type(exc).__name__)
            final_response = ""
        if not final_response:
            # The turn-completion explainer opt-out must not reintroduce the silent stop.
            final_response = (
                "No reply: the turn stopped while a tool result was still pending. "
                "Send `continue` to let the model summarize."
            )

    return final_response, failed, turn_exit_reason


def certify_before_cleanup(
    agent, *, final_response, interrupted, failed, completed, turn_exit_reason,
    preserved_verification_fallback, effective_task_id, turn_id, original_user_message, messages, logger,
):
    from agent.turn_finalizer import (
        PreDeliveryDecision, _append_file_mutation_footer, _explain_abnormal_exit,
        _prepare_output_for_delivery,
    )

    # Gate the complete protected candidate before ANY fallible cleanup or trajectory write.
    if getattr(agent, "_pre_delivery_gate_active", False) is True:
        if interrupted:
            final_response = None
        elif final_response:
            if getattr(agent, "_pre_delivery_failed", False) is not True:
                final_response = _append_file_mutation_footer(agent, final_response, logger)
                final_response = _explain_abnormal_exit(
                    agent, final_response, turn_exit_reason, preserved_verification_fallback, logger,
                )
            cache = getattr(agent, "_pre_delivery_output_cache", None)
            if not isinstance(cache, dict) or cache.get("response_text") != final_response:
                try:
                    decision = _prepare_output_for_delivery(
                        agent, final_response, logger, platform=getattr(agent, "platform", "") or "",
                        effective_task_id=effective_task_id, turn_id=turn_id,
                        original_user_message=original_user_message, messages=messages,
                        skip_transform=isinstance(cache, dict),
                    )
                # health: allow BLE001 -- Every preparation failure blocks delivery; traceback text may expose the clinical candidate.
                except Exception as exc:
                    logger.warning("pre_delivery preparation failed (%s)", type(exc).__name__)
                    decision = PreDeliveryDecision(action="block", response_text="")
                final_response = decision.response_text
                if decision.action == "block":
                    failed, completed = True, False
                    turn_exit_reason = "pre_delivery_blocked"
                    agent._pre_delivery_failed = True
                agent._pre_delivery_output_cache = {
                    "response_text": final_response, "transformed": decision.transformed,
                    "pre_transform": decision.pre_transform_response,
                }
        from agent.pre_delivery_projection import seal_protected_messages
        seal_protected_messages(messages, final_response, boundary=getattr(agent, "_current_turn_user_idx", 0))

    return final_response, failed, completed, turn_exit_reason


def attach_delivery_trace(agent, result):
    from agent.clinical_trace import snapshot
    if (delivery_trace := snapshot(agent)) is not None:
        result["clinical_delivery_trace"] = delivery_trace
