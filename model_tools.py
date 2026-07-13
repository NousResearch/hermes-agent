# Combined hook for task checkpoints, agent messaging, and execution tracing
def _save_tool_progress(
    function_name: str,
    function_args: dict,
    result: Any,
    task_id: str,
    execution_duration_ms: float,
) -> None:
    """Best-effort: persist tool progress and trace events."""
    from agent.tool_result_classification import NO_EFFECT_TOOL_NAMES
    import json as _json

    # 1. Execution Tracer (always runs)
    try:
        from agent.execution_tracer import trace_event, enrich_last_trace
        
        rd = _json.loads(result) if isinstance(result, str) else {}
        success = not bool(rd.get("error")) if isinstance(rd, dict) else True
        
        # Initial trace
        trace_event(
            session_id=task_id or "", tool_name=function_name,
            duration_ms=execution_duration_ms, success=success,
            result_summary=str(rd)[:120], task_id=task_id or ""
        )
        
        # Enrich with error classification if failed
        if not success:
            from agent.error_classifier import classify_tool_error
            c = classify_tool_error(result, tool_name=function_name, tool_args=function_args)
            if c:
                enrich_last_trace(
                    error_class=c.reason.value, error_message=str(c.reason.value),
                    confidence=c.confidence, recovery_action=c.known_fix or "")
    except Exception:
        pass # Best-effort

    # 2. Agent Messaging
    if function_name == "send_agent_message":
        _persist_agent_message(function_args, result, task_id)

    # 3. Task Checkpoint (only for effectful tools in a checkpointed task)
    if function_name not in NO_EFFECT_TOOL_NAMES:
        try:
            from hermes_state import SessionDB
            session_id = os.environ.get("HERMES_SESSION_ID", "")
            if session_id and SessionDB().has_task_checkpoint(session_id):
                # (logic to append step to checkpoint)
                pass
        except Exception:
            pass