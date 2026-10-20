"""Execute iteration-limit summaries through the ordinary LLM middleware."""

from __future__ import annotations


def execute_summary_call(agent, api_request_id: str, request, callback, *, retry_count: int, api_call_count: int = 0):
    from hermes_cli.middleware import _safe_copy, run_llm_execution_middleware

    api_mode = str(getattr(agent, "api_mode", "") or "chat_completions")
    # Match the request stage's snapshot semantics. Each attempt owns its mutable
    # copy so an execution middleware cannot rewrite the next summary retry.
    original_request = _safe_copy(request)
    request = _safe_copy(original_request)

    def perform(next_request):
        # Codex's interruptible driver already owns its request lifecycle.
        if api_mode == "codex_responses":
            return callback(next_request)
        from agent import relay_llm

        return relay_llm.execute_current(
            next_request, callback,
            name=str(getattr(agent, "provider", "") or "provider"),
            model_name=str(getattr(agent, "model", "") or ""),
            metadata={"api_mode": api_mode, "api_request_id": api_request_id,
                      "call_role": "iteration_summary", "retry_count": retry_count},
            defer_logical_completion=True,
        )

    # Middleware may return a provider-shaped refusal without invoking perform.
    # Keep the provider-specific driver and retry loop behind this same boundary.
    return run_llm_execution_middleware(
        request, perform, original_request=original_request, api_request_id=api_request_id,
        task_id=getattr(agent, "_current_task_id", "") or "",
        turn_id=getattr(agent, "_current_turn_id", "") or "",
        session_id=getattr(agent, "session_id", "") or "",
        platform=getattr(agent, "platform", "") or "",
        model=getattr(agent, "model", ""), provider=getattr(agent, "provider", ""),
        base_url=getattr(agent, "base_url", ""), api_mode=api_mode,
        api_call_count=api_call_count, middleware_trace=[],
    )
