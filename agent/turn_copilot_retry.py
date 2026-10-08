"""Opt-in Copilot 403 retries, isolated from credential and generic retry budgets."""

from agent.turn_recovery import (
    abort_turn_on_interrupt, compute_error_backoff, interruptible_backoff_sleep, nonretryable_client_error_result,
    settle_delivered_partial,
)


def copilot_403_retry_enabled(agent, api_error):
    from agent.conversation_loop import _is_copilot_provider

    return (
        getattr(api_error, "status_code", None) == 403
        and getattr(agent, "_copilot_403_max_retries", 0) > 0
        and _is_copilot_provider(agent)
    )


def copilot_retry_observation(agent, retry, retry_count, max_retries):
    """Observe the independent budget without changing generic loop accounting.

    Hooks/relay count prior retries from zero; their budget counts total attempts,
    matching generic semantics. A generic error cycle keeps its own observation.
    """
    from agent.conversation_loop import _is_copilot_provider

    limit = getattr(agent, "_copilot_403_max_retries", 0)
    used = getattr(retry, "copilot_403_retries_used", 0)
    policy = getattr(retry, "scheduled_retry_policy", None)
    if limit > 0 and (policy == "copilot_403" or (policy is None and retry_count == 0)) and _is_copilot_provider(agent):
        return used, limit + 1
    return retry_count, max_retries


def add_copilot_403_guidance(agent, api_error, result):
    """Keep the enable hint on every failed 403 surface, including billing/policy cards."""
    from agent.conversation_loop import _is_copilot_provider

    if not (getattr(api_error, "status_code", None) == 403
            and result.get("failed") and _is_copilot_provider(agent)):
        return
    hint = (
        "GitHub Copilot returned HTTP 403 (Forbidden). This may be a permanent "
        "authentication or entitlement problem. If you believe your login and model access "
        "are correct, enable bounded retries with "
        "`hermes config set agent.copilot_403_max_retries 3`, then restart Hermes "
        "(restart the CLI or run `hermes gateway restart`). "
        "Retries are disabled by default and cannot fix missing access."
    )
    if result.get("failure_reason") in {"auth", "auth_permanent"} and not result.get("partial"):
        result["final_response"] = f"{hint}\n\nProvider said: {result['error']}"
    else:
        result["final_response"] += f"\n\n{hint}"


def handle_copilot_403(agent, api_error, classified, retry, *, messages, api_messages,
                      api_kwargs, active_system_prompt, conversation_history,
                      api_call_count, approx_tokens, current_turn_user_idx):
    """Return (action, result, prompt); never reset or borrow the generic retry budget.

    Own the error until success or budget exhaustion: pool rotation, auth refresh,
    payload repair and auto-recovery must not add unbounded 403 replays.
    """
    from agent.conversation_loop import _arm_fallback_restart

    if agent._interrupt_requested:
        if agent.clear_interrupt(preserve_redirect=True):
            retry.restart_with_redirected_messages = True
            return "break", None, active_system_prompt
        result = abort_turn_on_interrupt(
            agent, messages, conversation_history, api_call_count,
            abort_message="Copilot 403 retry interrupted.",
            interrupt_text="Operation interrupted: handling GitHub Copilot HTTP 403.",
        )
        return "return", result, active_system_prompt

    limit = agent._copilot_403_max_retries
    if retry.copilot_403_retries_used < limit:
        retry.copilot_403_retries_used += 1
        wait = compute_error_backoff(
            agent, api_error, retry_count=retry.copilot_403_retries_used, max_retries=limit,
            is_rate_limited=False, is_zai_coding_overload=False,
            base_url=agent.base_url, model=agent.model,
        )
        interrupted = interruptible_backoff_sleep(
            agent, wait, retry, messages=messages, conversation_history=conversation_history,
            api_call_count=api_call_count, abort_message="Copilot 403 retry interrupted.",
            interrupt_text="Operation interrupted: retrying GitHub Copilot HTTP 403.",
            activity_label="Copilot 403 retry backoff",
        )
        if interrupted is not None:
            return "return", interrupted, active_system_prompt
        if retry.restart_with_redirected_messages:
            return "break", None, active_system_prompt
        return "continue", None, active_system_prompt

    # Still forbidden: allow the existing fallback, but no refresh/pool/generic retry
    # may replay this same request after the independently bounded budget is spent.
    if classified.should_fallback and agent._try_activate_fallback(reason=classified.reason):
        prompt = _arm_fallback_restart(agent, api_messages, active_system_prompt, retry)
        return "break", None, prompt
    delivered = settle_delivered_partial(agent, messages, current_turn_user_idx)
    result = nonretryable_client_error_result(
        agent, api_error, classified, status_code=403, api_kwargs=api_kwargs,
        api_messages=api_messages, messages=messages, conversation_history=conversation_history,
        api_call_count=api_call_count, approx_tokens=approx_tokens, provider=agent.provider,
        base_url=agent.base_url, model=agent.model, delivered=delivered,
    )
    return "return", result, active_system_prompt
