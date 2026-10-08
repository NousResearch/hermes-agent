"""One-turn consent for primary-provider fallback, shared by every interactive surface."""

from __future__ import annotations

import logging
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)
_QUESTION_ID = "fallback_route"
_ROUTE_STATE = (
    "_fallback_activated", "_provider_fallback_active", "_provider_fallback_route",
    "_rate_limited_until", "_rate_limit_backoff_count", "_credential_pool",
    "_credential_pool_entry_id", "_config_context_length",
)


class ManualFallbackStopped(Exception):
    """A human cancellation or failed chosen binding ends this turn without retries."""

    def __init__(self, message: str, *, cancelled: bool = False, preserve_redirect: bool = False):
        super().__init__(message)
        self.cancelled = cancelled
        self.preserve_redirect = preserve_redirect


def stopped_turn_result(agent, state, error: ManualFallbackStopped) -> dict:
    """Settle at the phase boundary while its exact current messages are still owned."""
    agent._clear_status_buffer()
    if error.cancelled:
        from agent.turn_recovery import abort_turn_on_interrupt
        pending = agent._drain_pending_redirect() if error.preserve_redirect else None
        result = abort_turn_on_interrupt(
            agent, state.messages, state.conversation_history, state.api_call_count,
            abort_message="Fallback selection cancelled.", interrupt_text="Operation interrupted: fallback selection cancelled.")
        if pending:
            result["pending_steer"] = pending
        return result
    agent._persist_session(state.messages, state.conversation_history)
    return {"final_response": str(error), "error": str(error), "messages": state.messages,
            "api_calls": state.api_call_count, "completed": False, "failed": True,
            "failure_reason": "unknown", "failure_retryable": False}


def uses_manual_fallback(agent) -> bool:
    return not getattr(agent, "_fallback_auto_activate", True)


def has_pending_fallback(agent) -> bool:
    chain = getattr(agent, "_fallback_chain", None) or []
    if uses_manual_fallback(agent):
        return bool(chain and not getattr(agent, "_fallback_manual_attempted", False)
                    and getattr(agent, "_fallback_selection_interactive", None) is not False
                    and callable(getattr(agent, "clarify_callback", None)))
    return getattr(agent, "_fallback_index", 0) < len(chain)


def manual_restore_required(agent) -> bool:
    """An automatic route also expires when its owner changes the policy to manual."""
    return bool(getattr(agent, "_fallback_activated", False) and (
        uses_manual_fallback(agent) or getattr(agent, "_fallback_manual_selected_index", None) is not None))


def reset_turn_selection(agent) -> None:
    """Called only after the preceding turn's required restoration has succeeded."""
    agent._fallback_manual_attempted = False
    agent._fallback_manual_selected_index = None
    agent._fallback_manual_declined = False
    agent._fallback_manual_cancelled = False


def prepare_turn_runtime(agent, publish) -> None:
    """Expire one-turn consent before publishing any route to auxiliary consumers."""
    if uses_manual_fallback(agent) and getattr(agent, "_fallback_bootstrap_active", False) is True:
        raise RuntimeError("The primary was unavailable at startup. Select a working primary with /model before continuing in manual fallback mode.")
    required = manual_restore_required(agent)
    if required:
        from agent.auxiliary_client import clear_runtime_main
        clear_runtime_main()
    restored = agent._restore_primary_runtime()
    if required and not restored:
        raise RuntimeError("Could not restore the primary after manual fallback; refusing to start another turn.")
    publish(agent, strict=required)
    reset_turn_selection(agent)


def fallback_attempt_status(agent, automatic_text: str) -> str:
    """Do not announce a provider switch before consent or without a usable surface."""
    if not uses_manual_fallback(agent):
        return automatic_text
    if has_pending_fallback(agent):
        return "⚠️ Provider unavailable — select a fallback route to continue this turn."
    return "⚠️ Provider unavailable — no fallback route was authorized for this turn."


def _choice_label(entry: dict) -> str:
    """Endpoint paths can carry credentials too; render only the origin."""
    provider, model = str(entry.get("provider") or "?"), str(entry.get("model") or "?")
    base = str(entry.get("base_url") or "").strip()
    if base:
        try:
            parts = urlsplit(base)
            host = parts.hostname
            if parts.scheme and host:
                host = f"[{host}]" if ":" in host else host
                if parts.port:
                    host = f"{host}:{parts.port}"
                base = urlunsplit((parts.scheme, host, "", "", ""))
            else:
                base = "[custom endpoint]"
        except ValueError:
            base = "[custom endpoint]"
    return f"Continue with {model} via {provider}" + (f" ({base})" if base else "")


def _candidates(agent) -> list[tuple[int, dict, str]]:
    from agent.chat_completion_helpers import _fallback_entry_key, _should_skip_fallback_candidate

    unavailable = getattr(agent, "_unavailable_fallback_keys", None)
    if unavailable is None:
        unavailable = agent._unavailable_fallback_keys = set()
    candidates = []
    for index, entry in enumerate(getattr(agent, "_fallback_chain", None) or []):
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider") or "").strip().lower()
        model = str(entry.get("model") or "").strip()
        if _should_skip_fallback_candidate(agent, entry, _fallback_entry_key(entry), provider, model, unavailable):
            continue
        candidates.append((index, dict(entry), _choice_label(entry)))
    return candidates


def _selected_route(agent, candidates):
    from tools.clarify_tool import MAX_CHOICES, MAX_CHOICE_CHARS
    from tools.human_input_hooks import human_input_request

    labels = [item[2] for item in candidates]
    if (not labels or len(labels) > MAX_CHOICES or len(set(labels)) != len(labels)
            or any(len(label) > MAX_CHOICE_CHARS for label in labels)):
        agent._buffer_diagnostic_status(
            "Manual fallback needs one to four eligible routes with distinct, safe labels; no route was selected.")
        return None
    # Surface callbacks consume this normalized batch. Generic clarify normalization would
    # truncate routes and recommend the first provider, neither of which is consent-neutral.
    question = {"qid": _QUESTION_ID, "question": "Select a fallback route for this turn:",
                "choices": labels, "choices_offered": labels, "multi_select": False}
    with human_input_request("clarify", prompt=question["question"]) as human:
        try:
            reply = agent.clarify_callback([question])
        except Exception:
            human.outcome = "error"
            logger.warning("Manual fallback selection could not be completed", exc_info=True)
            return None
        human.outcome = str(reply.get("outcome") or "") if isinstance(reply, dict) else "error"
        agent._fallback_manual_cancelled = human.outcome == "cancelled"
    if not isinstance(reply, dict) or reply.get("outcome") != "submitted":
        return None
    answers = reply.get("answers")
    answer = answers.get(_QUESTION_ID) if isinstance(answers, dict) else None
    if not isinstance(answer, str):
        return None
    return next((item for item in candidates if item[2] == answer.strip()), None)


def _route_identity(agent) -> tuple:
    return agent.model, agent.provider, agent.base_url, agent.api_mode, id(agent.client)


def _rollback_failed_selection(agent, before, identity) -> None:
    """A voice or /model-once route may already be active; restore that route, not the primary."""
    if _route_identity(agent) != identity:
        from agent.route_binding import reinstall_runtime_snapshot
        try:
            reinstall_runtime_snapshot(agent, before["runtime"])
        except Exception as exc:
            raise ManualFallbackStopped("The selected fallback could not be activated or safely restored; this turn has stopped.") from exc
    for name in _ROUTE_STATE:
        setattr(agent, name, before[name])


def try_activate_manual_fallback(agent, reason=None, reset_at=None) -> bool:
    """One submitted choice can authorize only that entry, even if binding fails."""
    if not has_pending_fallback(agent):
        return False
    agent._fallback_manual_attempted = True
    selected = _selected_route(agent, _candidates(agent))
    if selected is None or getattr(agent, "_interrupt_requested", False):
        agent._fallback_manual_declined = True
        if getattr(agent, "_fallback_manual_cancelled", False):
            agent.interrupt()
            raise ManualFallbackStopped("Fallback selection cancelled.", cancelled=True)
        if getattr(agent, "_interrupt_requested", False):
            raise ManualFallbackStopped("Fallback selection interrupted.", cancelled=True, preserve_redirect=True)
        return False
    index, entry, _ = selected
    agent._fallback_manual_selected_index = index
    original_chain, original_index = agent._fallback_chain, agent._fallback_index
    from agent.agent_runtime_helpers import _build_primary_runtime_snapshot
    before = {"runtime": _build_primary_runtime_snapshot(agent, agent.api_mode),
              **{name: getattr(agent, name, None) for name in _ROUTE_STATE}}
    identity = _route_identity(agent)
    activated = False
    try:
        # Reuse today's binder, protocol detection and notices through a singleton walk.
        # No skip/error path can ever authorize the next configured provider.
        agent._fallback_chain, agent._fallback_index = [entry], 0
        from agent.fallback_activation import activate_next_fallback
        activated = activate_next_fallback(agent, reason, reset_at)
        if not activated:
            agent._fallback_manual_declined = True
            _rollback_failed_selection(agent, before, identity)
            raise ManualFallbackStopped("The selected fallback could not be activated. No other provider was selected; this turn has stopped.")
        return activated
    finally:
        agent._fallback_chain = original_chain
        agent._fallback_index = index + 1 if activated else original_index
