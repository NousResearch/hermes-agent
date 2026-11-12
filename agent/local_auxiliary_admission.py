"""Admission boundary for auxiliary model calls and their Relay route state."""

from __future__ import annotations

import contextvars
from typing import Any, Optional


class LocalAuxiliaryAdmissionRequired(RuntimeError):
    """A local auxiliary route has no task/run-bound admission in this call path."""


_RELAY_AUX_CALL_CONTEXT: contextvars.ContextVar[Optional[dict[str, Any]]] = contextvars.ContextVar(
    "auxiliary_relay_call", default=None,
)


def require_nonlocal_auxiliary_route(
    client: Any, provider: str | None, task: str | None, *, base_url: str | None = None,
) -> None:
    """Refuse local/private auxiliary calls before Relay or provider I/O."""
    bases = []
    current = client
    for _ in range(4):
        if current is None:
            break
        base = str(getattr(current, "base_url", "") or "").strip()
        if base:
            bases.append(base)
        current = getattr(current, "_client", None)
    base_url = str(base_url or (bases[0] if bases else ""))
    try:
        from hermes_cli.middleware import _is_local_llm_route

        local = _is_local_llm_route(provider, base_url)
    except (ImportError, AttributeError, TypeError, ValueError):
        local = True
    if local:
        raise LocalAuxiliaryAdmissionRequired(
            f"Local auxiliary route refused before inference (task={task or 'call'}); "
            "this path has no task/run-bound GPU admission"
        )


def set_relay_auxiliary_route(provider: str | None, model: str | None, api_mode: str | None) -> None:
    context = _RELAY_AUX_CALL_CONTEXT.get()
    if context is None:
        return
    context["provider"] = str(provider or "auxiliary")
    context["model"] = str(model or "unknown")
    context["response_model"] = None
    context["api_mode"] = str(api_mode or "chat_completions")


def record_route_info(
    route_info: Optional[dict[str, str]], provider: Optional[str], model: Optional[str]
) -> None:
    """Expose the concrete route selected for one auxiliary call."""
    if route_info is not None:
        route_info["provider"] = provider or "auto"
        route_info["model"] = model or "default"


def rethrow_local_model_admission_refusal(error, compressor, short_error_text) -> None:
    """Keep policy refusals terminal in the auxiliary-summary recovery ladder."""
    from hermes_cli.middleware import LocalModelAdmissionBlockedError

    if isinstance(error, (LocalAuxiliaryAdmissionRequired, LocalModelAdmissionBlockedError)):
        compressor._last_summary_error = short_error_text(error)
        raise error


def raise_if_stale_attempt_or_admission_refusal(compressor, error, short_error_text) -> None:
    """Preserve cancellation ordering and keep local-model summary refusals terminal."""
    from agent.conversation_compression import _raise_if_stale_attempt

    _raise_if_stale_attempt(compressor)
    rethrow_local_model_admission_refusal(error, compressor, short_error_text)
