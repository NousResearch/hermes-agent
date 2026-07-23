"""Per-agent extension-middleware policy for reduced-authority turns."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List


@dataclass(frozen=True)
class BypassedRequestMiddleware:
    """Pass-through shape matching the request middleware result contract."""

    payload: Dict[str, Any]
    original_payload: Dict[str, Any]
    trace: List[Any]


def apply_agent_llm_request_middleware(agent: Any, payload: Dict[str, Any], **context):
    """Apply request middleware unless the agent forbids extension egress."""
    if getattr(agent, "_skip_extension_middleware", False):
        return BypassedRequestMiddleware(
            payload=payload,
            original_payload=dict(payload),
            trace=[],
        )

    from hermes_cli.middleware import apply_llm_request_middleware

    return apply_llm_request_middleware(payload, **context)


def run_agent_llm_execution_middleware(
    agent: Any,
    payload: Dict[str, Any],
    perform_api_call: Callable[[Dict[str, Any]], Any],
    **context,
):
    """Run execution middleware unless the agent forbids extension egress."""
    if getattr(agent, "_skip_extension_middleware", False):
        return perform_api_call(payload)

    from hermes_cli.middleware import run_llm_execution_middleware

    return run_llm_execution_middleware(payload, perform_api_call, **context)
