"""Evidence for "were these tool-call arguments cut off, or did the model just emit broken JSON?"

A tool call whose argument JSON never closes has two very different causes with one wire
shape: a real output-length cut (``finish_reason="length"``, or a router rewriting it to
``stop``/``tool_calls``), and a malformed generation that the model ended on its own —
fused argument bags, a dropped closing brace — well under its budget (qwen-code#12970).
Both the streaming assembler (``chat_completion_helpers``) and the tool-call validator
(``turn_tool_validation``) consult the same verdict here so they cannot disagree.

Shared by two readers: ``truncation_verdict`` is the pure three-state read, the two
``*_for_response`` helpers pull its inputs off a provider response and a prepared request.
"""

from __future__ import annotations

from typing import Any, Optional

# Completion tokens below this share of the output budget cannot be a token-limit cut: the
# model stopped on its own with most of its budget unused.
TRUNCATION_DISPROOF_RATIO = 0.5

# Hermes usually sends no ``max_tokens`` (provider default applies), so the wire carries no cap
# to compare against. ``turn_truncation.boosted_output_cap`` already assumes 4096 as the budget
# a cap-less request was working with; the verdict uses the same assumption rather than
# declaring every cap-less response inconclusive, which would blind the check for the
# chat-completions population that produces malformed bags most often.
ASSUMED_DEFAULT_OUTPUT_CAP = 4096


def truncation_verdict(output_tokens: Any, output_budget: Any) -> str:
    """Three-state read of a mid-JSON tool-call cut when the provider did NOT report ``length``:

    ``"disproved"`` — usage shows the reply ended well under the output budget, so the arguments
    were malformed generation, not a truncation; ``"corroborated"`` — usage sits at or near the
    budget (a router rewrote ``length`` → ``tool_calls``); ``"inconclusive"`` — no usable usage
    or no budget (a dropped stream never delivers its usage chunk).

    A missing, non-numeric or zero count is inconclusive, never a disproof: the response visibly
    produced the half-written arguments, so ``0`` means a provider that sends no usage."""
    try:
        used, budget = int(output_tokens), int(output_budget)
    except (TypeError, ValueError):
        return "inconclusive"
    if used <= 0 or budget <= 0:
        return "inconclusive"
    return "disproved" if used < budget * TRUNCATION_DISPROOF_RATIO else "corroborated"


def response_output_tokens(agent: Any, usage: Any) -> Optional[int]:
    """Completion/output tokens from a raw provider ``usage`` object (any wire), else None."""
    if not usage:
        return None
    from agent.usage_pricing import normalize_usage
    try:
        return int(normalize_usage(usage, provider=agent.provider, api_mode=agent.api_mode).output_tokens)
    except Exception:
        return None


def output_budget_for_request(agent: Any, api_kwargs: Any) -> int:
    """The output cap this request was working with: the cap actually sent on the wire, else the
    agent's configured ``max_tokens``, else the model's known output limit, else the same default
    the truncation-retry path assumes (``ASSUMED_DEFAULT_OUTPUT_CAP``)."""
    from agent.turn_truncation import _model_output_limit
    wire_cap = None
    if isinstance(api_kwargs, dict):
        for key in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
            wire_cap = wire_cap or api_kwargs.get(key)
    for candidate in (wire_cap, getattr(agent, "max_tokens", None), _model_output_limit(agent)):
        try:
            if candidate is not None and int(candidate) > 0:
                return int(candidate)
        except (TypeError, ValueError):
            continue
    return ASSUMED_DEFAULT_OUTPUT_CAP


def truncation_disproved(agent: Any, *, finish_reason: Any, usage: Any, api_kwargs: Any) -> bool:
    """True only when the provider reported a normal stop AND its usage disproves a cut.
    ``length`` (or no finish_reason at all) is the provider's own truncation verdict and is
    always trusted."""
    if finish_reason in (None, "length"):
        return False
    return truncation_verdict(
        response_output_tokens(agent, usage), output_budget_for_request(agent, api_kwargs)
    ) == "disproved"
