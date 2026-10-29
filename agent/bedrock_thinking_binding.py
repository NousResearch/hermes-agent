"""Bedrock Converse recovery for Claude thinking-block prefix-binding rejections.

Signed thinking blocks are bound to the conversation prefix that produced them. Bedrock rejects a
replayed block when earlier messages, the system prompt or tools change, including within the same
provider. Anthropic OAuth-to-Bedrock fallback is one trigger because the OAuth route transforms the
system prompt and tool names. AWS documents the opt-in remedy: send the ``thinking-binding-controls-2026-08-01``
beta with ``thinking.block_binding.prefix_mismatch_behavior=drop_block`` in
``additionalModelRequestFields`` and the server drops only the mismatched blocks
(https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-thinking-block-binding.html).
Healthy requests are never changed; the opt-in rides only the single retry after that exact rejection.
Related: the native Anthropic route change in NousResearch/hermes-agent#106426.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

THINKING_BINDING_BETA = "thinking-binding-controls-2026-08-01"

# Only the provider's binding verdict: the prefix-bound message or the remedy's field name. Generic
# signature or 400 errors are deliberately not matched.
_BINDING_REJECTION_PATTERN = re.compile(
    r"ValidationException.*(?:bound to a different conversation|prefix_mismatch_behavior)",
    re.IGNORECASE | re.DOTALL,
)


def _binding_thinking(model: str, thinking: Any) -> Optional[dict[str, Any]]:
    """``thinking`` with ``block_binding`` drop_block, or None when no opt-in is valid: explicitly
    disabled thinking stays off, and an omitted ``thinking`` is filled in only for Claude models that
    already default to adaptive thinking (no fabricated support for other models)."""
    if thinking is None:
        from agent.anthropic_adapter import _supports_adaptive_thinking
        from agent.bedrock_adapter import is_anthropic_bedrock_model

        if not (is_anthropic_bedrock_model(model) and _supports_adaptive_thinking(model)):
            return None
        thinking = {"type": "adaptive"}  # what the provider applies when the field is omitted
    if not isinstance(thinking, dict) or thinking.get("type") == "disabled":
        return None
    binding = thinking.get("block_binding")
    binding = dict(binding) if isinstance(binding, dict) else {}
    binding["prefix_mismatch_behavior"] = "drop_block"
    return {**thinking, "block_binding": binding}


def recover_from_thinking_binding_rejection(exc: BaseException, kwargs: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Retry kwargs carrying the binding opt-in, or None when ``exc`` is not a binding rejection or the
    request cannot opt in (caller re-raises). ``kwargs`` and everything it references stay untouched;
    the copy shares messages, system and tools by reference and only rebuilds
    ``additionalModelRequestFields``. Callers send the retry once and let its failure propagate."""
    if not _BINDING_REJECTION_PATTERN.search(str(exc)):
        return None
    model = str(kwargs.get("modelId") or "")
    fields = kwargs.get("additionalModelRequestFields")
    fields = fields if isinstance(fields, dict) else {}
    thinking = _binding_thinking(model, fields.get("thinking"))
    if thinking is None:
        return None
    betas = fields.get("anthropic_beta")
    betas = list(betas) if isinstance(betas, (list, tuple)) else []
    if THINKING_BINDING_BETA not in betas:
        betas.append(THINKING_BINDING_BETA)
    logger.warning(
        "bedrock: %s rejected replayed thinking bound to a different conversation prefix - resending "
        "once with block_binding.prefix_mismatch_behavior=drop_block.", model or "model",
    )
    return {**kwargs, "additionalModelRequestFields": {**fields, "anthropic_beta": betas, "thinking": thinking}}
