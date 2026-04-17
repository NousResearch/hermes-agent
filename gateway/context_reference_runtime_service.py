"""Shared runtime helpers for gateway @-context reference expansion."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from agent.context_references import preprocess_context_references_async
from agent.model_metadata import get_model_context_length


@dataclass(slots=True)
class GatewayContextReferenceOutcome:
    """Result of gateway-side @-context reference preprocessing."""

    message_text: str
    blocked_warning: str | None = None


def should_expand_gateway_context_references(message_text: str) -> bool:
    """Return True when the message might contain @-context references."""

    return "@" in str(message_text or "")


def resolve_gateway_context_reference_cwd(
    messaging_cwd: str | None,
    *,
    home_dir: str | None = None,
) -> str:
    """Resolve the cwd/root used for gateway @-context expansion."""

    return str(messaging_cwd or os.path.expanduser(home_dir or "~"))


async def expand_gateway_context_references(
    message_text: str,
    *,
    model: str,
    base_url: str = "",
    messaging_cwd: str | None = None,
    home_dir: str | None = None,
    preprocessor: Callable[..., Awaitable[Any]] = preprocess_context_references_async,
    context_length_loader: Callable[..., int] = get_model_context_length,
    logger=None,
) -> GatewayContextReferenceOutcome:
    """Expand gateway @-references while preserving current user-facing behavior."""

    if not should_expand_gateway_context_references(message_text):
        return GatewayContextReferenceOutcome(message_text=message_text)

    cwd = resolve_gateway_context_reference_cwd(
        messaging_cwd,
        home_dir=home_dir,
    )
    try:
        context_length = context_length_loader(model, base_url=base_url or "")
        result = await preprocessor(
            message_text,
            cwd=cwd,
            context_length=context_length,
            allowed_root=cwd,
        )
        if getattr(result, "blocked", False):
            warnings = list(getattr(result, "warnings", None) or [])
            return GatewayContextReferenceOutcome(
                message_text=message_text,
                blocked_warning="\n".join(warnings) or "Context injection refused.",
            )
        if getattr(result, "expanded", False):
            return GatewayContextReferenceOutcome(
                message_text=str(getattr(result, "message", message_text) or message_text)
            )
    except Exception as exc:
        if logger is not None:
            logger.debug("@ context reference expansion failed: %s", exc)

    return GatewayContextReferenceOutcome(message_text=message_text)
