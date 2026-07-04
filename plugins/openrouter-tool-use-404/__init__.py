"""openrouter-tool-use-404 — provider error classification as a plugin.

When OpenRouter routes a tool-calling request to a model with no
tool-capable endpoint, it returns HTTP 404 with a body like::

    No endpoints found that support tool use. Try disabling "browser_back".
    To learn more about provider routing, visit:
    https://openrouter.ai/docs/guides/routing/provider-selection

The body carries none of the core ``_MODEL_NOT_FOUND_PATTERNS`` signals,
so the built-in classifier files it under ``unknown`` (retryable) and the
retry loop burns several attempts on a deterministic rejection before
surfacing a generic error.

This plugin classifies it as ``model_not_found`` (``retryable=False``,
``should_fallback=True``) so the client-error fast-fallback path in
``conversation_loop.py`` switches to a configured fallback model before
the user ever sees the error.

It is also the reference implementation for the ``classify_api_error``
hook: a provider plugin self-scopes on the ``provider`` kwarg, matches
its provider's quirk in the pre-parsed error context, and returns a
classification dict — no core patches required. See the hook contract in
``hermes_cli.plugins.VALID_HOOKS``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

_PATTERN = "no endpoints found that support tool use"


def classify(
    provider: str = "",
    status_code: Optional[int] = None,
    error_message: str = "",
    **_kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """Claim OpenRouter's tool-use-404, pass on everything else."""
    if (provider or "").strip().lower() != "openrouter":
        return None
    # OpenRouter surfaces this as 404, but some SDK wrappers drop the
    # status; the phrase alone is unambiguous, so accept a missing code.
    if status_code not in (None, 404):
        return None
    if _PATTERN not in (error_message or "").lower():
        return None
    return {
        "reason": "model_not_found",
        "retryable": False,
        "should_fallback": True,
    }


def register(ctx) -> None:
    ctx.register_hook("classify_api_error", classify)
