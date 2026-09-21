"""Router-timeout shim predicates (#68396), extracted to a leaf module.

Every consumer imports these two pure predicates from this leaf: ``agent.chat_completion_helpers``
(eagerly at module load) and, lazily inside the function, ``agent.auxiliary_client`` and
``agent.transports.chat_completions.validate_response``. They used to live in
``agent.transports.chat_completions``, a heavy
transport module whose own top-level imports (``prompt_builder``, ``reasoning_effort``,
``message_sanitization``, ``moonshot_schema``, ...) pull in a large import graph.
``is_router_timeout_shim`` was defined ~110 lines BELOW those imports, so the eager
``from agent.transports.chat_completions import is_router_timeout_shim`` in
``chat_completion_helpers`` coupled that consumer's importability to the heavy module's
load ordering and to whatever version of it a long-lived gateway happened to have cached
in ``sys.modules``.

Incident: 2026-09-21, agent-cron turns died with
``ImportError: cannot import name 'is_router_timeout_shim' from
'agent.transports.chat_completions'`` inside a long-lived gateway. This is the bug class
pinned by ``tests/cron/test_stale_module_leaf_imports.py``: a foundational symbol must
live in a LEAF module with no heavy imports, so a consumer binds it atomically and can
never observe a partial or stale ``chat_completions``. There is no re-export shim in
``agent.transports.chat_completions``: internal paths are not API, so in-tree consumers
import from the defining leaf module (root AGENTS.md "no re-export shims for internal moves").
"""

from typing import Any

_ROUTER_TIMEOUT_SHIM = "Connect timeout, please try again later."


def _has_positive_completion_tokens(usage: Any) -> bool:
    """Return whether a response usage object proves text was generated."""
    for field in ("completion_tokens", "output_tokens"):
        value = usage.get(field) if isinstance(usage, dict) else getattr(usage, field, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return True
    return False


def router_timeout_shim_may_follow(text: str) -> bool:
    """True while streamed text is still a prefix of the shim sentinel (hold it back until judged)."""
    return bool(text) and _ROUTER_TIMEOUT_SHIM.startswith(text.lstrip())


def is_router_timeout_shim(response: Any) -> bool:
    """Recognize a router failure encoded as a successful ChatCompletion (#68396).

    Some OpenAI-compatible routers answer an upstream connect timeout with HTTP 200 and the
    sentinel as the sole assistant message. Only the exact sentinel, with no tool calls and no
    positive ``completion_tokens``/``output_tokens`` proof of generation, is a shim — a model
    that really produced those words keeps its usage evidence. Shared by every consumer of an
    OpenAI-compatible response: ``validate_response``, the stream assembler, the
    iteration-limit summary and the auxiliary ``_validate_llm_response``.
    """
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or len(choices) != 1:
        return False
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str) or content.strip() != _ROUTER_TIMEOUT_SHIM:
        return False
    if getattr(message, "tool_calls", None):
        return False
    return not _has_positive_completion_tokens(getattr(response, "usage", None))
