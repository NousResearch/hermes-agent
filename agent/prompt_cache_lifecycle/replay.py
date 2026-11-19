"""What a keep-alive replays: which requests can be replayed, with which output cap, through which client."""

from __future__ import annotations

import re
from typing import Any

#: Wires the warmer can replay a request on.
WARMABLE_API_MODES = frozenset({"anthropic_messages", "chat_completions"})

_OPENAI_REASONING_MODEL = re.compile(r"(^|/)(gpt-[5-9]|o[1-9])")


def has_cache_markers(api_kwargs: dict[str, Any]) -> bool:
    """True when the request carries at least one ``cache_control`` marker (system, tools or
    messages): Anthropic spec. Without one the route caches automatically (OpenAI spec)."""

    def _walk(node: Any) -> bool:
        if isinstance(node, dict):
            if "cache_control" in node:
                return True
            return any(_walk(v) for v in node.values())
        if isinstance(node, list):
            return any(_walk(v) for v in node)
        return False

    return any(_walk(api_kwargs.get(k)) for k in ("system", "tools", "messages"))


def is_replayable(api_kwargs: dict[str, Any]) -> bool:
    """Whether replaying with the small output cap (:func:`replay_output_cap`) leaves the cache entry untouched.

    Budget-based thinking (``{"type": "enabled", "budget_tokens": N}``) requires
    ``max_tokens > budget_tokens`` and Anthropic keys the message cache on the budget, so the
    replay is both rejected and useless. Adaptive thinking and no thinking replay fine.
    """
    thinking = api_kwargs.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "enabled" and "budget_tokens" in thinking:
        return False
    # Chat Completions gateways turn ``reasoning`` into a thinking budget for Claude models that
    # predate adaptive thinking, so a bounded replay would change (or be refused for) that budget.
    reasoning = (api_kwargs.get("extra_body") or {}).get("reasoning")
    model = str(api_kwargs.get("model") or "")
    if ("claude" in model.lower() and isinstance(reasoning, dict) and reasoning.get("enabled", True)
            and reasoning.get("effort") != "none"):
        from agent.anthropic_adapter import _supports_adaptive_thinking

        return _supports_adaptive_thinking(model)
    return True


def replay_output_cap(model: str) -> int:
    """The smallest output cap the replay may carry: OpenAI's GPT-5 family and o-series refuse an
    output cap below 16 (``max_output_tokens`` minimum, also behind Chat Completions gateways that
    translate to Responses: Nous Portal answers 400 "Expected a value >= 16" to a cap of 1)."""
    return 16 if _OPENAI_REASONING_MODEL.search((model or "").lower()) else 1


#: ``extra_body`` keys that raise the output cap past ``max_tokens`` (Gemini's long-decode budget, or a
#: gateway's own cap field merged into the body): output-only, so the replay drops them.
_EXTRA_BODY_OUTPUT_CAPS = ("max_tokens", "max_completion_tokens", "max_output_tokens")


def bounded_chat_replay(api_kwargs: dict[str, Any], model: str) -> dict[str, Any]:
    """The Chat Completions keep-alive: the same prompt with the smallest output cap ``model`` accepts
    (:func:`replay_output_cap`), not streamed, and no ``extra_body`` field that could lift that cap."""
    kwargs = {k: v for k, v in api_kwargs.items() if k not in ("stream", "stream_options")}
    kwargs["max_completion_tokens" if "max_completion_tokens" in kwargs else "max_tokens"] = replay_output_cap(model)
    extra = kwargs.get("extra_body")
    if isinstance(extra, dict) and any(k in extra for k in _EXTRA_BODY_OUTPUT_CAPS):
        kwargs["extra_body"] = {k: v for k, v in extra.items() if k not in _EXTRA_BODY_OUTPUT_CAPS}
    return kwargs


def _sdk_client_types(api_mode: str) -> tuple[type, ...]:
    """The SDK clients that send a request's kwargs as given on ``api_mode``'s wire."""
    if api_mode == "chat_completions":
        from openai import OpenAI

        return (OpenAI,)
    try:
        import anthropic
    except ImportError:  # no SDK installed: no Anthropic client can exist either
        return ()
    return (anthropic.Anthropic, anthropic.AnthropicBedrock, anthropic.AnthropicVertex)


def replay_cap_honoured(client: Any, api_kwargs: dict[str, Any], model: str, api_mode: str) -> bool:
    """Whether ``client`` sends the bounded replay's output cap as given, so a keep-alive costs at most
    that many output tokens. The OpenAI and Anthropic SDKs send the kwargs as they are. Any other client
    must report the cap it would really send through ``effective_output_cap(**kwargs)``: the native
    Gemini clients translate to another wire and lift the cap to 65,535 tokens when thinking is on. A
    client that reports nothing (ACP subprocess clients, an unknown plugin client) or would raise the
    cap is never warmed."""
    replay = bounded_chat_replay(api_kwargs, model) if api_mode == "chat_completions" else {**api_kwargs, "max_tokens": 1}
    cap = replay.get("max_completion_tokens", replay.get("max_tokens"))
    effective = getattr(client, "effective_output_cap", None)
    if callable(effective):
        return effective(**replay) <= cap
    return isinstance(client, _sdk_client_types(api_mode))
