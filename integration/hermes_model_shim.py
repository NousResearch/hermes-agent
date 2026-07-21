"""Provider-adapter shim — example for ``agent/anthropic_adapter.py``.

The kernel's ``ModelProtocol`` is intentionally minimal:

    def generate(messages, tools=None, **kwargs) -> ModelOutput: ...

Every hermes provider adapter (`anthropic_adapter.py`, `gemini_native_adapter.py`,
`bedrock_adapter.py`, `codex_responses_adapter.py`, …) already has a
provider-specific call path. The shim wraps that call path and translates
the result into a single ``ModelOutput``. No adapter internals change.

Drop this pattern into each adapter as a new class or function. The
existing adapter code is the engine; the shim is the bumper.
"""

from __future__ import annotations

from typing import Any

from agent.runtime import ModelOutput, ToolCall


# ----- placeholder for the hermes-side adapter -------------------------------
#
# In hermes-agent, replace this stub with the real import:
#
#     from agent.anthropic_adapter import AnthropicAdapter
#
# The real adapter exposes (varies per provider) a method like:
#
#     def call(self, messages, tools=None, **kwargs) -> AnthropicResponse: ...
#
# where ``AnthropicResponse`` is a hermes-internal dataclass / SDK type.


class _PlaceholderHermesAdapter:
    """Stand-in for whichever hermes adapter you're shimming.

    Replace every reference to this class with the real adapter import
    (e.g. ``AnthropicAdapter``, ``GeminiNativeAdapter``).
    """

    def call(self, messages, tools=None, **kwargs) -> dict:
        raise NotImplementedError("replace with real hermes adapter")


# ----- the actual shim -------------------------------------------------------


class HermesModelShim:
    """Wraps a hermes adapter into ``ModelProtocol``.

    Pattern:
      * delegate ``generate()`` to the underlying adapter's native call,
      * translate the adapter's response shape into ``ModelOutput`` and
        ``ToolCall`` tuples.

    Behavior preserved:
      * credential pool, rate guard, prompt caching, compression, redaction
        — all of these live *inside* the wrapped adapter and are unchanged.
      * The shim is the *only* place the kernel reaches into provider land.
    """

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelOutput:
        # TODO(hermes): call the real adapter method.
        response = self._adapter.call(messages, tools=tools, **kwargs)

        # TODO(hermes): translate the response. The exact mapping depends
        # on the adapter — Anthropic returns content blocks + tool_use blocks;
        # Gemini returns parts; OpenAI Responses returns output items.
        # The pattern below is illustrative.
        content = _extract_text(response)
        tool_calls = tuple(
            ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
            for tc in _extract_tool_calls(response)
        )
        input_tokens, output_tokens = _extract_token_usage(response)
        finish_reason = _extract_finish_reason(response)

        return ModelOutput(
            content=content,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
            raw=response,
        )


# ----- adapter-specific extractors (REPLACE per adapter) ---------------------


def _extract_text(response: Any) -> str:
    """Return the assistant text. For Anthropic this iterates content
    blocks and concatenates text blocks; for Gemini it walks parts."""
    # TODO(hermes): implement against the real response type.
    return response.get("content", "") if isinstance(response, dict) else ""


def _extract_tool_calls(response: Any) -> list[dict[str, Any]]:
    """Return [{id, name, arguments}, ...]. For Anthropic this filters
    ``tool_use`` blocks; for OpenAI it reads ``response.choices[0].message.tool_calls``."""
    # TODO(hermes): implement against the real response type.
    return response.get("tool_calls", []) if isinstance(response, dict) else []


def _extract_token_usage(response: Any) -> tuple[int, int]:
    """Return (input_tokens, output_tokens)."""
    # TODO(hermes): implement against the real response type.
    if isinstance(response, dict):
        return response.get("input_tokens", 0), response.get("output_tokens", 0)
    return 0, 0


def _extract_finish_reason(response: Any) -> str:
    """Normalize to: 'stop' | 'tool_calls' | 'length' | 'content_filter' | 'error'."""
    # TODO(hermes): implement against the real response type.
    if isinstance(response, dict):
        return response.get("finish_reason", "stop")
    return "stop"
