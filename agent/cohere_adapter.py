"""Cohere v2 chat adapter for Hermes Agent.

Thin wrapper around the ``cohere`` SDK that:

  - lazy-imports ``cohere`` so non-Cohere sessions don't pay the import cost
    or the ``provider.cohere`` lazy-install probe;
  - constructs ``cohere.ClientV2`` instances with the right timeouts and
    base-URL handling;
  - processes a ``chat_stream`` SSE event iterator into an OpenAI-shaped
    ``SimpleNamespace`` response (so the agent loop can treat streaming and
    non-streaming Cohere responses identically, same pattern as the Bedrock
    adapter).

All format conversion (messages, tools, request kwargs, response
normalization) lives in :mod:`agent.transports.cohere`. This module owns
only the SDK lifecycle and the streaming event-pump.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

from utils import base_url_host_matches, normalize_proxy_env_vars

logger = logging.getLogger(__name__)

# Lazy SDK accessor — mirrors the pattern in anthropic_adapter._get_anthropic_sdk.
_cohere_sdk: Any = ...  # sentinel — None means "tried and missing"


def _get_cohere_sdk():
    """Return the ``cohere`` SDK module, importing lazily.

    Returns None when the package is not installed and the lazy-install
    path also failed (offline / opted out). Callers must handle None
    with a clear error message.
    """
    global _cohere_sdk
    if _cohere_sdk is ...:
        try:
            from tools.lazy_deps import ensure as _lazy_ensure

            _lazy_ensure("provider.cohere", prompt=False)
        except ImportError:
            pass
        except Exception:
            # FeatureUnavailable — fall through to the ImportError handling below
            pass
        try:
            import cohere as _sdk

            _cohere_sdk = _sdk
        except ImportError:
            _cohere_sdk = None
    return _cohere_sdk


def is_cohere_url(base_url: str | None) -> bool:
    """Return True when *base_url* points at Cohere's public API host."""
    if not base_url:
        return False
    return base_url_host_matches(base_url, "api.cohere.com") or base_url_host_matches(
        base_url, "api.cohere.ai"
    )


def build_cohere_client(
    api_key: str,
    base_url: str | None = None,
    timeout: float | None = None,
):
    """Construct a ``cohere.ClientV2`` instance.

    The Cohere SDK accepts a ``client_name`` for usage telemetry; we tag
    every request with ``hermes-agent`` so Cohere can attribute traffic
    and so users can identify their requests in dashboards.

    Args:
        api_key: Cohere API key. Required — Cohere has no anonymous tier.
        base_url: Optional override (defaults to ``https://api.cohere.com``).
            Useful for staging / proxy / self-hosted Cohere deployments.
        timeout: Read timeout in seconds. ``None`` uses the SDK default.

    Raises:
        ImportError: if the ``cohere`` package is not installed and the
            lazy-install path is unavailable.
    """
    sdk = _get_cohere_sdk()
    if sdk is None:
        raise ImportError(
            "The 'cohere' package is required for the Cohere provider. "
            "Install it with: pip install cohere>=5.13\n"
            "Or install Hermes with Cohere support: pip install -e '.[cohere]'"
        )

    normalize_proxy_env_vars()

    kwargs: Dict[str, Any] = {
        "api_key": api_key,
        "client_name": "hermes-agent",
    }
    if base_url:
        kwargs["base_url"] = base_url
    if isinstance(timeout, (int, float)) and timeout > 0:
        kwargs["timeout"] = float(timeout)

    return sdk.ClientV2(**kwargs)


# ---------------------------------------------------------------------------
# Streaming: turn a chat_stream event iterator into a non-streaming response.
# ---------------------------------------------------------------------------


def _coerce_event(event: Any) -> Dict[str, Any]:
    """Return a dict view of a Cohere stream event.

    The SDK emits typed Pydantic objects; we accept dicts as well so tests
    can drive this with hand-built fixtures. Falls back to ``__dict__`` if
    the object exposes neither ``.model_dump()`` nor ``.dict()``.
    """
    if isinstance(event, dict):
        return event
    for attr in ("model_dump", "dict"):
        fn = getattr(event, attr, None)
        if callable(fn):
            try:
                value = fn()
                if isinstance(value, dict):
                    return value
            except Exception:
                continue
    return getattr(event, "__dict__", {}) or {}


def _event_type(event_dict: Dict[str, Any]) -> str:
    """Extract the canonical event type string from a stream event."""
    t = event_dict.get("type") or event_dict.get("event_type") or ""
    return str(t)


def _nested_get(d: Dict[str, Any], *path: str) -> Any:
    """Walk a nested dict by key path, returning None on the first miss."""
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def stream_chat_with_callbacks(
    stream_iter,
    on_text_delta: Optional[Callable[[str], None]] = None,
    on_tool_start: Optional[Callable[[str], None]] = None,
    on_reasoning_delta: Optional[Callable[[str], None]] = None,
    on_citation: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_interrupt_check: Optional[Callable[[], bool]] = None,
) -> SimpleNamespace:
    """Process a Cohere ``chat_stream`` event iterator with real-time callbacks.

    Mirrors the shape of ``stream_converse_with_callbacks`` in
    ``bedrock_adapter.py``: builds up tool calls, text, reasoning, and
    citations as events arrive, then returns an OpenAI-shaped response
    object so the rest of the agent loop is provider-agnostic.

    Event types we handle (from the Cohere v2 streaming spec):
      - ``message-start``           → ignored (response id available later)
      - ``content-start`` / ``content-end``
      - ``content-delta``           → ``delta.message.content.text`` chunks
      - ``tool-plan-delta``         → ``delta.message.tool_plan`` chunks (reasoning)
      - ``tool-call-start``         → start of a tool call (id + name)
      - ``tool-call-delta``         → ``delta.message.tool_calls.function.arguments``
      - ``tool-call-end``
      - ``citation-start`` / ``citation-end``
      - ``message-end``             → ``delta.finish_reason``, ``delta.usage``

    Args:
        stream_iter: An iterable of stream events from ``ClientV2.chat_stream``.
        on_text_delta: Called with each text chunk while no tool call is in
            flight. Same semantics as the Anthropic and Bedrock paths — once
            a tool call starts, text deltas are buffered but not fired.
        on_tool_start: Called with the tool name when a tool call begins.
        on_reasoning_delta: Called with each tool_plan chunk (Cohere's
            equivalent of Anthropic's thinking deltas).
        on_citation: Called once per emitted citation event with the raw
            citation dict (start/end offsets, sources, text).
        on_interrupt_check: Called on each event; truthy return aborts.
    """
    text_parts: List[str] = []
    reasoning_parts: List[str] = []
    citations: List[Dict[str, Any]] = []
    tool_calls: List[SimpleNamespace] = []
    current_tool: Optional[Dict[str, Any]] = None
    has_tool_use = False
    finish_reason_raw = "COMPLETE"
    usage_data: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
    response_id = ""

    for raw in stream_iter:
        if on_interrupt_check and on_interrupt_check():
            break

        event = _coerce_event(raw)
        etype = _event_type(event)

        if etype == "message-start":
            response_id = str(event.get("id") or _nested_get(event, "message", "id") or "")

        elif etype == "content-delta":
            text = _nested_get(event, "delta", "message", "content", "text")
            if isinstance(text, str) and text:
                text_parts.append(text)
                if on_text_delta and not has_tool_use:
                    try:
                        on_text_delta(text)
                    except Exception:
                        pass

        elif etype == "tool-plan-delta":
            chunk = _nested_get(event, "delta", "message", "tool_plan")
            if isinstance(chunk, str) and chunk:
                reasoning_parts.append(chunk)
                if on_reasoning_delta:
                    try:
                        on_reasoning_delta(chunk)
                    except Exception:
                        pass

        elif etype == "tool-call-start":
            has_tool_use = True
            tool_call = _nested_get(event, "delta", "message", "tool_calls") or {}
            fn = tool_call.get("function") or {}
            current_tool = {
                "id": tool_call.get("id", "") or "",
                "name": fn.get("name", "") or "",
                "arguments": fn.get("arguments", "") or "",
            }
            if on_tool_start and current_tool["name"]:
                try:
                    on_tool_start(current_tool["name"])
                except Exception:
                    pass

        elif etype == "tool-call-delta":
            if current_tool is not None:
                fn = _nested_get(event, "delta", "message", "tool_calls", "function") or {}
                args_chunk = fn.get("arguments")
                if isinstance(args_chunk, str):
                    current_tool["arguments"] += args_chunk

        elif etype == "tool-call-end":
            if current_tool is not None:
                args_str = current_tool["arguments"] or "{}"
                # Validate JSON; if Cohere streamed a malformed args object
                # we still emit the tool call with the raw string so the
                # agent's tool dispatcher can surface a useful error.
                try:
                    json.loads(args_str)
                except (json.JSONDecodeError, TypeError):
                    args_str = current_tool["arguments"] or "{}"
                tool_calls.append(
                    SimpleNamespace(
                        id=current_tool["id"],
                        type="function",
                        function=SimpleNamespace(
                            name=current_tool["name"],
                            arguments=args_str,
                        ),
                    )
                )
                current_tool = None

        elif etype == "citation-start":
            citation = _nested_get(event, "delta", "message", "citations") or {}
            if isinstance(citation, dict) and citation:
                citations.append(citation)
                if on_citation:
                    try:
                        on_citation(citation)
                    except Exception:
                        pass

        elif etype == "message-end":
            fr = _nested_get(event, "delta", "finish_reason")
            if isinstance(fr, str) and fr:
                finish_reason_raw = fr
            tokens = _nested_get(event, "delta", "usage", "tokens") or {}
            if isinstance(tokens, dict):
                usage_data["input_tokens"] = int(tokens.get("input_tokens") or 0)
                usage_data["output_tokens"] = int(tokens.get("output_tokens") or 0)

    # Flush a partial tool call if the stream ended mid-flight — preserves
    # the same lossy-but-useful behaviour the Bedrock adapter has.
    if current_tool is not None:
        tool_calls.append(
            SimpleNamespace(
                id=current_tool.get("id", ""),
                type="function",
                function=SimpleNamespace(
                    name=current_tool.get("name", ""),
                    arguments=current_tool.get("arguments", "") or "{}",
                ),
            )
        )

    finish_reason = _cohere_finish_reason_to_openai(finish_reason_raw)
    if tool_calls and finish_reason == "stop":
        finish_reason = "tool_calls"

    msg = SimpleNamespace(
        role="assistant",
        content="".join(text_parts) if text_parts else None,
        tool_calls=tool_calls if tool_calls else None,
        reasoning_content="".join(reasoning_parts) if reasoning_parts else None,
        citations=citations or None,
        tool_plan="".join(reasoning_parts) if reasoning_parts else None,
    )

    usage = SimpleNamespace(
        prompt_tokens=usage_data["input_tokens"],
        completion_tokens=usage_data["output_tokens"],
        total_tokens=usage_data["input_tokens"] + usage_data["output_tokens"],
    )

    choice = SimpleNamespace(
        index=0,
        message=msg,
        finish_reason=finish_reason,
    )

    return SimpleNamespace(
        id=response_id,
        choices=[choice],
        usage=usage,
        model="",
        finish_reason=finish_reason_raw,
        message=msg,
    )


# ---------------------------------------------------------------------------
# Finish reason mapping — shared between transport and adapter.
# ---------------------------------------------------------------------------

# Cohere finish_reason → OpenAI finish_reason
COHERE_FINISH_REASON_MAP: Dict[str, str] = {
    "COMPLETE": "stop",
    "STOP_SEQUENCE": "stop",
    "MAX_TOKENS": "length",
    "TOOL_CALL": "tool_calls",
    "ERROR": "stop",
    "ERROR_TOXIC": "content_filter",
    "ERROR_LIMIT": "length",
    "USER_CANCEL": "stop",
}


def _cohere_finish_reason_to_openai(raw: str) -> str:
    """Map a Cohere ``finish_reason`` enum value to its OpenAI equivalent."""
    if not raw:
        return "stop"
    return COHERE_FINISH_REASON_MAP.get(str(raw).upper(), "stop")
