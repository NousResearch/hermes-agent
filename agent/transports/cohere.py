"""Cohere v2 chat transport.

Owns format conversion and response normalization for ``api_mode="cohere_chat"``.
Client construction, streaming, and credential handling live elsewhere:

  - SDK lifecycle + streaming pump → :mod:`agent.cohere_adapter`
  - Client construction + retry/timeout policy → :mod:`run_agent`

The transport bridges Hermes's internal OpenAI-shaped message/tool format
to Cohere's native ``ClientV2.chat`` request shape and back. Cohere v2 is
close to OpenAI's chat schema, with three notable deltas:

  1. Tool result messages carry a list of ``document`` content blocks
     instead of a plain string.
  2. Responses surface ``message.tool_plan`` (chain-of-thought) and
     ``message.citations`` (RAG grounding) in addition to ``content`` and
     ``tool_calls``. We map ``tool_plan`` → ``reasoning`` and stash
     ``citations`` in ``provider_data``.
  3. ``finish_reason`` uses Cohere's enum (``COMPLETE``/``TOOL_CALL``/...).
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Dict, List, Optional

from agent.transports.base import ProviderTransport
from agent.transports.types import NormalizedResponse, ToolCall, Usage

logger = logging.getLogger(__name__)


# OpenAI fields on tool schemas that Cohere's v2 chat endpoint does not
# accept. Stripping them keeps callers that pass through the canonical
# OpenAI-shaped tool list working without per-call manual scrubbing.
_TOOL_FIELDS_TO_STRIP = ("strict", "cache_control")


def _strip_tool_fields(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a deep-copied tool list with Cohere-incompatible fields removed."""
    cleaned: List[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        t = copy.deepcopy(tool)
        for field_name in _TOOL_FIELDS_TO_STRIP:
            t.pop(field_name, None)
            fn = t.get("function")
            if isinstance(fn, dict):
                fn.pop(field_name, None)
        cleaned.append(t)
    return cleaned


def _coerce_tool_result_content(content: Any) -> List[Dict[str, Any]]:
    """Normalize an OpenAI tool-message ``content`` field to Cohere's shape.

    Cohere expects tool results as a list of ``{"type": "document",
    "document": {"data": <str>}}`` blocks. OpenAI tool messages carry a
    plain string (or sometimes a list of content parts when the producer
    is gemini-via-openai). We accept both and emit the Cohere shape.
    """
    if isinstance(content, str):
        data = content
    elif isinstance(content, list):
        # Best-effort flattening of OpenAI content-parts → a single string.
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(part.get("data"), str):
                    parts.append(part["data"])
        data = "\n".join(parts)
    elif content is None:
        data = ""
    else:
        try:
            data = json.dumps(content, ensure_ascii=False)
        except (TypeError, ValueError):
            data = str(content)
    return [{"type": "document", "document": {"data": data}}]


def _convert_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a single OpenAI-shape message to Cohere v2."""
    role = msg.get("role")
    if role == "tool":
        # tool_call_id + content (string or content parts) → Cohere doc blocks
        return {
            "role": "tool",
            "tool_call_id": msg.get("tool_call_id") or msg.get("id") or "",
            "content": _coerce_tool_result_content(msg.get("content")),
        }

    if role == "assistant" and msg.get("tool_calls"):
        # Re-emit assistant tool-call turns in Cohere's shape. Tool plan
        # (reasoning) is preserved when the caller stored it on the
        # message via the canonical ``reasoning`` field.
        out: Dict[str, Any] = {"role": "assistant"}
        content = msg.get("content")
        if isinstance(content, str) and content:
            out["content"] = content
        tool_plan = msg.get("reasoning") or msg.get("tool_plan")
        if isinstance(tool_plan, str) and tool_plan:
            out["tool_plan"] = tool_plan
        out_tool_calls: List[Dict[str, Any]] = []
        for tc in msg["tool_calls"]:
            if not isinstance(tc, dict):
                tc_id = getattr(tc, "id", "") or ""
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", "") if fn is not None else ""
                arguments = getattr(fn, "arguments", "") if fn is not None else ""
            else:
                tc_id = tc.get("id") or ""
                fn = tc.get("function") or {}
                name = fn.get("name", "") if isinstance(fn, dict) else ""
                arguments = fn.get("arguments", "") if isinstance(fn, dict) else ""
            if isinstance(arguments, (dict, list)):
                arguments = json.dumps(arguments)
            out_tool_calls.append(
                {
                    "id": tc_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments or "{}"},
                }
            )
        out["tool_calls"] = out_tool_calls
        return out

    # System / user / assistant (text-only) — pass through. Strip
    # OpenAI-specific extras Cohere ignores or rejects.
    cleaned: Dict[str, Any] = {"role": role}
    content = msg.get("content")
    if content is not None:
        cleaned["content"] = content
    name = msg.get("name")
    if isinstance(name, str) and name:
        cleaned["name"] = name
    return cleaned


class CohereTransport(ProviderTransport):
    """Transport for ``api_mode="cohere_chat"``."""

    @property
    def api_mode(self) -> str:
        return "cohere_chat"

    # ── Message + tool conversion ─────────────────────────────────

    def convert_messages(
        self, messages: List[Dict[str, Any]], **kwargs
    ) -> List[Dict[str, Any]]:
        """Translate OpenAI-shape messages to Cohere v2 chat shape.

        Empty / None messages are dropped to keep Cohere from rejecting
        empty assistant turns produced by upstream sanitisation.
        """
        out: List[Dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            converted = _convert_message(msg)
            out.append(converted)
        return out

    def convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cohere v2 accepts the OpenAI function-tool shape — strip extras only."""
        if not tools:
            return []
        return _strip_tool_fields(tools)

    # ── Request kwargs ────────────────────────────────────────────

    def build_kwargs(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **params,
    ) -> Dict[str, Any]:
        """Assemble ``ClientV2.chat`` kwargs.

        Recognised params (all optional):
            max_tokens: int
            temperature: float
            top_p: float
            stop_sequences: list[str]
            reasoning_config: dict | None — Hermes generic reasoning knob
            tool_choice: "auto" | "any" | "none"
            provider_profile: ProviderProfile | None — drives extra_body
            # Cohere-only knobs, may also flow via provider_profile.build_extra_body
            safety_mode: str
            citation_options: dict
            documents: list
            connectors: list
            force_single_step: bool
            thinking_token_budget: int | None
        """
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": self.convert_messages(messages),
        }

        converted_tools = self.convert_tools(tools or [])
        if converted_tools:
            kwargs["tools"] = converted_tools

        max_tokens = params.get("max_tokens")
        if isinstance(max_tokens, int) and max_tokens > 0:
            kwargs["max_tokens"] = max_tokens

        temperature = params.get("temperature")
        if isinstance(temperature, (int, float)):
            kwargs["temperature"] = float(temperature)

        top_p = params.get("top_p")
        if isinstance(top_p, (int, float)):
            kwargs["p"] = float(top_p)

        stop_sequences = params.get("stop_sequences")
        if isinstance(stop_sequences, list) and stop_sequences:
            kwargs["stop_sequences"] = [str(s) for s in stop_sequences]

        tool_choice = params.get("tool_choice")
        if isinstance(tool_choice, str) and tool_choice in {"auto", "any", "none"}:
            kwargs["tool_choice"] = tool_choice.upper() if tool_choice != "auto" else "AUTO"

        # ── Cohere-native knobs via provider profile ──────────────
        profile = params.get("provider_profile")
        extra_body: Dict[str, Any] = {}
        if profile is not None:
            try:
                profile_body = profile.build_extra_body(
                    session_id=params.get("session_id"),
                    model=model,
                    reasoning_config=params.get("reasoning_config"),
                    safety_mode=params.get("safety_mode"),
                    citation_options=params.get("citation_options"),
                    documents=params.get("documents"),
                    connectors=params.get("connectors"),
                    force_single_step=params.get("force_single_step"),
                    thinking_token_budget=params.get("thinking_token_budget"),
                )
                if isinstance(profile_body, dict):
                    extra_body.update(profile_body)
            except Exception as exc:
                logger.debug("CohereProfile.build_extra_body failed: %s", exc)

        # Allow direct override from params even when the profile path
        # isn't taken (e.g. unit tests that exercise the transport in
        # isolation).
        for key in (
            "safety_mode",
            "citation_options",
            "documents",
            "connectors",
            "force_single_step",
            "thinking",
        ):
            if key in extra_body:
                continue
            value = params.get(key)
            if value is None:
                continue
            if key == "force_single_step":
                if value is True:
                    extra_body["force_single_step"] = True
            elif key == "safety_mode":
                if isinstance(value, str) and value.strip():
                    extra_body["safety_mode"] = value.strip().upper()
            elif key in {"citation_options"}:
                if isinstance(value, dict) and value:
                    extra_body[key] = value
            elif key in {"documents", "connectors"}:
                if isinstance(value, list) and value:
                    extra_body[key] = value
            elif key == "thinking":
                if isinstance(value, dict) and value:
                    extra_body[key] = value

        # Apply Cohere-native fields as top-level kwargs (the SDK uses
        # these names on ClientV2.chat directly).
        for key, value in extra_body.items():
            kwargs[key] = value

        return kwargs

    # ── Response normalization ────────────────────────────────────

    def normalize_response(self, response: Any, **kwargs) -> NormalizedResponse:
        """Translate a Cohere ``ClientV2.chat`` response to ``NormalizedResponse``."""
        message = getattr(response, "message", None)

        # Text content — Cohere returns content as a list of typed blocks
        # ([{type: "text", text: "..."}] for assistant text, possibly
        # multiple blocks when citations chunk the response).
        text_parts: List[str] = []
        if message is not None:
            content_blocks = getattr(message, "content", None)
            if isinstance(content_blocks, list):
                for block in content_blocks:
                    block_type = getattr(block, "type", None) or (
                        block.get("type") if isinstance(block, dict) else None
                    )
                    if block_type == "text":
                        text = getattr(block, "text", None)
                        if text is None and isinstance(block, dict):
                            text = block.get("text")
                        if isinstance(text, str):
                            text_parts.append(text)
            elif isinstance(content_blocks, str):
                text_parts.append(content_blocks)

        # Reasoning — Cohere's tool_plan is the model's chain of thought
        # before issuing tool calls. Surfacing it as `reasoning` lines up
        # with how Anthropic thinking blocks are exposed.
        reasoning_text: Optional[str] = None
        if message is not None:
            tp = getattr(message, "tool_plan", None)
            if isinstance(tp, str) and tp:
                reasoning_text = tp

        # Tool calls
        tool_calls: List[ToolCall] = []
        if message is not None:
            raw_tool_calls = getattr(message, "tool_calls", None) or []
            for tc in raw_tool_calls:
                tc_id = getattr(tc, "id", None) or (
                    tc.get("id") if isinstance(tc, dict) else None
                )
                fn = getattr(tc, "function", None)
                if fn is None and isinstance(tc, dict):
                    fn = tc.get("function")
                name = ""
                arguments: Any = ""
                if fn is not None:
                    name = getattr(fn, "name", None) or (
                        fn.get("name") if isinstance(fn, dict) else ""
                    ) or ""
                    arguments = getattr(fn, "arguments", None)
                    if arguments is None and isinstance(fn, dict):
                        arguments = fn.get("arguments")
                if isinstance(arguments, (dict, list)):
                    arguments_str = json.dumps(arguments)
                else:
                    arguments_str = arguments or "{}"
                tool_calls.append(
                    ToolCall(
                        id=tc_id,
                        name=name,
                        arguments=arguments_str,
                    )
                )

        # Citations — RAG grounding metadata. Stash in provider_data so
        # protocol-aware consumers can surface them without polluting
        # the shared NormalizedResponse surface.
        citations: List[Any] = []
        if message is not None:
            raw_citations = getattr(message, "citations", None)
            if isinstance(raw_citations, list):
                for c in raw_citations:
                    if hasattr(c, "model_dump") and callable(c.model_dump):
                        try:
                            citations.append(c.model_dump())
                            continue
                        except Exception:
                            pass
                    if isinstance(c, dict):
                        citations.append(c)
                    else:
                        citations.append(getattr(c, "__dict__", {}) or {})

        # Finish reason
        raw_finish = getattr(response, "finish_reason", None) or ""
        from agent.cohere_adapter import _cohere_finish_reason_to_openai

        finish_reason = _cohere_finish_reason_to_openai(raw_finish)
        if tool_calls and finish_reason == "stop":
            finish_reason = "tool_calls"

        # Usage
        usage: Optional[Usage] = None
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            tokens = getattr(raw_usage, "tokens", None) or raw_usage
            input_tokens = (
                getattr(tokens, "input_tokens", None)
                if not isinstance(tokens, dict)
                else tokens.get("input_tokens")
            )
            output_tokens = (
                getattr(tokens, "output_tokens", None)
                if not isinstance(tokens, dict)
                else tokens.get("output_tokens")
            )
            if input_tokens is not None or output_tokens is not None:
                input_n = int(input_tokens or 0)
                output_n = int(output_tokens or 0)
                usage = Usage(
                    prompt_tokens=input_n,
                    completion_tokens=output_n,
                    total_tokens=input_n + output_n,
                )

        provider_data: Dict[str, Any] = {}
        if citations:
            provider_data["citations"] = citations
        if reasoning_text:
            # Mirror what other transports do so reasoning_content is
            # available via NormalizedResponse.reasoning_content.
            provider_data["reasoning_content"] = reasoning_text

        return NormalizedResponse(
            content="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls or None,
            finish_reason=finish_reason,
            reasoning=reasoning_text,
            usage=usage,
            provider_data=provider_data or None,
        )

    def validate_response(self, response: Any) -> bool:
        """Empty text is valid when the model is handing off to tools."""
        if response is None:
            return False
        message = getattr(response, "message", None)
        if message is None:
            return False
        # Tool-call turns may have no text content; assistant text turns
        # must have non-empty content (mirrors Anthropic's behaviour).
        content_blocks = getattr(message, "content", None)
        has_text = False
        if isinstance(content_blocks, list):
            for block in content_blocks:
                btype = getattr(block, "type", None) or (
                    block.get("type") if isinstance(block, dict) else None
                )
                if btype == "text":
                    text = getattr(block, "text", None)
                    if text is None and isinstance(block, dict):
                        text = block.get("text")
                    if isinstance(text, str) and text.strip():
                        has_text = True
                        break
        elif isinstance(content_blocks, str) and content_blocks.strip():
            has_text = True

        if has_text:
            return True

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            return True

        raw_finish = str(getattr(response, "finish_reason", "") or "").upper()
        return raw_finish == "TOOL_CALL"

    def extract_cache_stats(self, response: Any) -> Optional[Dict[str, int]]:
        """Cohere does not expose prompt-cache statistics."""
        return None

    def map_finish_reason(self, raw_reason: str) -> str:
        """Map Cohere's finish_reason enum to the OpenAI vocabulary."""
        from agent.cohere_adapter import _cohere_finish_reason_to_openai

        return _cohere_finish_reason_to_openai(raw_reason)


# Auto-register on import
from agent.transports import register_transport  # noqa: E402

register_transport("cohere_chat", CohereTransport)
