# SPDX-License-Identifier: MIT
"""OTel GenAI span emitter for the observability/otel plugin.

This module is intentionally **Hermes-agnostic**: it knows nothing about the
Hermes plugin contract (no ``register(ctx)``, no hook names, no kwargs shape).
It only knows how to turn the three observability concepts Hermes already
produces — *turn*, *LLM API call*, *tool call* — into OpenTelemetry spans that
follow the GenAI semantic conventions (``gen_ai.operation.name`` of
``invoke_agent`` / ``chat`` / ``execute_tool``).

Keeping it separate means the span-mapping logic is unit-testable with only the
OpenTelemetry SDK installed (no Hermes import), and the plugin glue in
``__init__.py`` stays a thin adapter over the real Hermes hooks.

Span model:

    invoke_agent   one span per Hermes turn  (gen_ai.conversation.id == session)
      |
      +-- chat           one span per LLM API call (tokens, cost, finish reason)
      |
      +-- execute_tool   one span per tool call   (tool name, type)

GenAI attribute conventions emitted (the names the OTel GenAI spec defines):

    gen_ai.operation.name            invoke_agent | chat | execute_tool
    gen_ai.system                    hermes
    gen_ai.agent.name                hermes
    gen_ai.conversation.id           session / conversation id
    gen_ai.request.model             model id requested
    gen_ai.response.model            model id that answered
    gen_ai.response.finish_reasons   [str]  (stop / length / tool_calls / ...)
    gen_ai.usage.input_tokens        int
    gen_ai.usage.output_tokens       int
    gen_ai.tool.name                 tool name (execute_tool span)
    gen_ai.tool.type                 function | extension

Extras (interoperable, safe no-ops on any conventions-aware backend):

    cost_usd                         float   (estimated, if available)
    copilot_chat.time_to_first_token TTFT ms (shared TTFT field)
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence

logger = logging.getLogger(__name__)

# OTel GenAI operation names (stable across the spec).
OP_INVOKE_AGENT = "invoke_agent"
OP_CHAT = "chat"
OP_EXECUTE_TOOL = "execute_tool"

# Stable identity for the contributed agent.
GEN_AI_SYSTEM = "hermes"
DEFAULT_AGENT_NAME = "hermes"


def _set_if(span: Any, key: str, value: Any) -> None:
    """Set an attribute only when ``value`` is meaningful.

    OTel rejects ``None`` and silently drops empty collections; skipping them
    keeps spans clean and avoids per-call ``if`` noise at the call sites.
    """
    if value is None:
        return
    if isinstance(value, (str, bytes)) and len(value) == 0:
        return
    span.set_attribute(key, value)


class OtelGenAIEmitter:
    """Translate Hermes observability events into OTel GenAI spans.

    Parameters
    ----------
    tracer:
        An OpenTelemetry ``Tracer``. Injected so tests can pass an in-memory
        tracer and the plugin can pass the globally-configured one.
    agent_name:
        Value for ``gen_ai.agent.name`` (defaults to ``hermes``).
    capture_content:
        When true, prompt/response text is attached to spans (gated because
        content capture has privacy implications — off by default).
    """

    def __init__(
        self,
        tracer: Any,
        *,
        agent_name: str = DEFAULT_AGENT_NAME,
        capture_content: bool = False,
    ) -> None:
        self._tracer = tracer
        self._agent_name = agent_name
        self._capture_content = capture_content

    # -- turn / agent invocation -------------------------------------------

    @contextmanager
    def turn_span(
        self,
        *,
        session_id: str | None,
        model: str | None = None,
        user_prompt: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Iterator[Any]:
        """Open the ``invoke_agent`` span for one Hermes turn.

        This is the session-boundary span. ``gen_ai.conversation.id`` carries
        the Hermes session id so any backend can group spans by session.
        """
        with self._tracer.start_as_current_span(OP_INVOKE_AGENT) as span:
            span.set_attribute("gen_ai.operation.name", OP_INVOKE_AGENT)
            span.set_attribute("gen_ai.system", GEN_AI_SYSTEM)
            span.set_attribute("gen_ai.agent.name", self._agent_name)
            _set_if(span, "gen_ai.conversation.id", session_id)
            # Belt-and-suspenders: also stamp the bare session.id attribute
            # some backends group on directly.
            _set_if(span, "session.id", session_id)
            _set_if(span, "gen_ai.request.model", model)
            if self._capture_content:
                _set_if(span, "gen_ai.prompt", user_prompt)
            self._apply_extra(span, extra)
            yield span

    def stamp_completion(self, span: Any, response_text: str | None) -> None:
        """Attach the model's response to an already-open ``invoke_agent`` span.

        Hermes delivers the final response text on ``transform_llm_output``
        (fired once per turn after the tool loop, before the turn closes) — the
        per-call ``post_api_request`` hook carries only token/finish metadata,
        not the text. Stamping it on the still-open turn span enables
        response-side evaluation (toxicity / hallucination / output PII)
        alongside the prompt-side eval. Content-gated; a no-op on a
        missing/empty response, so it is safe to call unconditionally.
        """
        if self._capture_content:
            _set_if(span, "gen_ai.completion", response_text)

    # -- LLM call -----------------------------------------------------------

    def record_llm_call(
        self,
        *,
        request_model: str | None,
        response_model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost_usd: float | None = None,
        finish_reasons: Sequence[str] | None = None,
        ttft_ms: float | None = None,
        response_text: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Emit one ``chat`` span for a single LLM API call.

        Nested under the active ``invoke_agent`` span when called inside
        :meth:`turn_span`. Carries tokens, cost, finish reason, and TTFT.
        """
        with self._tracer.start_as_current_span(OP_CHAT) as span:
            span.set_attribute("gen_ai.operation.name", OP_CHAT)
            span.set_attribute("gen_ai.system", GEN_AI_SYSTEM)
            _set_if(span, "gen_ai.request.model", request_model)
            _set_if(span, "gen_ai.response.model", response_model or request_model)
            _set_if(span, "gen_ai.usage.input_tokens", _as_int(input_tokens))
            _set_if(span, "gen_ai.usage.output_tokens", _as_int(output_tokens))
            _set_if(span, "cost_usd", _as_float(cost_usd))
            if finish_reasons:
                # OTel models this as an array; backends read index [0].
                span.set_attribute(
                    "gen_ai.response.finish_reasons", list(finish_reasons)
                )
            # Map onto the shared TTFT field used across agent-coding telemetry
            # (copilot_chat.time_to_first_token).
            _set_if(span, "copilot_chat.time_to_first_token", _as_float(ttft_ms))
            if self._capture_content:
                _set_if(span, "gen_ai.completion", response_text)
            self._apply_extra(span, extra)

    # -- tool call ----------------------------------------------------------

    def record_tool_call(
        self,
        *,
        tool_name: str,
        tool_type: str = "function",
        arguments: Mapping[str, Any] | None = None,
        result: Any = None,
        error: BaseException | str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Emit one ``execute_tool`` span for a single tool call.

        ``gen_ai.tool.name`` drives a per-session tool breakdown. Errors are
        recorded on the span (not swallowed) so failed tool calls are visible.
        """
        with self._tracer.start_as_current_span(OP_EXECUTE_TOOL) as span:
            span.set_attribute("gen_ai.operation.name", OP_EXECUTE_TOOL)
            span.set_attribute("gen_ai.system", GEN_AI_SYSTEM)
            span.set_attribute("gen_ai.tool.name", tool_name)
            span.set_attribute("gen_ai.tool.type", tool_type)
            if self._capture_content and arguments is not None:
                # Best-effort: stringify so non-serializable args never crash
                # observability (observability must never break the agent).
                _set_if(span, "gen_ai.tool.arguments", _safe_str(arguments))
            if self._capture_content and result is not None:
                _set_if(span, "gen_ai.tool.result", _safe_str(result))
            if error is not None:
                self._record_error(span, error)
            self._apply_extra(span, extra)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _record_error(span: Any, error: BaseException | str) -> None:
        try:
            if isinstance(error, BaseException):
                span.record_exception(error)
                span.set_attribute("error.type", type(error).__name__)
            else:
                span.set_attribute("error.type", str(error))
        except Exception:  # pragma: no cover - defensive
            logger.warning("failed to record tool error on span", exc_info=True)

    @staticmethod
    def _apply_extra(span: Any, extra: Mapping[str, Any] | None) -> None:
        if not extra:
            return
        for key, value in extra.items():
            try:
                _set_if(span, key, value)
            except Exception:  # pragma: no cover - defensive
                logger.warning("dropping un-settable span attribute %s", key)


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_str(value: Any, limit: int = 8192) -> str:
    try:
        text = value if isinstance(value, str) else repr(value)
    except Exception:  # pragma: no cover - defensive
        return "<unrepr-able>"
    return text if len(text) <= limit else text[:limit] + "...<truncated>"
