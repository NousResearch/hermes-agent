"""One physical OpenAI-wire request for an isolated reasoning job."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Mapping
from types import SimpleNamespace

from agent.codex_runtime import _consume_codex_event_stream, _sanitize_consumer_codex_request
from agent.sdk_transform_bypass import bypass_sdk_request_transform
from agent.chat_completion_helpers import bypass_chat_sdk_request_transform


def _tier(value: Any) -> str | None:
    return None if value in (None, "auto", "default") else str(value)


@dataclass(frozen=True)
class PriceQuote:
    version: str
    provider: str
    model: str
    input_usd_per_million: float
    output_usd_per_million: float
    service_tier: str | None = None

    def __post_init__(self):
        import math
        if not self.version or not self.provider or not self.model or (self.service_tier is not None and not self.service_tier) or any(
            not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0
            for v in (self.input_usd_per_million, self.output_usd_per_million)
        ):
            raise ValueError("versioned, finite route pricing required")


class CompletedUsageInterrupted(TimeoutError):
    """A completed response was interrupted; retain only measured numeric usage."""

    def __init__(self, usage: Mapping[str, int | float | None]):
        super().__init__("reasoning request cancelled or deadline exceeded")
        self.usage = dict(usage)


class CoreSingleAttemptTransport:
    """Use Core's resolved route and request slot, with no Relay or SDK retry.

    ``input_bound`` is supplied by the host and must include the whole wire request,
    including tool schemas. Missing bounds or pricing stop the runner before a send.
    """

    single_attempt = True

    def __init__(self, *, input_bound: Callable[[Mapping[str, Any], Any], int] | None,
                 price: PriceQuote | None, cancelled: Callable[[], bool], deadline_monotonic: float):
        self.input_bound = input_bound
        self.price = price
        self.cancelled = cancelled
        self.deadline = deadline_monotonic
        self._requested_tier: str | None = None

    def supports_effort(self, route) -> bool:
        return route.api_mode in {"chat_completions", "codex_responses"}

    def effective_effort(self, request: Mapping[str, Any], route) -> str | None:
        self._requested_tier = _tier(request.get("service_tier"))
        if route.api_mode == "codex_responses":
            reasoning = request.get("reasoning")
            return reasoning.get("effort") if isinstance(reasoning, dict) else None
        top = request.get("reasoning_effort")
        if isinstance(top, str):
            return top
        body = request.get("extra_body")
        reasoning = body.get("reasoning") if isinstance(body, dict) else None
        return reasoning.get("effort") if isinstance(reasoning, dict) else None

    def input_token_upper_bound(self, request: Mapping[str, Any], route) -> int:
        return self.input_bound(request, route) if self.input_bound is not None else 0

    def _priced(self, input_tokens: int, output_tokens: int, route) -> float | None:
        quote = self.price
        if (quote is None or (quote.provider, quote.model) != (route.provider, route.model)
                or _tier(quote.service_tier) != self._requested_tier):
            return None
        return (input_tokens * quote.input_usd_per_million +
                output_tokens * quote.output_usd_per_million) / 1_000_000

    def cost_upper_bound(self, input_tokens: int, output_tokens: int, route) -> float | None:
        return self._priced(input_tokens, output_tokens, route)

    def actual_cost(self, response: Any, route) -> float | None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        if self.price is None or _tier(getattr(response, "service_tier", None)) != _tier(self.price.service_tier):
            return None
        if route.api_mode == "codex_responses":
            input_tokens, output_tokens = getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None)
        else:
            input_tokens, output_tokens = getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)
        if type(input_tokens) is not int or type(output_tokens) is not int or min(input_tokens, output_tokens) < 0:
            return None
        return self._priced(input_tokens, output_tokens, route)

    def complete(self, request: Mapping[str, Any], resolved_agent: Any) -> Any:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0 or self.cancelled():
            raise TimeoutError("reasoning request expired before dispatch")
        client = resolved_agent._create_request_openai_client(reason="memory_reasoning", api_kwargs=dict(request))
        finished = threading.Event()
        aborted = threading.Event()

        def watch() -> None:
            while not finished.is_set():
                remaining = self.deadline - time.monotonic()
                if self.cancelled() or remaining <= 0:
                    aborted.set()
                    resolved_agent._abort_request_openai_client(client, reason="memory_reasoning_cancel")
                    return
                finished.wait(min(0.02, remaining))

        watcher = threading.Thread(target=watch, name="memory-reasoning-request-watch", daemon=True)
        watcher.start()
        stream = None
        try:
            self._raise_if_stopped(aborted)
            wire = dict(request)
            wire["timeout"] = max(0.001, self.deadline - time.monotonic())
            if resolved_agent.api_mode == "codex_responses":
                if wire.get("context_management"):
                    raise ValueError("automatic Responses compaction forbidden in restricted task")
                wire = _sanitize_consumer_codex_request(resolved_agent, wire)
                wire["stream"] = True
                self._raise_if_stopped(aborted)
                stream = client.responses.create(**bypass_sdk_request_transform(wire))
                result = _consume_codex_event_stream(
                    stream, model=str(wire["model"]),
                    interrupt_check=lambda: self._raise_if_stopped(aborted),
                )
            else:
                self._raise_if_stopped(aborted)
                result = client.chat.completions.create(**bypass_chat_sdk_request_transform(wire, client))
            try:
                self._raise_if_stopped(aborted)
            except TimeoutError:
                usage = getattr(result, "usage", None)
                if resolved_agent.api_mode == "codex_responses":
                    input_tokens = getattr(usage, "input_tokens", None)
                    output_tokens = getattr(usage, "output_tokens", None)
                else:
                    input_tokens = getattr(usage, "prompt_tokens", None)
                    output_tokens = getattr(usage, "completion_tokens", None)
                route = SimpleNamespace(provider=resolved_agent.provider,
                                        model=str(request.get("model") or resolved_agent.model),
                                        api_mode=resolved_agent.api_mode)
                try:
                    cost = self.actual_cost(result, route)
                except Exception:
                    cost = None
                raise CompletedUsageInterrupted({
                    "input_tokens": input_tokens if type(input_tokens) is int and input_tokens >= 0 else None,
                    "output_tokens": output_tokens if type(output_tokens) is int and output_tokens >= 0 else None,
                    "cost_usd": cost if type(cost) in (int, float) else None,
                }) from None
            return result
        finally:
            finished.set()
            watcher.join(timeout=0.1)
            try:
                if stream is not None:
                    stream.close()
            finally:
                resolved_agent._close_request_openai_client(client, reason="memory_reasoning_complete")

    def _raise_if_stopped(self, aborted: threading.Event) -> bool:
        if aborted.is_set() or self.cancelled() or time.monotonic() >= self.deadline:
            raise TimeoutError("reasoning request cancelled or deadline exceeded")
        return False
