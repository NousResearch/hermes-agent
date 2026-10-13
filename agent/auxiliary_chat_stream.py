"""Accumulate native chat-stream facts for auxiliary calls."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Optional

_REASONING_DETAIL_TEXT_FIELDS = ("summary", "thinking", "content", "text")


class _ChatStreamAccumulator:
    """Shared per-chunk accumulation so sync and async aggregation cannot drift."""

    def __init__(
        self,
        model: str = "",
        total_ceiling: Optional[float] = None,
        host_deadline: Optional[float] = None,
    ):
        self._started = time.monotonic()
        self._total_ceiling = total_ceiling
        # Absolute instant the waiting host gives up; checked alongside (not instead of) the
        # ceiling, and unaffected by pre-construction dispatch/TTFT.
        self._host_deadline = host_deadline
        self.refusal_parts: list[str] = []
        self.content_parts: list[str] = []
        self.reasoning_parts: list[str] = []
        self.reasoning_details: list[Any] = []
        self.tool_calls_acc: dict[int, dict[str, Any]] = {}
        self.finish_reason = self.usage = None
        self.resp_id = None
        self.resp_model = None
        self.native = None

    def _check_deadlines(self) -> None:
        """Raise TimeoutError past the total ceiling or the host deadline."""
        now = time.monotonic()
        if (
            self._total_ceiling is not None
            and (now - self._started) >= self._total_ceiling
        ):
            raise TimeoutError(
                f"Auxiliary streamed call timed out after {self._total_ceiling:.0f}s "
                "total ceiling (stream still open but over budget)"
            )
        if self._host_deadline is not None and now >= self._host_deadline:
            raise TimeoutError(
                "Auxiliary streamed call timed out at the host compression "
                f"deadline after {time.monotonic() - self._started:.0f}s "
                "(the caller already stopped waiting; streaming on would only "
                "pin its session lease)"
            )

    def _feed_reasoning_details(self, delta: Any) -> bool:
        """Collect ``reasoning_details`` (OpenRouter-style thinking); True only when a detail
        carries text, so structural/signed envelopes can't keep a stall alive."""
        reasoning_details = getattr(delta, "reasoning_details", None)
        if reasoning_details is None:
            model_extra = getattr(delta, "model_extra", None)
            if isinstance(model_extra, dict):
                reasoning_details = model_extra.get("reasoning_details")
        if not isinstance(reasoning_details, list):
            return False
        made_progress = False
        for detail in reasoning_details:
            self.reasoning_details.append(detail)
            if isinstance(detail, dict) and any(
                isinstance(detail.get(f), str) and detail[f]
                for f in _REASONING_DETAIL_TEXT_FIELDS
            ):
                made_progress = True
        return made_progress

    def _feed_tool_calls(self, delta: Any) -> bool:
        """Merge tool-call fragments by index; True when any fragment carried data."""
        made_progress = False
        for tc in getattr(delta, "tool_calls", None) or []:
            idx = getattr(tc, "index", 0) or 0
            acc = self.tool_calls_acc.setdefault(
                idx, {"id": "", "name": "", "arguments": []}
            )
            if getattr(tc, "id", None):
                acc["id"] = tc.id
                made_progress = True
            fn = getattr(tc, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    acc["name"] = fn.name
                    made_progress = True
                if getattr(fn, "arguments", None):
                    acc["arguments"].append(fn.arguments)
                    made_progress = True
        return made_progress

    def feed(self, chunk: Any) -> bool:
        # Every frame records transport timing (TTFP); only a substantive payload ticks the
        # forward-progress hook that keeps compression alive (and is reported as True).
        from agent import auxiliary_client

        auxiliary_client._notify_aux_timing_response()
        self._check_deadlines()
        self.resp_id = getattr(chunk, "id", None) or self.resp_id
        self.resp_model = getattr(chunk, "model", None) or self.resp_model
        self.native = getattr(chunk, "native", None) or self.native
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage:
            self.usage = chunk_usage
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return False
        choice = choices[0]
        self.finish_reason = (
            getattr(choice, "finish_reason", None) or self.finish_reason
        )
        delta = getattr(choice, "delta", None)
        if delta is None:
            return False
        made_progress = False
        from agent.message_content import flatten_message_text

        piece = flatten_message_text(getattr(delta, "content", None), sep="")
        if piece:
            self.content_parts.append(piece)
            made_progress = True
        refusal_piece = getattr(delta, "refusal", None)
        if refusal_piece:
            self.refusal_parts.append(refusal_piece)
            made_progress = True
        reasoning_piece = getattr(delta, "reasoning", None) or getattr(
            delta, "reasoning_content", None
        )
        if reasoning_piece is None and isinstance(
            getattr(delta, "model_extra", None), dict
        ):
            reasoning_piece = delta.model_extra.get(
                "reasoning"
            ) or delta.model_extra.get("reasoning_content")
        reasoning_piece = flatten_message_text(reasoning_piece, sep="")
        if reasoning_piece:
            self.reasoning_parts.append(reasoning_piece)
            made_progress = True
        # Evaluate both unconditionally: they accumulate state, not just progress.
        made_progress |= self._feed_reasoning_details(delta)
        made_progress |= self._feed_tool_calls(delta)
        if made_progress:
            auxiliary_client._notify_aux_progress()
        return made_progress

    def finish(self) -> Any:
        tool_calls = None
        if self.tool_calls_acc and not (
            self.finish_reason == "length"
            and getattr(self.native, "finish_reason", None) == "MAX_TOKENS"
        ):
            tool_calls = [
                SimpleNamespace(
                    id=acc["id"],
                    type="function",
                    function=SimpleNamespace(
                        name=acc["name"], arguments="".join(acc["arguments"])
                    ),
                )
                for _idx, acc in sorted(self.tool_calls_acc.items())
            ]
        message = SimpleNamespace(
            role="assistant",
            content="".join(self.content_parts),
            tool_calls=tool_calls,
            reasoning="".join(self.reasoning_parts) or None,
            reasoning_details=self.reasoning_details or None,
            **({"refusal": "".join(self.refusal_parts)} if self.refusal_parts else {}),
        )
        choice = SimpleNamespace(
            index=0, message=message, finish_reason=self.finish_reason
        )
        return SimpleNamespace(
            id=self.resp_id,
            model=self.resp_model,
            object="chat.completion",
            choices=[choice],
            usage=self.usage,
            **({"native": self.native} if self.native is not None else {}),
        )
