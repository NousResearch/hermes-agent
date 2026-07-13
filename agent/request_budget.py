"""Coarse per-turn request budget diagnostics.

The metrics intentionally contain timing and payload-size buckets only. Request
contents, tool arguments, prompts, and model responses never enter the log.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
import json
import logging
import time
from typing import Any


def _ceil_tokens_from_bytes(byte_count: int) -> int:
    if byte_count <= 0:
        return 0
    return (byte_count + 3) // 4


def _json_bytes(value: Any) -> int:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except Exception:
        rendered = str(value)
    return len(rendered.encode("utf-8"))


def estimate_tool_schema_tokens(tools: Any) -> tuple[int, int, int]:
    """Return ``(schema_count, estimated_tokens, byte_count)``."""
    if not isinstance(tools, list) or not tools:
        return 0, 0, 0
    byte_count = _json_bytes(tools)
    return len(tools), _ceil_tokens_from_bytes(byte_count), byte_count


def estimate_skill_index_tokens(prompt: str | None) -> int:
    """Estimate the rendered skills-index weight inside a system prompt."""
    if not prompt:
        return 0
    marker = "## Skills (mandatory)"
    start = prompt.find(marker)
    if start < 0:
        return 0
    tail = prompt[start:]
    end_marker = "Only proceed without loading a skill"
    end = tail.find(end_marker)
    if end >= 0:
        tail = tail[: end + len(end_marker)]
    return _ceil_tokens_from_bytes(len(tail.encode("utf-8")))


@dataclass
class RequestBudget:
    session_id: str
    turn_id: str
    model: str
    provider: str
    platform: str
    started_at: float = field(default_factory=time.perf_counter)
    tool_schema_count: int = 0
    tool_schema_tokens: int = 0
    tool_schema_bytes: int = 0
    skill_index_tokens: int = 0
    skill_index_build_ms: int = 0
    model_ttfb_ms: int | None = None
    model_ttfb_source: str | None = None
    model_ttfb_max_ms: int | None = None
    model_request_ms: int = 0
    model_call_count: int = 0
    tool_execution_ms: int = 0
    tool_call_count: int = 0
    tool_names: list[str] = field(default_factory=list)

    _active_model_start: float | None = field(default=None, init=False, repr=False)
    _active_first_byte_at: float | None = field(default=None, init=False, repr=False)

    def record_tool_schema(self, tools: Any) -> None:
        count, tokens, byte_count = estimate_tool_schema_tokens(tools)
        self.tool_schema_count = count
        self.tool_schema_tokens = tokens
        self.tool_schema_bytes = byte_count

    def record_skill_index(
        self,
        prompt: str | None,
        *,
        build_ms: float | int | None = None,
    ) -> None:
        self.skill_index_tokens = estimate_skill_index_tokens(prompt)
        if build_ms is not None:
            self.skill_index_build_ms += max(0, int(round(float(build_ms))))

    def mark_model_request_start(self) -> None:
        self.model_call_count += 1
        self._active_model_start = time.perf_counter()
        self._active_first_byte_at = None

    def mark_model_first_byte(self, *, source: str = "stream_delta") -> None:
        if self._active_model_start is None or self._active_first_byte_at is not None:
            return
        self._active_first_byte_at = time.perf_counter()
        ttfb_ms = max(
            0,
            int(round((self._active_first_byte_at - self._active_model_start) * 1000)),
        )
        if self.model_ttfb_ms is None:
            self.model_ttfb_ms = ttfb_ms
            self.model_ttfb_source = source
        if self.model_ttfb_max_ms is None or ttfb_ms > self.model_ttfb_max_ms:
            self.model_ttfb_max_ms = ttfb_ms

    def mark_model_request_end(self) -> None:
        if self._active_model_start is None:
            return
        if self._active_first_byte_at is None:
            self.mark_model_first_byte(source="response_complete")
        elapsed_ms = max(
            0,
            int(round((time.perf_counter() - self._active_model_start) * 1000)),
        )
        self.model_request_ms += elapsed_ms
        self._active_model_start = None
        self._active_first_byte_at = None

    def add_tool_execution(
        self,
        tool_names: Iterable[str],
        duration_s: float,
    ) -> None:
        names = [str(name) for name in tool_names if name]
        self.tool_execution_ms += max(0, int(round(duration_s * 1000)))
        self.tool_call_count += len(names)
        self.tool_names.extend(names)

    def snapshot(self, *, reason: str, api_calls: int) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "model": self.model,
            "provider": self.provider,
            "platform": self.platform,
            "reason": reason,
            "api_calls": api_calls,
            "total_ms": max(
                0, int(round((time.perf_counter() - self.started_at) * 1000))
            ),
            "model_ttfb_ms": self.model_ttfb_ms,
            "model_ttfb_source": self.model_ttfb_source,
            "model_ttfb_max_ms": self.model_ttfb_max_ms,
            "model_request_ms": self.model_request_ms,
            "model_call_count": self.model_call_count,
            "tool_schema_count": self.tool_schema_count,
            "tool_schema_tokens": self.tool_schema_tokens,
            "tool_schema_bytes": self.tool_schema_bytes,
            "skill_index_tokens": self.skill_index_tokens,
            "skill_index_build_ms": self.skill_index_build_ms,
            "tool_execution_ms": self.tool_execution_ms,
            "tool_call_count": self.tool_call_count,
            "tool_names": list(dict.fromkeys(self.tool_names)),
        }

    def log_agent_turn(
        self,
        *,
        logger: logging.Logger,
        reason: str,
        api_calls: int,
    ) -> dict[str, Any]:
        payload = self.snapshot(reason=reason, api_calls=api_calls)
        logger.info(
            "request_budget.v1 %s",
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        return payload


def log_gateway_delivery_budget(
    *,
    logger: logging.Logger,
    platform: str,
    chat_id: str | None,
    session_key: str | None,
    gateway_delivery_ms: float,
    response_chars: int,
    delivery_succeeded: bool,
    delivery_kind: str,
) -> dict[str, Any]:
    payload = {
        "platform": platform,
        "chat_id": chat_id or "",
        "session_key": session_key or "",
        "delivery_kind": delivery_kind,
        "gateway_delivery_ms": max(0, int(round(gateway_delivery_ms))),
        "response_chars": max(0, int(response_chars)),
        "delivery_succeeded": bool(delivery_succeeded),
    }
    logger.info(
        "request_budget.gateway_delivery.v1 %s",
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    return payload
