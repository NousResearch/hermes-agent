"""Reusable PydanticAI event/progress recording helpers."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from .storage import utc_now


@dataclass
class PydanticAgentRecorder:
    label: str
    progress: bool = False
    env_prefix: str = "IA_PYDANTIC_PROGRESS"
    logger: logging.Logger | None = None
    include_tool_call_deltas: bool = True
    events: list[dict[str, Any]] = field(default_factory=list)
    started_monotonic: float = field(default_factory=time.monotonic)

    def add_event(self, event: dict[str, Any]) -> None:
        self.events.append(
            {
                "time": utc_now(),
                "t_rel_s": round(time.monotonic() - self.started_monotonic, 3),
                **event,
            }
        )

    def progress_log(self, stage: str, **fields: Any) -> None:
        if not self.progress:
            return
        record = {
            "time": utc_now(),
            "t_rel_s": round(time.monotonic() - self.started_monotonic, 3),
            "label": self.label,
            "stage": stage,
            **fields,
        }
        message = f"{self.env_prefix} {_preview(record, _env_int(f'{self.env_prefix}_MAX_CHARS', 2400))}"
        if self.logger:
            self.logger.info(message)
        print(message, file=sys.stderr, flush=True)

    def event_stream_handler(self):
        async def handler(_ctx: Any, events: Any) -> None:
            async for event in events:
                summary = summarize_pydantic_event(
                    event,
                    max_chars=_env_int(f"{self.env_prefix}_EVENT_MAX_CHARS", 1200),
                    include_tool_call_deltas=self.include_tool_call_deltas,
                )
                if summary is None:
                    continue
                self.add_event(summary)
                self.progress_log("agent_event", **summary)

        return handler


def summarize_pydantic_event(
    event: Any,
    *,
    max_chars: int = 1200,
    include_tool_call_deltas: bool = True,
) -> dict[str, Any] | None:
    kind = getattr(event, "event_kind", type(event).__name__)
    part = getattr(event, "part", None)
    if kind in {"function_tool_call", "output_tool_call"} and part is not None:
        return {
            "event": kind,
            "tool": getattr(part, "tool_name", ""),
            "call_id": getattr(part, "tool_call_id", ""),
            "args": _part_args(part, max_chars),
        }
    if kind in {"function_tool_result", "output_tool_result"} and part is not None:
        return {
            "event": kind,
            "tool": getattr(part, "tool_name", ""),
            "call_id": getattr(part, "tool_call_id", ""),
            "content_preview": _preview(getattr(part, "content", ""), max_chars),
        }
    if kind == "builtin_tool_call":
        native_part = getattr(event, "part", None)
        return {
            "event": kind,
            "tool": getattr(native_part, "tool_name", ""),
            "call_id": getattr(native_part, "tool_call_id", ""),
            "args": _part_args(native_part, max_chars),
        }
    if kind == "builtin_tool_result":
        native_result = getattr(event, "result", None)
        return {
            "event": kind,
            "tool": getattr(native_result, "tool_name", ""),
            "call_id": getattr(native_result, "tool_call_id", ""),
            "content_preview": _preview(getattr(native_result, "content", ""), max_chars),
        }
    if kind in {"part_start", "part_end"} and part is not None:
        return {
            "event": kind,
            "index": getattr(event, "index", ""),
            "part_kind": getattr(part, "part_kind", ""),
            "tool": getattr(part, "tool_name", ""),
            "call_id": getattr(part, "tool_call_id", ""),
        }
    if kind == "part_delta":
        delta = getattr(event, "delta", None)
        delta_kind = getattr(delta, "part_delta_kind", "")
        if delta_kind != "tool_call" or not include_tool_call_deltas:
            return None
        args_delta = getattr(delta, "args_delta", None)
        if not args_delta:
            return None
        return {
            "event": kind,
            "index": getattr(event, "index", ""),
            "delta_kind": delta_kind,
            "tool_name_delta": getattr(delta, "tool_name_delta", ""),
            "call_id_delta": getattr(delta, "tool_call_id_delta", ""),
            "delta_len": len(str(args_delta or "")),
            "delta_preview": _preview(args_delta, max_chars),
        }
    if kind == "final_result":
        return {
            "event": kind,
            "tool": getattr(event, "tool_name", ""),
            "call_id": getattr(event, "tool_call_id", ""),
        }
    return None


def run_pydantic_agent_sync(
    agent: Any,
    payload: dict[str, Any],
    recorder: PydanticAgentRecorder | None = None,
):
    kwargs: dict[str, Any] = {}
    if recorder is not None:
        kwargs["event_stream_handler"] = recorder.event_stream_handler()
    return agent.run_sync(json.dumps(payload, ensure_ascii=False, sort_keys=True), **kwargs)


def _part_args(part: Any, max_chars: int) -> str:
    if not hasattr(part, "args_as_json_str"):
        return ""
    try:
        return _preview(part.args_as_json_str(), max_chars)
    except Exception:
        return _preview(repr(getattr(part, "args", "")), max_chars)


def _preview(value: Any, max_chars: int) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = repr(value)
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name) or default).strip())
    except ValueError:
        return default
