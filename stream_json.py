"""
Structured JSON output for programmatic use.

Emits newline-delimited JSON (JSONL) to stdout so external tools
(CI runners, agent orchestrators, TeamForge, etc.) can parse
Hermes output in real-time without scraping human-readable text.
"""

import json
import sys
import time
from typing import Any


class StreamJsonEmitter:
    """Emits structured JSONL events to stdout for machine-readable output."""

    __slots__ = (
        "_model",
        "_session_id",
        "_start_time",
        "_tool_start_times",
        "_pending_tool_use_names",
    )

    def __init__(self, model: str = "", session_id: str = ""):
        self._model = model
        self._session_id = session_id
        self._start_time = time.time()
        self._tool_start_times: dict[str, float] = {}
        self._pending_tool_use_names: dict[str, int] = {}
        self._emit(
            {
                "type": "system",
                "subtype": "init",
                "model": model,
                "session_id": session_id,
                "timestamp": _now_ms(),
            }
        )

    # ------------------------------------------------------------------
    # Callbacks — wire these to the corresponding agent callback slots.
    # ------------------------------------------------------------------

    def on_text_delta(self, text: str) -> None:
        """Stream delta callback — emits text chunks as they arrive."""
        if text and str(text).strip():
            self._emit({"type": "text", "text": text, "timestamp": _now_ms()})

    def on_tool_gen_start(self, tool_name: str) -> None:
        """Tool generation callback — emits when tool-call generation starts.

        Current Hermes also emits a later ``tool.started`` progress callback once
        full arguments are available.  Track the generation event so the later
        started event does not produce a duplicate ``tool_use`` line.
        """
        name = tool_name or "unknown"
        self._pending_tool_use_names[name] = self._pending_tool_use_names.get(name, 0) + 1
        self._tool_start_times[name] = time.time()
        self._emit(
            {
                "type": "tool_use",
                "name": name,
                "timestamp": _now_ms(),
            }
        )

    def on_tool_progress(
        self,
        event_type: str,
        tool_name: str | None = None,
        preview: Any = None,
        args: Any = None,
        **kwargs: Any,
    ) -> None:
        """Tool progress callback.

        Supports both the original PR test shape::

            on_tool_progress("complete", name, args, result, duration=...)

        and the current Hermes callback shape::

            on_tool_progress("tool.started", name, preview, args)
            on_tool_progress("tool.completed", name, None, None,
                             duration=..., is_error=..., result=...)
        """
        name = tool_name or "unknown"
        event = event_type or ""

        if event == "tool.started":
            pending = self._pending_tool_use_names.get(name, 0)
            if pending:
                if pending == 1:
                    self._pending_tool_use_names.pop(name, None)
                else:
                    self._pending_tool_use_names[name] = pending - 1
                return

            self._tool_start_times[name] = time.time()
            payload = {"type": "tool_use", "name": name, "timestamp": _now_ms()}
            if isinstance(args, dict):
                payload["input"] = args
            elif isinstance(preview, dict):
                payload["input"] = preview
            self._emit(payload)
            return

        if event not in {"complete", "tool.completed"}:
            return

        output = kwargs.get("result")
        if output is None:
            output = kwargs.get("output")
        if output is None:
            # Backward-compatible positional shape used by the original tests:
            # ("complete", name, args, result, duration=...)
            if isinstance(args, str):
                output = args
            elif isinstance(preview, str):
                output = preview
            else:
                output = ""

        duration = kwargs.get("duration", 0) or 0
        is_error = bool(kwargs.get("is_error", False))

        if not duration:
            start = self._tool_start_times.get(name)
            if start is not None:
                duration = time.time() - start

        self._emit(
            {
                "type": "tool_result",
                "name": name,
                "output": _truncate_str(str(output or ""), 5000),
                "duration_ms": int(float(duration) * 1000) if duration else 0,
                "is_error": is_error,
                "timestamp": _now_ms(),
            }
        )

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------

    def emit_result(
        self,
        result: dict | None,
        session_id: str = "",
        exit_code: int = 0,
    ) -> int:
        """Emit the final result envelope.  Call once after the conversation ends."""
        response = ""
        tokens = {}
        failed = False
        error = None

        if isinstance(result, dict):
            response = result.get("final_response", "") or ""
            failed = bool(result.get("failed", False))
            error = result.get("error")
            tokens = {
                "input": result.get("input_tokens", 0) or 0,
                "output": result.get("output_tokens", 0) or 0,
                "total": result.get("total_tokens", 0) or 0,
                "cache_read": result.get("cache_read_tokens", 0) or 0,
                "cache_write": result.get("cache_write_tokens", 0) or 0,
            }
        elif result is not None:
            response = str(result)

        effective_exit = exit_code if exit_code != 0 else (1 if failed else 0)

        payload = {
            "type": "result",
            "session_id": session_id or self._session_id,
            "exit_code": effective_exit,
            "text": response,
            "tokens": tokens,
            "duration_ms": int((time.time() - self._start_time) * 1000),
            "timestamp": _now_ms(),
        }
        if error:
            payload["error"] = str(error)

        self._emit(payload)

        # Session ID to stderr (backward compatible with quiet mode)
        print(f"\nsession_id: {session_id or self._session_id}", file=sys.stderr)

        return effective_exit

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit(self, obj: dict) -> None:
        """Write one JSON line to stdout and flush immediately."""
        try:
            sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        except (BrokenPipeError, OSError):
            # Pipe closed by consumer — stop writing.
            pass


def _now_ms() -> int:
    """Current time as milliseconds since epoch."""
    return int(time.time() * 1000)


def _truncate_str(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."
