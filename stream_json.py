"""
Structured JSON output for programmatic use.

Emits newline-delimited JSON (JSONL) to stdout so external tools
(CI runners, agent orchestrators, TeamForge, etc.) can parse
Hermes output in real-time without scraping human-readable text.

Usage (from cli.py quiet-mode path)::

    from stream_json import StreamJsonEmitter

    emitter = StreamJsonEmitter(agent)
    agent.stream_delta_callback = emitter.on_text_delta
    agent.tool_gen_callback = emitter.on_tool_gen_start
    agent.tool_progress_callback = emitter.on_tool_progress

    # ... run conversation ...

    emitter.emit_result(result, session_id, exit_code)

Output format (one JSON object per line)::

    {"type":"system","subtype":"init","model":"anthropic/claude-sonnet-4","session_id":"..."}
    {"type":"text","text":"I'll fix that bug...","timestamp":1745000000}
    {"type":"tool_use","name":"read_file","input":{"path":"main.go"},"timestamp":1745000001}
    {"type":"tool_result","name":"read_file","output":"...","duration_ms":120,"is_error":false,"timestamp":1745000002}
    {"type":"text","text":"Found the issue.","timestamp":1745000003}
    {"type":"result","session_id":"...","exit_code":0,"tokens":{"input":5000,"output":200,"total":5200},"duration_ms":3500}
"""

import json
import sys
import time


class StreamJsonEmitter:
    """Emits structured JSONL events to stdout for machine-readable output."""

    __slots__ = ("_model", "_session_id", "_start_time", "_tool_start_times")

    def __init__(self, model: str = "", session_id: str = ""):
        self._model = model
        self._session_id = session_id
        self._start_time = time.time()
        self._tool_start_times: dict[str, float] = {}
        self._emit(
            {
                "type": "system",
                "subtype": "init",
                "model": model,
                "session_id": session_id,
            }
        )

    # ------------------------------------------------------------------
    # Callbacks — wire these to the corresponding agent callback slots.
    # ------------------------------------------------------------------

    def on_text_delta(self, text: str) -> None:
        """Stream delta callback — emits text chunks as they arrive."""
        # Only emit non-empty text.  The final assembled text will also
        # appear in the result envelope, so callers can ignore deltas
        # and just read the final line.
        if text and text.strip():
            self._emit(
                {"type": "text", "text": text, "timestamp": _now_ms()}
            )

    def on_tool_gen_start(self, tool_name: str) -> None:
        """Tool generation callback — emits when the model starts generating tool args."""
        key = f"{tool_name}:{_now_ms()}"
        self._tool_start_times[key] = time.time()
        self._emit(
            {
                "type": "tool_use",
                "name": tool_name,
                "timestamp": _now_ms(),
            }
        )

    def on_tool_progress(
        self,
        event_type: str,
        tool_name: str,
        args: dict | None,
        result: str | None,
        duration: float = 0,
        is_error: bool = False,
    ) -> None:
        """Tool progress callback — emits tool results when available.

        Only emits on 'complete' events to avoid duplicate tool_use lines.
        """
        if event_type == "complete":
            self._emit(
                {
                    "type": "tool_result",
                    "name": tool_name,
                    "output": _truncate_str(result or "", 5000),
                    "duration_ms": int(duration * 1000) if duration else 0,
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
    ) -> None:
        """Emit the final result envelope.  Call once after the conversation ends."""
        response = ""
        tokens = {}
        failed = False

        if isinstance(result, dict):
            response = result.get("final_response", "")
            failed = result.get("failed", False)
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

        self._emit(
            {
                "type": "result",
                "session_id": session_id or self._session_id,
                "exit_code": effective_exit,
                "tokens": tokens,
                "duration_ms": int((time.time() - self._start_time) * 1000),
            }
        )

        # Session ID to stderr (backward compatible with quiet mode)
        print(f"\nsession_id: {session_id}", file=sys.stderr)

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
