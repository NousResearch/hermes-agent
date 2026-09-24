"""``hermes chat -q … --format stream-json``: one JSON object per stdout line.

CI runners and orchestrators consume a one-shot run without scraping human-formatted text:
``system/init`` → ``text`` deltas / ``tool_use`` / ``tool_result`` → one terminal ``result``
envelope (exit code, final text, token stats). Diagnostics and ``session_id`` stay on stderr.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

_TOOL_OUTPUT_CAP = 5000


def stream_json_requested(args) -> bool:
    """True when ``--format stream-json`` was passed; exits 2 on the combinations the protocol forbids
    (no query to answer, or the interactive TUI transport) and forces quiet mode on ``args``."""
    if getattr(args, "output_format", "text") != "stream-json":
        return False
    if not (getattr(args, "query", None) or getattr(args, "query_file", None)):
        print("Error: --format stream-json requires -q/--query.", file=sys.stderr)
        raise SystemExit(2)
    if getattr(args, "tui", False):
        print("Error: --format stream-json cannot be used with --tui.", file=sys.stderr)
        raise SystemExit(2)
    args.quiet = True
    return True


def _now_ms() -> int:
    return int(time.time() * 1000)


class StreamJsonEmitter:
    """Agent-callback sink that writes JSONL events to stdout and flushes each line."""

    def __init__(self, model: str = "", session_id: str = ""):
        self._session_id = session_id
        self._start = time.time()
        self._tool_started: dict[str, float] = {}
        self._emit({"type": "system", "subtype": "init", "model": model, "session_id": session_id})

    def attach(self, agent) -> "StreamJsonEmitter":
        """Route the agent's streaming/tool callbacks into this emitter (``init`` was already written at
        construction, before credentials/agent init, so a failed start still yields init + result)."""
        agent.stream_delta_callback = self.on_text_delta
        agent.tool_progress_callback = self.on_tool_progress
        return self

    def bind_session(self, session_id: str) -> None:
        """Record the id the gateway assigned after ``init`` was written, so an interrupted or failed
        run still reports which session holds the accepted work."""
        self._session_id = session_id or self._session_id

    def on_text_delta(self, text: str | None) -> None:
        # Only None/"" (the turn-end sentinel) is dropped: whitespace deltas are part of the text, and
        # a consumer concatenating ``text`` events must reproduce the answer byte for byte.
        if text:
            self._emit({"type": "text", "text": str(text)})

    def on_tool_progress(self, event_type: str, tool_name: str | None = None, preview: Any = None, args: Any = None,
                         **kwargs: Any) -> None:
        """``tool.started`` → ``tool_use`` (with ``input`` when the args are a dict); ``tool.completed`` →
        ``tool_result``. Other progress events (reasoning, output risk) are not part of the protocol."""
        name = tool_name or "unknown"
        # Parallel same-name calls would clobber each other's start time under a name-only key.
        key = kwargs.get("tool_call_id") or name
        if event_type == "tool.started":
            self._tool_started[key] = time.time()
            payload: dict[str, Any] = {"type": "tool_use", "name": name}
            if kwargs.get("tool_call_id"):
                payload["tool_call_id"] = kwargs["tool_call_id"]
            if isinstance(args, dict):
                payload["input"] = args
            self._emit(payload)
        elif event_type == "tool.completed":
            duration = kwargs.get("duration") or (time.time() - self._tool_started.pop(key, time.time()))
            output = str(kwargs.get("result") or "")
            self._emit({"type": "tool_result", "name": name,
                        **({"tool_call_id": kwargs["tool_call_id"]} if kwargs.get("tool_call_id") else {}),
                        "output": output if len(output) <= _TOOL_OUTPUT_CAP else output[:_TOOL_OUTPUT_CAP] + "...",
                        "duration_ms": int(float(duration) * 1000), "is_error": bool(kwargs.get("is_error", False))})

    def emit_result(self, result: Any, session_id: str = "", exit_code: int = 0) -> int:
        """Write the terminal ``result`` record (once) and return the process exit code it reports."""
        data = result if isinstance(result, dict) else {"final_response": "" if result is None else str(result)}
        exit_code = exit_code or (1 if data.get("failed") else 0)
        payload = {"type": "result", "session_id": session_id or self._session_id, "exit_code": exit_code,
                   "text": data.get("final_response") or "",
                   "tokens": {"input": data.get("input_tokens") or 0, "output": data.get("output_tokens") or 0,
                              "total": data.get("total_tokens") or 0, "cache_read": data.get("cache_read_tokens") or 0,
                              "cache_write": data.get("cache_write_tokens") or 0},
                   "duration_ms": int((time.time() - self._start) * 1000)}
        if data.get("error"):
            payload["error"] = str(data["error"])
        self._emit(payload)
        print(f"\nsession_id: {session_id or self._session_id}", file=sys.stderr)  # same stderr contract as -Q
        return exit_code

    def _emit(self, obj: dict) -> None:
        try:
            sys.stdout.write(json.dumps({**obj, "timestamp": _now_ms()}, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        except (BrokenPipeError, OSError):
            pass  # consumer closed the pipe — nothing left to report to
