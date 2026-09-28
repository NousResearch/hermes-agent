"""``hermes chat -q … --format stream-json``: one JSON object per stdout line.

CI runners and orchestrators consume a one-shot run without scraping human-formatted text:
``system/init`` → ``text`` deltas / ``tool_use`` / ``tool_result`` → one terminal ``result``
envelope (exit code, final text, token stats). Diagnostics and ``session_id`` stay on stderr.
"""

from __future__ import annotations

import json
from contextlib import contextmanager, redirect_stdout
import sys
import time
from typing import Any

_TOOL_OUTPUT_CAP = 5000


def stream_json_requested(args) -> bool:
    """Validate one-shot output options and force quiet mode on ``args``.

    Kept at the stream protocol boundary so text/schema/file output and both
    machine formats reject interactive/TUI combinations before startup.
    """
    options = []
    if getattr(args, "output_format", "text") in {"json", "stream-json"}:
        options.append(f"--format {args.output_format}")
    for option in ("output_schema", "output_last_message"):
        if getattr(args, option, None) is not None:
            options.append("--" + option.replace("_", "-"))
    if not options:
        return False
    label = ", ".join(options)
    if not (getattr(args, "query", None) or getattr(args, "query_file", None)):
        print(f"Error: {label} requires -q/--query or --query-file.", file=sys.stderr)
        raise SystemExit(2)
    if getattr(args, "tui", False):
        print(f"Error: {label} cannot be used with --tui.", file=sys.stderr)
        raise SystemExit(2)
    args.quiet = True
    return True


@contextmanager
def output_protocol(output_format, model=""):
    """Keep startup/cleanup chatter off JSON stdout and close early failures once."""
    emitter = None
    try:
        emitter = StreamJsonEmitter(model=model, final_only=output_format != "stream-json",
                                    text_only=output_format == "text", defer_init=True)
        if output_format in {"json", "stream-json"}:
            with redirect_stdout(sys.stderr):
                yield emitter
        else:
            # Schema/file text output needs the same completion guard: an early
            # return or exit(0) before the one-shot runner is not a successful run.
            yield emitter
    except ValueError as exc:
        if emitter is not None:
            emitter.emit_result({"failed": True, "error": str(exc)}, exit_code=1)
        if output_format != "text":
            print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except SystemExit as exc:
        if emitter is not None and not emitter.result_emitted:
            code = exc.code if isinstance(exc.code, int) and exc.code else 1
            emitter.emit_result({"failed": True, "error": "Chat exited before producing a final response; see stderr."},
                                exit_code=code)
            raise SystemExit(code) from None
        raise
    except (KeyboardInterrupt, InterruptedError):
        if emitter is None:
            raise
        emitter.emit_result({"failed": True, "interrupted": True,
                             "error": "Interrupted before completion"}, exit_code=130)
        raise SystemExit(130) from None
    except Exception as exc:
        if emitter is None:
            raise
        emitter.emit_result({"failed": True, "error": f"{type(exc).__name__}: {exc}"}, exit_code=1)
        if output_format != "text":
            print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    else:
        if emitter is not None and not emitter.result_emitted:
            emitter.emit_result({"failed": True, "error": "Chat ended before producing a final response"}, exit_code=1)
            raise SystemExit(1)


def _now_ms() -> int:
    return int(time.time() * 1000)


class StreamJsonEmitter:
    """Agent-callback sink that writes JSONL events to stdout and flushes each line."""

    def __init__(self, model: str = "", session_id: str = "", *, final_only: bool = False,
                 defer_init: bool = False, text_only: bool = False):
        self._stdout = sys.stdout
        self.result_emitted = False
        self._exit_code = 0
        self._final_only = final_only or text_only
        self._text_only = text_only
        self._session_id = session_id
        self._start = time.time()
        self._tool_started: dict[str, float] = {}
        self._model = model
        self._initialized = False
        if not defer_init:
            self.initialize(model, session_id)

    def initialize(self, model: str = "", session_id: str = "") -> None:
        """Emit init once, after CLI routing resolves the model/session (or on early failure)."""
        if not self._initialized:
            self._model = model or self._model
            self._session_id = session_id or self._session_id
            self._initialized = True
            self._emit({"type": "system", "subtype": "init", "model": self._model, "session_id": self._session_id})

    def attach(self, agent) -> "StreamJsonEmitter":
        """Route the agent's streaming/tool callbacks into this emitter (``init`` was already written at
        construction, before credentials/agent init, so a failed start still yields init + result)."""
        if not self._final_only:
            agent.stream_delta_callback = self.on_text_delta
            agent.tool_progress_callback = self.on_tool_progress
        return self

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

    def emit_result(self, result: Any, session_id: str = "", exit_code: int = 0, *, output_last_message=None) -> int:
        """Write the terminal ``result`` record (once) and return the process exit code it reports."""
        if self.result_emitted:
            return self._exit_code
        self.initialize(session_id=session_id)
        data = result if isinstance(result, dict) else {"final_response": "" if result is None else str(result)}
        exit_code = exit_code or (1 if data.get("failed") else 0)
        payload = {"type": "result", "session_id": session_id or self._session_id, "exit_code": exit_code,
                   "text": data.get("final_response") or "",
                   "tokens": {"input": data.get("input_tokens") or 0, "output": data.get("output_tokens") or 0,
                              "total": data.get("total_tokens") or 0, "cache_read": data.get("cache_read_tokens") or 0,
                              "cache_write": data.get("cache_write_tokens") or 0},
                   "duration_ms": int((time.time() - self._start) * 1000)}
        for key in ("structured_output", "schema_errors", "failed", "failure_reason", "partial", "interrupted"):
            if key in data:
                payload[key] = data[key]
        if data.get("error"):
            payload["error"] = str(data["error"])
        # Serialize before publishing the artifact: a malformed envelope must not
        # replace a previous successful answer or suppress the failure envelope.
        if self._text_only:
            line = payload["text"] + "\n" if payload["text"] else ""
            line.encode(getattr(self._stdout, "encoding", None) or "utf-8",
                        errors=getattr(self._stdout, "errors", None) or "strict")
        else:
            line = self._serialize(payload)
        if output_last_message is not None and exit_code == 0:
            from pathlib import Path
            from hermes_cli.structured_output import write_last_message
            try:
                write_last_message(Path(output_last_message), payload["text"])
            except ValueError as exc:
                return self.emit_result({**data, "failed": True, "completed": False,
                                         "failure_reason": "output_file", "error": str(exc)},
                                        session_id=session_id, exit_code=1)
        if self._text_only and exit_code and payload.get("error"):
            print(f"Error: {payload['error']}", file=sys.stderr)
        self._write(line)
        self.result_emitted = True
        self._exit_code = exit_code
        print(f"\nsession_id: {session_id or self._session_id}", file=sys.stderr)  # same stderr contract as -Q
        return exit_code

    def _emit(self, obj: dict) -> None:
        if self._final_only and obj["type"] != "result":
            return
        self._write(self._serialize(obj))

    @staticmethod
    def _serialize(obj: dict) -> str:
        # JSON permits escaped lone surrogates; emitting them literally fails on
        # strict UTF-8 stdout after an otherwise valid schema result.
        return json.dumps({**obj, "timestamp": _now_ms()}, ensure_ascii=True) + "\n"

    def _write(self, line: str) -> None:
        try:
            self._stdout.write(line)
            self._stdout.flush()
        except (BrokenPipeError, OSError):
            pass  # consumer closed the pipe — nothing left to report to
