"""Stdio bridge for embedding Hermes in external hosts.

The bridge reads JSON requests from stdin (one JSON object per line) and emits
JSON events on stdout (also JSONL). This keeps transport deterministic for
Electron/local-server integrations.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

from hermes_cli.config import get_env_value, load_config
from hermes_constants import OPENROUTER_BASE_URL
from run_agent import AIAgent


@dataclass
class BridgeDefaults:
    model: str
    base_url: str
    api_key: Optional[str]
    max_iterations: int
    toolsets: Optional[List[str]]
    disabled_toolsets: Optional[List[str]]


@dataclass
class BridgeRunOptions:
    assistant_delta: bool
    assistant_delta_chunk_size: int
    native_assistant_delta: bool
    native_stream: bool
    stream_first_chunk_timeout_ms: Optional[int]
    stream_idle_timeout_ms: Optional[int]


class JsonlEmitter:
    """Write JSON events to a stream as single-line JSON objects."""

    def __init__(self, stream: TextIO):
        self._stream = stream
        self._lock = threading.Lock()

    def emit(self, event: Dict[str, Any]) -> None:
        with self._lock:
            self._stream.write(json.dumps(_make_json_safe(event), ensure_ascii=False) + "\n")
            self._stream.flush()


def _make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(v) for v in value]
    return str(value)


def _parse_toolsets(raw: Any) -> Optional[List[str]]:
    if raw is None:
        return None
    if isinstance(raw, str):
        items = [part.strip() for part in raw.split(",") if part.strip()]
        return items or None
    if isinstance(raw, (list, tuple, set)):
        items = [str(x).strip() for x in raw if str(x).strip()]
        return items or None
    value = str(raw).strip()
    return [value] if value else None


def _resolve_max_iterations(raw: Any, fallback: int) -> int:
    if raw in (None, ""):
        return fallback
    try:
        value = int(raw)
        return value if value > 0 else fallback
    except (TypeError, ValueError):
        return fallback


def _resolve_defaults(args: argparse.Namespace) -> BridgeDefaults:
    config = load_config()

    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        config_model = model_cfg.get("default")
        config_base_url = model_cfg.get("base_url")
    else:
        config_model = model_cfg
        config_base_url = None

    max_turns_cfg = config.get("max_turns", 60)
    env_max_turns = os.getenv("HERMES_MAX_ITERATIONS")

    toolsets = _parse_toolsets(args.toolsets)
    if toolsets is None:
        toolsets = _parse_toolsets(config.get("toolsets"))

    disabled_toolsets = _parse_toolsets(args.disabled_toolsets)

    base_url = (
        args.base_url
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENROUTER_BASE_URL")
        or config_base_url
        or OPENROUTER_BASE_URL
    )

    return BridgeDefaults(
        model=args.model or config_model or "anthropic/claude-opus-4.6",
        base_url=base_url,
        api_key=args.api_key or get_env_value("OPENAI_API_KEY") or get_env_value("OPENROUTER_API_KEY"),
        max_iterations=_resolve_max_iterations(
            env_max_turns,
            _resolve_max_iterations(args.max_iterations, int(max_turns_cfg or 60)),
        ),
        toolsets=toolsets,
        disabled_toolsets=disabled_toolsets,
    )


def _resolve_run_options(request: Dict[str, Any], args: argparse.Namespace) -> BridgeRunOptions:
    chunk_size = request.get("assistant_delta_chunk_size", getattr(args, "assistant_delta_chunk_size", 120))
    try:
        chunk_size_int = int(chunk_size)
    except (TypeError, ValueError):
        chunk_size_int = 120
    if chunk_size_int <= 0:
        chunk_size_int = 120

    assistant_delta = bool(request.get("assistant_delta", getattr(args, "assistant_delta", False)))
    native_assistant_delta = bool(
        request.get("native_assistant_delta", getattr(args, "native_assistant_delta", True))
    )
    native_stream = bool(request.get("native_stream", getattr(args, "native_stream", False)))

    first_chunk_timeout = request.get(
        "stream_first_chunk_timeout_ms",
        getattr(args, "stream_first_chunk_timeout_ms", None),
    )
    idle_timeout = request.get(
        "stream_idle_timeout_ms",
        getattr(args, "stream_idle_timeout_ms", None),
    )

    def _coerce_timeout(raw: Any) -> Optional[int]:
        if raw in (None, ""):
            return None
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    return BridgeRunOptions(
        assistant_delta=assistant_delta,
        assistant_delta_chunk_size=chunk_size_int,
        native_assistant_delta=native_assistant_delta,
        native_stream=native_stream,
        stream_first_chunk_timeout_ms=_coerce_timeout(first_chunk_timeout),
        stream_idle_timeout_ms=_coerce_timeout(idle_timeout),
    )


@contextlib.contextmanager
def _temporary_cwd(cwd: Optional[str]):
    if not cwd:
        yield
        return

    previous_cwd = os.getcwd()
    os.chdir(cwd)
    try:
        yield
    finally:
        os.chdir(previous_cwd)


@contextlib.contextmanager
def _temporary_terminal_cwd(cwd: Optional[str]):
    if not cwd:
        yield
        return

    prev_terminal_cwd = os.environ.get("TERMINAL_CWD")
    os.environ["TERMINAL_CWD"] = cwd
    try:
        yield
    finally:
        if prev_terminal_cwd is None:
            os.environ.pop("TERMINAL_CWD", None)
        else:
            os.environ["TERMINAL_CWD"] = prev_terminal_cwd


def _iter_chunks(text: str, chunk_size: int):
    if not text:
        return
    for start in range(0, len(text), chunk_size):
        yield text[start:start + chunk_size]


def _emit_assistant_delta_events(
    event: Dict[str, Any],
    run_options: BridgeRunOptions,
    request_id: str,
    session_id: str,
    emitter: JsonlEmitter,
) -> bool:
    """Synthetic assistant_delta fallback from assistant_message events only."""
    if event.get("type") != "assistant_message" or not run_options.assistant_delta:
        return False

    content = event.get("content")
    if not isinstance(content, str) or not content:
        return False

    total = 0
    for idx, chunk in enumerate(_iter_chunks(content, run_options.assistant_delta_chunk_size), start=1):
        total = idx
        emitter.emit(
            {
                "type": "assistant_delta",
                "request_id": request_id,
                "session_id": session_id,
                "timestamp": event.get("timestamp"),
                "delta": chunk,
                "delta_index": idx,
                "native": False,
            }
        )

    emitter.emit(
        {
            "type": "assistant_message_end",
            "request_id": request_id,
            "session_id": session_id,
            "timestamp": event.get("timestamp"),
            "delta_chunks": total,
            "has_tool_calls": bool(event.get("has_tool_calls")),
            "finish_reason": event.get("finish_reason"),
            "reasoning_present": bool(event.get("reasoning_present")),
            "native": False,
        }
    )
    return True


def _handle_run_request(
    request: Dict[str, Any],
    sessions: Dict[str, List[Dict[str, Any]]],
    sessions_lock: threading.Lock,
    active_agents: Dict[str, AIAgent],
    active_lock: threading.Lock,
    defaults: BridgeDefaults,
    args: argparse.Namespace,
    emitter: JsonlEmitter,
    stderr: TextIO,
) -> None:
    request_id = str(request.get("request_id") or uuid.uuid4().hex)

    user_message = request.get("message")
    if not isinstance(user_message, str) or not user_message.strip():
        emitter.emit(
            {
                "type": "error",
                "request_id": request_id,
                "stage": "request_validation",
                "message": "Run request requires a non-empty 'message' string.",
                "retryable": False,
            }
        )
        return

    session_id = str(request.get("session_id") or uuid.uuid4().hex)
    resume = bool(request.get("resume", True))

    seeded_history = request.get("conversation_history")
    if isinstance(seeded_history, list):
        history = seeded_history
    elif resume:
        with sessions_lock:
            history = sessions.get(session_id, [])
    else:
        history = []

    toolsets = _parse_toolsets(request.get("toolsets")) or defaults.toolsets
    disabled_toolsets = _parse_toolsets(request.get("disabled_toolsets")) or defaults.disabled_toolsets

    model = str(request.get("model") or defaults.model)
    base_url = str(request.get("base_url") or defaults.base_url)
    api_key = request.get("api_key") or defaults.api_key
    max_iterations = _resolve_max_iterations(request.get("max_iterations"), defaults.max_iterations)
    run_options = _resolve_run_options(request, args)

    cwd_value = request.get("cwd") or args.cwd
    cwd = str(cwd_value) if cwd_value else None

    if cwd:
        target = Path(cwd)
        if not target.exists() or not target.is_dir():
            emitter.emit(
                {
                    "type": "error",
                    "request_id": request_id,
                    "session_id": session_id,
                    "stage": "request_validation",
                    "message": f"Invalid cwd: '{cwd}'",
                    "retryable": False,
                }
            )
            return

    with active_lock:
        if session_id in active_agents:
            emitter.emit(
                {
                    "type": "error",
                    "request_id": request_id,
                    "session_id": session_id,
                    "stage": "request_validation",
                    "message": "A run is already active for this session_id.",
                    "retryable": True,
                }
            )
            return

    emitter.emit({"type": "session", "request_id": request_id, "session_id": session_id})

    native_deltas_in_turn = {"seen": False}

    def _forward_event(event: Dict[str, Any]) -> None:
        event = dict(event or {})
        event.setdefault("request_id", request_id)
        event.setdefault("session_id", session_id)

        event_type = event.get("type")

        if event_type == "assistant_delta":
            native_deltas_in_turn["seen"] = True
            if run_options.assistant_delta and run_options.native_assistant_delta:
                emitter.emit(event)
            return

        if event_type == "assistant_message_end":
            if native_deltas_in_turn["seen"]:
                if run_options.assistant_delta and run_options.native_assistant_delta:
                    emitter.emit(event)
                return
            # suppress stray end markers when no prior deltas were forwarded
            return

        if event_type == "assistant_message":
            # If native deltas were already emitted for this turn, avoid synthetic conversion.
            if not native_deltas_in_turn["seen"]:
                if _emit_assistant_delta_events(event, run_options, request_id, session_id, emitter):
                    native_deltas_in_turn["seen"] = False
                    return
            native_deltas_in_turn["seen"] = False
            emitter.emit(event)
            return

        emitter.emit(event)

    agent = AIAgent(
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_iterations=max_iterations,
        enabled_toolsets=toolsets,
        disabled_toolsets=disabled_toolsets,
        quiet_mode=True,
        verbose_logging=bool(getattr(args, "verbose", False)),
        session_id=session_id,
        event_callback=_forward_event,
        reasoning_config={"enabled": True, "effort": "high"},
        enable_native_streaming=run_options.native_stream,
        emit_assistant_delta_events=run_options.assistant_delta and run_options.native_assistant_delta,
        stream_first_chunk_timeout=(
            (run_options.stream_first_chunk_timeout_ms / 1000.0)
            if run_options.stream_first_chunk_timeout_ms
            else None
        ),
        stream_idle_timeout=(
            (run_options.stream_idle_timeout_ms / 1000.0)
            if run_options.stream_idle_timeout_ms
            else None
        ),
    )

    with active_lock:
        active_agents[session_id] = agent

    try:
        with _temporary_cwd(cwd), _temporary_terminal_cwd(cwd), contextlib.redirect_stdout(stderr):
            result = agent.run_conversation(
                user_message=user_message,
                system_message=request.get("system_message"),
                conversation_history=history,
                task_id=request.get("task_id"),
                request_id=request_id,
            )
    except Exception as exc:
        emitter.emit(
            {
                "type": "error",
                "request_id": request_id,
                "session_id": session_id,
                "stage": "bridge_run",
                "message": str(exc),
                "retryable": False,
            }
        )
        return
    finally:
        with active_lock:
            current = active_agents.get(session_id)
            if current is agent:
                active_agents.pop(session_id, None)

    with sessions_lock:
        sessions[session_id] = result.get("messages") or history

    emitter.emit(
        {
            "type": "final",
            "request_id": request_id,
            "session_id": session_id,
            "text": result.get("final_response"),
            "api_calls": result.get("api_calls"),
            "completed": bool(result.get("completed")),
            "partial": bool(result.get("partial")),
            "interrupted": bool(result.get("interrupted")),
        }
    )


def _start_run_thread(
    request: Dict[str, Any],
    sessions: Dict[str, List[Dict[str, Any]]],
    sessions_lock: threading.Lock,
    active_agents: Dict[str, AIAgent],
    active_lock: threading.Lock,
    defaults: BridgeDefaults,
    args: argparse.Namespace,
    emitter: JsonlEmitter,
    stderr: TextIO,
) -> threading.Thread:
    thread = threading.Thread(
        target=_handle_run_request,
        kwargs={
            "request": request,
            "sessions": sessions,
            "sessions_lock": sessions_lock,
            "active_agents": active_agents,
            "active_lock": active_lock,
            "defaults": defaults,
            "args": args,
            "emitter": emitter,
            "stderr": stderr,
        },
        daemon=True,
    )
    thread.start()
    return thread


def _cleanup_finished_threads(threads: List[threading.Thread]) -> None:
    alive = [t for t in threads if t.is_alive()]
    threads[:] = alive


def _interrupt_session(
    request: Dict[str, Any],
    active_agents: Dict[str, AIAgent],
    active_lock: threading.Lock,
    emitter: JsonlEmitter,
) -> None:
    request_id = request.get("request_id")
    session_id = request.get("session_id")
    if not session_id:
        emitter.emit(
            {
                "type": "error",
                "request_id": request_id,
                "stage": "request_validation",
                "message": "interrupt request requires 'session_id'.",
                "retryable": False,
            }
        )
        return

    with active_lock:
        agent = active_agents.get(str(session_id))

    if not agent:
        emitter.emit(
            {
                "type": "error",
                "request_id": request_id,
                "session_id": str(session_id),
                "stage": "request_validation",
                "message": "No active run for the provided session_id.",
                "retryable": True,
            }
        )
        return

    agent.interrupt(message=request.get("message"))
    emitter.emit(
        {
            "type": "interrupt_ack",
            "request_id": request_id,
            "session_id": str(session_id),
            "accepted": True,
        }
    )


def bridge_command(
    args: argparse.Namespace,
    stdin: Optional[TextIO] = None,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
) -> int:
    """Run the Hermes JSONL stdio bridge."""

    in_stream: TextIO = sys.stdin if stdin is None else stdin
    out_stream: TextIO = sys.stdout if stdout is None else stdout
    err_stream: TextIO = sys.stderr if stderr is None else stderr

    defaults = _resolve_defaults(args)
    emitter = JsonlEmitter(out_stream)
    sessions: Dict[str, List[Dict[str, Any]]] = {}
    active_agents: Dict[str, AIAgent] = {}
    sessions_lock = threading.Lock()
    active_lock = threading.Lock()
    run_threads: List[threading.Thread] = []

    for raw_line in in_stream:
        line = raw_line.strip()
        if not line:
            continue

        _cleanup_finished_threads(run_threads)

        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            emitter.emit(
                {
                    "type": "error",
                    "stage": "request_parse",
                    "message": f"Invalid JSON input: {exc}",
                    "retryable": True,
                }
            )
            continue

        req_type = str(request.get("type") or "run").lower()

        if req_type == "ping":
            emitter.emit({"type": "pong", "request_id": request.get("request_id")})
            if args.once:
                break
            continue

        if req_type == "interrupt":
            _interrupt_session(request, active_agents, active_lock, emitter)
            if args.once:
                break
            continue

        if req_type in {"shutdown", "exit", "quit"}:
            emitter.emit({"type": "shutdown_ack", "request_id": request.get("request_id")})
            break

        if req_type != "run":
            emitter.emit(
                {
                    "type": "error",
                    "request_id": request.get("request_id"),
                    "stage": "request_validation",
                    "message": f"Unsupported request type '{req_type}'.",
                    "retryable": False,
                }
            )
            if args.once:
                break
            continue

        if args.once:
            _handle_run_request(
                request=request,
                sessions=sessions,
                sessions_lock=sessions_lock,
                active_agents=active_agents,
                active_lock=active_lock,
                defaults=defaults,
                args=args,
                emitter=emitter,
                stderr=err_stream,
            )
            break

        run_threads.append(
            _start_run_thread(
                request=request,
                sessions=sessions,
                sessions_lock=sessions_lock,
                active_agents=active_agents,
                active_lock=active_lock,
                defaults=defaults,
                args=args,
                emitter=emitter,
                stderr=err_stream,
            )
        )

    with active_lock:
        running_agents = list(active_agents.values())

    for agent in running_agents:
        try:
            agent.interrupt(message="bridge shutdown")
        except Exception:
            pass

    for thread in run_threads:
        thread.join(timeout=10)

    return 0
