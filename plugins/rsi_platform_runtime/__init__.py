from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any


PLUGIN_NAME = "rsi_platform_runtime"
PLUGIN_VERSION = "0.1.0"
CONTRACT_VERSION = "execution-envelope/v1"
CAPABILITIES = {
    "execution_scoped_context_supported": True,
    "execution_envelope_v1_producer": True,
    "atomic_envelope_writer": True,
}

_STATE_BY_EXECUTION: dict[str, dict[str, Any]] = {}
_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(bearer)\s+([A-Za-z0-9._~+/=-]{8,})"),
    re.compile(r"\b(xox[baprs]-[A-Za-z0-9-]{8,})\b"),
    re.compile(r"\b(sk-[A-Za-z0-9_-]{8,})\b"),
    re.compile(r"(?i)(token|secret|password|api[_-]?key)(['\"\s:=]+)([^\s'\",}\]]{4,})"),
)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _string(value: Any) -> str:
    return str(value or "").strip()


def _json_object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _json_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _parse_json_maybe(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except Exception:
        return value


def _redact_text(value: str) -> str:
    out = value
    for pattern in _SECRET_PATTERNS:
        if pattern.pattern.startswith("(?i)\\b(bearer)"):
            out = pattern.sub(lambda match: f"{match.group(1)} [redacted]", out)
        elif "token|secret|password" in pattern.pattern:
            out = pattern.sub(lambda match: f"{match.group(1)}{match.group(2)}[redacted]", out)
        else:
            out = pattern.sub("[redacted]", out)
    return out


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if any(fragment in key_text.lower() for fragment in ("token", "secret", "password", "api_key", "apikey")):
                redacted[key_text] = "[redacted]"
            else:
                redacted[key_text] = _redact(item)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _runtime_path(env_name: str) -> Path:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        raise RuntimeError(f"{env_name} is required for RSI native workflow execution.")
    return Path(raw).expanduser().resolve()


def _runtime_ids() -> dict[str, str]:
    ids = {
        "execution_id": os.getenv("RSI_EXECUTION_ID", "").strip(),
        "operation_id": os.getenv("RSI_OPERATION_ID", "").strip(),
        "trace_id": os.getenv("RSI_TRACE_ID", "").strip(),
        "workflow_id": os.getenv("RSI_WORKFLOW_ID", "").strip(),
    }
    missing = [key for key, value in ids.items() if not value]
    if missing:
        raise RuntimeError("Missing RSI runtime identifier(s): " + ", ".join(missing))
    return ids


def _load_context() -> dict[str, Any]:
    context_path = _runtime_path("RSI_RUNTIME_CONTEXT_PATH")
    if not context_path.exists():
        raise RuntimeError(f"RSI runtime context not found at {context_path}.")
    parsed = json.loads(context_path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError(f"RSI runtime context at {context_path} must be a JSON object.")
    ids = _runtime_ids()
    for key, expected in ids.items():
        actual = str(parsed.get(key, "") or "").strip()
        if actual != expected:
            raise RuntimeError(f"RSI runtime context {key} mismatch: expected {expected!r}, got {actual!r}.")
    return parsed


def _fsync_parent(path: Path) -> None:
    try:
        flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
        fd = os.open(str(path.parent), flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(_redact(payload), handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_parent(path)
    finally:
        temp_path.unlink(missing_ok=True)


def _state() -> dict[str, Any]:
    ids = _runtime_ids()
    execution_id = ids["execution_id"]
    state = _STATE_BY_EXECUTION.setdefault(
        execution_id,
        {
            **ids,
            "created_at": _now_iso(),
            "ledger_events": [],
            "artifacts": [],
            "deliveries": [],
        },
    )
    if "context" not in state:
        state["context"] = _load_context()
    return state


def _tool_call_event(tool_name: str, args: dict[str, Any], result: Any, tool_call_id: str, duration_ms: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "args": args,
        "result": result,
    }
    try:
        payload["duration_ms"] = int(duration_ms)
    except Exception:
        pass
    return {
        "event_id": hashlib.sha256(f"{tool_call_id}:{tool_name}:{time.time_ns()}".encode("utf-8")).hexdigest()[:24],
        "kind": "tool.call.completed",
        "phase_id": "main",
        "status": "completed",
        "sequence": 0,
        "idempotency_key": tool_call_id or "",
        "recorded_at": _now_iso(),
        "payload": payload,
    }


def _delivery_from_tool_result(tool_name: str, args: dict[str, Any], result: Any, tool_call_id: str) -> dict[str, Any] | None:
    if not (tool_name == "send_message" or tool_name.endswith(".send_message")):
        return None
    parsed = _parse_json_maybe(result)
    if not isinstance(parsed, dict) or not parsed.get("success"):
        return None
    platform = _string(parsed.get("platform")) or _string(args.get("target")).split(":", 1)[0]
    chat_id = _string(parsed.get("chat_id"))
    thread_id = _string(parsed.get("thread_id"))
    provider_ref = _string(parsed.get("message_id"))
    delivery: dict[str, Any] = {
        "send_status": "posted",
        "channel_id": chat_id,
        "thread_ts": thread_id,
        "body": _string(args.get("message")),
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "provider_ref": provider_ref,
        "message_link": _string(parsed.get("message_link")),
    }
    if platform:
        delivery["platform"] = platform
    if parsed.get("idempotency_key"):
        delivery["idempotency_key"] = _string(parsed.get("idempotency_key"))
    return delivery


def _artifact_from_tool_result(tool_name: str, result: Any) -> dict[str, Any] | None:
    parsed = _parse_json_maybe(result)
    if not isinstance(parsed, dict):
        return None
    output = _json_object(parsed.get("output"))
    path = _string(output.get("path")) or _string(parsed.get("path"))
    if not path:
        return None
    if not any(fragment in tool_name for fragment in ("write_file", "artifact")):
        return None
    return {
        "kind": "file",
        "workspace_path": path,
        "file_ref": path,
        "share_status": "not_shared",
    }


def post_tool_call(tool_name: str = "", args: dict[str, Any] | None = None, result: Any = None, tool_call_id: str = "", duration_ms: Any = None, **_kwargs) -> None:
    state = _state()
    safe_args = args if isinstance(args, dict) else {}
    parsed_result = _parse_json_maybe(result)
    event = _tool_call_event(tool_name, safe_args, parsed_result, tool_call_id, duration_ms)
    event["sequence"] = len(state["ledger_events"]) + 1
    state["ledger_events"].append(event)
    delivery = _delivery_from_tool_result(tool_name, safe_args, parsed_result, tool_call_id)
    if delivery:
        state["deliveries"].append(delivery)
    artifact = _artifact_from_tool_result(tool_name, parsed_result)
    if artifact:
        state["artifacts"].append(artifact)


def _phase_runs(state: dict[str, Any], completion: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "phase_id": "main",
            "phase_type": "workflow",
            "status": "completed" if completion.get("ok") else "failed",
            "completion_verdict": completion.get("completion_verdict", ""),
            "termination_reason": completion.get("termination_reason", ""),
        }
    ]


def _write_envelope(final_response: str) -> None:
    state = _state()
    envelope_path = _runtime_path("RSI_RUNTIME_ENVELOPE_PATH")
    completion = {
        "ok": True,
        "completion_verdict": "complete",
        "termination_reason": "normal_completion",
        "partial": False,
        "max_iterations_reached": False,
    }
    envelope = {
        "contract_version": CONTRACT_VERSION,
        "producer": PLUGIN_NAME,
        "producer_version": PLUGIN_VERSION,
        "created_at": _now_iso(),
        "facts_source": [
            "rsi_platform_runtime.post_tool_call",
            "rsi_platform_runtime.post_llm_call",
        ],
        "execution_id": state["execution_id"],
        "operation_id": state["operation_id"],
        "trace_id": state["trace_id"],
        "workflow_id": state["workflow_id"],
        "session_id": _string(state.get("context", {}).get("hermes_session_id")),
        "phase_runs": _phase_runs(state, completion),
        "ledger_events": list(state["ledger_events"]),
        "artifacts": list(state["artifacts"]),
        "deliveries": list(state["deliveries"]),
        "completion": completion,
        "final_response": str(final_response or ""),
    }
    _atomic_write_json(envelope_path, envelope)


def post_llm_call(assistant_response: str = "", **_kwargs) -> None:
    _write_envelope(assistant_response)


def on_session_end(completed: bool = False, **_kwargs) -> None:
    if not completed:
        return
    envelope_path_raw = os.getenv("RSI_RUNTIME_ENVELOPE_PATH", "").strip()
    if envelope_path_raw and not Path(envelope_path_raw).expanduser().exists():
        _write_envelope("")


def _capabilities_handler(_args=None, **_kwargs) -> str:
    return json.dumps(
        {
            "plugin": PLUGIN_NAME,
            "version": PLUGIN_VERSION,
            "capabilities": CAPABILITIES,
        },
        ensure_ascii=True,
        sort_keys=True,
    )


def register(ctx) -> None:
    ctx.register_hook("post_tool_call", post_tool_call)
    ctx.register_hook("post_llm_call", post_llm_call)
    ctx.register_hook("on_session_end", on_session_end)
    ctx.register_tool(
        name="rsi_platform_runtime_capabilities",
        toolset="rsi-platform-runtime",
        schema={
            "name": "rsi_platform_runtime_capabilities",
            "description": "Report RSI platform runtime plugin static capabilities.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        handler=_capabilities_handler,
        description="Report RSI platform runtime plugin static capabilities.",
    )
