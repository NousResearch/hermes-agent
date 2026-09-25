"""Local, content-free run diagnostics. Enabled explicitly as a Hermes plugin.

The provider/turn lifecycle is the source of truth. A snapshot is replaced after
each boundary so even a killed process leaves a visibly *running* attempt,
never an invented success. This module has no network client or payload capture.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path
from functools import wraps
from typing import Any

from hermes_constants import get_hermes_home

_LOCK = threading.RLock()
_RUNS: dict[tuple[str, str], dict[str, Any]] = {}
_CHILD_PARENTS: dict[tuple[str, str], str] = {}
_SAFE_ID = re.compile(r"^[\w.:-]{1,160}$", re.ASCII)
_SAFE_ROUTE = re.compile(r"^[\w./:@+-]{1,160}$", re.ASCII)
logger = logging.getLogger(__name__)


def _id(value: Any) -> str | None:
    text = str(value or "")
    return text if _SAFE_ID.fullmatch(text) else None


def _route(value: Any) -> str | None:
    text = str(value or "")
    return text if _SAFE_ROUTE.fullmatch(text) else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if 0 <= number < 1e15 else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number.is_integer() else None


def _exit_reason(value: Any) -> str | None:
    """Do not persist exception snippets embedded in a turn exit label."""
    raw = str(value or "")
    if raw.startswith("text_response("):
        match = re.fullmatch(r"text_response\(finish_reason=([a-z_]+)\)", raw)
        return match.group(0) if match else "text_response"
    return _route(raw.split("(", 1)[0])


def _key(event: dict[str, Any]) -> tuple[str, str] | None:
    turn_id = _id(event.get("turn_id"))
    return (str(get_hermes_home()), turn_id) if turn_id else None


def _run(event: dict[str, Any], *, create: bool = True) -> dict[str, Any] | None:
    key = _key(event)
    if key is None:
        return None
    existing = _RUNS.get(key)
    if existing is not None or not create:
        return existing
    from hermes_cli.version_info import get_version_info

    run_id = uuid.uuid4().hex
    home = Path(key[0])
    data = {
        "schema_version": 1,
        "run_id": run_id,
        "parent_run_id": _CHILD_PARENTS.get((key[0], _id(event.get("session_id")) or "")),
        "task_id": _id(event.get("task_id")),
        "turn_id": key[1],
        "session_id": _id(event.get("session_id")),
        "platform": _route(event.get("platform")),
        "agent_role": _route(event.get("agent_role")),
        "phase": _route(event.get("phase")),
        "hermes_commit": get_version_info().commit,
        "started_at": _number(event.get("started_at")) or time.time(),
        "ended_at": None,
        "status": "running",
        "turn_exit_reason": None,
        "attempts": [],
        "tools": [],
        "subagents": [],
        "compactions": [],
    }
    state = {"data": data, "path": home / "logs" / "run-metrics" / f"{run_id}.json"}
    _RUNS[key] = state
    return state


def _summary(data: dict[str, Any]) -> dict[str, Any]:
    attempts = data["attempts"]
    tools = data["tools"]
    physical_attempts = sum(len(a["transport_attempts"]) or 1 for a in attempts)
    complete_transport_coverage = all(
        a["transport_attempts"] or a.get("api_mode") not in {"codex_responses", "bedrock_converse"}
        for a in attempts
    )
    model_time = sum(
        sum(t["duration_s"] or 0 for t in a["transport_attempts"])
        if a["transport_attempts"] else (a["duration_s"] or 0)
        for a in attempts
    )
    actual_inputs = [a["input_tokens"] for a in attempts if a["token_source"] == "provider_reported" and a["input_tokens"] is not None]
    actual_outputs = [a["output_tokens"] for a in attempts if a["token_source"] == "provider_reported" and a["output_tokens"] is not None]
    decode_samples = []
    for a in attempts:
        delivered = next((t for t in reversed(a["transport_attempts"])
                          if t["token_source"] == "provider_reported"), None)
        timing = delivered or (a if not a["transport_attempts"] else None)
        if timing and a["output_tokens"] is not None and timing["duration_s"] is not None:
            first_delta = timing["time_to_first_delta_s"]
            if first_delta is not None and timing["duration_s"] > first_delta:
                decode_samples.append((a["output_tokens"], timing["duration_s"] - first_delta))
    return {
        "logical_model_calls": len({a["api_request_id"] for a in attempts if a["api_request_id"]}),
        "provider_attempts": physical_attempts,
        "provider_attempts_complete": complete_transport_coverage,
        "retry_attempts": physical_attempts - len({a["api_request_id"] for a in attempts if a["api_request_id"]}),
        "model_time_s": round(model_time, 3),
        "tool_time_s": round(sum(t["duration_s"] or 0 for t in tools), 3),
        "provider_reported_input_tokens": sum(actual_inputs) if actual_inputs else None,
        "provider_reported_output_tokens": sum(actual_outputs) if actual_outputs else None,
        "attempts_with_missing_usage": sum(
            sum(t["token_source"] != "provider_reported" for t in a["transport_attempts"])
            if a["transport_attempts"] else int(a["token_source"] != "provider_reported")
            for a in attempts
        ),
        "measured_decode_tokens_per_s": round(sum(t for t, _ in decode_samples) / sum(s for _, s in decode_samples), 2) if decode_samples else None,
        "output_capped_attempts": sum(a["status"] == "output_capped" for a in attempts),
        "tool_calls": len(tools),
        "compactions": len(data["compactions"]),
        "wall_time_s": round(max(0, (data["ended_at"] or time.time()) - data["started_at"]), 3),
    }


def _persist(state: dict[str, Any]) -> None:
    data = state["data"]
    path: Path = state["path"]
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = {**data, "summary": _summary(data)}
    fd, temp_path = tempfile.mkstemp(prefix=f".{data['run_id']}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            os.chmod(temp_path, 0o600)
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _attempt(state: dict[str, Any], event: dict[str, Any], *, create: bool = True) -> dict[str, Any] | None:
    request_id = _id(event.get("api_request_id"))
    if not request_id:
        return None
    entries = state["data"]["attempts"]
    retry = _integer(event.get("retry_count"))
    for entry in reversed(entries):
        if entry["api_request_id"] == request_id and entry["status"] == "running" and (retry is None or entry["retry_count"] == retry):
            return entry
    if not create:
        return None
    entry = {
        "api_request_id": request_id, "retry_count": retry or 0,
        "provider": _route(event.get("provider")), "model": _route(event.get("model")),
        "api_mode": _route(event.get("api_mode")),
        "response_model": None, "context_limit_tokens": _integer(event.get("context_limit_tokens")),
        "max_output_tokens": _integer(event.get("max_tokens")),
        "estimated_input_tokens": _integer(event.get("approx_input_tokens")),
        "estimated_context_utilization": None,
        "input_tokens": None, "uncached_input_tokens": None,
        "cache_read_tokens": None, "cache_write_tokens": None,
        "output_tokens": None, "reasoning_tokens": None,
        "token_source": "unavailable", "started_at": _number(event.get("attempt_started_at")) or _number(event.get("started_at")) or time.time(),
        "ended_at": None, "duration_s": None, "time_to_first_chunk_s": None,
        "time_to_first_delta_s": None, "finish_reason": None, "status": "running",
        "error_type": None, "error_reason": None,
        "transport_attempts": [],
    }
    if entry["context_limit_tokens"]:
        estimate = entry["estimated_input_tokens"]
        if estimate is not None:
            entry["estimated_context_utilization"] = round(estimate / entry["context_limit_tokens"], 4)
    entries.append(entry)
    return entry


def on_pre_llm_call(**event: Any) -> None:
    with _LOCK:
        state = _run(event)
        if state:
            for field in ("platform", "agent_role", "phase"):
                if state["data"].get(field) is None:
                    state["data"][field] = _route(event.get(field))
            _persist(state)


def on_turn_start(**event: Any) -> None:
    """The turn exists before preflight compression and its first model call."""
    with _LOCK:
        state = _run(event)
        if state:
            _persist(state)


def on_pre_api_request(**event: Any) -> None:
    with _LOCK:
        state = _run(event)
        if state:
            # Some core recovery paths restart without an api_request_error hook.
            # Preserve the observed attempt, but never invent its terminal cause.
            request_id = _id(event.get("api_request_id"))
            for prior in state["data"]["attempts"]:
                if prior["status"] == "running" and prior["api_request_id"] == request_id:
                    prior["status"] = "outcome_unobserved"
                    prior["ended_at"] = _number(event.get("attempt_started_at")) or _number(event.get("started_at")) or time.time()
                    prior["duration_s"] = round(max(0, prior["ended_at"] - prior["started_at"]), 3)
                    for nested in prior["transport_attempts"]:
                        if nested["status"] == "running":
                            nested["status"] = "outcome_unobserved"
                            nested["ended_at"] = prior["ended_at"]
                            nested["duration_s"] = round(max(0, nested["ended_at"] - nested["started_at"]), 3)
            _attempt(state, event)
            _persist(state)


def on_stream_attempt(**event: Any) -> None:
    """Nested transport events reveal retries hidden inside one core API call."""
    with _LOCK:
        state = _run(event, create=False)
        if not state:
            return
        attempt = _attempt(state, event, create=False)
        stream_id = _integer(event.get("stream_attempt_id"))
        if not attempt or stream_id is None:
            return
        entries = attempt["transport_attempts"]
        if event.get("phase") == "start":
            entries.append({
                "stream_attempt_id": stream_id, "started_at": _number(event.get("started_at")) or time.time(),
                "ended_at": None, "duration_s": None, "status": "running", "error_type": None,
                "time_to_first_chunk_s": None, "time_to_first_delta_s": None,
                "finish_reason": None, "input_tokens": None, "output_tokens": None,
                "reasoning_tokens": None, "token_source": "unavailable",
            })
        elif event.get("phase") == "end":
            nested = next((t for t in reversed(entries)
                           if t["stream_attempt_id"] == stream_id and t["status"] == "running"), None)
            if nested is None:
                return
            nested["ended_at"] = _number(event.get("ended_at")) or time.time()
            nested["duration_s"] = round(max(0, nested["ended_at"] - nested["started_at"]), 3)
            nested["status"] = "error" if event.get("status") == "error" else "completed"
            nested["error_type"] = _route(event.get("error_type"))
            for source, target in (("first_chunk_at", "time_to_first_chunk_s"),
                                   ("first_delta_at", "time_to_first_delta_s")):
                at = _number(event.get(source))
                if at is not None and at >= nested["started_at"]:
                    nested[target] = round(at - nested["started_at"], 3)
        else:
            return
        _persist(state)


def _finish_attempt(event: dict[str, Any], *, error: bool) -> None:
    with _LOCK:
        state = _run(event)
        if not state:
            return
        attempt = _attempt(state, event, create=False)
        if attempt is None and error:
            # HTTP-200 content_filter is first reported as a completed response,
            # then the refusal handler emits api_request_error for that SAME wire
            # attempt. Keep its usage and do not manufacture a second request.
            request_id = _id(event.get("api_request_id"))
            retry = _integer(event.get("retry_count"))
            attempt = next((a for a in reversed(state["data"]["attempts"])
                            if a["api_request_id"] == request_id
                            and (retry is None or a["retry_count"] == retry)
                            and a["finish_reason"] == "content_filter"), None)
        if attempt is None:
            attempt = _attempt(state, event)
        if not attempt:
            return
        ended_at = _number(event.get("ended_at")) or time.time()
        started_at = attempt["started_at"]
        if attempt["ended_at"] is None:
            attempt["ended_at"] = ended_at
            attempt["duration_s"] = round(max(0, ended_at - started_at), 3)
            for source, target in (("first_chunk_at", "time_to_first_chunk_s"), ("first_delta_at", "time_to_first_delta_s")):
                at = _number(event.get(source))
                if at is not None and at >= started_at:
                    attempt[target] = round(at - started_at, 3)
        if error:
            error_data = event.get("error") or {}
            attempt["status"] = "error"
            attempt["error_type"] = _route(error_data.get("type")) if isinstance(error_data, dict) else None
            attempt["error_reason"] = _route(event.get("reason"))
        else:
            attempt["finish_reason"] = _route(event.get("finish_reason"))
            attempt["synthetic_response"] = bool(event.get("synthetic_response"))
            attempt["status"] = ("incomplete_stream" if attempt["synthetic_response"] else
                                 "output_capped" if attempt["finish_reason"] == "length" else "completed")
            attempt["response_model"] = _route(event.get("response_model"))
            usage = event.get("usage")
            if isinstance(usage, dict):
                presence = usage.get("reported_usage_fields")

                def reported(field: str) -> bool:
                    return bool(presence.get(field)) if isinstance(presence, dict) else field in usage

                attempt["input_tokens"] = (_integer(usage.get("prompt_tokens")) if reported("prompt_tokens")
                                           else _integer(usage.get("input_tokens"))
                                           if not isinstance(presence, dict) and reported("input_tokens") else None)
                for source, target in (("input_tokens", "uncached_input_tokens"),
                                       ("cache_read_tokens", "cache_read_tokens"),
                                       ("cache_write_tokens", "cache_write_tokens"),
                                       ("output_tokens", "output_tokens"),
                                       ("reasoning_tokens", "reasoning_tokens")):
                    attempt[target] = _integer(usage.get(source)) if reported(source) else None
                if any(attempt[field] is not None for field in ("input_tokens", "output_tokens", "reasoning_tokens")):
                    attempt["token_source"] = "provider_reported"
            delivered = next((t for t in reversed(attempt["transport_attempts"])
                              if t["status"] == "completed"), None)
            if delivered:
                delivered["finish_reason"] = attempt["finish_reason"]
                delivered["input_tokens"] = attempt["input_tokens"]
                delivered["output_tokens"] = attempt["output_tokens"]
                delivered["reasoning_tokens"] = attempt["reasoning_tokens"]
                delivered["token_source"] = attempt["token_source"]
        _persist(state)


def on_post_api_request(**event: Any) -> None:
    _finish_attempt(event, error=False)


def on_api_request_error(**event: Any) -> None:
    _finish_attempt(event, error=True)


def on_post_tool_call(**event: Any) -> None:
    with _LOCK:
        state = _run(event, create=False)
        if not state:
            return
        call_id = _id(event.get("tool_call_id"))
        tool = next((t for t in reversed(state["data"]["tools"]) if t["tool_call_id"] == call_id and t["status"] == "running"), None)
        if tool is None:
            tool = {"tool_call_id": call_id, "tool_name": _route(event.get("tool_name")), "status": "running", "duration_s": None}
            state["data"]["tools"].append(tool)
        tool["status"] = _route(event.get("status")) or "unknown"
        milliseconds = _number(event.get("duration_ms"))
        tool["duration_s"] = round(milliseconds / 1000, 3) if milliseconds is not None else None
        _persist(state)


def on_subagent_start(**event: Any) -> None:
    with _LOCK:
        parent_key = _key({"turn_id": event.get("parent_turn_id")})
        parent = _RUNS.get(parent_key) if parent_key else None
        child_id = _id(event.get("child_session_id"))
        if parent and child_id:
            _CHILD_PARENTS[(parent_key[0], child_id)] = parent["data"]["run_id"]
            parent["data"]["subagents"].append({"session_id": child_id, "role": _route(event.get("child_role")), "status": "running", "duration_s": None})
            _persist(parent)


def on_subagent_stop(**event: Any) -> None:
    with _LOCK:
        parent_key = _key({"turn_id": event.get("parent_turn_id")})
        parent = _RUNS.get(parent_key) if parent_key else None
        child_id = _id(event.get("child_session_id"))
        if parent and child_id:
            child = next((c for c in reversed(parent["data"]["subagents"]) if c["session_id"] == child_id and c["status"] == "running"), None)
            if child:
                child["status"] = _route(event.get("child_status")) or "unknown"
                milliseconds = _number(event.get("duration_ms"))
                child["duration_s"] = round(milliseconds / 1000, 3) if milliseconds is not None else None
                _persist(parent)
            _CHILD_PARENTS.pop((parent_key[0], child_id), None)


def on_context_compaction(**event: Any) -> None:
    with _LOCK:
        state = _run(event, create=False)
        if state:
            state["data"]["compactions"].append({
                "at": time.time(), "old_session_id": _id(event.get("old_session_id")),
                "new_session_id": _id(event.get("session_id")),
            })
            _persist(state)


def on_session_end(**event: Any) -> None:
    with _LOCK:
        state = _run(event, create=False)
        if not state:
            return
        data = state["data"]
        data["ended_at"] = time.time()
        data["turn_exit_reason"] = reason = _exit_reason(
            event.get("turn_exit_reason") or event.get("failure_reason"))
        if event.get("interrupted"):
            data["status"] = "interrupted"
        elif event.get("failed") or not event.get("completed"):
            data["status"] = "failed"
        elif (reason == "text_response(finish_reason=length)"
              and any(a["status"] == "output_capped" for a in data["attempts"])):
            data["status"] = "output_capped"
        else:
            data["status"] = "completed"
        for attempt in data["attempts"]:
            if attempt["status"] == "running":
                attempt["status"] = "aborted"
                attempt["ended_at"] = data["ended_at"]
                attempt["duration_s"] = round(max(0, data["ended_at"] - attempt["started_at"]), 3)
                for nested in attempt["transport_attempts"]:
                    if nested["status"] == "running":
                        nested["status"] = "aborted"
                        nested["ended_at"] = data["ended_at"]
                        nested["duration_s"] = round(max(0, data["ended_at"] - nested["started_at"]), 3)
        _persist(state)
        _RUNS.pop(_key(event), None)


def register(ctx: Any) -> None:
    for hook, callback in (
        ("on_turn_start", on_turn_start), ("pre_llm_call", on_pre_llm_call),
        ("pre_api_request", on_pre_api_request),
        ("stream_attempt", on_stream_attempt),
        ("post_api_request", on_post_api_request), ("api_request_error", on_api_request_error),
        ("post_tool_call", on_post_tool_call),
        ("subagent_start", on_subagent_start), ("subagent_stop", on_subagent_stop),
        ("context_compaction", on_context_compaction), ("on_session_end", on_session_end),
        ("on_turn_result", on_session_end),
    ):
        ctx.register_hook(hook, _fail_open(callback))


def _fail_open(callback: Any) -> Any:
    """Policy hooks fail closed; a metrics observer must never veto agent work."""
    @wraps(callback)
    def observe(**event: Any) -> None:
        try:
            callback(**event)
        except Exception:
            logger.warning("Run metrics observer failed", exc_info=True)

    return observe
