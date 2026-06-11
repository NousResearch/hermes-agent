"""pi_observability — Hermes plugin for pi-compatible event export.

The plugin is intentionally stdlib-only and fail-open. When enabled through the
Hermes plugin system, it emits bounded, batched JSON events to ``POST /events``
on the configured pi-observability server.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

MAX_TEXT_FIELD = 32_000
MAX_ARGS_BYTES = 16_000
MAX_RESULT_BYTES = 32_000

_LOCK = threading.RLock()
_RUNTIME: "PiObservabilityRuntime | None" = None


@dataclass(frozen=True)
class PiObservabilityConfig:
    server_url: str = "http://127.0.0.1:43190"
    token: str = ""
    pool: str = "hermes"
    agent_name: str = "Hermes Agent"
    tags: tuple[str, ...] = ("hermes",)
    timeout_s: float = 2.0
    queue_max: int = 1000
    batch_size: int = 25
    batch_interval_s: float = 0.5
    enabled: bool = True


@dataclass
class SessionState:
    session_id: str
    seq: int = 0
    started: bool = False
    agent_started: bool = False
    turn_index: int = -1
    active_turn: bool = False
    provider: str = ""
    model: str = ""
    cwd: str = ""
    usage: dict[str, Any] = field(default_factory=dict)


class EventSender:
    """Bounded async JSON sender; drops instead of blocking the agent."""

    def __init__(self, config: PiObservabilityConfig) -> None:
        self.config = config
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=max(1, config.queue_max))
        self._stopped = threading.Event()
        self.sent_batches: list[list[dict[str, Any]]] = []  # useful for tests/subclasses
        self._thread = threading.Thread(target=self._run, name="hermes-pi-observability", daemon=True)
        self._thread.start()

    def send(self, event: dict[str, Any]) -> None:
        if not self.config.enabled:
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            logger.debug("pi-observability queue full; dropping event")

    def close(self, *, drain: bool = True) -> None:
        if self._stopped.is_set():
            return
        self._stopped.set()
        if drain:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            self._thread.join(timeout=max(0.1, self.config.timeout_s + 0.2))

    def _run(self) -> None:
        batch: list[dict[str, Any]] = []
        deadline = time.monotonic() + self.config.batch_interval_s
        while not self._stopped.is_set():
            timeout = max(0.0, deadline - time.monotonic())
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                item = None
            if item is None:
                if batch:
                    self._post_batch(batch)
                    batch = []
                deadline = time.monotonic() + self.config.batch_interval_s
                if self._stopped.is_set():
                    break
                continue
            batch.append(item)
            if len(batch) >= self.config.batch_size:
                self._post_batch(batch)
                batch = []
                deadline = time.monotonic() + self.config.batch_interval_s

    def _post_batch(self, batch: list[dict[str, Any]]) -> None:
        self.sent_batches.append(list(batch))
        try:
            _post_events(self.config, batch)
        except Exception as exc:  # pragma: no cover - fail-open network path
            logger.debug("pi-observability POST failed: %s", exc, exc_info=True)


class PiObservabilityRuntime:
    def __init__(self, config: PiObservabilityConfig, sender: Optional[EventSender] = None) -> None:
        self.config = config
        self.sender = sender or EventSender(config)
        self.sessions: dict[str, SessionState] = {}
        self._lock = threading.RLock()

    def emit(self, event_type: str, payload: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        state = self.ensure_session(kwargs)
        with self._lock:
            provider = str(kwargs.get("provider") or state.provider or "")
            model = str(kwargs.get("model") or state.model or "")
            if provider:
                state.provider = provider
            if model:
                state.model = model
            event = _envelope(
                event_type,
                payload,
                state=state,
                config=self.config,
                provider=provider or None,
                model=model or None,
                session_file=_str_or_none(kwargs.get("session_file")),
            )
            state.seq += 1
        self.sender.send(event)
        return event

    def ensure_session(self, kwargs: dict[str, Any]) -> SessionState:
        session_id = _session_id(kwargs)
        with self._lock:
            state = self.sessions.get(session_id)
            if state is None:
                state = SessionState(session_id=session_id, cwd=str(kwargs.get("cwd") or os.getcwd()))
                self.sessions[session_id] = state
            return state

    def ensure_started(self, kwargs: dict[str, Any]) -> None:
        state = self.ensure_session(kwargs)
        if not state.started:
            state.started = True
            self.emit("session_start", {"reason": str(kwargs.get("reason") or "startup")}, kwargs)
        if not state.agent_started:
            state.agent_started = True
            prompt = _prompt_from_kwargs(kwargs)
            self.emit(
                "agent_start",
                {"prompt": prompt, "images_count": _images_count(kwargs), "session_id": state.session_id},
                kwargs,
            )

    def start_turn(self, kwargs: dict[str, Any]) -> None:
        self.ensure_started(kwargs)
        state = self.ensure_session(kwargs)
        if state.active_turn:
            return
        state.turn_index += 1
        state.active_turn = True
        self.emit("turn_start", {"turn_index": state.turn_index}, kwargs)

    def end_turn(self, kwargs: dict[str, Any], usage: Optional[dict[str, Any]] = None) -> None:
        state = self.ensure_session(kwargs)
        if not state.active_turn:
            return
        payload: dict[str, Any] = {"turn_index": state.turn_index}
        if usage is not None:
            payload["usage"] = usage
            state.usage = usage
        self.emit("turn_end", payload, kwargs)
        state.active_turn = False

    def close(self) -> None:
        self.sender.close()


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return default


def _env_bool(*names: str, default: bool = True) -> bool:
    value = _env(*names)
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def load_config_from_env() -> PiObservabilityConfig:
    tags_raw = _env("HERMES_PI_OBS_TAGS", "OBS_TAGS", default="hermes")
    tags = tuple(tag.strip() for tag in tags_raw.split(",") if tag.strip()) or ("hermes",)
    return PiObservabilityConfig(
        server_url=_env("HERMES_PI_OBS_SERVER_URL", "OBS_SERVER_URL", default="http://127.0.0.1:43190").rstrip("/"),
        token=_env("HERMES_PI_OBS_TOKEN", "OBS_TOKEN"),
        pool=_env("HERMES_PI_OBS_POOL", "OBS_POOL", default="hermes"),
        agent_name=_env("HERMES_PI_OBS_AGENT_NAME", "OBS_AGENT_NAME", default="Hermes Agent"),
        tags=tags,
        timeout_s=_float_env("HERMES_PI_OBS_TIMEOUT_S", "OBS_TIMEOUT_S", default=2.0),
        queue_max=_int_env("HERMES_PI_OBS_QUEUE_MAX", "OBS_QUEUE_MAX", default=1000),
        batch_size=_int_env("HERMES_PI_OBS_BATCH_SIZE", "OBS_BATCH_SIZE", default=25),
        batch_interval_s=_float_env("HERMES_PI_OBS_BATCH_INTERVAL_S", "OBS_BATCH_INTERVAL_S", default=0.5),
        enabled=_env_bool("HERMES_PI_OBS_ENABLED", "OBS_ENABLED", default=True),
    )


def _int_env(*names: str, default: int) -> int:
    try:
        return max(1, int(_env(*names, default=str(default))))
    except ValueError:
        return default


def _float_env(*names: str, default: float) -> float:
    try:
        return max(0.05, float(_env(*names, default=str(default))))
    except ValueError:
        return default


def _get_runtime() -> PiObservabilityRuntime:
    global _RUNTIME
    with _LOCK:
        if _RUNTIME is None:
            _RUNTIME = PiObservabilityRuntime(load_config_from_env())
        return _RUNTIME


def _post_events(config: PiObservabilityConfig, events: list[dict[str, Any]]) -> None:
    url = urllib.parse.urljoin(config.server_url + "/", "events")
    data = json.dumps(events, separators=(",", ":")).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if config.token:
        headers["Authorization"] = f"Bearer {config.token}"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=config.timeout_s) as response:  # noqa: S310 - configured local/remote URL
        response.read()


def _envelope(
    event_type: str,
    payload: dict[str, Any],
    *,
    state: SessionState,
    config: PiObservabilityConfig,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    session_file: Optional[str] = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "event_id": str(uuid.uuid4()),
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "type": event_type,
        "session_id": state.session_id,
        "cwd": state.cwd or os.getcwd(),
        "agent_name": config.agent_name,
        "pool": config.pool,
        "tags": list(config.tags),
        "payload": payload,
        "seq": state.seq,
    }
    if session_file:
        event["session_file"] = session_file
    if provider:
        event["provider"] = provider
    if model:
        event["model"] = model
    return event


def normalize_usage(usage: Any = None, response: Any = None, **kwargs: Any) -> dict[str, Any]:
    raw = usage if usage is not None else _value(response, "usage", {})
    data = _jsonable(raw) if raw is not None else {}
    if not isinstance(data, dict):
        data = {}
    input_tokens = _first_int(data, "input", "input_tokens", "prompt_tokens")
    output_tokens = _first_int(data, "output", "output_tokens", "completion_tokens")
    cache_read = _first_int(data, "cache_read", "cache_read_tokens", "cached_tokens", "prompt_cache_hit_tokens")
    cache_write = _first_int(data, "cache_write", "cache_write_tokens", "cache_creation_input_tokens", "prompt_cache_miss_tokens")
    total = _first_int(data, "total_tokens", "total") or (input_tokens + output_tokens + cache_read + cache_write)
    cost_total = _first_float(data, "cost_total", "total_cost", "cost", "amount_usd")
    if not cost_total:
        cost_total = _estimate_cost(kwargs, input_tokens, output_tokens, cache_read, cache_write)
    return {
        "input": input_tokens,
        "output": output_tokens,
        "cache_read": cache_read,
        "cache_write": cache_write,
        "total_tokens": total,
        "cost_total": float(cost_total or 0.0),
    }


def _estimate_cost(kwargs: dict[str, Any], input_tokens: int, output_tokens: int, cache_read: int, cache_write: int) -> float:
    try:
        from agent.usage_pricing import CanonicalUsage, estimate_usage_cost

        cost = estimate_usage_cost(
            str(kwargs.get("model") or ""),
            CanonicalUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
            ),
            provider=str(kwargs.get("provider") or ""),
            base_url=str(kwargs.get("base_url") or ""),
            api_key="",
        )
        return float(cost.amount_usd or 0.0)
    except Exception:
        return 0.0


def _first_int(data: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = data.get(key)
        if isinstance(value, dict):
            nested = _first_int(value, *keys)
            if nested:
                return nested
        try:
            if value is not None:
                return max(0, int(value))
        except (TypeError, ValueError):
            continue
    return 0


def _first_float(data: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = data.get(key)
        try:
            if value is not None:
                return max(0.0, float(value))
        except (TypeError, ValueError):
            continue
    return 0.0


def _session_id(kwargs: dict[str, Any]) -> str:
    return str(kwargs.get("session_id") or kwargs.get("task_id") or f"thread-{threading.get_ident()}")


def _prompt_from_kwargs(kwargs: dict[str, Any]) -> str:
    for key in ("user_message", "prompt", "message"):
        value = kwargs.get(key)
        if isinstance(value, str):
            return _truncate_text(value, MAX_TEXT_FIELD)[0]
    messages = kwargs.get("messages") or kwargs.get("request_messages") or kwargs.get("conversation_history")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                return _truncate_text(str(message.get("content") or ""), MAX_TEXT_FIELD)[0]
    return ""


def _images_count(kwargs: dict[str, Any]) -> int:
    value = kwargs.get("images_count")
    if isinstance(value, int):
        return value
    return 0


def _assistant_text(kwargs: dict[str, Any]) -> str:
    for key in ("assistant_response", "content", "text"):
        value = kwargs.get(key)
        if isinstance(value, str):
            return _truncate_text(value, MAX_TEXT_FIELD)[0]
    msg = kwargs.get("assistant_message")
    value = _value(msg, "content")
    return _truncate_text(str(value or ""), MAX_TEXT_FIELD)[0]


def _assistant_tool_ids(kwargs: dict[str, Any]) -> list[str]:
    msg = kwargs.get("assistant_message")
    calls = _value(msg, "tool_calls", None) or kwargs.get("tool_calls") or []
    ids: list[str] = []
    if isinstance(calls, list):
        for call in calls:
            call_id = _value(call, "id")
            if call_id:
                ids.append(str(call_id))
    count = kwargs.get("assistant_tool_call_count")
    if not ids and isinstance(count, int) and count > 0:
        ids = [f"tool-{idx}" for idx in range(count)]
    return ids


def _stop_reason(kwargs: dict[str, Any]) -> str:
    reason = kwargs.get("finish_reason") or kwargs.get("stop_reason") or "stop"
    if reason == "tool_calls":
        return "toolUse"
    return str(reason)


def _tool_args(args: Any) -> tuple[dict[str, Any], bool]:
    value = _jsonable(args if args is not None else {})
    if not isinstance(value, dict):
        value = {"value": value}
    encoded = json.dumps(value, default=str, separators=(",", ":"))
    if len(encoded.encode("utf-8")) <= MAX_ARGS_BYTES:
        return value, False
    return {"_truncated": True, "preview": _truncate_text(encoded, MAX_ARGS_BYTES)[0]}, True


def _tool_result_payload(result: Any) -> tuple[str, bool, dict[str, Any]]:
    value = _jsonable(result)
    if isinstance(value, dict):
        text = str(value.get("content") or value.get("text") or value.get("stdout") or value.get("result") or "")
        details = {k: v for k, v in value.items() if k in {"exit_code", "status", "error", "duration_ms"}}
    else:
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        details = {}
    truncated_text, truncated = _truncate_text(text, MAX_RESULT_BYTES)
    return truncated_text, truncated, details


def _truncate_text(text: str, max_bytes: int) -> tuple[str, bool]:
    if len(text.encode("utf-8")) <= max_bytes:
        return text, False
    marker = "\n…[truncated]"
    budget = max(0, max_bytes - len(marker.encode("utf-8")))
    out = ""
    used = 0
    for ch in text:
        size = len(ch.encode("utf-8"))
        if used + size > budget:
            break
        out += ch
        used += size
    return out + marker, True


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump(mode="json"))
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _jsonable(vars(value))
        except Exception:
            pass
    return str(value)


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _str_or_none(value: Any) -> Optional[str]:
    return str(value) if value else None


def _safe(fn) -> None:
    try:
        fn()
    except Exception as exc:
        logger.debug("pi-observability hook failed: %s", exc, exc_info=True)


def on_pre_llm_call(**kwargs: Any) -> None:
    runtime = _get_runtime()
    _safe(lambda: runtime.start_turn(kwargs))


def on_pre_api_request(**kwargs: Any) -> None:
    runtime = _get_runtime()
    _safe(lambda: runtime.start_turn(kwargs))


def on_post_api_request(**kwargs: Any) -> None:
    runtime = _get_runtime()

    def _record() -> None:
        usage = normalize_usage(**kwargs)
        state = runtime.ensure_session(kwargs)
        payload = {
            "text": _assistant_text(kwargs),
            "thinking": str(kwargs.get("reasoning") or ""),
            "tool_call_ids": _assistant_tool_ids(kwargs),
            "stop_reason": _stop_reason(kwargs),
            "usage": usage,
            "turn_index": state.turn_index,
        }
        duration = kwargs.get("api_duration")
        if isinstance(duration, (int, float)) and duration > 0:
            payload["latency_ms"] = int(duration * 1000)
        runtime.emit("assistant_message", payload, kwargs)
        if not payload["tool_call_ids"]:
            runtime.end_turn(kwargs, usage)

    _safe(_record)


def on_post_llm_call(**kwargs: Any) -> None:
    runtime = _get_runtime()
    _safe(lambda: runtime.end_turn(kwargs, normalize_usage(**kwargs)))


def on_api_request_error(**kwargs: Any) -> None:
    runtime = _get_runtime()
    _safe(lambda: runtime.emit("error", {"message": str(kwargs.get("error") or kwargs.get("message") or ""), "where": "llm"}, kwargs))


def on_pre_tool_call(**kwargs: Any) -> None:
    runtime = _get_runtime()

    def _record() -> None:
        args, truncated = _tool_args(kwargs.get("args"))
        runtime.emit(
            "tool_call",
            {
                "tool_call_id": str(kwargs.get("tool_call_id") or ""),
                "tool_name": str(kwargs.get("tool_name") or "tool"),
                "args": args,
                "args_truncated": truncated,
            },
            kwargs,
        )

    _safe(_record)


def on_post_tool_call(**kwargs: Any) -> None:
    runtime = _get_runtime()

    def _record() -> None:
        content, truncated, details = _tool_result_payload(kwargs.get("result"))
        status = kwargs.get("status")
        is_error = bool(kwargs.get("is_error") or status == "error" or (isinstance(details.get("exit_code"), int) and details["exit_code"] != 0))
        payload: dict[str, Any] = {
            "tool_call_id": str(kwargs.get("tool_call_id") or ""),
            "tool_name": str(kwargs.get("tool_name") or "tool"),
            "content_text": content,
            "content_truncated": truncated,
            "is_error": is_error,
        }
        if details:
            payload["details_summary"] = details
        runtime.emit("tool_result", payload, kwargs)
        if is_error:
            runtime.emit("error", {"message": content[:500], "where": f"tool:{payload['tool_name']}"}, kwargs)

    _safe(_record)


def register(ctx) -> None:
    ctx.register_hook("pre_api_request", on_pre_api_request)
    ctx.register_hook("post_api_request", on_post_api_request)
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("post_llm_call", on_post_llm_call)
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("api_request_error", on_api_request_error)


def reset_for_tests() -> None:
    global _RUNTIME
    with _LOCK:
        if _RUNTIME is not None:
            _RUNTIME.close()
        _RUNTIME = None


atexit.register(reset_for_tests)
