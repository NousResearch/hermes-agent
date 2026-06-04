"""Codex API runtime — App Server and Responses-API streaming paths.

Extracted from :class:`AIAgent` to keep the agent loop file focused.
Each function takes the parent ``AIAgent`` as its first argument
(``agent``).  AIAgent keeps thin forwarder methods for backward
compatibility.

* ``run_codex_app_server_turn`` — drives one turn through the
  ``codex_app_server`` subprocess client (used when a Codex CLI install
  is the active provider).
* ``run_codex_stream`` — streams a Codex Responses API call (the
  ``codex_responses`` api_mode).
* ``run_codex_create_stream_fallback`` — recovery path when the
  Responses ``stream=True`` initial create fails.
"""

from __future__ import annotations

from copy import deepcopy
import logging
import json
import os
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)


def run_codex_app_server_turn(
    agent,
    *,
    user_message: str,
    original_user_message: Any,
    messages: List[Dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool = False,
) -> Dict[str, Any]:
    """Codex app-server runtime path. Hands the entire turn to a `codex
    app-server` subprocess and projects its events back into Hermes'
    messages list so memory/skill review keep working.

    Called from run_conversation() when agent.api_mode == "codex_app_server".
    Returns the same dict shape as the chat_completions path.
    """
    from agent.transports.codex_app_server_session import CodexAppServerSession

    # Lazy session: one CodexAppServerSession per AIAgent instance.
    # Spawned on first turn, reused across turns, closed at AIAgent
    # shutdown (see _cleanup hook).
    if not hasattr(agent, "_codex_session") or agent._codex_session is None:
        cwd = getattr(agent, "session_cwd", None) or os.getcwd()
        # Approval callback: defer to Hermes' standard prompt flow if a
        # CLI thread has installed one. Gateway / cron contexts get the
        # codex-side fail-closed default.
        try:
            from tools.terminal_tool import _get_approval_callback
            approval_callback = _get_approval_callback()
        except Exception:
            approval_callback = None
        agent._codex_session = CodexAppServerSession(
            cwd=cwd,
            approval_callback=approval_callback,
        )

    # NOTE: the user message is ALREADY appended to messages by the
    # standard run_conversation() flow (line ~11823) before the early
    # return reaches us. Do NOT append again — that would duplicate.

    try:
        turn = agent._codex_session.run_turn(user_input=user_message)
    except Exception as exc:
        logger.exception("codex app-server turn failed")
        # Crash → unconditionally drop the session so the next turn
        # respawns from scratch instead of reusing a dead client.
        try:
            agent._codex_session.close()
        except Exception:
            pass
        agent._codex_session = None
        return {
            "final_response": (
                f"Codex app-server turn failed: {exc}. "
                f"Fall back to default runtime with `/codex-runtime auto`."
            ),
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "error": str(exc),
        }

    # If the turn signalled the underlying client is wedged (deadline
    # blown, post-tool watchdog tripped, OAuth refresh died, subprocess
    # exited), retire the session so the next turn respawns codex
    # rather than riding the broken process. Mirrors openclaw beta.8's
    # "retire timed-out app-server clients" fix.
    if getattr(turn, "should_retire", False):
        logger.warning(
            "codex app-server session retired (turn error: %s)",
            turn.error,
        )
        try:
            agent._codex_session.close()
        except Exception:
            pass
        agent._codex_session = None

    # Splice projected messages into the conversation. The projector emits
    # standard {role, content, tool_calls, tool_call_id} entries, which
    # is exactly what curator.py / sessions DB expect.
    if turn.projected_messages:
        messages.extend(turn.projected_messages)

    # Counter ticks for the agent-improvement loop.
    # _turns_since_memory and _user_turn_count are ALREADY incremented
    # in the run_conversation() pre-loop block (lines ~11793-11817) so we
    # do NOT touch them here — that would double-count.
    # Only _iters_since_skill needs explicit increment, since the
    # chat_completions loop bumps it per tool iteration (line ~12110)
    # and that loop is bypassed on this path.
    agent._iters_since_skill = (
        getattr(agent, "_iters_since_skill", 0) + turn.tool_iterations
    )

    # Now check the skill nudge AFTER iters were incremented — same
    # pattern the chat_completions path uses (line ~15432).
    should_review_skills = False
    if (
        agent._skill_nudge_interval > 0
        and agent._iters_since_skill >= agent._skill_nudge_interval
        and "skill_manage" in agent.valid_tool_names
    ):
        should_review_skills = True
        agent._iters_since_skill = 0

    # External memory provider sync (mirrors line ~15439). Skipped on
    # interrupt/error to avoid feeding partial transcripts to memory.
    if not turn.interrupted and turn.error is None:
        try:
            agent._sync_external_memory_for_turn(
                original_user_message=original_user_message,
                final_response=turn.final_text,
                interrupted=False,
            )
        except Exception:
            logger.debug("external memory sync raised", exc_info=True)

    # Background review fork — same cadence + signature as the default
    # path (line ~15449). Only fires when a trigger actually tripped AND
    # we have a real final response.
    if (
        turn.final_text
        and not turn.interrupted
        and (should_review_memory or should_review_skills)
    ):
        try:
            agent._spawn_background_review(
                messages_snapshot=list(messages),
                review_memory=should_review_memory,
                review_skills=should_review_skills,
            )
        except Exception:
            logger.debug("background review spawn raised", exc_info=True)

    return {
        "final_response": turn.final_text,
        "messages": messages,
        "api_calls": 1,  # one app-server "turn" maps to one logical API call
        "completed": not turn.interrupted and turn.error is None,
        "partial": turn.interrupted or turn.error is not None,
        "error": turn.error,
        "codex_thread_id": turn.thread_id,
        "codex_turn_id": turn.turn_id,
    }


# ---------------------------------------------------------------------------
# Event-driven Responses streaming
#
# OpenAI ships its consumer Codex backend (chatgpt.com/backend-api/codex) on
# a different schedule from the openai Python SDK.  The high-level
# ``client.responses.stream(...)`` helper reconstructs a typed Response from
# the terminal ``response.completed`` event's ``response.output`` field, and
# when that field drifts to ``null`` (gpt-5.5, May 2026) the SDK raises
# ``TypeError: 'NoneType' object is not iterable`` mid-iteration.
#
# We sidestep the whole class of failure by going one level lower:
# ``client.responses.create(stream=True)`` returns the raw AsyncIterable of
# SSE events, and we assemble the final response object purely from
# ``response.output_item.done`` events as they arrive.  We never read
# ``response.completed.response.output`` for content reconstruction, so the
# backend can return ``null``, ``[]``, a string, or omit the field entirely
# and we don't care.
#
# This mirrors what the OpenClaw TS implementation does for the same backend
# and is structurally immune to the bug class rather than patched.
# ---------------------------------------------------------------------------


_TERMINAL_EVENT_TYPES = frozenset({
    "response.completed",
    "response.incomplete",
    "response.failed",
})

_WEBSOCKET_TRUE_VALUES = frozenset({
    "1",
    "true",
    "yes",
    "on",
    "enabled",
    "enable",
    "websocket",
    "websocket_mode",
    "ws",
    "wss",
})

_WEBSOCKET_TRANSPORT_VALUES = frozenset({
    "websocket",
    "websocket_mode",
    "ws",
    "wss",
})

_RESPONSES_WEBSOCKETS_BETA_VALUE = "responses_websockets=2026-02-06"
_WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE = "websocket_connection_limit_reached"
_WEBSOCKET_ERROR_MARKER = "_hermes_codex_responses_websocket_error"
_CODEX_TURN_STATE_HEADER = "x-codex-turn-state"

_WEBSOCKET_BODY_EXCLUDED_KEYS = frozenset({
    "timeout",
    "extra_headers",
    "extra_body",
    "stream",
    "background",
})

_GENERATED_RESPONSE_ITEM_TYPES = frozenset({
    "function_call",
    "custom_tool_call",
    "reasoning",
})


def load_config() -> Dict[str, Any]:
    """Load Hermes config lazily so tests can monkeypatch this module symbol."""
    from hermes_cli.config import load_config as _load_config
    return _load_config()


def codex_responses_websocket_url(base_url: str) -> str:
    """Derive the Responses WebSocket endpoint from a provider base URL."""
    raw = str(base_url or "").strip().rstrip("/")
    if not raw:
        raise ValueError("codex_responses WebSocket transport requires provider base_url")

    parts = urlsplit(raw)
    scheme = parts.scheme.lower()
    if scheme == "https":
        websocket_scheme = "wss"
    elif scheme == "http":
        websocket_scheme = "ws"
    elif scheme in {"ws", "wss"}:
        websocket_scheme = scheme
    else:
        raise ValueError(
            f"unsupported codex_responses WebSocket base_url scheme: {parts.scheme!r}"
        )
    if not parts.netloc:
        raise ValueError("codex_responses WebSocket base_url must include a host")

    path = parts.path.rstrip("/")
    if not path.endswith("/responses"):
        path = f"{path}/responses" if path else "/responses"

    return urlunsplit((websocket_scheme, parts.netloc, path, parts.query, ""))


def _normalized_provider_name(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalized_provider_url(value: Any) -> str:
    return str(value or "").strip().rstrip("/").lower()


def _codex_websocket_transport_key(agent) -> tuple[str, str]:
    return (
        _normalized_provider_name(getattr(agent, "provider", "")),
        _normalized_provider_url(getattr(agent, "base_url", "")),
    )


def _codex_response_issuer_kind_for_agent(agent) -> str:
    from agent.codex_responses_adapter import _classify_responses_issuer

    return _classify_responses_issuer(
        is_xai_responses=bool(getattr(agent, "_is_xai_responses", False)),
        is_github_responses=bool(getattr(agent, "_is_github_responses", False)),
        is_codex_backend=bool(getattr(agent, "_is_codex_backend", False)),
        base_url=getattr(agent, "base_url", None),
    )


def _entry_base_url(entry: Dict[str, Any]) -> str:
    for key in ("base_url", "url", "api"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _entry_enables_codex_responses_websocket(entry: Dict[str, Any]) -> bool:
    raw_flag = entry.get("codex_responses_websocket")
    if isinstance(raw_flag, bool):
        return raw_flag
    if isinstance(raw_flag, str):
        return raw_flag.strip().lower() in _WEBSOCKET_TRUE_VALUES

    raw_transport = entry.get("codex_responses_transport")
    if isinstance(raw_transport, bool):
        return raw_transport
    if isinstance(raw_transport, str):
        return raw_transport.strip().lower() in _WEBSOCKET_TRANSPORT_VALUES

    return False


def _iter_config_provider_entries(config: Dict[str, Any]):
    providers = config.get("providers")
    if isinstance(providers, dict):
        for provider_key, entry in providers.items():
            if isinstance(entry, dict):
                yield str(provider_key), entry

    custom_providers = config.get("custom_providers")
    if isinstance(custom_providers, list):
        for entry in custom_providers:
            if isinstance(entry, dict):
                name = entry.get("name")
                yield str(name or ""), entry


def _provider_entry_name_matches_agent(agent, provider_key: str, entry: Dict[str, Any]) -> bool:
    provider = _normalized_provider_name(getattr(agent, "provider", ""))
    provider_no_custom = provider.removeprefix("custom:")
    names = {
        _normalized_provider_name(provider_key),
        _normalized_provider_name(entry.get("name")),
    }
    names = {name for name in names if name}
    return bool(provider and (provider in names or provider_no_custom in names))


def _provider_entry_url_matches_agent(agent, entry: Dict[str, Any]) -> bool:
    agent_url = _normalized_provider_url(getattr(agent, "base_url", ""))
    entry_url = _normalized_provider_url(_entry_base_url(entry))
    return bool(agent_url and entry_url and agent_url == entry_url)


def codex_responses_websocket_enabled(agent) -> bool:
    """Return True when this codex_responses provider explicitly opts into WebSocket."""
    if getattr(agent, "api_mode", None) != "codex_responses":
        return False
    fallback_key = getattr(agent, "_codex_responses_websocket_http_fallback_key", None)
    if fallback_key is not None and fallback_key == _codex_websocket_transport_key(agent):
        return False

    try:
        config = load_config()
    except Exception:
        logger.debug("Unable to load config for codex_responses WebSocket gate", exc_info=True)
        return False
    if not isinstance(config, dict):
        return False

    entries = list(_iter_config_provider_entries(config))
    name_matches = [
        entry
        for provider_key, entry in entries
        if _provider_entry_name_matches_agent(agent, provider_key, entry)
    ]
    if name_matches:
        return any(_entry_enables_codex_responses_websocket(entry) for entry in name_matches)

    for _provider_key, entry in entries:
        if _provider_entry_url_matches_agent(agent, entry) and _entry_enables_codex_responses_websocket(entry):
            return True

    return False


def _json_to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(
            **{str(key): _json_to_namespace(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_json_to_namespace(item) for item in value]
    return value


def _codex_websocket_response_body(api_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    body = {
        key: value
        for key, value in api_kwargs.items()
        if key not in _WEBSOCKET_BODY_EXCLUDED_KEYS and value is not None
    }
    extra_body = api_kwargs.get("extra_body")
    if isinstance(extra_body, dict):
        body.update(extra_body)
    body.pop("stream", None)
    body.pop("background", None)
    return body


def _header_present(headers: Dict[str, str], name: str) -> bool:
    target = name.lower()
    return any(str(key).lower() == target for key in headers)


def _pop_header_case_insensitive(headers: Dict[str, str], name: str) -> Optional[str]:
    target = name.lower()
    for key in list(headers):
        if str(key).lower() == target:
            return headers.pop(key)
    return None


def _set_header_if_missing(headers: Dict[str, str], name: str, value: str) -> None:
    if not value or _header_present(headers, name):
        return
    headers[name] = value


def _codex_responses_turn_state(agent) -> str:
    return str(getattr(agent, "_codex_responses_websocket_turn_state", "") or "").strip()


def _set_codex_responses_turn_state(agent, value: Any) -> None:
    turn_state = str(value or "").strip()
    if not turn_state:
        return
    agent._codex_responses_websocket_turn_state = turn_state
    logger.info(
        "codex_responses_websocket_event=turn_state_captured "
        "Codex Responses WebSocket captured same-turn sticky routing token. %s",
        getattr(agent, "_client_log_context", lambda: "")(),
    )


def _headers_get_case_insensitive(headers: Any, name: str) -> Optional[str]:
    if headers is None:
        return None
    try:
        value = headers.get(name)
    except Exception:
        value = None
    if value is not None:
        return str(value).strip()
    try:
        items = headers.items()
    except Exception:
        return None
    target = name.lower()
    for key, value in items:
        if str(key).lower() == target and value is not None:
            return str(value).strip()
    return None


def _codex_websocket_response_header(websocket: Any, raw_context: Any, name: str) -> Optional[str]:
    for source in (websocket, raw_context):
        response = getattr(source, "response", None)
        value = _headers_get_case_insensitive(getattr(response, "headers", None), name)
        if value:
            return value
        value = _headers_get_case_insensitive(getattr(source, "response_headers", None), name)
        if value:
            return value
        value = _headers_get_case_insensitive(getattr(source, "headers", None), name)
        if value:
            return value
    return None


def _codex_responses_session_header_values(agent) -> tuple[str, str]:
    session_id = str(getattr(agent, "session_id", "") or "").strip()
    thread_id = str(getattr(agent, "_thread_id", "") or "").strip() or session_id
    return session_id, thread_id


def _codex_websocket_headers(agent, api_kwargs: Dict[str, Any]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    extra_headers = api_kwargs.get("extra_headers")
    if isinstance(extra_headers, dict):
        headers.update({
            str(key): str(value)
            for key, value in extra_headers.items()
            if key and value is not None
        })
    # This is a dynamic handshake token. Keeping it in the stable session
    # headers would make later API calls recreate an otherwise reusable socket.
    _pop_header_case_insensitive(headers, _CODEX_TURN_STATE_HEADER)

    session_id, thread_id = _codex_responses_session_header_values(agent)
    _set_header_if_missing(headers, "session-id", session_id)
    _set_header_if_missing(headers, "thread-id", thread_id)
    _set_header_if_missing(headers, "x-client-request-id", thread_id)

    beta_key = next((key for key in headers if key.lower() == "openai-beta"), "OpenAI-Beta")
    beta_values = [
        value.strip()
        for value in str(headers.get(beta_key, "") or "").split(",")
        if value.strip()
    ]
    if _RESPONSES_WEBSOCKETS_BETA_VALUE not in beta_values:
        beta_values.append(_RESPONSES_WEBSOCKETS_BETA_VALUE)
    headers[beta_key] = ", ".join(beta_values)

    api_key = str(getattr(agent, "api_key", "") or "").strip()
    has_authorization = any(key.lower() == "authorization" for key in headers)
    if api_key and not has_authorization:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _connect_responses_websocket(uri: str, **kwargs):
    try:
        from websockets.sync.client import connect
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "codex_responses WebSocket transport requires the websockets package"
        ) from exc
    return connect(uri, **kwargs)


def _codex_websocket_user_agent_header(headers: Dict[str, str]) -> Optional[str]:
    if any(str(key).lower() == "user-agent" for key in headers):
        return None
    try:
        from hermes_cli import __version__ as hermes_version
    except Exception:
        hermes_version = "unknown"
    return f"HermesAgent/{hermes_version}"


def _websocket_state_name(websocket: Any) -> Optional[str]:
    state = getattr(websocket, "state", None)
    if state is None:
        return None
    name = getattr(state, "name", None)
    if isinstance(name, str):
        return name.upper()
    if isinstance(state, str):
        return state.upper()
    return None


def _websocket_is_open(websocket: Any) -> bool:
    state_name = _websocket_state_name(websocket)
    return state_name is None or state_name == "OPEN"


def _safe_snapshot(value: Any) -> Any:
    try:
        return deepcopy(value)
    except Exception:
        return value


def _plain_jsonish(value: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return {
            key: _plain_jsonish(val)
            for key, val in vars(value).items()
        }
    if isinstance(value, dict):
        return {
            key: _plain_jsonish(val)
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [_plain_jsonish(item) for item in value]
    if isinstance(value, tuple):
        return [_plain_jsonish(item) for item in value]
    return value


def _safe_plain_snapshot(value: Any) -> Any:
    return _safe_snapshot(_plain_jsonish(value))


def _normalize_codex_response_item_status(
    value: Any,
    *,
    default: str = "completed",
    allowed: Optional[set[str]] = None,
) -> str:
    if allowed is None:
        allowed = {"completed", "incomplete", "in_progress"}
    if isinstance(value, str):
        status = value.strip().lower().replace("-", "_").replace(" ", "_")
        if status in allowed:
            return status
    return default


def _is_prefix_items(prefix: Any, full: Any) -> bool:
    if not isinstance(prefix, list) or not isinstance(full, list):
        return False
    if len(prefix) > len(full):
        return False
    return full[: len(prefix)] == prefix


def _is_generated_response_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    role = str(item.get("role") or "").strip().lower()
    if role == "assistant":
        return True
    item_type = str(item.get("type") or "").strip()
    return item_type in _GENERATED_RESPONSE_ITEM_TYPES


def _client_incremental_input_items(items: Any) -> List[Any]:
    if not isinstance(items, list):
        return items
    return [item for item in items if not _is_generated_response_item(item)]


def _codex_websocket_baseline_input(
    previous_input: Any,
    response_output_items: Any = None,
) -> Any:
    previous_input = _plain_jsonish(previous_input)
    response_output_items = _plain_jsonish(response_output_items)
    if not isinstance(previous_input, list):
        return previous_input
    if not isinstance(response_output_items, list) or not response_output_items:
        return previous_input
    return previous_input + response_output_items


def _incremental_input_since_previous_response(
    previous_input: Any,
    current_input: Any,
    response_output_items: Any = None,
):
    baseline_input = _codex_websocket_baseline_input(previous_input, response_output_items)
    current_input = _plain_jsonish(current_input)
    if not _is_prefix_items(baseline_input, current_input):
        return current_input, False
    delta = current_input[len(baseline_input) :]
    return _client_incremental_input_items(delta), True


def _codex_websocket_request_fingerprint(payload: Dict[str, Any]) -> Any:
    return _safe_plain_snapshot({
        key: value
        for key, value in payload.items()
        if key not in {"input", "previous_response_id"}
    })


def _codex_response_output_items(final: SimpleNamespace) -> List[Any]:
    output = _plain_jsonish(getattr(final, "output", None))
    if not isinstance(output, list):
        return []
    reasoning_items: List[Any] = []
    message_items: List[Any] = []
    function_call_items: List[Any] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").strip()
        if item_type == "message":
            content = item.get("content")
            if not isinstance(content, list):
                continue
            content_parts = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = str(part.get("type") or "").strip()
                if part_type not in {"output_text", "text"}:
                    continue
                text = part.get("text", "")
                if text is None:
                    text = ""
                if not isinstance(text, str):
                    text = str(text)
                content_parts.append({"type": "output_text", "text": text})
            if not content_parts:
                continue
            message_item = {
                "type": "message",
                "role": "assistant",
                "status": _normalize_codex_response_item_status(item.get("status")),
                "content": content_parts,
            }
            item_id = item.get("id")
            if isinstance(item_id, str) and item_id.strip():
                message_item["id"] = item_id.strip()
            phase = item.get("phase")
            if isinstance(phase, str) and phase.strip():
                message_item["phase"] = phase.strip()
            message_items.append(message_item)
            continue
        if item_type == "reasoning":
            encrypted = item.get("encrypted_content")
            if not isinstance(encrypted, str) or not encrypted:
                continue
            reasoning_item = {"type": "reasoning", "encrypted_content": encrypted}
            item_id = item.get("id")
            if isinstance(item_id, str):
                item_id = item_id.strip()
                if item_id and not item_id.startswith("rs_tmp_"):
                    reasoning_item["id"] = item_id
            summary = item.get("summary")
            reasoning_item["summary"] = _plain_jsonish(summary) if isinstance(summary, list) else []
            reasoning_items.append(reasoning_item)
            continue
        if item_type == "function_call":
            status = _normalize_codex_response_item_status(
                item.get("status"),
                default="",
                allowed={"queued", "completed", "incomplete", "in_progress"},
            )
            if status in {"queued", "in_progress", "incomplete"}:
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id.strip():
                continue
            arguments = item.get("arguments", "{}")
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False)
            elif not isinstance(arguments, str):
                arguments = str(arguments)
            function_call_items.append({
                "type": "function_call",
                "call_id": call_id.strip(),
                "name": name.strip(),
                "arguments": arguments.strip() or "{}",
            })
            continue
    # Mirror _chat_messages_to_responses_input(): replay reasoning first,
    # then assistant content/sentinel, then tool calls. The WebSocket prefix
    # matcher must see the same sequence the next turn's adapter will emit.
    normalized: List[Any] = []
    normalized.extend(reasoning_items)
    normalized.extend(message_items)
    if reasoning_items and not message_items:
        normalized.append({"role": "assistant", "content": ""})
    normalized.extend(function_call_items)
    return normalized


class _CodexResponsesWebSocketSession:
    _MAX_CACHED_CHAINS = 16

    def __init__(
        self,
        *,
        uri: str,
        headers: Dict[str, str],
        open_timeout: float,
        turn_state_supplier=None,
        on_turn_state=None,
    ) -> None:
        self.uri = uri
        self.headers = dict(headers)
        self.open_timeout = open_timeout
        self._turn_state_supplier = turn_state_supplier or (lambda: "")
        self._on_turn_state = on_turn_state or (lambda value: None)
        self.websocket = None
        self._context = None
        self.previous_response_id: Optional[str] = None
        self.last_full_input: Any = None
        self.response_output_items: Any = None
        self._chains: List[Dict[str, Any]] = []
        self._pending_chain_index: Optional[int] = None
        self._pending_request_fingerprint: Any = None
        self.closed = False

    def matches(self, *, uri: str, headers: Dict[str, str], open_timeout: float) -> bool:
        return (
            not self.closed
            and self.uri == uri
            and self.headers == dict(headers)
            and self.open_timeout == open_timeout
        )

    def open(self):
        if self.websocket is not None and not self.closed:
            if _websocket_is_open(self.websocket):
                return self.websocket
            logger.info(
                "codex_responses_websocket_event=cached_connection_closed "
                "Codex Responses WebSocket cached connection is no longer open; reconnecting "
                "with full input (state=%s).",
                _websocket_state_name(self.websocket) or "unknown",
            )
            self.close()
        connect_headers = dict(self.headers)
        turn_state = str(self._turn_state_supplier() or "").strip()
        if turn_state:
            _set_header_if_missing(connect_headers, _CODEX_TURN_STATE_HEADER, turn_state)
            logger.info(
                "codex_responses_websocket_event=turn_state_replayed "
                "Codex Responses WebSocket replaying same-turn sticky routing token."
            )
        raw = _connect_responses_websocket(
            self.uri,
            additional_headers=connect_headers,
            user_agent_header=_codex_websocket_user_agent_header(connect_headers),
            open_timeout=self.open_timeout,
            close_timeout=10,
            max_size=None,
        )
        self._context = raw
        if hasattr(raw, "__enter__"):
            self.websocket = raw.__enter__()
        else:
            self.websocket = raw
        self.closed = False
        turn_state = _codex_websocket_response_header(
            self.websocket,
            raw,
            _CODEX_TURN_STATE_HEADER,
        )
        if turn_state:
            self._on_turn_state(turn_state)
        return self.websocket

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self._context is not None and hasattr(self._context, "__exit__"):
                self._context.__exit__(None, None, None)
            elif self.websocket is not None and hasattr(self.websocket, "close"):
                self.websocket.close()
        except Exception:
            logger.debug("Codex Responses WebSocket close failed", exc_info=True)
        finally:
            self.websocket = None
            self._context = None
            self._clear_chains()

    def _clear_active_chain(self) -> None:
        self.previous_response_id = None
        self.last_full_input = None
        self.response_output_items = None
        self._pending_chain_index = None
        self._pending_request_fingerprint = None

    def _clear_chains(self) -> None:
        self._chains = []
        self._clear_active_chain()

    def _set_active_chain(self, chain: Optional[Dict[str, Any]]) -> None:
        if chain is None:
            self.previous_response_id = None
            self.last_full_input = None
            self.response_output_items = None
            return
        self.previous_response_id = chain.get("previous_response_id")
        self.last_full_input = _safe_snapshot(chain.get("last_full_input"))
        self.response_output_items = _safe_snapshot(chain.get("response_output_items"))

    def _ensure_legacy_chain(self) -> None:
        if self._chains or not self.previous_response_id or self.last_full_input is None:
            return
        self._chains.append({
            "previous_response_id": self.previous_response_id,
            "last_full_input": _safe_plain_snapshot(self.last_full_input),
            "response_output_items": _safe_plain_snapshot(self.response_output_items or []),
            "request_fingerprint": None,
        })

    def _fingerprint_matches(self, chain: Dict[str, Any], request_fingerprint: Any) -> bool:
        chain_fingerprint = chain.get("request_fingerprint")
        return chain_fingerprint is None or chain_fingerprint == request_fingerprint

    def _find_continuation_chain(
        self,
        *,
        current_input: Any,
        request_fingerprint: Any,
    ) -> tuple[Optional[int], Optional[Dict[str, Any]], Any]:
        self._ensure_legacy_chain()
        for index in range(len(self._chains) - 1, -1, -1):
            chain = self._chains[index]
            if not self._fingerprint_matches(chain, request_fingerprint):
                continue
            incremental_input, can_continue = _incremental_input_since_previous_response(
                chain.get("last_full_input"),
                current_input,
                chain.get("response_output_items"),
            )
            if can_continue:
                return index, chain, incremental_input
        return None, None, current_input

    def _drop_pending_chain(self) -> None:
        if self._pending_chain_index is None:
            return
        if 0 <= self._pending_chain_index < len(self._chains):
            del self._chains[self._pending_chain_index]
        self._clear_active_chain()

    def build_payload(
        self,
        api_kwargs: Dict[str, Any],
        *,
        force_full_input: bool = False,
    ) -> tuple[Dict[str, Any], Any, bool]:
        payload = _codex_websocket_response_body(api_kwargs)
        payload["type"] = "response.create"
        full_input = _safe_plain_snapshot(payload.get("input"))
        request_fingerprint = _codex_websocket_request_fingerprint(payload)
        used_continuation = False
        self._pending_chain_index = None
        self._pending_request_fingerprint = request_fingerprint

        if force_full_input:
            self._clear_chains()
            self._pending_request_fingerprint = request_fingerprint
            return payload, full_input, used_continuation

        chain_index, chain, incremental_input = self._find_continuation_chain(
            current_input=full_input,
            request_fingerprint=request_fingerprint,
        )
        if chain is not None:
            self._pending_chain_index = chain_index
            self._set_active_chain(chain)
            payload["previous_response_id"] = chain["previous_response_id"]
            payload["input"] = incremental_input
            used_continuation = True
        else:
            # The current prompt no longer extends any cached chain exactly.
            # Start a fresh chain on the same socket with full input rather
            # than sending an unsafe incremental payload.
            self._set_active_chain(None)
            if self._chains:
                logger.debug(
                    "codex_responses_websocket_event=incremental_chain_miss "
                    "sending full input because no cached chain matched request "
                    "(chains=%d).",
                    len(self._chains),
                )

        return payload, full_input, used_continuation

    def record_response(self, final: SimpleNamespace, full_input: Any) -> None:
        status = str(getattr(final, "status", "") or "").strip().lower()
        response_id = getattr(final, "id", None)
        if status in {"failed", "cancelled"} or not isinstance(response_id, str) or not response_id:
            self._drop_pending_chain()
            return
        chain = {
            "previous_response_id": response_id,
            "last_full_input": _safe_plain_snapshot(full_input),
            "response_output_items": _safe_plain_snapshot(_codex_response_output_items(final)),
            "request_fingerprint": _safe_snapshot(self._pending_request_fingerprint),
        }
        if (
            self._pending_chain_index is not None
            and 0 <= self._pending_chain_index < len(self._chains)
        ):
            del self._chains[self._pending_chain_index]
        self._chains.append(chain)
        if len(self._chains) > self._MAX_CACHED_CHAINS:
            self._chains = self._chains[-self._MAX_CACHED_CHAINS :]
        self._set_active_chain(chain)
        self._pending_chain_index = None
        self._pending_request_fingerprint = None


def close_codex_responses_websocket_session(agent) -> None:
    session = getattr(agent, "_codex_responses_websocket_session", None)
    if session is not None:
        try:
            session.close()
        finally:
            agent._codex_responses_websocket_session = None


def reset_codex_responses_websocket_turn_fallback(agent) -> None:
    """Reset per-turn stream state while preserving session-scoped HTTP fallback."""
    agent._codex_responses_websocket_output_committed = False
    agent._codex_responses_websocket_turn_state = None


def mark_codex_responses_websocket_error(exc: BaseException) -> None:
    """Tag an exception as coming from the codex_responses WebSocket transport."""
    try:
        setattr(exc, _WEBSOCKET_ERROR_MARKER, True)
    except Exception:
        pass


def is_codex_responses_websocket_error(exc: BaseException) -> bool:
    return bool(getattr(exc, _WEBSOCKET_ERROR_MARKER, False))


def prepare_codex_responses_websocket_retry(
    agent,
    *,
    reason: str,
    error: BaseException | None = None,
) -> None:
    """Close the failed socket and force the next WebSocket attempt to send full input."""
    if error is not None:
        mark_codex_responses_websocket_error(error)
    key = _codex_websocket_transport_key(agent)
    close_codex_responses_websocket_session(agent)
    agent._codex_responses_websocket_chain_invalidated_key = key
    agent._codex_responses_websocket_output_committed = False
    logger.warning(
        "codex_responses_websocket_event=retry_transport_error "
        "Codex Responses WebSocket transport failed before output; retrying with a fresh "
        "WebSocket connection and full input (reason=%s code=%s param=%s). %s error=%s",
        reason,
        getattr(error, "code", None),
        getattr(error, "param", None),
        getattr(agent, "_client_log_context", lambda: "")(),
        error,
    )


def disable_codex_responses_websocket_for_session(
    agent,
    *,
    reason: str,
    error: BaseException | None = None,
) -> None:
    """Disable WebSocket for this provider for the current agent session.

    The continuation chain is also invalidated so the next WebSocket attempt
    for the same provider starts with full input instead of reusing stale
    ``previous_response_id`` state from the failed socket.
    """
    key = _codex_websocket_transport_key(agent)
    close_codex_responses_websocket_session(agent)
    agent._codex_responses_websocket_http_fallback_key = key
    agent._codex_responses_websocket_http_fallback_reason = reason
    agent._codex_responses_websocket_chain_invalidated_key = key
    agent._codex_responses_websocket_output_committed = False
    logger.warning(
        "codex_responses_websocket_event=http_fallback "
        "Codex Responses WebSocket disabled for this session; falling back to HTTP "
        "(reason=%s). %s error=%s",
        reason,
        getattr(agent, "_client_log_context", lambda: "")(),
        error,
    )


def disable_codex_responses_websocket_for_turn(
    agent,
    *,
    reason: str,
    error: BaseException | None = None,
) -> None:
    """Backward-compatible wrapper for the now session-scoped HTTP fallback."""
    disable_codex_responses_websocket_for_session(agent, reason=reason, error=error)


def should_immediately_fallback_codex_responses_websocket_to_http(
    exc: BaseException,
) -> bool:
    """Return True for WebSocket handshake errors Codex routes directly to HTTP."""
    return getattr(exc, "status_code", None) == 426


def switch_codex_responses_websocket_to_http_after_retries(
    agent,
    exc: BaseException,
    *,
    reason: str,
) -> bool:
    """Activate Codex-style session HTTP fallback after WebSocket retry exhaustion."""
    if not is_codex_responses_websocket_error(exc):
        return False
    if not codex_responses_websocket_enabled(agent):
        return False
    if not should_fallback_codex_responses_websocket_to_http(agent, exc):
        return False
    disable_codex_responses_websocket_for_session(agent, reason=reason, error=exc)
    return True


def clear_codex_responses_websocket_chain_invalidated(agent) -> None:
    key = getattr(agent, "_codex_responses_websocket_chain_invalidated_key", None)
    if key is not None and key == _codex_websocket_transport_key(agent):
        agent._codex_responses_websocket_chain_invalidated_key = None


def _codex_responses_websocket_chain_invalidated(agent) -> bool:
    key = getattr(agent, "_codex_responses_websocket_chain_invalidated_key", None)
    return key is not None and key == _codex_websocket_transport_key(agent)


def codex_responses_websocket_output_committed(agent) -> bool:
    """Return True only after visible text was delivered outside this request.

    WebSocket streams can receive internal output events (function calls,
    reasoning items, or buffered text) before ``response.completed``. Those
    events are safe to retry/fallback when nothing has been sent to the user.
    The unsafe boundary is external delivery via stream callbacks, because an
    HTTP retry would duplicate already-visible text.
    """
    return bool(getattr(agent, "_codex_responses_websocket_output_committed", False))


def _mark_codex_responses_websocket_output_committed(agent) -> None:
    agent._codex_responses_websocket_output_committed = True


def _get_codex_responses_websocket_session(
    agent,
    *,
    uri: str,
    headers: Dict[str, str],
    open_timeout: float,
) -> _CodexResponsesWebSocketSession:
    session = getattr(agent, "_codex_responses_websocket_session", None)
    if isinstance(session, _CodexResponsesWebSocketSession) and session.matches(
        uri=uri,
        headers=headers,
        open_timeout=open_timeout,
    ):
        return session

    close_codex_responses_websocket_session(agent)
    session = _CodexResponsesWebSocketSession(
        uri=uri,
        headers=headers,
        open_timeout=open_timeout,
        turn_state_supplier=lambda: _codex_responses_turn_state(agent),
        on_turn_state=lambda value: _set_codex_responses_turn_state(agent, value),
    )
    agent._codex_responses_websocket_session = session
    return session


def _iter_codex_websocket_events(
    websocket,
    *,
    interrupt_check=None,
    recv_timeout: Optional[float] = 1.0,
):
    while True:
        if interrupt_check is not None and interrupt_check():
            break
        try:
            raw = websocket.recv(timeout=recv_timeout)
        except TimeoutError:
            continue
        except Exception as exc:
            if exc.__class__.__name__.startswith("ConnectionClosed"):
                raise ConnectionError(
                    f"websocket closed before response.completed: {exc}"
                ) from exc
            raise

        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        event = json.loads(raw)
        event_type = event.get("type") if isinstance(event, dict) else None
        yield _json_to_namespace(event)
        if event_type in _TERMINAL_EVENT_TYPES or event_type == "error":
            break


def run_codex_websocket(
    agent,
    api_kwargs: Dict[str, Any],
    *,
    on_text_delta=None,
    on_reasoning_delta=None,
    on_first_delta=None,
    on_event=None,
    interrupt_check=None,
) -> SimpleNamespace:
    """Execute one Responses API request over WebSocket mode."""
    websocket_url = codex_responses_websocket_url(getattr(agent, "base_url", ""))
    issuer_kind = _codex_response_issuer_kind_for_agent(agent)
    timeout = api_kwargs.get("timeout")
    open_timeout = (
        float(timeout)
        if isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and timeout > 0
        else 10
    )
    headers = _codex_websocket_headers(agent, api_kwargs)
    force_full_input = _codex_responses_websocket_chain_invalidated(agent)
    if force_full_input:
        logger.info(
            "codex_responses_websocket_event=restore_full_input "
            "Codex Responses WebSocket chain invalidated; reopening with full input. %s",
            getattr(agent, "_client_log_context", lambda: "")(),
        )
        close_codex_responses_websocket_session(agent)
    session = _get_codex_responses_websocket_session(
        agent,
        uri=websocket_url,
        headers=headers,
        open_timeout=open_timeout,
    )

    send_state = {"used_continuation": False}

    def _send_once(
        active_session: _CodexResponsesWebSocketSession,
        *,
        force_full: bool = False,
    ) -> tuple[SimpleNamespace, bool]:
        agent._codex_responses_websocket_output_committed = False
        send_state["used_continuation"] = False
        websocket = active_session.open()
        payload, full_input, used_continuation = active_session.build_payload(
            api_kwargs,
            force_full_input=force_full,
        )
        send_state["used_continuation"] = used_continuation
        websocket.send(json.dumps(payload))

        final_response = _consume_codex_event_stream(
            _iter_codex_websocket_events(
                websocket,
                interrupt_check=interrupt_check,
                recv_timeout=1.0,
            ),
            model=api_kwargs.get("model"),
            on_text_delta=on_text_delta,
            on_reasoning_delta=on_reasoning_delta,
            on_first_delta=on_first_delta,
            on_event=on_event,
            interrupt_check=interrupt_check,
        )
        final_response.issuer_kind = issuer_kind
        active_session.record_response(final_response, full_input)
        if force_full and active_session.previous_response_id:
            clear_codex_responses_websocket_chain_invalidated(agent)
            logger.info(
                "codex_responses_websocket_event=restore_full_input_completed "
                "Codex Responses WebSocket recovered; full-input request completed "
                "(response_id=%s). %s",
                active_session.previous_response_id,
                getattr(agent, "_client_log_context", lambda: "")(),
            )
        return final_response, used_continuation

    try:
        final, used_continuation = _send_once(session, force_full=force_full_input)
    except Exception:
        exc = _current_exception()
        close_codex_responses_websocket_session(agent)
        if _is_previous_response_missing_error(exc):
            logger.warning(
                "codex_responses_websocket_event=previous_response_rejected "
                "Codex Responses WebSocket previous_response_id rejected; reopening with full input. "
                "%s error=%s",
                getattr(agent, "_client_log_context", lambda: "")(),
                exc,
            )
            recovery_session = _get_codex_responses_websocket_session(
                agent,
                uri=websocket_url,
                headers=headers,
                open_timeout=open_timeout,
            )
            final, _used_continuation = _send_once(recovery_session, force_full=True)
            return final
        raise

    return final


def _current_exception() -> BaseException:
    import sys
    exc = sys.exc_info()[1]
    if isinstance(exc, BaseException):
        return exc
    return RuntimeError("unknown exception")


def _is_previous_response_missing_error(exc: BaseException) -> bool:
    code = str(getattr(exc, "code", "") or "").strip().lower()
    param = str(getattr(exc, "param", "") or "").strip().lower()
    message = str(getattr(exc, "message", "") or exc).strip().lower()
    return (
        code == "previous_response_not_found"
        or param == "previous_response_id"
        or "previous_response_not_found" in message
        or "previous response" in message and "not found" in message
    )


def _event_field(event: Any, name: str, default: Any = None) -> Any:
    """Field access that handles both attr-style (SDK objects) and dict (raw JSON) events."""
    value = getattr(event, name, None)
    if value is None and isinstance(event, dict):
        value = event.get(name, default)
    return value if value is not None else default


def should_fallback_codex_responses_websocket_to_http(agent, exc: BaseException) -> bool:
    """Return True for pre-output WebSocket transport failures only."""
    if codex_responses_websocket_output_committed(agent):
        return False
    if isinstance(exc, InterruptedError):
        return False
    if isinstance(exc, ValueError):
        return False
    class_name = exc.__class__.__name__.lower()
    message = str(getattr(exc, "message", None) or exc).lower()
    if "websocket" in class_name or class_name.startswith("connectionclosed"):
        return True
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    if _is_previous_response_missing_error(exc):
        return False
    code = str(getattr(exc, "code", "") or "").strip().lower()
    if code == _WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE:
        return True
    if should_immediately_fallback_codex_responses_websocket_to_http(exc):
        return True
    if getattr(exc, "status_code", None) is not None:
        return False
    if getattr(exc, "code", None) or getattr(exc, "param", None):
        return False

    if "did not emit a terminal response" in message:
        return True
    return any(
        phrase in message
        for phrase in (
            "connection closed",
            "connection reset",
            "connection refused",
            "network connection",
            "remote protocol",
            "timed out",
            "timeout",
            "websocket",
        )
    )


def _raise_stream_error(event: Any) -> None:
    """Raise a ``_StreamErrorEvent`` from a ``type=error`` SSE frame.

    Imported lazily so this module stays importable from places that don't
    pull in ``run_agent`` (e.g. plugin code, doc tools).
    """
    from run_agent import _StreamErrorEvent
    nested_error = _event_field(event, "error")
    if not isinstance(nested_error, dict):
        nested_error = {
            "message": getattr(nested_error, "message", None),
            "code": getattr(nested_error, "code", None),
            "param": getattr(nested_error, "param", None),
        } if nested_error is not None else {}
    message = (
        _event_field(event, "message", "")
        or nested_error.get("message")
        or "stream emitted error event"
    ).strip()
    raise _StreamErrorEvent(
        message,
        code=_event_field(event, "code") or nested_error.get("code"),
        param=_event_field(event, "param") or nested_error.get("param"),
        status_code=_event_field(event, "status"),
    )


def _consume_codex_event_stream(
    event_iter: Any,
    *,
    model: str,
    on_text_delta=None,
    on_reasoning_delta=None,
    on_first_delta=None,
    on_event=None,
    interrupt_check=None,
) -> SimpleNamespace:
    """Consume a Codex Responses SSE event stream and return a final response.

    The returned object is a ``SimpleNamespace`` shaped like the SDK's typed
    ``Response`` for the fields downstream code actually reads:

    * ``output``: list of output items, assembled from ``response.output_item.done``.
      For tool-call turns this contains the function_call items; for plain-text
      turns it contains a synthesized ``message`` item built from streamed deltas
      if no message item was emitted directly.
    * ``output_text``: assembled text from ``response.output_text.delta`` deltas.
    * ``usage``: copied from the terminal event's ``response.usage`` (when present).
    * ``status``: ``completed`` / ``incomplete`` / ``failed`` (or ``completed`` if
      the stream ended without a terminal frame but produced content).
    * ``id``: ``response.id`` when present.
    * ``incomplete_details``: passed through for ``response.incomplete`` frames.
    * ``error``: passed through for ``response.failed`` frames.
    * ``model``: from kwargs (the wire model name is not authoritative).

    Critically, we never read ``response.output`` from the terminal event for
    content reconstruction — only ``usage``, ``status``, ``id``.  That field
    being ``null`` / ``[]`` / missing is fine.

    Callbacks:

    * ``on_text_delta(str)`` — fires per ``response.output_text.delta``, suppressed
      once a function_call event is seen (so tool-call turns don't bleed text
      into the chat).
    * ``on_reasoning_delta(str)`` — fires per ``response.reasoning.*.delta``.
    * ``on_first_delta()`` — one-shot, fires on the first text delta only.
    * ``on_event(event)`` — fires for every event before any other processing.
      Used for watchdog activity, debug logging, anything wire-shape-agnostic.
    * ``interrupt_check()`` — returns True to break the loop early.
    """
    collected_output_items: List[Any] = []
    collected_text_deltas: List[str] = []
    has_tool_calls = False
    first_delta_fired = False
    terminal_status: str = "completed"
    terminal_usage: Any = None
    terminal_response_id: str = None
    terminal_incomplete_details: Any = None
    terminal_error: Any = None
    saw_terminal = False

    for event in event_iter:
        if on_event is not None:
            try:
                on_event(event)
            except (TimeoutError, InterruptedError):
                # Control-flow signals from watchdog/cancellation hooks must
                # propagate, not get swallowed as "debug noise".
                raise
            except Exception:
                # Genuine bugs in third-party debug/log hooks shouldn't break
                # stream consumption.
                logger.debug("Codex stream on_event hook raised", exc_info=True)
        if interrupt_check is not None and interrupt_check():
            break

        event_type = _event_field(event, "type", "")
        if not isinstance(event_type, str):
            event_type = ""

        # ``error`` SSE frames carry the provider's real failure reason
        # (subscription / quota / model-not-available / rejected-reasoning-replay)
        # but never appear in the terminal set.  Surface them as a structured
        # exception so the credential pool + error classifier see the body.
        if event_type == "error":
            _raise_stream_error(event)

        if "output_text.delta" in event_type or event_type == "response.output_text.delta":
            delta_text = _event_field(event, "delta", "")
            if delta_text:
                collected_text_deltas.append(delta_text)
                if not has_tool_calls:
                    if not first_delta_fired:
                        first_delta_fired = True
                        if on_first_delta is not None:
                            try:
                                on_first_delta()
                            except Exception:
                                logger.debug("Codex stream on_first_delta raised", exc_info=True)
                    if on_text_delta is not None:
                        try:
                            on_text_delta(delta_text)
                        except Exception:
                            logger.debug("Codex stream on_text_delta raised", exc_info=True)
            continue

        if "function_call" in event_type:
            has_tool_calls = True
            # fall through — function_call items still get added on output_item.done

        if "reasoning" in event_type and "delta" in event_type:
            reasoning_text = _event_field(event, "delta", "")
            if reasoning_text and on_reasoning_delta is not None:
                try:
                    on_reasoning_delta(reasoning_text)
                except Exception:
                    logger.debug("Codex stream on_reasoning_delta raised", exc_info=True)
            continue

        if event_type == "response.output_item.done":
            done_item = _event_field(event, "item")
            if done_item is not None:
                collected_output_items.append(done_item)
            continue

        if event_type in _TERMINAL_EVENT_TYPES:
            saw_terminal = True
            resp_obj = _event_field(event, "response")
            if resp_obj is not None:
                terminal_usage = getattr(resp_obj, "usage", None)
                if terminal_usage is None and isinstance(resp_obj, dict):
                    terminal_usage = resp_obj.get("usage")
                rid = getattr(resp_obj, "id", None)
                if rid is None and isinstance(resp_obj, dict):
                    rid = resp_obj.get("id")
                terminal_response_id = rid
                rstatus = getattr(resp_obj, "status", None)
                if rstatus is None and isinstance(resp_obj, dict):
                    rstatus = resp_obj.get("status")
                if isinstance(rstatus, str):
                    terminal_status = rstatus
                if event_type == "response.incomplete":
                    terminal_incomplete_details = getattr(resp_obj, "incomplete_details", None)
                    if terminal_incomplete_details is None and isinstance(resp_obj, dict):
                        terminal_incomplete_details = resp_obj.get("incomplete_details")
                if event_type == "response.failed":
                    terminal_error = getattr(resp_obj, "error", None)
                    if terminal_error is None and isinstance(resp_obj, dict):
                        terminal_error = resp_obj.get("error")
            if event_type == "response.completed":
                terminal_status = terminal_status or "completed"
            elif event_type == "response.incomplete":
                terminal_status = terminal_status or "incomplete"
            elif event_type == "response.failed":
                terminal_status = terminal_status or "failed"
            # Stop on terminal event.
            break

    # Build the final output list.  Prefer items observed via output_item.done;
    # if none arrived but we streamed plain text deltas (no tool calls), synthesize
    # a single message item so downstream normalization has something to work with.
    if collected_output_items:
        output = list(collected_output_items)
    elif collected_text_deltas and not has_tool_calls:
        assembled = "".join(collected_text_deltas)
        output = [SimpleNamespace(
            type="message",
            role="assistant",
            status="completed",
            content=[SimpleNamespace(type="output_text", text=assembled)],
        )]
    else:
        output = []

    # If the stream ended without any terminal event AND produced no usable
    # content (no items, no text deltas), surface that as a RuntimeError so
    # callers can distinguish "stream truncated mid-flight / provider rejected
    # the call" from "stream completed with empty body".  This preserves the
    # signal the SDK's high-level helper used to raise as
    # ``RuntimeError("Didn't receive a `response.completed` event.")``.
    if not saw_terminal and not output:
        raise RuntimeError(
            "Codex Responses stream did not emit a terminal response"
        )

    assembled_text = "".join(collected_text_deltas)

    final = SimpleNamespace(
        output=output,
        output_text=assembled_text,
        usage=terminal_usage,
        status=terminal_status,
        id=terminal_response_id,
        model=model,
        incomplete_details=terminal_incomplete_details,
        error=terminal_error,
    )
    return final


def run_codex_stream(agent, api_kwargs: dict, client: Any = None, on_first_delta=None):
    """Execute one streaming Responses API request and return the final response.

    Uses ``responses.create(stream=True)`` (low-level raw event iteration)
    rather than the high-level ``responses.stream(...)`` helper.  This makes
    us structurally immune to backend drift in the ``response.completed``
    payload shape — we never let the SDK reconstruct a typed object from
    the terminal event's ``output`` field.
    """
    import httpx as _httpx

    max_stream_retries = 1
    # Accumulate streamed text so callers / compat shims can read it.
    agent._codex_streamed_text_parts: list = []

    def _on_text_delta(text: str) -> None:
        agent._codex_streamed_text_parts.append(text)
        before = getattr(agent, "_current_streamed_assistant_text", "") or ""
        agent._fire_stream_delta(text)
        after = getattr(agent, "_current_streamed_assistant_text", "") or ""
        if after != before:
            _mark_codex_responses_websocket_output_committed(agent)

    def _on_reasoning_delta(text: str) -> None:
        agent._fire_reasoning_delta(text)

    def _on_event(event: Any) -> None:
        # TTFB watchdog and activity touch — runs once per SSE event.
        agent._codex_stream_last_event_ts = time.time()
        agent._touch_activity("receiving stream response")

    def _interrupt_check() -> bool:
        return bool(agent._interrupt_requested)

    def _warn_terminal_status(final: SimpleNamespace) -> None:
        if final.status in {"incomplete", "failed"}:
            logger.warning(
                "Codex Responses stream terminal status=%s "
                "(incomplete_details=%s, error=%s, streamed_chars=%d). %s",
                final.status, final.incomplete_details, final.error,
                sum(len(p) for p in agent._codex_streamed_text_parts),
                agent._client_log_context(),
            )

    if codex_responses_websocket_enabled(agent):
        final = run_codex_websocket(
            agent,
            api_kwargs,
            on_text_delta=_on_text_delta,
            on_reasoning_delta=_on_reasoning_delta,
            on_first_delta=on_first_delta,
            on_event=_on_event,
            interrupt_check=_interrupt_check,
        )
        _warn_terminal_status(final)
        return final

    active_client = client or agent._ensure_primary_openai_client(reason="codex_stream_direct")

    for attempt in range(max_stream_retries + 1):
        if agent._interrupt_requested:
            raise InterruptedError("Agent interrupted before Codex stream retry")

        stream_kwargs = dict(api_kwargs)
        stream_kwargs["stream"] = True

        try:
            event_stream = active_client.responses.create(**stream_kwargs)
        except (_httpx.RemoteProtocolError, _httpx.ReadTimeout, _httpx.ConnectError, ConnectionError) as exc:
            if attempt < max_stream_retries:
                logger.debug(
                    "Codex Responses stream connect failed (attempt %s/%s); retrying. %s error=%s",
                    attempt + 1, max_stream_retries + 1,
                    agent._client_log_context(), exc,
                )
                continue
            raise

        try:
            # Compatibility: some mocks/providers return a concrete response
            # instead of an iterable.  Pass it straight through.
            if hasattr(event_stream, "output") and not hasattr(event_stream, "__iter__"):
                return event_stream

            try:
                final = _consume_codex_event_stream(
                    event_stream,
                    model=api_kwargs.get("model"),
                    on_text_delta=_on_text_delta,
                    on_reasoning_delta=_on_reasoning_delta,
                    on_first_delta=on_first_delta,
                    on_event=_on_event,
                    interrupt_check=_interrupt_check,
                )
            except (_httpx.RemoteProtocolError, _httpx.ReadTimeout, _httpx.ConnectError, ConnectionError) as exc:
                if attempt < max_stream_retries:
                    logger.debug(
                        "Codex Responses stream transport failed mid-iteration "
                        "(attempt %s/%s); retrying. %s error=%s",
                        attempt + 1, max_stream_retries + 1,
                        agent._client_log_context(), exc,
                    )
                    continue
                raise

            _warn_terminal_status(final)

            return final
        finally:
            close_fn = getattr(event_stream, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass


def run_codex_create_stream_fallback(agent, api_kwargs: dict, client: Any = None):
    """Backward-compatible alias for the unified event-driven path.

    Historically this was the fallback when the SDK's high-level
    ``responses.stream(...)`` helper raised on shape drift.  The primary
    path now does exactly what the fallback did, so this just forwards.
    Kept as a public symbol because tests and a small number of call sites
    still reference it by name.
    """
    return run_codex_stream(agent, api_kwargs, client=client)


__all__ = [
    "run_codex_app_server_turn",
    "run_codex_stream",
    "run_codex_websocket",
    "run_codex_create_stream_fallback",
    "close_codex_responses_websocket_session",
    "codex_responses_websocket_enabled",
    "codex_responses_websocket_url",
    "disable_codex_responses_websocket_for_session",
    "disable_codex_responses_websocket_for_turn",
    "is_codex_responses_websocket_error",
    "mark_codex_responses_websocket_error",
    "prepare_codex_responses_websocket_retry",
    "reset_codex_responses_websocket_turn_fallback",
    "should_immediately_fallback_codex_responses_websocket_to_http",
    "should_fallback_codex_responses_websocket_to_http",
    "switch_codex_responses_websocket_to_http_after_retries",
    "_consume_codex_event_stream",
]
