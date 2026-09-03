"""langfuse — Hermes plugin for Langfuse observability.

Traces Hermes conversations, LLM calls, and tool usage to Langfuse.

Activation is handled by the Hermes plugin system — standalone plugins only
load when listed in ``plugins.enabled`` (via ``hermes plugins enable
observability/langfuse`` or ``hermes tools → Langfuse Observability``). At
runtime the plugin also requires the ``langfuse`` SDK and credentials; if
either is missing the hooks are inert.

Required env vars (set via ``hermes tools`` or ~/.hermes/.env):
  HERMES_LANGFUSE_PUBLIC_KEY  - Langfuse project public key (pk-lf-...)
  HERMES_LANGFUSE_SECRET_KEY  - Langfuse project secret key (sk-lf-...)
  HERMES_LANGFUSE_BASE_URL    - Langfuse server URL (default: https://cloud.langfuse.com)

Optional env vars:
  HERMES_LANGFUSE_ENV         - environment tag (e.g. "production", "local")
  HERMES_LANGFUSE_RELEASE     - release/version tag
  HERMES_LANGFUSE_SAMPLE_RATE - sampling rate 0.0–1.0 (default: 1.0)
  HERMES_LANGFUSE_MAX_CHARS   - max chars per field (default: 12000)
  HERMES_LANGFUSE_CAPTURE     - content capture mode (default: "sanitized")
      metadata  - no content: sizes, roles, tool names, IDs, usage, cost only
      sanitized - content with secret-pattern redaction + truncation
      full      - raw content (truncated only); explicit opt-in
  HERMES_LANGFUSE_DEBUG       - set to "true" for verbose logging

Exported (not read) while a turn is traced:
  HERMES_LANGFUSE_TRACE_ID    - the current turn's trace id, published into the
      environment of subprocesses spawned during the turn so a child run (the
      terminal tool's claude shim, execute_code) can stamp its own telemetry
      and join back to the exact spawning turn.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from langfuse import Langfuse, propagate_attributes
except Exception:  # pragma: no cover - fail-open when optional dep is missing
    Langfuse = None
    propagate_attributes = None


@dataclass
class TraceState:
    trace_id: str
    root_ctx: Any
    root_span: Any
    generations: Dict[str, Any] = field(default_factory=dict)
    tools: Dict[str, Any] = field(default_factory=dict)
    pending_tools_by_name: Dict[str, list] = field(default_factory=dict)
    turn_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # Keyed by child_session_id: subagent_stop carries no child_subagent_id.
    subagents: Dict[str, Any] = field(default_factory=dict)
    # Fingerprints of MoA fan-outs already recorded. The client holds its last
    # fan-out until the next one, so a tool-loop turn would re-emit the same
    # advisors on every API call without this.
    moa_emitted: set = field(default_factory=set)
    last_updated_at: float = field(default_factory=time.time)


_STATE_LOCK = threading.Lock()
_TRACE_STATE: Dict[str, TraceState] = {}
# Hard cap on live trace state. Each turn keys _TRACE_STATE by a unique
# turn_id, and an entry is normally reclaimed by _finish_trace when a turn
# ends cleanly (final response has content and no tool calls). A turn that
# never reaches that state — interrupted, a tool-only final step, or empty
# final content — would otherwise linger forever, so over the cap we evict
# the least-recently-updated entries (ending their root span first). The cap
# is far above any realistic concurrent-live-turn working set; it exists only
# to bound the leak from non-finalizing turns, not to limit concurrency.
_MAX_TRACE_STATE = 256
_LANGFUSE_CLIENT = None
# Guards _LANGFUSE_CLIENT initialization against the TOCTOU race: two
# concurrent first callers both pass the ``is not None`` guard, both
# construct a Langfuse(**kwargs) client, and the loser's client leaks an
# open HTTPS connection + background flush thread.  _STATE_LOCK is not
# reused here because it guards _TRACE_STATE (hot path) and nesting the
# two locks would risk deadlock with future callers.
_LANGFUSE_CLIENT_LOCK = threading.Lock()
_READ_FILE_LINE_RE = re.compile(r"^\s*(\d+)\|(.*)$")
_READ_FILE_HEAD_LINES = 25
_READ_FILE_TAIL_LINES = 15

# Langfuse-issued keys always carry these prefixes (cloud or self-hosted —
# the prefix is baked into the server-side issuance flow, not a UI hint).
# Anything else (`placeholder`, `test-key`, `your-langfuse-key`, etc.) is a
# leftover template value and would cause the SDK to silently accept the
# credentials at construction time but drop every trace at flush time.
# See #23823 — the silent-failure bug this guard fixes.
_LANGFUSE_KEY_PREFIXES: Dict[str, str] = {
    "HERMES_LANGFUSE_PUBLIC_KEY": "pk-lf-",
    "HERMES_LANGFUSE_SECRET_KEY": "sk-lf-",
}


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_bool(*names: str) -> bool:
    for name in names:
        value = _env(name).lower()
        if value:
            return value in {"1", "true", "yes", "on"}
    return False


def _debug_enabled() -> bool:
    return _env_bool("HERMES_LANGFUSE_DEBUG")


def _debug(message: str) -> None:
    if _debug_enabled():
        logger.info("Langfuse tracing: %s", message)


# ---------------------------------------------------------------------------
# Capture modes
# ---------------------------------------------------------------------------

_CAPTURE_MODES = ("metadata", "sanitized", "full")
_DEFAULT_CAPTURE_MODE = "sanitized"
_warned_invalid_capture = False


def _capture_mode() -> str:
    """Resolve the content-capture mode: ``metadata | sanitized | full``.

    Read per call (cheap env lookup) so tests and long-lived processes can
    flip modes without a client reset. Invalid values warn once per process
    and fall back to the default rather than silently capturing more than
    the operator intended.
    """
    global _warned_invalid_capture
    value = _env("HERMES_LANGFUSE_CAPTURE").lower()
    if not value:
        return _DEFAULT_CAPTURE_MODE
    if value in _CAPTURE_MODES:
        return value
    if not _warned_invalid_capture:
        _warned_invalid_capture = True
        logger.warning(
            "Langfuse plugin: invalid HERMES_LANGFUSE_CAPTURE=%r, falling back "
            "to %r (valid: %s)",
            value, _DEFAULT_CAPTURE_MODE, ", ".join(_CAPTURE_MODES),
        )
    return _DEFAULT_CAPTURE_MODE


# Secret redaction in ``sanitized`` mode reuses the project-wide
# ``agent.redact.redact_sensitive_text(force=True)`` — which covers 50+ credential
# patterns, private keys, JWTs, auth headers, DB connection strings, and env
# assignments with pre-check-gated regex. The ``force=True`` flag ensures
# redaction runs even if the user has ``security.redact_secrets: false`` set —
# appropriate for an observability plugin exporting to an external service.


def _redact_secrets(value: str) -> str:
    try:
        from agent.redact import redact_sensitive_text
        return redact_sensitive_text(value, force=True)
    except Exception:
        return value


def _describe_content(value: Any, *, depth: int = 0) -> Any:
    """Metadata-mode stand-in for content: shape and size, never payload."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return {"omitted": True, "type": "number"}
    if isinstance(value, bytes):
        return {"omitted": True, "type": "bytes", "length": len(value)}
    if isinstance(value, str):
        return {"omitted": True, "type": "text", "chars": len(value)}
    if isinstance(value, dict):
        return {
            "omitted": True,
            "type": "object",
            "keys": [str(k) for k in list(value.keys())[:20]],
        }
    if isinstance(value, (list, tuple, set)):
        return {"omitted": True, "type": "array", "items": len(value)}
    return {"omitted": True, "type": type(value).__name__}


def _capture_content(value: Any, *, parse_json_strings: bool = False,
                     tool_name: str = "", args: Any = None) -> Any:
    """Apply the active capture mode to a CONTENT value.

    Metadata fields (provider, model, IDs, counts) should NOT go through
    this — they stay as-is in every mode. Only prompt/response text, tool
    arguments, and tool results are content.
    """
    mode = _capture_mode()
    if mode == "metadata":
        return _describe_content(value)
    if tool_name or args is not None:
        value = _normalize_payload(value, tool_name=tool_name, args=args)
    return _safe_value(value, parse_json_strings=parse_json_strings)


# Sentinel: "_get_langfuse() has tried and failed". Lets us short-circuit
# every subsequent hook call without re-checking env vars or re-attempting
# SDK init. Tests clear this by reloading the module via
# ``sys.modules.pop(...) + importlib.import_module(...)`` rather than via a
# dedicated reset function. Runtime callers cannot reset the cache; if an
# operator fixes a misconfigured credential they must restart the process.
_INIT_FAILED = object()


def _redact_key_preview(value: str) -> str:
    """Return a brief, log-safe preview of a credential value.

    Keeps enough characters to disambiguate common placeholders
    (``placeholder``, ``test-key``, ``your-key``) without echoing a
    real secret in full if an operator pasted one into the wrong env
    var.  Used only for the once-per-process placeholder-detection
    warning in :func:`_get_langfuse`.
    """
    if not value:
        return "<empty>"
    if len(value) <= 12:
        return repr(value)
    return repr(value[:6] + "...")


def _validate_langfuse_key(env_name: str, value: str) -> Optional[str]:
    """Return an error message if ``value`` is not a real Langfuse key.

    Returns ``None`` when the value matches the documented Langfuse
    prefix for ``env_name``, or when no prefix is registered for the
    name (in which case we trust the operator).  When validation
    fails the returned string is suitable for direct inclusion in a
    single log line — it names the env var and shows a safe preview.
    """
    expected = _LANGFUSE_KEY_PREFIXES.get(env_name, "")
    if not expected:
        return None
    if value.startswith(expected):
        return None
    return (
        f"{env_name}={_redact_key_preview(value)} "
        f"(expected {expected!r} prefix)"
    )


def _get_langfuse() -> Optional[Langfuse]:
    """Return a cached Langfuse client, or ``None`` if unavailable.

    Activation of this plugin is controlled by the Hermes plugin system —
    this function only handles the runtime-availability gate (SDK installed
    + credentials present). The result is cached: on the first call we try
    to construct a client, and every subsequent call returns that client
    (or fast-returns ``None`` if init failed).

    Thread-safe: ``_LANGFUSE_CLIENT_LOCK`` serializes the first build so
    concurrent callers can't both pass the ``is not None`` guard, both
    construct a ``Langfuse(**kwargs)`` client, and leak the loser's open
    HTTP connection and background flush thread (same TOCTOU class fixed
    for the Honcho and FAL clients in ``plugins/plugin_utils.py``).
    """
    global _LANGFUSE_CLIENT
    # Fast path — already settled (success or _INIT_FAILED); no lock needed.
    if _LANGFUSE_CLIENT is _INIT_FAILED:
        return None
    if _LANGFUSE_CLIENT is not None:
        return _LANGFUSE_CLIENT

    with _LANGFUSE_CLIENT_LOCK:
        # Re-check inside the lock: a racing thread may have completed init
        # while we were waiting.
        if _LANGFUSE_CLIENT is _INIT_FAILED:
            return None
        if _LANGFUSE_CLIENT is not None:
            return _LANGFUSE_CLIENT

        if Langfuse is None:
            logger.warning(
                "Langfuse plugin is enabled but the langfuse SDK is unavailable; "
                "tracing is disabled. Run `hermes tools` and configure Langfuse "
                "Observability to reinstall it."
            )
            _LANGFUSE_CLIENT = _INIT_FAILED
            return None

        public_key = _env("HERMES_LANGFUSE_PUBLIC_KEY") or _env("LANGFUSE_PUBLIC_KEY")
        secret_key = _env("HERMES_LANGFUSE_SECRET_KEY") or _env("LANGFUSE_SECRET_KEY")
        if not (public_key and secret_key):
            _LANGFUSE_CLIENT = _INIT_FAILED
            return None

        # Reject placeholder credentials with a one-shot warning so the
        # operator sees the misconfiguration instead of silently shipping a
        # broken observability stack (#23823).  The SDK does not validate
        # keys at construction time — it queues traces in memory and only
        # discovers the auth failure when the background flush thread tries
        # to post them, by which point the warning is buried under whatever
        # else the process is logging.  Catch it here, surface it once, and
        # short-circuit via the same _INIT_FAILED path as the empty case.
        placeholder_issues = [
            msg
            for msg in (
                _validate_langfuse_key("HERMES_LANGFUSE_PUBLIC_KEY", public_key),
                _validate_langfuse_key("HERMES_LANGFUSE_SECRET_KEY", secret_key),
            )
            if msg
        ]
        if placeholder_issues:
            logger.warning(
                "Langfuse plugin: credentials look like placeholders, traces will "
                "NOT be emitted (%s). Set real Langfuse keys (pk-lf-... / sk-lf-...) "
                "or unset HERMES_LANGFUSE_PUBLIC_KEY / HERMES_LANGFUSE_SECRET_KEY to "
                "silence this warning.",
                "; ".join(placeholder_issues),
            )
            _LANGFUSE_CLIENT = _INIT_FAILED
            return None

        base_url = _env("HERMES_LANGFUSE_BASE_URL") or _env("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com"
        environment = _env("HERMES_LANGFUSE_ENV") or _env("LANGFUSE_ENV")
        release = _env("HERMES_LANGFUSE_RELEASE") or _env("LANGFUSE_RELEASE")
        sample_rate = _env("HERMES_LANGFUSE_SAMPLE_RATE")

        kwargs: Dict[str, Any] = {
            "public_key": public_key,
            "secret_key": secret_key,
            "base_url": base_url,
        }
        if environment:
            kwargs["environment"] = environment
        if release:
            kwargs["release"] = release
        if sample_rate:
            try:
                kwargs["sample_rate"] = float(sample_rate)
            except ValueError:
                logger.warning("Invalid HERMES_LANGFUSE_SAMPLE_RATE=%r", sample_rate)

        try:
            _LANGFUSE_CLIENT = Langfuse(**kwargs)
        except Exception as exc:  # pragma: no cover - fail-open
            logger.warning("Could not initialize Langfuse client: %s", exc)
            _LANGFUSE_CLIENT = _INIT_FAILED
            return None

        # atexit is LIFO: registering AFTER the SDK's constructor (which installs
        # its own shutdown flush) means our finalizer runs FIRST at exit — root
        # spans ended there are still picked up by the SDK's exporter. Closes the
        # short-lived-process gap (kanban workers / hermes chat -q / cron): exit
        # with tool calls still queued left the root span un-ended → anonymous
        # trace with no name/session/metadata on the backend.
        try:
            import atexit

            atexit.register(_finalize_all_traces)
        except Exception:  # pragma: no cover - fail-open
            pass

        return _LANGFUSE_CLIENT


def _scope_prefix(task_id: str, session_id: str) -> str:
    """The task/session/thread prefix shared by every trace-key shape."""
    if task_id:
        return f"task:{task_id}"
    if session_id:
        return f"session:{session_id}"
    return f"thread:{threading.get_ident()}"


def _trace_key(
    task_id: str,
    session_id: str,
    *,
    turn_id: str = "",
    api_request_id: str = "",
) -> str:
    """Build a stable in-process trace scope key for one agent turn.

    Older Hermes paths only expose ``task_id``/``session_id``. Newer paths
    pass ``turn_id`` and ``api_request_id`` in LLM/tool hooks; when present,
    they must scope trace state so concurrent requests sharing one task/session
    never collide. ``turn_id`` is preferred over ``api_request_id`` so the
    turn-level ``post_llm_call`` hook (which carries ``turn_id`` but no
    ``api_request_id``) resolves to the same key as the request-level hooks.
    """
    if turn_id:
        return f"{_scope_prefix(task_id, session_id)}:turn:{turn_id}"
    if api_request_id:
        return f"{_scope_prefix(task_id, session_id)}:api:{api_request_id}"
    # Legacy shape: a bare ``task_id`` (NOT the ``task:`` prefix) when present,
    # otherwise the session/thread prefix. Kept distinct for backward
    # compatibility with keys minted before turn/request scoping existed.
    if task_id:
        return task_id
    return _scope_prefix(task_id, session_id)


def _state_for_turn(turn_id: str) -> Optional[str]:
    """Resolve a live trace key from a turn id alone.

    The subagent hooks carry ``parent_turn_id`` but no ``task_id``, and
    ``_scope_prefix`` prefers ``task_id`` when the LLM hooks minted the key —
    so rebuilding the key here would miss whenever a task id is in play.
    ``turn_id`` is already unique per turn, so match on its suffix instead.
    Caller must hold ``_STATE_LOCK``.
    """
    if not turn_id:
        return None
    suffix = f":turn:{turn_id}"
    for key in _TRACE_STATE:
        if key.endswith(suffix):
            return key
    return None


def _is_base64_data_uri(value: str) -> bool:
    prefix = value[:200].lower()
    return prefix.startswith("data:") and ";base64," in prefix


def _redact_data_uri(value: str) -> dict[str, Any]:
    header = value.split(",", 1)[0] if "," in value else "data:"
    media_type = header[5:].split(";", 1)[0] if header.startswith("data:") else ""
    return {
        "type": "data_uri",
        "media_type": media_type or None,
        "omitted": True,
        "length": len(value),
    }


def _truncate_text(value: str, max_chars: int) -> Any:
    # Langfuse SDK treats data:*;base64 strings as media and attempts to
    # decode them. Truncating those strings produces invalid base64 and noisy
    # "Error parsing base64 data URI" logs. Observability only needs metadata,
    # not raw image/audio payloads, so redact the whole data URI before it
    # reaches the SDK.
    if _is_base64_data_uri(value):
        return _redact_data_uri(value)
    # Redact BEFORE truncating so a secret straddling the cut point cannot
    # leak its prefix. Truncation is a size control, not redaction.
    if _capture_mode() == "sanitized":
        value = _redact_secrets(value)
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + f"... [truncated {len(value) - max_chars} chars]"


def _maybe_parse_json_string(value: str) -> Any:
    stripped = value.strip()
    if len(stripped) < 2 or stripped[0] not in "{[" or stripped[-1] not in "}]":
        if len(stripped) < 2 or stripped[0] not in "{[":
            return value
    try:
        parsed, idx = json.JSONDecoder().raw_decode(stripped)
    except Exception:
        return value
    if not isinstance(parsed, (dict, list)):
        return value

    trailing = stripped[idx:].strip()
    if not trailing:
        return parsed

    hint_key = "_hint" if trailing.startswith("[Hint:") else "_trailing_text"
    if isinstance(parsed, dict):
        merged = dict(parsed)
        key = hint_key if hint_key not in merged else "_trailing_text"
        merged[key] = trailing
        return merged

    return {"data": parsed, hint_key: trailing}


def _looks_like_read_file_payload(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    content = value.get("content")
    return (
        isinstance(content, str)
        and "total_lines" in value
        and "file_size" in value
        and "is_binary" in value
        and "is_image" in value
        and not value.get("error")
    )


def _parse_read_file_lines(content: str) -> list[dict[str, Any]]:
    if not isinstance(content, str) or not content:
        return []

    lines = []
    for raw_line in content.splitlines():
        match = _READ_FILE_LINE_RE.match(raw_line)
        if not match:
            return []
        lines.append({
            "line": int(match.group(1)),
            "text": match.group(2),
        })
    return lines


def _build_read_file_preview(lines: list[dict[str, Any]]) -> dict[str, Any]:
    if len(lines) <= (_READ_FILE_HEAD_LINES + _READ_FILE_TAIL_LINES):
        return {"lines": lines}

    return {
        "head": lines[:_READ_FILE_HEAD_LINES],
        "tail": lines[-_READ_FILE_TAIL_LINES:],
        "omitted_line_count": len(lines) - _READ_FILE_HEAD_LINES - _READ_FILE_TAIL_LINES,
    }


def _normalize_read_file_payload(value: dict[str, Any], *, args: Any = None) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    if isinstance(args, dict):
        path = args.get("path")
        offset = args.get("offset")
        limit = args.get("limit")
        if isinstance(path, str) and path:
            normalized["path"] = path
        if isinstance(offset, int):
            normalized["offset"] = offset
        if isinstance(limit, int):
            normalized["limit"] = limit

    lines = _parse_read_file_lines(value.get("content", ""))
    if lines:
        normalized["returned_lines"] = {
            "start": lines[0]["line"],
            "end": lines[-1]["line"],
            "count": len(lines),
        }
        normalized["content_preview"] = _build_read_file_preview(lines)
    elif value.get("content"):
        normalized["content_preview"] = {
            "text": value.get("content", ""),
        }

    for key in (
        "total_lines",
        "file_size",
        "truncated",
        "is_binary",
        "is_image",
        "hint",
        "_warning",
        "mime_type",
        "dimensions",
        "similar_files",
        "error",
    ):
        if key in value:
            normalized[key] = value[key]

    base64_content = value.get("base64_content")
    if isinstance(base64_content, str) and base64_content:
        normalized["base64_content"] = {
            "omitted": True,
            "length": len(base64_content),
        }

    return normalized


def _normalize_payload(value: Any, *, tool_name: str = "", args: Any = None) -> Any:
    if _looks_like_read_file_payload(value):
        return _normalize_read_file_payload(
            value,
            args=args if tool_name == "read_file" else None,
        )
    return value


def _safe_value(value: Any, *, max_chars: Optional[int] = None, depth: int = 0,
                parse_json_strings: bool = False) -> Any:
    max_chars = max_chars if max_chars is not None else int(_env("HERMES_LANGFUSE_MAX_CHARS", "12000") or "12000")
    if depth > 4:
        return "<max-depth>"
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "len": len(value)}
    if isinstance(value, str):
        if parse_json_strings:
            parsed = _maybe_parse_json_string(value)
            if parsed is not value:
                return _safe_value(parsed, max_chars=max_chars, depth=depth, parse_json_strings=True)
        return _truncate_text(value, max_chars)
    if isinstance(value, dict):
        normalized = _normalize_payload(value)
        if normalized is not value:
            return _safe_value(normalized, max_chars=max_chars, depth=depth, parse_json_strings=parse_json_strings)
        return {
            str(k): _safe_value(v, max_chars=max_chars, depth=depth + 1, parse_json_strings=parse_json_strings)
            for k, v in list(value.items())[:50]
        }
    if isinstance(value, (list, tuple, set)):
        return [
            _safe_value(v, max_chars=max_chars, depth=depth + 1, parse_json_strings=parse_json_strings)
            for v in list(value)[:50]
        ]
    if hasattr(value, "__dict__"):
        return _safe_value(vars(value), max_chars=max_chars, depth=depth + 1, parse_json_strings=parse_json_strings)
    return _truncate_text(repr(value), max_chars)


def _extract_last_user_message(messages: Any) -> Any:
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return {
                "role": "user",
                "content": _capture_content(message.get("content")),
            }
    return None


def _coerce_request_messages(
    *,
    request_messages: Any = None,
    messages: Any = None,
    conversation_history: Any = None,
    user_message: Any = None,
) -> list[dict[str, Any]]:
    for candidate in (request_messages, messages, conversation_history):
        if isinstance(candidate, list):
            return candidate
    if user_message is None:
        return []
    return [{"role": "user", "content": user_message}]


def _serialize_system_prompt(system_prompt: Any) -> Optional[dict[str, Any]]:
    """Normalize Anthropic/Bedrock ``system`` param or OpenAI-style system content for Langfuse."""
    if system_prompt is None:
        return None
    if isinstance(system_prompt, str):
        text = system_prompt.strip()
        if not text:
            return None
        return {"role": "system", "content": _capture_content(text)}
    if isinstance(system_prompt, list):
        parts: list[str] = []
        for block in system_prompt:
            if isinstance(block, dict):
                # Anthropic blocks carry {"type": "text", "text": ...}; Bedrock
                # Converse system blocks are {"text": ...} with no "type" key.
                block_type = block.get("type")
                if block_type == "text" or (block_type is None and "text" in block):
                    piece = block.get("text", "")
                    if isinstance(piece, str) and piece:
                        parts.append(piece)
            elif isinstance(block, str) and block:
                parts.append(block)
        if not parts:
            return None
        return {"role": "system", "content": _capture_content("\n\n".join(parts))}
    return None


def _messages_for_langfuse_input(
    *,
    request_messages: Any = None,
    messages: Any = None,
    conversation_history: Any = None,
    user_message: Any = None,
    system_prompt: Any = None,
    pre_coerced: Any = None,
) -> list[dict[str, Any]]:
    """Build generation input: include Anthropic ``system`` when split out of ``messages``.

    Pass ``pre_coerced`` to skip the internal ``_coerce_request_messages`` call
    when the caller already has the result — avoids double-coercion per hook.
    """
    raw = pre_coerced if pre_coerced is not None else _coerce_request_messages(
        request_messages=request_messages,
        messages=messages,
        conversation_history=conversation_history,
        user_message=user_message,
    )
    if raw and raw[0].get("role") == "system":
        return _serialize_messages(raw)
    system_msg = _serialize_system_prompt(system_prompt)
    if system_msg is None:
        return _serialize_messages(raw)
    return [system_msg, *_serialize_messages(raw)]


def _serialize_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    serialized = []
    for message in messages[-12:]:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        item = {
            "role": role,
            "content": _capture_content(
                message.get("content"),
                parse_json_strings=(role == "tool"),
            ),
        }
        if role == "tool":
            if message.get("tool_call_id"):
                item["tool_call_id"] = message.get("tool_call_id")
            if message.get("name"):
                item["name"] = _safe_value(message.get("name"))
        if message.get("tool_calls"):
            item["tool_calls"] = _capture_content(message.get("tool_calls"), parse_json_strings=True)
        serialized.append(item)
    return serialized


def _serialize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not tool_calls:
        return []
    serialized = []
    for tool_call in tool_calls:
        fn = getattr(tool_call, "function", None)
        name = getattr(fn, "name", None) if fn else None
        arguments = getattr(fn, "arguments", None) if fn else None
        safe_arguments = _capture_content(arguments, parse_json_strings=False)
        serialized.append({
            "id": getattr(tool_call, "id", None),
            "type": getattr(tool_call, "type", None) or "function",
            "name": name,
            "arguments": safe_arguments,
            "function": {
                "name": name,
                "arguments": safe_arguments,
            },
        })
    return serialized


def _extract_assistant_reasoning(message: Any) -> Any:
    for field in ("reasoning", "reasoning_content", "reasoning_details"):
        value = getattr(message, field, None)
        if value is not None:
            return _capture_content(value)
    return None


def _serialize_assistant_message(message: Any) -> dict[str, Any]:
    return {
        "content": _capture_content(getattr(message, "content", None)),
        "reasoning": _extract_assistant_reasoning(message),
        "tool_calls": _serialize_tool_calls(getattr(message, "tool_calls", None)),
    }


def _canonical_usage_and_cost(
    canonical: Any,
    *,
    provider: str,
    model: str,
    base_url: str,
) -> tuple[dict[str, int], dict[str, float]]:
    """Translate canonical Hermes usage into Langfuse usage and cost maps."""
    usage_details: Dict[str, int] = {
        "input": canonical.input_tokens,
        "output": canonical.output_tokens,
    }
    if canonical.cache_read_tokens:
        usage_details["cache_read_input_tokens"] = canonical.cache_read_tokens
    if canonical.cache_write_tokens:
        usage_details["cache_creation_input_tokens"] = canonical.cache_write_tokens
    if canonical.reasoning_tokens:
        usage_details["reasoning_tokens"] = canonical.reasoning_tokens

    cost_details: Dict[str, float] = {}
    try:
        from agent.usage_pricing import estimate_usage_cost, resolve_billing_route

        # Subscription-included routes (e.g. openai-codex): Langfuse treats
        # provided cost_details as authoritative and will not recalculate
        # from model pricing when explicit zeros are sent. Omit cost_details
        # entirely so Langfuse falls back to its own estimation.
        route = resolve_billing_route(model, provider=provider, base_url=base_url)
        if getattr(route, "billing_mode", "") == "subscription_included":
            return usage_details, cost_details

        cost = estimate_usage_cost(
            model,
            canonical,
            provider=provider,
            base_url=base_url,
            api_key="",
        )
    except Exception as exc:  # pragma: no cover - fail-open
        _debug(f"usage pricing failed: {exc}")
        return usage_details, cost_details

    # A partial component breakdown is not a valid request total.  In
    # particular, cache pricing may be unavailable even when input/output
    # rates are known.  Leave all costs absent so Langfuse cannot mistake a
    # partial subtotal for the full generation cost.
    if cost.amount_usd is None:
        return usage_details, cost_details

    # Langfuse only derives a total for the built-in input/output cost keys.
    # Cache/custom keys therefore need an explicit canonical total.  Use the
    # Hermes estimate rather than summing components because it also includes
    # request-level pricing.  Preserve the existing component-only payload for
    # subscription-included routes; their export policy is handled separately.
    # A zero estimate is not exported either: a priced model that billed no
    # tokens writes no per-type keys, and an explicit 0.0 total would be
    # treated as authoritative by Langfuse instead of left for it to derive.
    if cost.status != "included" and float(cost.amount_usd) > 0:
        cost_details["total"] = float(cost.amount_usd)

    # Langfuse cost_details keys match usage_details keys.  Keep the per-type
    # breakdown for dashboards in addition to the authoritative request total.
    try:
        from decimal import Decimal

        from agent.usage_pricing import get_pricing_entry

        one_million = Decimal("1000000")
        entry = get_pricing_entry(model, provider=provider, base_url=base_url)
        if entry:
            if entry.input_cost_per_million is not None and canonical.input_tokens:
                cost_details["input"] = float(
                    Decimal(canonical.input_tokens)
                    * entry.input_cost_per_million
                    / one_million
                )
            if entry.output_cost_per_million is not None and canonical.output_tokens:
                cost_details["output"] = float(
                    Decimal(canonical.output_tokens)
                    * entry.output_cost_per_million
                    / one_million
                )
            if entry.cache_read_cost_per_million is not None and canonical.cache_read_tokens:
                cost_details["cache_read_input_tokens"] = float(
                    Decimal(canonical.cache_read_tokens)
                    * entry.cache_read_cost_per_million
                    / one_million
                )
            if entry.cache_write_cost_per_million is not None and canonical.cache_write_tokens:
                cost_details["cache_creation_input_tokens"] = float(
                    Decimal(canonical.cache_write_tokens)
                    * entry.cache_write_cost_per_million
                    / one_million
                )
    except Exception:  # pragma: no cover - canonical total remains usable
        pass

    return usage_details, cost_details


def _usage_and_cost(response: Any, *, provider: str, api_mode: str, model: str, base_url: str) -> tuple[dict[str, int], dict[str, float]]:
    raw_usage = getattr(response, "usage", None)
    if not raw_usage:
        return {}, {}

    try:
        from agent.usage_pricing import normalize_usage

        canonical = normalize_usage(raw_usage, provider=provider, api_mode=api_mode)
        return _canonical_usage_and_cost(
            canonical,
            provider=provider,
            model=model,
            base_url=base_url,
        )
    except Exception as exc:  # pragma: no cover - fail-open
        _debug(f"usage normalization failed: {exc}")
        return {}, {}


def _publish_turn_trace_id(trace_id: str) -> None:
    """Export this turn's trace id to subprocesses spawned during the turn.

    Routed through ``gateway.session_context`` rather than written straight to
    ``os.environ``: that layer owns the task-local ContextVar and the
    subprocess-env bridge, so concurrent turns (MoA fan-out, gateway message
    tasks) each publish their own id instead of racing one process-global slot.

    Fail-open like every other path here — a Hermes build without the setter
    must still trace.
    """
    try:
        from gateway.session_context import set_current_turn_trace_id

        set_current_turn_trace_id(trace_id)
    except Exception as exc:  # pragma: no cover - fail-open
        _debug(f"publishing turn trace id failed: {exc}")


def _unpublish_turn_trace_id(trace_id: str) -> None:
    """Withdraw a finished turn's trace id (see :func:`_publish_turn_trace_id`).

    The id lives in two stores: the task-local ContextVar (authoritative for the
    subprocess-env bridge once the session machinery is engaged) and the
    process-global ``os.environ`` mirror (the CLI/cron fallback, skipped on the
    delegated-child path).  They can hold different ids, so each is checked and
    withdrawn on its own:

    * ContextVar holds *trace_id* — the ordinary same-task finish.  Clear
      through the setter, which drops the mirror too.
    * Only the mirror holds *trace_id* — ``_finish_trace`` ran in a different
      task than ``_start_root_trace`` published in (a session-end reap, or any
      teardown deferred to a background task), so this task never had the id to
      begin with.  Pop the mirror alone: blanking the ContextVar here would
      unpublish whatever *this* task is publishing, which may be a live turn —
      a delegated child's ContextVar-only publish leaves exactly that shape.
    * Neither holds it — a newer turn has already published over it.  Nothing to
      withdraw.

    So a stale reap can never unpublish the id a live turn is still handing to
    its subprocesses, and a cross-task finish still cleans up after itself.
    Note the ContextVar has two "absent" spellings (``_UNSET`` before any bind,
    ``""`` after a clear) and neither can carry an id across a task boundary —
    only the mirror can, which is why it has to be consulted separately.
    """
    try:
        from gateway.session_context import get_session_env, set_current_turn_trace_id

        if get_session_env("HERMES_LANGFUSE_TRACE_ID", "") == trace_id:
            set_current_turn_trace_id("")
        elif os.environ.get("HERMES_LANGFUSE_TRACE_ID") == trace_id:
            os.environ.pop("HERMES_LANGFUSE_TRACE_ID", None)
    except Exception as exc:  # pragma: no cover - fail-open
        _debug(f"clearing turn trace id failed: {exc}")


def _start_root_trace(task_key: str, *, task_id: str, session_id: str, platform: str, provider: str, model: str,
                      api_mode: str, messages: Any, client: Langfuse,
                      turn_id: str = "", api_request_id: str = "") -> TraceState:
    trace_id = client.create_trace_id(seed=f"{session_id or 'sessionless'}::{task_id or task_key}")
    _publish_turn_trace_id(trace_id)
    trace_input = _extract_last_user_message(messages)
    metadata = {
        "source": "hermes",
        "task_id": task_id,
        "turn_id": turn_id,
        "api_request_id": api_request_id,
        "platform": platform,
        "provider": provider,
        "model": model,
        "api_mode": api_mode,
        "capture_mode": _capture_mode(),
    }

    # session_id must be passed in trace_context for Langfuse session grouping.
    trace_ctx: Dict[str, Any] = {"trace_id": trace_id}
    if session_id:
        trace_ctx["session_id"] = session_id

    if propagate_attributes is not None:
        try:
            with propagate_attributes(
                session_id=session_id or task_key,
                trace_name="Hermes turn",
                tags=["hermes", "langfuse"],
            ):
                root_ctx = client.start_as_current_observation(
                    trace_context=trace_ctx,
                    name="Hermes turn",
                    as_type="chain",
                    input=trace_input,
                    metadata=metadata,
                    end_on_exit=False,
                )
                root_span = root_ctx.__enter__()
        except Exception:
            root_ctx = client.start_as_current_observation(
                trace_context=trace_ctx,
                name="Hermes turn",
                as_type="chain",
                input=trace_input,
                metadata=metadata,
                end_on_exit=False,
            )
            root_span = root_ctx.__enter__()
    else:
        root_ctx = client.start_as_current_observation(
            trace_context=trace_ctx,
            name="Hermes turn",
            as_type="chain",
            input=trace_input,
            metadata=metadata,
            end_on_exit=False,
        )
        root_span = root_ctx.__enter__()

    # SDK v3 uses update_trace() (not set_trace_io). Failures must never block
    # the rest of the turn — the observation still carries input from start.
    try:
        root_span.update_trace(input=trace_input)
    except Exception as exc:
        _debug(f"update_trace(input) failed: {exc}")

    _debug(f"started trace {trace_id} for {task_key}")
    return TraceState(trace_id=trace_id, root_ctx=root_ctx, root_span=root_span)


def _start_child_observation(state: TraceState, *, client: Langfuse, name: str, as_type: str,
                             input_value: Any, metadata: Optional[dict] = None,
                             model: Optional[str] = None, model_parameters: Optional[dict] = None) -> Any:
    return state.root_span.start_observation(
        name=name,
        as_type=as_type,
        input=input_value,
        metadata=metadata or {},
        model=model,
        model_parameters=model_parameters,
    )


def _end_observation(observation: Any, *, output: Any = None, metadata: Optional[dict] = None,
                     usage_details: Optional[dict] = None, cost_details: Optional[dict] = None) -> None:
    if observation is None:
        return
    try:
        update_kwargs: Dict[str, Any] = {}
        if output is not None:
            update_kwargs["output"] = output
        if metadata:
            update_kwargs["metadata"] = metadata
        if usage_details:
            update_kwargs["usage_details"] = usage_details
        if cost_details:
            update_kwargs["cost_details"] = cost_details
        if update_kwargs:
            observation.update(**update_kwargs)
        observation.end()
    except Exception as exc:  # pragma: no cover - fail-open
        _debug(f"end observation failed: {exc}")


def _merge_trace_output(output: Any, state: TraceState) -> Any:
    if not state.turn_tool_calls:
        return output

    merged = dict(output) if isinstance(output, dict) else {"content": output}
    merged["tool_calls"] = list(state.turn_tool_calls)
    return merged


def _evict_stale_locked() -> None:
    """Drop least-recently-updated trace state to make room for a new entry.

    Caller MUST hold ``_STATE_LOCK`` and call this immediately before inserting
    one new entry. Bounds the leak from turns that never reach ``_finish_trace``
    (interrupted / tool-only final step / empty final content), whose unique
    per-turn key would otherwise linger forever. We evict down to
    ``_MAX_TRACE_STATE - 1`` so that the about-to-be-added entry leaves the dict
    at ``_MAX_TRACE_STATE`` — a true ceiling. The evicted entry's root span is
    ended so it is not left dangling on the Langfuse side.
    """
    over = len(_TRACE_STATE) - (_MAX_TRACE_STATE - 1)
    if over <= 0:
        return
    # Oldest-first by last_updated_at; evict just enough to make room.
    stale = sorted(_TRACE_STATE.items(), key=lambda kv: kv[1].last_updated_at)[:over]
    for key, state in stale:
        _TRACE_STATE.pop(key, None)
        try:
            state.root_span.end()
            if state.root_ctx is not None:
                try:
                    state.root_ctx.__exit__(None, None, None)
                except Exception:  # pragma: no cover - fail-open
                    pass
        except Exception as exc:  # pragma: no cover - fail-open
            _debug(f"evict stale trace failed: {exc}")


def _finalize_all_traces() -> None:
    """End every open root span so short-lived processes export complete traces.

    Gateway turns normally end their root span via ``_finish_trace`` (final
    assistant message with no tool calls). But short-lived CLI processes —
    kanban workers, ``hermes chat -q`` one-shots, cron jobs — can exit while
    the last LLM call still has tool calls queued, leaving the root span
    un-ended. Ended children DO export via the SDK's own atexit flush, so the
    backend shows an anonymous trace (no name/session/metadata) whose
    observations all point at a root that never arrived (observed live
    2026-08-08: kanban workers produced 4 such traces in one tick).

    Registered with ``atexit`` AFTER the Langfuse client is constructed:
    atexit is LIFO, so this runs BEFORE the SDK's own shutdown hook — spans
    ended here still get flushed by the SDK's exporter.
    """
    with _STATE_LOCK:
        states = list(_TRACE_STATE.items())
        _TRACE_STATE.clear()
    for _key, state in states:
        try:
            for observation in state.generations.values():
                _end_observation(observation)
            for observation in state.tools.values():
                _end_observation(observation)
            for queue in state.pending_tools_by_name.values():
                for observation in queue:
                    _end_observation(observation)
            for observation in state.subagents.values():
                _end_observation(observation)
            state.root_span.end()
            # Exit the root observation's context manager so its generator
            # unwinds now, while opentelemetry.trace.Span is still a real
            # type — otherwise GC closes it during interpreter teardown and
            # use_span's isinstance(span, Span) raises TypeError.
            if state.root_ctx is not None:
                try:
                    state.root_ctx.__exit__(None, None, None)
                except Exception:  # pragma: no cover - fail-open
                    pass
        except Exception as exc:  # pragma: no cover - fail-open
            _debug(f"atexit finalize failed for {_key}: {exc}")
    if states:
        client = _get_langfuse()
        if client is not None:
            try:
                client.flush()
            except Exception:
                pass


def _finish_trace(task_key: str, *, output: Any = None) -> None:
    client = _get_langfuse()
    if client is None:
        return

    with _STATE_LOCK:
        state = _TRACE_STATE.pop(task_key, None)
    if state is None:
        return

    try:
        for observation in state.generations.values():
            _end_observation(observation)
        for observation in state.tools.values():
            _end_observation(observation)
        for queue in state.pending_tools_by_name.values():
            for observation in queue:
                _end_observation(observation)
        final_output = _merge_trace_output(output, state)
        if final_output is not None:
            # update_trace sets TRACE-level Input/Output columns in the UI
            # (SDK v3; set_trace_io was the pre-v3 spelling). Root observation
            # I/O is set via update(); never let either call prevent end() —
            # otherwise generations/tools export without a CHAIN root and the
            # list view looks half-empty.
            try:
                state.root_span.update_trace(output=final_output)
            except Exception as exc:
                _debug(f"update_trace(output) failed: {exc}")
            try:
                state.root_span.update(output=final_output)
            except Exception as exc:
                _debug(f"root update(output) failed: {exc}")
        try:
            state.root_span.end()
        except Exception as exc:
            _debug(f"root end() failed: {exc}")
        # Properly exit the root context manager so the generator unwinds
        # now, while opentelemetry.trace.Span is still a real type.  Without
        # this the generator is left suspended; at interpreter teardown the
        # GC calls .close(), which throws GeneratorExit through use_span
        # __exit__ -> isinstance(span, Span) -- but Span has been torn down
        # to None by then, producing the TypeError traceback on quit.
        if state.root_ctx is not None:
            try:
                state.root_ctx.__exit__(None, None, None)
            except Exception:  # pragma: no cover - fail-open
                pass
    except Exception as exc:  # pragma: no cover - fail-open
        _debug(f"finish trace failed: {exc}")
        # Last-chance end so an earlier unexpected error still exports the root.
        try:
            state.root_span.end()
        except Exception:
            pass
    finally:
        # Withdraw before the flush so a finished turn's id can never reach a
        # subprocess spawned by whatever runs next in this context.
        _unpublish_turn_trace_id(state.trace_id)
        try:
            client.flush()
        except Exception:
            pass


def _assistant_has_tool_calls(message: Any) -> bool:
    return bool(getattr(message, "tool_calls", None))


def _request_key(api_call_count: Any) -> str:
    return str(api_call_count or 0)


def on_pre_llm_call(*, task_id: str = "", session_id: str = "", platform: str = "", model: str = "",
                    provider: str = "", base_url: str = "", api_mode: str = "",
                    api_call_count: int = 0, messages: Any = None, turn_type: str = "user",
                    conversation_history: Any = None, user_message: Any = None,
                    turn_id: str = "", api_request_id: str = "", **_: Any) -> None:
    # Older Hermes branches used pre_llm_call for request-scoped tracing and
    # passed the actual API messages. Current Hermes also has a turn-scoped
    # pre_llm_call used for context injection; tracing that hook creates an
    # extra orphan/root trace before the real request trace. Only trace the
    # legacy request-shaped call here.
    if not isinstance(messages, list):
        return

    client = _get_langfuse()
    if client is None:
        return

    # messages is a list only for legacy Hermes branches that fired
    # pre_llm_call with API messages directly. Current Hermes fires
    # pre_llm_call for context injection (conversation_history/user_message,
    # no messages list) — tracing that would create orphan traces.
    task_key = _trace_key(
        task_id,
        session_id,
        turn_id=turn_id,
        api_request_id=api_request_id,
    )

    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        if state is None:
            state = _start_root_trace(
                task_key,
                task_id=task_id,
                session_id=session_id,
                platform=platform,
                provider=provider,
                model=model,
                api_mode=api_mode,
                messages=messages,
                client=client,
                turn_id=turn_id,
                api_request_id=api_request_id,
            )
            _evict_stale_locked()
            _TRACE_STATE[task_key] = state
        state.last_updated_at = time.time()


def _emit_moa_reference_generations(state: TraceState, *, client: Langfuse,
                                    references: Any) -> None:
    """Record each MoA advisor as its own generation under the turn.

    MoA returns only the aggregator's response, so without this the whole
    fan-out collapses into one generation priced at the aggregator's model.
    Each advisor carries its own model, usage, and dollars — advisors routinely
    run on a different provider than the aggregator, so their spend cannot be
    priced at the aggregator's rate.
    """
    if not isinstance(references, list) or not references:
        return
    fingerprint = json.dumps(
        [
            [r.get("label"), r.get("model"), (r.get("usage") or {}).get("output_tokens")]
            for r in references
            if isinstance(r, dict)
        ],
        sort_keys=True,
        default=str,
    )
    with _STATE_LOCK:
        if fingerprint in state.moa_emitted:
            return
        state.moa_emitted.add(fingerprint)

    for ref in references:
        if not isinstance(ref, dict):
            continue
        usage = ref.get("usage") or {}
        usage_details = {}
        if isinstance(usage, dict):
            if usage.get("input_tokens"):
                usage_details["input"] = usage["input_tokens"]
            if usage.get("output_tokens"):
                usage_details["output"] = usage["output_tokens"]
            if usage.get("cache_read_tokens"):
                usage_details["cache_read_input_tokens"] = usage["cache_read_tokens"]
            if usage.get("cache_write_tokens"):
                usage_details["cache_creation_input_tokens"] = usage["cache_write_tokens"]
            if usage.get("reasoning_tokens"):
                usage_details["reasoning_tokens"] = usage["reasoning_tokens"]
        cost_details = {}
        cost_usd = ref.get("cost_usd")
        if isinstance(cost_usd, (int, float)):
            cost_details["total"] = float(cost_usd)

        label = ref.get("label") or "advisor"
        metadata = {"moa_role": "reference", "label": label}
        for key in ("provider", "cost_status", "cost_source", "temperature"):
            if ref.get(key) is not None:
                metadata[key] = ref[key]

        observation = _start_child_observation(
            state,
            client=client,
            name=f"MoA advisor: {label}",
            as_type="generation",
            input_value=None,
            metadata=metadata,
            model=ref.get("model"),
        )
        _end_observation(
            observation,
            output=_capture_content(ref.get("output")),
            usage_details=usage_details,
            cost_details=cost_details,
            metadata=metadata,
        )


def on_pre_llm_request(
    *,
    task_id: str = "",
    session_id: str = "",
    platform: str = "",
    model: str = "",
    provider: str = "",
    base_url: str = "",
    api_mode: str = "",
    api_call_count: int = 0,
    request_messages: Any = None,
    messages: Any = None,
    turn_type: str = "user",
    message_count: int = 0,
    tool_count: int = 0,
    approx_input_tokens: int = 0,
    request_char_count: int = 0,
    max_tokens: Any = None,
    conversation_history: Any = None,
    user_message: Any = None,
    turn_id: str = "",
    api_request_id: str = "",
    request: Any = None,
    system_prompt: Any = None,
    **_: Any,
) -> None:
    client = _get_langfuse()
    if client is None:
        return

    # ``model`` is the agent's current attribute at hook time; the request
    # body carries the model actually being dispatched. They can diverge
    # (mid-session /model switch propagation, provider fallback, middleware
    # rewrites) — prefer the wire truth for generation attribution.
    if isinstance(request, dict):
        body = request.get("body")
        if isinstance(body, dict):
            body_model = body.get("model")
            if isinstance(body_model, str) and body_model:
                model = body_model

    input_messages = _coerce_request_messages(
        request_messages=request_messages,
        messages=messages,
        conversation_history=conversation_history,
        user_message=user_message,
    )
    langfuse_input = _messages_for_langfuse_input(
        request_messages=request_messages,
        messages=messages,
        conversation_history=conversation_history,
        user_message=user_message,
        system_prompt=system_prompt,
        pre_coerced=input_messages,
    )
    system_chars = 0
    if langfuse_input and langfuse_input[0].get("role") == "system":
        system_chars = len(str(langfuse_input[0].get("content") or ""))

    task_key = _trace_key(
        task_id,
        session_id,
        turn_id=turn_id,
        api_request_id=api_request_id,
    )
    req_key = _request_key(api_call_count)

    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        if state is None:
            state = _start_root_trace(
                task_key,
                task_id=task_id,
                session_id=session_id,
                platform=platform,
                provider=provider,
                model=model,
                api_mode=api_mode,
                messages=input_messages,
                client=client,
                turn_id=turn_id,
                api_request_id=api_request_id,
            )
            _evict_stale_locked()
            _TRACE_STATE[task_key] = state
        state.last_updated_at = time.time()
        previous = state.generations.pop(req_key, None)
        if previous is not None:
            _end_observation(previous)
        gen_metadata = {
            "provider": provider,
            "platform": platform,
            "api_mode": api_mode,
            "base_url": base_url,
            "message_count": message_count,
            "approx_input_tokens": approx_input_tokens,
        }
        if system_chars:
            gen_metadata["system_prompt_chars"] = system_chars
        state.generations[req_key] = _start_child_observation(
            state,
            client=client,
            name=f"LLM call {api_call_count}",
            as_type="generation",
            input_value=langfuse_input,
            metadata=gen_metadata,
            model=model,
            model_parameters={"api_mode": api_mode, "provider": provider},
        )


def on_post_llm_call(*, task_id: str = "", session_id: str = "", provider: str = "", base_url: str = "",
                     api_mode: str = "", model: str = "", api_call_count: int = 0,
                     assistant_message: Any = None, response: Any = None,
                     api_duration: float = 0.0, finish_reason: str = "",
                     usage: Any = None, assistant_content_chars: int = 0,
                     assistant_tool_call_count: int = 0, assistant_response: Any = None,
                     turn_id: str = "", api_request_id: str = "",
                     response_model: Any = None, moa_references: Any = None,
                     **_: Any) -> None:
    client = _get_langfuse()
    if client is None:
        return

    # The provider's response echoes the model that actually served the
    # request. Prefer it over the agent attribute, which can be stale after
    # a mid-session model switch or fallback.
    if isinstance(response_model, str) and response_model:
        model = response_model

    task_key = _trace_key(
        task_id,
        session_id,
        turn_id=turn_id,
        api_request_id=api_request_id,
    )
    req_key = _request_key(api_call_count)

    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        generation = state.generations.pop(req_key, None) if state else None
    if state is None or generation is None:
        return

    if moa_references:
        _emit_moa_reference_generations(state, client=client, references=moa_references)

    # Handle both call patterns:
    # 1. post_api_request: passes usage (dict), assistant_content_chars, assistant_tool_call_count
    # 2. post_llm_call: passes assistant_message (object), response (object), assistant_response (str)
    if assistant_message is not None:
        output = _serialize_assistant_message(assistant_message)
    elif assistant_response is not None:
        # post_llm_call passes assistant_response as a plain string
        output = {"content": _capture_content(assistant_response), "reasoning": None, "tool_calls": []}
    else:
        # post_api_request path — reconstruct from summary kwargs
        output = {
            "content": f"[{assistant_content_chars} chars]" if assistant_content_chars else None,
            "reasoning": None,
            "tool_calls": [{"id": f"tc_{i}"} for i in range(assistant_tool_call_count)] if assistant_tool_call_count else [],
        }

    if output.get("tool_calls"):
        state.turn_tool_calls.extend(output["tool_calls"])

    # Extract usage: prefer a real response object that carries usage, else
    # fall back to the usage summary dict from post_api_request.
    #
    # post_api_request passes `response` as a SANITIZED dict (no ``.usage``
    # attribute) alongside a separate `usage` summary dict. Gating on
    # ``response is not None`` here took the response-object path on that dict,
    # where ``getattr(response, "usage", None)`` is always None — so usage and
    # cost were silently dropped for every gateway turn. Gate on a real
    # ``.usage`` attribute instead so the usage-dict fallback below is reached.
    if getattr(response, "usage", None) is not None:
        usage_details, cost_details = _usage_and_cost(
            response,
            provider=provider,
            api_mode=api_mode,
            model=model,
            base_url=base_url,
        )
    elif isinstance(usage, dict) and usage:
        # post_api_request passes a pre-built CanonicalUsage summary dict.
        _input = usage.get("input_tokens", 0)
        _output = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
        _cache_read = usage.get("cache_read_tokens", 0)
        _cache_write = usage.get("cache_write_tokens", 0)
        _reasoning = usage.get("reasoning_tokens", 0)
        try:
            from agent.usage_pricing import CanonicalUsage

            _cu = CanonicalUsage(
                input_tokens=_input,
                output_tokens=_output,
                cache_read_tokens=_cache_read,
                cache_write_tokens=_cache_write,
                reasoning_tokens=_reasoning,
                request_count=usage.get("request_count", 1),
            )
            usage_details, cost_details = _canonical_usage_and_cost(
                _cu,
                provider=provider,
                model=model,
                base_url=base_url,
            )
        except Exception:
            usage_details, cost_details = {}, {}
    else:
        usage_details, cost_details = {}, {}

    tool_count = len(output.get("tool_calls", [])) or assistant_tool_call_count
    gen_metadata: Dict[str, Any] = {"tool_call_count": tool_count}
    if api_duration and api_duration > 0:
        gen_metadata["api_duration_s"] = round(api_duration, 3)
    if finish_reason:
        gen_metadata["finish_reason"] = finish_reason
    _end_observation(
        generation,
        output=output,
        usage_details=usage_details,
        cost_details=cost_details,
        metadata=gen_metadata,
    )

    has_tools = _assistant_has_tool_calls(assistant_message) if assistant_message else (assistant_tool_call_count > 0)
    has_content = bool(output.get("content"))
    if not has_tools and has_content:
        _finish_trace(task_key, output=output)


def on_pre_tool_call(*, tool_name: str = "", args: Any = None, task_id: str = "",
                     session_id: str = "", tool_call_id: str = "",
                     turn_id: str = "", api_request_id: str = "", **_: Any) -> None:
    client = _get_langfuse()
    if client is None:
        return

    task_key = _trace_key(
        task_id,
        session_id,
        turn_id=turn_id,
        api_request_id=api_request_id,
    )

    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        if state is None:
            return
        observation = _start_child_observation(
            state,
            client=client,
            name=f"Tool: {tool_name}",
            as_type="tool",
            input_value=_capture_content(args),
            metadata={"tool_name": tool_name, "tool_call_id": tool_call_id},
        )
        if tool_call_id:
            state.tools[tool_call_id] = observation
        else:
            state.pending_tools_by_name.setdefault(tool_name, []).append(observation)


def on_post_tool_call(*, tool_name: str = "", args: Any = None, result: Any = None,
                      task_id: str = "", session_id: str = "", tool_call_id: str = "",
                      turn_id: str = "", api_request_id: str = "", **_: Any) -> None:
    task_key = _trace_key(
        task_id,
        session_id,
        turn_id=turn_id,
        api_request_id=api_request_id,
    )
    observation = None

    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        if state is None:
            return
        if tool_call_id:
            observation = state.tools.pop(tool_call_id, None)
        if observation is None:
            queue = state.pending_tools_by_name.get(tool_name)
            if queue:
                observation = queue.pop(0)
                if not queue:
                    state.pending_tools_by_name.pop(tool_name, None)

    if observation is None:
        return

    if _capture_mode() == "metadata":
        safe_result_value = _describe_content(result)
    else:
        if isinstance(result, str):
            result_value = _maybe_parse_json_string(result)
        else:
            result_value = result
        result_value = _normalize_payload(result_value, tool_name=tool_name, args=args)
        safe_result_value = _safe_value(result_value, parse_json_strings=True)

    # Backfill so the generation's tool_call record carries the result alongside arguments.
    if tool_call_id:
        with _STATE_LOCK:
            state = _TRACE_STATE.get(task_key)
            if state is not None:
                for tool_call in reversed(state.turn_tool_calls):
                    if tool_call.get("id") == tool_call_id:
                        tool_call["output"] = safe_result_value
                        function_payload = tool_call.get("function")
                        if isinstance(function_payload, dict):
                            function_payload["output"] = safe_result_value
                        break

    _end_observation(
        observation,
        output=safe_result_value,
        metadata={"tool_name": tool_name, "args": _capture_content(args, parse_json_strings=True)},
    )


def on_api_request_error(*, task_id: str = "", session_id: str = "", provider: str = "",
                         model: str = "", api_mode: str = "", api_call_count: int = 0,
                         api_duration: float = 0.0, status_code: Any = None,
                         retry_count: Any = None, max_retries: Any = None,
                         retryable: Any = None, reason: Any = None, error: Any = None,
                         turn_id: str = "", api_request_id: str = "",
                         **_: Any) -> None:
    """Close the open generation for a failed API request.

    Without this, a failed request leaves its generation open until trace
    eviction — the failure is invisible in Langfuse and the turn appears
    to hang. Marks the generation with ERROR level and the error summary.
    If the request was not retryable (or retries are exhausted), the turn
    is finished too, since the agent loop is about to unwind.
    """
    client = _get_langfuse()
    if client is None:
        return

    task_key = _trace_key(
        task_id,
        session_id,
        turn_id=turn_id,
        api_request_id=api_request_id,
    )
    req_key = _request_key(api_call_count)

    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        generation = state.generations.pop(req_key, None) if state else None
    if state is None:
        return

    error_type = ""
    error_message = ""
    if isinstance(error, dict):
        error_type = str(error.get("type") or "")
        error_message = str(error.get("message") or "")

    error_metadata: Dict[str, Any] = {
        "error": True,
        "error_type": error_type,
        # Error messages can embed request fragments (URLs w/ keys, prompt
        # echoes) — run them through the capture pipeline like content.
        "error_message": _capture_content(error_message),
    }
    if status_code is not None:
        error_metadata["status_code"] = status_code
    if retry_count is not None:
        error_metadata["retry_count"] = retry_count
    if max_retries is not None:
        error_metadata["max_retries"] = max_retries
    if retryable is not None:
        error_metadata["retryable"] = retryable
    if reason:
        error_metadata["reason"] = str(reason)
    if api_duration and api_duration > 0:
        error_metadata["api_duration_s"] = round(api_duration, 3)

    if generation is not None:
        try:
            generation.update(
                level="ERROR",
                status_message=(error_type or "api_request_error")[:200],
            )
        except Exception as exc:  # pragma: no cover - fail-open
            _debug(f"error-level update failed: {exc}")
        _end_observation(generation, metadata=error_metadata)

    # A retryable failure will be followed by another pre_api_request on the
    # same trace; keep the turn open. A terminal failure ends the turn.
    if retryable is False:
        _finish_trace(task_key, output={"error": error_metadata})
    else:
        state.last_updated_at = time.time()


def on_session_finalize(*, session_id: str = "", reason: str = "", **_: Any) -> None:
    """True session-end boundary: close any traces still open and flush.

    A turn that ended on a tool-only or empty final response never reaches
    ``_finish_trace``; without this hook its root span dangles until state
    eviction and queued events can be lost on process exit.
    """
    # Only act on an already-constructed client — do NOT lazily initialize
    # one at finalize time; if init never happened there are no traces.
    client = _LANGFUSE_CLIENT
    if client is None or client is _INIT_FAILED or not hasattr(client, "flush"):
        return

    # Close every trace belonging to this session (or all, when no
    # session_id is provided — process-level finalization). Trace keys carry
    # the session as either "session:<id>" (no task) or "task:<id>" (gateway
    # sets task_id == session_id), plus the legacy bare-task_id shape — match
    # on the id in any segment.
    if session_id:
        fragments = (f"session:{session_id}", f"task:{session_id}")
        with _STATE_LOCK:
            keys = [
                k for k in _TRACE_STATE
                if k == session_id or any(f in k for f in fragments)
            ]
    else:
        with _STATE_LOCK:
            keys = list(_TRACE_STATE)

    for key in keys:
        _finish_trace(key)

    try:
        client.flush()
    except Exception as exc:  # pragma: no cover - fail-open
        _debug(f"finalize flush failed: {exc}")

    # Explicitly shut down the Langfuse client — but only at a true
    # process-exit boundary.  on_session_finalize also fires on /new, /reset
    # (reason "session_boundary"/"new_session") and gateway session expiry
    # ("session_expired"), where the process lives on and the cached client
    # must keep exporting for subsequent sessions.  The shutdown matters at
    # exit because the Langfuse SDK's own atexit handler runs during
    # interpreter finalization — by then module globals (notably
    # opentelemetry.trace.Span) may already be torn down to None, and the
    # SDK's span-finalization path (use_span -> isinstance(span, Span))
    # raises "TypeError: isinstance() arg 2 must be a type" which surfaces
    # as a noisy "Exception ignored in: <generator>" traceback on quit.
    # Calling shutdown() here flushes pending spans and joins the background
    # export threads while all modules are intact; the SDK's atexit handler
    # then becomes a no-op.
    if reason == "shutdown":
        shutdown = getattr(client, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception as exc:  # pragma: no cover - fail-open
                _debug(f"langfuse shutdown failed: {exc}")


def on_subagent_start(*, parent_session_id: Any = None, parent_turn_id: str = "",
                      parent_subagent_id: Any = None, child_session_id: Any = None,
                      child_subagent_id: Any = None, child_role: str = "",
                      child_goal: Any = None, **_: Any) -> None:
    client = _get_langfuse()
    if client is None or not child_session_id:
        return

    with _STATE_LOCK:
        key = _state_for_turn(parent_turn_id)
        state = _TRACE_STATE.get(key) if key else None
        if state is None:
            return
        metadata = {
            "child_session_id": child_session_id,
            "child_subagent_id": child_subagent_id,
            "child_role": child_role,
        }
        if parent_subagent_id:
            metadata["parent_subagent_id"] = parent_subagent_id
        state.subagents[str(child_session_id)] = _start_child_observation(
            state,
            client=client,
            name=f"Subagent: {child_role or 'delegate'}",
            as_type="span",
            input_value=_capture_content(child_goal),
            metadata=metadata,
        )


def on_subagent_stop(*, parent_session_id: Any = None, parent_turn_id: str = "",
                     child_session_id: Any = None, child_role: str = "",
                     child_summary: Any = None, child_status: Any = None,
                     tool_call_history: Any = None, duration_ms: Any = None,
                     **_: Any) -> None:
    if not child_session_id:
        return

    with _STATE_LOCK:
        key = _state_for_turn(parent_turn_id)
        state = _TRACE_STATE.get(key) if key else None
        if state is None:
            return
        observation = state.subagents.pop(str(child_session_id), None)

    if observation is None:
        return

    metadata: Dict[str, Any] = {"child_role": child_role}
    if child_status:
        metadata["status"] = child_status
    if duration_ms:
        metadata["duration_ms"] = duration_ms
    if isinstance(tool_call_history, list):
        metadata["tool_call_count"] = len(tool_call_history)
        metadata["tool_calls"] = _capture_content(tool_call_history)
    _end_observation(
        observation,
        output=_capture_content(child_summary),
        metadata=metadata,
    )


def register(ctx) -> None:
    # Register for both hook name variants so the plugin works across
    # Hermes versions.  pre_api_request / post_api_request fire per API
    # call (preferred); pre_llm_call / post_llm_call fire once per turn.
    ctx.register_hook("pre_api_request", on_pre_llm_request)
    ctx.register_hook("post_api_request", on_post_llm_call)
    ctx.register_hook("api_request_error", on_api_request_error)
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("post_llm_call", on_post_llm_call)
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("on_session_finalize", on_session_finalize)
    ctx.register_hook("on_session_end", on_session_finalize)
    ctx.register_hook("subagent_start", on_subagent_start)
    ctx.register_hook("subagent_stop", on_subagent_stop)
