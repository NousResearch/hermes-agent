"""langfuse — Hermes plugin tracing conversations, LLM calls and tool usage to Langfuse.

Activated via ``plugins.enabled``; hooks are inert without the ``langfuse`` SDK
and credentials. Env: HERMES_LANGFUSE_PUBLIC_KEY / SECRET_KEY (required),
BASE_URL, ENV, RELEASE, SAMPLE_RATE, MAX_CHARS (12000), DEBUG, and CAPTURE =
metadata (sizes/ids/usage only) | sanitized (default: secret redaction +
truncation) | full (truncated raw content). See README.md.
"""
from __future__ import annotations

import atexit
import contextlib
import hashlib
import hmac
import importlib.metadata
import json
import logging
import os
import re
import secrets
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

try:
    from langfuse.types import MaskOtelSpansResult, OtelSpanPatch
except Exception:  # pragma: no cover - legacy SDK compatibility
    MaskOtelSpansResult = None
    OtelSpanPatch = None


@dataclass
class TraceState:
    trace_id: str
    root_ctx: Any
    root_span: Any
    content_sampled: bool = True
    generations: Dict[str, Any] = field(default_factory=dict)
    tools: Dict[str, Any] = field(default_factory=dict)
    pending_tools_by_name: Dict[str, list] = field(default_factory=dict)
    turn_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # Keyed by child_session_id: subagent_stop carries no child_subagent_id.
    subagents: Dict[str, Any] = field(default_factory=dict)
    # Fingerprints of MoA fan-outs already recorded: the client holds its last
    # fan-out until the next one, so tool-loop turns would re-emit advisors.
    moa_emitted: set = field(default_factory=set)
    last_updated_at: float = field(default_factory=time.time)


_STATE_LOCK = threading.Lock()
_TRACE_STATE: Dict[str, TraceState] = {}
# Ceiling on live trace state (per turn_id): turns that never reach _finish_trace
# would leak forever, so over the cap the least-recently-updated are evicted.
# Bounds the leak, not concurrency.
_MAX_TRACE_STATE = 256
_LANGFUSE_CLIENT = None
# Separate from _STATE_LOCK (hot path) so the two never nest; serializes the
# first client build so racing callers can't each construct a client.
_LANGFUSE_CLIENT_LOCK = threading.Lock()
_READ_FILE_LINE_RE = re.compile(r"^\s*(\d+)\|(.*)$")
_READ_FILE_HEAD_LINES = 25
_READ_FILE_TAIL_LINES = 15
_READ_FILE_META_KEYS = ("total_lines", "file_size", "truncated", "is_binary", "is_image", "hint",
                        "_warning", "mime_type", "dimensions", "similar_files", "error")

# Langfuse-issued keys always carry these prefixes. Anything else is a leftover
# template value: the SDK accepts it at construction time but silently drops
# every trace at flush time (#23823).
_LANGFUSE_KEY_PREFIXES: Dict[str, str] = {
    "HERMES_LANGFUSE_PUBLIC_KEY": "pk-lf-",
    "HERMES_LANGFUSE_SECRET_KEY": "sk-lf-",
}

# (langfuse usage key, CanonicalUsage attribute / summary-dict key, PricingEntry attribute)
_USAGE_FIELDS = (
    ("input", "input_tokens", "input_cost_per_million"),
    ("output", "output_tokens", "output_cost_per_million"),
    ("cache_read_input_tokens", "cache_read_tokens", "cache_read_cost_per_million"),
    ("cache_creation_input_tokens", "cache_write_tokens", "cache_write_cost_per_million"),
    ("reasoning_tokens", "reasoning_tokens", None),
)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _debug(message: str) -> None:
    if _env("HERMES_LANGFUSE_DEBUG").lower() in {"1", "true", "yes", "on"}:
        logger.info("Langfuse tracing: %s", message)


@contextlib.contextmanager
def _failsafe(label: str):
    """Swallow + debug-log any exception: telemetry must never block the agent turn."""
    try:
        yield
    except Exception as exc:  # pragma: no cover - fail-open
        _debug(f"{label} failed: {exc}")


_CAPTURE_MODES = ("metadata", "sanitized", "full")
_DEFAULT_CAPTURE_MODE = "sanitized"
_warned_invalid_capture = False


def _capture_mode() -> str:
    """Resolve ``metadata | sanitized | full``; read per call so long-lived processes
    can flip modes. Invalid values warn once and fall back to the default (never
    capture more than the operator intended)."""
    global _warned_invalid_capture
    value = _env("HERMES_LANGFUSE_CAPTURE").lower()
    if not value or value in _CAPTURE_MODES:
        return value or _DEFAULT_CAPTURE_MODE
    if not _warned_invalid_capture:
        _warned_invalid_capture = True
        logger.warning(
            "Langfuse plugin: invalid HERMES_LANGFUSE_CAPTURE=%r, falling back "
            "to %r (valid: %s)",
            value, _DEFAULT_CAPTURE_MODE, ", ".join(_CAPTURE_MODES),
        )
    return _DEFAULT_CAPTURE_MODE


_TELEMETRY_SCHEMA_VERSION = "hermes.telemetry.v1"
_MASK_VERSION = "hermes-mask.v1"
_SAMPLE_POLICY_VERSION = "hermes-content-sample.v1"
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d .()\-]{7,}\d)(?!\w)")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _fingerprint(value: Any) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _hermes_release() -> Optional[str]:
    configured = _env("HERMES_LANGFUSE_RELEASE") or _env("LANGFUSE_RELEASE")
    if configured:
        return configured
    try:
        return importlib.metadata.version("hermes-agent")
    except importlib.metadata.PackageNotFoundError:
        return None


def _pseudonymize_identifier(kind: str, value: Any) -> Optional[str]:
    """Return a keyed, domain-separated identifier or omit it fail-closed."""
    if value is None or value == "":
        return None
    key = _env("HERMES_LANGFUSE_PSEUDONYM_KEY")
    if not key or len(key.encode("utf-8")) < 16:
        return None
    digest = hmac.new(
        key.encode("utf-8"),
        f"{kind}:{value}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"hmac-sha256:{digest}"


def _text_chars(value: Any) -> int:
    """Count actual text characters without estimating tokens or media bytes."""
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return sum(_text_chars(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_text_chars(item) for item in value)
    return 0


def _message_content_chars(messages: Any) -> int:
    """Count message payload text without counting structural keys/roles."""
    if not isinstance(messages, list):
        return _text_chars(messages)
    total = 0
    for message in messages:
        if isinstance(message, dict):
            total += _text_chars(message.get("content"))
        else:
            total += _text_chars(getattr(message, "content", None))
    return total


def _telemetry_metadata(
    *,
    task_id: str,
    session_id: str,
    provider: str,
    model: str,
    api_mode: str,
    messages: Any,
    system_prompt: Any,
    request: Any,
    enabled_tool_count: Any = None,
    tool_schema_bytes: Any = None,
    tool_policy_fingerprint: Any = None,
) -> Dict[str, Any]:
    """Build the stable, content-free Hermes telemetry v1 root contract."""
    metadata: Dict[str, Any] = {
        "source": "hermes",
        "telemetry_schema_version": _TELEMETRY_SCHEMA_VERSION,
        "capture_mode": _capture_mode(),
        "mask_version": _MASK_VERSION,
        "sample_policy_version": _SAMPLE_POLICY_VERSION,
        "config_fingerprint": _fingerprint(
            {"provider": provider, "model": model, "api_mode": api_mode}
        ),
    }
    release = _hermes_release()
    if release:
        metadata["hermes_release"] = release

    pseudonym_key_version = _env("HERMES_LANGFUSE_PSEUDONYM_KEY_VERSION", "v1")
    pseudonymous_session = _pseudonymize_identifier("session", session_id)
    pseudonymous_task = _pseudonymize_identifier("task", task_id)
    if pseudonymous_session or pseudonymous_task:
        metadata["pseudonym_key_version"] = pseudonym_key_version
    if pseudonymous_session:
        metadata["session_id"] = pseudonymous_session
    if pseudonymous_task:
        metadata["task_id"] = pseudonymous_task

    if system_prompt is not None:
        metadata["prompt_fingerprint"] = _fingerprint(system_prompt)

    if isinstance(enabled_tool_count, int) and not isinstance(enabled_tool_count, bool):
        metadata["enabled_tool_count"] = max(0, enabled_tool_count)
    if isinstance(tool_schema_bytes, int) and not isinstance(tool_schema_bytes, bool):
        metadata["tool_schema_bytes"] = max(0, tool_schema_bytes)
    if isinstance(tool_policy_fingerprint, str) and re.fullmatch(
        r"sha256:[0-9a-f]{64}", tool_policy_fingerprint
    ):
        metadata["tool_policy_fingerprint"] = tool_policy_fingerprint

    context_chars: Dict[str, int] = {}
    if system_prompt is not None:
        context_chars["system_prompt"] = _text_chars(system_prompt)
    if messages is not None:
        context_chars["conversation"] = _message_content_chars(messages)
    if context_chars:
        metadata["context_source_chars"] = context_chars
    return metadata


def _content_sample_rate() -> float:
    raw = (
        _env("HERMES_LANGFUSE_CONTENT_SAMPLE_RATE")
        or _env("HERMES_LANGFUSE_SAMPLE_RATE")
        or "1.0"
    )
    try:
        rate = float(raw)
    except ValueError:
        logger.warning("Invalid Langfuse content sample rate %r; using 0.0", raw)
        return 0.0
    if not 0.0 <= rate <= 1.0:
        logger.warning(
            "Langfuse content sample rate outside 0..1: %r; using 0.0", raw
        )
        return 0.0
    return rate


def _content_is_sampled(trace_id: str, rate: Optional[float] = None) -> bool:
    if _capture_mode() == "metadata":
        return False
    probability = _content_sample_rate() if rate is None else rate
    if probability <= 0:
        return False
    if probability >= 1:
        return True
    sample = int(hashlib.sha256(trace_id.encode("utf-8")).hexdigest()[:16], 16)
    return sample / float(0xFFFFFFFFFFFFFFFF) < probability


def _mask_text(value: str) -> str:
    from agent.redact import redact_sensitive_text

    masked = redact_sensitive_text(value, force=True)
    masked = _EMAIL_RE.sub("[EMAIL_REDACTED]", masked)
    return _PHONE_RE.sub("[PHONE_REDACTED]", masked)


def _mask_export_value(value: Any) -> Any:
    """Recursively mask an export attribute; exceptions intentionally escape."""
    if isinstance(value, str):
        return _mask_text(value)
    if isinstance(value, dict):
        return {key: _mask_export_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_mask_export_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_mask_export_value(item) for item in value)
    return value


def _mask_otel_spans(*, params: Any) -> Any:
    """Langfuse export-stage mask; masking failure drops the whole batch."""
    if MaskOtelSpansResult is None:
        raise RuntimeError("Langfuse MaskOtelSpansResult type unavailable")
    try:
        patches = {}
        for identifier, span in params.spans.items():
            replacements: Dict[str, Any] = {}
            for key, value in (getattr(span, "attributes", None) or {}).items():
                masked = _mask_export_value(value)
                if masked != value:
                    replacements[key] = masked
            if replacements:
                if OtelSpanPatch is None:
                    raise RuntimeError("Langfuse OtelSpanPatch type unavailable")
                patches[identifier] = OtelSpanPatch(set_attributes=replacements)
        return MaskOtelSpansResult(span_patches=patches)
    except Exception as exc:
        # The SDK's explicit drop signal is the only acceptable fallback: a
        # masking failure must never cause the original attributes to export.
        _debug(f"export masking failed; dropping span batch: {exc}")
        return MaskOtelSpansResult(drop=True)


def _legacy_mask(*, data: Any, **_: Any) -> Any:
    """Fail-closed fallback for Langfuse SDKs predating export-stage masking."""
    try:
        return _mask_export_value(data)
    except Exception:
        logger.error("Langfuse SDK masking failed; replacing content", exc_info=True)
        return "<redacted:mask-failed>"


def _token_source_buckets(usage: Any) -> Dict[str, int]:
    """Map only provider-reported token buckets; never manufacture estimates."""
    if not isinstance(usage, dict):
        return {}
    names = {
        "input": "input",
        "output": "output",
        "input_tokens": "input",
        "output_tokens": "output",
        "cache_read_tokens": "cache_read",
        "cache_write_tokens": "cache_write",
        "cache_read_input_tokens": "cache_read",
        "cache_creation_input_tokens": "cache_write",
        "reasoning_tokens": "reasoning",
    }
    buckets: Dict[str, int] = {}
    for source, target in names.items():
        value = usage.get(source)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            buckets[target] = int(value)
    return buckets


def _tool_policy_decision(middleware_trace: Any) -> Optional[str]:
    if not isinstance(middleware_trace, list):
        return None
    normalized = {
        "allow": "allowed",
        "allowed": "allowed",
        "approve": "allowed",
        "approved": "allowed",
        "deny": "denied",
        "denied": "denied",
        "block": "denied",
        "blocked": "denied",
        "prompt": "approval_required",
        "approval_required": "approval_required",
    }
    for entry in reversed(middleware_trace):
        if not isinstance(entry, dict):
            continue
        for key in ("decision", "policy", "outcome", "status"):
            value = str(entry.get(key) or "").lower()
            if value in normalized:
                return normalized[value]
    return None


def _middleware_route_reason(middleware_trace: Any) -> Optional[str]:
    """Report the existing middleware route signal without exporting free text."""
    if not isinstance(middleware_trace, list):
        return None
    for entry in reversed(middleware_trace):
        if not isinstance(entry, dict):
            continue
        if isinstance(entry.get("reason"), str) and entry["reason"]:
            return "middleware_rewrite"
    return None


def _structured_code(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return value if re.fullmatch(r"[a-z][a-z0-9_.:-]{0,63}", value) else None


def _structured_outcome_metadata(
    *,
    route_reason_code: Any = None,
    fallback_count: Any = None,
    retry_count: Any = None,
    quota_result_code: Any = None,
) -> Dict[str, Any]:
    """Accept only bounded codes/counts, never arbitrary provider text."""
    metadata: Dict[str, Any] = {}
    for key, value in (
        ("route_reason_code", route_reason_code),
        ("quota_result_code", quota_result_code),
    ):
        code = _structured_code(value)
        if code is not None:
            metadata[key] = code
    for key, value in (
        ("fallback_count", fallback_count),
        ("retry_count", retry_count),
    ):
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value <= 1_000_000
        ):
            metadata[key] = value
    return metadata


# Secret redaction in ``sanitized`` mode reuses the project-wide
# ``agent.redact.redact_sensitive_text(force=True)`` — which covers 50+ credential
# patterns, private keys, JWTs, auth headers, DB connection strings, and env
# assignments with pre-check-gated regex. The ``force=True`` flag ensures
# redaction runs even if the user has ``security.redact_secrets: false`` set —
# appropriate for an observability plugin exporting to an external service.


def _redact_secrets(value: str) -> str:
    # force=True: redact even if the user disabled security.redact_secrets —
    # this content is exported to an external service.
    try:
        return _mask_text(value)
    except Exception:
        return "[CONTENT OMITTED: MASKING FAILED]"


# (types, shape builder) for _describe_content; first match wins (bool handled before).
_CONTENT_SHAPES = (
    ((int, float), lambda v: {"type": "number"}),
    (bytes, lambda v: {"type": "bytes", "length": len(v)}),
    (str, lambda v: {"type": "text", "chars": len(v)}),
    (dict, lambda v: {"type": "object", "keys": [str(k) for k in list(v.keys())[:20]]}),
    ((list, tuple, set), lambda v: {"type": "array", "items": len(v)}),
)


def _describe_content(value: Any, *, include_keys: bool = True) -> Any:
    """Metadata-mode stand-in for content: shape and size, never payload."""
    if value is None or isinstance(value, bool):
        return value
    shape = next((build(value) for types, build in _CONTENT_SHAPES if isinstance(value, types)), None)
    if isinstance(value, dict) and shape is not None:
        shape["items"] = len(value)
        if not include_keys:
            shape.pop("keys", None)
    return {"omitted": True, **(shape or {"type": type(value).__name__})}


def _capture_content(
    value: Any, *, parse_json_strings: bool = False,
    tool_result_of: Optional[tuple] = None, content_sampled: bool = True,
) -> Any:
    """Apply the active capture mode to a CONTENT value.

    Only prompt/response text, tool arguments and tool results are content;
    metadata fields (provider, model, IDs, counts) stay as-is in every mode.
    ``tool_result_of=(tool_name, args)`` marks a tool result: JSON strings are
    parsed first so a read_file payload can be collapsed to a preview keyed by
    the call's ``args``.
    """
    if not content_sampled:
        if value is None:
            return {"omitted": True, "type": "null"}
        if isinstance(value, bool):
            return {"omitted": True, "type": "boolean"}
        return _describe_content(value, include_keys=False)
    if _capture_mode() == "metadata":
        return _describe_content(value)
    if tool_result_of is not None:
        tool_name, args = tool_result_of
        value = _maybe_parse_json_string(value) if isinstance(value, str) else value
        value, parse_json_strings = _normalize_payload(value, tool_name=tool_name, args=args), True
    return _safe_value(value, parse_json_strings=parse_json_strings)


# Sentinel: "_get_langfuse() has tried and failed". Tests reset by reloading
# the module; runtime callers must restart the process after fixing credentials.
_INIT_FAILED = object()


def _validate_langfuse_key(env_name: str, value: str) -> Optional[str]:
    """Log-ready error if ``value`` lacks the prefix for ``env_name``; the preview
    exposes placeholders without echoing a real secret pasted into the wrong var."""
    expected = _LANGFUSE_KEY_PREFIXES.get(env_name, "")
    if not expected or value.startswith(expected):
        return None
    preview = "<empty>" if not value else repr(value) if len(value) <= 12 else repr(value[:6] + "...")
    return f"{env_name}={preview} (expected {expected!r} prefix)"


def _get_langfuse() -> Optional[Langfuse]:
    """Cached Langfuse client, or ``None`` if the SDK/credentials are unavailable.
    The first build is serialized so racing callers can't each construct a client
    and leak the loser's HTTP connection + flush thread."""
    global _LANGFUSE_CLIENT
    # Fast path — already settled (success or _INIT_FAILED) needs no lock;
    # re-check under it since a racing thread may have finished init.
    if _LANGFUSE_CLIENT is None:
        with _LANGFUSE_CLIENT_LOCK:
            if _LANGFUSE_CLIENT is None:
                client = _build_client()
                _LANGFUSE_CLIENT = _INIT_FAILED if client is None else client
                if client is not None:
                    # atexit is LIFO: registering AFTER the SDK's constructor means our
                    # finalizer runs first, so root spans ended there still get flushed
                    # by the SDK (short-lived processes: kanban workers, chat -q, cron).
                    atexit.register(_finalize_all_traces)
    return None if _LANGFUSE_CLIENT is _INIT_FAILED else _LANGFUSE_CLIENT


def _build_client() -> Optional[Langfuse]:
    """Construct the SDK client from env, or None (with one warning) when it can't be."""
    if Langfuse is None:
        logger.warning(
            "Langfuse plugin is enabled but the langfuse SDK is unavailable; "
            "tracing is disabled. Run `hermes tools` and configure Langfuse "
            "Observability to reinstall it."
        )
        return None

    public_key, secret_key = (_env(f"HERMES_LANGFUSE_{n}") or _env(f"LANGFUSE_{n}") for n in ("PUBLIC_KEY", "SECRET_KEY"))
    if not (public_key and secret_key):
        return None

    # The SDK does not validate keys at construction; placeholder keys
    # would fail silently at flush time (#23823). Warn once here instead.
    placeholder_issues = [issue for issue in (
        _validate_langfuse_key("HERMES_LANGFUSE_PUBLIC_KEY", public_key),
        _validate_langfuse_key("HERMES_LANGFUSE_SECRET_KEY", secret_key),
    ) if issue]
    if placeholder_issues:
        logger.warning(
            "Langfuse plugin: credentials look like placeholders, traces will "
            "NOT be emitted (%s). Set real Langfuse keys (pk-lf-... / sk-lf-...) "
            "or unset HERMES_LANGFUSE_PUBLIC_KEY / HERMES_LANGFUSE_SECRET_KEY to "
            "silence this warning.",
            "; ".join(placeholder_issues),
        )
        return None

    kwargs: Dict[str, Any] = {"public_key": public_key, "secret_key": secret_key}
    for key, name, default in (("base_url", "BASE_URL", "https://cloud.langfuse.com"), ("environment", "ENV", ""),
                               ("release", "RELEASE", "")):
        value = _env(f"HERMES_LANGFUSE_{name}") or _env(f"LANGFUSE_{name}") or default
        if value:
            kwargs[key] = value
    # Retain operational spans and sample only content. Mandatory SDK masking is
    # still applied in full-capture mode.
    if MaskOtelSpansResult is not None and OtelSpanPatch is not None:
        kwargs["mask_otel_spans"] = _mask_otel_spans
    else:
        kwargs["mask"] = _legacy_mask

    try:
        return Langfuse(**kwargs)
    except Exception as exc:  # pragma: no cover - fail-open
        logger.warning("Could not initialize Langfuse client: %s", exc)
        return None


def _trace_key(task_id: str, session_id: str, *, turn_id: str = "", api_request_id: str = "") -> str:
    """In-process trace scope key for one agent turn. ``turn_id`` wins over
    ``api_request_id`` so the turn-level post_llm_call hook (no api_request_id)
    resolves to the same key as request-level hooks; a bare ``task_id`` is the
    legacy shape from before turn/request scoping."""
    scope = f"task:{task_id}" if task_id else f"session:{session_id}" if session_id else f"thread:{threading.get_ident()}"
    if turn_id:
        return f"{scope}:turn:{turn_id}"
    if api_request_id:
        return f"{scope}:api:{api_request_id}"
    return task_id or scope


def _state_for_turn(turn_id: str) -> Optional[TraceState]:
    """Live trace state for a turn id alone (caller holds ``_STATE_LOCK``). Subagent
    hooks carry ``parent_turn_id`` but no ``task_id``, so rebuilding the key would
    miss; match on the unique ``:turn:<id>`` suffix instead."""
    if not turn_id:
        return None
    suffix = f":turn:{turn_id}"
    return next((state for key, state in _TRACE_STATE.items() if key.endswith(suffix)), None)


def _truncate_text(value: str, max_chars: int) -> Any:
    # The SDK decodes data:*;base64 strings as media; a truncated one is
    # invalid base64 and logs noisily, so redact the whole URI instead.
    prefix = value[:200].lower()
    if prefix.startswith("data:") and ";base64," in prefix:
        header = value.split(",", 1)[0] if "," in value else "data:"
        media_type = header[5:].split(";", 1)[0] if header.startswith("data:") else ""
        return {"type": "data_uri", "media_type": media_type or None, "omitted": True, "length": len(value)}
    # Redact BEFORE truncating so a secret straddling the cut cannot leak.
    if _capture_mode() == "sanitized":
        value = _redact_secrets(value)
    over = len(value) - max_chars
    return value if over <= 0 else value[:max_chars] + f"... [truncated {over} chars]"


def _maybe_parse_json_string(value: str) -> Any:
    stripped = value.strip()
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
        return {**parsed, (hint_key if hint_key not in parsed else "_trailing_text"): trailing}
    return {"data": parsed, hint_key: trailing}


def _normalize_payload(value: Any, *, tool_name: str = "", args: Any = None) -> Any:
    """Collapse a read_file result (line-numbered content + file metadata) into a compact preview."""
    is_read_file = (
        isinstance(value, dict)
        and isinstance(value.get("content"), str)
        and all(k in value for k in ("total_lines", "file_size", "is_binary", "is_image"))
        and not value.get("error")
    )
    if not is_read_file:
        return value
    normalized: dict[str, Any] = {}
    if tool_name == "read_file" and isinstance(args, dict):
        if isinstance(args.get("path"), str) and args["path"]:
            normalized["path"] = args["path"]
        normalized.update({key: args[key] for key in ("offset", "limit") if isinstance(args.get(key), int)})

    content = value.get("content", "")
    matches = [_READ_FILE_LINE_RE.match(raw) for raw in content.splitlines()] if isinstance(content, str) and content else []
    lines = [{"line": int(m.group(1)), "text": m.group(2)} for m in matches] if matches and all(matches) else []
    if lines:
        normalized["returned_lines"] = {"start": lines[0]["line"], "end": lines[-1]["line"], "count": len(lines)}
        head, tail = _READ_FILE_HEAD_LINES, _READ_FILE_TAIL_LINES
        normalized["content_preview"] = {"lines": lines} if len(lines) <= head + tail else {
            "head": lines[:head], "tail": lines[-tail:], "omitted_line_count": len(lines) - head - tail,
        }
    elif value.get("content"):
        normalized["content_preview"] = {"text": value.get("content", "")}

    normalized.update({key: value[key] for key in _READ_FILE_META_KEYS if key in value})

    b64 = value.get("base64_content")
    if isinstance(b64, str) and b64:
        normalized["base64_content"] = {"omitted": True, "length": len(b64)}
    return normalized


def _safe_value(value: Any, *, max_chars: Optional[int] = None, depth: int = 0,
                parse_json_strings: bool = False) -> Any:
    max_chars = max_chars if max_chars is not None else int(_env("HERMES_LANGFUSE_MAX_CHARS", "12000") or "12000")
    if depth > 4:
        return "<max-depth>"
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "len": len(value)}
    recurse = lambda v, d: _safe_value(v, max_chars=max_chars, depth=d, parse_json_strings=parse_json_strings)  # noqa: E731
    if isinstance(value, str):
        parsed = _maybe_parse_json_string(value) if parse_json_strings else value
        return recurse(parsed, depth) if parsed is not value else _truncate_text(value, max_chars)
    if isinstance(value, dict):
        normalized = _normalize_payload(value)
        if normalized is not value:
            return recurse(normalized, depth)
        return {str(k): recurse(v, depth + 1) for k, v in list(value.items())[:50]}
    if isinstance(value, (list, tuple, set)):
        return [recurse(v, depth + 1) for v in list(value)[:50]]
    if hasattr(value, "__dict__"):
        return recurse(vars(value), depth + 1)
    return _truncate_text(repr(value), max_chars)


def _coerce_request_messages(*, request_messages: Any = None, messages: Any = None,
                             conversation_history: Any = None, user_message: Any = None) -> list[dict[str, Any]]:
    for candidate in (request_messages, messages, conversation_history):
        if isinstance(candidate, list):
            return candidate
    return [] if user_message is None else [{"role": "user", "content": user_message}]


def _serialize_system_prompt(system_prompt: Any) -> Optional[dict[str, Any]]:
    """Normalize Anthropic/Bedrock ``system`` param or OpenAI-style system content."""
    if isinstance(system_prompt, str):
        text = system_prompt.strip()
    elif isinstance(system_prompt, list):
        # Anthropic: {"type": "text", "text": ...}; Bedrock Converse: {"text": ...}; or bare strings.
        blocks = ((b.get("text", "") if b.get("type") in ("text", None) and "text" in b else None)
                  if isinstance(b, dict) else b for b in system_prompt)
        text = "\n\n".join(b for b in blocks if isinstance(b, str) and b)
    else:
        return None
    return {"role": "system", "content": _capture_content(text)} if text else None


def _messages_for_langfuse_input(*, request_messages: Any = None, messages: Any = None,
                                 conversation_history: Any = None, user_message: Any = None,
                                 system_prompt: Any = None) -> list[dict[str, Any]]:
    """Generation input, prepending ``system_prompt`` when the provider split it out of messages."""
    raw = _coerce_request_messages(request_messages=request_messages, messages=messages,
                                   conversation_history=conversation_history, user_message=user_message)
    system_msg = None if raw and raw[0].get("role") == "system" else _serialize_system_prompt(system_prompt)
    serialized = _serialize_messages(raw)
    return serialized if system_msg is None else [system_msg, *serialized]


def _serialize_message(message: dict[str, Any]) -> dict[str, Any]:
    role, is_tool = message.get("role"), message.get("role") == "tool"
    return {
        "role": role, "content": _capture_content(message.get("content"), parse_json_strings=is_tool),
        **({"tool_call_id": message["tool_call_id"]} if is_tool and message.get("tool_call_id") else {}),
        **({"name": _safe_value(message["name"])} if is_tool and message.get("name") else {}),
        **({"tool_calls": _capture_content(message["tool_calls"], parse_json_strings=True)} if message.get("tool_calls") else {}),
    }


def _serialize_messages(messages: Any) -> list[dict[str, Any]]:
    return [_serialize_message(m) for m in messages[-12:] if isinstance(m, dict)] if isinstance(messages, list) else []


def _serialize_tool_call(tool_call: Any) -> dict[str, Any]:
    fn = getattr(tool_call, "function", None)
    name, safe_arguments = getattr(fn, "name", None), _capture_content(getattr(fn, "arguments", None))
    return {"id": getattr(tool_call, "id", None), "type": getattr(tool_call, "type", None) or "function",
            "name": name, "arguments": safe_arguments, "function": {"name": name, "arguments": safe_arguments}}


def _serialize_assistant_message(message: Any) -> dict[str, Any]:
    reasoning = next((getattr(message, attr, None) for attr in ("reasoning", "reasoning_content", "reasoning_details")
                      if getattr(message, attr, None) is not None), None)
    return {
        "content": _capture_content(getattr(message, "content", None)),
        "reasoning": None if reasoning is None else _capture_content(reasoning),
        "tool_calls": [_serialize_tool_call(tc) for tc in getattr(message, "tool_calls", None) or ()],
    }


def _canonical_usage_and_cost(canonical: Any, *, provider: str, model: str,
                              base_url: str) -> tuple[dict[str, int], dict[str, float]]:
    """Translate canonical Hermes usage into Langfuse usage and cost maps."""
    usage_details: Dict[str, int] = {
        key: tokens for key, attr, _ in _USAGE_FIELDS
        if (tokens := getattr(canonical, attr)) or key in ("input", "output")
    }
    cost_details: Dict[str, float] = {}
    try:
        from agent.usage_pricing import estimate_usage_cost, resolve_billing_route

        # Subscription-included routes: Langfuse treats explicit cost_details
        # (even zeros) as authoritative, so omit them and let it estimate.
        route = resolve_billing_route(model, provider=provider, base_url=base_url)
        if getattr(route, "billing_mode", "") == "subscription_included":
            return usage_details, cost_details
        cost = estimate_usage_cost(model, canonical, provider=provider, base_url=base_url, api_key="")
    except Exception as exc:  # pragma: no cover - fail-open
        _debug(f"usage pricing failed: {exc}")
        return usage_details, cost_details

    # No total (e.g. cache pricing unknown) => export no costs at all, so a
    # partial component subtotal is never mistaken for the request total.
    if cost.amount_usd is None:
        return usage_details, cost_details

    # Langfuse only derives totals from input/output keys, so cache/custom keys
    # need an explicit total (Hermes estimate also includes request pricing).
    # A zero total is not exported: Langfuse would treat it as authoritative.
    if cost.status != "included" and float(cost.amount_usd) > 0:
        cost_details["total"] = float(cost.amount_usd)

    # Per-type breakdown for dashboards; keys mirror usage_details.
    try:
        from decimal import Decimal

        from agent.usage_pricing import get_pricing_entry

        entry = get_pricing_entry(model, provider=provider, base_url=base_url)
        for key, attr, rate_attr in _USAGE_FIELDS if entry else ():
            rate = getattr(entry, rate_attr, None) if rate_attr else None
            tokens = getattr(canonical, attr)
            if rate is not None and tokens:
                cost_details[key] = float(Decimal(tokens) * rate / Decimal("1000000"))
    except Exception:  # pragma: no cover - canonical total remains usable
        pass

    return usage_details, cost_details


def _usage_and_cost(response: Any, *, provider: str, model: str, base_url: str, api_mode: str = "",
                    usage: Optional[dict] = None) -> tuple[dict[str, int], dict[str, float]]:
    """Langfuse usage/cost maps from ``response.usage`` (post_llm_call) or, when ``usage``
    is given (post_api_request), from that pre-built CanonicalUsage summary dict."""
    raw_usage = getattr(response, "usage", None)
    if usage is None and not raw_usage:
        return {}, {}
    try:
        from agent.usage_pricing import CanonicalUsage, normalize_usage

        canonical = normalize_usage(raw_usage, provider=provider, api_mode=api_mode) if usage is None else CanonicalUsage(
            output_tokens=usage.get("output_tokens", 0) or usage.get("completion_tokens", 0),
            request_count=usage.get("request_count", 1),
            **{attr: usage.get(attr, 0) for attr in ("input_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens")},
        )
        return _canonical_usage_and_cost(canonical, provider=provider, model=model, base_url=base_url)
    except Exception as exc:  # pragma: no cover - fail-open
        if usage is None:
            _debug(f"usage normalization failed: {exc}")
        return {}, {}


def _start_root_trace(
    task_key: str, *, task_id: str, session_id: str, platform: str,
    provider: str, model: str, api_mode: str, messages: Any, client: Any,
    turn_id: str = "", api_request_id: str = "", system_prompt: Any = None,
    request: Any = None, route_reason_code: Any = None,
    fallback_count: Any = None, retry_count: Any = None,
    quota_result_code: Any = None, enabled_tool_count: Any = None,
    tool_schema_bytes: Any = None, tool_policy_fingerprint: Any = None,
) -> TraceState:
    raw_trace_seed = (
        f"{session_id or 'sessionless'}::"
        f"{task_id or task_key}::{turn_id or api_request_id or task_key}"
    )
    trace_seed = _pseudonymize_identifier("trace", raw_trace_seed)
    if trace_seed is None:
        trace_seed = f"random-v1:{secrets.token_hex(32)}"
    trace_id = client.create_trace_id(seed=trace_seed)
    content_sampled = _content_is_sampled(trace_id)
    last_user = next(
        (m for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "user"),
        None,
    ) if isinstance(messages, list) else None
    raw_trace_input = None if last_user is None else {
        "role": "user", "content": last_user.get("content"),
    }
    trace_input = _capture_content(raw_trace_input, content_sampled=content_sampled)
    metadata = _telemetry_metadata(
        task_id=task_id, session_id=session_id, provider=provider, model=model,
        api_mode=api_mode, messages=messages, system_prompt=system_prompt,
        request=request, enabled_tool_count=enabled_tool_count,
        tool_schema_bytes=tool_schema_bytes,
        tool_policy_fingerprint=tool_policy_fingerprint,
    )
    metadata.update({
        "platform": platform, "provider": provider, "model": model,
        "api_mode": api_mode, "sample_probability": _content_sample_rate(),
        "content_sampled": content_sampled,
    })
    pseudonymous_turn = _pseudonymize_identifier("turn", turn_id)
    pseudonymous_request = _pseudonymize_identifier("api_request", api_request_id)
    if pseudonymous_turn:
        metadata["turn_id"] = pseudonymous_turn
    if pseudonymous_request:
        metadata["api_request_id"] = pseudonymous_request
    metadata.update(_structured_outcome_metadata(
        route_reason_code=route_reason_code, fallback_count=fallback_count,
        retry_count=retry_count, quota_result_code=quota_result_code,
    ))

    pseudonymous_session = _pseudonymize_identifier("session", session_id)
    trace_ctx: Dict[str, Any] = {"trace_id": trace_id}
    if pseudonymous_session:
        trace_ctx["session_id"] = pseudonymous_session

    def open_root():
        ctx = client.start_as_current_observation(
            trace_context=trace_ctx, name="Hermes turn", as_type="chain",
            input=trace_input, metadata=metadata, end_on_exit=False,
        )
        return ctx, ctx.__enter__()

    root_ctx = root_span = None
    if propagate_attributes is not None:
        try:
            with propagate_attributes(
                session_id=pseudonymous_session, trace_name="Hermes turn",
                tags=["hermes", "langfuse"],
            ):
                root_ctx, root_span = open_root()
        except Exception:
            root_ctx = None
    if root_ctx is None:
        root_ctx, root_span = open_root()

    with _failsafe("update_trace(input)"):
        root_span.update_trace(input=trace_input)
    if turn_id:
        with _failsafe("turn correlation publish"):
            from hermes_cli.lifecycle import publish_turn_correlation
            observation_id = getattr(root_span, "id", None)
            publish_turn_correlation(
                turn_id=turn_id, trace_id=trace_id,
                observation_id=str(observation_id) if observation_id else None,
            )
    _debug(f"started trace {trace_id} for {task_key}")
    return TraceState(
        trace_id=trace_id, root_ctx=root_ctx, root_span=root_span,
        content_sampled=content_sampled,
    )


def _start_child_observation(state: TraceState, *, name: str, as_type: str, input_value: Any,
                             metadata: Optional[dict] = None, model: Optional[str] = None,
                             model_parameters: Optional[dict] = None) -> Any:
    return state.root_span.start_observation(name=name, as_type=as_type, input=input_value, metadata=metadata or {},
                                             model=model, model_parameters=model_parameters)


def _end_observation(observation: Any, *, output: Any = None, metadata: Optional[dict] = None,
                     usage_details: Optional[dict] = None, cost_details: Optional[dict] = None) -> None:
    if observation is None:
        return
    with _failsafe("end observation"):
        update_kwargs = {**({} if output is None else {"output": output}),
                         **{k: v for k, v in (("metadata", metadata), ("usage_details", usage_details),
                                              ("cost_details", cost_details)) if v}}
        if update_kwargs:
            observation.update(**update_kwargs)
        observation.end()


def _end_children(state: TraceState, *, include_subagents: bool = False) -> None:
    pending = [obs for queue in state.pending_tools_by_name.values() for obs in queue]
    subagents = state.subagents.values() if include_subagents else ()
    for observation in (*state.generations.values(), *state.tools.values(), *pending, *subagents):
        _end_observation(observation)


def _end_root(state: TraceState, label: str) -> None:
    """End the root span then unwind its context; never raises."""
    with _failsafe(label):
        state.root_span.end()
        # Unwind the root context manager now, while opentelemetry.trace.Span is
        # still a real type; GC-driven close at interpreter teardown raises
        # TypeError inside use_span's isinstance check.
        if state.root_ctx is not None:
            state.root_ctx.__exit__(None, None, None)


def _finalize_all_traces() -> None:
    """atexit: end every open root span. Short-lived processes (kanban workers,
    ``chat -q``, cron) exit with tool calls queued; children export via the SDK
    flush but an un-ended root leaves an anonymous trace. Registered after the
    client is built so (LIFO) it runs before the SDK's shutdown hook."""
    with _STATE_LOCK:
        states = list(_TRACE_STATE.items())
        _TRACE_STATE.clear()
    for key, state in states:
        with _failsafe(f"atexit finalize for {key}"):  # _end_root never raises
            _end_children(state, include_subagents=True)
            _end_root(state, f"atexit finalize for {key}")
    if states:
        _flush(_get_langfuse())


def _flush(client: Any) -> None:
    if client is not None:
        with contextlib.suppress(Exception):
            client.flush()


def _merge_trace_output(output: Any, state: TraceState) -> Any:
    """Attach tool calls only when the trace's content sample is enabled."""
    if not state.content_sampled or not state.turn_tool_calls:
        return output
    merged = dict(output) if isinstance(output, dict) else {"content": output}
    merged["tool_calls"] = list(state.turn_tool_calls)
    return merged


def _finish_trace(task_key: str, *, output: Any = None) -> None:
    client = _get_langfuse()
    with _STATE_LOCK:
        state = _TRACE_STATE.pop(task_key, None) if client is not None else None
    if state is None:
        return

    try:
        _end_children(state)
        final_output = _merge_trace_output(output, state)
        if final_output is not None:
            # update_trace sets TRACE-level I/O (SDK v3); root I/O via update().
            # Neither may prevent end(), else children export without a root.
            for method, label in (("update_trace", "update_trace(output)"), ("update", "root update(output)")):
                with _failsafe(label):
                    getattr(state.root_span, method)(output=final_output)
        _end_root(state, "root end()")
    except Exception as exc:  # pragma: no cover - fail-open
        _debug(f"finish trace failed: {exc}")
        with contextlib.suppress(Exception):  # last-chance end so the root still exports
            state.root_span.end()
    finally:
        _flush(client)


def _request_key(api_call_count: Any) -> str:
    return str(api_call_count or 0)


def _client_and_key(task_id: str, session_id: str, turn_id: str, api_request_id: str) -> tuple[Any, str]:
    """(client, trace key) for a hook; client is None when tracing is unavailable."""
    client = _get_langfuse()
    if client is None:
        return None, ""
    return client, _trace_key(task_id, session_id, turn_id=turn_id, api_request_id=api_request_id)


def _duration_meta(api_duration: Any) -> Dict[str, Any]:
    return {"api_duration_s": round(api_duration, 3)} if api_duration and api_duration > 0 else {}


def _pop_generation(task_key: str, api_call_count: Any) -> tuple[Optional[TraceState], Any]:
    """Detach the open generation for one API call. Returns (state, generation); either may be None."""
    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        return state, state.generations.pop(_request_key(api_call_count), None) if state else None


def _get_or_start_state_locked(task_key: str, **root_kwargs: Any) -> TraceState:
    """Caller must hold ``_STATE_LOCK``. Starts a root trace if the key is new, first
    evicting least-recently-updated state down to ``_MAX_TRACE_STATE - 1`` (evicted
    roots are ended so they don't dangle on the Langfuse side)."""
    state = _TRACE_STATE.get(task_key)
    if state is None:
        state = _start_root_trace(task_key, **root_kwargs)
        over = len(_TRACE_STATE) - (_MAX_TRACE_STATE - 1)
        for key, stale in sorted(_TRACE_STATE.items(), key=lambda kv: kv[1].last_updated_at)[:max(over, 0)]:
            _TRACE_STATE.pop(key, None)
            _end_root(stale, "evict stale trace")
        _TRACE_STATE[task_key] = state
    state.last_updated_at = time.time()
    return state


def on_pre_llm_call(*, task_id: str = "", session_id: str = "", platform: str = "", model: str = "",
                    provider: str = "", api_mode: str = "", messages: Any = None,
                    turn_id: str = "", api_request_id: str = "", **_: Any) -> None:
    # Only legacy request-shaped calls carry an API ``messages`` list; the
    # turn-scoped pre_llm_call would otherwise open an orphan root trace.
    if not isinstance(messages, list):
        return
    client, task_key = _client_and_key(task_id, session_id, turn_id, api_request_id)
    if client is None:
        return
    with _STATE_LOCK:
        _get_or_start_state_locked(task_key, task_id=task_id, session_id=session_id, platform=platform, provider=provider, model=model,
                                   api_mode=api_mode, messages=messages, client=client, turn_id=turn_id, api_request_id=api_request_id)


def _emit_moa_reference_generations(state: TraceState, *, client: Langfuse, references: Any) -> None:
    """Record each MoA advisor as its own generation: advisors routinely run on a
    different provider/model, so otherwise the fan-out would collapse into one
    generation priced at the aggregator's rate."""
    if not isinstance(references, list) or not references:
        return
    fingerprint = json.dumps(
        [[r.get("label"), r.get("model"), (r.get("usage") or {}).get("output_tokens")]
         for r in references if isinstance(r, dict)],
        sort_keys=True, default=str,
    )
    with _STATE_LOCK:
        if fingerprint in state.moa_emitted:
            return
        state.moa_emitted.add(fingerprint)

    for ref in references:
        if not isinstance(ref, dict):
            continue
        usage = ref.get("usage") or {}
        usage_details = {key: usage[attr] for key, attr, _ in _USAGE_FIELDS if usage.get(attr)} if isinstance(usage, dict) else {}
        cost_usd = ref.get("cost_usd")
        cost_details = {"total": float(cost_usd)} if isinstance(cost_usd, (int, float)) else {}

        label = ref.get("label") or "advisor"
        metadata = {"moa_role": "reference", "label": label,
                    **{k: ref[k] for k in ("provider", "cost_status", "cost_source", "temperature") if ref.get(k) is not None}}

        observation = _start_child_observation(state, name=f"MoA advisor: {label}", as_type="generation", input_value=None,
                                               metadata=metadata, model=ref.get("model"))
        _end_observation(observation, output=_capture_content(
            ref.get("output"), content_sampled=state.content_sampled,
        ), usage_details=usage_details,
                         cost_details=cost_details, metadata=metadata)


def on_pre_llm_request(*, task_id: str = "", session_id: str = "", platform: str = "", model: str = "",
                       provider: str = "", base_url: str = "", api_mode: str = "", api_call_count: int = 0,
                       request_messages: Any = None, messages: Any = None, message_count: int = 0,
                       approx_input_tokens: int = 0, conversation_history: Any = None,
                       user_message: Any = None, turn_id: str = "", api_request_id: str = "",
                       request: Any = None, system_prompt: Any = None,
                       enabled_tool_count: Any = None, route_reason_code: Any = None,
                       fallback_count: Any = None, retry_count: Any = None,
                       quota_result_code: Any = None, middleware_trace: Any = None,
                       tool_schema_bytes: Any = None,
                       tool_policy_fingerprint: Any = None, **_: Any) -> None:
    client, task_key = _client_and_key(task_id, session_id, turn_id, api_request_id)
    if client is None:
        return

    # The request body carries the model actually dispatched (mid-session
    # switch, fallback, middleware rewrite) — prefer it over the agent attribute.
    body_model = request["body"].get("model") if isinstance(request, dict) and isinstance(request.get("body"), dict) else None
    if isinstance(body_model, str) and body_model:
        model = body_model
    if route_reason_code is None:
        route_reason_code = _middleware_route_reason(middleware_trace)

    input_messages = _coerce_request_messages(request_messages=request_messages, messages=messages,
                                              conversation_history=conversation_history, user_message=user_message)
    langfuse_input = _messages_for_langfuse_input(request_messages=input_messages, system_prompt=system_prompt)
    has_system = bool(langfuse_input) and langfuse_input[0].get("role") == "system"
    system_chars = len(str(langfuse_input[0].get("content") or "")) if has_system else 0
    req_key = _request_key(api_call_count)

    with _STATE_LOCK:
        state = _get_or_start_state_locked(
            task_key, task_id=task_id, session_id=session_id, platform=platform, provider=provider, model=model,
            api_mode=api_mode, messages=input_messages, client=client, turn_id=turn_id,
            api_request_id=api_request_id, system_prompt=system_prompt, request=request,
            route_reason_code=route_reason_code, fallback_count=fallback_count,
            retry_count=retry_count, quota_result_code=quota_result_code,
            enabled_tool_count=enabled_tool_count, tool_schema_bytes=tool_schema_bytes,
            tool_policy_fingerprint=tool_policy_fingerprint)
        previous = state.generations.pop(req_key, None)
        if previous is not None:
            _end_observation(previous)
        gen_metadata = {
            "provider": provider, "platform": platform, "api_mode": api_mode, "base_url": base_url,
            "message_count": message_count, "approx_input_tokens": approx_input_tokens,
            **({"system_prompt_chars": system_chars} if system_chars else {}),
        }
        gen_metadata.update(_structured_outcome_metadata(
            route_reason_code=route_reason_code, fallback_count=fallback_count,
            retry_count=retry_count, quota_result_code=quota_result_code,
        ))
        state.generations[req_key] = _start_child_observation(
            state, name=f"LLM call {api_call_count}", as_type="generation",
            input_value=_capture_content(
                langfuse_input, content_sampled=state.content_sampled,
            ), metadata=gen_metadata, model=model,
            model_parameters={"api_mode": api_mode, "provider": provider},
        )


def on_post_llm_call(*, task_id: str = "", session_id: str = "", provider: str = "", base_url: str = "",
                     api_mode: str = "", model: str = "", api_call_count: int = 0, assistant_message: Any = None,
                     response: Any = None, api_duration: float = 0.0, finish_reason: str = "", usage: Any = None,
                     assistant_content_chars: int = 0, assistant_tool_call_count: int = 0,
                     assistant_response: Any = None, turn_id: str = "", api_request_id: str = "",
                     response_model: Any = None, moa_references: Any = None,
                     route_reason_code: Any = None, fallback_count: Any = None,
                     retry_count: Any = None, quota_result_code: Any = None, **_: Any) -> None:
    client, task_key = _client_and_key(task_id, session_id, turn_id, api_request_id)
    if client is None:
        return

    # The response echoes the model that actually served the request.
    if isinstance(response_model, str) and response_model:
        model = response_model

    state, generation = _pop_generation(task_key, api_call_count)
    if state is None or generation is None:
        return

    if moa_references:
        _emit_moa_reference_generations(state, client=client, references=moa_references)

    # Two call shapes: post_llm_call passes assistant_message / assistant_response
    # objects; post_api_request passes summary counts + a usage dict.
    if assistant_message is not None:
        output = _serialize_assistant_message(assistant_message)
    elif assistant_response is not None:
        output = {"content": _capture_content(
            assistant_response, content_sampled=state.content_sampled,
        ), "reasoning": None, "tool_calls": []}
    else:
        output = {"content": f"[{assistant_content_chars} chars]" if assistant_content_chars else None, "reasoning": None,
                  "tool_calls": [{"id": f"tc_{i}"} for i in range(assistant_tool_call_count or 0)]}

    if output.get("tool_calls") and state.content_sampled:
        state.turn_tool_calls.extend(output["tool_calls"])

    # post_api_request's ``response`` is a sanitized dict with no ``.usage``;
    # gate on the attribute so the usage-dict fallback is actually reached.
    if getattr(response, "usage", None) is not None:
        usage_details, cost_details = _usage_and_cost(response, provider=provider, api_mode=api_mode, model=model, base_url=base_url)
    elif isinstance(usage, dict) and usage:
        usage_details, cost_details = _usage_and_cost(None, provider=provider, model=model, base_url=base_url, usage=usage)
    else:
        usage_details, cost_details = {}, {}

    gen_metadata = {"tool_call_count": len(output.get("tool_calls", [])) or assistant_tool_call_count,
                    **_duration_meta(api_duration), **({"finish_reason": finish_reason} if finish_reason else {})}
    token_bucket_source = usage if isinstance(usage, dict) else usage_details
    token_buckets = _token_source_buckets(token_bucket_source)
    if token_buckets:
        gen_metadata["token_source_buckets"] = token_buckets
        if "input" in token_buckets:
            gen_metadata["context_total_tokens"] = token_buckets["input"]
    gen_metadata.update(_structured_outcome_metadata(
        route_reason_code=route_reason_code, fallback_count=fallback_count,
        retry_count=retry_count, quota_result_code=quota_result_code,
    ))
    export_output = output if state.content_sampled else _capture_content(
        output, content_sampled=False,
    )
    _end_observation(generation, output=export_output, usage_details=usage_details,
                     cost_details=cost_details, metadata=gen_metadata)

    has_tools = bool(getattr(assistant_message, "tool_calls", None)) if assistant_message else assistant_tool_call_count > 0
    if not has_tools and output.get("content"):
        _finish_trace(task_key, output=export_output)


def on_pre_tool_call(*, tool_name: str = "", args: Any = None, task_id: str = "",
                     session_id: str = "", tool_call_id: str = "",
                     turn_id: str = "", api_request_id: str = "", **_: Any) -> None:
    client, task_key = _client_and_key(task_id, session_id, turn_id, api_request_id)
    if client is None:
        return
    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        if state is None:
            return
        observation = _start_child_observation(state, name=f"Tool: {tool_name}", as_type="tool", input_value=_capture_content(
            args, content_sampled=state.content_sampled,
        ),
                                               metadata={"tool_name": tool_name, "tool_call_id": tool_call_id})
        if tool_call_id:
            state.tools[tool_call_id] = observation
        else:
            state.pending_tools_by_name.setdefault(tool_name, []).append(observation)


def on_post_tool_call(*, tool_name: str = "", args: Any = None, result: Any = None,
                      task_id: str = "", session_id: str = "", tool_call_id: str = "",
                      turn_id: str = "", api_request_id: str = "",
                      status: Any = None, error_type: Any = None,
                      duration_ms: Any = None, middleware_trace: Any = None,
                      **_: Any) -> None:
    task_key = _trace_key(task_id, session_id, turn_id=turn_id, api_request_id=api_request_id)
    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        if state is None:
            return
        observation = state.tools.pop(tool_call_id, None) if tool_call_id else None
        queue = state.pending_tools_by_name.get(tool_name) if observation is None else None
        if queue:
            observation = queue.pop(0)
            if not queue:
                state.pending_tools_by_name.pop(tool_name, None)
    if observation is None:
        return

    safe_result_value = _capture_content(
        result, tool_result_of=(tool_name, args),
        content_sampled=state.content_sampled,
    )

    # Backfill so the generation's tool_call record carries the result alongside arguments.
    if tool_call_id:
        with _STATE_LOCK:
            state = _TRACE_STATE.get(task_key)
            calls = state.turn_tool_calls if state is not None else []
            tool_call = next((tc for tc in reversed(calls) if tc.get("id") == tool_call_id), None)
            for target in (tool_call, tool_call.get("function")) if tool_call is not None else ():
                if isinstance(target, dict):
                    target["output"] = safe_result_value

    tool_metadata: Dict[str, Any] = {"tool_name": tool_name}
    if tool_call_id:
        tool_metadata["tool_call_id"] = tool_call_id
    if status is not None:
        tool_metadata["outcome_code"] = str(status)
    if error_type:
        tool_metadata["error_type"] = str(error_type)
    if isinstance(duration_ms, (int, float)) and not isinstance(duration_ms, bool):
        tool_metadata["duration_ms"] = duration_ms
    policy_decision = _tool_policy_decision(middleware_trace)
    if policy_decision:
        tool_metadata["tool_policy_decision"] = policy_decision
    _end_observation(observation, output=safe_result_value, metadata=tool_metadata)


def on_api_request_error(*, task_id: str = "", session_id: str = "", api_call_count: int = 0,
                         api_duration: float = 0.0, status_code: Any = None, retry_count: Any = None,
                         max_retries: Any = None, retryable: Any = None, reason: Any = None, error: Any = None,
                         turn_id: str = "", api_request_id: str = "", **_: Any) -> None:
    """Close (as ERROR) the open generation for a failed API request so the turn
    doesn't look hung until eviction; a non-retryable failure also finishes the
    turn, since the agent loop is about to unwind."""
    client, task_key = _client_and_key(task_id, session_id, turn_id, api_request_id)
    if client is None:
        return
    state, generation = _pop_generation(task_key, api_call_count)
    if state is None:
        return

    error = error if isinstance(error, dict) else {}
    error_type, error_message = str(error.get("type") or ""), str(error.get("message") or "")

    # Error messages can embed request fragments (URLs w/ keys, prompt echoes) — capture-pipeline them.
    error_metadata: Dict[str, Any] = {
        "error": True, "error_type": error_type,
        "error_message": _capture_content(
            error_message, content_sampled=state.content_sampled,
        ), **_duration_meta(api_duration),
    }
    if isinstance(status_code, int) and not isinstance(status_code, bool) and 0 <= status_code <= 999:
        error_metadata["status_code"] = status_code
    error_metadata.update(_structured_outcome_metadata(retry_count=retry_count))
    if isinstance(max_retries, int) and not isinstance(max_retries, bool) and 0 <= max_retries <= 1_000_000:
        error_metadata["max_retries"] = max_retries
    if isinstance(retryable, bool):
        error_metadata["retryable"] = retryable
    reason_code = _structured_code(reason)
    if reason_code:
        error_metadata["reason_code"] = reason_code

    if generation is not None:
        with _failsafe("error-level update"):
            generation.update(level="ERROR", status_message=(error_type or "api_request_error")[:200])
        _end_observation(generation, metadata=error_metadata)

    # A retryable failure is followed by another pre_api_request on the same
    # trace; keep the turn open. A terminal failure ends the turn.
    if retryable is False:
        _finish_trace(task_key, output={"error": error_metadata})
    else:
        state.last_updated_at = time.time()


def on_session_finalize(*, session_id: str = "", reason: str = "", **_: Any) -> None:
    """Session-end boundary: close still-open traces and flush. A turn ending on a
    tool-only or empty final response never reaches ``_finish_trace``; its root
    would dangle until eviction and queued events could be lost on exit."""
    # Never lazily initialize a client here — if init never happened there are no traces.
    client = _LANGFUSE_CLIENT
    if client is None or client is _INIT_FAILED or not hasattr(client, "flush"):
        return

    # This session's traces (all, when no session_id). Keys carry the session as
    # "session:<id>" or "task:<id>" (gateway: task_id == session_id) or bare legacy id.
    fragments = (f"session:{session_id}", f"task:{session_id}")
    with _STATE_LOCK:
        keys = [k for k in _TRACE_STATE if not session_id or k == session_id or any(f in k for f in fragments)]
    for key in keys:
        _finish_trace(key)
    with _failsafe("finalize flush"):
        client.flush()

    # Shut down only at true process exit (not /new, /reset, session expiry: the
    # cached client must keep exporting). Doing it while modules are intact keeps
    # the SDK's atexit handler off torn-down opentelemetry globals (TypeError on quit).
    if reason == "shutdown" and callable(getattr(client, "shutdown", None)):
        with _failsafe("langfuse shutdown"):
            client.shutdown()


def on_subagent_start(*, parent_turn_id: str = "", parent_subagent_id: Any = None,
                      child_session_id: Any = None, child_subagent_id: Any = None,
                      child_role: str = "", child_goal: Any = None, **_: Any) -> None:
    client = _get_langfuse()
    if client is None or not child_session_id:
        return

    with _STATE_LOCK:
        state = _state_for_turn(parent_turn_id)
        if state is None:
            return
        metadata = {"child_role": child_role}
        for field, kind, value in (
            ("child_session_id", "subagent_session", child_session_id),
            ("child_subagent_id", "subagent", child_subagent_id),
            ("parent_subagent_id", "parent_subagent", parent_subagent_id),
        ):
            pseudonym = _pseudonymize_identifier(kind, value)
            if pseudonym is not None:
                metadata[field] = pseudonym
        state.subagents[str(child_session_id)] = _start_child_observation(
            state, name=f"Subagent: {child_role or 'delegate'}", as_type="span",
            input_value=_capture_content(
                child_goal, content_sampled=state.content_sampled,
            ), metadata=metadata)


def on_subagent_stop(*, parent_turn_id: str = "", child_session_id: Any = None, child_role: str = "",
                     child_summary: Any = None, child_status: Any = None,
                     tool_call_history: Any = None, duration_ms: Any = None, **_: Any) -> None:
    if not child_session_id:
        return

    with _STATE_LOCK:
        state = _state_for_turn(parent_turn_id)
        if state is None:
            return
        observation = state.subagents.pop(str(child_session_id), None)
    if observation is None:
        return

    metadata = {"child_role": child_role,
                **{k: v for k, v in (("status", child_status), ("duration_ms", duration_ms)) if v}}
    if isinstance(tool_call_history, list):
        metadata["tool_call_count"] = len(tool_call_history)
        metadata["tool_calls"] = _capture_content(
            tool_call_history, content_sampled=state.content_sampled,
        )
    _end_observation(observation, output=_capture_content(
        child_summary, content_sampled=state.content_sampled,
    ), metadata=metadata)


def register(ctx) -> None:
    # Both hook-name variants so the plugin works across Hermes versions:
    # *_api_request fire per API call (preferred); *_llm_call once per turn.
    hooks = (
        ("pre_api_request", on_pre_llm_request), ("post_api_request", on_post_llm_call),
        ("api_request_error", on_api_request_error), ("pre_llm_call", on_pre_llm_call),
        ("post_llm_call", on_post_llm_call), ("pre_tool_call", on_pre_tool_call),
        ("post_tool_call", on_post_tool_call), ("on_session_finalize", on_session_finalize),
        ("on_session_end", on_session_finalize), ("subagent_start", on_subagent_start),
        ("subagent_stop", on_subagent_stop),
    )
    for name, fn in hooks:
        ctx.register_hook(name, fn)
