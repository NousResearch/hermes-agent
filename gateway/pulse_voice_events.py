"""Best-effort local voice-out bridge for Pulse/Aegis.

Aegis must not speak raw executor output.  The gateway publishes only short,
purpose-built voice UX events to ``$HERMES_HOME/pulse/voice-out.jsonl`` and
Aegis consumes that file/SSE stream for TTS.

This is intentionally best-effort: failures must never affect chat delivery.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

from hermes_constants import get_hermes_home
from gateway.ambient_voice_policy import AmbientVoicePolicy, VoiceContext as AmbientVoiceContext
from gateway.final_speech_summarizer import (
    FinalSpeechSummarizer,
    VoiceContext as FinalSpeechVoiceContext,
)

_LOCK = threading.Lock()
_MAX_BYTES = 2_000_000
_MAX_TEXT_CHARS = 180
_ALLOWED_KINDS = {"ack", "completion", "error", "question", "progress"}
_MEDIA_RE = re.compile(r"MEDIA:\S+")
_DIRECTIVE_RE = re.compile(r"\[\[[^\]]+\]\]")
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]{1,120})`")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
_PATH_RE = re.compile(r"(?<!\w)(?:[A-Za-z]:[\\/][^\s,;:)\]'\"]+|(?:~?/|/)[^\s,;:)\]'\"]+)")
_FILE_REF_RE = re.compile(
    r"(?<![\w/.-])(?:[\w.-]+/)*[\w.-]+\.(?:py|json|ya?ml|toml|md|txt|log|js|jsx|ts|tsx|css|html|sh|env)\b",
    re.IGNORECASE,
)
_SECRET_RE = re.compile(
    r"""
    (?:
        \b(?:bearer|authorization)\b\s*[:=]?\s+[A-Za-z0-9][A-Za-z0-9._~+/\-]{7,}
      | \b(?:api[_ -]?key|secret(?:[_ -]?key)?|password|passwd|pwd|token|access[_ -]?key|aws[_ -]?key)\b
        \s*(?:is|=|:)?\s+[A-Za-z0-9][A-Za-z0-9._~+/\-]{7,}
      | \b(?:sk-[A-Za-z0-9._-]{6,}|sk-[A-Za-z0-9]{2,}\.\.\.[A-Za-z0-9]{2,}|gh[pousr]_[A-Za-z0-9_]{10,})\b
      | \b(?:AKIA|ASIA)[A-Z0-9.]{10,}\b
      | \bhk_[A-Za-z0-9._-]{10,}\b
      | \b[a-z]{2,}_(?:test|live|prod|secret|key)_[A-Za-z0-9]{10,}\b
      | \beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
_TOOL_LOG_LINE_RE = re.compile(
    r'(?m)^\s*(?:\$ .+|>>> .+|FAILED .+|Traceback .+|File "[^"]+", line \d+.*|E\s+.+|={5,}.*)$'
)
_DEFAULT_FINAL_SUMMARY_CONFIG = {
    "mode": "hybrid",
    "timeout_ms": 1000,
    "max_spoken_chars": _MAX_TEXT_CHARS,
    "voice_profile": "eon",
    "fallback": "deterministic_sanitizer",
    "on_empty": "silence",
}
_STACK_TRACE_RE = re.compile(r"(?is)^\s*Traceback \(most recent call last\):.*")
_SENSITIVE_TOPIC_RE = re.compile(r"(?i)\b(?:trading\s+pnl|portfolio\s+exposure)\b")
_AMBIENT_POLICY = AmbientVoicePolicy()


@dataclass(frozen=True)
class SanitizedVoiceText:
    text: str
    policy: dict[str, Any]


@dataclass(frozen=True)
class FinalVoiceSummaryResult:
    kind: str
    text: str
    source: str
    derived_from: str
    voice_profile: str
    summarizer: dict[str, Any]
    policy: dict[str, Any]


def _enabled() -> bool:
    value = str(os.getenv("HERMES_PULSE_VOICE_EVENTS", "1")).strip().lower()
    return value not in {"0", "false", "no", "off"}


def voice_out_path() -> Path:
    """Return the canonical Jarvis-style voice-out JSONL path."""
    return get_hermes_home() / "pulse" / "voice-out.jsonl"


def voice_events_path() -> Path:
    """Return the legacy voice-events JSONL path.

    Kept as a compatibility mirror for older Aegis builds.  New consumers should
    read :func:`voice_out_path`.
    """
    return get_hermes_home() / "pulse" / "voice-events.jsonl"


def _default_policy() -> dict[str, Any]:
    return {
        "allowed": True,
        "sanitized": True,
        "truncated": False,
        "suppressed": False,
        "rule_profile": "living_room_default",
        "reason_codes": [],
        "classifiers": {
            "dropped_code": False,
            "dropped_tool_logs": False,
            "dropped_paths": False,
            "blocked_secret_like": False,
            "blocked_sensitive_topic": False,
            "blocked_stack_trace": False,
        },
    }


def _ambient_context(kind: str, metadata: dict[str, Any]) -> AmbientVoiceContext:
    """Build the deterministic ambient-policy context from safe event metadata."""
    return AmbientVoiceContext(
        source=str(kind or metadata.get("source") or "completion"),
        platform=metadata.get("platform"),
        channel_id=metadata.get("channel_id"),
        chat_id=metadata.get("chat_id"),
        thread_id=metadata.get("thread_id"),
        source_message_id=metadata.get("source_message_id"),
        input_modality=metadata.get("input_modality"),
        output_device=metadata.get("output_device"),
        profile=metadata.get("voice_profile") or metadata.get("profile") or "eon",
        explicit_spoken_request=bool(metadata.get("explicit_spoken_request", False)),
        is_private_context=bool(metadata.get("is_private_context", False)),
        config_scope=str(metadata.get("config_scope") or "living_room_default"),
    )


def _safe_reason_code(reason: Any) -> str | None:
    """Return a safe short reason code, never raw evidence/snippets."""
    code = str(reason or "").strip()
    if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code):
        return code
    return None


def _policy_metadata_from_decision(
    decision: Any,
    base_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert AmbientVoicePolicy decisions into the Pulse v2 safe metadata shape."""
    policy = _default_policy()
    base_reason_codes: list[Any] = []
    if base_policy:
        for key in ("allowed", "sanitized", "truncated", "suppressed", "rule_profile"):
            if key in base_policy:
                policy[key] = base_policy[key]
        base_classifiers = base_policy.get("classifiers") or {}
        classifier_aliases = {
            "dropped_code": ("dropped_code", "code"),
            "dropped_tool_logs": ("dropped_tool_logs", "command_log"),
            "dropped_paths": ("dropped_paths", "raw_path"),
            "blocked_secret_like": ("blocked_secret_like", "secret_like"),
            "blocked_sensitive_topic": ("blocked_sensitive_topic", "sensitive_topic"),
            "blocked_stack_trace": ("blocked_stack_trace", "stack_trace"),
        }
        for safe_key, aliases in classifier_aliases.items():
            policy["classifiers"][safe_key] = any(bool(base_classifiers.get(alias)) for alias in aliases)
        base_reason_codes = list(base_policy.get("reason_codes") or [])

    reason_aliases = {
        "raw_path_stripped": "path_stripped",
        "raw_path": "path_stripped",
        "empty_after_sanitization": "empty_after_sanitize",
        "secret_like": "secret_like_blocked",
        "sensitive_topic": "sensitive_topic_blocked",
        "stack_trace": "stack_trace_blocked",
    }
    reason_codes: list[str] = []
    for reason in base_reason_codes + list(getattr(decision, "reasons", ()) or ()):
        mapped = _safe_reason_code(reason_aliases.get(str(reason), str(reason)))
        if mapped and mapped not in reason_codes:
            reason_codes.append(mapped)

    classifiers = getattr(decision, "classifiers", {}) or {}
    policy.update(
        {
            "allowed": bool(getattr(decision, "allowed", policy["allowed"])),
            "sanitized": bool(policy.get("sanitized") or getattr(decision, "sanitized", False)),
            "truncated": bool(policy.get("truncated") or getattr(decision, "truncated", False)),
            "suppressed": bool(getattr(decision, "suppressed", policy["suppressed"])),
            "rule_profile": str(getattr(decision, "rule_profile", policy["rule_profile"])),
            "reason_codes": reason_codes,
        }
    )
    policy["classifiers"].update(
        {
            "dropped_code": bool(policy["classifiers"].get("dropped_code") or classifiers.get("code")),
            "dropped_tool_logs": bool(policy["classifiers"].get("dropped_tool_logs") or classifiers.get("command_log")),
            "dropped_paths": bool(policy["classifiers"].get("dropped_paths") or classifiers.get("raw_path")),
            "blocked_secret_like": bool(policy["classifiers"].get("blocked_secret_like") or classifiers.get("secret_like")),
            "blocked_sensitive_topic": bool(
                policy["classifiers"].get("blocked_sensitive_topic") or classifiers.get("sensitive_topic")
            ),
            "blocked_stack_trace": bool(policy["classifiers"].get("blocked_stack_trace") or classifiers.get("stack_trace")),
        }
    )
    return policy


def _final_summary_config() -> dict[str, Any]:
    """Load pulse.voice.final_summary with conservative defaults.

    Config loading is best-effort because voice event publication must never
    block or break gateway text delivery.
    """
    config = dict(_DEFAULT_FINAL_SUMMARY_CONFIG)
    try:
        from hermes_cli.config import load_config

        loaded = load_config() or {}
        pulse_voice = ((loaded.get("pulse") or {}).get("voice") or {}) if isinstance(loaded, dict) else {}
        user_config = pulse_voice.get("final_summary") or {}
        if isinstance(user_config, dict):
            config.update({k: v for k, v in user_config.items() if v is not None})
    except Exception:
        pass
    return config


def _trim_if_needed(path: Path) -> None:
    try:
        if not path.exists() or path.stat().st_size <= _MAX_BYTES:
            return
        data = path.read_bytes()[-(_MAX_BYTES // 2):]
        first_newline = data.find(b"\n")
        if first_newline >= 0:
            data = data[first_newline + 1 :]
        path.write_bytes(data)
    except OSError:
        return


def _first_sentence(text: str) -> str:
    match = re.match(r"^.{1,150}?[.!?。！？](?=\s|$)", text)
    return match.group(0) if match else text


def sanitize_voice_text(text: str, *, max_chars: int = _MAX_TEXT_CHARS) -> SanitizedVoiceText:
    """Return speech-safe text plus non-sensitive policy metadata."""
    original = str(text or "")
    policy = _default_policy()
    classifiers = policy["classifiers"]

    classifiers["dropped_code"] = bool(_CODE_FENCE_RE.search(original) or _INLINE_CODE_RE.search(original))
    classifiers["dropped_tool_logs"] = bool(_TOOL_LOG_LINE_RE.search(original))
    classifiers["dropped_paths"] = bool(_PATH_RE.search(original) or _FILE_REF_RE.search(original))
    classifiers["blocked_secret_like"] = bool(_SECRET_RE.search(original))
    classifiers["blocked_sensitive_topic"] = bool(_SENSITIVE_TOPIC_RE.search(original))
    classifiers["blocked_stack_trace"] = bool(_STACK_TRACE_RE.search(original))

    if classifiers["blocked_secret_like"] or classifiers["blocked_sensitive_topic"] or classifiers["blocked_stack_trace"]:
        policy["allowed"] = False
        policy["suppressed"] = True
        if classifiers["blocked_secret_like"]:
            policy["reason_codes"].append("secret_like_blocked")
        if classifiers["blocked_sensitive_topic"]:
            policy["reason_codes"].append("sensitive_topic_blocked")
        if classifiers["blocked_stack_trace"]:
            policy["reason_codes"].append("stack_trace_blocked")
        return SanitizedVoiceText("", policy)

    cleaned = _CODE_FENCE_RE.sub("", original)
    cleaned = _TOOL_LOG_LINE_RE.sub("", cleaned)
    cleaned = _MEDIA_RE.sub("", cleaned)
    cleaned = _DIRECTIVE_RE.sub("", cleaned)
    cleaned = _MARKDOWN_LINK_RE.sub(r"\1", cleaned)
    cleaned = _INLINE_CODE_RE.sub("", cleaned)
    if classifiers["dropped_paths"]:
        cleaned = _PATH_RE.sub("", cleaned)
        cleaned = _FILE_REF_RE.sub("", cleaned)
        policy["reason_codes"].append("path_stripped")
    cleaned = re.sub(r"^[#>*\-•\s]+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        policy["allowed"] = False
        policy["suppressed"] = True
        policy["reason_codes"].append("empty_after_sanitize")
        return SanitizedVoiceText("", policy)

    cleaned = _first_sentence(cleaned).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 1].rstrip(" ,.;:") + "…"
        policy["truncated"] = True
        policy["reason_codes"].append("truncated")
    policy["reason_codes"].append("bounded_to_one_sentence")
    return SanitizedVoiceText(cleaned, policy)


def voice_safe_text(text: str, *, max_chars: int = _MAX_TEXT_CHARS) -> str:
    """Return short text safe enough to speak aloud.

    This intentionally strips code blocks, media tags, markdown links, transport
    directives, bullets/headings, paths, and excess whitespace.  It then keeps
    only a single bounded sentence so executor prose/logs cannot leak into room
    audio.
    """
    return sanitize_voice_text(text, max_chars=max_chars).text


def summarize_final_voice_response(
    final_response: str,
    *,
    summarizer: Callable[[str], str] | None = None,
    max_chars: int = _MAX_TEXT_CHARS,
) -> FinalVoiceSummaryResult:
    """Derive the final spoken line and metadata from assistant final text.

    The generated path is bounded by ``pulse.voice.final_summary.timeout_ms``
    through ``FinalSpeechSummarizer``. Any timeout, exception, invalid generated
    output, or missing generated model falls back to deterministic sanitization;
    empty deterministic output stays silent.
    """
    config = _final_summary_config()
    mode = str(config.get("mode") or "hybrid").strip().lower()
    if mode not in {"deterministic", "generated", "hybrid", "off"}:
        mode = "hybrid"
    timeout_ms = int(config.get("timeout_ms") or _DEFAULT_FINAL_SUMMARY_CONFIG["timeout_ms"])
    max_spoken_chars = int(config.get("max_spoken_chars") or max_chars or _MAX_TEXT_CHARS)
    voice_profile = str(config.get("voice_profile") or "eon")

    context = FinalSpeechVoiceContext(
        max_spoken_chars=max_spoken_chars,
        timeout_ms=max(1, timeout_ms),
        voice_profile=voice_profile,
    )
    result = FinalSpeechSummarizer(generator=summarizer, mode=mode).summarize(final_response, context)
    fallback_used = result.method == "deterministic" and (
        summarizer is not None or mode in {"generated", "hybrid"}
    )
    summarizer_meta = {
        "mode": mode,
        "method": result.method,
        "fallback_used": fallback_used,
        "timeout_ms": timeout_ms,
        "validation_failed": bool(str(result.reason or "").startswith("generated_invalid")),
    }


    if result.reason:
        summarizer_meta["reason"] = result.reason

    summary_policy = sanitize_voice_text(result.text).policy if result.text else _default_policy()
    summary_policy.update(result.policy)
    summary_policy.update({
        "pre_sanitized": True,
        "post_sanitized": True,
    })
    if not result.text:
        summary_policy["allowed"] = False
        summary_policy["suppressed"] = True
    return FinalVoiceSummaryResult(
        kind=result.kind,
        text=result.text,
        source="assistant_final",
        derived_from="final_response",
        voice_profile=voice_profile,
        summarizer=summarizer_meta,
        policy=summary_policy,
    )

def completion_voice_text(
    final_response: str,
    *,
    summarizer: Callable[[str], str] | None = None,
) -> tuple[str, str]:
    """Derive a concise completion/question/error event from executor output."""
    result = summarize_final_voice_response(final_response, summarizer=summarizer)
    return result.kind, result.text


def _write_event(path: Path, event: dict[str, Any]) -> None:
    line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
    _trim_if_needed(path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)


def publish_voice_out(kind: str, text: str, **metadata: Any) -> None:
    """Append a canonical voice-out event for local Pulse/Aegis subscribers.

    ``kind`` must be one of ``ack``, ``completion``, ``error``, ``question``, or
    ``progress``.  ``text`` is sanitized and bounded here even if callers forget.
    """
    if not _enabled():
        return
    kind = str(kind or "progress").strip().lower()
    if kind not in _ALLOWED_KINDS:
        kind = "progress"
    sanitized = sanitize_voice_text(text)
    if not sanitized.text:
        return
    try:
        context = _ambient_context(kind, metadata)
        decision = _AMBIENT_POLICY.evaluate(sanitized.text, context)
        if not decision.allowed or not decision.text:
            return
        default_seconds = 2 if kind in {"ack", "progress"} else 4
        base_policy = metadata.pop("policy", None)
        if isinstance(base_policy, dict):
            merged_policy = dict(sanitized.policy)
            merged_policy.update({k: v for k, v in base_policy.items() if k not in {"reason_codes", "classifiers"}})
            merged_policy["reason_codes"] = list(base_policy.get("reason_codes") or []) + list(
                sanitized.policy.get("reason_codes") or []
            )
            merged_classifiers = dict(sanitized.policy.get("classifiers") or {})
            merged_classifiers.update(base_policy.get("classifiers") or {})
            merged_policy["classifiers"] = merged_classifiers
            policy = _policy_metadata_from_decision(decision, merged_policy)
        else:
            policy = _policy_metadata_from_decision(decision, sanitized.policy)
        event = {
            "id": f"{time.time_ns()}",
            "ts": time.time(),
            "schema_version": 2,
            "kind": kind,
            "text": decision.text,
            "max_seconds": int(metadata.pop("max_seconds", decision.max_seconds or default_seconds) or default_seconds),
            "source": metadata.pop("source", kind),
            "derived_from": metadata.pop("derived_from", "pulse_voice_candidate"),
            "voice_profile": metadata.pop("voice_profile", context.profile or "eon"),
            "policy": policy,
            **{k: v for k, v in metadata.items() if v is not None},
        }
        canonical = voice_out_path()
        legacy = voice_events_path()
        canonical.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            _write_event(canonical, event)
            # Compatibility mirror for older Aegis builds; same canonical schema,
            # not raw delta/commentary.
            if legacy != canonical:
                _write_event(legacy, event)
    except Exception:
        return


def publish_completion_voice_out(
    final_response: str,
    *,
    summarizer: Callable[[str], str] | None = None,
    **metadata: Any,
) -> None:
    """Publish a short spoken completion derived from executor final text."""
    result = summarize_final_voice_response(final_response, summarizer=summarizer)
    publish_voice_out(
        result.kind,
        result.text,
        source=result.source,
        derived_from=result.derived_from,
        voice_profile=result.voice_profile,
        summarizer=result.summarizer,
        policy=result.policy,
        **metadata,
    )


def publish_voice_event(kind: str, text: str, **metadata: Any) -> None:
    """Backward-compatible wrapper for older gateway call sites.

    Raw streaming ``delta`` events are deliberately ignored.  Interim assistant
    ``commentary`` maps to a short ``progress`` voice-out event only when the
    model actually generated that assistant text.
    """
    legacy_kind = str(kind or "").strip().lower()
    if legacy_kind == "delta":
        return
    mapped = "progress" if legacy_kind == "commentary" else legacy_kind
    publish_voice_out(mapped, text, **metadata)
