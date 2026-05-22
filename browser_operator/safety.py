"""Safety helpers for Hermes Universal Browser Operator.

The browser sees untrusted web pages.  These helpers keep page text useful for
UI navigation while preventing obvious page-level prompt injections and secret
strings from entering Hermes context/results.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Iterable

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # OpenAI/Anthropic-ish API keys and many generic provider keys.
    re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_\-]{10,}\b"),
    # GitHub PAT families.
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    # Bearer tokens in copied error messages/URLs.
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/\-]{16,}={0,2}"),
    # Query/body secret-looking assignments.
    re.compile(r"(?i)(\b(?:api[_-]?key|token|secret|password|passwd|pwd|access[_-]?token|refresh[_-]?token)=)[^\s&?#]{6,}"),
)

_PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions"),
    re.compile(r"(?i)disregard\s+(?:all\s+)?(?:previous|prior|above)\s+instructions"),
    re.compile(r"(?i)reveal\s+(?:your\s+)?(?:system\s+prompt|secrets?|passwords?|1password)"),
    re.compile(r"(?i)send\s+(?:all\s+)?(?:secrets?|passwords?|tokens?)"),
    re.compile(r"(?i)exfiltrate\s+(?:secrets?|passwords?|tokens?|credentials)"),
    re.compile(r"(?i)you\s+are\s+now\s+(?:a|an)\s+"),
)


def redact_secrets(text: str) -> str:
    """Return *text* with common token/password shapes replaced.

    This is intentionally conservative and independent from Hermes' global
    redaction toggle because browser pages are untrusted input.
    """
    redacted = text
    for pattern in _SECRET_PATTERNS:
        if pattern.pattern.startswith("(?i)(\\b"):
            redacted = pattern.sub(lambda m: f"{m.group(1)}[REDACTED_SECRET]", redacted)
        else:
            redacted = pattern.sub("[REDACTED_SECRET]", redacted)
    return redacted


def _count_secret_matches(text: str) -> int:
    return sum(len(pattern.findall(text)) for pattern in _SECRET_PATTERNS)


def detect_prompt_injection(text: str) -> bool:
    """Return True when text looks like page-borne instructions to the agent."""
    return any(pattern.search(text or "") for pattern in _PROMPT_INJECTION_PATTERNS)


def _walk_and_sanitize(value: Any, stats: dict[str, Any]) -> Any:
    if isinstance(value, str):
        if detect_prompt_injection(value):
            stats["prompt_injection_detected"] = True
        count = _count_secret_matches(value)
        if count:
            stats["secrets_redacted"] += count
        return redact_secrets(value)
    if isinstance(value, dict):
        return {str(k): _walk_and_sanitize(v, stats) for k, v in value.items()}
    if isinstance(value, list):
        return [_walk_and_sanitize(v, stats) for v in value]
    if isinstance(value, tuple):
        return [_walk_and_sanitize(v, stats) for v in value]
    return value


def safety_flags(*, prompt_injection_detected: bool, secrets_redacted: int) -> list[str]:
    flags: list[str] = []
    if prompt_injection_detected:
        flags.append("prompt_injection")
    if secrets_redacted:
        flags.append("secret_redaction")
    return flags


def sanitize_browser_payload(payload: Any) -> Any:
    """Sanitize browser/tool payloads before returning them to Hermes.

    Dict payloads receive a `safety` metadata object.  Non-dict payloads are
    wrapped so callers still get safety metadata.
    """
    stats = {"prompt_injection_detected": False, "secrets_redacted": 0}
    sanitized = _walk_and_sanitize(copy.deepcopy(payload), stats)
    metadata = {
        "raw_page_content_trusted": False,
        "prompt_injection_detected": bool(stats["prompt_injection_detected"]),
        "secrets_redacted": int(stats["secrets_redacted"]),
        "flags": safety_flags(
            prompt_injection_detected=bool(stats["prompt_injection_detected"]),
            secrets_redacted=int(stats["secrets_redacted"]),
        ),
    }
    if isinstance(sanitized, dict):
        existing_value = sanitized.get("safety")
        existing: dict[str, Any] = existing_value if isinstance(existing_value, dict) else {}
        sanitized["safety"] = {**existing, **metadata}
        return sanitized
    return {"result": sanitized, "safety": metadata}


def sanitize_json_text(text: str) -> str:
    """Parse JSON if possible, sanitize it, and serialize a JSON string."""
    import json

    try:
        parsed = json.loads(text)
    except Exception:
        parsed = {"text": text}
    return json.dumps(sanitize_browser_payload(parsed), ensure_ascii=False)
