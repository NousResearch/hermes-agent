"""DLP redaction for outbound Teams messages.

Scrubs secrets / PII from every outbound surface before it reaches Teams.
Pure logic — the adapter calls :func:`redact`
at its delivery choke point (block sends, streamed replies, edits, card strings).

Built-in categories: ``email``, ``secret`` (API keys / tokens). Operators can add
``custom_patterns`` (regexes). Each hit becomes a configurable placeholder, e.g.
``[REDACTED:email]``. Because per-token streaming can't retract a secret that
completes across chunk boundaries, the adapter downgrades partial streaming to
progress when DLP is on (handled adapter-side; this module is the scrubber).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ── built-in detectors ───────────────────────────────────────────────────────

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Secrets: known-prefixed tokens (full-match redaction).
_SECRET_PATTERNS = (
    re.compile(r"\bsk-(?:proj|ant|live|test)-[A-Za-z0-9_-]{8,}\b"),  # OpenAI/Anthropic/Stripe
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),  # generic sk- key
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),  # Slack token
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),  # GitHub token
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}\b"),  # bearer header value
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{6,}\b"),  # JWT
)
# "api_key: <value>" / "password=<value>" — redact the VALUE only.
_SECRET_ASSIGN = re.compile(
    r"(?i)\b(api[_-]?key|secret|token|password|passwd|pwd)\b(\s*[:=]\s*)(\S{6,})"
)

_BUILTIN = ("email", "secret")


@dataclass(frozen=True)
class DlpConfig:
    enabled: bool = False
    categories: tuple[str, ...] = _BUILTIN
    custom_patterns: tuple[str, ...] = ()
    placeholder: str = "[REDACTED:{label}]"

    @classmethod
    def from_dict(cls, raw: dict | None) -> "DlpConfig":
        raw = raw or {}
        cats = raw.get("categories")
        custom = raw.get("custom_patterns") or raw.get("customPatterns") or []
        return cls(
            enabled=bool(raw.get("enabled", False)),
            categories=tuple(cats) if cats else _BUILTIN,
            custom_patterns=tuple(str(p) for p in custom),
            placeholder=str(raw.get("placeholder") or "[REDACTED:{label}]"),
        )


def _ph(config: DlpConfig, label: str) -> str:
    try:
        return config.placeholder.format(label=label)
    except (KeyError, IndexError, ValueError):
        return f"[REDACTED:{label}]"


def redact(text: str, config: DlpConfig) -> tuple[str, int]:
    """Return ``(redacted_text, num_redactions)``. No-op when disabled/empty."""
    if not config.enabled or not text:
        return text, 0
    count = 0

    if "secret" in config.categories:
        for pat in _SECRET_PATTERNS:
            text, n = pat.subn(_ph(config, "secret"), text)
            count += n
        text, n = _SECRET_ASSIGN.subn(
            lambda m: f"{m.group(1)}{m.group(2)}{_ph(config, 'secret')}", text
        )
        count += n

    if "email" in config.categories:
        text, n = _EMAIL.subn(_ph(config, "email"), text)
        count += n

    for raw in config.custom_patterns:
        try:
            text, n = re.subn(raw, _ph(config, "custom"), text)
            count += n
        except re.error:
            continue  # skip a malformed operator pattern rather than crash

    return text, count
