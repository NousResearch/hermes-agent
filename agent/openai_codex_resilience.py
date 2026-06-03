"""OpenAI Codex transient-error classification and plain-English cron notices."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

OPENAI_CODEX_PROVIDER = "openai-codex"

TRANSIENT_ERROR_CLASSES = frozenset(
    {
        "upstream_503",
        "connection_refused",
        "connection_reset",
        "sse_no_event",
        "sse_idle_timeout",
    }
)

_SECRET_PATTERNS = (
    re.compile(r"\bsk-[a-zA-Z0-9_-]{8,}\b"),
    re.compile(r"\bBearer\s+[a-zA-Z0-9._-]{8,}\b", re.IGNORECASE),
    re.compile(r"\baccess[_-]?token[\"\\s:=]+[a-zA-Z0-9._-]{8,}", re.IGNORECASE),
)


@dataclass(frozen=True)
class ClassifiedOpenAiCodexError:
    error_class: str
    summary: str
    status_code: Optional[int]
    transient: bool


def redact_provider_error_text(text: str, *, max_len: int = 400) -> str:
    out = " ".join(str(text or "").split())
    for pattern in _SECRET_PATTERNS:
        out = pattern.sub("[redacted]", out)
    if len(out) > max_len:
        return out[: max_len - 1] + "…"
    return out


def _haystack(error: BaseException | str, status_code: Optional[int] = None) -> tuple[str, Optional[int]]:
    parts: list[str] = []
    if isinstance(error, BaseException):
        parts.append(str(error))
        parts.append(type(error).__name__)
    else:
        parts.append(str(error))
    code = status_code
    if code is None and isinstance(error, BaseException):
        code = getattr(error, "status_code", None)
    if code is not None:
        parts.append(f"http {code}")
    return " ".join(parts).lower(), code


def classify_openai_codex_error(
    error: BaseException | str,
    *,
    status_code: Optional[int] = None,
) -> ClassifiedOpenAiCodexError:
    haystack, code = _haystack(error, status_code)
    raw = str(error) if not isinstance(error, BaseException) else str(error)

    if code == 402 or any(
        token in haystack
        for token in ("billing", "credit", "payment required", "insufficient_quota")
    ):
        return ClassifiedOpenAiCodexError(
            "billing",
            redact_provider_error_text(raw or "Billing or credits exhausted"),
            code,
            False,
        )
    if code in (401, 403) or "unauthorized" in haystack or "forbidden" in haystack:
        return ClassifiedOpenAiCodexError(
            "auth",
            redact_provider_error_text(raw or "Authentication failed"),
            code,
            False,
        )
    if code == 429 or "rate limit" in haystack or "too many requests" in haystack:
        return ClassifiedOpenAiCodexError(
            "rate_limit",
            redact_provider_error_text(raw or "Rate limited"),
            code,
            True,
        )
    if code == 503 or any(
        token in haystack
        for token in (
            "upstream connect error",
            "connection termination",
            "remote connection failure",
            "delayed connect error",
        )
    ):
        return ClassifiedOpenAiCodexError(
            "upstream_503",
            redact_provider_error_text(raw or "ChatGPT Codex backend returned HTTP 503"),
            code or 503,
            True,
        )
    if "connection refused" in haystack:
        return ClassifiedOpenAiCodexError(
            "connection_refused",
            redact_provider_error_text(raw or "Connection refused by ChatGPT Codex backend"),
            code,
            True,
        )
    if "no sse events" in haystack or "no bytes within" in haystack or "codex stream produced no" in haystack:
        sse_class = (
            "sse_no_event"
            if any(token in haystack for token in ("no bytes", "ttfb", "first byte"))
            else "sse_idle_timeout"
        )
        default = (
            "Codex stream opened but sent no events"
            if sse_class == "sse_no_event"
            else "Codex stream stalled after the first event"
        )
        return ClassifiedOpenAiCodexError(
            sse_class,
            redact_provider_error_text(raw or default),
            code,
            True,
        )
    if any(
        token in haystack
        for token in ("connection reset", "connection lost", "connection closed", "disconnect/reset")
    ):
        return ClassifiedOpenAiCodexError(
            "connection_reset",
            redact_provider_error_text(raw or "Connection reset while talking to ChatGPT Codex"),
            code,
            True,
        )

    return ClassifiedOpenAiCodexError(
        "unknown",
        redact_provider_error_text(raw or "Unknown provider error"),
        code,
        False,
    )


def resolve_openai_codex_retry_budget(
    *,
    platform: Optional[str],
    provider: Optional[str],
    default_retries: int = 3,
) -> int:
    base = max(1, int(default_retries))
    if (provider or "").strip().lower() != OPENAI_CODEX_PROVIDER:
        return base
    if (platform or "").strip().lower() != "cron":
        return base
    raw = os.environ.get("HERMES_CODEX_CRON_API_MAX_RETRIES", "").strip()
    if not raw:
        return max(base, 5)
    try:
        parsed = int(raw)
    except ValueError:
        return max(base, 5)
    return max(base, max(1, parsed))


_ERROR_CLASS_PLAIN = {
    "upstream_503": "ChatGPT Codex backend outage (HTTP 503)",
    "connection_refused": "ChatGPT Codex backend refused the connection",
    "connection_reset": "ChatGPT Codex connection dropped mid-request",
    "sse_no_event": "Codex stream never started (no SSE events)",
    "sse_idle_timeout": "Codex stream stalled (no SSE events after the first byte)",
    "rate_limit": "ChatGPT Codex rate limit",
    "billing": "ChatGPT Codex billing or quota limit",
    "auth": "ChatGPT Codex authentication problem",
    "unknown": "ChatGPT Codex provider error",
}


def format_openai_codex_failure_notice(
    *,
    job_name: Optional[str],
    provider: Optional[str],
    model: Optional[str],
    error_class: str,
    summary: str,
    attempts: int,
    degraded: bool = False,
) -> str:
    label = (job_name or "").strip() or "Scheduled Hermes job"
    prov = (provider or "").strip() or OPENAI_CODEX_PROVIDER
    mdl = (model or "").strip() or "(unknown model)"
    what = _ERROR_CLASS_PLAIN.get(error_class, _ERROR_CLASS_PLAIN["unknown"])
    mode = "degraded" if degraded else "failed"
    return "\n".join(
        [
            f"⚠️ {label} — AI provider {mode} (not a Kynver harness failure).",
            "",
            f"Provider: {prov} (ChatGPT Codex subscription)",
            f"Model: {mdl}",
            f"What happened: {what} after {attempts} attempt(s) with backoff.",
            f"Details: {summary}",
            "",
            "This is a ChatGPT/Codex API issue on chatgpt.com — not AgentOS, not the Kynver runtime, and not your local harness.",
            "The cron job will run again on its next schedule. If it persists, check ChatGPT/Codex status or re-auth with: hermes auth status openai-codex",
        ]
    )


def format_cron_job_delivery_failure(job_name: str, notice: str) -> str:
    return f"⚠️ Cron job '{job_name}' — provider degraded\n\n{notice}"


def maybe_format_codex_cron_failure(
    agent: object,
    error: BaseException | str,
    *,
    attempts: int,
    job_name: Optional[str] = None,
) -> Optional[str]:
    """Return a plain-English cron notice when this is an openai-codex cron failure."""
    platform = str(getattr(agent, "platform", "") or "").lower()
    provider = str(getattr(agent, "provider", "") or "").lower()
    if platform != "cron" or provider != OPENAI_CODEX_PROVIDER:
        return None
    status_code = getattr(error, "status_code", None) if isinstance(error, BaseException) else None
    classified = classify_openai_codex_error(error, status_code=status_code)
    return format_openai_codex_failure_notice(
        job_name=job_name,
        provider=provider,
        model=str(getattr(agent, "model", "") or ""),
        error_class=classified.error_class,
        summary=classified.summary,
        attempts=attempts,
        degraded=classified.transient,
    )
