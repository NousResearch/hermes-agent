"""Localized automatic compression progress and shared catalog-based classification."""

import re
from functools import lru_cache
from typing import Any

from agent.i18n import SUPPORTED_LANGUAGES, t

def _count(value: Any) -> str:
    """Render numeric status values with the grouping used by legacy strings."""
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def compaction_status(*, lang: str | None = None) -> str:
    return t("gateway.compress.status_compacting", lang=lang)


def compaction_done_status(*, lang: str | None = None) -> str:
    return t("gateway.compress.status_compacted", lang=lang)


def pre_api_compression_status(tokens: int, *, lang: str | None = None) -> str:
    return t("gateway.compress.status_pre_api", lang=lang, tokens=_count(tokens))


def preflight_compression_status(
    tokens: int,
    threshold: int,
    *,
    lang: str | None = None,
) -> str:
    return t(
        "gateway.compress.status_preflight",
        lang=lang,
        tokens=_count(tokens),
        threshold=_count(threshold),
    )


def idle_compaction_status(
    idle_seconds: int,
    tokens: int,
    *,
    lang: str | None = None,
) -> str:
    return t(
        "gateway.compress.status_idle",
        lang=lang,
        idle_seconds=str(idle_seconds),
        tokens=_count(tokens),
    )


def compression_retry_too_large_status(
    tokens: int,
    attempt: int,
    cap: int,
    *,
    lang: str | None = None,
) -> str:
    return t(
        "gateway.compress.status_retry_too_large",
        lang=lang,
        tokens=_count(tokens),
        attempt=attempt,
        cap=cap,
    )


def compression_retry_messages_status(
    before: int,
    after: int,
    *,
    lang: str | None = None,
) -> str:
    return t(
        "gateway.compress.status_retry_messages",
        lang=lang,
        before=before,
        after=after,
    )


def compression_retry_tokens_status(
    before: int,
    after: int,
    *,
    lang: str | None = None,
) -> str:
    return t(
        "gateway.compress.status_retry_tokens",
        lang=lang,
        before=_count(before),
        after=_count(after),
    )


def compression_retry_context_reduced_status(
    new_ctx: int,
    old_ctx: int,
    *,
    lang: str | None = None,
) -> str:
    return t(
        "gateway.compress.status_retry_reduced",
        lang=lang,
        new_ctx=_count(new_ctx),
        old_ctx=_count(old_ctx),
    )


def compression_retry_payload_too_large_status(
    attempt: int,
    cap: int,
    *,
    lang: str | None = None,
) -> str:
    return t(
        "gateway.compress.status_retry_payload_too_large",
        lang=lang,
        attempt=attempt,
        cap=cap,
    )


def compression_retry_retained_vision_status(
    *,
    lang: str | None = None,
) -> str:
    return t(
        "gateway.compress.status_retry_retained_vision",
        lang=lang,
    )


def routine_compression_status_samples(
    *, lang: str | None = None
) -> tuple[str, ...]:
    """Return representative output for every routine compression status."""
    return (
        compaction_status(lang=lang),
        compaction_heartbeat_status(lang=lang),
        compaction_done_status(lang=lang),
        compression_retry_bytes_status(123456, 12345, lang=lang),
        pre_api_compression_status(123456, lang=lang),
        preflight_compression_status(120000, 100000, lang=lang),
        idle_compaction_status(3600, 120000, lang=lang),
        compression_retry_too_large_status(250000, 1, 3, lang=lang),
        compression_retry_messages_status(30, 12, lang=lang),
        compression_retry_tokens_status(250000, 120000, lang=lang),
        compression_retry_context_reduced_status(120000, 250000, lang=lang),
        compression_retry_payload_too_large_status(1, 3, lang=lang),
        compression_retry_retained_vision_status(lang=lang),
    )



def compaction_heartbeat_status(*, lang: str | None = None) -> str:
    return t("gateway.compress.status_heartbeat", lang=lang)


def compression_retry_bytes_status(before: int, after: int, *, lang: str | None = None) -> str:
    return t("gateway.compress.status_retry_bytes", lang=lang, before=_count(before), after=_count(after))


_STATUS_KEYS = (
    "status_compacting", "status_heartbeat", "status_compacted", "status_pre_api", "status_preflight",
    "status_idle", "status_retry_too_large", "status_retry_messages", "status_retry_tokens",
    "status_retry_reduced", "status_retry_payload_too_large", "status_retry_retained_vision", "status_retry_bytes",
)


@lru_cache(maxsize=64)
def _status_patterns(templates: tuple[str, ...]):
    return tuple(re.compile(r"[\d,]+".join(re.escape(part) for part in re.split(r"\{[^{}]*\}", text)))
                 for text in templates)


def compression_status_kind(text: str) -> str | None:
    """Classify catalog output across profiles without matching arbitrary warning prose."""
    for lang in SUPPORTED_LANGUAGES:
        templates = tuple(t(f"gateway.compress.{key}", lang=lang) for key in _STATUS_KEYS)
        for key, pattern in zip(_STATUS_KEYS, _status_patterns(templates)):
            if pattern.fullmatch(text.strip()):
                return "compacted" if key == "status_compacted" else "compacting"
    return None
