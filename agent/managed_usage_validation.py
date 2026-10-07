"""Conservative raw-usage checks for opt-in managed turn observations.

Ordinary usage accounting can normalize absent/malformed optional fields to zero;
that is useful for display, but it must never become proof of complete spend.
"""
from __future__ import annotations

from typing import cast
from agent.usage_pricing import (
    _ANTHROPIC_USAGE_SHAPE, _CHAT_USAGE_SHAPE, _CODEX_USAGE_SHAPE,
)

_MISSING = object()
_MAX_SAFE_INTEGER = 2**53 - 1


def _raw_field(raw, path):
    for name in path:
        raw = raw.get(name, _MISSING) if isinstance(raw, dict) else getattr(raw, name, _MISSING)
        if raw is _MISSING:
            break
    return raw


def _bucket(raw, paths) -> tuple[list[int], int] | None:
    raw_values = [value for path in paths
                  if (value := _raw_field(raw, path)) is not _MISSING and value is not None]
    if any(type(value) is not int or not 0 <= value <= _MAX_SAFE_INTEGER for value in raw_values):
        return None
    values = [cast(int, value) for value in raw_values]
    nonzero = {value for value in values if value}
    if len(nonzero) > 1:  # conflicting aliases are not proof of a unique bucket
        return None
    return (values, next(iter(nonzero), 0))


def has_complete_raw_usage(agent, raw, canonical) -> bool:
    """Require both raw input/output and coherent optional cache/total buckets."""
    if not raw or canonical is None:
        return False
    mode = str(getattr(agent, "api_mode", "") or "").lower()
    provider = str(getattr(agent, "provider", "") or "").lower()
    shape = (_ANTHROPIC_USAGE_SHAPE if mode == "anthropic_messages" or provider == "anthropic"
             else _CODEX_USAGE_SHAPE if mode == "codex_responses" else _CHAT_USAGE_SHAPE)
    input_bucket, output_bucket = _bucket(raw, shape[0]), _bucket(raw, shape[1])
    read_bucket, write_bucket = _bucket(raw, shape[2]), _bucket(raw, shape[3])
    if input_bucket is None or output_bucket is None or read_bucket is None or write_bucket is None:
        return False
    input_values, prompt = input_bucket
    output_values, output = output_bucket
    _, read = read_bucket
    _, write = write_bucket
    if not input_values or not output_values:
        return False
    if shape is _ANTHROPIC_USAGE_SHAPE:
        if canonical.input_tokens != prompt:
            return False
    elif read + write > prompt or canonical.prompt_tokens != prompt:
        return False
    if (canonical.output_tokens != output or canonical.cache_read_tokens != read
            or canonical.cache_write_tokens != write):
        return False
    reasoning = _bucket(raw, (("output_tokens_details", "reasoning_tokens"),
                              ("completion_tokens_details", "reasoning_tokens")))
    if reasoning is None or reasoning[1] > output or canonical.reasoning_tokens != reasoning[1]:
        return False
    totals = _bucket(raw, (("total_tokens",), ("totalTokens",)))
    if totals is None:
        return False
    total_values, reported_total = totals
    return not total_values or reported_total == canonical.total_tokens
