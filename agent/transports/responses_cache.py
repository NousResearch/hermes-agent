"""Content-free cache accounting projected from a terminal Responses frame."""
from dataclasses import dataclass
from typing import Any


_DIAGNOSTIC_TYPES = ("cache_hit", "cache_miss", "comparison_response_not_found", "unavailable")
_MISS_REASONS = (
    "model_changed", "prompt_cache_key_changed", "service_tier_changed", "tools_changed",
    "text_format_changed", "reasoning_effort_changed", "verbosity_changed", "context_compacted", "input_changed",
)


def _field(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _counts(obj: Any, *names: str) -> dict[str, int]:
    return {key: n for key in names if type(n := _field(obj, key)) is int and 0 <= n < 2**63}


def _diagnostics(obj: Any) -> dict:
    kind = _field(obj, "type")
    if kind not in _DIAGNOSTIC_TYPES:
        return {}
    diag: dict[str, str | int] = {"type": kind}
    if kind == "cache_miss":
        if (reason := _field(obj, "reason")) in _MISS_REASONS:
            diag["reason"] = reason
        diag.update(_counts(obj, "cache_missed_tokens", "comparison_reusable_tokens"))
    return diag


def _usage_evidence(usage: Any) -> dict:
    details = _field(usage, "input_tokens_details")
    evidence: dict = {**_counts(usage, "input_tokens"), **_counts(details, "cached_tokens", "cache_write_tokens")}
    fields = _field(_field(usage, "attribution"), "request_fields")
    counters = ("input_tokens", "cached_tokens", "cache_write_tokens", "output_tokens")
    request_fields = {key: counts for key in ("instructions", "tools", "input")
                      if (counts := _counts(_field(fields, key), *counters))}
    if request_fields:
        evidence["request_fields"] = request_fields
    return evidence


@dataclass(frozen=True)
class TerminalCacheMetadata:
    payload: dict

    @classmethod
    def from_terminal(cls, response: Any) -> "TerminalCacheMetadata | None":
        evidence: dict = {}
        if diag := _diagnostics(_field(response, "prompt_cache_diagnostics")):
            evidence["prompt_cache_diagnostics"] = diag
        if isinstance(transformations := _field(response, "input_transformations"), list):
            evidence["input_transformation_count"] = len(transformations)
        return cls(evidence) if evidence else None


def responses_cache_hook_fields(response: Any, api_mode: str) -> dict:
    if api_mode != "codex_responses":
        return {}
    evidence = _usage_evidence(_field(response, "usage"))
    metadata = getattr(response, "_hermes_cache_metadata", None)
    if isinstance(metadata, TerminalCacheMetadata):
        evidence.update(metadata.payload)
    return {"responses_cache": evidence} if evidence else {}
