"""Exact, dated wire contracts for provider fast-mode features.

This module is deliberately dependency-free so both the model resolver and
transport adapters can consume the same immutable catalog without creating an
import cycle.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Optional


# Snapshot of https://openai.com/api-priority-processing/ on 2026-07-12.
# This is the source contract, not the Hermes picker inventory. The shipped
# openai_priority contract below is its exact intersection with the current
# ``openai-api`` provider catalog.
OPENAI_PRIORITY_SOURCE_MODELS: tuple[str, ...] = (
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.4-mini",
    "gpt-5.4",
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5.1-codex",
    "gpt-5-codex",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-2024-11-20",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-05-13",
    "gpt-4o-mini",
    "o3",
    "o4-mini",
)


def _contract(*, source_url: str, models: tuple[str, ...]) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "source_url": source_url,
            "checked_date": "2026-07-12",
            "models": models,
        }
    )


FAST_MODE_CAPABILITY_CATALOG: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "openai_priority": _contract(
            source_url="https://openai.com/api-priority-processing/",
            models=(
                "gpt-5.6-sol",
                "gpt-5.6-terra",
                "gpt-5.6-luna",
                "gpt-5.5",
                "gpt-5.4-mini",
                "gpt-5.4",
                "gpt-5-mini",
                "gpt-4.1",
                "gpt-4o",
                "gpt-4o-mini",
            ),
        ),
        "codex_fast": _contract(
            source_url="https://developers.openai.com/codex/speed",
            models=("gpt-5.5", "gpt-5.4"),
        ),
        # Opus 4.7 is intentionally absent. Anthropic deprecated its Fast
        # contract on 2026-06-25 and will reject speed=fast after 2026-07-24.
        "anthropic_fast": _contract(
            source_url=(
                "https://platform.claude.com/docs/en/build-with-claude/fast-mode"
            ),
            models=("claude-opus-4-8",),
        ),
    }
)


def normalize_fast_model_id(model_id: Optional[str]) -> str:
    """Normalize only documented spelling aliases; retain all other suffixes."""
    normalized = str(model_id or "").strip().lower()
    if normalized.startswith(("anthropic/", "openai/")):
        normalized = normalized.split("/", 1)[1]
    if normalized == "claude-opus-4.8":
        return "claude-opus-4-8"
    return normalized


def anthropic_fast_contract_accepts(model_id: Optional[str]) -> bool:
    """Return whether *model_id* exactly matches the native Anthropic contract."""
    return normalize_fast_model_id(model_id) in FAST_MODE_CAPABILITY_CATALOG[
        "anthropic_fast"
    ]["models"]
