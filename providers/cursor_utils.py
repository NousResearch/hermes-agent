"""Shared helpers for the Cursor external-process provider."""

from __future__ import annotations

import os
import shlex
import shutil

DEFAULT_CURSOR_COMMAND = "cursor-agent"

# Cursor can expose a very large live catalog. Keep the models most likely to
# be useful for Hermes-agent workflows at the top of picker/discovery results,
# while preserving every other live model after them.
CURSOR_PREFERRED_MODELS: tuple[str, ...] = (
    "auto",
    "composer-2.5",
    "composer-2.5-fast",
    "composer-2",
    "composer-2-fast",
    "gpt-5.5-medium",
    "gpt-5.5-medium-fast",
    "gpt-5.5-high",
    "gpt-5.5-high-fast",
    "gpt-5.5-low",
    "gpt-5.5-low-fast",
    "claude-opus-4-7-medium",
    "claude-opus-4-7-high",
    "claude-4.6-sonnet-medium",
    "claude-4.6-sonnet-medium-thinking",
    "gemini-3.1-pro",
    "gemini-3-flash",
    "grok-4.3",
    "kimi-k2.5",
)


def prioritize_cursor_models(model_ids: list[str]) -> list[str]:
    """Move high-value Cursor models to the top without dropping live entries."""
    seen: set[str] = set()
    deduped: list[str] = []
    for model_id in model_ids:
        cleaned = str(model_id).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)

    by_lower = {mid.lower(): mid for mid in deduped}
    preferred = [by_lower[p.lower()] for p in CURSOR_PREFERRED_MODELS if p.lower() in by_lower]
    preferred_keys = {mid.lower() for mid in preferred}
    return preferred + [mid for mid in deduped if mid.lower() not in preferred_keys]


def resolve_cursor_command() -> str:
    """Return the configured cursor-agent command or the default binary name."""
    return (
        os.getenv("HERMES_CURSOR_COMMAND", "").strip()
        or os.getenv("CURSOR_AGENT_PATH", "").strip()
        or DEFAULT_CURSOR_COMMAND
    )


def resolve_cursor_extra_args() -> list[str]:
    """Return optional Cursor CLI args configured through the environment."""
    raw = os.getenv("HERMES_CURSOR_ARGS", "").strip()
    if not raw:
        return []
    return shlex.split(raw)


def resolve_cursor_command_path(command: str | None = None) -> str | None:
    """Resolve Cursor command while preserving explicit wrapper paths.

    Bare command names must resolve on PATH. Explicit paths, including relative
    paths such as ``./bin/cursor-wrapper`` and Windows-style paths with a
    backslash, are returned as-is when PATH lookup cannot resolve them.
    """
    candidate = (command or "").strip()
    if not candidate:
        return None
    resolved = shutil.which(candidate)
    if resolved:
        return resolved
    if "/" in candidate or "\\" in candidate:
        return candidate
    return None
