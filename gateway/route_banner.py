"""Route announcement and context-token banner helpers."""

from __future__ import annotations


BASE_CONTEXT_VERSION = "gateway-context-v1"


def format_context_window_token_line(
    *,
    approx_tokens: int,
    context_length: int,
    threshold_tokens: int = 0,
    base_context_version: str = BASE_CONTEXT_VERSION,
    loaded_skill_names: list[str] | None = None,
) -> str:
    skills = ", ".join(loaded_skill_names or []) or "none"
    prefix = f"Context: {base_context_version}; skills: {skills}."
    if context_length > 0:
        pct = min(999.0, approx_tokens / context_length * 100)
        threshold_part = (
            f", compress at ~{threshold_tokens:,}" if threshold_tokens else ""
        )
        return (
            f"🧮 {prefix} Loaded ~{approx_tokens:,} / {context_length:,} "
            f"prompt tokens ({pct:.0f}%{threshold_part})."
        )
    return f"🧮 {prefix} Loaded ~{approx_tokens:,} prompt tokens."
