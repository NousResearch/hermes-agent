"""Shared Anthropic prior-turn thinking retention policy.

Anthropic preserves and bills prior assistant thinking on Opus 4.5+ and
Sonnet 4.6+ (plus the newer Fable/Mythos families). Older models accept
replayed blocks but strip them server-side. Keep the model capability in
one place so message conversion and context accounting cannot diverge.
"""

from __future__ import annotations

import re
from typing import Any

from agent.anthropic_endpoints import (
    _is_nous_portal_endpoint,
    _is_third_party_anthropic_endpoint,
)


_CLAUDE_VERSION_RE = re.compile(
    r"claude[-_.](opus|sonnet|fable|mythos)[-_.](\d+)(?:[-_.](\d+))?",
    re.IGNORECASE,
)


def _claude_family_version(model: Any) -> tuple[str, tuple[int, int]] | None:
    if not isinstance(model, str):
        return None
    match = _CLAUDE_VERSION_RE.search(model.strip())
    if not match:
        return None
    family = match.group(1).lower()
    major = int(match.group(2))
    minor = int(match.group(3) or 0)
    return family, (major, minor)


def model_preserves_prior_thinking(model: Any) -> bool:
    """Whether Anthropic keeps prior assistant thinking in model-visible context."""
    if isinstance(model, str) and re.search(
        r"claude[-_.]mythos[-_.]preview", model.strip(), re.IGNORECASE
    ):
        return True
    parsed = _claude_family_version(model)
    if parsed is None:
        return False
    family, version = parsed
    if family == "opus":
        return version >= (4, 5)
    if family == "sonnet":
        return version >= (4, 6)
    if family in {"fable", "mythos"}:
        return version in {(5, 0), (5, 1)}
    return False


def native_anthropic_preserves_prior_thinking(base_url: Any, model: Any) -> bool:
    """True for direct Anthropic/Nous Portal routes whose model retains old thinking."""
    native_route = (
        not _is_third_party_anthropic_endpoint(base_url)
        or _is_nous_portal_endpoint(base_url)
    )
    return native_route and model_preserves_prior_thinking(model)
