"""Catalog substring matching and residual-identity checks (sibling of ``model_metadata``)."""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Families whose generation digit is part of the name (``solar-mini`` vs ``solar-mini4``): their keys
# match only on an id boundary, after folding aggregator slugs (``solar-pro-3``) into the native form.
_BOUNDARY_MATCHED_KEY_PREFIXES = ("solar-",)
_HYPHENATED_GENERATION_RE = re.compile(
    rf"((?:{'|'.join(map(re.escape, _BOUNDARY_MATCHED_KEY_PREFIXES))})[a-z]+)-(\d{{1,2}})(?=[-:.@]|$)")

# Residuals that decorate a model id without changing its advertised window: region/mode markers
# (``:US``, ``:batch``), channel tags (``-latest``, ``-preview``). Generation/snapshot continuations
# (``.4``, ``-1017``) stay family scope so catch-alls still answer new lineup members. Anything else
# after a full catalog key is a different model, not a variant.
_DECORATIVE_RESIDUAL_RE = re.compile(
    r"^(?:[-:.@](?:us|eu|uk|ca|jp|global|intl|batch|chat|instruct|thinking|reasoning|cot|tee"
    r"|latest|preview|beta|alpha|dev|nightly|experimental|online|offline|stable|high|low|fast))+$")


def _normalize_model_version(model: str) -> str:
    """Dots -> dashes so Nous ids (claude-opus-4-6) compare with OpenRouter's (claude-opus-4.6)."""
    return model.replace(".", "-")


def _catalog_key_matches(key: str, model_lower: str) -> bool:
    """Substring match with version separators normalised on both sides, so a relay slug like
    ``z-ai-glm-5-3`` still hits the ``glm-5.3`` entry instead of the ``glm`` catch-all (#97398).
    Boundary-matched families: a key must be followed by ``-:.@`` or the end, and a key ending in
    ``-`` is the family default for bare names (``org/`` allowed) continuing with a lineup letter."""
    if key.startswith(_BOUNDARY_MATCHED_KEY_PREFIXES):
        model_lower = _HYPHENATED_GENERATION_RE.sub(r"\1\2", model_lower)
        if key.endswith("-"):
            return re.match(re.escape(key) + "[a-z]", model_lower.rsplit("/", 1)[-1]) is not None
        return re.search(re.escape(key) + r"(?:[-:.@]|$)", model_lower) is not None
    return key in model_lower or _normalize_model_version(key) in _normalize_model_version(model_lower)


def _catalog_hit_accepts_residual(key: str, residual: Optional[str]) -> bool:
    """Gate on how much identity the catalog key already carries.

    A bare family head without version digits (``minimax``, ``qwen``, ``glm``) is a catch-all by
    design — it answers new lineup members (``MiniMax-M2.7``). A key that already carries a full
    model identity (``glm-4.5``, ``kimi-k3``, ``qwen3.8-flash``) transfers its window only to a
    decoration (``:batch``, ``-cot``, ``-1017``, ``.4``), never to a sibling model id
    (``-air``, ``-next``) whose window is its own to advertise."""
    if not residual:
        return True
    if not any(ch.isdigit() for ch in key):
        return True  # bare family head: continuations are the point
    if _DECORATIVE_RESIDUAL_RE.match(residual):
        return True
    # ``.4-mini`` / ``-1017-free``: a numeric generation or snapshot continuation of the family.
    return re.match(r"^[ -:.@]*\.?\d", residual) is not None


def _catalog_key_residual(key: str, model_lower: str) -> Optional[str]:
    """Text following the catalog key's occurrence in the model id (version separators
    normalised); ``None`` for boundary-matched family keys, which carry their own semantics."""
    if key.startswith(_BOUNDARY_MATCHED_KEY_PREFIXES):
        return None
    idx = model_lower.find(key)
    if idx >= 0:
        return model_lower[idx + len(key):]
    norm_model, norm_key = _normalize_model_version(model_lower), _normalize_model_version(key)
    idx = norm_model.find(norm_key)
    if idx < 0:
        return None
    return norm_model[idx + len(norm_key):]


def _longest_key_match(
    table: dict[str, int],
    model_lower: str,
    *,
    require_decorative_residual: bool = False,
) -> Optional[tuple[str, int]]:
    """First ``(key, value)`` whose key is a substring of ``model_lower``, longest key first so
    specific entries (``gpt-5.4-mini``) beat their family catch-all (``gpt-5``); ties keep table order.

    ``require_decorative_residual=True`` (the custom-endpoint last resort) rejects a hit whose key is
    followed by an identity-bearing token — ``qwen3.8-flash`` matching ``qwen3.8-flash-next``,
    ``glm-4.5`` matching ``glm-4.5-air`` — and keeps scanning. Substring matching decodes relay slugs
    (``z-ai-glm-5-3`` → ``glm-5.3``, #97398), but a sibling model id is not a decoration: inheriting a
    sibling's window is false-safe, inflating compression budgets past what the endpoint accepts."""
    for key, value in sorted(table.items(), key=lambda x: len(x[0]), reverse=True):
        if not _catalog_key_matches(key, model_lower):
            continue
        if require_decorative_residual:
            residual = _catalog_key_residual(key, model_lower)
            if not _catalog_hit_accepts_residual(key, residual):
                logger.info(
                    "Catalog key %r matches model %r but trailing %r is a distinct model identity — "
                    "skipping its inherited %s window", key, model_lower, residual, f"{value:,}")
                continue
        return key, value
    return None
