"""Neutral provider/model string utility shared across surfaces.

Lives in a low-level module that BOTH ``gateway/runtime_footer.py`` and
``agent/conversation_compression.py`` can import without creating a
gateway↔agent circular import (gateway already imports agent).
"""

from __future__ import annotations

from typing import Optional, Tuple


def split_provider_model(
    provider: Optional[str], model: Optional[str]
) -> Tuple[str, str]:
    """Resolve a clean ``(provider, model)`` pair.

    When the ``model`` ALREADY carries a ``provider/`` prefix, that embedded
    prefix wins and any separately-supplied ``provider`` is ignored — this
    avoids an ugly triple like ``claude-app/claude-app/claude-opus-4-8`` when a
    caller passes both a provider and a prefixed model (the live config has
    ``model.default: claude-app/claude-opus-4-8`` AND ``model.provider:
    claude-app``). The model's own prefix is the more specific source.
    """
    prov = (provider or "").strip()
    mdl = (model or "").strip()
    if "/" in mdl:
        # The model carries its own provider prefix — it's authoritative.
        prov, _, mdl = mdl.partition("/")
    return prov, mdl


def format_provider_model(provider: Optional[str], model: Optional[str]) -> str:
    """Render ``provider/model`` (or bare ``model`` when no provider), de-duped."""
    prov, mdl = split_provider_model(provider, model)
    if prov and mdl:
        return f"{prov}/{mdl}"
    return mdl
