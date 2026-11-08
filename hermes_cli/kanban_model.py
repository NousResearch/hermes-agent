"""Palier resolution for Kanban ``model_override`` writes.

A card may pin ``model: std`` — a palier name from ``model.aliases``, not a wire id. Stored
verbatim it reaches the provider as ``Model "std" is not supported on this endpoint`` (HTTP 400)
and burns the card's attempt budget, so every path that STORES an override (tool create, CLI
create, CLI set-model) expands it first, and the board always carries the id the worker sends.

Targets are stored verbatim: the prefix in ``commandcode-anthropic/claude-haiku-5-5`` is a
routing prefix inside the Command Code endpoint, not a ``provider/model`` split (``xiaomi/`` is no
provider either), so the provider stays whatever the caller or the alias entry declares — and a
caller-declared provider that contradicts the entry is an error, not a silent mixture.
"""

from __future__ import annotations

from typing import Any, Optional

_ALIASES_KEY = "model.aliases"


def _alias_table(config: Optional[dict]) -> dict[str, Any]:
    """The active profile's ``model.aliases``, keyed lower-case ({} when undeclared)."""
    if config is None:
        from hermes_cli.config import load_config

        config = load_config()
    section = config.get("model") if isinstance(config, dict) else None
    aliases = section.get("aliases") if isinstance(section, dict) else None
    if not isinstance(aliases, dict):
        return {}
    return {str(name).strip().lower(): value for name, value in aliases.items()}


def _palier_shaped(name: str) -> bool:
    """A bare word — no ``/`` and no version token (every wire id carries a digit).

    Catches a mistyped or foreign palier (``stdd``, ``leger``, ``frontier``) while every real
    model id (``gpt-6-luna``, ``claude-haiku-5-5``, ``xiaomi/mimo-v2.6-pro``) passes.
    """
    return "/" not in name and not any(ch.isdigit() for ch in name)


def resolve_model_override(
    model: Optional[str], provider: Optional[str] = None, *, config: Optional[dict] = None,
) -> tuple[Optional[str], Optional[str]]:
    """``(model, provider)`` with a palier expanded to its wire id.

    Raises ``ValueError`` — nothing is stored and no card is created — for an unknown
    palier-shaped name, an entry declared without a model, or a provider contradicting the entry.
    ``config`` lets a caller pass an already-loaded profile config (tests, batched callers).
    """
    model = str(model or "").strip() or None
    provider = str(provider or "").strip() or None
    if not model:
        if provider:
            raise ValueError("'provider' requires 'model' to be set as well")
        return None, None
    aliases = _alias_table(config)
    entry = aliases.get(model.lower())
    if entry is None:
        if _palier_shaped(model):
            known = ", ".join(sorted(aliases)) or "none declared"
            raise ValueError(
                f"unknown palier {model!r}: this profile's {_ALIASES_KEY} declares {known}. "
                f"Pass a wire id (one carrying a version token or a '/'), or declare the palier "
                f"in {_ALIASES_KEY}.")
        return model, provider
    if isinstance(entry, dict):
        target = str(entry.get("model") or "").strip()
        entry_provider = str(entry.get("provider") or "").strip() or None
    else:
        target, entry_provider = str(entry or "").strip(), None
    if not target:
        raise ValueError(f"palier {model!r} is declared in {_ALIASES_KEY} without a model")
    if provider and entry_provider and provider != entry_provider:
        raise ValueError(
            f"palier {model!r} targets provider {entry_provider!r}, not {provider!r}")
    return target, provider or entry_provider
