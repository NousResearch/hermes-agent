"""Palier resolution for Kanban ``model_override`` writes.

A card may pin ``model: std`` — a palier from ``model.aliases``, not a wire id. Stored verbatim it
reaches the provider as ``Model "std" is not supported on this endpoint`` (HTTP 400) and burns the
card's attempt budget, so every path that STORES an override (tool create, CLI create, CLI
set-model) expands it to the route the worker launches: the model id AND the provider it must be
sent to.

The provider is not decoration. ``commandcode`` and ``commandcode-anthropic`` are two registered
providers over the same endpoint — ``chat_completions`` vs ``anthropic_messages`` — so
``commandcode-anthropic/claude-haiku-5-5`` belongs on the board as model ``claude-haiku-5-5`` plus
provider ``commandcode-anthropic``; left whole in the model string it is sent unsplit to whichever
provider the card kept, and 400s. The pair comes from the owners CLI startup uses
(``_load_direct_aliases`` for the alias table, ``resolve_startup_model_route`` for a
``provider/model`` override), so a card launches what ``/model <palier>`` would.
"""

from __future__ import annotations

from typing import Optional

_ALIASES_KEY = "model.aliases"


def _profile_config(config: Optional[dict]) -> dict:
    """The caller's config snapshot, else the active profile's ``config.yaml``."""
    if config is None:
        from hermes_cli.config import load_config

        config = load_config()
    return config if isinstance(config, dict) else {}


def _current_provider(config: dict) -> str:
    section = config.get("model")
    return str(section.get("provider") or "").strip().lower() if isinstance(section, dict) else ""


def _palier_shaped(name: str) -> bool:
    """A bare word — no ``/`` and no version token (every wire id carries a digit).

    Catches a mistyped or foreign palier (``stdd``, ``leger``, ``frontier``) while every real
    model id (``gpt-6-luna``, ``claude-haiku-5-5``, ``xiaomi/mimo-v2.6-pro``) passes.
    """
    return "/" not in name and not any(ch.isdigit() for ch in name)


def resolve_model_override(
    model: Optional[str], provider: Optional[str] = None, *, config: Optional[dict] = None,
) -> tuple[Optional[str], Optional[str]]:
    """``(model, provider)`` with a palier expanded to the route its worker launches.

    Raises ``ValueError`` — nothing is stored and no card is created — for an unknown
    palier-shaped name, a palier whose route a card cannot carry (it pins a ``base_url``), an
    entry declared without a model, or a provider contradicting the palier's own.
    ``config`` lets a caller pass an already-loaded profile config (tests, batched callers).
    """
    model = str(model or "").strip() or None
    provider = str(provider or "").strip() or None
    if not model:
        if provider:
            raise ValueError("'provider' requires 'model' to be set as well")
        return None, None
    from hermes_cli.model_switch import _load_direct_aliases, resolve_startup_model_route

    cfg = _profile_config(config)
    aliases = _load_direct_aliases(cfg)
    entry = aliases.get(model.lower())
    if entry is not None:
        if entry.base_url:
            # A card stores the model/provider pair only: an alias carrying its own endpoint
            # would otherwise launch against the profile's endpoint, silently.
            raise ValueError(
                f"palier {model!r} pins its own base_url, which a card cannot carry; pass the "
                f"wire model and its provider explicitly")
        target, target_provider = entry.model, entry.provider or "custom"
        if provider and provider != target_provider:
            raise ValueError(
                f"palier {model!r} targets provider {target_provider!r}, not {provider!r}")
        return target, provider or target_provider
    route = resolve_startup_model_route(
        model, current_provider=_current_provider(cfg), user_providers=cfg.get("providers"),
        custom_providers=cfg.get("custom_providers"), alias_config=cfg)
    if route is None:
        if _palier_shaped(model):
            known = ", ".join(sorted(aliases)) or "none declared"
            raise ValueError(
                f"unknown palier {model!r}: this profile's {_ALIASES_KEY} declares {known}. "
                f"Pass a wire id (one carrying a version token or a '/'), or declare the palier "
                f"in {_ALIASES_KEY}.")
        return model, provider
    if route.base_url:
        raise ValueError(
            f"{model!r} resolves to an alias with its own base_url, which a card cannot carry; "
            f"pass the wire model and its provider explicitly")
    if provider and provider != route.provider:
        raise ValueError(f"{model!r} routes to provider {route.provider!r}, not {provider!r}")
    return route.model, route.provider or None
