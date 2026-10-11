"""Opt-in policy: pinning a worker to some providers needs a stated reason.

Some providers are shared pools, e.g. one subscription seat or one
quota-limited account that several agents draw from. Pinning a card, a batch
of cards, or a whole lane to such a provider can starve everything else that
uses it, so an operator may want that to be a deliberate act rather than a
typo away.

``kanban.pin_reason_required_providers`` in ``config.yaml`` lists
case-insensitive fnmatch globs matched against the provider name. A route
write (``create --provider``, ``set-model --provider``, ``lane-model set``)
whose provider matches is refused unless it carries a non-empty
``pin_reason``, which is then recorded on the card's event.

The list is empty by default, so a board that never sets it behaves exactly
as before this module existed.
"""

from __future__ import annotations

import fnmatch
from typing import Optional, Sequence

CONFIG_KEY = "pin_reason_required_providers"


def required_provider_globs(cfg: Optional[dict] = None) -> tuple[str, ...]:
    """Configured globs (``()`` when unset). ``cfg=None`` reads config.yaml.

    An unreadable config yields ``()``: the policy is opt-in, so a broken read
    falls back to the default (no policy) rather than refusing every route.
    """
    if cfg is None:
        try:
            from hermes_cli.config import load_config

            cfg = load_config() or {}
        except Exception:
            return ()
    kanban = cfg.get("kanban") if isinstance(cfg, dict) else None
    raw = kanban.get(CONFIG_KEY) if isinstance(kanban, dict) else None
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(g.strip() for g in raw if isinstance(g, str) and g.strip())


def matching_glob(provider: Optional[str], globs: Sequence[str]) -> Optional[str]:
    """The first glob ``provider`` matches, or None."""
    name = (provider or "").strip().lower()
    if not name:
        return None
    for glob in globs:
        if fnmatch.fnmatchcase(name, glob.lower()):
            return glob
    return None


def check_route_pin(
    provider: Optional[str],
    pin_reason: Optional[str],
    *,
    globs: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Validate one route write; return the normalized reason (or None).

    Raises ``ValueError`` when ``provider`` matches a configured glob and no
    reason was given, or when a reason is given without a provider (it would
    pin nothing, so it is almost certainly a mistyped command).
    """
    reason = (pin_reason or "").strip() or None
    provider = (provider or "").strip() or None
    if reason and not provider:
        raise ValueError("--pin-reason needs a --provider to pin")
    if not provider:
        return None
    glob = matching_glob(provider, required_provider_globs() if globs is None else globs)
    if glob and not reason:
        raise ValueError(
            f"provider {provider!r} matches kanban.{CONFIG_KEY} ({glob!r}); "
            f"pinning a worker to it needs a stated reason: pass --pin-reason \"<why>\""
        )
    return reason
