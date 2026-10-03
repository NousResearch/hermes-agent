"""Resolve a cron job's model through the same alias resolver ``/model`` uses.

A job's ``model`` is user-set (``hermes cron create/edit --model``, the dashboard, ``cron.model`` in
config.yaml) and used to reach the provider verbatim, so the short names every other model surface
accepts — ``kimi``, ``gpt5``, a ``model_aliases:`` tier like ``fast`` — produced an opaque 400 at
fire time. Resolution happens once at create/update (the stored job is the concrete route, and an
ambiguous alias is refused where the user can see it) and again at fire time for the ``cron.model``
fleet default and jobs stored before this existed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _configured_provider(cfg: Dict[str, Any]) -> str:
    cron_cfg = cfg.get("cron") or {}
    model_cfg = cfg.get("model") or {}
    for value in (
        cron_cfg.get("model_provider") if isinstance(cron_cfg, dict) else None,
        model_cfg.get("provider") if isinstance(model_cfg, dict) else None,
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def resolve_job_model_route(
    model: Optional[str], provider: Optional[str] = None, base_url: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[str]]:
    """``{"model", "provider", "base_url"}`` for the job with a short alias expanded.

    A catalog alias (``kimi`` → ``moonshotai/kimi-k2.5``) rewrites the model only, so the job keeps
    following the provider it was created against exactly like a full ``--model <id>`` does; a
    ``model_aliases:`` direct alias also carries its provider and base_url onto a job that pins
    neither. A full model id, or anything the resolver cannot classify, passes through unchanged —
    only a hit by alias NAME rewrites the route, since a reverse match of a full id against a direct
    alias would silently move the job to that alias's provider. Raises ``ValueError`` with ``/model``'s
    disambiguation text when the alias matches several catalog models.
    """
    route: Dict[str, Optional[str]] = {"model": model, "provider": provider, "base_url": base_url}
    raw = (model or "").strip()
    if not raw:
        return route
    try:
        from hermes_cli.model_switch import (
            DIRECT_ALIASES, AmbiguousAliasError, _ambiguous_alias_message, resolve_alias)
        if cfg is None:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
        current_provider = (provider or "").strip() or _configured_provider(cfg)
        user_providers = cfg.get("providers") if isinstance(cfg.get("providers"), dict) else None
        custom_providers = cfg.get("custom_providers") if isinstance(cfg.get("custom_providers"), list) else None
        try:
            hit = resolve_alias(raw, current_provider, user_providers, custom_providers)
        except AmbiguousAliasError as err:
            raise ValueError(_ambiguous_alias_message(
                err, verb="scheduling it", hint="Pin one with --model <exact-model-name>.")) from err
    except ValueError:
        raise
    except Exception as exc:  # catalog/config trouble must never block scheduling a job
        logger.debug("cron model alias resolution skipped for %r: %s", raw, exc)
        return route
    if hit is None or hit[2] != raw.lower():
        return route
    hit_provider, resolved_model, alias_name = hit
    route["model"] = resolved_model
    direct = DIRECT_ALIASES.get(alias_name)  # populated by resolve_alias for the active profile
    if direct is not None and not (provider or "").strip() and not (base_url or "").strip():
        route["provider"] = hit_provider or None
        route["base_url"] = direct.base_url or None
    return route
