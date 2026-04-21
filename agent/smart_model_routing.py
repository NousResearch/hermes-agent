"""Match-based model routing via ``model.routes``.

A pure helper that lets callers pick a model + provider bundle based on a
context dict describing the turn (``platform``, ``source_kind``, etc.).
The router iterates ``model.routes`` — a list of ``{match, model, provider,
api_key, base_url, ...}`` entries — and applies the first route whose
``match`` predicates are all satisfied by the context.

Two legacy shorthand forms are supported for backwards compatibility and
config ergonomics (the ``routes`` list is canonical; shims synthesize
entries from them at match time):

    model.platforms.<name>:       routes by platform (pre-unify shorthand)
    model.by_source.<kind>:       routes by source identity (owner /
                                  hub_peer / stranger / cron)

Example config::

    model:
      default: my-fast-model
      routes:
        - match: { source_kind: owner }
          model: my-strong-model
          api_key: sk-owner
        - match: { platform: hub }
          model: my-fast-model
        - match: { source_kind: cron }
          model: my-fast-model
        - match: { platform: discord, source_kind: stranger }
          model: some-other

Explicit ``routes`` always evaluate first; legacy shims run last, so a new
``routes:`` block always takes priority over an equivalent legacy entry.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple


SOURCE_KIND_OWNER = "owner"
SOURCE_KIND_HUB_PEER = "hub_peer"
SOURCE_KIND_STRANGER = "stranger"
SOURCE_KIND_CRON = "cron"
KNOWN_SOURCE_KINDS = frozenset({
    SOURCE_KIND_OWNER,
    SOURCE_KIND_HUB_PEER,
    SOURCE_KIND_STRANGER,
    SOURCE_KIND_CRON,
})

_OVERRIDE_RUNTIME_KEYS = ("api_key", "base_url", "provider", "api_mode", "command", "args")

logger = logging.getLogger(__name__)


def _normalize_routes(model_config: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collect the effective ordered route list from ``model_config``.

    Precedence (first match wins during lookup):

    1. Explicit ``routes:`` list as declared in config.
    2. Legacy ``platforms.<name>`` entries (synthesized as
       ``{match: {platform: <name>}, ...}``).
    3. Legacy ``by_source.<kind>`` entries (synthesized as
       ``{match: {source_kind: <kind>}, ...}``).
    """
    if not isinstance(model_config, dict):
        return []

    routes: List[Dict[str, Any]] = []

    explicit = model_config.get("routes")
    if isinstance(explicit, list):
        for item in explicit:
            if isinstance(item, dict) and isinstance(item.get("match"), dict):
                routes.append(item)

    platforms = model_config.get("platforms")
    if isinstance(platforms, dict):
        for name, override in platforms.items():
            if isinstance(override, str) and override:
                routes.append({"match": {"platform": str(name)}, "model": override})
            elif isinstance(override, dict) and override:
                routes.append({"match": {"platform": str(name)}, **override})

    by_source = model_config.get("by_source")
    if isinstance(by_source, dict):
        for kind, override in by_source.items():
            if isinstance(override, dict) and override:
                routes.append({"match": {"source_kind": str(kind)}, **override})

    return routes


def _route_matches(match_spec: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """Return True when every key in ``match_spec`` equals the context value.

    Missing keys in ``context`` never match (empty context satisfies empty
    match only). String comparison is case-sensitive.
    """
    if not isinstance(match_spec, dict) or not match_spec:
        return False
    for key, expected in match_spec.items():
        actual = context.get(key)
        if actual is None:
            return False
        if str(actual) != str(expected):
            return False
    return True


def apply_route(
    model: str,
    runtime_kwargs: Dict[str, Any],
    model_config: Optional[Dict[str, Any]],
    context: Optional[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    """Apply the first matching ``model.routes`` entry to ``(model, runtime_kwargs)``.

    ``context`` is a dict describing the turn. Current recognised keys:

    * ``platform``   — inbound platform string ("telegram", "hub", "cli", ...)
    * ``source_kind`` — ``"owner"`` / ``"hub_peer"`` / ``"stranger"`` /
                        ``"cron"`` as classified by the caller.

    Callers may add additional keys (e.g. ``user_id``); routes that don't
    reference them are unaffected.

    Each route entry has shape::

        {
          "match": { "platform": "hub", "source_kind": "stranger", ... },
          "model": "my-model",            # optional; preserves base when missing
          "provider": "custom",           # optional
          "api_key": "sk-...",
          "base_url": "https://...",
          "api_mode": "chat_completions",
          "command": [...],  "args": [...]
        }

    Partial overrides are supported — any field not set on the matched route
    keeps its base value. First match wins; no match leaves inputs unchanged.
    """
    if not context:
        return model, runtime_kwargs
    routes = _normalize_routes(model_config)
    if not routes:
        return model, runtime_kwargs

    for entry in routes:
        match_spec = entry.get("match")
        if not isinstance(match_spec, dict):
            continue
        if not _route_matches(match_spec, context):
            continue

        new_model = model
        new_runtime = dict(runtime_kwargs or {})
        applied = []

        if entry.get("model"):
            new_model = str(entry["model"])
            applied.append("model")
        for key in _OVERRIDE_RUNTIME_KEYS:
            val = entry.get(key)
            if val in (None, "", []):
                continue
            if key == "args":
                new_runtime[key] = list(val)
            else:
                new_runtime[key] = val
            applied.append(key)

        if applied:
            logger.info(
                "model.routes matched: context=%s fields=%s model=%s provider=%s",
                context, applied, new_model, new_runtime.get("provider"),
            )
        return new_model, new_runtime

    return model, runtime_kwargs
