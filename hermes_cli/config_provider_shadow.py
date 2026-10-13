"""Detect ``providers:`` / ``custom_providers:`` entries that a built-in provider shadows (#43026).

``runtime_provider_custom._shadowed_by_builtin`` deliberately ignores user entries named after a
canonical built-in provider id when they are referenced by raw name, so their ``base_url`` /
``api_key`` silently do nothing and the built-in's default endpoint + env-var credentials are used.
This sibling makes that ignore loud: ``validate_config_structure`` (hermes doctor) and the runtime
resolver both call :func:`find_shadowed_builtin_provider_entries`.

Split out of :mod:`hermes_cli.config` (code-health size ratchet); callers late-import it.
"""

from __future__ import annotations

from typing import Any, Optional

# Fields that make a ``providers.<id>`` entry a *routing* definition (endpoint / credentials)
# rather than the documented per-provider timeout tuning schema (request_timeout_seconds /
# stale_timeout_seconds / models), which is a valid use of a built-in provider's id.
_PROVIDER_ROUTING_FIELDS = {
    "base_url", "api", "url", "api_key", "key_env", "api_key_env",
    "api_mode", "transport", "model", "default_model",
}

# Canonical built-in provider ids that are deliberately NOT keys of ``auth.PROVIDER_REGISTRY``.
# ``auth.py`` keeps ``openrouter`` out on purpose (runtime_provider resolution relies on
# ``openrouter not in PROVIDER_REGISTRY``), so registry membership alone would stop flagging a
# ``providers.openrouter`` entry that the runtime still ignores. ``custom`` is excluded instead —
# it is the explicit user-supplied escape hatch, not something that can be shadowed.
_BUILTIN_PROVIDERS_OUTSIDE_REGISTRY = frozenset({"openrouter"})


def _normalize(name: Any) -> str:
    return str(name or "").strip().lower().replace(" ", "-")


def _canonical_shadow(name: Any, registry: dict[str, Any]) -> Optional[str]:
    """``name`` when it IS a canonical built-in id; aliases (``kimi`` -> ``kimi-coding``) are
    registered as extra registry keys pointing at the canonical entry and are honored by the
    runtime as custom-provider names, so they never count as shadowing."""
    norm = _normalize(name)
    if not norm or norm in {"custom", "auto"}:
        return None
    if norm in _BUILTIN_PROVIDERS_OUTSIDE_REGISTRY:
        return norm
    entry = registry.get(norm)
    if entry is None:
        return None
    return norm if _normalize(getattr(entry, "id", "")) == norm else None


def find_shadowed_builtin_provider_entries(config: Optional[dict[str, Any]] = None) -> list[str]:
    """Return canonical built-in provider ids that user config tries to redefine.

    The built-in check is a static lookup against ``auth.PROVIDER_REGISTRY`` and must stay that
    way: ``resolve_provider()`` calls ``validate_config_structure()`` back to build its
    unknown-provider hint (``auth._get_config_hint_for_unknown_provider``), so asking it here
    would close an unbounded mutual recursion.
    """
    if config is None:
        from hermes_cli.config import load_config_readonly

        try:
            config = load_config_readonly()
        except (OSError, RuntimeError):  # unreadable home/config: nothing to inspect
            return []
    from hermes_cli.auth import PROVIDER_REGISTRY

    names: list[Any] = []
    providers = config.get("providers")
    if isinstance(providers, dict):
        names += [key for key, entry in providers.items()
                  if isinstance(entry, dict) and _PROVIDER_ROUTING_FIELDS & set(entry)]
    custom_providers = config.get("custom_providers")
    if isinstance(custom_providers, list):
        names += [entry.get("name") for entry in custom_providers if isinstance(entry, dict)]

    shadowed: list[str] = []
    for name in names:
        hit = _canonical_shadow(name, PROVIDER_REGISTRY)
        if hit and hit not in shadowed:
            shadowed.append(hit)
    return shadowed


def _referenced_provider_names(config: dict[str, Any]) -> set[str]:
    """Provider names the config routes to by raw name (``model.provider`` / ``fallback_model``)."""
    model_cfg = config.get("model")
    fb_cfg = config.get("fallback_model")
    entries = [model_cfg] + (fb_cfg if isinstance(fb_cfg, list) else [fb_cfg])
    names = {_normalize(e.get("provider")) for e in entries if isinstance(e, dict)}
    names.discard("")
    return names


def validate_shadowed_builtin_providers(config: dict[str, Any], issues: list) -> None:
    """Flag shadowed entries the config actually routes to by raw name.

    Entries selected via an explicit ``custom:<name>`` menu key are still honored by the runtime,
    so only names referenced from ``model.provider`` / ``fallback_model`` providers are flagged.
    """
    from hermes_cli.config import _issue

    referenced = _referenced_provider_names(config)
    for name in find_shadowed_builtin_provider_entries(config):
        if name not in referenced:
            continue
        _issue(issues, "warning",
               f"providers/custom_providers entry '{name}' shadows the built-in '{name}' "
               f"provider — its base_url/api_key are ignored and the built-in's default "
               f"endpoint + environment credentials are used instead",
               f"Rename the entry to a non-built-in name (e.g. '{name}-custom') and set "
               f"model.provider to that name, or configure the built-in provider through "
               f"its environment variables / 'hermes auth add {name}' and remove the entry")
