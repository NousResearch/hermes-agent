"""Copy the model and its selected custom endpoint into a new profile, and merge that endpoint
over same-identity entries the new profile already has."""

from hermes_cli.config import _expand_env_vars, find_provider_entry
from hermes_cli.config_providers import _normalize_custom_provider_entry
from hermes_cli.providers import custom_provider_aliases
from hermes_cli.runtime_provider_custom import (
    _get_named_custom_provider,
    _normalize_custom_provider_name,
    _shadowed_by_builtin,
)


def _requested_provider(model_cfg) -> str | None:
    provider = model_cfg.get("provider") if isinstance(model_cfg, dict) else None
    # Runtime expands ``model.provider: ${VAR}`` before resolving; select with the same value.
    return str(_expand_env_vars(provider)) if isinstance(provider, str) else None


# Fields name lookups match on. Runtime sees them ``${VAR}``-expanded (load_config expands the
# whole file), so matching expands only these; copied entries stay raw and secrets stay refs.
_ROUTING_FIELDS = ("name", "provider_key", "base_url", "baseUrl", "url", "api", "enabled")


def _expand_routing_fields(entry):
    if not isinstance(entry, dict):
        return entry
    return {k: _expand_env_vars(v) if k in _ROUTING_FIELDS else v for k, v in entry.items()}


def _routing_view(cfg: dict) -> dict:
    """*cfg* with endpoint routing fields expanded; same keys and list order, so a key or index
    found here addresses the raw entry in *cfg*."""
    view = dict(cfg)
    providers = cfg.get("providers")
    if isinstance(providers, dict):
        view["providers"] = {k: _expand_routing_fields(v) for k, v in providers.items()}
    custom = cfg.get("custom_providers")
    if isinstance(custom, list):
        view["custom_providers"] = [_expand_routing_fields(e) for e in custom]
    return view


def _resolve(provider: str, view: dict):
    """Runtime named-endpoint lookup on a routing *view* without reading any credential."""
    return _get_named_custom_provider(provider, config=view, resolve_credentials=False)


def _source_of(cfg: dict, resolved: dict):
    """Where in *cfg* the resolver found *resolved*: ``("providers", key)`` or
    ``("custom_providers", index)``. Keyed results carry ``provider_key``; legacy results do not
    (their normalizer runs without one), so they are located by normalized name and URL."""
    key, entry = find_provider_entry(cfg.get("providers"), resolved.get("provider_key"))
    if entry is not None:
        return "providers", key
    custom = cfg.get("custom_providers")
    for index, entry in enumerate(custom if isinstance(custom, list) else ()):
        normalized = _normalize_custom_provider_entry(entry)
        if normalized and (normalized["name"], normalized["base_url"]) == (
            resolved["name"],
            resolved["base_url"],
        ):
            return "custom_providers", index
    return None


def launch_model_seed(source_cfg: dict) -> dict:
    """The config a fresh profile needs to run the launch profile's model: its ``model`` block plus
    the endpoint definition it points at. ``model.provider: custom:lab`` alone is "Unknown provider"
    on the first turn. The selected ``providers:`` or ``custom_providers:`` entry is copied as
    written (``${VAR}`` refs stay refs), and an exact ``providers.<built-in>`` / ``providers.custom``
    key travels for its tuning even without a URL. ``{}`` when the launch profile has no model."""
    model_cfg = source_cfg.get("model")
    if not model_cfg:
        return {}
    seed = {"model": model_cfg}
    provider = _requested_provider(model_cfg)
    if provider is None:
        return seed
    requested = _normalize_custom_provider_name(provider)
    providers = source_cfg.get("providers")
    providers = providers if isinstance(providers, dict) else {}
    # Resolve and locate on ONE view so the found key/index addresses the raw entry.
    view = _routing_view(source_cfg)
    selected = _resolve(provider, view)
    if selected is None:
        exact = providers.get(provider)
        # Bare ``custom`` runs as provider ``custom`` and reads ``providers.custom`` tuning too.
        if (
            isinstance(exact, dict)
            and requested not in {"", "auto"}
            and (requested == "custom" or _shadowed_by_builtin(requested))
        ):
            seed["providers"] = {provider: exact}
        return seed
    found = _source_of(view, selected)
    if found is not None:
        section, where = found
        seed[section] = (
            {where: providers[where]}
            if section == "providers"
            else [source_cfg["custom_providers"][where]]
        )
    return seed


def _aliases(entry, key: str = "") -> frozenset[str]:
    """Every identity name-based lookups (runtime, ``/model`` pinning, pickers) accept for *entry*."""
    if not isinstance(entry, dict):
        return frozenset()
    entry = _expand_routing_fields(entry)
    key = key or str(entry.get("provider_key") or "")
    return custom_provider_aliases(str(entry.get("name") or key), key)


def merge_endpoint_seed(dst_cfg: dict, seed: dict) -> dict:
    """Return a copy of *dst_cfg* with the seeded endpoint written first; a same-key destination
    entry is replaced whole. When the seed is an endpoint (it resolves), destination entries that
    claim any of its identities are dropped in both sections: Hermes has several name lookups with
    different skip rules (URL validation, ``enabled``, keyed-first), so only removing every
    same-identity entry guarantees each of them lands on the inherited endpoint. A URL-less tuning
    seed (``providers.custom``) claims no identity and replaces only its own key. Dropped entries
    include URL-less or disabled ones sharing a name (e.g. ``providers.openrouter`` tuning under a
    custom endpoint named OpenRouter); this only runs for a new profile without a model."""
    seed_providers = seed.get("providers") or {}
    seed_custom = seed.get("custom_providers") or []
    provider = _requested_provider(seed.get("model"))
    claimed: frozenset[str] = frozenset()
    if provider and _resolve(provider, _routing_view(seed)) is not None:
        claimed = frozenset().union(
            *(_aliases(e, str(k)) for k, e in seed_providers.items()),
            *(_aliases(e) for e in seed_custom),
        )

    merged = dict(dst_cfg)
    dst_providers = dst_cfg.get("providers")
    if isinstance(dst_providers, dict) or seed_providers:
        kept = {
            k: v for k, v in (dst_providers if isinstance(dst_providers, dict) else {}).items()
            if k not in seed_providers and not claimed & _aliases(v, str(k))
        }
        merged["providers"] = {**seed_providers, **kept}
    dst_custom = dst_cfg.get("custom_providers")
    if isinstance(dst_custom, list) or seed_custom:
        merged["custom_providers"] = seed_custom + [
            e for e in (dst_custom if isinstance(dst_custom, list) else [])
            if not claimed & _aliases(e)
        ]
    return merged
