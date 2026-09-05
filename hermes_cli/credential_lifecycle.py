"""Unified provider-credential lifecycle across every store Hermes reads.

Deleting a key from ``.env`` alone leaves the stale ``credential_pool`` entry (and the
``provider_models_cache.json`` row) behind, so the provider keeps appearing in the model picker
even across restarts (the pool loader is additive-only). Every surface that saves or removes a
provider credential should route through :func:`save_provider_env_credential` /
:func:`remove_provider_env_credential` so all stores stay consistent.
"""

from __future__ import annotations

from typing import Any, Dict, List

__all__ = [
    "save_provider_env_credential",
    "remove_provider_env_credential",
    "purge_env_credential_references"]


def _providers_for_env_var(env_var: str) -> List[str]:
    """Provider ids whose registered api_key_env_vars include ``env_var``."""
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
    except Exception:
        return []
    hits: List[str] = []
    for pid, cfg in PROVIDER_REGISTRY.items():
        try:
            if env_var in (cfg.api_key_env_vars or ()):
                hits.append(pid)
        except Exception:
            continue
    return hits


def _for_each_provider(providers: List[str], import_path: str, *args: Any) -> None:
    """Best-effort ``module.fn(provider, *args)`` for every provider; failures never propagate."""
    try:
        import importlib

        module_name, fn_name = import_path.rsplit(".", 1)
        fn = getattr(importlib.import_module(module_name), fn_name)
        for provider in providers:
            fn(provider, *args)
    except Exception:
        pass


def _prune_env_pool_entries(env_var: str) -> List[str]:
    """Drop ``credential_pool`` entries seeded from ``env:<env_var>``; return providers pruned.

    Spans ALL providers (shared vars like GITHUB_TOKEN seed several). Entries with any other
    source (OAuth, device-code, manual, borrowed-CLI) are preserved verbatim.
    """
    from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store

    source = f"env:{env_var}"
    pruned: List[str] = []
    with _auth_store_lock():
        auth_store = _load_auth_store()
        pool = auth_store.get("credential_pool")
        if not isinstance(pool, dict):
            return pruned
        for provider in list(pool.keys()):
            entries = pool[provider]
            if not isinstance(entries, list):
                continue
            kept = [e for e in entries if not (isinstance(e, dict) and e.get("source") == source)]
            if len(kept) == len(entries):
                continue
            pruned.append(provider)
            if kept:
                pool[provider] = kept
            else:
                del pool[provider]
        if pruned:
            _save_auth_store(auth_store)
    return pruned


def _scrub_config_yaml_mirrors(old_value: str, new_value: str | None) -> List[str]:
    """Reconcile config.yaml api_key mirrors holding ``old_value``; return dotted paths touched.

    Value-matched on purpose: only an entry holding the SAME credential that just changed in
    ``.env`` is touched. ``new_value=None`` removes the field. Operates on the RAW user config
    so defaults are never baked into the user's file.
    """
    if not old_value:
        return []
    from utils import atomic_yaml_write, fast_safe_load

    from hermes_cli.config import get_config_path, require_readable_config_before_write

    config_path = get_config_path()
    if not config_path.exists():
        return []
    try:
        with open(config_path, encoding="utf-8") as f:
            user_config = fast_safe_load(f) or {}
    except Exception:
        return []
    if not isinstance(user_config, dict):
        return []

    touched: List[str] = []

    def _fix(section: Any, key_path: str, fields: tuple[str, ...] = ("api_key", "api")) -> None:
        # "api" is the legacy alias for model.api_key in older configs. In the keyed ``providers``
        # schema ``api`` means the base_url, not a credential, so that section passes
        # ``fields=("api_key",)``.
        if not isinstance(section, dict):
            return
        for field in fields:
            current = section.get(field)
            if isinstance(current, str) and current == old_value:
                if new_value:
                    section[field] = new_value
                else:
                    section.pop(field, None)
                touched.append(f"{key_path}.{field}")

    def _items(value: Any, allow_list: bool):
        if isinstance(value, dict):
            return value.items()
        return enumerate(value) if allow_list and isinstance(value, list) else ()

    _fix(user_config.get("model"), "model")
    for task, slot_cfg in _items(user_config.get("auxiliary"), False):
        _fix(slot_cfg, f"auxiliary.{task}")
    for name, entry in _items(user_config.get("custom_providers"), True):
        _fix(entry, f"custom_providers.{name}")

    # ``providers.<id>.api_key`` (v12+) is where dashboard/desktop write custom-endpoint
    # credentials. It is a real inline secret with higher precedence than the env var, so a stale
    # copy shadows a rotation (persistent 401 with a key the UI no longer shows) and survives a
    # removal that promised to clear EVERY store.
    for provider_id, entry in _items(user_config.get("providers"), False):
        _fix(entry, f"providers.{provider_id}", fields=("api_key",))

    if touched:
        require_readable_config_before_write(config_path)
        atomic_yaml_write(config_path, user_config, sort_keys=False)
    return touched


def purge_env_credential_references(
    env_var: str, *, clear_models_cache: bool = True) -> Dict[str, Any]:
    """Remove non-.env references to an env-var credential.

    Prunes env-seeded pool entries and (optionally) the affected ``provider_models_cache.json`` rows
    so the model picker stops advertising a provider whose key is gone.

    See #59761.
    """
    pruned = _prune_env_pool_entries(env_var)
    providers = sorted(set(pruned) | set(_providers_for_env_var(env_var)))
    # Make the removal sticky the same way `hermes auth remove` does: a lingering shell export (or
    # another live process's os.environ) would otherwise re-seed the pool entry on the next
    # load_pool(). The save path lifts the suppression on an explicit re-add.
    _for_each_provider(providers, "hermes_cli.auth.suppress_credential_source", f"env:{env_var}")
    if clear_models_cache and providers:
        # Best-effort — a cache failure must not block the credential removal itself.
        _for_each_provider(providers, "hermes_cli.models.clear_provider_models_cache")
    return {"pool_pruned": pruned, "providers": providers}


def save_provider_env_credential(env_var: str, value: str) -> Dict[str, Any]:
    """Save/update a credential in ``.env`` and reconcile every mirror.

    config.yaml mirrors of the PREVIOUS value are updated so a stale higher-precedence copy cannot
    shadow the rotation, and ``load_pool()`` runs now so the env-seeded ``credential_pool`` entry
    lands in ``auth.json`` (a ``.env``-only write left env-backed providers 401'ing).

    Suppressed ``env:<VAR>`` pool sources are re-enabled so a deliberate re-add through the UI behaves like
    ``hermes auth add``. See #62269.
    The save also forces an immediate ``load_pool()`` for every provider registered against this env var so
    the env-seeded ``credential_pool`` entry is materialized to ``auth.json`` right now — the live runtime
    reads from the pool, and before #96058 the Desktop "Save" action only touched ``.env`` while
    ``auth.json``'s mtime stayed unchanged, so an OpenCode Go (or any other env-backed provider) request
    kept 401'ing until the user ran ``hermes auth add <provider> --type api-key`` separately. This makes the
    Desktop save's effect on disk match what ``hermes auth add`` does.
    """
    from hermes_cli.config import load_env, save_env_value

    old_value = load_env().get(env_var)
    save_env_value(env_var, value)

    config_updates: List[str] = []
    if value and old_value and old_value != value:
        config_updates = _scrub_config_yaml_mirrors(old_value, value)

    # A prior removal may have suppressed this env source; a fresh save is an explicit re-add.
    providers = _providers_for_env_var(env_var)
    _for_each_provider(providers, "hermes_cli.auth.unsuppress_credential_source", f"env:{env_var}")

    # ``load_pool`` is idempotent and additive-only for env sources, so re-running is safe even when
    # the pool already had this entry. Best-effort: never masks the successful .env write above.
    _for_each_provider(providers, "agent.credential_pool.load_pool")

    return {"ok": True, "key": env_var, "config_updates": config_updates}


def remove_provider_env_credential(env_var: str) -> Dict[str, Any]:
    """Remove a credential from EVERY store: ``.env`` (and process env), env-seeded
    ``credential_pool`` entries, model-cache rows, config.yaml mirrors of the same value."""
    from hermes_cli.config import load_env, remove_env_value

    old_value = load_env().get(env_var)
    removed_from_env = remove_env_value(env_var)
    refs = purge_env_credential_references(env_var)
    config_scrubbed = _scrub_config_yaml_mirrors(old_value, None) if old_value else []

    return {
        "ok": True,
        "key": env_var,
        "removed": removed_from_env,
        "pool_pruned": refs["pool_pruned"],
        "providers": refs["providers"],
        "config_scrubbed": config_scrubbed,
        "found": bool(removed_from_env or refs["pool_pruned"] or config_scrubbed)}
