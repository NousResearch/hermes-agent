"""Source suppression and coordinated credential lifecycle policy."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional
from auth import store
from auth.context import CredentialScope
from hermes_constants import get_hermes_home


def _suppressed_source_list(
    suppressed: Dict[str, Any], provider_id: str
) -> Optional[List[str]]:
    """Canonical (list-form) suppressed sources for *provider_id*; a legacy mapping (keys = source
    names) is migrated to the list form in place."""
    raw_sources = suppressed.get(provider_id)
    if isinstance(raw_sources, list):
        return raw_sources
    if isinstance(raw_sources, dict):
        suppressed[provider_id] = [str(name) for name in raw_sources]
        return suppressed[provider_id]
    return None


def suppress_credential_source(provider_id: str, source: str) -> None:
    """Mark a credential source as suppressed so it won't be re-seeded."""
    with store._auth_store_lock():
        auth_store = store._load_auth_store()
        suppressed = store._store_section(auth_store, "suppressed_sources")
        provider_list = _suppressed_source_list(suppressed, provider_id)
        if provider_list is None:
            provider_list = suppressed[provider_id] = []
        if source not in provider_list:
            provider_list.append(source)
        store._save_auth_store(auth_store)


def is_source_suppressed(provider_id: str, source: str) -> bool:
    """Check if a credential source has been suppressed by the user."""
    try:
        return source in store._load_auth_store().get("suppressed_sources", {}).get(
            provider_id, []
        )
    except Exception:
        return False


def unsuppress_credential_source(provider_id: str, source: str) -> bool:
    """Clear a suppression marker so the source will be re-seeded on the next load."""
    with store._auth_store_lock():
        auth_store = store._load_auth_store()
        suppressed = auth_store.get("suppressed_sources")
        if not isinstance(suppressed, dict):
            return False
        provider_list = _suppressed_source_list(suppressed, provider_id)
        if provider_list is None or source not in provider_list:
            return False
        provider_list.remove(source)
        if not provider_list:
            suppressed.pop(provider_id, None)
        if not suppressed:
            auth_store.pop("suppressed_sources", None)
        store._save_auth_store(auth_store)
        return True


@dataclass(frozen=True)
class CredentialEnvironment:
    """Explicit application operations, bound to the current execution profile."""

    scope: CredentialScope
    read_env: Callable[[], Mapping[str, str]] = field(repr=False)
    save_env: Callable[[str, str], Any] = field(repr=False)
    remove_env: Callable[[str], bool] = field(repr=False)
    reconcile_mirrors: Callable[[str, str | None], List[str]] = field(repr=False)
    providers_for_env: Callable[[str], List[str]] = field(repr=False)
    clear_models_cache: Callable[[str], Any] = field(repr=False)
    seed_pool: Callable[[str], Any] = field(repr=False)

    def require_current_scope(self) -> None:
        if self.scope != CredentialScope(get_hermes_home()):
            raise ValueError("Credential environment belongs to another profile")


def _for_each_provider(
    providers: List[str], fn: Callable[..., Any], *args: Any
) -> None:
    """Best-effort collaborator calls; preserve existing failure isolation."""
    try:
        for provider in providers:
            fn(provider, *args)
    except Exception:
        pass


def _prune_env_pool_entries(env_var: str) -> List[str]:
    """Drop ``credential_pool`` entries seeded from ``env:<env_var>``; return providers pruned.

    Spans ALL providers (shared vars like GITHUB_TOKEN seed several). Entries with any other
    source (OAuth, device-code, manual, borrowed-CLI) are preserved verbatim.
    """

    source = f"env:{env_var}"
    pruned: List[str] = []
    with store._auth_store_lock():
        auth_store = store._load_auth_store()
        pool = auth_store.get("credential_pool")
        if not isinstance(pool, dict):
            return pruned
        for provider in list(pool.keys()):
            entries = pool[provider]
            if not isinstance(entries, list):
                continue
            kept = [
                e
                for e in entries
                if not (isinstance(e, dict) and e.get("source") == source)
            ]
            if len(kept) == len(entries):
                continue
            pruned.append(provider)
            if kept:
                pool[provider] = kept
            else:
                del pool[provider]
        if pruned:
            store._save_auth_store(auth_store)
    return pruned


def purge_env_credential_references(
    env_var: str, *, environment: CredentialEnvironment, clear_models_cache: bool = True
) -> Dict[str, Any]:
    """Remove non-.env references to an env-var credential.

    Prunes env-seeded pool entries and (optionally) the affected ``provider_models_cache.json`` rows
    so the model picker stops advertising a provider whose key is gone.

    See #59761.
    """
    environment.require_current_scope()
    pruned = _prune_env_pool_entries(env_var)
    providers = sorted(set(pruned) | set(environment.providers_for_env(env_var)))
    # Make the removal sticky the same way `hermes auth remove` does: a lingering shell export (or
    # another live process's os.environ) would otherwise re-seed the pool entry on the next
    # load_pool(). The save path lifts the suppression on an explicit re-add.
    _for_each_provider(providers, suppress_credential_source, f"env:{env_var}")
    if clear_models_cache and providers:
        # Best-effort — a cache failure must not block the credential removal itself.
        _for_each_provider(providers, environment.clear_models_cache)
    return {"pool_pruned": pruned, "providers": providers}


def save_provider_env_credential(
    env_var: str, value: str, *, environment: CredentialEnvironment
) -> Dict[str, Any]:
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
    environment.require_current_scope()

    old_value = environment.read_env().get(env_var)
    environment.save_env(env_var, value)

    config_updates: List[str] = []
    if value and old_value and old_value != value:
        config_updates = environment.reconcile_mirrors(old_value, value)

    # A prior removal may have suppressed this env source; a fresh save is an explicit re-add.
    providers = environment.providers_for_env(env_var)
    _for_each_provider(providers, unsuppress_credential_source, f"env:{env_var}")

    # ``load_pool`` is idempotent and additive-only for env sources, so re-running is safe even when
    # the pool already had this entry. Best-effort: never masks the successful .env write above.
    _for_each_provider(providers, environment.seed_pool)

    return {"ok": True, "key": env_var, "config_updates": config_updates}


def remove_provider_env_credential(
    env_var: str, *, environment: CredentialEnvironment
) -> Dict[str, Any]:
    """Remove a credential from EVERY store: ``.env`` (and process env), env-seeded
    ``credential_pool`` entries, model-cache rows, config.yaml mirrors of the same value."""
    environment.require_current_scope()

    old_value = environment.read_env().get(env_var)
    removed_from_env = environment.remove_env(env_var)
    refs = purge_env_credential_references(env_var, environment=environment)
    config_scrubbed = (
        environment.reconcile_mirrors(old_value, None) if old_value else []
    )

    return {
        "ok": True,
        "key": env_var,
        "removed": removed_from_env,
        "pool_pruned": refs["pool_pruned"],
        "providers": refs["providers"],
        "config_scrubbed": config_scrubbed,
        "found": bool(removed_from_env or refs["pool_pruned"] or config_scrubbed),
    }


def unsuppress_provider_sources(provider: str) -> None:
    """Clear ALL suppressions for this provider — re-adding a credential is a strong signal the
    user wants auth re-enabled. Covers env:* (shell-exported vars), gh_cli (copilot), claude_code,
    qwen-cli, device_code (codex), etc. — one consistent re-engagement pattern."""
    try:
        suppressed = store._load_auth_store().get("suppressed_sources", {})
        for src in list(suppressed.get(provider, []) or []):
            unsuppress_credential_source(provider, src)
    except Exception:
        pass
