"""Provider removal: config references, owning schema, and credential pool in one operation.

Adding a provider is a guided flow that writes to several stores; removing one had no
equivalent. ``config.yaml`` can hold the same provider in two shapes (the legacy
``custom_providers`` list and the v12+ ``providers`` map), and other sections carry
*references* to it (``model``, ``fallback_providers``, ``auxiliary.<slot>``, ``moa``). The
interactive "Remove a saved custom provider" row only ever popped the legacy list, so a
migrated entry survived, and nothing cleaned the ``auth.json`` pool behind it.

This module is the shared implementation behind both the interactive row and
``hermes provider remove``. It never writes on a dry run, never emits secret material, and
refuses to leave a dangling ``model.provider`` behind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# Sections holding a direct provider *reference* (not a provider definition), as dotted paths.
# ``<slot>`` is expanded over the auxiliary task names; ``[]`` marks a list of mappings.
REFERENCE_PATHS: tuple[str, ...] = (
    "model.provider",
    "fallback_providers[].provider",
    "auxiliary.<slot>.provider",
    "moa.aggregator.provider",
    "moa.reference_models[].provider",
)


@dataclass
class Reference:
    """One provider reference found in config, and what removal did to it."""

    path: str
    provider: str
    action: str = "pending"  # pending | reassigned | cleared | kept
    to: str = ""

    def describe(self) -> str:
        if self.action == "reassigned":
            return f"{self.path}: {self.provider} -> {self.to}"
        if self.action == "cleared":
            return f"{self.path}: {self.provider} (cleared)"
        return f"{self.path}: {self.provider}"


@dataclass
class RemovalPlan:
    """What a removal would do. Built without writing anything."""

    provider: str
    found: bool = False
    # Config schemas the definition actually lives in (never both, unless duplicated).
    legacy_index: Optional[int] = None
    providers_key: Optional[str] = None
    credential_pool_keys: list[str] = field(default_factory=list)
    exposed_env_vars: list[str] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)

    @property
    def stores(self) -> list[str]:
        stores: list[str] = []
        if self.legacy_index is not None:
            stores.append("custom_providers")
        if self.providers_key is not None:
            stores.append("providers")
        if self.credential_pool_keys:
            stores.append("auth.json")
        return stores


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _coerce_slot_names(config: dict[str, Any]) -> list[str]:
    auxiliary = config.get("auxiliary")
    if not isinstance(auxiliary, dict):
        return []
    return [str(slot) for slot, value in auxiliary.items() if isinstance(value, dict)]


def _iter_reference_targets(config: dict[str, Any]) -> list[tuple[str, str]]:
    """Yield ``(dotted_path, provider_value)`` for every configured provider reference."""
    targets: list[tuple[str, str]] = []

    model = config.get("model")
    if isinstance(model, dict):
        targets.append(("model.provider", str(model.get("provider") or "")))

    fallbacks = config.get("fallback_providers")
    if isinstance(fallbacks, list):
        for i, entry in enumerate(fallbacks):
            if isinstance(entry, dict):
                targets.append((f"fallback_providers[{i}].provider", str(entry.get("provider") or "")))

    for slot in _coerce_slot_names(config):
        entry = config["auxiliary"][slot]
        targets.append((f"auxiliary.{slot}.provider", str(entry.get("provider") or "")))

    moa = config.get("moa")
    if isinstance(moa, dict):
        aggregator = moa.get("aggregator")
        if isinstance(aggregator, dict):
            targets.append(("moa.aggregator.provider", str(aggregator.get("provider") or "")))
        references = moa.get("reference_models")
        if isinstance(references, list):
            for i, entry in enumerate(references):
                if isinstance(entry, dict):
                    targets.append((f"moa.reference_models[{i}].provider", str(entry.get("provider") or "")))

    return targets


def _resolves_to(provider_value: str, target: str) -> bool:
    """Does *provider_value* name the provider *target*?

    Matches the raw key, the legacy ``custom:<name>`` pool identity, and the same
    normalized form, so ``cerebras``, ``Cerebras`` and ``custom:cerebras`` all resolve.
    """
    candidate = _norm(provider_value)
    target_norm = _norm(target)
    if not candidate or not target_norm:
        return False
    if candidate == target_norm:
        return True
    # ``custom:<name>`` and the bare slug are the same provider.
    prefix = "custom:"
    values = {candidate, target_norm}
    stripped = {v[len(prefix):] for v in values if v.startswith(prefix)}
    plain = {v for v in values if not v.startswith(prefix)}
    return bool(stripped & plain)


def _credential_pool_keys(config: dict[str, Any], target: str) -> list[str]:
    """Pool keys in ``auth.json`` that belong to *target*, using the canonical resolver."""
    try:
        from agent.credential_pool import CUSTOM_POOL_PREFIX, _normalize_custom_pool_name
        from hermes_cli.config import get_compatible_custom_providers
    except Exception:
        return []

    keys: list[str] = []
    target_norm = _norm(target)
    for entry in get_compatible_custom_providers(config) or []:
        if not isinstance(entry, dict):
            continue
        names = {_norm(entry.get("name")), _norm(entry.get("provider_key"))}
        names.discard("")
        if target_norm not in names and not (target_norm.startswith(CUSTOM_POOL_PREFIX)
                                             and target_norm[len(CUSTOM_POOL_PREFIX):] in names):
            continue
        provider_key = _normalize_custom_pool_name(str(entry.get("provider_key") or ""))
        name = _normalize_custom_pool_name(str(entry.get("name") or ""))
        for key in (provider_key, f"{CUSTOM_POOL_PREFIX}{name}" if name else ""):
            if key and key not in keys:
                keys.append(key)
    # A bare target may itself be the pool key.
    for key in (target_norm, f"{CUSTOM_POOL_PREFIX}{_normalize_custom_pool_name(target)}"):
        if key and key not in keys:
            keys.append(key)
    return keys


def _pool_existing_keys(candidate_keys: list[str]) -> list[str]:
    """Narrow *candidate_keys* to those actually present in ``auth.json``."""
    try:
        from hermes_cli.auth import _load_auth_store
        store = _load_auth_store() or {}
    except Exception:
        return []
    pool = store.get("credential_pool")
    if not isinstance(pool, dict):
        return []
    return [key for key in candidate_keys if isinstance(pool.get(key), list) and pool.get(key)]


def _exposed_env_vars(config: dict[str, Any], target: str, plan: RemovalPlan) -> list[str]:
    """Env var names declared by the provider *definition* (names only, never values)."""
    names: list[str] = []
    providers = config.get("providers")
    keys = [plan.providers_key] if plan.providers_key else []
    if isinstance(providers, dict) and _norm(target) in {_norm(k) for k in providers}:
        keys.append(_norm(target))
    for key in filter(None, keys):
        entry = providers.get(key) if isinstance(providers, dict) else None
        if isinstance(entry, dict):
            env_var = str(entry.get("key_env") or "").strip()
            if env_var and env_var not in names:
                names.append(env_var)
    if plan.legacy_index is not None:
        legacy = config.get("custom_providers")
        entry = legacy[plan.legacy_index] if isinstance(legacy, list) else None
        if isinstance(entry, dict):
            env_var = str(entry.get("key_env") or "").strip()
            if env_var and env_var not in names:
                names.append(env_var)
    return names


def build_removal_plan(config: dict[str, Any], target: str) -> RemovalPlan:
    """Locate *target* across every store and collect its references. Writes nothing."""
    plan = RemovalPlan(provider=target)
    target_norm = _norm(target)

    legacy = config.get("custom_providers")
    if isinstance(legacy, list):
        for i, entry in enumerate(legacy):
            if isinstance(entry, dict) and _norm(entry.get("name")) == target_norm:
                plan.legacy_index = i
                plan.found = True
                break

    providers = config.get("providers")
    if isinstance(providers, dict):
        for key in providers:
            if _norm(key) == target_norm:
                plan.providers_key = str(key)
                plan.found = True
                break

    if not plan.found and target_norm.startswith("custom:"):
        bare = target_norm[len("custom:"):]
        if isinstance(legacy, list):
            for i, entry in enumerate(legacy):
                if isinstance(entry, dict) and _norm(entry.get("name")) == bare:
                    plan.legacy_index = i
                    plan.found = True
                    break
        if not plan.found and isinstance(providers, dict):
            for key in providers:
                if _norm(key) == bare:
                    plan.providers_key = str(key)
                    plan.found = True
                    break

    for path, value in _iter_reference_targets(config):
        if _resolves_to(value, target):
            plan.references.append(Reference(path=path, provider=value))

    plan.credential_pool_keys = _pool_existing_keys(_credential_pool_keys(config, target))
    plan.exposed_env_vars = _exposed_env_vars(config, target, plan)
    return plan


def _apply_to_reference(config: dict[str, Any], ref: Reference, action: str, to: str = "") -> None:
    """Rewrite one reference in *config*; ``action`` is ``reassigned`` or ``cleared``."""
    path = ref.path
    new_value: Any = to if action == "reassigned" else ""

    if path == "model.provider":
        model = config.get("model")
        if isinstance(model, dict):
            model["provider"] = new_value
            # A reassignment or clear must not leave the removed provider's endpoint behind:
            # base_url is part of the provider's identity, so a stale one sends every request
            # to a dead host.
            model.pop("base_url", None)
        ref.action = action
        ref.to = to
        return

    if path.startswith("fallback_providers[") and path.endswith("].provider"):
        index = int(path[len("fallback_providers["):-len("].provider")])
        fallbacks = config.get("fallback_providers")
        if isinstance(fallbacks, list) and 0 <= index < len(fallbacks) and isinstance(fallbacks[index], dict):
            if action == "reassigned":
                fallbacks[index]["provider"] = new_value
                ref.action, ref.to = action, to
            else:
                fallbacks.pop(index)
                ref.action, ref.to = action, ""
        return

    if path.startswith("auxiliary.") and path.endswith(".provider"):
        slot = path[len("auxiliary."):-len(".provider")]
        auxiliary = config.get("auxiliary")
        if isinstance(auxiliary, dict) and isinstance(auxiliary.get(slot), dict):
            auxiliary[slot]["provider"] = new_value
            ref.action, ref.to = action, to
        return

    if path == "moa.aggregator.provider":
        aggregator = config.get("moa", {}).get("aggregator") if isinstance(config.get("moa"), dict) else None
        if isinstance(aggregator, dict):
            aggregator["provider"] = new_value
            ref.action, ref.to = action, to
        return

    if path.startswith("moa.reference_models[") and path.endswith("].provider"):
        index = int(path[len("moa.reference_models["):-len("].provider")])
        moa = config.get("moa")
        references = moa.get("reference_models") if isinstance(moa, dict) else None
        if isinstance(references, list) and 0 <= index < len(references) and isinstance(references[index], dict):
            if action == "reassigned":
                references[index]["provider"] = new_value
                ref.action, ref.to = action, to
            else:
                references.pop(index)
                ref.action, ref.to = action, ""
        return


def _purge_definition(config: dict[str, Any], plan: RemovalPlan) -> list[str]:
    """Drop the provider definition from its owning schema(s). Returns stores touched."""
    touched: list[str] = []

    if plan.providers_key is not None:
        providers = config.get("providers")
        if isinstance(providers, dict) and plan.providers_key in providers:
            del providers[plan.providers_key]
            touched.append("providers")

    if plan.legacy_index is not None:
        legacy = config.get("custom_providers")
        if isinstance(legacy, list) and 0 <= plan.legacy_index < len(legacy):
            entry = legacy[plan.legacy_index]
            if isinstance(entry, dict) and _norm(entry.get("name")) == _norm(plan.provider):
                legacy.pop(plan.legacy_index)
            elif isinstance(entry, dict):
                # Positions may have shifted after clearing list-backed references.
                for i, candidate in enumerate(legacy):
                    if isinstance(candidate, dict) and _norm(candidate.get("name")) == _norm(plan.provider):
                        legacy.pop(i)
                        break
                else:
                    return touched
            touched.append("custom_providers")
            if not legacy:
                config.pop("custom_providers", None)

    return touched


def _prune_credential_pools(keys: list[str]) -> dict[str, int]:
    """Delete the given pool keys from ``auth.json``. Returns key -> entries removed."""
    if not keys:
        return {}
    try:
        from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store
    except Exception:
        return {}

    removed: dict[str, int] = {}
    with _auth_store_lock():
        store = _load_auth_store() or {}
        pool = store.get("credential_pool")
        if not isinstance(pool, dict):
            return {}
        for key in keys:
            entries = pool.pop(key, None)
            if isinstance(entries, list):
                removed[key] = len(entries)
        if removed:
            _save_auth_store(store)
    return removed


def execute_removal(
    config: dict[str, Any],
    plan: RemovalPlan,
    *,
    reassign: Optional[dict[str, str]] = None,
    clear: Optional[list[str]] = None,
    keep_credentials: bool = False,
) -> dict[str, Any]:
    """Apply *plan* to *config* in memory and prune the credential pools.

    Mutates *config*; the caller persists it. ``reassign`` maps a reference path to a
    replacement provider; ``clear`` names reference paths to drop outright. Any reference
    not covered by either keeps its original value (reported as ``kept``).
    """
    reassign = reassign or {}
    clear_paths = set(clear or [])

    for ref in plan.references:
        if ref.path in reassign:
            _apply_to_reference(config, ref, "reassigned", reassign[ref.path])
        elif ref.path in clear_paths:
            _apply_to_reference(config, ref, "cleared")
        else:
            ref.action = "kept"

    stores = _purge_definition(config, plan)
    credentials = {} if keep_credentials else _prune_credential_pools(plan.credential_pool_keys)

    return {
        "provider": plan.provider,
        "found": plan.found,
        "stores": stores,
        "credential_pools_removed": credentials,
        "credentials_removed": sum(credentials.values()),
        "env_vars": list(plan.exposed_env_vars),
        "references": [r.describe() for r in plan.references],
        "unresolved": [r.path for r in plan.references if r.action == "kept"],
    }


def format_plan(plan: RemovalPlan, *, dry_run: bool = False) -> list[str]:
    """Human-readable, secret-free rendering of a plan."""
    headline = "Would remove" if dry_run else "Removing"
    if not plan.found:
        return [f"Provider '{plan.provider}' is not configured."]

    lines = [f"{headline} provider '{plan.provider}'"]
    if plan.providers_key is not None:
        lines.append(f"  config.yaml: providers.{plan.providers_key}")
    if plan.legacy_index is not None:
        lines.append("  config.yaml: custom_providers[]")
    for key in plan.credential_pool_keys:
        lines.append(f"  auth.json:   credential_pool[{key}]")
    for env_var in plan.exposed_env_vars:
        lines.append(f"  env var:     {env_var} (name only)")
    if plan.references:
        lines.append("  references:")
        lines.extend(f"    {ref.path} -> {ref.provider}" for ref in plan.references)
    return lines

