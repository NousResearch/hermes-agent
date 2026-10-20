"""Unified provider-credential lifecycle across every store Hermes reads.

Deleting a key from ``.env`` alone leaves the stale ``credential_pool`` entry (and the
``provider_models_cache.json`` row) behind, so the provider keeps appearing in the model picker
even across restarts (the pool loader is additive-only). Every surface that saves or removes a
provider credential should route through :func:`save_provider_env_credential` /
:func:`remove_provider_env_credential` so all stores stay consistent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

__all__ = [
    "propagate_env_credential_to_siblings",
    "purge_env_credential_references",
    "remove_provider_env_credential",
    "save_provider_env_credential",
    "sibling_profiles_with_env_var",
]

# Distinguishes "the value is the empty string" from "the key is absent".
_MISSING = object()


def _providers_for_env_var(env_var: str) -> list[str]:
    """Provider ids whose registered api_key_env_vars include ``env_var``."""
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
    except Exception:
        return []
    hits: list[str] = []
    for pid, cfg in PROVIDER_REGISTRY.items():
        try:
            if env_var in (cfg.api_key_env_vars or ()):
                hits.append(pid)
        except Exception:
            continue
    return hits


def _for_each_provider(providers: list[str], import_path: str, *args: Any) -> None:
    """Best-effort ``module.fn(provider, *args)`` for every provider; failures never propagate."""
    try:
        import importlib

        module_name, fn_name = import_path.rsplit(".", 1)
        fn = getattr(importlib.import_module(module_name), fn_name)
        for provider in providers:
            fn(provider, *args)
    except Exception:
        pass


def _prune_env_pool_entries(env_var: str) -> list[str]:
    """Drop ``credential_pool`` entries seeded from ``env:<env_var>``; return providers pruned.

    Spans ALL providers (shared vars like GITHUB_TOKEN seed several). Entries with any other
    source (OAuth, device-code, manual, borrowed-CLI) are preserved verbatim.
    """
    from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store

    source = f"env:{env_var}"
    pruned: list[str] = []
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


def _scrub_config_yaml_mirrors(old_value: str, new_value: str | None) -> list[str]:
    """Reconcile config.yaml api_key mirrors holding ``old_value``; return dotted paths touched.

    Value-matched on purpose: only an entry holding the SAME credential that just changed in
    ``.env`` is touched. ``new_value=None`` removes the field. Operates on the RAW user config
    so defaults are never baked into the user's file.
    """
    if not old_value:
        return []
    from hermes_cli.config import atomic_config_replace, get_config_path, read_user_config_raw

    config_path = get_config_path()
    if not config_path.exists():
        return []
    try:
        user_config = read_user_config_raw(config_path)
    except Exception:
        return []
    if not user_config:
        return []

    touched: list[str] = []

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
        atomic_config_replace(config_path, user_config)
    return touched


def purge_env_credential_references(
    env_var: str, *, clear_models_cache: bool = True) -> dict[str, Any]:
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


def save_provider_env_credential(env_var: str, value: str) -> dict[str, Any]:
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
    from hermes_cli.config import load_env, require_env_writable, save_env_value

    # A locked key must fail here: save_env_value's refusal returns like a success, and the mirror
    # scrub below would still move the new value into config.yaml.
    require_env_writable(env_var, "set")
    old_value = load_env().get(env_var)
    save_env_value(env_var, value)

    config_updates: list[str] = []
    if value and old_value and old_value != value:
        config_updates = _scrub_config_yaml_mirrors(old_value, value)

    # A prior removal may have suppressed this env source; a fresh save is an explicit re-add.
    providers = _providers_for_env_var(env_var)
    _for_each_provider(providers, "hermes_cli.auth.unsuppress_credential_source", f"env:{env_var}")

    # ``load_pool`` is idempotent and additive-only for env sources, so re-running is safe even when
    # the pool already had this entry. Best-effort: never masks the successful .env write above.
    _for_each_provider(providers, "agent.credential_pool.load_pool")

    return {"ok": True, "key": env_var, "config_updates": config_updates}


def remove_provider_env_credential(env_var: str) -> dict[str, Any]:
    """Remove a credential from EVERY store: ``.env`` (and process env), env-seeded
    ``credential_pool`` entries, model-cache rows, config.yaml mirrors of the same value."""
    from hermes_cli.config import load_env, remove_env_value, require_env_writable

    # Before the pool prune and mirror scrub: a refused remove must not strip the other stores.
    require_env_writable(env_var, "remove")
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


# ---------------------------------------------------------------------------
# Multi-profile fan-out
#
# Rotating a key through ``.env`` alone fixes ONE profile. Every sibling keeps
# the dead value and keeps 401'ing, so a rotation that "succeeded" leaves the
# install half-broken (real incident: 11 profiles, 10 stale copies, every agent
# 401'ing plus a memory backend that could no longer authenticate).
#
# The fan-out is deliberately narrow:
#   * sibling     — a real named-profile dir under <root>/profiles, never the
#                   active home, never a tombstoned (profiles/.deleted/<name>)
#                   or hidden entry, deduped by resolved path;
#   * already has — the profile's ``.env`` must ALREADY define the variable.
#                   Provisioning the key into a profile that deliberately never
#                   had it would hand a credential to a tenant that opted out.
#   * opt-in      — enumeration is read-only; nothing is written unless the
#                   caller asks for ``apply=True``.
# ---------------------------------------------------------------------------


def _canonical_home(path: str | Path) -> Path:
    """Best-effort absolute path for identity/dedup comparisons (never raises)."""
    try:
        return Path(path).expanduser().resolve(strict=False)
    except (OSError, RuntimeError):  # pragma: no cover - exotic filesystem
        return Path(path).expanduser()


def sibling_profiles_with_env_var(
    env_var: str, *, active_home: str | Path | None = None
) -> list[tuple[str, Path]]:
    """``(name, home)`` for every sibling profile whose ``.env`` defines ``env_var``.

    ``active_home`` defaults to the current (context-scoped) HERMES_HOME — the profile whose
    key just rotated. Read-only: this never creates a directory, writes a file, or drops a
    cache, so it is safe to call on every dashboard save to populate the opt-in prompt.
    """
    from hermes_cli.profiles import _PROFILE_ID_RE, _get_profiles_root
    from hermes_constants import (
        get_hermes_home,
        named_profile_has_identity,
        named_profile_is_deleted,
    )

    try:
        active = _canonical_home(active_home if active_home is not None else get_hermes_home())
    except Exception:
        return []
    try:
        root = _get_profiles_root()
        entries = sorted(root.iterdir()) if root.is_dir() else []
    except OSError:
        return []

    from agent.secret_scope import load_env_file

    seen: set[Path] = {active}
    found: list[tuple[str, Path]] = []
    for entry in entries:
        name = entry.name
        # ``default`` is the install root, not a sibling; hidden dirs (``.deleted``) are
        # tombstones/scratch; anything off the profile-id grammar is not a profile.
        if name == "default" or name.startswith(".") or not _PROFILE_ID_RE.match(name):
            continue
        try:
            if not entry.is_dir():
                continue
            if named_profile_is_deleted(entry) or not named_profile_has_identity(entry):
                continue
            resolved = _canonical_home(entry)
            if resolved in seen:
                continue
            if env_var not in load_env_file(entry / ".env"):
                continue
        except OSError:
            continue
        seen.add(resolved)
        found.append((name, entry))
    return found


def _propagation_row(
    name: str, home: Path, status: str, reason: str, *, stale: bool | None = None,
    config_updates: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "profile": name,
        "home": str(home),
        "status": status,
        "reason": reason,
        "stale": stale,
        "config_updates": list(config_updates or ()),
    }


def _apply_credential_to_home(env_var: str, value: str, home: Path) -> dict[str, Any]:
    """Write one sibling home's ``.env`` through the audited save path, then verify it.

    ``set_hermes_home_override`` re-points ``get_env_path()`` / ``get_config_path()`` for the
    block, so this is the SAME writer the active profile uses: atomic replace, identical
    quoting, the file's existing mode preserved, and the sibling's own ``config.yaml``
    mirrors of its previous value scrubbed by ``save_provider_env_credential``.
    """
    from agent.secret_scope import load_env_file
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    previous = load_env_file(home / ".env").get(env_var, _MISSING)
    token = set_hermes_home_override(home)
    try:
        result = save_provider_env_credential(env_var, value)
    finally:
        reset_hermes_home_override(token)

    # Verify before claiming success: the save path refuses (jobs locked by a managed
    # install / administrator) by RETURNING, which its caller cannot tell from a write.
    written = load_env_file(home / ".env").get(env_var, _MISSING)
    if written != value:
        return _propagation_row(
            "", home, "failed",
            f"{env_var} did not land in {home / '.env'} (refused or overwritten)",
        )
    config_updates: list[str] = []
    raw_updates = result.get("config_updates")
    if isinstance(raw_updates, list):
        config_updates = [str(path) for path in raw_updates]
    if config_updates and isinstance(previous, str) and previous and previous != value:
        try:
            mirror_text = (home / "config.yaml").read_text(encoding="utf-8")
        except OSError as exc:
            return _propagation_row(
                "", home, "failed", f"config.yaml unreadable after scrub: {exc}",
            )
        if previous in mirror_text:
            return _propagation_row(
                "", home, "failed",
                f"stale value still present in {home / 'config.yaml'}",
            )
    return _propagation_row("", home, "updated", "updated", config_updates=config_updates)


def propagate_env_credential_to_siblings(
    env_var: str,
    value: str,
    *,
    old_value: str | None = None,
    active_home: str | Path | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    """Fan a rotated credential out to the sibling profiles that already define it.

    ``apply=False`` (dry run): enumerate + classify only — nothing on disk changes.
    ``apply=True``: write each eligible sibling's ``.env`` through the audited save path
    (home-overridden), scrub that home's ``config.yaml`` mirrors of the previous value, and
    verify each write landed. One profile's failure never aborts the rest.

    A sibling is updated when it already defines ``env_var`` and its value differs from
    ``value`` (a profile holding the new value is untouched). ``old_value`` — the value the
    active profile held before the rotation, when the caller knows it — only labels a row as
    stale (``stale=True``) for reporting; it never changes what gets written.

    Returns ``{"key", "dry_run", "results": [...], "updated", "pending", "skipped", "failed"}``
    where each row is ``{"profile", "home", "status", "reason", "stale", "config_updates"}``
    and ``status`` is ``updated`` / ``skipped`` / ``failed`` (``pending`` for a dry run's
    to-be-updated rows).
    """
    results: list[dict[str, Any]] = []
    from agent.secret_scope import load_env_file

    for name, home in sibling_profiles_with_env_var(env_var, active_home=active_home):
        current = load_env_file(home / ".env").get(env_var, _MISSING)
        if current is _MISSING:
            results.append(_propagation_row(
                name, home, "failed", f"{env_var} is not readable from {home / '.env'}",
            ))
            continue
        if current == value:
            results.append(_propagation_row(
                name, home, "skipped", "already holds the new value", stale=False,
            ))
            continue
        stale = (current == old_value) if old_value is not None else None
        label = "holds the previous value" if stale else "holds a different value"
        if not apply:
            results.append(_propagation_row(name, home, "pending", label, stale=stale))
            continue
        try:
            row = _apply_credential_to_home(env_var, value, home)
        except Exception as exc:  # one bad home must not abort the others
            row = _propagation_row(name, home, "failed", f"{type(exc).__name__}: {exc}")
        row["profile"] = name
        row["stale"] = stale
        results.append(row)

    def _count(status: str) -> int:
        return sum(1 for r in results if r["status"] == status)

    return {
        "key": env_var,
        "dry_run": not apply,
        "results": results,
        "updated": _count("updated"),
        "pending": _count("pending"),
        "skipped": _count("skipped"),
        "failed": _count("failed"),
    }
