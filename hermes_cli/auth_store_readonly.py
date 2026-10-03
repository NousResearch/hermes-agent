"""Import-safe auth-store diagnostics and shared per-provider pool selection.

No provider discovery, auth recovery, refresh, cache, filesystem writes or locks.
Ordinary auth operations retain their existing reader and recovery behavior.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any


def select_credential_pool(profile_store: dict, global_store: dict, provider_id=None):
    """One selection rule shared by operational auth and read-only diagnostics."""
    pool = profile_store.get("credential_pool")
    pool = pool if isinstance(pool, dict) else {}
    global_pool = global_store.get("credential_pool")
    global_pool = global_pool if isinstance(global_pool, dict) else {}
    if provider_id is None:
        merged = dict(pool)
        for gp_key, gp_entries in global_pool.items():
            existing = merged.get(gp_key)
            if not (isinstance(gp_entries, list) and gp_entries):
                continue
            if not (isinstance(existing, list) and existing):  # profile wins when it has ANY entries
                merged[gp_key] = list(gp_entries)
        return merged

    provider_entries = pool.get(provider_id)
    if isinstance(provider_entries, list) and provider_entries:
        return list(provider_entries)
    global_entries = global_pool.get(provider_id)
    return list(global_entries) if isinstance(global_entries, list) else []



def _auth_store_paths() -> tuple[Path, Path | None]:
    from hermes_constants import get_hermes_home, get_default_hermes_root
    profile = get_hermes_home() / "auth.json"
    root = get_default_hermes_root() / "auth.json"
    return profile, None if profile.resolve() == root.resolve() else root


def _read_store(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}
    except (ValueError, UnicodeError):
        raise ValueError("Auth store is not valid JSON") from None
    if not isinstance(raw, dict):
        raise ValueError("Auth store has an unsupported shape")
    if "credential_pool" in raw:
        pool = raw["credential_pool"]
        if not isinstance(pool, dict) or any(
            not isinstance(entries, list) for entries in pool.values()
        ):
            raise ValueError("Auth store credential_pool must map providers to lists")
    # Empty, metadata-only and legacy dictionaries have no pool. Do not migrate them.
    return raw


def read_credential_pool(provider_id=None):
    """Read the same effective pool as native auth, failing closed without recovery."""
    profile, root = _auth_store_paths()
    return select_credential_pool(_read_store(profile), _read_store(root), provider_id)
