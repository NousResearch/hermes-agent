"""Single-use OAuth grant hygiene: strip cloned grants from profiles.

Split out of ``hermes_cli/auth.py`` and re-exported there; origin helpers are imported lazily
inside each function so ``hermes_cli.auth.<name>`` patches still intercept (and no import cycle).
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any, Dict

# Log-record parity with the origin module (caplog tests pin "hermes_cli.auth").
logger = logging.getLogger("hermes_cli.auth")

# Pool providers whose OAuth refresh tokens are SINGLE-USE: redeeming rotates the pair and
# revokes the old one, so a grant forked into two auth.json files is ONE credential with two
# owners — the first to refresh strands the other with ``invalid_grant`` /
# ``refresh_token_reused``.
# Profiles must never receive a copy; each profile signs in on its own (``hermes -p <name> auth add``).
SINGLE_USE_REFRESH_POOL_PROVIDERS = frozenset({"anthropic", "openai-codex", "xai-oauth"})

# Singleton credential files holding the same single-use grants outside ``auth.json``. Copying one
# into a profile re-seeds a forked pool row on the profile's next ``load_pool()``.
SINGLE_USE_OAUTH_SINGLETON_FILES = (".anthropic_oauth.json",)

# Providers whose device-code grants live under ``providers.<id>`` (not only the pool).
_DEVICE_CODE_BLOCK_PROVIDERS = ("openai-codex", "xai-oauth")


def _is_oauth_pool_payload(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    # Legacy rows predating ``auth_type``: an Anthropic OAuth access token or any row carrying a
    # refresh token is an OAuth grant.
    return (
        str(entry.get("auth_type") or "").strip().lower() == "oauth"
        or bool(str(entry.get("refresh_token") or "").strip())
        or str(entry.get("access_token") or "").startswith("sk-ant-oat"))


def strip_cloned_single_use_oauth_grants(profile_dir: Path) -> Dict[str, Any]:
    """Remove forked single-use OAuth grants from a freshly cloned profile.

    Called after any path that copies credential files between profiles (``hermes profile create
    --clone-all``, the dashboard/TUI ``mirror_credentials`` flow). API-key pool rows are kept — a
    static key is safe to duplicate; the clone signs into the OAuth provider itself. Returns ``{"pool": [...provider ids], "providers": [...],
    "files": [...]}`` of what was stripped. Never raises: a clone must not fail because hygiene
    could not run — the caller logs the summary.
    """
    from hermes_cli.auth import _same_path, _save_auth_store
    stripped: Dict[str, Any] = {"pool": [], "providers": [], "files": []}
    profile_dir = Path(profile_dir)
    for name in SINGLE_USE_OAUTH_SINGLETON_FILES:
        target = profile_dir / name
        try:
            if target.is_file() or target.is_symlink():
                target.unlink()
                stripped["files"].append(name)
        except OSError:
            logger.debug("Could not remove cloned %s from %s", name, profile_dir, exc_info=True)
    auth_path = profile_dir / "auth.json"
    if not auth_path.is_file():
        return stripped
    # A profile auth.json that IS the root store (symlink / hardlink) is not a clone; stripping it
    # would delete the root's single-use grants. Resolve identity positively: same path, or
    # samefile() says so; any error refuses (fail closed).
    try:
        from hermes_constants import get_default_hermes_root
        root_auth_path = get_default_hermes_root() / "auth.json"
        if _same_path(auth_path, root_auth_path) or (
            root_auth_path.exists() and auth_path.samefile(root_auth_path)
        ):
            return stripped
    except Exception:
        return stripped
    try:
        store = json.loads(auth_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        store = None
    if not isinstance(store, dict):
        return stripped
    changed = False
    pool = store.get("credential_pool")
    if isinstance(pool, dict):
        for provider_id in list(pool):
            entries = pool.get(provider_id)
            if (provider_id not in SINGLE_USE_REFRESH_POOL_PROVIDERS
                    or not isinstance(entries, list)):
                continue
            kept = [e for e in entries if not _is_oauth_pool_payload(e)]
            if len(kept) != len(entries):
                changed = True
                stripped["pool"].append(provider_id)
                if kept:
                    pool[provider_id] = kept
                else:
                    del pool[provider_id]
    providers = store.get("providers")
    if isinstance(providers, dict):
        for provider_id in _DEVICE_CODE_BLOCK_PROVIDERS:
            block = providers.get(provider_id)
            if isinstance(block, dict) and block:
                del providers[provider_id]
                stripped["providers"].append(provider_id)
                changed = True
    if not changed:
        return stripped
    try:
        _save_auth_store(store, target_path=auth_path)
    except Exception:
        logger.debug(
            "Failed to strip cloned single-use OAuth grants from %s", auth_path, exc_info=True)
    return stripped
