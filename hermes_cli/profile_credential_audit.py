"""Post-update audit: named profiles that have no credentials of their own.

Until Sep 2026 a named profile with an empty ``auth.json`` silently read the root profile's
``auth.json`` (provider state + credential pool) and even wrote refreshed OAuth tokens back to it.
Profiles are independent islands, so that fallback is gone (#111724): such a profile now gets the
"not connected to any AI provider" setup prompt instead of the owner's credentials. ``hermes update``
names every affected profile up front so nobody discovers the change from a dead bot.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def _auth_store_has_credentials(auth_path: Path) -> bool:
    try:
        store = json.loads(auth_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return False
    if not isinstance(store, dict):
        return False
    providers = store.get("providers")
    if isinstance(providers, dict) and any(isinstance(v, dict) and v for v in providers.values()):
        return True
    pool = store.get("credential_pool")
    return isinstance(pool, dict) and any(isinstance(v, list) and v for v in pool.values())


def _profile_has_own_credentials(profile_dir: Path) -> bool:
    """auth.json credentials, a provider key in ``.env``, or a model block carrying its own key/base URL."""
    if _auth_store_has_credentials(profile_dir / "auth.json"):
        return True
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.main import _dotenv_has_provider_key
    provider_env_vars = {"OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "OPENAI_BASE_URL"}
    for pconfig in PROVIDER_REGISTRY.values():
        if pconfig.auth_type == "api_key":
            provider_env_vars.update(pconfig.api_key_env_vars)
    if _dotenv_has_provider_key(profile_dir / ".env", provider_env_vars):
        return True
    try:
        import yaml
        cfg = yaml.safe_load((profile_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    except Exception:
        return False
    model_cfg = cfg.get("model") if isinstance(cfg, dict) else None
    return isinstance(model_cfg, dict) and any(
        str(model_cfg.get(k) or "").strip() for k in ("base_url", "api_key"))


def profiles_without_own_credentials() -> List[str]:
    """Names of live named profiles that would have inherited the root ``auth.json`` — i.e. the root
    holds credentials and the profile holds none. Empty when the root has nothing to inherit."""
    from hermes_constants import get_default_hermes_root
    from hermes_cli.profiles import _iter_named_profile_dirs
    if not _auth_store_has_credentials(get_default_hermes_root() / "auth.json"):
        return []
    return [p.name for p in _iter_named_profile_dirs() if not _profile_has_own_credentials(p)]


def print_profiles_without_credentials_notice() -> None:
    names = profiles_without_own_credentials()
    if not names:
        return
    print("\n\033[1;33m⚠  Profiles no longer inherit the root profile's credentials (auth.json).\033[0m")
    print("   These profiles have no provider of their own and will ask for one on their next turn:")
    for name in names:
        print(f"     hermes -p {name} model        # or: hermes -p {name} auth add <provider>")
