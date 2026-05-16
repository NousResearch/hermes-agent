"""Role-based tool gating for the API server gateway.

Loads per-profile ``config/role-map.yaml`` and ``config/role-tools.yaml``
to provide identity → role resolution and role → toolset filtering.

Architecture
------------
1. ``load_role_map()`` reads ``config/role-map.yaml``:
   ``identity_patterns`` maps request identity (API key hash, user_id, etc.)
   to a named role.

2. ``load_role_tools()`` reads ``config/role-tools.yaml``:
   Per-role ``allow`` / ``deny`` lists of toolset names, plus optional
   ``global_deny``.

3. ``filter_toolsets_by_role()`` intersects the enabled toolsets with the
   resolved role's allow list, minus the deny list.

All loaders are cached per-profile-dir so repeated calls within the same
gateway process are cheap (stat-based TTL of 60 s).
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

import yaml

logger = logging.getLogger(__name__)

# Cache: {(profile_dir, filename): (mtime, parsed_data)}
_cache: Dict[tuple, tuple[float, Any]] = {}
_CACHE_TTL = 60.0  # seconds


def _profile_config_dir() -> Path:
    """Return the active profile config directory."""
    profile_home = os.environ.get("HERMES_PROFILE_HOME", os.path.expanduser("~/.hermes/profiles/main"))
    return Path(profile_home) / "config"


def _load_yaml(filename: str) -> Optional[Dict[str, Any]]:
    """Load a YAML file from profile config/ with stat-based cache."""
    config_dir = _profile_config_dir()
    filepath = config_dir / filename

    if not filepath.is_file():
        return None

    mtime = filepath.stat().st_mtime
    cache_key = (str(config_dir), filename)
    cached = _cache.get(cache_key)
    if cached and cached[0] == mtime:
        return cached[1]

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        logger.warning("Failed to load %s", filepath, exc_info=True)
        return None

    if not isinstance(data, dict):
        return None

    _cache[cache_key] = (mtime, data)
    return data


def load_role_map() -> Dict[str, str]:
    """Load identity → role mapping from ``config/role-map.yaml``.

    Expected format::

        roles:
          admin:
            identities:
              - "sha256:abcdef..."
              - "user_id:12345"
          viewer:
            identities:
              - "user_id:67890"

    Returns a flat dict mapping identity strings to role names.
    """
    data = _load_yaml("role-map.yaml")
    if data is None:
        return {}

    mapping: Dict[str, str] = {}
    for role_name, role_def in data.get("roles", {}).items():
        if not isinstance(role_def, dict):
            continue
        for identity in role_def.get("identities", []):
            if isinstance(identity, str) and identity.strip():
                mapping[identity.strip()] = role_name

    return mapping


def load_role_tools() -> Dict[str, Dict[str, list]]:
    """Load role → tool allow/deny from ``config/role-tools.yaml``.

    Expected format::

        global_deny:
          - "terminal"
          - "file"

        roles:
          admin:
            allow: []          # empty = all (subject to global_deny)
          viewer:
            allow:
              - "web"
              - "search"
            deny:
              - "browser"

    Returns a dict with optional ``global_deny`` and ``roles`` keys.
    """
    data = _load_yaml("role-tools.yaml")
    if data is None:
        return {}

    result: Dict[str, Dict[str, list]] = {}
    if "global_deny" in data:
        result["global_deny"] = data["global_deny"]
    if "roles" in data:
        result["roles"] = data["roles"]
    return result


def resolve_role(
    identity: Optional[str],
    role_map: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Resolve a request identity to a role name.

    If *role_map* is None, it is loaded from disk.
    Returns None if the identity has no role mapping.
    """
    if not identity:
        return None
    if role_map is None:
        role_map = load_role_map()
    return role_map.get(identity)


def filter_toolsets_by_role(
    enabled_toolsets: Set[str],
    role: Optional[str] = None,
    role_tools: Optional[Dict[str, Dict[str, list]]] = None,
) -> Set[str]:
    """Filter *enabled_toolsets* by the resolved *role*.

    Rules:
    1. If *role* is None (no role-map configured or identity not found),
       return *enabled_toolsets* unchanged (no gating).
    2. Apply ``global_deny`` if present.
    3. If the role has an ``allow`` list and it is non-empty, intersect
       *enabled_toolsets* with the allow list.
    4. Remove any role-level ``deny`` entries.
    """
    if role is None:
        return enabled_toolsets

    if role_tools is None:
        role_tools = load_role_tools()

    result = set(enabled_toolsets)

    # Global deny
    global_deny = set(role_tools.get("global_deny", []))
    if global_deny:
        result -= global_deny

    # Role-specific rules
    roles = role_tools.get("roles", {})
    role_def = roles.get(role, {})
    if not isinstance(role_def, dict):
        return result

    allow_list = role_def.get("allow")
    if isinstance(allow_list, list) and allow_list:
        result &= set(allow_list)

    deny_list = role_def.get("deny")
    if isinstance(deny_list, list):
        result -= set(deny_list)

    return result


def hash_api_key(api_key: str) -> str:
    """Hash an API key for identity matching in role-map.yaml.

    Returns ``sha256:<hex>`` — the same format used in role-map.yaml
    identity entries.
    """
    return "sha256:" + hashlib.sha256(api_key.encode()).hexdigest()
