"""
Model profile helpers for multi-provider model configuration.

This module normalizes legacy model config formats and provides utilities
for profile CRUD and active-profile synchronization.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from hermes_constants import OPENROUTER_BASE_URL
from hermes_cli.provider_registry import (
    get_provider,
    normalize_provider_id,
    resolve_provider_base_url,
)

DEFAULT_MODEL_ID = "anthropic/claude-opus-4.6"
DEFAULT_PROFILE_NAME = "default-openrouter"


def _normalize_profile(raw: Any, idx: int) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None

    name = str(raw.get("name") or f"profile-{idx+1}").strip()
    provider = normalize_provider_id(str(raw.get("provider") or "openrouter").strip(), default="openrouter")
    model = str(raw.get("model") or raw.get("default") or DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID
    enabled = bool(raw.get("enabled", True))

    base_url = raw.get("base_url")
    if isinstance(base_url, str):
        base_url = base_url.strip().rstrip("/")
    else:
        base_url = None

    if not base_url:
        base_url = resolve_provider_base_url(provider)
        if not base_url:
            # Custom and OAuth providers should not silently inherit OpenRouter URLs.
            if provider in {"custom", "nous"}:
                base_url = ""
            else:
                base_url = OPENROUTER_BASE_URL

    return {
        "name": name,
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "enabled": enabled,
    }


def normalize_model_config(model_cfg: Any) -> Dict[str, Any]:
    """
    Convert model config (string or dict) into a normalized dict format.

    Normalized keys:
    - default
    - provider
    - base_url
    - profiles
    - active_profile
    - scoped_profiles
    """
    if isinstance(model_cfg, str):
        legacy_model = model_cfg.strip() or DEFAULT_MODEL_ID
        normalized: Dict[str, Any] = {
            "default": legacy_model,
            "provider": "openrouter",
            "base_url": OPENROUTER_BASE_URL,
            "profiles": [
                {
                    "name": DEFAULT_PROFILE_NAME,
                    "provider": "openrouter",
                    "model": legacy_model,
                    "base_url": OPENROUTER_BASE_URL,
                    "enabled": True,
                }
            ],
            "active_profile": DEFAULT_PROFILE_NAME,
            "scoped_profiles": [DEFAULT_PROFILE_NAME],
        }
        return normalized

    cfg = deepcopy(model_cfg) if isinstance(model_cfg, dict) else {}

    default_model = str(cfg.get("default") or DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID
    provider = normalize_provider_id(str(cfg.get("provider") or "openrouter").strip(), default="openrouter")

    base_url = cfg.get("base_url")
    if isinstance(base_url, str):
        base_url = base_url.strip().rstrip("/")
    else:
        base_url = None
    if not base_url:
        base_url = resolve_provider_base_url(provider)
        if not base_url:
            if provider in {"custom", "nous"}:
                base_url = ""
            else:
                base_url = OPENROUTER_BASE_URL

    raw_profiles = cfg.get("profiles")
    profiles: List[Dict[str, Any]] = []
    if isinstance(raw_profiles, list):
        for idx, entry in enumerate(raw_profiles):
            profile = _normalize_profile(entry, idx)
            if profile:
                profiles.append(profile)

    if not profiles:
        # Synthesize from legacy fields
        profiles = [
            {
                "name": DEFAULT_PROFILE_NAME,
                "provider": provider,
                "model": default_model,
                "base_url": base_url,
                "enabled": True,
            }
        ]

    # Ensure unique profile names (stable suffixing)
    seen = set()
    for idx, p in enumerate(profiles):
        base_name = p["name"]
        candidate = base_name
        suffix = 2
        while candidate in seen:
            candidate = f"{base_name}-{suffix}"
            suffix += 1
        p["name"] = candidate
        seen.add(candidate)

    active_profile = str(cfg.get("active_profile") or "").strip()
    active_names = {p["name"] for p in profiles if p.get("enabled", True)}
    if not active_profile or active_profile not in active_names:
        active_profile = profiles[0]["name"]

    scoped_profiles_raw = cfg.get("scoped_profiles")
    scoped_profiles: List[str] = []
    if isinstance(scoped_profiles_raw, list):
        for value in scoped_profiles_raw:
            name = str(value).strip()
            if name and any(p["name"] == name for p in profiles):
                scoped_profiles.append(name)
    if not scoped_profiles:
        scoped_profiles = [active_profile]

    normalized = {
        "default": default_model,
        "provider": provider,
        "base_url": base_url,
        "profiles": profiles,
        "active_profile": active_profile,
        "scoped_profiles": scoped_profiles,
    }
    return sync_legacy_model_fields(normalized)


def _active_profile_from_normalized(cfg: Dict[str, Any]) -> Dict[str, Any]:
    active_name = cfg["active_profile"]
    for p in cfg["profiles"]:
        if p["name"] == active_name:
            return p
    return cfg["profiles"][0]


def get_active_profile(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = normalize_model_config(model_cfg)
    return _active_profile_from_normalized(cfg)


def sync_legacy_model_fields(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(model_cfg)
    profiles = cfg.get("profiles", [])
    if not isinstance(profiles, list) or not profiles:
        return cfg

    active = _active_profile_from_normalized(cfg)
    cfg["default"] = active.get("model") or cfg.get("default") or DEFAULT_MODEL_ID
    cfg["provider"] = active.get("provider") or cfg.get("provider") or "openrouter"
    active_base_url = active.get("base_url")
    provider_base_url = resolve_provider_base_url(cfg["provider"])
    if cfg["provider"] in {"custom", "nous"}:
        cfg["base_url"] = active_base_url or provider_base_url or ""
    else:
        legacy_base_url = cfg.get("base_url")
        cfg["base_url"] = (
            active_base_url
            or provider_base_url
            or legacy_base_url
            or OPENROUTER_BASE_URL
        )
    return cfg


def upsert_profile(model_cfg: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    cfg = normalize_model_config(model_cfg)
    entry = _normalize_profile(profile, 0)
    if not entry:
        return cfg

    updated = False
    for idx, existing in enumerate(cfg["profiles"]):
        if existing.get("name") == entry["name"]:
            cfg["profiles"][idx] = entry
            updated = True
            break
    if not updated:
        cfg["profiles"].append(entry)

    if entry.get("enabled", True):
        cfg["active_profile"] = entry["name"]
    if entry["name"] not in cfg["scoped_profiles"]:
        cfg["scoped_profiles"].append(entry["name"])
    return sync_legacy_model_fields(cfg)


def remove_profile(model_cfg: Dict[str, Any], profile_name: str) -> Dict[str, Any]:
    cfg = normalize_model_config(model_cfg)
    cfg["profiles"] = [p for p in cfg["profiles"] if p.get("name") != profile_name]
    cfg["scoped_profiles"] = [n for n in cfg.get("scoped_profiles", []) if n != profile_name]
    if not cfg["profiles"]:
        cfg["profiles"] = [
            {
                "name": DEFAULT_PROFILE_NAME,
                "provider": "openrouter",
                "model": DEFAULT_MODEL_ID,
                "base_url": OPENROUTER_BASE_URL,
                "enabled": True,
            }
        ]
    if cfg.get("active_profile") == profile_name:
        cfg["active_profile"] = cfg["profiles"][0]["name"]
    if not cfg["scoped_profiles"]:
        cfg["scoped_profiles"] = [cfg["active_profile"]]
    return sync_legacy_model_fields(cfg)


def set_active_profile(model_cfg: Dict[str, Any], profile_name: str) -> Dict[str, Any]:
    cfg = normalize_model_config(model_cfg)
    if any(p.get("name") == profile_name for p in cfg["profiles"]):
        cfg["active_profile"] = profile_name
        if profile_name not in cfg["scoped_profiles"]:
            cfg["scoped_profiles"].append(profile_name)
    return sync_legacy_model_fields(cfg)


def get_profile_names(model_cfg: Dict[str, Any], *, enabled_only: bool = False) -> List[str]:
    cfg = normalize_model_config(model_cfg)
    names: List[str] = []
    for p in cfg["profiles"]:
        if enabled_only and not p.get("enabled", True):
            continue
        names.append(p["name"])
    return names
