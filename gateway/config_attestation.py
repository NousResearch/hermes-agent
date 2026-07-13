"""Sanitized provenance for gateway startup configuration."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from hermes_cli.fallback_config import get_fallback_chain


@dataclass(frozen=True)
class ProfileConfigSnapshot:
    """One immutable-on-disk config read and its derived runtime views."""

    parsed: dict[str, Any]
    normalized: dict[str, Any]
    effective: dict[str, Any]
    attestation: dict[str, Any]
    loaded_at: str


def load_profile_config_snapshot(profile_home: Path) -> ProfileConfigSnapshot:
    """Read config once and derive both runtime input and sanitized provenance."""
    profile_path = profile_home.resolve()
    config_path = (profile_path / "config.yaml").resolve()
    config_bytes = config_path.read_bytes()
    loaded_at = datetime.now(timezone.utc).isoformat()
    parsed = yaml.safe_load(config_bytes.decode("utf-8-sig")) or {}
    if not isinstance(parsed, dict):
        raise ValueError("config.yaml top level must be a mapping")

    from hermes_cli import managed_scope
    from hermes_cli.config import _expand_env_vars, _normalize_root_model_keys

    normalized = managed_scope.apply_managed_overlay(deepcopy(parsed))
    normalized = _normalize_root_model_keys(normalized)
    effective = _expand_env_vars(deepcopy(normalized))
    if not isinstance(effective, dict):
        raise ValueError("effective config must be a mapping")

    # Model selection and per-turn fallback refresh use the normalized config
    # path (`_load_gateway_config`), not the env-expanded runtime-helper view.
    # Attest the values that those actual model paths consume.
    model_config = normalized.get("model", {})
    if isinstance(model_config, str):
        provider = ""
        model = model_config.strip()
    elif isinstance(model_config, dict):
        provider = str(model_config.get("provider") or "").strip()
        model = str(
            model_config.get("default") or model_config.get("model") or ""
        ).strip()
    else:
        provider = ""
        model = ""

    fallbacks = [
        {
            "provider": str(entry.get("provider") or "").strip(),
            "model": str(entry.get("model") or "").strip(),
        }
        for entry in get_fallback_chain(normalized)
    ]

    attestation = {
        "profile_path": str(profile_path),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "primary": {"provider": provider, "model": model},
        "fallbacks": fallbacks,
    }
    return ProfileConfigSnapshot(
        parsed=deepcopy(parsed),
        normalized=normalized,
        effective=effective,
        attestation=attestation,
        loaded_at=loaded_at,
    )


__all__ = [
    "ProfileConfigSnapshot",
    "load_profile_config_snapshot",
]
