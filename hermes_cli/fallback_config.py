"""Helpers for reading the effective fallback provider chain from config."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import yaml

from agent.secret_scope import get_secret


def _replace_profile_env_reference(match: re.Match[str]) -> str:
    resolved = get_secret(match.group(1))
    return match.group(0) if resolved is None else resolved


def _expand_profile_env_vars(value: Any) -> Any:
    """Expand config templates through the active profile's secret scope."""
    if isinstance(value, str):
        return re.sub(r"\${([^}]+)}", _replace_profile_env_reference, value)
    if isinstance(value, dict):
        return {key: _expand_profile_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_profile_env_vars(item) for item in value]
    return value


def _normalized_base_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().rstrip("/")


def resolve_entry_api_key(entry: dict[str, Any] | None) -> str | None:
    """API key for one fallback entry: inline ``api_key``, else ``key_env``.

    Mirrors the custom-provider convention (``key_env`` names the env var
    holding the key; ``api_key_env`` accepted as an alias). Returns None when
    neither yields a non-empty value, letting ``resolve_runtime_provider``
    fall through to the provider's standard credential resolution.
    """
    if not isinstance(entry, dict):
        return None
    inline = str(entry.get("api_key") or "").strip()
    if inline:
        return inline
    key_env = str(entry.get("key_env") or entry.get("api_key_env") or "").strip()
    if key_env:
        return (get_secret(key_env) or "").strip() or None
    return None


def _iter_fallback_entries(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        candidates = [raw]
    elif isinstance(raw, list):
        candidates = raw
    else:
        return []

    entries: list[dict[str, Any]] = []
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider") or "").strip()
        model = str(entry.get("model") or "").strip()
        if not provider or not model:
            continue

        normalized = dict(entry)
        normalized["provider"] = provider
        normalized["model"] = model

        base_url = _normalized_base_url(entry.get("base_url"))
        if base_url:
            normalized["base_url"] = base_url

        entries.append(normalized)
    return entries


def _entry_identity(entry: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(entry.get("provider") or "").strip().lower(),
        str(entry.get("model") or "").strip().lower(),
        _normalized_base_url(entry.get("base_url")).lower(),
    )


def get_fallback_chain(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the effective fallback chain merged across old and new config keys.

    ``fallback_providers`` remains the primary source of truth and keeps its
    order. Legacy ``fallback_model`` entries are appended afterwards unless
    they target the same provider/model/base_url route as an earlier entry.
    The returned list always contains fresh dict copies.
    """

    config = config or {}
    chain: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for key in ("fallback_providers", "fallback_model"):
        for entry in _iter_fallback_entries(config.get(key)):
            identity = _entry_identity(entry)
            if identity in seen:
                continue
            seen.add(identity)
            chain.append(entry)

    return chain


def _read_yaml_mapping_strict(path: Path) -> dict[str, Any]:
    """Read one YAML mapping snapshot, raising on I/O or invalid content."""
    try:
        with open(path, encoding="utf-8") as config_file:
            parsed = yaml.safe_load(config_file)
    except FileNotFoundError:
        return {}
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError(f"{path} root must be a mapping")
    return parsed


def load_fallback_chain_strict() -> list[dict[str, Any]]:
    """Read the effective user + managed fallback chain without fail-open gaps.

    The general config loaders intentionally convert parse/read failures to
    defaults so startup can continue. A live-agent refresh cannot use that
    fail-open result: defaults are indistinguishable from a user deliberately
    removing every fallback and would erase the session's last-known-good
    chain. Read each source exactly once and raise on failures so the caller
    can retain its cached chain.
    """
    from hermes_cli import managed_scope
    from hermes_cli.config import (
        _CONFIG_LOCK,
        _deep_merge,
        _expand_env_vars,
        get_config_path,
    )

    with _CONFIG_LOCK:
        effective = cast(
            dict[str, Any],
            _expand_profile_env_vars(_read_yaml_mapping_strict(get_config_path())),
        )

        managed_dir = managed_scope.get_managed_dir()
        if managed_dir is not None:
            managed = _read_yaml_mapping_strict(managed_dir / "config.yaml")
            if managed:
                managed_expanded = cast(
                    dict[str, Any], _expand_env_vars(managed)
                )
                effective = _deep_merge(effective, managed_expanded)

        return get_fallback_chain(effective)
