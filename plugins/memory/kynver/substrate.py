"""Kynver substrate detection — default-on when configured and healthy."""

from __future__ import annotations

import os
from typing import Any, Mapping

from hermes_cli.config import cfg_get

from .agentos_bridge import (
    KynverAgentOSClient,
    agentos_available,
    load_kynver_agentos_config,
    probe_agentos_health,
)


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def kynver_explicitly_disabled(
    env: Mapping[str, str] | None = None,
    config: Mapping[str, Any] | None = None,
) -> bool:
    """Explicit opt-out only — never default-off gates."""

    merged = dict(env or os.environ)
    if _truthy(merged.get("KYNVER_DISABLED")):
        return True
    if _truthy(merged.get("HERMES_KYNVER_DISABLED")):
        return True
    if config:
        kynver_cfg = config.get("kynver") if isinstance(config.get("kynver"), dict) else {}
        if _truthy(cfg_get(config, "kynver", "disabled")):
            return True
        if _truthy(cfg_get(config, "memory", "kynver_disabled")):
            return True
    return False


def resolve_memory_provider_name(
    mem_config: Mapping[str, Any] | None,
    *,
    env: Mapping[str, str] | None = None,
    full_config: Mapping[str, Any] | None = None,
) -> str:
    """Return memory provider name to activate.

    - Explicit non-kynver provider in config wins.
    - Blank provider + configured healthy Kynver → ``kynver`` (default-on).
  """

    mem_config = mem_config or {}
    configured = (mem_config.get("provider") or "").strip()
    if configured and configured.lower() != "kynver":
        return configured
    if kynver_explicitly_disabled(env=env, config=full_config):
        return configured
    if not agentos_available(env):
        return configured
    client = KynverAgentOSClient(load_kynver_agentos_config(env))
    if probe_agentos_health(client):
        return "kynver"
    return configured


def substrate_active(
    *,
    env: Mapping[str, str] | None = None,
    config: Mapping[str, Any] | None = None,
    client: KynverAgentOSClient | None = None,
) -> bool:
    """True when Kynver AgentOS should own operating state this session."""

    if kynver_explicitly_disabled(env=env, config=config):
        return False
    if not agentos_available(env):
        return False
    return probe_agentos_health(client)


def allow_local_fallback(config: Mapping[str, Any] | None = None) -> bool:
    if config is None:
        return True
    val = cfg_get(config, "kynver", "allow_local_fallback", default=True)
    if isinstance(val, bool):
        return val
    return _truthy(val)
