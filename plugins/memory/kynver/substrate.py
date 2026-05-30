"""Kynver operating substrate detection — default-on when configured and healthy."""

from __future__ import annotations

import os
from typing import Any, Mapping

from hermes_cli.config import cfg_get

from .agentos_bridge import (
    KynverAgentOSClient,
    agentos_enabled,
    load_kynver_agentos_config,
    probe_agentos_health,
)
from .operating_config import kynver_operating_tools_enabled


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
    merged = dict(env or os.environ)
    if _truthy(merged.get("KYNVER_DISABLED")):
        return True
    if _truthy(merged.get("HERMES_KYNVER_DISABLED")):
        return True
    if config:
        if _truthy(cfg_get(config, "kynver", "disabled")):
            return True
        if _truthy(cfg_get(config, "memory", "kynver_disabled")):
            return True
    return False


def substrate_active(
    *,
    env: Mapping[str, str] | None = None,
    config: Mapping[str, Any] | None = None,
    client: KynverAgentOSClient | None = None,
) -> bool:
    """True when Kynver should own Hermes todo/current-focus for this session."""

    if kynver_explicitly_disabled(env=env, config=config):
        return False
    if not kynver_operating_tools_enabled(env):
        return False
    if client is not None:
        return probe_agentos_health(client)
    if not agentos_enabled(env):
        return False
    return probe_agentos_health(KynverAgentOSClient(load_kynver_agentos_config(env)))


def allow_local_fallback(config: Mapping[str, Any] | None = None) -> bool:
    if config is None:
        return True
    val = cfg_get(config, "kynver", "allow_local_fallback", default=True)
    if isinstance(val, bool):
        return val
    return _truthy(val)
