"""Flag resolution for Executive v2.

Default-off. Opt-in is resolved from an explicit per-instance override, an
explicit config value passed by an integration boundary, or the legacy internal
environment bridge. No global config mutation. No config.yaml writes.
"""

from __future__ import annotations

import os
from inspect import getattr_static
from typing import Any, Mapping

_ENV_VAR = "HERMES_EXECUTIVE_V2_ENABLED"
_TRUTHY = {"1", "true", "yes", "on"}
CONFIG_KEY = "executive_v2_enabled"
CONFIG_UNSET = object()


def _coerce_enabled(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY
    return bool(value)


def _explicit_agent_override(agent: Any | None) -> Any:
    if agent is None:
        return CONFIG_UNSET
    try:
        return getattr_static(agent, "_executive_v2_enabled")
    except AttributeError:
        return CONFIG_UNSET
    except Exception:
        return CONFIG_UNSET


def _explicit_config_value(config: Mapping[str, Any] | None) -> Any:
    if not isinstance(config, Mapping):
        return CONFIG_UNSET
    agent_config = config.get("agent")
    if not isinstance(agent_config, Mapping) or CONFIG_KEY not in agent_config:
        return CONFIG_UNSET
    return agent_config[CONFIG_KEY]


def resolve_v2_enabled(
    agent: Any | None = None,
    *,
    config_value: Any = CONFIG_UNSET,
    config: Mapping[str, Any] | None = None,
) -> bool:
    """Resolve whether Executive v2 is enabled.

    Resolution order:
    1. Explicit per-instance override, when present.
    2. Explicit ``agent.executive_v2_enabled`` config value supplied by an
       integration boundary.
    3. Legacy internal environment bridge.

    Default: False.
    """
    override = _explicit_agent_override(agent)
    if override is not CONFIG_UNSET:
        return _coerce_enabled(override)

    if config_value is CONFIG_UNSET:
        config_value = _explicit_config_value(config)
    if config_value is not CONFIG_UNSET:
        return _coerce_enabled(config_value)

    try:
        env = os.environ.get(_ENV_VAR, "")
        if env and env.strip().lower() in _TRUTHY:
            return True
    except Exception:
        pass
    return False
