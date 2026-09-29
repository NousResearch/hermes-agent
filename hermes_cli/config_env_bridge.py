"""The one writer for config.yaml -> ``os.environ`` bridges, and the record of what it wrote.

Startup bridges mirror the LAUNCH profile's config.yaml into env carriers that ``os.getenv``
readers consume: the gateway bridge (``gateway/run.py::_bridge_config_to_env``: agent timeouts,
``HERMES_TIMEZONE``, ``AUXILIARY_<TASK>_*``, busy/display knobs, top-level scalars), the CLI mirror
(``cli_config_load._mirror_config_to_env``), media policy (``gateway/media_policy.py``) and the
platform YAML hooks (``gateway/platforms/_shared.py::yaml_env_setter``). Each writes through
:func:`set_bridged_env`, so :func:`bridged_env_names` is exactly what this process bridged: the
set ``tools.environments.local.strip_launch_profile_env`` drops from a child built for ANOTHER
profile. That child bridges its own config; a kept value would run as its own wherever its config
is silent, and even where it is not when the bridge or the reader lets env win (``hermes_time``
reads ``HERMES_TIMEZONE`` before config.yaml). ``terminal.*`` bridges are not recorded here: every
``TERMINAL_*`` name is launch terminal policy by prefix.
"""

from __future__ import annotations

import os

_BRIDGED_NAMES: set[str] = set()


def set_bridged_env(name: str, value: str) -> None:
    """``os.environ[name] = value`` for a config-derived setting, recorded (case-folded)."""
    os.environ[name] = value
    _BRIDGED_NAMES.add(name.upper())


def bridged_env_names() -> frozenset[str]:
    """Upper-cased names this process wrote from its config.yaml via :func:`set_bridged_env`."""
    return frozenset(_BRIDGED_NAMES)
