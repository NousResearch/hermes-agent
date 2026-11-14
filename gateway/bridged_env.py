"""Ownership markers of the YAML -> env bridge (``gateway.platforms._shared.yaml_env_setter``).

Stdlib only: the subprocess env builders (``tools.environments.local``) read these on the
post-update restart path, which must import nothing heavy (``hermes_cli.update_cmd_commit``).
"""

from __future__ import annotations

import hashlib
from typing import Optional

BRIDGED_ENV_MARKER_PREFIX = "HERMES_BRIDGED_"


def bridged_env_digest(value: Optional[str]) -> Optional[str]:
    """What a marker stores for *value* (None stays None, so an unset name never matches)."""
    return None if value is None else "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def drop_bridge_markers(env: dict) -> dict:
    """Remove every marker from a child env, and with it each value the bridge owns (the value still
    equals its marker's digest). A marker-less bridged value reads as operator-set to the next
    gateway started from that env, which would then keep it over an edited ``config.yaml``; without
    the value the child re-derives it from its own YAML. An unmatched value stays (the operator's)."""
    markers = [key for key in env if key.upper().startswith(BRIDGED_ENV_MARKER_PREFIX)]
    for key in markers:
        name = key[len(BRIDGED_ENV_MARKER_PREFIX):]
        if name and bridged_env_digest(env.get(name)) == env[key]:
            env.pop(name, None)
    for key in markers:
        del env[key]
    return env

