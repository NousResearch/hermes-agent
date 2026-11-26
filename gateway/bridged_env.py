"""Ownership markers of the YAML -> env bridge (``gateway.platforms._shared.yaml_env_setter``).

Stdlib only: the subprocess env builders (``tools.environments.local``) read these on the
post-update restart path, which must import nothing heavy (``hermes_cli.update_cmd_commit``), and
``launch_secret_scope`` follows them from a remote config backend's cold token fetch, where importing
``gateway.platforms`` would read config.
"""

from __future__ import annotations

import hashlib
import os
import threading
from typing import Optional

BRIDGED_ENV_MARKER_PREFIX = "HERMES_BRIDGED_"
# Held for a whole bridging load (``gateway.platforms._shared.yaml_env_refresh``): its writes publish all-or-nothing.
YAML_ENV_LOCK = threading.RLock()


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


def follow_bridged_env(scope: dict) -> None:
    """Bring a launch-profile secret mapping's YAML-bridged values up to the bridge's current ones.

    The mapping starts from the env frozen at activation, which holds the bridged lists of THAT moment;
    the bridge keeps refreshing ``os.environ`` from the launch profile's config.yaml afterwards. Only a
    value the bridge owns in the mapping (its marker's digest matches) or a name the mapping lacks
    follows: it takes the live bridge-owned value, or is dropped when the bridge retracted it (name
    and marker both gone). Any other live state (a value another writer put there) keeps the frozen
    value, and a frozen operator value (systemd, ``op run``) is never replaced from the live env; the
    caller applies ``.env`` and secret sources over this.
    ``allow_all_users`` is bridged without a marker: a frozen grant the live env no longer holds was
    revoked (only the bridge clears it); a live one reaches the mapping via ``build_profile_secret_scope``."""
    from gateway.config_loader import bridged_allow_all_users

    with YAML_ENV_LOCK:  # a load in progress is published all-or-nothing: never read its partial writes
        live = dict(os.environ)
        grant = bridged_allow_all_users()
    # Revoke the grant BEFORE touching the lists: another thread of this body may read the mapping
    # mid-update, and "no list, grant still on" would admit any sender for that one lookup.
    if grant is None and "GATEWAY_ALLOW_ALL_USERS" not in live:
        scope.pop("GATEWAY_ALLOW_ALL_USERS", None)
    prefix = BRIDGED_ENV_MARKER_PREFIX
    names = {key[len(prefix):] for key in (*scope, *live) if key.startswith(prefix) and key != prefix}
    retracted = []
    for name in sorted(names):  # deterministic, not hash-seed order
        marker = prefix + name
        if name in scope and bridged_env_digest(scope[name]) != scope.get(marker):
            continue
        if live.get(marker) is not None and bridged_env_digest(live.get(name)) == live[marker]:
            scope[name], scope[marker] = live[name], live[marker]
        elif name not in live and marker not in live:  # retracted: only the bridge removes both
            retracted.append((name, marker))
    # Every new list lands before any old one goes: an edit moving users from one list to another
    # never shows a reader "no list at all", which a still-on grant would turn into admit-all.
    for name, marker in retracted:
        scope.pop(name, None)
        scope.pop(marker, None)
