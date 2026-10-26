"""Undo the system proxy the Desktop handed to an update's children.

The Desktop resolves the OS proxy (PAC included) and gives it to the update's
git, urllib, npm and uv children as HTTPS_PROXY/HTTP_PROXY with an extended
NO_PROXY (apps/desktop/electron/updater/system-proxy-env.ts). ``hermes update``
then restarts gateways and backends that inherit that env; for them it is not
the user's setting, and every host the PAC sends direct (LAN model servers,
internal names) would go through the proxy. ``HERMES_UPDATE_SYSTEM_PROXY`` names
the variables the Desktop added and ``HERMES_UPDATE_USER_<KEY>`` holds a user's
own NO_PROXY it extended; a long-lived process restores its env from them at
start. Same algorithm as restoreUserProxyEnv, posix.sh and windows.ps1.
"""
from __future__ import annotations

import os
from typing import MutableMapping

MARKER = "HERMES_UPDATE_SYSTEM_PROXY"
SAVED_PREFIX = "HERMES_UPDATE_USER_"
_ADDED_KEYS = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "NO_PROXY", "no_proxy")
_NO_PROXY_KEYS = ("NO_PROXY", "no_proxy")


def restore_user_proxy_env(environ: MutableMapping[str, str] | None = None) -> None:
    """Drop the proxy variables the Desktop added and restore the user's own NO_PROXY."""
    env = os.environ if environ is None else environ
    for key in env.get(MARKER, "").split(","):
        if key in _ADDED_KEYS:
            env.pop(key, None)
    for key in _NO_PROXY_KEYS:
        saved = env.pop(SAVED_PREFIX + key, None)
        if saved is not None:
            env[key] = saved
    env.pop(MARKER, None)
