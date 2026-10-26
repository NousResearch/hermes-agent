"""Undo the system proxy the Desktop handed to an update's children.

The Desktop resolves the OS proxy (PAC included) and gives it to the update's
git, urllib, npm and uv children as HTTPS_PROXY/HTTP_PROXY with an extended
NO_PROXY (apps/desktop/electron/updater/system-proxy-env.ts). ``hermes update``
then restarts gateways and backends that inherit that env; for them it is not
the user's setting, and every host the PAC sends direct (LAN model servers,
internal names) would go through the proxy. ``HERMES_UPDATE_SYSTEM_PROXY`` names
the variables the Desktop added and ``HERMES_UPDATE_USER_<KEY>`` holds a user's
own NO_PROXY it extended; a long-lived process restores its env from them at
start. Same algorithm as restoreUserProxyEnv, posix.sh and windows.ps1, plus one
rule only a Hermes process needs: a value the user's ``.env`` set wins. ``hermes
update`` and the restarted process load ``~/.hermes/.env`` with override before
this runs, so a marked key holding the value a dotenv load published is the
user's own setting (same value as the Desktop's or not) and is kept. The Desktop
and the hand-off scripts never load ``.env``, so their envs never hit that case.
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
    from hermes_cli.env_loader import dotenv_published_values

    env = os.environ if environ is None else environ

    def users_dotenv_value(key: str) -> bool:
        return key in env and env[key] in dotenv_published_values(key)

    for key in env.get(MARKER, "").split(","):
        if key in _ADDED_KEYS and not users_dotenv_value(key):
            env.pop(key, None)
    for key in _NO_PROXY_KEYS:
        saved = env.pop(SAVED_PREFIX + key, None)
        if saved is not None and not users_dotenv_value(key):
            env[key] = saved
    env.pop(MARKER, None)
