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


def prefer_dotenv_proxy_env(environ: MutableMapping[str, str] | None = None) -> None:
    """In ``hermes update``, after its ``.env`` load: the user's ``.env`` proxy beats the system one.

    The Desktop injects every case variant (``HTTPS_PROXY`` and ``https_proxy``) but the ``.env``
    load replaces only the spelling the file uses, and urllib and curl prefer the lower-case one, so
    the update would still go through the system proxy. A ``.env`` proxy is an explicit setting the
    Desktop could not see; had it seen one it would have injected nothing, so everything it added is
    undone. A ``.env`` NO_PROXY alone replaces the user's own entries in every injected NO_PROXY
    variant, keeping the entries the Desktop added; a ``*`` in it bypasses every proxy.
    """
    from agent.proxy_bypass import no_proxy_entries
    from hermes_cli.env_loader import dotenv_published_values

    env = os.environ if environ is None else environ
    if MARKER not in env:
        return
    if any(dotenv_published_values(name) for name in ("https_proxy", "http_proxy", "all_proxy")):
        restore_user_proxy_env(env)
        return
    published = dotenv_published_values("no_proxy")
    users = next((env[key] for key in env if key.lower() == "no_proxy" and env[key] in published), None)
    if users is None:
        return
    if "*" in no_proxy_entries(users):
        restore_user_proxy_env(env)
        return
    own = {e for key in _NO_PROXY_KEYS for e in no_proxy_entries(env.get(SAVED_PREFIX + key, ""))}
    added = [e for key in _NO_PROXY_KEYS if env.get(key, users) != users
             for e in no_proxy_entries(env[key]) if e not in own]
    merged = ",".join(dict.fromkeys([*no_proxy_entries(users), *added]))
    for key in _NO_PROXY_KEYS:
        if key in env:
            env[key] = merged
