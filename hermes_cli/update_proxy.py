"""Proxy configuration for the updater (``hermes update`` + in-app update).

The updater used to run with no proxy configuration at all: it never set
``http_proxy``/``https_proxy`` and read no proxy setting from config, so on
networks that cut direct TLS every update died mid-handshake while the shipped
``hermes-update-with-vpn.bat`` wrapper (which just exports proxy env for the
child ``hermes update``) succeeded on the same machine (#124022). The built-in
path now does what that wrapper does:

* ``updates.proxy`` (e.g. ``http://127.0.0.1:10808``) is the explicit opt-in;
  ambient ``HTTPS_PROXY``/``HTTP_PROXY``/``ALL_PROXY`` env still works as before.
* :class:`hermes_cli.release_channels.ChannelReader` routes through the
  configured proxy with zero caller changes (the one shared channel function).
* :func:`ensure_update_proxy_env` fills the process env from config at the
  update entry points, so the updater's git subprocesses (which inherit the
  env) and the remaining ``urllib`` call sites pick it up.

# ponytail: explicit config/env only — no automatic probing of localhost
# ports or the Windows system proxy. Auto-routing update traffic through
# whatever happens to listen locally is a product/security decision; add it
# here (probe + health-check, then feed resolve_update_proxy) when a
# maintainer asks for suggestion #2 of #124022.
"""
from __future__ import annotations

import logging
import os
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

PROXY_ENV_KEYS = ("https_proxy", "http_proxy", "HTTPS_PROXY", "HTTP_PROXY")
ALL_PROXY_ENV_KEYS = ("all_proxy", "ALL_PROXY")
# Schemes the updater can hand to both urllib and git/libcurl unchanged.
PROXY_SCHEMES = ("http", "https", "socks5", "socks5h")


def _updates_config() -> dict:
    """The ``updates:`` config section (``{}`` when absent/malformed/unreadable)."""
    try:
        from hermes_cli.config import load_config
        section = (load_config() or {}).get("updates", {})
    except Exception as exc:  # noqa: BLE001 — a proxy lookup must never break an update
        logger.debug("update proxy config unreadable: %s", exc)
        return {}
    return section if isinstance(section, dict) else {}


def valid_proxy_url(value: object) -> str | None:
    """The stripped proxy URL when *value* is a usable ``scheme://host`` URL, else None."""
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return None
    if parsed.scheme.lower() not in PROXY_SCHEMES or not parsed.hostname:
        return None
    return candidate


def configured_proxy() -> str | None:
    """The validated ``updates.proxy`` URL, else None (warns once on a malformed value)."""
    raw = _updates_config().get("proxy")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    proxy = valid_proxy_url(raw)
    if proxy is None:
        logger.warning("Ignoring malformed updates.proxy value %r; updating directly", raw)
    return proxy


def resolve_update_proxy() -> str | None:
    """``updates.proxy`` when set, else the first ambient proxy env value, else None."""
    proxy = configured_proxy()
    if proxy is not None:
        return proxy
    for key in (*PROXY_ENV_KEYS, *ALL_PROXY_ENV_KEYS):
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()
    return None


def proxy_env_overlay(base: dict | None = None) -> dict:
    """*base* (default: the ambient env) with proxy vars filled from ``updates.proxy``.

    Only fills vars that are unset — an explicitly exported proxy (the VPN
    wrapper, the user's shell) always wins over config.
    """
    env = dict(base if base is not None else os.environ)
    proxy = configured_proxy()
    if proxy is None:
        return env
    for key in PROXY_ENV_KEYS:
        env.setdefault(key, proxy)
    return env


def ensure_update_proxy_env() -> str | None:
    """Export ``updates.proxy`` into this process's env (setdefault) for the update run.

    Makes the built-in path do what ``hermes-update-with-vpn.bat`` does for
    the child: git subprocesses inherit it and ``urllib`` honors it. Returns
    the applied proxy URL, or None when nothing was applied.
    """
    proxy = configured_proxy()
    if proxy is None:
        return None
    applied = False
    for key in PROXY_ENV_KEYS:
        if not os.environ.get(key):
            os.environ[key] = proxy
            applied = True
    if applied:
        print(f"-> Using update proxy from updates.proxy: {proxy}")
    return proxy
