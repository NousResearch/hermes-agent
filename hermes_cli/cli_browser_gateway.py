"""Classic CLI ``/browser`` on the shared gateway (#134889 CLI-9): ``browser.manage``, as Ink and
Desktop send it. Split from ``cli_commands_mixin`` (the file is over its size target)."""
from __future__ import annotations

import os
from urllib.parse import urlparse

from hermes_cli.browser_connect import DEFAULT_BROWSER_CDP_URL
from hermes_cli.cli_commands_session_tools import _say_block, _t


def _gateway_is_remote() -> bool:
    """True when this CLI attaches to an explicitly named gateway WebSocket that is not loopback
    (``HERMES_TUI_GATEWAY_URL``): Desktop's ``mode === 'remote'``."""
    raw = os.environ.get("HERMES_TUI_GATEWAY_URL", "").strip()
    if not raw:
        return False
    host = (urlparse(raw).hostname or "").lower()
    return host not in {"127.0.0.1", "localhost", "::1"}


def gateway_browser(cli, connection, word: str, rest: str) -> None:
    """``/browser`` on the shared gateway (#134889 CLI-9): ``browser.manage``, the verb Ink and
    Desktop send. The browser tools run in the gateway, so changing this process's
    ``BROWSER_CDP_URL`` would change nothing the agent uses. ``connect``/``disconnect`` manage a
    browser on the gateway HOST: only against a local gateway (Desktop's rule). The gateway scopes
    them to this session's profile."""
    from hermes_cli.gateway_client import GatewayClientError
    if word in {"connect", "disconnect"} and _gateway_is_remote():
        return _say_block(_t("browser.remote_gateway"))
    params = {"action": word, "session_id": cli.session_id}
    if word == "connect":
        normalized = _normalize_cdp_url(rest or DEFAULT_BROWSER_CDP_URL)
        if normalized is None:
            return
        params["url"] = normalized[0]
        print(f"   {_t('browser.checking', url=params['url'])}")
    if word == "use":
        mode = rest.lower() or "on"
        if mode not in {"on", "off"}:
            return _say_block(
                _t("browser.use_usage"),
                f"   {_t('browser.use_on_hint')}", f"   {_t('browser.use_off_hint')}")
        params["enabled"] = mode == "on"
    try:
        result = connection.call("browser.manage", **params) or {}
    except GatewayClientError as exc:
        return _say_block(f"   ⚠ {exc}")
    for message in result.get("messages") or ():
        print(f"   {message}")
    if word == "use":
        return _say_block(_t("browser.use_enabled" if params["enabled"] else "browser.use_disabled"),
                          f"   {_t('browser.new_sessions_only')}")
    if word == "disconnect":
        return _say_block(_t("browser.disconnected"), f"   {_t('browser.reverted')}")
    if word == "connect":
        if not result.get("connected"):
            return _say_block(_t("browser.not_connected"))
        return _say_block(_t("browser.connected"), f"   {_t('browser.endpoint', url=result.get('url') or '')}")
    if result.get("browser_use"):
        return _say_block(_t("browser.status_browser_use"), f"   {_t('browser.status_use_off_hint')}")
    if result.get("connected"):
        return _say_block(_t("browser.status_connected"), f"   {_t('browser.endpoint', url=result.get('url') or '')}")
    _say_block(_t("browser.status_gateway_default"),
               f"   {_t('browser.status_connect_hint')}", f"   {_t('browser.status_disconnect_hint')}")


def _normalize_cdp_url(cdp_url: str):
    """Validate a /browser connect URL: ``(cdp_url, port)`` or None after printing the error.
    A ``/devtools/browser/<id>`` path is kept verbatim; anything else is reduced to the origin."""
    parsed = urlparse(cdp_url if "://" in cdp_url else f"http://{cdp_url}")
    if parsed.scheme not in {"http", "https", "ws", "wss"}:
        _say_block(f"   {_t('browser.unsupported_scheme', scheme=parsed.scheme or _t('browser.scheme_missing'))}")
        return None
    try:
        port = parsed.port or (443 if parsed.scheme in {"https", "wss"} else 80)
    except ValueError:
        _say_block(f"   {_t('browser.invalid_port', url=cdp_url)}")
        return None
    if not parsed.hostname:
        _say_block(f"   {_t('browser.missing_host', url=cdp_url)}")
        return None
    if parsed.path.startswith("/devtools/browser/"):
        return parsed.geturl(), port
    return parsed._replace(path="", params="", query="", fragment="").geturl(), port

