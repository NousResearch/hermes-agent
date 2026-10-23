"""Remote access to the gateway-served Webapp, for Desktop's Web Access pane.

The gateway serves the Webapp (``gateway.webapp_listener``) from the ``webapp:`` block of its home's
``config.yaml``, read when it starts, so this writes that home's config and the caller restarts the
gateway. Sign-in is the user's Nous account (a self-hosted portal client registered with the right
callback); a gated bind without it would keep the Webapp off. Two shapes:

- ``lan``: ``webapp.host: 0.0.0.0``, opened at ``http://<this machine's LAN address>:<port>``; the
  wildcard bind engages the sign-in gate and the callback is rebuilt from each request.
- ``public``: an HTTPS URL the user already runs that forwards to ``127.0.0.1:<port>``.
  ``webapp.public_url`` engages the gate on the loopback bind and fixes the callback.

Of the config it writes only the ``webapp:`` block; ``dashboard.public_url`` belongs to the standalone
dashboard, which may sit behind its own tunnel or proxy. Registering a callback saves the portal
client's id as ``HERMES_DASHBOARD_OAUTH_CLIENT_ID`` in ``.env``, the client the dashboard's Nous
sign-in reads.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from gateway.webapp_listener import DEFAULT_PORT

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_CLIENT_ID_ENV = "HERMES_DASHBOARD_OAUTH_CLIENT_ID"
_REGISTRATION_PATH = Path("webapp") / "sign-in-registration.json"


class WebappAccessError(RuntimeError):
    """Remote access cannot be set up as asked; ``str(exc)`` is the user-facing reason."""


def lan_address() -> Optional[str]:
    """This machine's primary LAN IPv4 address, or None. Connecting a UDP socket only asks the kernel
    which interface routes to a non-local address; no packet is sent."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))
            address = probe.getsockname()[0]
    except OSError:
        return None
    return None if address.startswith("127.") or address == "0.0.0.0" else address


def _public_url(raw: Optional[str]) -> str:
    """``https://host[:port][/prefix]`` without a trailing slash, or WebappAccessError."""
    value = (raw or "").strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.hostname in _LOOPBACK_HOSTS:
        raise WebappAccessError("The public URL must be an https:// address other devices use.")
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise WebappAccessError("The public URL cannot carry credentials, a query or a fragment.")
    return value


def gateway_home() -> Path:
    """The home whose gateway serves the current one; its config holds the Webapp's settings."""
    from hermes_cli.webapp_launcher import _gateway_home
    from hermes_constants import get_hermes_home

    return _gateway_home(get_hermes_home())


def nous_signed_in() -> bool:
    """Whether this home holds a Nous account to register sign-in under (a free-tier guest has none).
    Refresh-free: it reads the auth store and never spends a refresh token."""
    from hermes_cli.auth_nous import get_nous_auth_status_local

    status = get_nous_auth_status_local()
    return bool(status.get("logged_in") and not status.get("free_tier"))


def _registration_path() -> Path:
    """Where this home records the sign-in callback Web Access last registered with the Portal."""
    from hermes_constants import get_hermes_home

    return get_hermes_home() / _REGISTRATION_PATH


def forget_registration() -> None:
    """Drop the record of the callback Web Access registered, so its next start registers again.
    ``hermes dashboard register`` calls this: it updates the same Portal client, and the Portal may
    replace that client's redirect URI."""
    _registration_path().unlink(missing_ok=True)


def _registered(record: dict) -> bool:
    try:
        return json.loads(_registration_path().read_text(encoding="utf-8-sig")) == record
    except (OSError, ValueError):
        return False


def registered_lan_origin(home: Path, port: int) -> Optional[str]:
    """``http://<LAN address>:<port>`` when Web Access last registered Nous sign-in for *home*'s
    Webapp on the LAN at that address and *port*, and the address is still this machine's; else None.
    Sign-in returns only to the callback the Portal has, so ``hermes webapp`` opens this address."""
    try:
        record = json.loads((Path(home) / _REGISTRATION_PATH).read_text(encoding="utf-8-sig"))
        parts = urlsplit(record["callback"])
        recorded_port = parts.port
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    if (parts.scheme, parts.path, parts.query, parts.fragment) != ("http", "/auth/callback", "", ""):
        return None
    if parts.username is not None or recorded_port != port or parts.hostname != lan_address():
        return None
    return f"http://{parts.hostname}:{port}"


def _register_nous_sign_in(callback_url: str) -> None:
    """Register (or update in place) this home's portal client with *callback_url*, as
    ``hermes dashboard register --redirect-uri`` does.

    The client is the dashboard's (``HERMES_DASHBOARD_OAUTH_CLIENT_ID``) and the Portal may replace
    its redirect URI on every registration, so the same callback for the same client and Portal is
    never posted twice: a start with nothing changed asks the Portal nothing."""
    from hermes_cli.auth import resolve_nous_access_token
    from hermes_cli.config import save_env_value
    from hermes_cli.dashboard_register import (
        _env_value,
        _generate_dashboard_name,
        _register_self_hosted_client,
        _resolve_portal_base_url,
    )
    from utils import atomic_write_text

    existing = (_env_value(_CLIENT_ID_ENV) or "").strip() or None
    portal = _resolve_portal_base_url(os.environ.get("HERMES_DASHBOARD_PORTAL_URL"))
    if existing and _registered({"portal": portal, "client_id": existing, "callback": callback_url}):
        return
    try:
        result = _register_self_hosted_client(
            access_token=resolve_nous_access_token(),
            portal_base_url=portal,
            name=None if existing else _generate_dashboard_name(),
            custom_redirect_uri=callback_url, existing_client_id=existing)
    except Exception as exc:
        raise WebappAccessError(f"Couldn't register Nous sign-in: {exc}") from exc
    client_id = str(result["client_id"])
    save_env_value(_CLIENT_ID_ENV, client_id)
    path = _registration_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    record = {"portal": portal, "client_id": client_id, "callback": callback_url}
    atomic_write_text(path, json.dumps(record) + "\n", mode=0o600)


def plan(mode: str, public_url: Optional[str] = None) -> tuple[str, str, str]:
    """Check that *mode* can be served and signed in to; returns ``(host, public_url, callback)``:
    the bind host and the URL to pin for :func:`serve`, and the sign-in callback for
    :func:`register`. Changes nothing, so Start runs it before any slow or lasting step."""
    if mode == "lan":
        address = lan_address()
        if address is None:
            raise WebappAccessError("This machine has no network address other devices can reach.")
        host, url, pinned = "0.0.0.0", f"http://{address}:{_configured_port(gateway_home())}", ""
    elif mode == "public":
        host, url = "127.0.0.1", _public_url(public_url)
        pinned = url
    else:
        raise WebappAccessError(f"Unknown access mode {mode!r}.")
    if not nous_signed_in():
        raise WebappAccessError("Sign in to your Nous account under Providers first.")
    return host, pinned, f"{url}/auth/callback"


def build_bundle() -> None:
    """Build the Webapp bundle the gateway serves, with the function ``hermes webapp`` uses (it
    reuses a current bundle), so Start never turns on a Webapp with nothing to serve."""
    from hermes_cli.config import get_project_root
    from hermes_cli.webapp import WebappBuildError, prepare_webapp_renderer

    try:
        prepare_webapp_renderer(get_project_root())
    except WebappBuildError as exc:
        raise WebappAccessError(
            f"{str(exc).rstrip('.')}. Run hermes webapp --build-only to see the full output, then start again.") from exc


def register(callback_url: str) -> None:
    """Register Nous sign-in for *callback_url*, unless Web Access already registered it for this
    client and Portal, so the callback follows a changed LAN address or URL. Writes no config: it
    waits on the portal, so callers do it before taking the config lock."""
    _register_nous_sign_in(callback_url)


def serve(host: str, public_url: str) -> None:
    """Serve the Webapp on *host* from the gateway's next start. Call :func:`register` first."""
    _write_config(enabled=True, host=host, public_url=public_url)


def turn_off() -> None:
    """Stop serving the Webapp from the gateway's next start."""
    _write_config(enabled=False)


def _write_config(*, enabled: bool, host: str = "", public_url: str = "") -> None:
    """The ``webapp:`` block, and nothing outside it. Starting pins ``webapp.public_url`` for the
    own-URL mode and drops it for LAN, where the callback follows each request; stopping changes
    only ``enabled``, so the next start offers the same address. ``webapp.port`` is the user's:
    a second gateway on this machine needs a port of its own."""
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    webapp = cfg.get("webapp") if isinstance(cfg.get("webapp"), dict) else {}
    webapp.update(enabled=enabled, **({"host": host} if enabled else {}))
    if enabled and public_url:
        webapp["public_url"] = public_url
    elif enabled:
        webapp.pop("public_url", None)
    cfg["webapp"] = webapp
    save_config(cfg)


def _configured_block(home: Path) -> dict:
    """The ``webapp:`` block of *home*'s config, or ``{}`` (``webapp_settings`` reports an unreadable
    config)."""
    from ruamel.yaml.error import YAMLError

    from gateway.webapp_listener import _config_document

    try:
        document = _config_document(home)
    except (OSError, ValueError, YAMLError):
        return {}
    block = document.get("webapp") if isinstance(document, dict) else None
    return block if isinstance(block, dict) else {}


def _configured_public_url(home: Path) -> str:
    """``webapp.public_url`` from *home*'s config, or ``""``."""
    value = _configured_block(home).get("public_url")
    return value.strip().rstrip("/") if isinstance(value, str) else ""


def _configured_port(home: Path) -> int:
    """``webapp.port`` from *home*'s config, whether or not the Webapp is on (the default when unset
    or invalid): the address Web Access shows and registers is the one the gateway will bind."""
    port = _configured_block(home).get("port", DEFAULT_PORT)
    valid = isinstance(port, int) and not isinstance(port, bool) and 1 <= port <= 65535
    return port if valid else DEFAULT_PORT


def _gateway_state(home: Path) -> dict:
    """What *home*'s gateway serves now (``serving``, ``off``, ``unavailable`` with a ``reason``, or
    ``stopped`` when none answers). It reads the config at start, so it can lag until a restart."""
    from hermes_cli.webapp_launcher import GatewayUnreachable, _control

    try:
        response = _control(home, "webapp-status")
    except GatewayUnreachable:
        return {"state": "stopped", "url": None, "reason": None}
    if not response.get("ok"):
        return {"state": "unavailable", "url": None, "reason": str(response.get("error") or "")}
    return dict(response["result"])


def _mode(settings, public_url: str) -> Optional[str]:
    """Who the configured Webapp serves: ``public`` (its own URL pins the callback and Host),
    ``lan`` (a wildcard bind), ``local`` (a loopback bind, as ``hermes webapp`` sets it up), or None
    when it is off."""
    from gateway.webapp_listener import _is_wildcard

    if not settings.enabled:
        return None
    if public_url:
        return "public"
    return "lan" if _is_wildcard(settings.host) else "local"


def access_status() -> dict:
    """Everything the Web Access pane shows, read for the gateway home in scope."""
    from gateway.webapp_listener import webapp_settings

    home = gateway_home()
    settings = webapp_settings(home)
    public_url = _configured_public_url(home)
    return {
        "enabled": settings.enabled,
        "mode": _mode(settings, public_url),
        "port": _configured_port(home),
        "gateway": _gateway_state(home),
        "lan_address": lan_address(),
        "public_url": public_url,
        "nous_signed_in": nous_signed_in(),
    }
