"""The browser-hosted Desktop (Webapp) on the gateway's own HTTP listener.

``webapp.enabled`` in the gateway home's ``config.yaml`` turns it on. The gateway then binds its one
listener to ``webapp.host``/``webapp.port`` (a stable port, so the browser's URL survives restarts) and
its dashboard mount serves the Webapp bundle. ``webapp.host`` is ``127.0.0.1``, ``::1`` or ``localhost``
(the binds the sign-in gate leaves local-only) or a wildcard (``0.0.0.0``/``::``), and a wildcard bind
is gated by dashboard sign-in like any non-loopback listener.
``webapp.public_url`` is the address a reverse proxy or tunnel serves the Webapp at: it engages the same
sign-in on a loopback bind, and ``dashboard.public_url`` (``hermes dashboard``'s) never applies here.
Native clients (Desktop, the CLI, Ink) dial the loopback address this module derives from the bind.

The browser's credential is the dashboard session token. With the Webapp on, the gateway keeps that
token in a user-only file in its home, so open tabs keep working across a gateway restart; signing out
replaces the file.
"""
from __future__ import annotations

import ipaddress
import logging
import os
import re
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9120
TOKEN_PATH = Path("webapp") / "session-token"
# secrets.token_urlsafe(32); the launch page refuses any other shape.
_TOKEN_FORMAT = re.compile(r"[A-Za-z0-9_-]{43}")


@dataclass(frozen=True)
class WebappSettings:
    enabled: bool = False
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    #: ``webapp.public_url``: the address a reverse proxy or tunnel serves the Webapp at, or "" for none.
    public_url: str = ""
    #: Why a ``webapp:`` block that asks for the Webapp keeps it off (it cannot be used), else None.
    reason: str | None = None

    @property
    def bind(self) -> tuple[str, int]:
        """The listener's bind address: the configured one with the Webapp on, else a private
        loopback listener on an ephemeral port (main's default)."""
        return (self.host, self.port) if self.enabled else (DEFAULT_HOST, 0)


@dataclass(frozen=True)
class WebappMount:
    """What the gateway's dashboard mount needs to serve the Webapp: the built bundle, the home
    whose browser credential it uses and the Webapp's own public URL ("" for none)."""
    dist: Path
    home: Path
    public_url: str = ""


def _config_document(home: Path) -> dict:
    """*home*'s config: the merged one (managed overlay, ``${ENV}`` expansion) for this process's own
    home, which is always the case in the gateway; otherwise that home's file as written, which only
    ``hermes webapp`` reads, about the gateway it targets, before asking that gateway itself."""
    from hermes_cli.config import load_config_readonly, read_user_config_raw
    from hermes_constants import get_hermes_home
    try:
        own = home.resolve() == Path(get_hermes_home()).resolve()
    except OSError:
        own = False
    return load_config_readonly() if own else read_user_config_raw(home / "config.yaml")


def webapp_settings(home) -> WebappSettings:
    """``webapp:`` from *home*'s config. A missing, unreadable or malformed block reads as off, so a
    typo never exposes the listener; a block that cannot be used carries the reason in ``reason``."""
    path = Path(home) / "config.yaml"
    from ruamel.yaml.error import YAMLError

    def off(reason: str) -> WebappSettings:
        logger.warning("Webapp stays off: %s", reason)
        return WebappSettings(reason=reason)

    try:
        document = _config_document(Path(home))
    except (OSError, ValueError, YAMLError) as exc:  # the gateway's own config load reports it
        return off(f"{path} could not be read ({exc})")
    block = document.get("webapp") if isinstance(document, dict) else None
    if block is None:
        return WebappSettings()
    if not isinstance(block, dict):
        return off(f"webapp in {path} must be a mapping")
    enabled = block.get("enabled", False)
    if enabled is False:
        return WebappSettings()
    if enabled is not True:
        return off(f"webapp.enabled must be true or false in {path}")
    host = block.get("host", DEFAULT_HOST)
    port = block.get("port", DEFAULT_PORT)
    host = _canonical_host(_bare_host(host)) if isinstance(host, str) else None
    if host is None or not (_local_only(host) or _is_wildcard(host)):
        return off(f"webapp.host must be 127.0.0.1, ::1, localhost, 0.0.0.0 or :: in {path}")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        return off(f"webapp.port must be a port number (1-65535) in {path}")
    public_url = _public_url(block.get("public_url"))
    if public_url is None:
        return off(f"webapp.public_url must be an http:// or https:// address with no credentials, "
                   f"query or fragment in {path}")
    return WebappSettings(enabled=True, host=host, port=port, public_url=public_url)


def _bare_host(host: str) -> str:
    """*host* without surrounding space or the brackets a URL puts around an IPv6 address
    (``[::1]`` is ``::1``); brackets around anything else stay and fail the host check."""
    host = host.strip()
    if host.startswith("[") and host.endswith("]") and ":" in host[1:-1] and "[" not in host[1:-1]:
        return host[1:-1]
    return host


def _canonical_host(host: str) -> str:
    """*host* spelled as the listener's sign-in gate compares it: an IP address in its short form
    (``0:0:0:0:0:0:0:1`` is ``::1``), a name in lower case."""
    try:
        return ipaddress.ip_address(host).compressed
    except ValueError:
        return host.lower()


def _local_only(host: str) -> bool:
    """Whether the listener's sign-in gate leaves a bind to *host* open as local-only. It knows
    ``127.0.0.1``, ``::1`` and ``localhost``; any other address, another loopback one included,
    engages sign-in like a public bind."""
    from hermes_cli.web_server_boundary import should_require_auth

    return not should_require_auth(host)


def _public_url(value) -> str | None:
    """``webapp.public_url`` without a trailing slash, "" when unset, None when malformed (not an
    http(s) address, or one carrying credentials, a query or a fragment)."""
    if value is None or value == "":
        return ""
    if not isinstance(value, str) or any(c in value for c in "\"'<> \t\r\n?#@\\"):
        return None
    url = value.rstrip("/")
    try:
        parts = urlsplit(url)
        port = parts.port  # raises ValueError on a malformed port
    except ValueError:
        return None
    return url if parts.scheme in {"http", "https"} and parts.hostname and port != 0 else None


def _is_loopback(host: str) -> bool:
    """Whether native clients can dial *host* itself (any loopback address); which binds the
    sign-in gate leaves open is :func:`_local_only`."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _is_wildcard(host: str) -> bool:
    return host in {"0.0.0.0", "::"}


def dial_host(bind_host: str) -> str:
    """The loopback address native clients dial for a listener bound to *bind_host*."""
    if _is_loopback(bind_host):
        return bind_host
    return "::1" if bind_host == "::" else "127.0.0.1"


def dial_origin(bind_host: str, port: int) -> str:
    """The ``api_origin`` native clients get for a listener bound to *bind_host*."""
    host = dial_host(bind_host)
    return f"http://[{host}]:{port}" if ":" in host else f"http://{host}:{port}"


def listener_plan(home) -> tuple[str, int, WebappMount | None, str | None]:
    """The gateway listener's ``(host, port, webapp, reason)`` for the gateway serving *home*;
    *reason* says why a ``webapp:`` block that asks for the Webapp cannot be used."""
    settings = webapp_settings(home)
    if not settings.enabled:
        return (*settings.bind, None, settings.reason)
    from hermes_cli.config import get_project_root
    from hermes_cli.webapp import webapp_dist_dir
    mount = WebappMount(dist=webapp_dist_dir(get_project_root()), home=Path(home), public_url=settings.public_url)
    return (*settings.bind, mount, None)


def browser_session_token(home, *, rotate: bool = False) -> str:
    """The dashboard session token a Webapp browser holds, kept user-only in *home*.

    Created on first use; *rotate* replaces it, which signs every open tab out. A stored token other
    users could read is replaced the same way, and its folder is closed to them again (POSIX)."""
    from utils import atomic_write_text

    path = Path(home) / TOKEN_PATH
    _close_to_other_users(path.parent)
    if not rotate:
        try:
            token = path.read_text(encoding="utf-8-sig").strip()
        except OSError:
            token = ""
        if _TOKEN_FORMAT.fullmatch(token) and not _readable_by_other_users(path):
            return token
    token = secrets.token_urlsafe(32)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    atomic_write_text(path, token + "\n", mode=0o600)
    return token


def _readable_by_other_users(path: Path) -> bool:
    """POSIX: whether *path* grants group or other access, or belongs to another user. Windows homes
    keep their own ACL."""
    if os.name == "nt":
        return False
    try:
        info = path.stat()
    except OSError:
        return False
    return bool(info.st_mode & 0o077) or info.st_uid != os.geteuid()  # windows-footgun: ok — POSIX-only branch


def _close_to_other_users(folder: Path) -> None:
    """POSIX: make an existing credential folder user-only (0700) again."""
    if os.name == "nt":
        return
    try:
        if folder.stat().st_mode & 0o077:
            folder.chmod(0o700)
    except FileNotFoundError:
        return
    except OSError as exc:  # the token file itself stays 0600
        logger.warning("Could not make %s user-only: %s", folder, exc)


def _one_use_links_work() -> bool:
    """Whether a browser the server opens can redeem a one-use launch link here. Redemption needs
    proof that the browser's connection belongs to the gateway's OS user
    (``web_routers.webapp._peer_is_server_user``) on Linux, or a browser launcher whose arguments other
    users cannot read (Windows). macOS and the BSDs offer neither, so there the link is printed for
    the user to open by hand and spends without that proof."""
    return sys.platform in ("linux", "win32")


#: How every missing-bundle reason starts; ``hermes webapp --status`` asks for no restart after it.
NOT_BUILT = "The Webapp is not built"
#: Why a bundle ``hermes update`` could not rebuild is not served (``hermes_cli.webapp.mark_webapp_stale``).
NOT_REBUILT = (f"{NOT_BUILT} for this version of Hermes: hermes update could not rebuild it. "
               "Run hermes webapp to rebuild it.")


def bundle_stale_reason(dist) -> str | None:
    """:data:`NOT_REBUILT` while the bundle in *dist* is marked out of date, else None."""
    from hermes_cli.webapp import webapp_bundle_stale
    return NOT_REBUILT if webapp_bundle_stale(dist) else None


def bundle_missing_reason(dist) -> str | None:
    """Why the Webapp cannot be served from *dist* (no bundle, or one ``hermes update`` could not
    rebuild), or None once it is built. The listener serves a bundle the moment it appears or is
    rebuilt, so building it needs no gateway restart."""
    if (Path(dist) / "index.html").is_file():
        return bundle_stale_reason(dist)
    return f"{NOT_BUILT} ({Path(dist) / 'index.html'} is missing). Run hermes webapp to build it."


def _bundle_missing(web) -> str | None:
    """:func:`bundle_missing_reason` for the bundle the mounted Webapp serves, or None."""
    dist = getattr(getattr(getattr(web, "app", None), "state", None), "web_dist", None)
    return None if dist is None else bundle_missing_reason(dist)


def register_control_verbs(runner, home) -> None:
    """The private control-socket verbs ``hermes webapp`` uses; only an authenticated same-user peer
    reaches them. ``webapp-launch`` returns a one-use launch link, so the credential crosses neither
    the socket nor a browser's argv; where nothing can check who redeems such a link (macOS) it is
    marked private, for the user to open by hand. ``webapp-sign-out`` replaces the
    browser credential, which also ends every login bound to it behind the sign-in gate, closes the
    browser sockets those opened, and reports how many it closed."""
    server = getattr(runner, "session_control_server", None)
    if server is None:
        return

    def serving():
        status = runner.webapp_status
        if status["state"] == "off":
            raise RuntimeError("The Webapp is off. Set webapp.enabled: true in config.yaml and restart the gateway.")
        if status["state"] != "serving":
            raise RuntimeError(f"The Webapp is unavailable: {status['reason']}")
        web = getattr(getattr(getattr(runner, "session_api", None), "app", None), "web", None)
        if web is None:
            raise RuntimeError("The Webapp is still starting; try again.")
        return status, web

    def webapp_status(_params, _peer):
        # Serving only while the bundle is built; the listener serves one as soon as it appears.
        status = dict(runner.webapp_status)
        missing = _bundle_missing(getattr(getattr(getattr(runner, "session_api", None), "app", None), "web", None))
        if status["state"] == "serving" and missing:
            return {"state": "unavailable", "url": None, "reason": missing}
        return status

    def webapp_launch(params, _peer):
        from urllib.parse import urlencode

        from hermes_cli.web_routers.webapp import (
            LAUNCH_TICKET_TTL_SECONDS, mint_launch_ticket, mint_printed_launch_ticket)
        from hermes_constants import PROFILE_ID_RE
        profile = params.get("profile")
        if profile is not None and not (isinstance(profile, str) and PROFILE_ID_RE.match(profile)):
            raise ValueError("profile must be a profile name")
        query = f"?profile={profile}" if profile else ""  # the page opens on that profile
        status, web = serving()
        missing = _bundle_missing(web)
        if missing:
            raise RuntimeError(f"The Webapp is unavailable: {missing}")
        from hermes_cli.web_server_surface import private_launch
        if not private_launch(web.app.state):
            # Behind the dashboard's sign-in the page asks for it; a launch link opens only a private Webapp.
            # Sign-in returns to a callback the provider must know and checks a state cookie set where it
            # started, so the browser starts at that callback's origin: webapp.public_url when set (the
            # listener's own public URL), else on the LAN the address Web Access registered Nous sign-in
            # for, else the loopback address (a password provider, or sign-in set up by hand for it).
            # Native clients keep the loopback address.
            from hermes_cli.webapp_access import registered_lan_origin
            address = (getattr(web.app.state, "listener_public_url", "")
                       or registered_lan_origin(home, urlsplit(status["url"]).port) or status["url"])
            return {"url": f"{address}/{query}", "private": False, "expires_in_seconds": None, "sign_in": True}
        # Where the browser's connection cannot be checked (macOS, the BSDs), the link is for the user to
        # open by hand ("private": printed, never passed to a browser launcher), and its ticket spends
        # without that check. Either way it works once, for 60 seconds, and carries no lasting credential.
        checked = _one_use_links_work()
        ticket = mint_launch_ticket(web.app.state) if checked else mint_printed_launch_ticket(web.app.state)
        fragment = urlencode({"ticket": ticket, **({"query": query} if query else {})})
        return {"url": f"{status['url']}/webapp/launch#{fragment}", "private": not checked,
                "expires_in_seconds": LAUNCH_TICKET_TTL_SECONDS}

    def webapp_sign_out(_params, _peer):
        from hermes_cli import web_server_file_tickets, web_session_sockets
        if runner.webapp_status["state"] != "serving":
            # No browser reaches this gateway's Webapp and it holds no browser credential: replacing
            # the stored one is the whole sign-out, and the next start serves the new one.
            browser_session_token(home, rotate=True)
            return {"signed_out": True, "serving": False, "sign_in_gate": False, "connections_closed": 0}
        _, web = serving()
        web._SESSION_TOKEN = browser_session_token(home, rotate=True)
        # Every tab loses access at once: the sockets and media links the old credential opened end
        # now, not when each tab next reconnects. Behind the sign-in gate every login is bound to the
        # credential (hermes_cli.webapp_sign_in), so those browsers are signed out and their sockets
        # close too. Native clients never held it and stay connected.
        closed = web_session_sockets.close_all()
        web_server_file_tickets.revoke_all()
        return {"signed_out": True, "serving": True,
                "sign_in_gate": bool(getattr(web.app.state, "auth_required", False)),
                "connections_closed": closed}

    server.private_handlers.update({
        "webapp-status": webapp_status, "webapp-launch": webapp_launch, "webapp-sign-out": webapp_sign_out})
