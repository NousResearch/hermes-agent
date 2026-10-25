"""``hermes webapp``: open the browser-hosted Desktop that the gateway serves.

The gateway serves the Webapp on its own listener when its home's config sets ``webapp.enabled``
(``gateway.webapp_listener``). This command builds the bundle when it is stale, starts the gateway that
serves the current profile, asks it for a one-use launch link over the control socket (same OS user
only) and opens that link, so the credential appears in no argv, output or link. On Linux the gateway
checks that the browser's connection is the user's; on Windows nothing checks it, and the link travels
only in the browser launcher's command line, which other standard users cannot read. On macOS, the BSDs
and illumos neither holds, so it writes the link into a page only the user can read and opens that page:
the launcher's command line carries only the page's path. Where it cannot open a browser, and on WSL
(whose Windows browser reaches the gateway through a relay any Windows account can use), it prints the
link for the user to open instead.
``--status`` and ``--sign-out`` talk to a running gateway and never start one.
"""
from __future__ import annotations

import html
import json
import os
import secrets
import sys
import time
from pathlib import Path

RESTART_COMMAND = "hermes gateway restart"
# The gateway reads the Webapp's settings and sign-in providers only when it starts.
RETRY_AFTER_FIX = f"The gateway reads these settings when it starts: after fixing them, run {RESTART_COMMAND}."
STARTED_BEFORE_ON = f"The gateway started before the Webapp was turned on. Restart it to serve the Webapp: {RESTART_COMMAND}"
# A gateway answers a verb it does not know this way; the Webapp's are registered before its control
# socket answers, so the gateway running predates them (an upgrade it has not restarted onto).
OLDER_GATEWAY = ("The running gateway is older than the Webapp and does not know its commands. Restart it "
                 f"onto this version of Hermes: {RESTART_COMMAND}")
# How long --sign-out waits for a gateway that is still starting to answer (30 s).
_START_WAIT_ATTEMPTS, _START_WAIT_INTERVAL = 60, 0.5


class GatewayUnreachable(Exception):
    """The gateway for that home did not start or does not answer on its control socket."""


def _gateway_home(home: Path) -> Path:
    """The home whose gateway serves *home*: the default gateway for a named profile it serves (the
    route native Desktop takes), else the profile's own."""
    from hermes_cli.gateway_runtime_multiplex import implied_host_root
    return implied_host_root(home) or home


def _deleted_profile_refusal(home: Path) -> str | None:
    """What ``-p <name>`` says when *home* is a named profile that was deleted or never created, else
    None. ``HERMES_HOME`` set to such a home skips that check, and its gateway would be the default
    one, so the command refuses it the same way rather than act on the default profile."""
    from hermes_cli.profiles import _missing_profile_error
    from hermes_constants import named_profile_home, named_profile_is_live
    profile = named_profile_home(home)
    if profile is None or named_profile_is_live(profile):
        return None
    return str(_missing_profile_error(profile.name))


def _build(args):
    from hermes_cli.config import get_project_root
    from hermes_cli.webapp import prepare_webapp_renderer
    return prepare_webapp_renderer(
        get_project_root(), force=args.force_build, skip_build=args.skip_build,
        explicit=args.force_build or args.build_only)


def _ensure(home) -> None:
    from hermes_cli.gateway_runtime import attachable, ensure_gateway_runtime
    result = ensure_gateway_runtime(home, idle_exit=False)
    if not attachable(result):
        cause = ": ".join(part for part in (result.state, result.detail or result.reason_code) if part)
        raise GatewayUnreachable(f"The gateway did not start ({cause}).")


def _control(home, verb: str, params: dict | None = None) -> dict:
    from hermes_cli.gateway_client import private_control_query
    request = json.dumps({"protocol": 1, "id": 1, "verb": verb, "params": params or {}}).encode() + b"\n"
    try:
        data = private_control_query(Path(home), request)
        return json.loads(data.split(b"\n", 1)[0])
    except Exception as exc:  # no socket, a refused peer, a timeout or a garbled answer
        raise GatewayUnreachable(str(exc)) from exc


def _open(url: str) -> bool:
    import webbrowser
    from hermes_cli.auth_device_flow import _can_open_graphical_browser
    if not _can_open_graphical_browser():
        return False
    try:
        return bool(webbrowser.open(url))
    except (webbrowser.Error, OSError):
        return False


# The page a private launch link reaches the browser in: it moves on to the link at once and names nothing
# else, so the referrer and any script stay out of it.
_LAUNCH_PAGE = ('<!doctype html><meta charset="utf-8"><meta name="referrer" content="no-referrer">'
                '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'">'
                '<meta http-equiv="refresh" content="0;url={url}"><title>Opening Hermes</title>'
                '<p><a href="{url}">Open Hermes Webapp</a></p>\n')


def _open_privately(target: Path, url: str) -> bool:
    """Open *url*, a launch link nobody checks the redeemer of, in a browser without handing it to the
    browser launcher, whose command line other local users can read: write it into a page in *target*'s
    user-only ``webapp`` folder, readable by this OS user alone, and open that page. Pages of earlier
    launches go first, so at most one is left, and its link stops working within the link's lifetime.
    False, leaving no page, on WSL (the Windows browser cannot read the page as this user only), when the
    folder or page cannot be kept user-only, or when no browser opens it."""
    from gateway.webapp_listener import TOKEN_PATH, _close_to_other_users, _readable_by_other_users
    from hermes_constants import is_wsl
    if os.name == "nt" or is_wsl():
        return False
    folder = Path(target) / TOKEN_PATH.parent
    try:
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        _close_to_other_users(folder)
        if _readable_by_other_users(folder):
            return False
        for earlier in folder.glob("launch-*.html"):
            earlier.unlink(missing_ok=True)
        page = folder / f"launch-{secrets.token_hex(8)}.html"
        fd = os.open(page, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(_LAUNCH_PAGE.format(url=html.escape(url, quote=True)))
    except OSError:
        return False
    if _open(page.as_uri()):
        return True
    page.unlink(missing_ok=True)
    return False


def _profile_choice(home: Path, target: Path) -> dict:
    """``{"profile": name}`` when the gateway serves *home* as one of its named profiles, so the
    Webapp opens on the profile this command ran for."""
    from hermes_constants import profile_name_for_home
    if Path(home).resolve() == Path(target).resolve():
        return {}
    name = profile_name_for_home(home)
    return {"profile": name} if name and name != "default" else {}


def older_gateway(response: dict) -> bool:
    """Whether *response* is a gateway's answer to a verb it does not know: it predates the Webapp."""
    return not response.get("ok") and str(response.get("error") or "").startswith("unknown verb:")


def _refusal(response: dict) -> str:
    if older_gateway(response):
        return OLDER_GATEWAY
    error = str(response.get("error") or "The gateway refused the request.")
    kind, _, message = error.partition(": ")
    return message if kind.endswith("Error") and message else error


def _off_message(target: Path, then: str) -> str:
    """Why *target*'s config keeps the Webapp off and what to do about it, then *then*: the
    ``webapp:`` block's own problem when it asks for the Webapp but cannot be used."""
    from gateway.webapp_listener import webapp_settings
    settings = webapp_settings(target)
    if settings.reason:
        return f"The Webapp is off: {settings.reason}. Fix it, then {then}."
    if settings.enabled:
        return STARTED_BEFORE_ON
    return f"The Webapp is off. Set webapp.enabled: true in {target / 'config.yaml'}, then {then}."


def _status(target: Path) -> int:
    try:
        response = _control(target, "webapp-status")
    except GatewayUnreachable:
        from gateway.webapp_listener import webapp_settings
        if webapp_settings(target).enabled:
            print("The gateway is not running. Run hermes webapp to start it.")
        else:
            print(_off_message(target, "run hermes webapp"))
        return 1
    if not response.get("ok"):
        print(_refusal(response))
        return 1
    status = response["result"]
    if status["state"] == "serving":
        print(f"Serving at {status['url']}")
        from gateway.webapp_listener import webapp_settings
        if not webapp_settings(target).enabled:
            print(f"{target / 'config.yaml'} no longer turns the Webapp on: it stops at the next gateway restart "
                  f"({RESTART_COMMAND}).")
    elif status["state"] == "starting":
        print("The gateway is still starting the Webapp. Run hermes webapp --status again in a moment.")
    elif status["state"] == "off":
        print(_off_message(target, f"run {RESTART_COMMAND}"))
    else:
        print(f"Unavailable: {status['reason']}")
        if _restart_applies(status["reason"]):
            print(RETRY_AFTER_FIX)
    return 0


def _restart_applies(reason) -> bool:
    """Whether fixing *reason* (why the gateway does not serve the Webapp) takes a gateway restart. A
    missing or stale bundle does not: the gateway serves one as soon as it is built, and a restart
    keeps the stale mark."""
    from gateway.webapp_listener import NOT_BUILT
    return not str(reason).startswith(NOT_BUILT)


def _still_served(target: Path) -> str | None:
    """What to say when *target*'s config no longer turns the Webapp on but its running gateway, which
    reads the config only when it starts, still serves it; None otherwise. Never starts a gateway."""
    try:
        response = _control(target, "webapp-status")
    except GatewayUnreachable:
        return None
    status = (response.get("result") or {}) if response.get("ok") else {}
    if status.get("state") != "serving":
        return None
    return (f"{target / 'config.yaml'} no longer turns the Webapp on, but the running gateway still serves it at "
            f"{status['url']} until it restarts. Run {RESTART_COMMAND} to apply the config.")


def _signed_out_summary(result: dict) -> str:
    """What a gateway's ``webapp-sign-out`` answer says it did, for the user."""
    if not result.get("serving", True):
        return ("The gateway is not serving the Webapp, so no browser is connected to it. Replaced its stored "
                "browser credential: every browser must sign in again once it serves the Webapp.")
    if "connections_closed" not in result:
        # A gateway started before this Hermes: it replaced the credential and closed the sockets that
        # credential opened, but logins behind the sign-in gate outlive it.
        return (f"Replaced the Webapp's browser credential. Restart the gateway ({RESTART_COMMAND}) to sign "
                "out browsers that signed in with a password or an account too.")
    closed = result["connections_closed"]
    connections = f"closed {closed} browser connection{'' if closed == 1 else 's'}"
    if result.get("sign_in_gate"):
        return (f"Signed out every browser: {connections}. Every browser login behind sign-in has ended, "
                "and the account tokens it held no longer work here: browsers must sign in again. Native apps "
                "signed in to this gateway with their own sign-in stay signed in.")
    return f"Signed out of every Webapp tab: {connections}. Run hermes webapp to open it again."


def _sign_out(target: Path) -> int:
    from gateway.webapp_listener import BrowserCredentialError, browser_session_token
    for attempt in range(_START_WAIT_ATTEMPTS + 1):
        try:
            response = _control(target, "webapp-sign-out")
        except GatewayUnreachable:
            response = None
        if response is not None and response.get("ok"):
            print(_signed_out_summary(response.get("result") or {}))
            return 0
        state = _gateway_state(target)
        if response is None and state == "absent":
            # No gateway is running: replace the stored credential so no browser can use the old one
            # when one starts. Behind the sign-in gate, logins bound to it end too.
            try:
                browser_session_token(target, rotate=True)
            except BrowserCredentialError as exc:
                print(f"Nobody was signed out: {exc}")
                return 1
            print("No gateway is running for this profile. Replaced the stored Webapp credential: every browser "
                  "must sign in again when the gateway next serves the Webapp.")
            return 0
        if state != "starting":
            reason = _refusal(response) if response is not None else f"the gateway did not answer (state: {state})"
            print(f"Nobody was signed out: {reason}")
            return 1
        # A starting gateway may already hold the stored credential in memory, where rewriting the file
        # never reaches it: wait until it answers the sign-out itself.
        if attempt < _START_WAIT_ATTEMPTS:
            time.sleep(_START_WAIT_INTERVAL)
    print(f"The gateway is still starting, so nobody was signed out. Run hermes webapp --sign-out again once "
          f"it is up ({RESTART_COMMAND} if it never comes up).")
    return 1


def _gateway_state(home: Path) -> str:
    """What *home*'s gateway lock and control socket say: ``absent`` (none running), ``starting``,
    ``ready``, or why it cannot be reached."""
    from hermes_cli.gateway_runtime import discover_gateway_endpoint
    return discover_gateway_endpoint(home).state


def _explain_refused_launch(target: Path, response: dict) -> int:
    try:
        status = _control(target, "webapp-status").get("result") or {}
    except GatewayUnreachable:
        status = {}
    if status.get("state") == "off":
        print(STARTED_BEFORE_ON)
    else:
        print(_refusal(response))
        if status.get("state") == "unavailable" and _restart_applies(status.get("reason")):
            print(RETRY_AFTER_FIX)
    return 1


def cmd_webapp(args) -> int:
    from gateway.webapp_listener import webapp_settings
    from hermes_cli.webapp import WebappBuildError
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    if (refusal := _deleted_profile_refusal(home)) is not None:
        print(f"Error: {refusal}", file=sys.stderr)
        return 1
    target = _gateway_home(home)
    if args.status:
        return _status(target)
    if args.sign_out:
        return _sign_out(target)
    if not args.build_only and not webapp_settings(target).enabled:
        print(_still_served(target) or _off_message(target, "run hermes webapp again"))
        return 1
    try:
        _build(args)
    except WebappBuildError as exc:
        print(f"✗ {exc}")
        print("  Build it explicitly:")
        print("    hermes webapp --build-only")
        return 1
    if args.build_only:
        print("Webapp bundle ready.")
        return 0
    try:
        _ensure(target)
        response = _control(target, "webapp-launch", _profile_choice(home, target))
    except GatewayUnreachable as exc:
        print(str(exc))
        return 1
    if not response.get("ok"):
        return _explain_refused_launch(target, response)
    link = response["result"]
    if link.get("sign_in"):
        # The Webapp is behind the dashboard's sign-in: the page asks for it, so its address is no secret.
        if not args.no_open and _open(link["url"]):
            print("Opened the Webapp in your browser. Sign in with your dashboard account.")
        else:
            print(f"Open the Webapp and sign in with your dashboard account: {link['url']}")
        return 0
    if link.get("private"):
        # macOS, the BSDs, illumos, WSL: nothing checks who redeems this one-use link, so it never goes to a
        # browser launcher, whose arguments other users' processes can read. A page only this user can read
        # carries it to the browser; failing that, the user opens it by hand.
        if not args.no_open and _open_privately(target, link["url"]):
            print("Opened the Webapp in your browser.")
            return 0
        print(f"Open this link in your browser within {link['expires_in_seconds']} seconds. It works "
              "once and signs the browser in as you, so keep it private:")
        print(f"  {link['url']}")
        return 0
    if not args.no_open and _open(link["url"]):
        print("Opened the Webapp in your browser.")
        return 0
    print(f"Open the Webapp with this one-use link (valid for {link['expires_in_seconds']} seconds):")
    print(f"  {link['url']}")
    return 0
