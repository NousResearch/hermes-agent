"""Shared device-code / browser / TLS helpers for interactive OAuth logins.

Split out of ``hermes_cli/auth.py`` and re-exported there; origin helpers are imported lazily
inside each function so ``hermes_cli.auth.<name>`` patches still intercept (and no import cycle).
"""

from __future__ import annotations

import logging
import os
import ssl
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Optional
from urllib.parse import urlparse
from hermes_cli.auth_constants import (
    AuthError, DEFAULT_NOUS_PORTAL_URL, DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS,
    DEVICE_CODE_GRANT_TYPE, OAUTH_OVER_SSH_DOCS_URL, httpx)
from utils import is_truthy_value

# Log-record parity with the origin module (caplog tests pin "hermes_cli.auth").
logger = logging.getLogger("hermes_cli.auth")

# Console/text-mode browsers that ``webbrowser`` will launch INSIDE the terminal, hijacking the
# user's TTY with an unusable text browser. When the resolved browser is one of these we refuse
# to auto-open and fall back to the print-the-URL path, same as a remote session.
_CONSOLE_BROWSER_NAMES: FrozenSet[str] = frozenset({
    "w3m", "lynx", "links", "links2", "elinks", "www-browser",
    "browsh",  # TUI browser — still hijacks the terminal
})

# Browser-only remote IDEs / cloud shells (they don't set SSH_CLIENT / SSH_TTY). Keep this list
# narrow — well-known env vars set by the host platform — so a local shell never trips it.
_REMOTE_IDE_ENV_VARS = (
    "CLOUD_SHELL",  # GCP Cloud Shell
    "CODESPACES", "CODESPACE_NAME",  # GitHub Codespaces
    "GITPOD_WORKSPACE_ID",  # Gitpod
    "REPL_ID",  # Replit
    "STACKBLITZ",  # StackBlitz
)


def _is_remote_session() -> bool:
    """Detect environments where loopback OAuth can't reach the local browser.

    Historically only SSH was checked, but #26923 surfaced that **browser-only remote consoles** (GCP Cloud
    Shell, GitHub Codespaces, AWS EC2 Instance Connect, Gitpod, Replit, etc.) hit the exact same problem —
    the user has a browser on their laptop but the loopback listener is bound on the remote VM that the
    laptop's browser can't reach. These environments typically don't set ``SSH_CLIENT`` / ``SSH_TTY``, so
    the SSH-only check left them with no guidance and no fallback.
    """
    return bool(
        os.getenv("SSH_CLIENT") or os.getenv("SSH_TTY")
        or any(os.getenv(var) for var in _REMOTE_IDE_ENV_VARS))


def _names_console_browser(value: str) -> bool:
    token = value.strip().split()[0] if value.strip() else ""
    return os.path.basename(token).lower() in _CONSOLE_BROWSER_NAMES


def _can_open_graphical_browser() -> bool:
    """Return True only when a *graphical* browser is likely to open.

    On a headless Linux box ``webbrowser.open()`` often resolves to a text-mode browser that takes
    over the terminal. Heuristics: a ``$BROWSER`` naming a console browser refuses; on Linux a
    display server (``$DISPLAY`` / ``$WAYLAND_DISPLAY``) is required unless ``$BROWSER`` is set
    (a console one already returned False, so a set ``$BROWSER`` here is graphical).
    """
    browser_env = os.environ.get("BROWSER", "")
    if browser_env and _names_console_browser(browser_env):
        return False
    if sys.platform.startswith("linux"):
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if not has_display and not browser_env:
            return False
    try:
        controller = webbrowser.get()
    except Exception:
        return False  # No browser resolvable at all → definitely don't auto-open.
    candidate = getattr(controller, "name", "") or getattr(controller, "basename", "") or ""
    return not (candidate and _names_console_browser(candidate))


def _ssh_user_at_host() -> str:
    """Best-effort 'user@hostname' for the SSH tunnel hint; placeholders keep it valid syntax."""
    try:
        import socket as _socket
        hostname = _socket.gethostname() or "<this-host>"
    except OSError:
        hostname = "<this-host>"
    user = os.getenv("USER") or os.getenv("LOGNAME") or "<user>"
    return f"{user}@{hostname}"


def _print_loopback_ssh_hint(redirect_uri: str, *, docs_url: str | None = None) -> None:
    """Print an SSH tunnel hint when a loopback-redirect OAuth flow runs on a remote host.

    The auth server redirects the browser to ``127.0.0.1:<port>/callback``; when the browser is
    on another machine (the SSH case) the redirect needs a local port forward to reach us.
    """
    from hermes_cli.auth import _is_remote_session
    if not _is_remote_session():
        return
    try:
        parsed = urlparse(redirect_uri)
    except Exception:
        return
    host, port = parsed.hostname or "", parsed.port
    if host not in {"127.0.0.1", "::1", "localhost"} or not port:
        return
    divider = "-" * 60
    print(
        f"\n{divider}\nRemote session detected — SSH tunnel required\n{divider}\n"
        f"Hermes is waiting for the OAuth callback on {redirect_uri}\n"
        "but your browser is on a different machine. Run this command\n"
        "in a NEW terminal on your local machine BEFORE opening the URL:\n\n"
        f"  ssh -N -L {port}:127.0.0.1:{port} {_ssh_user_at_host()}\n\n"
        "Then open the authorize URL above in your local browser.")
    if docs_url:
        print(f"Provider docs:      {docs_url}")
    print(f"SSH/jump-box guide: {OAUTH_OVER_SSH_DOCS_URL}\n{divider}\n")


def _default_verify() -> bool | ssl.SSLContext:
    """Platform-aware default SSL verify for httpx clients.

    On macOS with Homebrew Python the system OpenSSL cannot find the system trust store, so pin
    certifi's bundle when importable; elsewhere defer to httpx's built-in default.
    """
    if sys.platform == "darwin":
        try:
            import certifi
            return ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            pass
    return True


def _resolve_verify(
    *, insecure: Optional[bool] = None, ca_bundle: Optional[str] = None,
    auth_state: Optional[Dict[str, Any]] = None) -> bool | ssl.SSLContext:
    from hermes_cli.auth import _default_verify
    tls_state = auth_state.get("tls") if isinstance(auth_state, dict) else {}
    tls_state = tls_state if isinstance(tls_state, dict) else {}
    effective_insecure = (
        is_truthy_value(insecure, default=False) if insecure is not None
        else is_truthy_value(tls_state.get("insecure", False), default=False))
    effective_ca = (
        ca_bundle or tls_state.get("ca_bundle") or os.getenv("HERMES_CA_BUNDLE")
        or os.getenv("SSL_CERT_FILE") or os.getenv("REQUESTS_CA_BUNDLE"))
    if effective_insecure:
        return False
    if effective_ca:
        ca_path = str(effective_ca)
        if not os.path.isfile(ca_path):
            logger.warning(
                "CA bundle path does not exist: %s — falling back to default certificates",
                ca_path)
            return _default_verify()
        return ssl.create_default_context(cafile=ca_path)
    return _default_verify()


def _request_device_code(
    client: httpx.Client, portal_base_url: str, client_id: str, scope: Optional[str],
) -> Dict[str, Any]:
    """POST to the device code endpoint. Returns device_code, user_code, etc."""
    response = client.post(
        f"{portal_base_url}/api/oauth/device/code",
        data={"client_id": client_id, **({"scope": scope} if scope else {})})
    response.raise_for_status()
    data = response.json()
    required_fields = [
        "device_code", "user_code", "verification_uri", "verification_uri_complete", "expires_in",
        "interval"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        raise ValueError(f"Device code response missing fields: {', '.join(missing)}")
    return data


def _nous_device_auth_timeout_message(portal_base_url: str) -> str:
    """Actionable timeout text: the usual cause is Portal sign-in failing in the browser tab.

    A bare "Timed out waiting for device authorization" gives the user nothing to act on. The most common
    cause is Portal sign-in failing in the opened browser tab (including the server-side CAPTCHA loop from
    20605), so point at the Portal login page and the retry command. See #20605.
    """
    portal = (portal_base_url or DEFAULT_NOUS_PORTAL_URL).rstrip("/")
    return (
        "Timed out waiting for device authorization.\n"
        "  Portal sign-in is required before the device code can be approved.\n"
        "  If the browser showed a CAPTCHA / 'You did not pass CAPTCHA' error,\n"
        "  finish signing in at the Portal in a normal browser tab, then retry:\n"
        "    hermes portal\n"
        f"  Portal login: {portal}/login")


def _print_device_code_instructions(
    verification_url: str, user_code: str, *, open_browser: bool, failure_dash: str = "--",
    swallow_open_errors: bool = False) -> None:
    """Print the shared "To continue" device-code block and optionally open the browser.

    Callers decide *whether* to open (remote-session / graphical-browser gating differs per
    provider); *failure_dash* keeps each provider's historical hint wording.
    """
    print()
    print("To continue:")
    print(f"  1. Open: {verification_url}")
    print(f"  2. If prompted, enter code: {user_code}")
    if not open_browser:
        return
    try:
        opened = webbrowser.open(verification_url)
    except Exception:
        if not swallow_open_errors:
            raise
        opened = False
    if opened:
        print("  (Opened browser for verification)")
    else:
        print(f"  Could not open browser automatically {failure_dash} use the URL above.")


def _poll_device_token_generic(
    post: Callable[[], "httpx.Response"], *, expires_in: int, poll_interval: int,
    validate_success: Callable[[Dict[str, Any]], None],
    on_non_json_error: Callable[["httpx.Response"], Exception],
    on_error: Callable[["httpx.Response", Dict[str, Any]], Exception],
    on_timeout: Callable[[], Exception]) -> Dict[str, Any]:
    """RFC 8628 device-code polling loop shared by the Nous and xAI flows.

    ``authorization_pending`` sleeps and retries; ``slow_down`` grows the interval by 1s (cap 30s).
    Every other error, a non-JSON error body, and the deadline become provider-specific exceptions
    via the supplied factories so each caller keeps its exact error contract.
    """
    deadline = time.monotonic() + max(1, expires_in)
    current_interval = poll_interval
    while time.monotonic() < deadline:
        response = post()
        if response.status_code == 200:
            payload = response.json()
            validate_success(payload)
            return payload
        try:
            error_payload = response.json()
        except Exception:
            response.raise_for_status()
            raise on_non_json_error(response)
        error_code = str(error_payload.get("error") or "")
        if error_code == "authorization_pending":
            time.sleep(current_interval)
            continue
        if error_code == "slow_down":
            current_interval = min(current_interval + 1, 30)
            time.sleep(current_interval)
            continue
        raise on_error(response, error_payload)
    raise on_timeout()


def _poll_for_token(
    client: httpx.Client, portal_base_url: str, client_id: str, device_code: str,
    expires_in: int, poll_interval: int) -> Dict[str, Any]:
    """Poll the Nous token endpoint until the user approves or the code expires."""
    def _validate(payload: Dict[str, Any]) -> None:
        if "access_token" not in payload:
            raise ValueError("Token response did not include access_token")

    def _error(_response, error_payload) -> Exception:
        error_code = error_payload.get("error", "")
        description = error_payload.get("error_description") or "Unknown authentication error"
        return RuntimeError(f"{error_code}: {description}")

    return _poll_device_token_generic(
        lambda: client.post(
            f"{portal_base_url}/api/oauth/token",
            data={
                "grant_type": DEVICE_CODE_GRANT_TYPE, "client_id": client_id,
                "device_code": device_code}),
        expires_in=expires_in,
        poll_interval=max(1, min(poll_interval, DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS)),
        validate_success=_validate, on_error=_error,
        on_non_json_error=lambda _r: RuntimeError(
            "Token endpoint returned a non-JSON error response"),
        # Enriched at the SOURCE so the CLI login and the dashboard/desktop poller
        # (web_server._nous_poller surfaces str(e) to the UI) both inherit the guidance.
        on_timeout=lambda: TimeoutError(_nous_device_auth_timeout_message(portal_base_url)))


def _prompt_yes_no(prompt: str, *, default: str) -> bool:
    """``input()`` a [Y/n]-style question; EOF/Ctrl-C count as *default*."""
    try:
        answer = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = default
    return answer in {"", "y", "yes"} if default == "y" else answer in {"y", "yes"}


def _print_login_success(
    provider_id: str, config_path: Path, *, show_auth_state: bool = False) -> None:
    print()
    print("Login successful!")
    if show_auth_state:
        from hermes_constants import display_hermes_home as _dhh
        print(f"  Auth state: {_dhh()}/auth.json")
    print(f"  Config updated: {config_path} (model.provider={provider_id})")


def _offer_existing_oauth_credentials(
    provider_id: str, *, resolve: Callable[[], Dict[str, Any]],
    is_expiring: Callable[[str, int], bool], display_name: str, default_base_url: str,
    expired_notice: Optional[str] = None) -> bool:
    """Offer to reuse still-valid stored OAuth credentials. Returns True when the user accepted.

    *resolve* attempts a refresh, so a resolved token should be valid — but double-check the
    expiry before telling the user "Login successful!".
    """
    from hermes_cli.auth import _update_config_for_provider
    try:
        existing = resolve()
        api_key = existing.get("api_key", "")
        if isinstance(api_key, str) and api_key and not is_expiring(api_key, 60):
            print(f"Existing {display_name} credentials found in Hermes auth store.")
            if _prompt_yes_no("Use existing credentials? [Y/n]: ", default="y"):
                config_path = _update_config_for_provider(
                    provider_id, existing.get("base_url", default_base_url))
                _print_login_success(provider_id, config_path)
                return True
        elif expired_notice:
            print(expired_notice)
    except AuthError:
        pass
    return False
