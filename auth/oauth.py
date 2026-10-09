"""Canonical oauth authentication mechanics; no CLI dependencies."""

from __future__ import annotations
from datetime import datetime, timezone


import base64

import hashlib

import logging

import os

import ssl

import sys

import threading

import time


from http.server import BaseHTTPRequestHandler, HTTPServer


from typing import Any, Callable, Dict, FrozenSet, Optional

from urllib.parse import parse_qs, urlparse

from auth.errors import AuthError

from auth.constants import (
    DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS,
    DEVICE_CODE_GRANT_TYPE,
    httpx,
)

from auth.store_migrations import DEFAULT_NOUS_PORTAL_URL

from utils import is_truthy_value

logger = logging.getLogger("hermes_cli.auth")

_CONSOLE_BROWSER_NAMES: FrozenSet[str] = frozenset({
    "w3m",
    "lynx",
    "links",
    "links2",
    "elinks",
    "www-browser",
    "browsh",  # TUI browser — still hijacks the terminal
})

_REMOTE_IDE_ENV_VARS = (
    "CLOUD_SHELL",  # GCP Cloud Shell
    "CODESPACES",
    "CODESPACE_NAME",  # GitHub Codespaces
    "GITPOD_WORKSPACE_ID",  # Gitpod
    "REPL_ID",  # Replit
    "STACKBLITZ",  # StackBlitz
)


def _pkce_code_verifier(length: int = 64) -> str:
    return (
        base64.urlsafe_b64encode(os.urandom(length)).decode("ascii").rstrip("=")[:128]
    )


def _pkce_code_challenge(code_verifier: str) -> str:
    digest = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _make_loopback_callback_handler(
    expected_path: str,
    *,
    display_name: str,
) -> tuple[type[BaseHTTPRequestHandler], dict[str, Any]]:
    """Handler class for an RFC 8252 loopback redirect plus the dict it fills in.

    Only a GET on *expected_path* is accepted (anything else is a 404 and leaves the result
    untouched), so a nonce embedded in the path acts as the CSRF ``state`` for authorization
    servers that do not echo an explicit ``state`` parameter.
    """
    result: dict[str, Any] = {
        "code": None,
        "state": None,
        "error": None,
        "error_description": None,
    }

    class _LoopbackCallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != expected_path:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not found.")
                return

            params = parse_qs(parsed.query)
            for key in result:
                result[key] = params.get(key, [None])[0]

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            outcome = "failed" if result["error"] else "received"
            self.wfile.write(
                f"<html><body><h1>{display_name} authorization {outcome}.</h1>"
                "You can close this tab.</body></html>".encode("utf-8")
            )

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return _LoopbackCallbackHandler, result


def _bind_loopback_callback_server(
    host: str,
    port: int,
    handler_cls: type[BaseHTTPRequestHandler],
    *,
    err: Callable[..., AuthError],
    bind_failed_code: str,
) -> HTTPServer:
    """Bind the loopback listener up front (``port=0`` = OS-assigned) so the redirect URI sent to
    the authorization server names a port we already own — no probe-close-rebind race."""

    class _ReuseHTTPServer(HTTPServer):
        allow_reuse_address = True

    try:
        return _ReuseHTTPServer((host, port), handler_cls)
    except OSError as exc:
        raise err(
            f"Could not bind callback server on {host}:{port}: {exc}", bind_failed_code
        ) from exc


def _serve_loopback_callback(
    server: HTTPServer,
    result: dict[str, Any],
    *,
    timeout_seconds: float,
    err: Callable[..., AuthError],
    timeout_code: str,
) -> dict[str, Any]:
    """Serve *server* until the redirect lands in *result* or the deadline passes; always closes."""
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True
    )
    thread.start()
    deadline = time.monotonic() + max(5.0, timeout_seconds)
    try:
        while time.monotonic() < deadline:
            if result["code"] or result["error"]:
                return result
            time.sleep(0.1)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)
    raise err("Authorization timed out waiting for the local callback.", timeout_code)


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
    *,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    auth_state: Optional[Dict[str, Any]] = None,
) -> bool | ssl.SSLContext:
    from auth.oauth import _default_verify

    tls_state = auth_state.get("tls") if isinstance(auth_state, dict) else {}
    tls_state = tls_state if isinstance(tls_state, dict) else {}
    effective_insecure = (
        is_truthy_value(insecure, default=False)
        if insecure is not None
        else is_truthy_value(tls_state.get("insecure", False), default=False)
    )
    effective_ca = (
        ca_bundle
        or tls_state.get("ca_bundle")
        or os.getenv("HERMES_CA_BUNDLE")
        or os.getenv("SSL_CERT_FILE")
        or os.getenv("REQUESTS_CA_BUNDLE")
    )
    if effective_insecure:
        return False
    if effective_ca:
        ca_path = str(effective_ca)
        if not os.path.isfile(ca_path):
            logger.warning(
                "CA bundle path does not exist: %s — falling back to default certificates",
                ca_path,
            )
            return _default_verify()
        return ssl.create_default_context(cafile=ca_path)
    return _default_verify()


def _request_device_code(
    client: httpx.Client,
    portal_base_url: str,
    client_id: str,
    scope: Optional[str],
) -> Dict[str, Any]:
    """POST to the device code endpoint. Returns device_code, user_code, etc."""
    response = client.post(
        f"{portal_base_url}/api/oauth/device/code",
        data={"client_id": client_id, **({"scope": scope} if scope else {})},
    )
    response.raise_for_status()
    data = response.json()
    required_fields = [
        "device_code",
        "user_code",
        "verification_uri",
        "verification_uri_complete",
        "expires_in",
        "interval",
    ]
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
        f"  Portal login: {portal}/login"
    )


def _poll_device_token_generic(
    post: Callable[[], "httpx.Response"],
    *,
    expires_in: int,
    poll_interval: int,
    validate_success: Callable[[Dict[str, Any]], None],
    on_non_json_error: Callable[["httpx.Response"], Exception],
    on_error: Callable[["httpx.Response", Dict[str, Any]], Exception],
    on_timeout: Callable[[], Exception],
) -> Dict[str, Any]:
    """RFC 8628 device-code polling loop shared by the Nous and xAI flows.

    ``authorization_pending`` sleeps and retries; ``slow_down`` grows the interval by 1s (cap 30s).
    A non-JSON 408/429/5xx, or a 403 carrying ``x-vercel-mitigated`` (edge/WAF mitigation, never a
    real OAuth error), backs off — honoring ``Retry-After``, capped at 60s and at the device-code
    deadline — instead of aborting a login the user may still be approving. Every other error, a
    non-JSON error body, and the deadline become provider-specific exceptions via the supplied
    factories so each caller keeps its exact error contract.
    """
    deadline = time.monotonic() + max(1, expires_in)
    current_interval = poll_interval
    edge_backoff = (
        0.0  # kept apart from current_interval so slow_down/pending pacing is untouched
    )
    while time.monotonic() < deadline:
        response = post()
        if response.status_code == 200:
            payload = response.json()
            validate_success(payload)
            return payload
        try:
            error_payload = response.json()
        except Exception:
            status = response.status_code
            # Edge/WAF mitigation: back off and keep polling until the device code expires.
            if (
                status in {408, 429}
                or status >= 500
                or (status == 403 and response.headers.get("x-vercel-mitigated"))
            ):
                from agent.retry_utils import parse_retry_after_seconds

                retry_after = parse_retry_after_seconds(response.headers)
                if retry_after is not None:
                    edge_backoff = min(max(current_interval, retry_after), 60)
                else:
                    edge_backoff = min(
                        max(edge_backoff * 2, current_interval * 2, 5), 60
                    )
                time.sleep(max(0.0, min(edge_backoff, deadline - time.monotonic())))
                continue
            response.raise_for_status()
            raise on_non_json_error(response)
        edge_backoff = 0.0
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
    client: httpx.Client,
    portal_base_url: str,
    client_id: str,
    device_code: str,
    expires_in: int,
    poll_interval: int,
) -> Dict[str, Any]:
    """Poll the Nous token endpoint until the user approves or the code expires."""

    def _validate(payload: Dict[str, Any]) -> None:
        if "access_token" not in payload:
            raise ValueError("Token response did not include access_token")

    def _error(_response, error_payload) -> Exception:
        # Plain copy per OAuth error code; the raw ``code: description`` stays on a Details line.
        from auth.oauth import device_flow_error

        return device_flow_error(
            str(error_payload.get("error", "") or ""),
            str(
                error_payload.get("error_description") or "Unknown authentication error"
            ),
        )

    return _poll_device_token_generic(
        lambda: client.post(
            f"{portal_base_url}/api/oauth/token",
            data={
                "grant_type": DEVICE_CODE_GRANT_TYPE,
                "client_id": client_id,
                "device_code": device_code,
            },
        ),
        expires_in=expires_in,
        poll_interval=max(1, min(poll_interval, DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS)),
        validate_success=_validate,
        on_error=_error,
        on_non_json_error=lambda _r: RuntimeError(
            "Token endpoint returned a non-JSON error response"
        ),
        # Enriched at the SOURCE so the CLI login and the dashboard/desktop poller
        # (web_server_oauth._nous_promotion_poller surfaces it to the UI) both inherit the guidance.
        on_timeout=lambda: TimeoutError(
            _nous_device_auth_timeout_message(portal_base_url)
        ),
    )


def _utc_now_z() -> str:
    """Current UTC time as an ISO-8601 string with a ``Z`` suffix (last_refresh format)."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _tls_state_from_verify(verify: Any) -> Dict[str, Any]:
    """Persistable ``tls`` block derived from an httpx ``verify`` value."""
    return {
        "insecure": verify is False,
        "ca_bundle": verify if isinstance(verify, str) else None,
    }


def _last_auth_error_marker(
    provider: str,
    error: "AuthError",
    *,
    reason: str,
    default_code: Optional[str] = None,
) -> Dict[str, Any]:
    """The ``last_auth_error`` record persisted when dead OAuth material is quarantined."""
    return {
        "provider": provider,
        "message": str(error),
        "reason": reason,
        "relogin_required": True,
        "code": error.code if default_code is None else (error.code or default_code),
        "at": datetime.now(timezone.utc).isoformat(),
    }


_FLAT_OAUTH_TOKEN_KEYS = (
    "access_token",
    "refresh_token",
    "expires_at",
    "expires_in",
    "obtained_at",
)


def _quarantine_flat_oauth_state(
    state: Dict[str, Any], provider: str, exc: "AuthError"
) -> None:
    """Strip dead tokens from a flat OAuth state after a terminal runtime refresh failure so
    subsequent calls fail fast without a network retry (mirrors the Nous / xAI / Codex pattern)."""
    for _k in _FLAT_OAUTH_TOKEN_KEYS:
        state.pop(_k, None)
    state["last_auth_error"] = _last_auth_error_marker(
        provider, exc, reason="runtime_refresh_failure", default_code="refresh_failed"
    )


def _coerce_ttl_seconds(expires_in: Any) -> int:
    try:
        return max(0, int(expires_in))
    except Exception:
        return 0


def _optional_base_url(value: Any) -> Optional[str]:
    cleaned = value.strip().rstrip("/") if isinstance(value, str) else ""
    return cleaned or None


DEVICE_FLOW_ERROR_COPY = {
    "expired_token": (
        "The sign-in code expired before it was approved in the browser. Run `{retry}` to get a new code."
    ),
    "access_denied": (
        "Sign-in was declined in the browser. Run `{retry}` to try again, or `hermes model` to pick a "
        "different provider."
    ),
    "invalid_grant": (
        "The sign-in code was not accepted by the server. Run `{retry}` to get a new code."
    ),
    "invalid_client": (
        "The server did not recognize this copy of Hermes. Run `hermes update`, then `{retry}` again."
    ),
}


class SignInCopyError(RuntimeError):
    """Exception whose ``str()`` is already user copy (lead line + ``Details:`` line)."""

    def __init__(self, message: str, *, oauth_error_code: str = "") -> None:
        super().__init__(message)
        self.oauth_error_code = oauth_error_code


def device_flow_error(
    code: str, description: str, *, retry_command: str = "hermes portal"
) -> SignInCopyError:
    """Exception for an OAuth device-flow error code whose text is already user-facing.

    Unknown codes keep the server's description as the lead (it is the only information available)
    but still name the retry command.
    """
    lead = DEVICE_FLOW_ERROR_COPY.get(code, "").format(retry=retry_command)
    if not lead:
        lead = (
            f"Sign-in did not complete: {description or 'the server rejected the request'}. "
            f"Run `{retry_command}` to try again."
        )
    details = f"{code}: {description}" if code else description
    return SignInCopyError(
        f"{lead}\n  Details: {details}" if details else lead, oauth_error_code=code
    )


from functools import partial
from auth.errors import _OAUTH_GRANT_DEAD_CODES

_NOUS_AUTH_MISSING_CODES = frozenset({
    "nous_auth_missing",
    "nous_auth_missing_access_token",
    "nous_auth_missing_refresh_token",
})
_TERMINAL_REFRESH_CODES = {
    "nous": _OAUTH_GRANT_DEAD_CODES | _NOUS_AUTH_MISSING_CODES,
    "openai-codex": _OAUTH_GRANT_DEAD_CODES
    | {"codex_refresh_failed", "codex_auth_missing_refresh_token"},
    "xai-oauth": frozenset({"xai_refresh_failed", "xai_auth_missing_refresh_token"}),
}


def _is_terminal_refresh_error(exc, provider):
    return (
        isinstance(exc, AuthError)
        and exc.provider == provider
        and exc.code in _TERMINAL_REFRESH_CODES.get(provider, frozenset())
        and bool(exc.relogin_required)
    )


_is_terminal_nous_refresh_error = partial(_is_terminal_refresh_error, provider="nous")
_is_terminal_codex_oauth_refresh_error = partial(
    _is_terminal_refresh_error, provider="openai-codex"
)
_is_terminal_xai_oauth_refresh_error = partial(
    _is_terminal_refresh_error, provider="xai-oauth"
)
