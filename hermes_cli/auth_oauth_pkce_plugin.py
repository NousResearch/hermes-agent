"""Declarative OAuth 2.0 Authorization-Code + PKCE login for out-of-tree model-provider plugins.

A plugin declares endpoints and public-client metadata in :class:`OAuthPKCEConfig` and plugs the two
factories into the ``ProviderProfile`` hooks::

    cfg = OAuthPKCEConfig(client_id="…", authorize_url="https://…", token_url="https://…", scopes=("…",))
    ProviderProfile(name="example", auth_type="oauth_external",
                    auth_handler=pkce_auth_handler(cfg), refresh_credential=pkce_refresh_credential(cfg))

Hermes owns the security boundary: HTTPS-only endpoints (plain HTTP only for a loopback-literal host,
i.e. a local development IdP), token endpoint host checked against the same allowlist as the authorize
URL BEFORE any request, S256 PKCE, CSRF ``state`` compared in constant time, an RFC 8252 loopback
listener on the literal ``127.0.0.1`` (explicit port, ``0`` = OS-assigned), persistence as a
``PooledCredential`` and single-use refresh tokens re-read from the store under the auth lock. No token,
``state`` or verifier is ever logged.

Lives in ``hermes_cli`` because everything it drives (loopback helpers, the auth store lock, the pool)
does; every core import is deferred into the callables so a plugin may import this module while
provider discovery is still running inside ``hermes_cli.auth``'s own import.
"""

from __future__ import annotations

import hmac
import logging
import secrets
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
from urllib.parse import urlencode, urlparse

logger = logging.getLogger(__name__)

POOL_SOURCE = "manual:loopback_pkce"  # ``manual:`` prefix = never pruned by load_pool() re-seeding
_LOOPBACK_LITERALS = frozenset({"127.0.0.1", "::1"})


@dataclass(frozen=True)
class OAuthPKCEConfig:
    """Public-client OAuth metadata for one provider. No client secret — PKCE is the proof."""

    client_id: str
    authorize_url: str
    token_url: str
    scopes: Tuple[str, ...] = ()
    redirect_port: int = 0  # 0 = OS-assigned; pin it when the IdP allowlists an exact redirect URI
    redirect_path: str = "/callback"
    audience: Optional[str] = None
    extra_authorize_params: Mapping[str, str] = field(default_factory=dict)
    extra_token_params: Mapping[str, str] = field(default_factory=dict)
    # Hosts the token endpoint may live on; default = the authorize URL's host (and its subdomains).
    allowed_hosts: Tuple[str, ...] = ()
    timeout_seconds: float = 180.0
    label: str = ""


def _err(provider: str, message: str, code: str):
    from hermes_cli.auth_constants import AuthError

    return AuthError(f"{provider}: {message}", provider=provider, code=code)


def _host_allowed(host: str, allowlist: Tuple[str, ...]) -> bool:
    return any(host == apex or host.endswith(f".{apex}") for apex in allowlist)


def _endpoint_host(provider: str, name: str, url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    host = (parsed.hostname or "").lower()
    if not host:
        raise _err(provider, f"OAuth {name} has no host.", "oauth_endpoint_invalid")
    if parsed.scheme != "https" and not (parsed.scheme == "http" and host in _LOOPBACK_LITERALS):
        raise _err(provider, f"OAuth {name} must use HTTPS.", "oauth_endpoint_invalid")
    return host


def validate_config(provider: str, cfg: OAuthPKCEConfig) -> None:
    """Refuse a misdeclared config before any network request (login AND refresh call this)."""
    if not str(cfg.client_id or "").strip():
        raise _err(provider, "OAuth client_id is missing.", "oauth_client_id_missing")
    authorize_host = _endpoint_host(provider, "authorize_url", cfg.authorize_url)
    token_host = _endpoint_host(provider, "token_url", cfg.token_url)
    allowlist = tuple(h.lower() for h in cfg.allowed_hosts) or (authorize_host,)
    if not _host_allowed(token_host, allowlist):
        raise _err(provider, f"OAuth token_url host {token_host!r} is not on the allowlist "
                   f"{sorted(allowlist)}.", "oauth_token_host_rejected")
    if not 0 <= int(cfg.redirect_port) <= 65535:
        raise _err(provider, "OAuth redirect_port must be within 0..65535.", "oauth_redirect_invalid")


def _post_token(provider: str, cfg: OAuthPKCEConfig, data: Dict[str, str], *, code: str) -> Dict[str, Any]:
    """POST the token endpoint and return the rotated pool fields; the payload is never logged."""
    from hermes_cli.auth import _coerce_ttl_seconds, _default_verify, _utc_now_z
    from hermes_cli.auth_constants import httpx

    body = {**cfg.extra_token_params, **data, "client_id": cfg.client_id}
    if cfg.audience:
        body.setdefault("audience", cfg.audience)
    try:
        response = httpx.post(cfg.token_url, data=body, headers={"Accept": "application/json"},
                              timeout=30.0, verify=_default_verify())
    except Exception as exc:
        raise _err(provider, f"OAuth token request failed: {type(exc).__name__}", code) from exc
    if response.status_code >= 400:
        raise _token_http_error(provider, response, code)
    payload = response.json()
    access_token = str(payload.get("access_token") or "").strip()
    if not access_token:
        raise _err(provider, "OAuth token response carried no access_token.", code)
    ttl = _coerce_ttl_seconds(payload.get("expires_in", 0))
    return {
        "access_token": access_token,
        "refresh_token": str(payload.get("refresh_token") or data.get("refresh_token") or "").strip() or None,
        "expires_at_ms": int(time.time() * 1000) + ttl * 1000 if ttl else None,
        "last_refresh": _utc_now_z(),
    }


def _token_http_error(provider: str, response: Any, fallback_code: str):
    """Map a failed token HTTP response. A grant-dead JSON ``error`` value becomes the
    error's ``code`` — the pool's plugin recovery treats those codes as terminal. The
    response body is not logged."""
    from hermes_cli.auth import _OAUTH_GRANT_DEAD_CODES

    error = ""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            error = str(payload.get("error") or "").strip()
    except Exception:
        error = ""
    if error in _OAUTH_GRANT_DEAD_CODES:
        return _err(provider, f"OAuth token request failed with HTTP {response.status_code} ({error}).", error)
    return _err(provider, f"OAuth token request failed with HTTP {response.status_code}.", fallback_code)


def _pool_provider(args: Any) -> str:
    """Canonical profile name for the credential pool. ``args.provider`` may be an alias."""
    raw = str(getattr(args, "provider", "") or "").strip().lower()
    from providers import get_provider_profile
    profile = get_provider_profile(raw)
    return profile.name if profile is not None else raw


def login(provider: str, cfg: OAuthPKCEConfig, *, open_browser: bool = True) -> Dict[str, Any]:
    """Run the browser Authorization-Code + PKCE flow; returns the pool fields for the new grant."""
    from hermes_cli.auth_device_flow import (
        _bind_loopback_callback_server, _can_open_graphical_browser, _make_loopback_callback_handler,
        _pkce_code_challenge, _pkce_code_verifier, _print_loopback_ssh_hint, _serve_loopback_callback)

    validate_config(provider, cfg)
    path = cfg.redirect_path if cfg.redirect_path.startswith("/") else f"/{cfg.redirect_path}"
    err = lambda message, code: _err(provider, message, code)  # noqa: E731
    handler_cls, result = _make_loopback_callback_handler(path, display_name=cfg.label or provider)
    server = _bind_loopback_callback_server(
        "127.0.0.1", int(cfg.redirect_port), handler_cls, err=err, bind_failed_code="oauth_callback_bind_failed")
    redirect_uri = f"http://127.0.0.1:{server.server_address[1]}{path}"
    verifier = _pkce_code_verifier()
    state = secrets.token_urlsafe(32)
    params = {
        **cfg.extra_authorize_params, "client_id": cfg.client_id, "response_type": "code",
        "redirect_uri": redirect_uri, "state": state, "code_challenge": _pkce_code_challenge(verifier),
        "code_challenge_method": "S256"}
    if cfg.scopes:
        params["scope"] = " ".join(cfg.scopes)
    if cfg.audience:
        params["audience"] = cfg.audience
    authorize_url = f"{cfg.authorize_url}{'&' if urlparse(cfg.authorize_url).query else '?'}{urlencode(params)}"

    print(f"\nOpen this URL to authorize Hermes with {cfg.label or provider}:\n  {authorize_url}\n")
    print(f"Waiting for callback on {redirect_uri} (timeout {int(cfg.timeout_seconds)}s, Ctrl+C to cancel)...")
    _print_loopback_ssh_hint(redirect_uri)
    if open_browser and _can_open_graphical_browser():
        try:
            webbrowser.open(authorize_url)
        except Exception:
            print("Could not open the browser automatically; use the URL above.")
    try:
        callback = _serve_loopback_callback(
            server, result, timeout_seconds=cfg.timeout_seconds, err=err, timeout_code="oauth_callback_timeout")
    except KeyboardInterrupt:
        print("\nLogin cancelled.")
        raise SystemExit(130)

    if callback.get("error"):
        raise err(f"authorization failed: {callback.get('error_description') or callback['error']}",
                  "oauth_authorization_denied")
    if not hmac.compare_digest(str(callback.get("state") or ""), state):
        raise err("callback state mismatch — the redirect did not come from this login. Aborting.",
                  "oauth_state_mismatch")
    code = str(callback.get("code") or "").strip()
    if not code:
        raise err("callback carried no authorization code.", "oauth_no_code")
    return _post_token(provider, cfg, {
        "grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri, "code_verifier": verifier,
    }, code="oauth_token_exchange_failed")


def _is_usable(access_token: Any, expires_at_ms: Any, now_ms: int) -> bool:
    return bool(str(access_token or "").strip()) and (expires_at_ms is None or int(expires_at_ms) > now_ms)


def pkce_auth_handler(cfg: OAuthPKCEConfig) -> Callable[[str, Any], bool]:
    """``ProviderProfile.auth_handler`` owning add/status/logout; ``refresh`` is declined so the
    credential pool's generic refresh (which calls :func:`pkce_refresh_credential`) handles it."""

    def handler(action: str, args: Any) -> bool:
        from agent.credential_pool import AUTH_TYPE_OAUTH, PooledCredential, load_pool

        provider = _pool_provider(args)
        if action == "add":
            tokens = login(provider, cfg, open_browser=not getattr(args, "no_browser", False))
            entry = load_pool(provider).add_entry(PooledCredential(
                provider=provider, id=uuid.uuid4().hex[:6], label=cfg.label or provider,
                auth_type=AUTH_TYPE_OAUTH, priority=0, source=POOL_SOURCE, **tokens,
                extra={"oauth_pkce": {"client_id": cfg.client_id, "scope": " ".join(cfg.scopes)}}))
            print(f"Signed in to {cfg.label or provider}; credential {entry.id} added to the pool.")
            return True
        if action == "status":
            entries = load_pool(provider).entries()
            now_ms = int(time.time() * 1000)
            if not entries:
                print(f"{provider}: logged out")
            elif any(_is_usable(e.access_token, e.expires_at_ms, now_ms) for e in entries):
                print(f"{provider}: logged in\n  auth_type: oauth (pkce)\n  credentials: {len(entries)}")
            else:
                print(f"{provider}: expired (needs refresh) — run `hermes auth refresh {provider}`")
            return True
        if action == "logout":
            pool = load_pool(provider)
            count = len(pool.entries())
            for index in range(count, 0, -1):
                pool.remove_index(index)
            print(f"Logged out of {provider} ({count} credential(s) removed)")
            return True
        return False

    return handler


def pkce_refresh_credential(cfg: OAuthPKCEConfig) -> Callable[[Any], Mapping[str, Any]]:
    """``ProviderProfile.refresh_credential``: rotate one pooled row via the refresh_token grant.

    Refresh tokens are single-use, so the store is re-read under the auth lock first: a peer that
    already rotated this row is adopted instead of spending its (now-revoked) refresh token again.
    """

    def refresh(entry: Any) -> Mapping[str, Any]:
        from hermes_cli.auth import _auth_store_lock, read_credential_pool

        provider = str(entry.provider)
        validate_config(provider, cfg)
        with _auth_store_lock():
            on_disk = next((row for row in read_credential_pool(provider)
                            if isinstance(row, dict) and row.get("id") == entry.id), None) or {}
            disk_refresh = str(on_disk.get("refresh_token") or "").strip()
            if disk_refresh and disk_refresh != (entry.refresh_token or "") and _is_usable(
                    on_disk.get("access_token"), on_disk.get("expires_at_ms"), int(time.time() * 1000)):
                logger.debug("%s entry %s: adopting a peer's rotation, refresh token not spent", provider, entry.id)
                return {k: on_disk.get(k) for k in ("access_token", "refresh_token", "expires_at_ms", "last_refresh")}
            refresh_token = disk_refresh or str(entry.refresh_token or "").strip()
            if not refresh_token:
                raise _err(provider, "no refresh_token on the pooled credential.", "oauth_refresh_no_token")
            data = {"grant_type": "refresh_token", "refresh_token": refresh_token}
            if cfg.scopes:
                data["scope"] = " ".join(cfg.scopes)
            return _post_token(provider, cfg, data, code="oauth_refresh_failed")

    return refresh
