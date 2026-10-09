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
from auth.providers.plugin_pkce import OAuthPKCEConfig, POOL_SOURCE, _err, _is_usable, _post_token, validate_config
# Documented external plugin factory; internal refresh consumers use auth directly.
from auth.providers.plugin_pkce import pkce_refresh_credential as pkce_refresh_credential


import hmac
import logging
import secrets
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
from urllib.parse import urlencode, urlparse


  # ``manual:`` prefix = never pruned by load_pool() re-seeding


def login(provider: str, cfg: OAuthPKCEConfig, *, open_browser: bool = True) -> Dict[str, Any]:
    """Run the browser Authorization-Code + PKCE flow; returns the pool fields for the new grant."""
    from auth.oauth import _bind_loopback_callback_server, _make_loopback_callback_handler, _pkce_code_challenge, _pkce_code_verifier, _serve_loopback_callback
    from hermes_cli.auth_device_flow import _can_open_graphical_browser, _print_loopback_ssh_hint

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


def pkce_auth_handler(cfg: OAuthPKCEConfig) -> Callable[[str, Any], bool]:
    """``ProviderProfile.auth_handler`` owning add/status/logout; ``refresh`` is declined so the
    credential pool's generic refresh (which calls :func:`pkce_refresh_credential`) handles it."""

    def handler(action: str, args: Any) -> bool:
        from hermes_cli.config_credentials import credential_pool_environment
        from auth.credential_pool import AUTH_TYPE_OAUTH, PooledCredential, load_pool

        provider = _pool_provider(args)
        if action == "add":
            tokens = login(provider, cfg, open_browser=not getattr(args, "no_browser", False))
            entry = load_pool(provider, environment=credential_pool_environment()).add_entry(PooledCredential(
                provider=provider, id=uuid.uuid4().hex[:6], label=cfg.label or provider,
                auth_type=AUTH_TYPE_OAUTH, priority=0, source=POOL_SOURCE, **tokens,
                extra={"oauth_pkce": {"client_id": cfg.client_id, "scope": " ".join(cfg.scopes)}}))
            print(f"Signed in to {cfg.label or provider}; credential {entry.id} added to the pool.")
            return True
        if action == "status":
            entries = load_pool(provider, environment=credential_pool_environment()).entries()
            now_ms = int(time.time() * 1000)
            if not entries:
                print(f"{provider}: logged out")
            elif any(_is_usable(e.access_token, e.expires_at_ms, now_ms) for e in entries):
                print(f"{provider}: logged in\n  auth_type: oauth (pkce)\n  credentials: {len(entries)}")
            else:
                print(f"{provider}: expired (needs refresh) — run `hermes auth refresh {provider}`")
            return True
        if action == "logout":
            pool = load_pool(provider, environment=credential_pool_environment())
            count = len(pool.entries())
            for index in range(count, 0, -1):
                pool.remove_index(index)
            print(f"Logged out of {provider} ({count} credential(s) removed)")
            return True
        return False

    return handler


def _pool_provider(args: Any) -> str:
    """Canonical profile name for the credential pool. ``args.provider`` may be an alias."""
    raw = str(getattr(args, "provider", "") or "").strip().lower()
    from providers import get_provider_profile
    profile = get_provider_profile(raw)
    return profile.name if profile is not None else raw
