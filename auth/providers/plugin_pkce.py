"""Canonical plugin_pkce authentication mechanics; no CLI dependencies."""

from __future__ import annotations


import logging


import time


from dataclasses import dataclass, field

from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from urllib.parse import urlparse

logger = logging.getLogger(__name__)

POOL_SOURCE = "manual:loopback_pkce"

_LOOPBACK_LITERALS = frozenset({"127.0.0.1", "::1"})


@dataclass(frozen=True)
class OAuthPKCEConfig:
    """Public-client OAuth metadata for one provider. No client secret — PKCE is the proof."""

    client_id: str
    authorize_url: str
    token_url: str
    scopes: Tuple[str, ...] = ()
    redirect_port: int = (
        0  # 0 = OS-assigned; pin it when the IdP allowlists an exact redirect URI
    )
    redirect_path: str = "/callback"
    audience: Optional[str] = None
    extra_authorize_params: Mapping[str, str] = field(default_factory=dict)
    extra_token_params: Mapping[str, str] = field(default_factory=dict)
    # Hosts the token endpoint may live on; default = the authorize URL's host (and its subdomains).
    allowed_hosts: Tuple[str, ...] = ()
    timeout_seconds: float = 180.0
    label: str = ""


def _err(provider: str, message: str, code: str):
    from auth.errors import AuthError

    return AuthError(f"{provider}: {message}", provider=provider, code=code)


def _host_allowed(host: str, allowlist: Tuple[str, ...]) -> bool:
    return any(host == apex or host.endswith(f".{apex}") for apex in allowlist)


def _endpoint_host(provider: str, name: str, url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    host = (parsed.hostname or "").lower()
    if not host:
        raise _err(provider, f"OAuth {name} has no host.", "oauth_endpoint_invalid")
    if parsed.scheme != "https" and not (
        parsed.scheme == "http" and host in _LOOPBACK_LITERALS
    ):
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
        raise _err(
            provider,
            f"OAuth token_url host {token_host!r} is not on the allowlist "
            f"{sorted(allowlist)}.",
            "oauth_token_host_rejected",
        )
    if not 0 <= int(cfg.redirect_port) <= 65535:
        raise _err(
            provider,
            "OAuth redirect_port must be within 0..65535.",
            "oauth_redirect_invalid",
        )


def _post_token(
    provider: str, cfg: OAuthPKCEConfig, data: Dict[str, str], *, code: str
) -> Dict[str, Any]:
    """POST the token endpoint and return the rotated pool fields; the payload is never logged."""
    from auth.oauth import _coerce_ttl_seconds, _default_verify, _utc_now_z
    from auth.constants import httpx

    body = {**cfg.extra_token_params, **data, "client_id": cfg.client_id}
    if cfg.audience:
        body.setdefault("audience", cfg.audience)
    try:
        response = httpx.post(
            cfg.token_url,
            data=body,
            headers={"Accept": "application/json"},
            timeout=30.0,
            verify=_default_verify(),
        )
    except Exception as exc:
        raise _err(
            provider, f"OAuth token request failed: {type(exc).__name__}", code
        ) from exc
    if response.status_code >= 400:
        raise _token_http_error(provider, response, code)
    payload = response.json()
    access_token = str(payload.get("access_token") or "").strip()
    if not access_token:
        raise _err(provider, "OAuth token response carried no access_token.", code)
    ttl = _coerce_ttl_seconds(payload.get("expires_in", 0))
    return {
        "access_token": access_token,
        "refresh_token": str(
            payload.get("refresh_token") or data.get("refresh_token") or ""
        ).strip()
        or None,
        "expires_at_ms": int(time.time() * 1000) + ttl * 1000 if ttl else None,
        "last_refresh": _utc_now_z(),
    }


def _token_http_error(provider: str, response: Any, fallback_code: str):
    """Map a failed token HTTP response. A grant-dead JSON ``error`` value becomes the
    error's ``code`` — the pool's plugin recovery treats those codes as terminal. The
    response body is not logged."""
    from auth.errors import _OAUTH_GRANT_DEAD_CODES

    error = ""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            error = str(payload.get("error") or "").strip()
    except Exception:
        error = ""
    if error in _OAUTH_GRANT_DEAD_CODES:
        return _err(
            provider,
            f"OAuth token request failed with HTTP {response.status_code} ({error}).",
            error,
        )
    return _err(
        provider,
        f"OAuth token request failed with HTTP {response.status_code}.",
        fallback_code,
    )


def _is_usable(access_token: Any, expires_at_ms: Any, now_ms: int) -> bool:
    return bool(str(access_token or "").strip()) and (
        expires_at_ms is None or int(expires_at_ms) > now_ms
    )


def pkce_refresh_credential(cfg: OAuthPKCEConfig) -> Callable[[Any], Mapping[str, Any]]:
    """``ProviderProfile.refresh_credential``: rotate one pooled row via the refresh_token grant.

    Refresh tokens are single-use, so the store is re-read under the auth lock first: a peer that
    already rotated this row is adopted instead of spending its (now-revoked) refresh token again.
    """

    def refresh(entry: Any) -> Mapping[str, Any]:
        from auth.store import _auth_store_lock
        from auth.pool_persistence import read_credential_pool

        provider = str(entry.provider)
        validate_config(provider, cfg)
        with _auth_store_lock():
            on_disk = (
                next(
                    (
                        row
                        for row in read_credential_pool(provider)
                        if isinstance(row, dict) and row.get("id") == entry.id
                    ),
                    None,
                )
                or {}
            )
            disk_refresh = str(on_disk.get("refresh_token") or "").strip()
            if (
                disk_refresh
                and disk_refresh != (entry.refresh_token or "")
                and _is_usable(
                    on_disk.get("access_token"),
                    on_disk.get("expires_at_ms"),
                    int(time.time() * 1000),
                )
            ):
                logger.debug(
                    "%s entry %s: adopting a peer's rotation, refresh token not spent",
                    provider,
                    entry.id,
                )
                return {
                    k: on_disk.get(k)
                    for k in (
                        "access_token",
                        "refresh_token",
                        "expires_at_ms",
                        "last_refresh",
                    )
                }
            refresh_token = disk_refresh or str(entry.refresh_token or "").strip()
            if not refresh_token:
                raise _err(
                    provider,
                    "no refresh_token on the pooled credential.",
                    "oauth_refresh_no_token",
                )
            data = {"grant_type": "refresh_token", "refresh_token": refresh_token}
            if cfg.scopes:
                data["scope"] = " ".join(cfg.scopes)
            return _post_token(provider, cfg, data, code="oauth_refresh_failed")

    return refresh
