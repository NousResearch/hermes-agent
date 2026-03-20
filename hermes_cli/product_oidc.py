"""Provider-neutral OIDC helpers for the hermes-core product layer."""

from __future__ import annotations

import base64
import hashlib
import secrets
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlencode

import httpx

from hermes_cli.config import get_env_value
from hermes_cli.product_config import load_product_config


@dataclass(frozen=True)
class ProductOIDCClientSettings:
    issuer_url: str
    client_id: str
    client_secret: str
    redirect_uri: str
    scopes: tuple[str, ...]


@dataclass(frozen=True)
class ProductOIDCProviderMetadata:
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: str | None = None
    end_session_endpoint: str | None = None
    jwks_uri: str | None = None


def _required_string(value: str, field_name: str) -> str:
    candidate = (value or "").strip()
    if not candidate:
        raise ValueError(f"{field_name} must not be empty")
    return candidate


def load_product_oidc_client_settings(config: Mapping[str, Any] | None = None) -> ProductOIDCClientSettings:
    product_config = dict(config or load_product_config())
    auth = dict(product_config.get("auth", {}))
    network = dict(product_config.get("network", {}))
    public_host = str(network.get("public_host", "localhost")).strip() or "localhost"
    app_port = int(network.get("app_port", 8086))

    issuer_url = _required_string(str(auth.get("issuer_url", "")), "auth.issuer_url")
    client_id = _required_string(str(auth.get("client_id", "")), "auth.client_id")
    client_secret_ref = _required_string(
        str(auth.get("client_secret_ref", "")),
        "auth.client_secret_ref",
    )
    client_secret = _required_string(
        get_env_value(client_secret_ref) or "",
        client_secret_ref,
    )
    return ProductOIDCClientSettings(
        issuer_url=issuer_url.rstrip("/"),
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=f"http://{public_host}:{app_port}/api/auth/oidc/callback",
        scopes=("openid", "profile", "email"),
    )


def discover_product_oidc_provider_metadata(
    settings: ProductOIDCClientSettings,
    *,
    client: httpx.Client | None = None,
) -> ProductOIDCProviderMetadata:
    well_known_url = f"{settings.issuer_url}/.well-known/openid-configuration"
    owns_client = client is None
    http_client = client or httpx.Client(timeout=10.0)
    try:
        response = http_client.get(well_known_url)
        response.raise_for_status()
        payload = response.json()
    finally:
        if owns_client:
            http_client.close()

    issuer = _required_string(str(payload.get("issuer", settings.issuer_url)), "oidc issuer")
    authorization_endpoint = _required_string(
        str(payload.get("authorization_endpoint", "")),
        "authorization_endpoint",
    )
    token_endpoint = _required_string(
        str(payload.get("token_endpoint", "")),
        "token_endpoint",
    )
    userinfo_endpoint = str(payload.get("userinfo_endpoint", "")).strip() or None
    end_session_endpoint = (
        str(payload.get("end_session_endpoint", "")).strip()
        or str(payload.get("end_session_endpoint_uri", "")).strip()
        or None
    )
    jwks_uri = str(payload.get("jwks_uri", "")).strip() or None
    return ProductOIDCProviderMetadata(
        issuer=issuer,
        authorization_endpoint=authorization_endpoint,
        token_endpoint=token_endpoint,
        userinfo_endpoint=userinfo_endpoint,
        end_session_endpoint=end_session_endpoint,
        jwks_uri=jwks_uri,
    )


def create_pkce_verifier() -> str:
    return secrets.token_urlsafe(64)


def create_pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def create_oidc_login_request(
    settings: ProductOIDCClientSettings,
    metadata: ProductOIDCProviderMetadata,
    *,
    state: str | None = None,
    nonce: str | None = None,
    verifier: str | None = None,
    scopes: tuple[str, ...] | None = None,
) -> dict[str, str]:
    chosen_state = state or secrets.token_urlsafe(24)
    chosen_nonce = nonce or secrets.token_urlsafe(24)
    chosen_verifier = verifier or create_pkce_verifier()
    chosen_scopes = scopes or settings.scopes
    code_challenge = create_pkce_challenge(chosen_verifier)
    query = urlencode(
        {
            "client_id": settings.client_id,
            "redirect_uri": settings.redirect_uri,
            "response_type": "code",
            "scope": " ".join(chosen_scopes),
            "state": chosen_state,
            "nonce": chosen_nonce,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
    )
    return {
        "state": chosen_state,
        "nonce": chosen_nonce,
        "verifier": chosen_verifier,
        "authorization_url": f"{metadata.authorization_endpoint}?{query}",
    }


def exchange_product_oidc_code(
    settings: ProductOIDCClientSettings,
    metadata: ProductOIDCProviderMetadata,
    *,
    code: str,
    verifier: str,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    owns_client = client is None
    http_client = client or httpx.Client(timeout=10.0)
    try:
        response = http_client.post(
            metadata.token_endpoint,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": settings.redirect_uri,
                "client_id": settings.client_id,
                "client_secret": settings.client_secret,
                "code_verifier": verifier,
            },
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        return response.json()
    finally:
        if owns_client:
            http_client.close()
