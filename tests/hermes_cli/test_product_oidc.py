from urllib.parse import parse_qs, urlparse

import httpx

from hermes_cli.product_config import load_product_config
from hermes_cli.product_oidc import (
    create_oidc_login_request,
    create_pkce_challenge,
    discover_product_oidc_provider_metadata,
    exchange_product_oidc_code,
    load_product_oidc_client_settings,
)


def test_load_product_oidc_client_settings_reads_product_config_and_secret(monkeypatch):
    config = load_product_config()
    config["network"]["public_host"] = "officebox.local"
    config["auth"]["issuer_url"] = "http://officebox.local:1411"
    config["auth"]["client_id"] = "hermes-core"

    monkeypatch.setattr("hermes_cli.product_oidc.get_env_value", lambda key: "oidc-secret")

    settings = load_product_oidc_client_settings(config)

    assert settings.issuer_url == "http://officebox.local:1411"
    assert settings.client_id == "hermes-core"
    assert settings.client_secret == "oidc-secret"
    assert settings.redirect_uri == "http://officebox.local:8086/api/auth/oidc/callback"
    assert settings.scopes == ("openid", "profile", "email")


def test_discover_product_oidc_provider_metadata_uses_well_known(monkeypatch):
    monkeypatch.setattr("hermes_cli.product_oidc.get_env_value", lambda key: "oidc-secret")

    def _handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://officebox.local:1411/.well-known/openid-configuration"
        return httpx.Response(
            200,
            json={
                "issuer": "http://officebox.local:1411",
                "authorization_endpoint": "http://officebox.local:1411/authorize",
                "token_endpoint": "http://officebox.local:1411/token",
                "userinfo_endpoint": "http://officebox.local:1411/userinfo",
                "jwks_uri": "http://officebox.local:1411/jwks",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(_handler))
    settings = load_product_oidc_client_settings(
        {
            "auth": {
                "issuer_url": "http://officebox.local:1411",
                "client_id": "hermes-core",
                "client_secret_ref": "HERMES_PRODUCT_OIDC_CLIENT_SECRET",
            },
            "network": {"public_host": "officebox.local", "app_port": 8086},
        }
    )

    metadata = discover_product_oidc_provider_metadata(settings, client=client)

    assert metadata.authorization_endpoint == "http://officebox.local:1411/authorize"
    assert metadata.token_endpoint == "http://officebox.local:1411/token"
    assert metadata.userinfo_endpoint == "http://officebox.local:1411/userinfo"
    assert metadata.jwks_uri == "http://officebox.local:1411/jwks"


def test_create_oidc_login_request_uses_pkce_and_standard_scopes(monkeypatch):
    monkeypatch.setattr("hermes_cli.product_oidc.secrets.token_urlsafe", lambda _n=0: "fixed-token")
    monkeypatch.setattr("hermes_cli.product_oidc.get_env_value", lambda key: "oidc-secret")
    settings = load_product_oidc_client_settings(
        {
            "auth": {
                "issuer_url": "http://officebox.local:1411",
                "client_id": "hermes-core",
                "client_secret_ref": "HERMES_PRODUCT_OIDC_CLIENT_SECRET",
            },
            "network": {"public_host": "officebox.local", "app_port": 8086},
        }
    )
    metadata = discover_product_oidc_provider_metadata(
        settings,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={
                        "issuer": "http://officebox.local:1411",
                        "authorization_endpoint": "http://officebox.local:1411/authorize",
                        "token_endpoint": "http://officebox.local:1411/token",
                    },
                )
            )
        ),
    )

    login = create_oidc_login_request(
        settings,
        metadata,
        state="state-123",
        nonce="nonce-123",
        verifier="verifier-123",
    )

    parsed = urlparse(login["authorization_url"])
    params = parse_qs(parsed.query)
    assert login["state"] == "state-123"
    assert login["nonce"] == "nonce-123"
    assert login["verifier"] == "verifier-123"
    assert params["client_id"] == ["hermes-core"]
    assert params["response_type"] == ["code"]
    assert params["scope"] == ["openid profile email"]
    assert params["code_challenge_method"] == ["S256"]
    assert params["code_challenge"] == [create_pkce_challenge("verifier-123")]


def test_exchange_product_oidc_code_posts_expected_token_request(monkeypatch):
    monkeypatch.setattr("hermes_cli.product_oidc.get_env_value", lambda key: "oidc-secret")
    seen = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.read().decode("utf-8")
        return httpx.Response(
            200,
            json={"access_token": "access", "id_token": "id-token", "token_type": "Bearer"},
        )

    client = httpx.Client(transport=httpx.MockTransport(_handler))
    settings = load_product_oidc_client_settings(
        {
            "auth": {
                "issuer_url": "http://officebox.local:1411",
                "client_id": "hermes-core",
                "client_secret_ref": "HERMES_PRODUCT_OIDC_CLIENT_SECRET",
            },
            "network": {"public_host": "officebox.local", "app_port": 8086},
        }
    )
    metadata = discover_product_oidc_provider_metadata(
        settings,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={
                        "issuer": "http://officebox.local:1411",
                        "authorization_endpoint": "http://officebox.local:1411/authorize",
                        "token_endpoint": "http://officebox.local:1411/token",
                    },
                )
            )
        ),
    )

    tokens = exchange_product_oidc_code(
        settings,
        metadata,
        code="auth-code",
        verifier="verifier-123",
        client=client,
    )

    assert seen["url"] == "http://officebox.local:1411/token"
    assert "grant_type=authorization_code" in seen["body"]
    assert "code=auth-code" in seen["body"]
    assert "code_verifier=verifier-123" in seen["body"]
    assert "client_secret=oidc-secret" in seen["body"]
    assert tokens["access_token"] == "access"
