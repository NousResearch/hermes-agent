from fastapi.testclient import TestClient

from hermes_cli.product_app import create_product_app


def test_product_app_index_shows_login_link_when_signed_out(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.product_app.load_product_config",
        lambda: {"product": {"brand": {"name": "Hermes Core"}}},
    )
    monkeypatch.setattr(
        "hermes_cli.product_app.resolve_product_urls",
        lambda config: {"app_base_url": "http://officebox.local:8086", "issuer_url": "http://officebox.local:1411"},
    )
    monkeypatch.setattr("hermes_cli.product_app._session_secret", lambda: "test-secret")

    client = TestClient(create_product_app())
    response = client.get("/")

    assert response.status_code == 200
    assert "Sign in with Pocket ID" in response.text
    assert "Your Agent" in response.text
    assert "User Management" in response.text
    assert 'id="sessionCard"' not in response.text
    assert 'id="chatForm"' in response.text


def test_product_app_healthz_reports_auth_provider(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.product_app.load_product_config",
        lambda: {
            "auth": {"provider": "pocket-id", "issuer_url": "http://officebox.local:1411"},
            "network": {"public_host": "officebox.local", "app_port": 8086, "pocket_id_port": 1411},
        },
    )
    monkeypatch.setattr(
        "hermes_cli.product_app.resolve_product_urls",
        lambda config: {
            "app_base_url": "http://officebox.local:8086",
            "issuer_url": "http://officebox.local:1411",
        },
    )
    monkeypatch.setattr("hermes_cli.product_app._session_secret", lambda: "test-secret")

    client = TestClient(create_product_app())
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "auth_provider": "pocket-id",
        "issuer_url": "http://officebox.local:1411",
        "app_base_url": "http://officebox.local:8086",
    }


def test_product_app_login_redirects_and_stores_pending_pkce(monkeypatch):
    monkeypatch.setattr("hermes_cli.product_app.load_product_config", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.product_app.resolve_product_urls",
        lambda config: {"app_base_url": "http://officebox.local:8086", "issuer_url": "http://officebox.local:1411"},
    )
    monkeypatch.setattr("hermes_cli.product_app._session_secret", lambda: "test-secret")
    monkeypatch.setattr(
        "hermes_cli.product_app.load_product_oidc_client_settings",
        lambda: object(),
    )
    monkeypatch.setattr(
        "hermes_cli.product_app.discover_product_oidc_provider_metadata",
        lambda settings: object(),
    )
    monkeypatch.setattr(
        "hermes_cli.product_app.create_oidc_login_request",
        lambda settings, metadata: {
            "state": "state-123",
            "nonce": "nonce-123",
            "verifier": "verifier-123",
            "authorization_url": "http://officebox.local:1411/authorize?client_id=hermes-core",
        },
    )

    client = TestClient(create_product_app())
    response = client.get("/api/auth/login", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "http://officebox.local:1411/authorize?client_id=hermes-core"

    session = client.get("/api/auth/session")
    assert session.json() == {"authenticated": False, "user": None}


def test_product_app_callback_establishes_session(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.product_app.load_product_config",
        lambda: {"bootstrap": {"first_admin_username": "admin"}},
    )
    monkeypatch.setattr(
        "hermes_cli.product_app.resolve_product_urls",
        lambda config: {"app_base_url": "http://officebox.local:8086", "issuer_url": "http://officebox.local:1411"},
    )
    monkeypatch.setattr("hermes_cli.product_app._session_secret", lambda: "test-secret")
    monkeypatch.setattr("hermes_cli.product_app.load_product_oidc_client_settings", lambda: object())
    monkeypatch.setattr("hermes_cli.product_app.discover_product_oidc_provider_metadata", lambda settings: object())
    monkeypatch.setattr(
        "hermes_cli.product_app.create_oidc_login_request",
        lambda settings, metadata: {
            "state": "state-123",
            "nonce": "nonce-123",
            "verifier": "verifier-123",
            "authorization_url": "http://officebox.local:1411/authorize?client_id=hermes-core",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.product_app.exchange_product_oidc_code",
        lambda settings, metadata, code, verifier: {"access_token": "access-token"},
    )
    monkeypatch.setattr(
        "hermes_cli.product_app.fetch_product_oidc_userinfo",
        lambda access_token, metadata: {
            "sub": "user-1",
            "email": "admin@example.com",
            "name": "Admin User",
            "preferred_username": "admin",
            "email_verified": True,
        },
    )

    client = TestClient(create_product_app())
    client.get("/api/auth/login", follow_redirects=False)
    callback = client.get("/api/auth/oidc/callback?code=auth-code&state=state-123", follow_redirects=False)

    assert callback.status_code == 303
    assert callback.headers["location"] == "http://officebox.local:8086"

    session = client.get("/api/auth/session")
    assert session.status_code == 200
    assert session.json() == {
        "authenticated": True,
        "user": {
            "sub": "user-1",
            "email": "admin@example.com",
            "name": "Admin User",
            "preferred_username": "admin",
            "email_verified": True,
            "is_admin": True,
        },
    }


def test_product_app_callback_rejects_state_mismatch(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.product_app.load_product_config",
        lambda: {"bootstrap": {"first_admin_username": "admin"}},
    )
    monkeypatch.setattr(
        "hermes_cli.product_app.resolve_product_urls",
        lambda config: {"app_base_url": "http://officebox.local:8086", "issuer_url": "http://officebox.local:1411"},
    )
    monkeypatch.setattr("hermes_cli.product_app._session_secret", lambda: "test-secret")
    monkeypatch.setattr("hermes_cli.product_app.load_product_oidc_client_settings", lambda: object())
    monkeypatch.setattr("hermes_cli.product_app.discover_product_oidc_provider_metadata", lambda settings: object())
    monkeypatch.setattr(
        "hermes_cli.product_app.create_oidc_login_request",
        lambda settings, metadata: {
            "state": "state-123",
            "nonce": "nonce-123",
            "verifier": "verifier-123",
            "authorization_url": "http://officebox.local:1411/authorize?client_id=hermes-core",
        },
    )

    client = TestClient(create_product_app())
    client.get("/api/auth/login", follow_redirects=False)
    response = client.get("/api/auth/oidc/callback?code=auth-code&state=wrong-state")

    assert response.status_code == 400
    assert response.json() == {"detail": "OIDC state mismatch"}


def test_product_app_logout_clears_session(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.product_app.load_product_config",
        lambda: {"bootstrap": {"first_admin_username": "admin"}},
    )
    monkeypatch.setattr(
        "hermes_cli.product_app.resolve_product_urls",
        lambda config: {"app_base_url": "http://officebox.local:8086", "issuer_url": "http://officebox.local:1411"},
    )
    monkeypatch.setattr("hermes_cli.product_app._session_secret", lambda: "test-secret")
    monkeypatch.setattr("hermes_cli.product_app.load_product_oidc_client_settings", lambda: object())
    monkeypatch.setattr("hermes_cli.product_app.discover_product_oidc_provider_metadata", lambda settings: object())
    monkeypatch.setattr(
        "hermes_cli.product_app.create_oidc_login_request",
        lambda settings, metadata: {
            "state": "state-123",
            "nonce": "nonce-123",
            "verifier": "verifier-123",
            "authorization_url": "http://officebox.local:1411/authorize?client_id=hermes-core",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.product_app.exchange_product_oidc_code",
        lambda settings, metadata, code, verifier: {"access_token": "access-token"},
    )
    monkeypatch.setattr(
        "hermes_cli.product_app.fetch_product_oidc_userinfo",
        lambda access_token, metadata: {
            "sub": "user-1",
            "email": "admin@example.com",
            "name": "Admin User",
            "preferred_username": "admin",
            "email_verified": True,
        },
    )

    client = TestClient(create_product_app())
    client.get("/api/auth/login", follow_redirects=False)
    client.get("/api/auth/oidc/callback?code=auth-code&state=state-123", follow_redirects=False)

    response = client.post("/api/auth/logout")
    assert response.status_code == 200
    assert response.json() == {"authenticated": False, "user": None}
    assert client.get("/api/auth/session").json() == {"authenticated": False, "user": None}


def test_product_app_chat_session_requires_auth(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.product_app.load_product_config",
        lambda: {"product": {"brand": {"name": "Hermes Core"}}},
    )
    monkeypatch.setattr(
        "hermes_cli.product_app.resolve_product_urls",
        lambda config: {"app_base_url": "http://officebox.local:8086", "issuer_url": "http://officebox.local:1411"},
    )
    monkeypatch.setattr("hermes_cli.product_app._session_secret", lambda: "test-secret")

    client = TestClient(create_product_app())
    response = client.get("/api/chat/session")

    assert response.status_code == 401
    assert response.json() == {"detail": "Not authenticated"}


def test_product_app_chat_session_returns_payload(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.product_app.load_product_config",
        lambda: {"bootstrap": {"first_admin_username": "admin"}},
    )
    monkeypatch.setattr(
        "hermes_cli.product_app.resolve_product_urls",
        lambda config: {"app_base_url": "http://officebox.local:8086", "issuer_url": "http://officebox.local:1411"},
    )
    monkeypatch.setattr("hermes_cli.product_app._session_secret", lambda: "test-secret")
    monkeypatch.setattr("hermes_cli.product_app.load_product_oidc_client_settings", lambda: object())
    monkeypatch.setattr("hermes_cli.product_app.discover_product_oidc_provider_metadata", lambda settings: object())
    monkeypatch.setattr(
        "hermes_cli.product_app.create_oidc_login_request",
        lambda settings, metadata: {
            "state": "state-123",
            "nonce": "nonce-123",
            "verifier": "verifier-123",
            "authorization_url": "http://officebox.local:1411/authorize?client_id=hermes-core",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.product_app.exchange_product_oidc_code",
        lambda settings, metadata, code, verifier: {"access_token": "access-token"},
    )
    monkeypatch.setattr(
        "hermes_cli.product_app.fetch_product_oidc_userinfo",
        lambda access_token, metadata: {
            "sub": "user-1",
            "email": "admin@example.com",
            "name": "Admin User",
            "preferred_username": "admin",
            "email_verified": True,
        },
    )
    monkeypatch.setattr(
        "hermes_cli.product_app.get_product_runtime_session",
        lambda user: {
            "session_id": "product_admin_123",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
        },
    )

    client = TestClient(create_product_app())
    client.get("/api/auth/login", follow_redirects=False)
    client.get("/api/auth/oidc/callback?code=auth-code&state=state-123", follow_redirects=False)

    response = client.get("/api/chat/session")

    assert response.status_code == 200
    assert response.json() == {
        "session_id": "product_admin_123",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ],
    }


def test_product_app_chat_stream_returns_sse(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.product_app.load_product_config",
        lambda: {"bootstrap": {"first_admin_username": "admin"}},
    )
    monkeypatch.setattr(
        "hermes_cli.product_app.resolve_product_urls",
        lambda config: {"app_base_url": "http://officebox.local:8086", "issuer_url": "http://officebox.local:1411"},
    )
    monkeypatch.setattr("hermes_cli.product_app._session_secret", lambda: "test-secret")
    monkeypatch.setattr("hermes_cli.product_app.load_product_oidc_client_settings", lambda: object())
    monkeypatch.setattr("hermes_cli.product_app.discover_product_oidc_provider_metadata", lambda settings: object())
    monkeypatch.setattr(
        "hermes_cli.product_app.create_oidc_login_request",
        lambda settings, metadata: {
            "state": "state-123",
            "nonce": "nonce-123",
            "verifier": "verifier-123",
            "authorization_url": "http://officebox.local:1411/authorize?client_id=hermes-core",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.product_app.exchange_product_oidc_code",
        lambda settings, metadata, code, verifier: {"access_token": "access-token"},
    )
    monkeypatch.setattr(
        "hermes_cli.product_app.fetch_product_oidc_userinfo",
        lambda access_token, metadata: {
            "sub": "user-1",
            "email": "admin@example.com",
            "name": "Admin User",
            "preferred_username": "admin",
            "email_verified": True,
        },
    )
    monkeypatch.setattr(
        "hermes_cli.product_app.stream_product_runtime_turn",
        lambda user, message: iter(
            [
                'event: start\ndata: {"session_id": "product_admin_123"}\n\n',
                'event: final\ndata: {"session_id": "product_admin_123", "final_response": "done", "messages": []}\n\n',
            ]
        ),
    )

    client = TestClient(create_product_app())
    client.get("/api/auth/login", follow_redirects=False)
    client.get("/api/auth/oidc/callback?code=auth-code&state=state-123", follow_redirects=False)

    response = client.post("/api/chat/turn/stream", json={"user_message": "hello"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert 'event: start' in response.text
    assert '"final_response": "done"' in response.text


def test_product_app_index_shows_session_details_when_signed_in(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.product_app.load_product_config",
        lambda: {
            "product": {"brand": {"name": "Hermes Core"}},
            "bootstrap": {"first_admin_username": "admin"},
        },
    )
    monkeypatch.setattr(
        "hermes_cli.product_app.resolve_product_urls",
        lambda config: {"app_base_url": "http://officebox.local:8086", "issuer_url": "http://officebox.local:1411"},
    )
    monkeypatch.setattr("hermes_cli.product_app._session_secret", lambda: "test-secret")
    monkeypatch.setattr("hermes_cli.product_app.load_product_oidc_client_settings", lambda: object())
    monkeypatch.setattr("hermes_cli.product_app.discover_product_oidc_provider_metadata", lambda settings: object())
    monkeypatch.setattr(
        "hermes_cli.product_app.create_oidc_login_request",
        lambda settings, metadata: {
            "state": "state-123",
            "nonce": "nonce-123",
            "verifier": "verifier-123",
            "authorization_url": "http://officebox.local:1411/authorize?client_id=hermes-core",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.product_app.exchange_product_oidc_code",
        lambda settings, metadata, code, verifier: {"access_token": "access-token"},
    )
    monkeypatch.setattr(
        "hermes_cli.product_app.fetch_product_oidc_userinfo",
        lambda access_token, metadata: {
            "sub": "user-1",
            "email": "admin@example.com",
            "name": "Admin User",
            "preferred_username": "admin",
            "email_verified": True,
        },
    )
    monkeypatch.setattr(
        "hermes_cli.product_app.get_product_runtime_session",
        lambda user: {"session_id": "product_admin_123", "messages": []},
    )

    client = TestClient(create_product_app())
    client.get("/api/auth/login", follow_redirects=False)
    client.get("/api/auth/oidc/callback?code=auth-code&state=state-123", follow_redirects=False)
    response = client.get("/")

    assert response.status_code == 200
    session = client.get("/api/auth/session")
    assert session.json()["user"]["is_admin"] is True
    assert "Hermes Core" in response.text
    assert "Shared Files" in response.text
    assert 'id="sessionCard"' not in response.text
