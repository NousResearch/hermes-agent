from __future__ import annotations

from agent.ninerouter_usage import fetch_ninerouter_account_usage


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class _Client:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, headers=None):
        self.calls.append((url, dict(headers or {})))
        return self.responses[url]


def test_management_redirect_is_not_followed(monkeypatch):
    base = "http://127.0.0.1:20128"
    client = _Client({f"{base}/api/providers": _Response({}, status_code=302)})
    client_kwargs = {}

    def make_client(**kwargs):
        client_kwargs.update(kwargs)
        return client

    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", make_client)

    snapshot = fetch_ninerouter_account_usage(
        base_url=f"{base}/v1",
        cli_token="[REDACTED]",
        scope="profile:default",
    )

    assert snapshot.available is False
    assert len(client.calls) == 1
    assert client_kwargs["follow_redirects"] is False


def test_route_quota_redirect_is_not_followed(monkeypatch):
    base = "http://127.0.0.1:20128"
    client = _Client(
        {
            f"{base}/api/providers": _Response(
                {"connections": [{"id": "conn-a", "provider": "route-provider"}]}
            ),
            f"{base}/api/usage/conn-a": _Response({}, status_code=302),
        }
    )
    client_kwargs = {}

    def _client_factory(**kwargs):
        client_kwargs.update(kwargs)
        return client

    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", _client_factory)

    snapshot = fetch_ninerouter_account_usage(
        base_url=f"{base}/v1",
        cli_token="[REDACTED]",
    )

    assert snapshot.available is False
    assert client_kwargs["follow_redirects"] is False
    assert [url for url, _headers in client.calls] == [
        f"{base}/api/providers",
        f"{base}/api/usage/conn-a",
    ]


def test_management_cookie_is_forwarded_without_authorization_header(monkeypatch):
    base = "http://127.0.0.1:20128"
    connection_id = "cookie-conn"
    client = _Client(
        {
            f"{base}/api/providers": _Response(
                {
                    "connections": [
                        {"id": connection_id, "provider": "route-provider", "name": "account"}
                    ]
                }
            ),
            f"{base}/api/usage/{connection_id}": _Response(
                {
                    "quotas": {
                        "window": {
                            "used": 1,
                            "total": 2,
                            "remaining": 1,
                            "unit": "requests",
                        }
                    }
                }
            ),
        }
    )
    client_kwargs = {}

    def _client_factory(**kwargs):
        client_kwargs.update(kwargs)
        return client

    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", _client_factory)

    snapshot = fetch_ninerouter_account_usage(
        base_url=f"{base}/v1",
        auth_cookie="session=[REDACTED]",
        inference_api_key="[REDACTED]",
    )

    assert snapshot.available is True
    assert client_kwargs["trust_env"] is False
    assert len(client.calls) == 2
    for url, headers in client.calls:
        assert url in {
            f"{base}/api/providers",
            f"{base}/api/usage/{connection_id}",
        }
        assert headers["Cookie"] == "session=[REDACTED]"
        assert "Authorization" not in headers


def test_malformed_management_cookie_is_not_forwarded(monkeypatch):
    base = "http://127.0.0.1:20128"
    client = _Client({f"{base}/api/providers": _Response({"connections": []})})
    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", lambda **kwargs: client)

    snapshot = fetch_ninerouter_account_usage(
        base_url=f"{base}/v1",
        auth_cookie="bad\r\nX-Injected: yes",
    )

    assert snapshot.available is False
    assert client.calls == [(f"{base}/api/providers", {"Accept": "application/json"})]


def test_management_cookie_is_rejected_for_non_loopback_url(monkeypatch):
    client = _Client({})
    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", lambda **kwargs: client)

    snapshot = fetch_ninerouter_account_usage(
        base_url="https://router.example.invalid/v1",
        management_base_url="https://router.example.invalid",
        auth_cookie="session=[REDACTED]",
    )

    assert snapshot.available is False
    assert "management endpoint" in (snapshot.unavailable_reason or "")
    assert client.calls == []


def test_management_requests_require_loopback_even_without_credentials(monkeypatch):
    client = _Client({})
    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", lambda **kwargs: client)

    snapshot = fetch_ninerouter_account_usage(
        base_url="https://router.example.invalid/v1",
        management_base_url="https://router.example.invalid",
    )

    assert snapshot.available is False
    assert "loopback" in (snapshot.unavailable_reason or "")
    assert client.calls == []


def test_management_401_is_unavailable_and_inference_auth_is_not_forwarded(monkeypatch):
    client = _Client(
        {
            "http://127.0.0.1:20128/api/providers": _Response({}, status_code=401),
        }
    )
    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", lambda **kwargs: client)

    snapshot = fetch_ninerouter_account_usage(
        base_url="http://localhost:20128/v1",
        inference_api_key="[REDACTED]",
        scope="profile:default",
    )

    assert snapshot is not None
    assert snapshot.provider == "9router"
    assert snapshot.available is False
    assert snapshot.routes == ()
    assert snapshot.unavailable_reason == (
        "9Router management authentication is required; "
        "the OpenAI-compatible inference credential was not sent."
    )
    assert client.calls == [
        (
            "http://127.0.0.1:20128/api/providers",
            {"Accept": "application/json"},
        )
    ]


def test_authenticated_connections_are_rendered_as_independent_routes(monkeypatch):
    base = "http://127.0.0.1:20128"
    client = _Client(
        {
            f"{base}/api/providers": _Response(
                {
                    "connections": [
                        {"id": "conn-a", "provider": "route-provider", "name": "account-a", "isActive": True},
                        {"id": "conn-b", "provider": "route-provider", "name": "account-b", "isActive": True},
                    ]
                }
            ),
            f"{base}/api/usage/conn-a": _Response(
                {
                    "plan": "balanced",
                    "quotas": {
                        "weekly": {
                            "used": 12.0,
                            "total": 100.0,
                            "remaining": 88.0,
                            "unit": "tokens",
                            "resetAt": "2026-09-20T00:00:00Z",
                        }
                    },
                }
            ),
            f"{base}/api/usage/conn-b": _Response(
                {"message": "usage unavailable"}
            ),
        }
    )
    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", lambda **kwargs: client)

    snapshot = fetch_ninerouter_account_usage(
        base_url=f"{base}/v1",
        scope="profile:default",
    )

    assert snapshot is not None
    assert snapshot.available is True
    assert snapshot.partial is True
    assert snapshot.scope == "profile:default"
    assert len(snapshot.routes) == 2
    reported, unknown = snapshot.routes
    assert reported.route == "weekly"
    assert reported.account == "account-a"
    assert reported.provider == "route-provider"
    assert reported.usage == 12.0
    assert reported.limit == 100.0
    assert reported.remaining == 88.0
    assert reported.remaining_percent == 88.0
    assert reported.unit == "tokens"
    assert reported.status == "reported"
    assert unknown.route == "quota"
    assert unknown.account == "account-b"
    assert unknown.status == "unknown"
    assert unknown.usage is None
    assert unknown.limit is None
    assert unknown.remaining is None
    assert all("Authorization" not in headers for _, headers in client.calls)


def test_malformed_quota_does_not_derive_remaining_or_coerce_strings(monkeypatch):
    base = "http://127.0.0.1:20128"
    client = _Client(
        {
            f"{base}/api/providers": _Response(
                {"connections": [{"id": "conn-a", "provider": "route-provider", "name": "account-a"}]}
            ),
            f"{base}/api/usage/conn-a": _Response(
                {
                    "quotas": {
                        "bad-window": {
                            "used": "12",
                            "total": 100.0,
                        }
                    }
                }
            ),
        }
    )
    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", lambda **kwargs: client)

    snapshot = fetch_ninerouter_account_usage(base_url=f"{base}/v1")

    assert snapshot is not None
    assert len(snapshot.routes) == 1
    route = snapshot.routes[0]
    assert route.status == "unknown"
    assert route.usage is None
    assert route.limit is None
    assert route.remaining is None
    assert route.remaining_percent is None
    assert route.detail == "Quota fields were incomplete or used an unsupported type."


def test_invalid_route_range_does_not_create_percentage_gauge():
    from agent.ninerouter_usage import _remaining_percent

    assert _remaining_percent(100.0, 101.0) is None
    assert _remaining_percent(0.0, 0.0) is None
    assert _remaining_percent(100.0, -1.0) is None


def test_non_loopback_inference_endpoint_requires_explicit_management_url(monkeypatch):
    client = _Client({})
    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", lambda **kwargs: client)

    snapshot = fetch_ninerouter_account_usage(
        base_url="https://example.invalid/v1",
    )

    assert snapshot is not None
    assert snapshot.available is False
    assert snapshot.routes == ()
    assert snapshot.unavailable_reason == (
        "9Router management endpoint is not configured for this provider endpoint."
    )
    assert client.calls == []


def test_explicit_management_url_can_be_used_without_inference_auth(monkeypatch):
    management = "http://127.0.0.1:20128"
    client = _Client(
        {
            f"{management}/api/providers": _Response({"connections": []}),
        }
    )
    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", lambda **kwargs: client)

    snapshot = fetch_ninerouter_account_usage(
        base_url="https://example.invalid/v1",
        management_base_url=management,
        inference_api_key="[REDACTED]",
    )

    assert snapshot is not None
    assert snapshot.available is False
    assert snapshot.routes == ()
    assert snapshot.unavailable_reason == "9Router returned no provider connections."
    assert client.calls == [(f"{management}/api/providers", {"Accept": "application/json"})]


def test_cli_token_is_forwarded_separately_from_inference_auth(monkeypatch):
    management = "http://127.0.0.1:20128"
    client = _Client(
        {
            f"{management}/api/providers": _Response({"connections": []}),
        }
    )
    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", lambda **kwargs: client)

    snapshot = fetch_ninerouter_account_usage(
        base_url=f"{management}/v1",
        cli_token="[REDACTED]",
        inference_api_key="[REDACTED]",
    )

    assert snapshot is not None
    assert snapshot.available is False
    assert client.calls == [
        (
            f"{management}/api/providers",
            {"Accept": "application/json", "x-9r-cli-token": "[REDACTED]"},
        )
    ]


def test_cli_token_is_rejected_for_non_loopback_management_url(monkeypatch):
    management = "https://router.example.invalid"
    client = _Client({})
    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", lambda **kwargs: client)

    snapshot = fetch_ninerouter_account_usage(
        base_url="https://router.example.invalid/v1",
        management_base_url=management,
        cli_token="[REDACTED]",
    )

    assert snapshot.available is False
    assert snapshot.unavailable_reason == "9Router management requests are restricted to a loopback management endpoint."
    assert client.calls == []


def test_cli_token_request_uses_numeric_loopback_and_bypasses_environment_proxy(monkeypatch):
    management = "http://localhost:20128"
    client = _Client({"http://127.0.0.1:20128/api/providers": _Response({"connections": []})})
    client_kwargs = {}

    def _client_factory(**kwargs):
        client_kwargs.update(kwargs)
        return client

    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", _client_factory)
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda *args, **kwargs: [
            (2, 1, 6, "", ("127.0.0.1", 20128)),
        ],
    )

    snapshot = fetch_ninerouter_account_usage(
        base_url=f"{management}/v1",
        cli_token="[REDACTED]",
        inference_api_key="[REDACTED]",
    )

    assert snapshot.available is False
    assert client_kwargs["trust_env"] is False
    assert client.calls == [
        ("http://127.0.0.1:20128/api/providers", {"Accept": "application/json", "x-9r-cli-token": "[REDACTED]"})
    ]


def test_cli_token_accepts_direct_numeric_loopback_address(monkeypatch):
    management = "http://127.0.0.2:20128"
    client = _Client({f"{management}/api/providers": _Response({"connections": []})})
    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", lambda **kwargs: client)

    snapshot = fetch_ninerouter_account_usage(
        base_url=f"{management}/v1",
        cli_token="[REDACTED]",
    )

    assert snapshot.available is False
    assert client.calls == [
        (
            f"{management}/api/providers",
            {"Accept": "application/json", "x-9r-cli-token": "[REDACTED]"},
        )
    ]


def test_cli_token_request_uses_an_address_returned_by_dns(monkeypatch):
    client = _Client({"http://127.0.0.2:20128/api/providers": _Response({"connections": []})})
    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", lambda **kwargs: client)
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda *args, **kwargs: [
            (2, 1, 6, "", ("127.0.0.2", 20128)),
        ],
    )

    snapshot = fetch_ninerouter_account_usage(
        base_url="http://localhost:20128/v1",
        cli_token="[REDACTED]",
    )

    assert snapshot.available is False
    assert client.calls == [
        ("http://127.0.0.2:20128/api/providers", {"Accept": "application/json", "x-9r-cli-token": "[REDACTED]"})
    ]


def test_cli_token_is_rejected_when_localhost_resolves_off_loopback(monkeypatch):
    client = _Client({})
    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", lambda **kwargs: client)
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda *args, **kwargs: [
            (2, 1, 6, "", ("192.0.2.10", 20128)),
        ],
    )

    snapshot = fetch_ninerouter_account_usage(
        base_url="http://localhost:20128/v1",
        cli_token="[REDACTED]",
    )

    assert snapshot.available is False
    assert snapshot.unavailable_reason == "9Router management requests are restricted to a loopback management endpoint."
    assert client.calls == []


def test_over_limit_provider_quota_is_not_reported_as_valid(monkeypatch):
    management = "http://127.0.0.1:20128"
    client = _Client(
        {
            f"{management}/api/providers": _Response(
                {"connections": [{"id": "conn-a", "provider": "route-provider", "name": "account-a"}]}
            ),
            f"{management}/api/usage/conn-a": _Response(
                {"quotas": {"weekly": {"used": 110, "total": 100, "remaining": 0, "remainingPercentage": 0}}}
            ),
        }
    )
    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", lambda **kwargs: client)

    snapshot = fetch_ninerouter_account_usage(base_url=f"{management}/v1")

    route = snapshot.routes[0]
    assert route.status == "unknown"
    assert route.remaining_percent is None
    assert "valid range" in (route.detail or "").lower()


def test_usage_plus_remaining_must_match_limit(monkeypatch):
    management = "http://127.0.0.1:20128"
    client = _Client(
        {
            f"{management}/api/providers": _Response(
                {"connections": [{"id": "conn-a", "provider": "route-provider", "name": "account-a"}]}
            ),
            f"{management}/api/usage/conn-a": _Response(
                {"quotas": {"weekly": {"used": 10, "total": 100, "remaining": 20, "remainingPercentage": 20}}}
            ),
        }
    )
    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", lambda **kwargs: client)

    snapshot = fetch_ninerouter_account_usage(base_url=f"{management}/v1")

    route = snapshot.routes[0]
    assert route.status == "unknown"
    assert route.remaining_percent is None
    assert "inconsistent" in (route.detail or "").lower()


def test_reported_percentage_must_match_usage_when_remaining_is_missing(monkeypatch):
    management = "http://127.0.0.1:20128"
    client = _Client(
        {
            f"{management}/api/providers": _Response(
                {"connections": [{"id": "conn-a", "provider": "route-provider", "name": "account-a"}]}
            ),
            f"{management}/api/usage/conn-a": _Response(
                {"quotas": {"weekly": {"used": 10, "total": 100, "remainingPercentage": 20}}}
            ),
        }
    )
    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", lambda **kwargs: client)

    snapshot = fetch_ninerouter_account_usage(base_url=f"{management}/v1")

    route = snapshot.routes[0]
    assert route.status == "unknown"
    assert route.remaining_percent is None
    assert "inconsistent" in (route.detail or "").lower()


def test_inconsistent_provider_quota_is_not_reported_as_valid(monkeypatch):
    management = "http://127.0.0.1:20128"
    client = _Client(
        {
            f"{management}/api/providers": _Response(
                {"connections": [{"id": "conn-a", "provider": "route-provider", "name": "account-a"}]}
            ),
            f"{management}/api/usage/conn-a": _Response(
                {
                    "quotas": {
                        "weekly": {
                            "used": 10,
                            "total": 100,
                            "remaining": 20,
                            "remainingPercentage": 50,
                        }
                    }
                }
            ),
        }
    )
    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", lambda **kwargs: client)

    snapshot = fetch_ninerouter_account_usage(base_url=f"{management}/v1")

    route = snapshot.routes[0]
    assert route.status == "unknown"
    assert "inconsistent" in (route.detail or "").lower()


def test_timezone_less_reset_is_not_assumed_to_be_utc(monkeypatch):
    management = "http://127.0.0.1:20128"
    client = _Client(
        {
            f"{management}/api/providers": _Response(
                {"connections": [{"id": "conn-a", "provider": "route-provider", "name": "account-a"}]}
            ),
            f"{management}/api/usage/conn-a": _Response(
                {
                    "quotas": {
                        "weekly": {
                            "used": 10,
                            "total": 100,
                            "remaining": 90,
                            "resetAt": "2026-09-20T00:00:00",
                        }
                    }
                }
            ),
        }
    )
    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", lambda **kwargs: client)

    snapshot = fetch_ninerouter_account_usage(base_url=f"{management}/v1")

    route = snapshot.routes[0]
    assert route.status == "unknown"
    assert route.reset_at is None
    assert "reset" in (route.detail or "").lower()


def test_credential_like_labels_are_not_serialized(monkeypatch):
    import json
    from dataclasses import asdict

    management = "http://127.0.0.1:20128"
    connection_id = "connection-labels"
    client = _Client(
        {
            f"{management}/api/providers": _Response(
                {
                    "connections": [
                        {
                            "id": connection_id,
                            "provider": "provider=hidden",
                            "name": "token=[REDACTED]",
                            "email": "authorization=[REDACTED]",
                        }
                    ]
                }
            ),
            f"{management}/api/usage/{connection_id}": _Response(
                {
                    "quotas": {
                        "cookie=[REDACTED]": {
                            "used": 1,
                            "total": 2,
                            "remaining": 1,
                            "unit": "https://user:[REDACTED]@unit.invalid",
                        },
                        "https://user:[REDACTED]@route.invalid": {
                            "used": 1,
                            "total": 2,
                            "remaining": 1,
                        },
                    }
                }
            ),
        }
    )
    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", lambda **kwargs: client)

    snapshot = fetch_ninerouter_account_usage(
        base_url=management,
        cli_token="[REDACTED]",
    )
    serialized = json.dumps([asdict(route) for route in snapshot.routes], default=str)

    assert "[REDACTED]" not in serialized
    assert "provider=hidden" not in serialized
    assert "token=" not in serialized
    assert "authorization=" not in serialized
    assert "cookie=" not in serialized
    assert "credential=" not in serialized


def test_display_label_sanitizer_preserves_safe_labels_and_rejects_opaque_material():
    from agent.ninerouter_usage import _clean_display_label, _looks_like_credential_material

    safe_labels = [
        "route-provider",
        "user@example.com",
        "gemini-3.8-flash-high",
        "claude_gpt_weekly",
        "requests",
        "tokens",
        "profile:default",
        "normal short names",
    ]
    for label in safe_labels:
        assert _looks_like_credential_material(label) is False
        assert _clean_display_label(label) == label

    opaque_labels = [
        "sk-synthetic-opaque-body-1234567890",
        "eyJsyntheticHeaderSegment.eyJsyntheticPayload.eyJsyntheticSignature",
        "0123456789abcdef0123456789abcdef0123456789abcdef",
        "A7fK2mP9xQ4vR8sL1nC6dH3jT5wY0uZ8qW6eR4tY2uI0oP",
        "QWxhZGRpblN5bnRoZXRpY0Jsb2I+/Q==",
    ]
    for label in opaque_labels:
        assert _looks_like_credential_material(label) is True
        assert _clean_display_label(label) is None


def test_opaque_display_labels_are_not_serialized(monkeypatch):
    import json
    from dataclasses import asdict

    management = "http://127.0.0.1:20128"
    connection_id = "synthetic-label-connection"
    opaque_provider = "providerOpaqueA7fK2mP9xQ4vR8sL1nC6dH3jT5wY0uZ"
    opaque_account = "0123456789abcdef0123456789abcdef0123456789abcdef"
    opaque_route = "QWxhZGRpblN5bnRoZXRpY0Jsb2I+/Q=="
    opaque_unit = "eyJsyntheticUnitHeader.eyJsyntheticUnitPayload"
    opaque_scope = "A7fK2mP9xQ4vR8sL1nC6dH3jT5wY0uZ8qW6eR4tY2uI0oP"
    client = _Client(
        {
            f"{management}/api/providers": _Response(
                {
                    "connections": [
                        {
                            "id": connection_id,
                            "provider": opaque_provider,
                            "name": opaque_account,
                        }
                    ]
                }
            ),
            f"{management}/api/usage/{connection_id}": _Response(
                {
                    "quotas": {
                        opaque_route: {
                            "used": 1,
                            "total": 2,
                            "remaining": 1,
                            "unit": opaque_unit,
                        }
                    }
                }
            ),
        }
    )
    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", lambda **kwargs: client)

    snapshot = fetch_ninerouter_account_usage(
        base_url=management,
        cli_token="[REDACTED]",
        scope=opaque_scope,
    )
    serialized = json.dumps(asdict(snapshot), default=str)

    for raw_value in (opaque_provider, opaque_account, opaque_route, opaque_unit, opaque_scope):
        assert raw_value not in serialized
    assert snapshot.scope is None
    assert snapshot.routes[0].provider == "Unknown provider"
    assert snapshot.routes[0].account == "Unknown provider #1"
    assert snapshot.routes[0].route == "quota"
    assert snapshot.routes[0].unit is None


def test_zero_limit_with_provider_percentage_is_unknown(monkeypatch):
    management = "http://127.0.0.1:20128"
    client = _Client(
        {
            f"{management}/api/providers": _Response(
                {"connections": [{"id": "conn-a", "provider": "route-provider", "name": "account-a"}]}
            ),
            f"{management}/api/usage/conn-a": _Response(
                {"quotas": {"zero": {"used": 0, "total": 0, "remainingPercentage": 100}}}
            ),
        }
    )
    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", lambda **kwargs: client)

    snapshot = fetch_ninerouter_account_usage(base_url=f"{management}/v1")

    route = snapshot.routes[0]
    assert route.status == "unknown"
    assert route.remaining_percent is None
    assert "zero" in (route.detail or "").lower()


def test_present_malformed_remaining_is_not_treated_as_missing(monkeypatch):
    management = "http://127.0.0.1:20128"
    client = _Client(
        {
            f"{management}/api/providers": _Response(
                {"connections": [{"id": "conn-a", "provider": "route-provider", "name": "account-a"}]}
            ),
            f"{management}/api/usage/conn-a": _Response(
                {
                    "quotas": {
                        "weekly": {
                            "used": 10,
                            "total": 100,
                            "remaining": "not-a-number",
                            "remainingPercentage": 90,
                        }
                    }
                }
            ),
        }
    )
    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", lambda **kwargs: client)

    snapshot = fetch_ninerouter_account_usage(base_url=f"{management}/v1")

    route = snapshot.routes[0]
    assert route.status == "unknown"
    assert route.remaining_percent is None
    assert "remaining" in (route.detail or "").lower()


def test_named_profile_does_not_fall_back_to_process_global_data_dir(tmp_path, monkeypatch):
    from agent.ninerouter_usage import resolve_ninerouter_cli_token

    global_dir = tmp_path / "global"
    (global_dir / "auth").mkdir(parents=True)
    (global_dir / "machine-id").write_text("machine", encoding="utf-8")
    (global_dir / "auth" / "cli-secret").write_text("material", encoding="utf-8")
    monkeypatch.setenv("HERMES_9ROUTER_DATA_DIR", str(global_dir))

    assert resolve_ninerouter_cli_token(
        "http://127.0.0.1:20128/v1",
        use_process_env=False,
        allow_default_data_dir=False,
    ) is None


def test_profile_bound_data_dir_can_derive_local_cli_token(tmp_path):
    import hashlib
    from agent.ninerouter_usage import resolve_ninerouter_cli_token

    profile_dir = tmp_path / "profile-router"
    (profile_dir / "auth").mkdir(parents=True)
    machine_id = "profile-machine"
    local_material = "profile-material"
    (profile_dir / "machine-id").write_text(machine_id, encoding="utf-8")
    (profile_dir / "auth" / "cli-secret").write_text(local_material, encoding="utf-8")

    expected = hashlib.sha256((machine_id + "9r-cli-auth" + local_material).encode()).hexdigest()[:16]
    assert resolve_ninerouter_cli_token(
        "http://127.0.0.1:20128/v1",
        data_dir=str(profile_dir),
        use_process_env=False,
        allow_default_data_dir=False,
    ) == expected


def test_present_invalid_unit_marks_route_unknown(monkeypatch):
    base = "http://127.0.0.1:20128"
    client = _Client(
        {
            f"{base}/api/providers": _Response(
                {"connections": [{"id": "invalid-unit", "provider": "route-provider"}]}
            ),
            f"{base}/api/usage/invalid-unit": _Response(
                {
                    "quotas": {
                        "daily": {
                            "used": 1,
                            "total": 2,
                            "remaining": 1,
                            "unit": "https://user:[REDACTED]@unit.invalid",
                        }
                    }
                }
            ),
        }
    )
    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", lambda **kwargs: client)

    snapshot = fetch_ninerouter_account_usage(base_url=f"{base}/v1")

    assert len(snapshot.routes) == 1
    route = snapshot.routes[0]
    assert route.status == "unknown"
    assert route.unit is None
    assert "invalid quota unit" in (route.detail or "")


def test_provider_remaining_percentage_is_reported_without_inventing_missing_fields(monkeypatch):
    management = "http://127.0.0.1:20128"
    client = _Client(
        {
            f"{management}/api/providers": _Response(
                {"connections": [{"id": "conn-a", "provider": "gemini-cli", "name": "account-a"}]}
            ),
            f"{management}/api/usage/conn-a": _Response(
                {
                    "quotas": {
                        "gemini-3.8-flash-high": {
                            "used": 18,
                            "total": 1000,
                            "resetAt": "2026-09-20T00:00:00Z",
                            "remainingPercentage": 98.21777,
                            "unlimited": False,
                        }
                    }
                }
            ),
        }
    )
    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", lambda **kwargs: client)

    snapshot = fetch_ninerouter_account_usage(base_url=f"{management}/v1")

    assert snapshot is not None
    assert snapshot.available is True
    route = snapshot.routes[0]
    assert route.route == "gemini-3.8-flash-high"
    assert route.status == "reported"
    assert route.usage == 18.0
    assert route.limit == 1000.0
    assert route.remaining is None
    assert route.remaining_percent == 98.21777
    assert route.unit is None
    assert route.reset_at is not None


def test_local_cli_token_derivation_is_loopback_scoped(tmp_path, monkeypatch):
    import hashlib
    from agent.ninerouter_usage import resolve_ninerouter_cli_token

    machine_id = "machine-id-fixture"
    local_material = "material-fixture"
    (tmp_path / "machine-id").write_text(machine_id, encoding="utf-8")
    (tmp_path / "auth").mkdir()
    (tmp_path / "auth" / "cli-secret").write_text(local_material, encoding="utf-8")
    monkeypatch.setenv("HERMES_9ROUTER_DATA_DIR", str(tmp_path))

    expected = hashlib.sha256((machine_id + "9r-cli-auth" + local_material).encode()).hexdigest()[:16]
    assert resolve_ninerouter_cli_token("http://127.0.0.1:20128/v1") == expected
    assert resolve_ninerouter_cli_token("https://example.invalid/v1") is None


def test_duplicate_connection_labels_remain_separate_with_deterministic_suffix(monkeypatch):
    import json
    from dataclasses import asdict

    base = "http://127.0.0.1:20128"
    conn_id_1 = "conn-unique-id-alpha"
    conn_id_2 = "conn-unique-id-beta"
    shared_label = "primary-work-account"

    client = _Client(
        {
            f"{base}/api/providers": _Response(
                {
                    "connections": [
                        {
                            "id": conn_id_1,
                            "provider": "route-provider",
                            "name": shared_label,
                            "isActive": True,
                        },
                        {
                            "id": conn_id_2,
                            "provider": "route-provider",
                            "name": shared_label,
                            "isActive": True,
                        },
                    ]
                }
            ),
            f"{base}/api/usage/{conn_id_1}": _Response(
                {
                    "quotas": {
                        "quota-first": {
                            "used": 10.0,
                            "total": 100.0,
                            "remaining": 90.0,
                            "unit": "tokens",
                        }
                    }
                }
            ),
            f"{base}/api/usage/{conn_id_2}": _Response(
                {
                    "quotas": {
                        "quota-second": {
                            "used": 25.0,
                            "total": 50.0,
                            "remaining": 25.0,
                            "unit": "tokens",
                        }
                    }
                }
            ),
        }
    )
    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", lambda **kwargs: client)

    snapshot = fetch_ninerouter_account_usage(
        base_url=f"{base}/v1",
        cli_token="[REDACTED]",
        scope="profile:default",
    )

    assert snapshot is not None
    assert snapshot.available is True
    assert len(snapshot.routes) == 2

    route1, route2 = snapshot.routes

    # The first friendly label and deterministic suffix #2
    assert route1.account == shared_label
    assert route2.account == f"{shared_label} #2"
    assert route1.account != route2.account

    # Routes from each connection stay under the right account
    assert route1.route == "quota-first"
    assert route1.account == shared_label
    assert route1.provider == "route-provider"
    assert route1.usage == 10.0
    assert route1.limit == 100.0
    assert route1.remaining == 90.0
    assert route1.remaining_percent == 90.0
    assert route1.unit == "tokens"
    assert route1.status == "reported"

    assert route2.route == "quota-second"
    assert route2.account == f"{shared_label} #2"
    assert route2.provider == "route-provider"
    assert route2.usage == 25.0
    assert route2.limit == 50.0
    assert route2.remaining == 25.0
    assert route2.remaining_percent == 50.0
    assert route2.unit == "tokens"
    assert route2.status == "reported"

    # Neither raw connection ID appears in account labels or serialized/display values
    assert conn_id_1 not in route1.account
    assert conn_id_1 not in route2.account
    assert conn_id_2 not in route1.account
    assert conn_id_2 not in route2.account

    serialized_routes = json.dumps([asdict(r) for r in snapshot.routes], default=str)
    assert conn_id_1 not in serialized_routes
    assert conn_id_2 not in serialized_routes
    assert conn_id_1 not in str(snapshot)
    assert conn_id_2 not in str(snapshot)

    called_urls = [url for url, _ in client.calls]
    assert called_urls == [
        f"{base}/api/providers",
        f"{base}/api/usage/{conn_id_1}",
        f"{base}/api/usage/{conn_id_2}",
    ]


def test_fallback_account_labels_remain_separate_when_metadata_missing(monkeypatch):
    import json
    from dataclasses import asdict

    base = "http://127.0.0.1:20128"
    conn_id_1 = "conn-raw-fallback-alpha"
    conn_id_2 = "conn-raw-fallback-beta"

    client = _Client(
        {
            f"{base}/api/providers": _Response(
                {
                    "connections": [
                        {
                            "id": conn_id_1,
                            "provider": "route-provider",
                            "name": "",
                            "email": "   ",
                            "account": None,
                            "label": "bearer credential-material",
                            "isActive": True,
                        },
                        {
                            "id": conn_id_2,
                            "provider": "route-provider",
                            "name": "token=secret-token",
                            "email": None,
                            "account": "   ",
                            "label": None,
                            "isActive": True,
                        },
                    ]
                }
            ),
            f"{base}/api/usage/{conn_id_1}": _Response(
                {
                    "quotas": {
                        "quota-first": {
                            "used": 5.0,
                            "total": 50.0,
                            "remaining": 45.0,
                            "unit": "requests",
                        }
                    }
                }
            ),
            f"{base}/api/usage/{conn_id_2}": _Response(
                {
                    "quotas": {
                        "quota-second": {
                            "used": 15.0,
                            "total": 50.0,
                            "remaining": 35.0,
                            "unit": "requests",
                        }
                    }
                }
            ),
        }
    )
    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", lambda **kwargs: client)

    snapshot = fetch_ninerouter_account_usage(
        base_url=f"{base}/v1",
        cli_token="[REDACTED]",
        scope="profile:default",
    )

    assert snapshot is not None
    assert snapshot.available is True
    assert len(snapshot.routes) == 2

    route1, route2 = snapshot.routes

    # Safe fallback labels: distinct within the provider
    assert route1.account == "route-provider #1"
    assert route2.account == "route-provider #2"
    assert route1.account != route2.account

    # Routes stay under their respective fallback accounts
    assert route1.route == "quota-first"
    assert route1.account == "route-provider #1"
    assert route1.provider == "route-provider"
    assert route1.usage == 5.0
    assert route1.limit == 50.0
    assert route1.remaining == 45.0
    assert route1.status == "reported"

    assert route2.route == "quota-second"
    assert route2.account == "route-provider #2"
    assert route2.provider == "route-provider"
    assert route2.usage == 15.0
    assert route2.limit == 50.0
    assert route2.remaining == 35.0
    assert route2.status == "reported"

    # Raw connection IDs and credential-like inputs are not exposed
    assert conn_id_1 not in route1.account
    assert conn_id_1 not in route2.account
    assert conn_id_2 not in route1.account
    assert conn_id_2 not in route2.account

    serialized_routes = json.dumps([asdict(r) for r in snapshot.routes], default=str)
    assert conn_id_1 not in serialized_routes
    assert conn_id_2 not in serialized_routes
    assert "credential-material" not in serialized_routes
    assert "secret-token" not in serialized_routes
    assert conn_id_1 not in str(snapshot)
    assert conn_id_2 not in str(snapshot)

    called_urls = [url for url, _ in client.calls]
    assert called_urls == [
        f"{base}/api/providers",
        f"{base}/api/usage/{conn_id_1}",
        f"{base}/api/usage/{conn_id_2}",
    ]


def test_reset_at_alias_normalization_and_conflict(monkeypatch):
    base = "http://127.0.0.1:20128"
    client = _Client(
        {
            f"{base}/api/providers": _Response(
                {
                    "connections": [
                        {"id": "conn-1", "provider": "route-provider", "name": "account-1"}
                    ]
                }
            ),
            f"{base}/api/usage/conn-1": _Response(
                {
                    "quotas": {
                        "matching_aliases": {
                            "used": 10,
                            "total": 100,
                            "remaining": 90,
                            "unit": "requests",
                            "resetAt": "2026-09-13T10:52:41Z",
                            "reset_at": "2026-09-13T10:52:41Z",
                        },
                        "conflicting_aliases": {
                            "used": 10,
                            "total": 100,
                            "remaining": 90,
                            "unit": "requests",
                            "resetAt": "2026-09-13T10:00:00Z",
                            "reset_at": "2026-09-20T00:00:00Z",
                        },
                        "invalid_alias": {
                            "used": 10,
                            "total": 100,
                            "remaining": 90,
                            "unit": "requests",
                            "resetAt": "not-a-valid-date",
                            "reset_at": "2026-09-13T10:52:41Z",
                        },
                        "one_empty_alias": {
                            "used": 10,
                            "total": 100,
                            "remaining": 90,
                            "unit": "requests",
                            "resetAt": None,
                            "reset_at": "2026-09-13T10:52:41Z",
                        },
                    }
                }
            ),
        }
    )
    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", lambda **kwargs: client)

    snapshot = fetch_ninerouter_account_usage(
        base_url=f"{base}/v1",
        cli_token="[REDACTED]",
    )

    routes_by_name = {r.route: r for r in snapshot.routes}

    # Equivalent aliases: accepted as reported
    matching = routes_by_name["matching_aliases"]
    assert matching.status == "reported"
    assert matching.reset_at is not None
    assert matching.reset_at.year == 2026

    # Conflicting aliases: fails closed to unknown with no reset_at
    conflicting = routes_by_name["conflicting_aliases"]
    assert conflicting.status == "unknown"
    assert conflicting.reset_at is None
    assert "conflict" in (conflicting.detail or "").lower()

    # Invalid alias present: fails closed to unknown
    invalid = routes_by_name["invalid_alias"]
    assert invalid.status == "unknown"
    assert invalid.reset_at is None

    # One empty alias and one non-empty: contradictory, fails closed to unknown
    one_empty = routes_by_name["one_empty_alias"]
    assert one_empty.status == "unknown"
    assert one_empty.reset_at is None


def test_remaining_percentage_alias_normalization_and_conflict(monkeypatch):
    base = "http://127.0.0.1:20128"
    client = _Client(
        {
            f"{base}/api/providers": _Response(
                {
                    "connections": [
                        {"id": "conn-1", "provider": "route-provider", "name": "account-1"}
                    ]
                }
            ),
            f"{base}/api/usage/conn-1": _Response(
                {
                    "quotas": {
                        "matching_percent": {
                            "used": 12,
                            "total": 100,
                            "remaining": 88,
                            "remainingPercentage": 88.0,
                            "remaining_percent": 88,
                        },
                        "conflicting_percent": {
                            "used": 12,
                            "total": 100,
                            "remaining": 88,
                            "remainingPercentage": 88.0,
                            "remaining_percent": 50.0,
                        },
                        "malformed_percent": {
                            "used": 12,
                            "total": 100,
                            "remaining": 88,
                            "remainingPercentage": 88.0,
                            "remaining_percent": "not-a-number",
                        },
                        "out_of_range_percent": {
                            "used": 12,
                            "total": 100,
                            "remaining": 88,
                            "remainingPercentage": 88.0,
                            "remaining_percent": 150.0,
                        },
                    }
                }
            ),
        }
    )
    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", lambda **kwargs: client)

    snapshot = fetch_ninerouter_account_usage(
        base_url=f"{base}/v1",
        cli_token="[REDACTED]",
    )

    routes_by_name = {r.route: r for r in snapshot.routes}

    # Matching percent: accepted as reported
    matching = routes_by_name["matching_percent"]
    assert matching.status == "reported"
    assert matching.remaining_percent == 88.0

    # Conflicting percent: fails closed to unknown
    conflicting = routes_by_name["conflicting_percent"]
    assert conflicting.status == "unknown"
    assert conflicting.remaining_percent is None
    assert "conflict" in (conflicting.detail or "").lower()

    # Malformed alias: fails closed to unknown
    malformed = routes_by_name["malformed_percent"]
    assert malformed.status == "unknown"
    assert malformed.remaining_percent is None

    # Out of range alias: fails closed to unknown
    out_of_range = routes_by_name["out_of_range_percent"]
    assert out_of_range.status == "unknown"
    assert out_of_range.remaining_percent is None


def test_display_label_sanitization_rejects_nul_and_control_characters(monkeypatch):
    from agent.ninerouter_usage import _clean_display_label

    # Safe labels with whitespace preserved after normalization
    assert _clean_display_label("safe label") == "safe label"
    assert _clean_display_label("  safe   label  ") == "safe label"

    # Reject control characters (C0 controls including \t, \n, \r, \v, \f, DEL, and C1)
    assert _clean_display_label("safe\tlabel") is None
    assert _clean_display_label("safe\nlabel") is None
    assert _clean_display_label("safe\rlabel") is None
    assert _clean_display_label("safe\vlabel") is None
    assert _clean_display_label("safe\flabel") is None
    assert _clean_display_label("safe\tlabel\nwith   spaces") is None

    # Reject NUL
    assert _clean_display_label("bad\x00label") is None
    assert _clean_display_label("\x00") is None

    # Reject control characters (C0, DEL, C1)
    assert _clean_display_label("bad\x1blabel") is None
    assert _clean_display_label("bad\x07label") is None
    assert _clean_display_label("bad\x7flabel") is None
    assert _clean_display_label("bad\x80label") is None
    assert _clean_display_label("bad\x9flabel") is None

    # Account label with control characters falls back safely without exposing raw ID
    base = "http://127.0.0.1:20128"
    raw_id = "sensitive-conn-id-999"
    client = _Client(
        {
            f"{base}/api/providers": _Response(
                {
                    "connections": [
                        {"id": raw_id, "provider": "route-provider", "name": "bad\x00account"}
                    ]
                }
            ),
            f"{base}/api/usage/{raw_id}": _Response(
                {
                    "quotas": {
                        "daily": {"used": 1, "total": 10, "remaining": 9}
                    }
                }
            ),
        }
    )
    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", lambda **kwargs: client)

    snapshot = fetch_ninerouter_account_usage(
        base_url=f"{base}/v1",
        cli_token="[REDACTED]",
    )
    route = snapshot.routes[0]
    assert raw_id not in (route.account or "")
    assert "\x00" not in (route.account or "")
    assert route.account == "route-provider #1"


def test_sanitize_scope_sanitizes_fallback_and_rejects_unsafe_fallback():
    from agent.ninerouter_usage import sanitize_scope

    # Safe value returns sanitized value
    assert sanitize_scope("profile:custom", fallback="profile:fallback") == "profile:custom"

    # Unsafe value with safe fallback returns sanitized fallback
    assert sanitize_scope("bearer secret-token", fallback="profile:fallback") == "profile:fallback"
    assert sanitize_scope(None, fallback="profile:fallback") == "profile:fallback"
    assert sanitize_scope("bad\x00scope", fallback="profile:fallback") == "profile:fallback"
    assert sanitize_scope("bad\nscope", fallback="profile:fallback") == "profile:fallback"

    # Unsafe value with unsafe fallback returns fixed safe fallback "profile:current"
    assert sanitize_scope(None, fallback="bearer secret-token") == "profile:current"
    assert sanitize_scope("bad\x00scope", fallback="bad\x00fallback") == "profile:current"
    assert sanitize_scope("bad\x1bscope", fallback="bad\nfallback") == "profile:current"
    assert sanitize_scope("bad\x7fscope", fallback="bad\tfallback") == "profile:current"
    assert sanitize_scope("", fallback="") == "profile:current"


def test_named_profile_malformed_data_dir_rejects_and_does_not_use_global_dir(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    from agent.ninerouter_usage import resolve_ninerouter_cli_token

    # Populate global directory with valid files that would derive a token
    global_dir = tmp_path / "global-9router"
    (global_dir / "auth").mkdir(parents=True)
    (global_dir / "machine-id").write_text("synthetic-global-machine", encoding="utf-8")
    (global_dir / "auth" / "cli-secret").write_text("synthetic-global-secret", encoding="utf-8")

    # Set platform global defaults pointing to this global directory
    monkeypatch.setenv("HERMES_9ROUTER_DATA_DIR", str(global_dir))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setattr("agent.ninerouter_usage.Path.home", lambda: tmp_path)

    appdata_dir = tmp_path / "9router"
    (appdata_dir / "auth").mkdir(parents=True, exist_ok=True)
    (appdata_dir / "machine-id").write_text("synthetic-global-machine", encoding="utf-8")
    (appdata_dir / "auth" / "cli-secret").write_text("synthetic-global-secret", encoding="utf-8")

    dot_dir = tmp_path / ".9router"
    (dot_dir / "auth").mkdir(parents=True, exist_ok=True)
    (dot_dir / "machine-id").write_text("synthetic-global-machine", encoding="utf-8")
    (dot_dir / "auth" / "cli-secret").write_text("synthetic-global-secret", encoding="utf-8")

    # Verify that when allow_default_data_dir=True, a token can be derived
    default_token = resolve_ninerouter_cli_token(
        "http://127.0.0.1:20128/v1",
        use_process_env=True,
        allow_default_data_dir=True,
    )
    assert default_token is not None

    # Track reads to prove global files are not used
    accessed_paths: list[str] = []
    orig_read_text = Path.read_text

    def tracking_read_text(self, *args, **kwargs):
        accessed_paths.append(str(self))
        return orig_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", tracking_read_text)

    malformed_values = [
        "malformed\x00path",
        "x" * 4097,
        "   ",
        "",
        12345,
        None,
        "\x00",
    ]

    for val in malformed_values:
        accessed_before = len(accessed_paths)
        token = resolve_ninerouter_cli_token(
            "http://127.0.0.1:20128/v1",
            data_dir=val,
            use_process_env=False,
            allow_default_data_dir=False,
        )
        assert token is None
        assert len(accessed_paths) == accessed_before


def test_number_and_reset_at_overflow_handling():
    from agent.ninerouter_usage import _number, _reset_at

    huge_int = 10 ** 1000
    assert _number(huge_int) is None
    assert _reset_at(huge_int) is None


def test_quota_parsing_huge_integers_do_not_crash_or_abort_snapshot(monkeypatch):
    base = "http://127.0.0.1:20128"
    huge_int = 10 ** 1000

    client = _Client(
        {
            f"{base}/api/providers": _Response(
                {
                    "connections": [
                        {"id": "conn-valid", "provider": "provider-a", "name": "valid-account"},
                        {"id": "conn-huge", "provider": "provider-b", "name": "huge-account"},
                    ]
                }
            ),
            f"{base}/api/usage/conn-valid": _Response(
                {
                    "quotas": {
                        "normal-route": {
                            "used": 10,
                            "total": 100,
                            "remaining": 90,
                            "remainingPercentage": 90.0,
                            "resetAt": "2026-09-20T00:00:00Z",
                        }
                    }
                }
            ),
            f"{base}/api/usage/conn-huge": _Response(
                {
                    "quotas": {
                        "huge-used": {
                            "used": huge_int,
                            "total": 100,
                            "remaining": 90,
                        },
                        "huge-total": {
                            "used": 10,
                            "total": huge_int,
                            "remaining": 90,
                        },
                        "huge-remaining": {
                            "used": 10,
                            "total": 100,
                            "remaining": huge_int,
                        },
                        "huge-percentage": {
                            "used": 10,
                            "total": 100,
                            "remaining": 90,
                            "remainingPercentage": huge_int,
                        },
                        "huge-reset": {
                            "used": 10,
                            "total": 100,
                            "remaining": 90,
                            "resetAt": huge_int,
                        },
                    }
                }
            ),
        }
    )
    monkeypatch.setattr("agent.ninerouter_usage.httpx.Client", lambda **kwargs: client)

    snapshot = fetch_ninerouter_account_usage(
        base_url=f"{base}/v1",
        cli_token="[REDACTED]",
    )

    assert snapshot is not None
    assert snapshot.available is True
    assert snapshot.unavailable_reason is None
    assert snapshot.partial is True

    route_by_name = {r.route: r for r in snapshot.routes}
    assert "normal-route" in route_by_name
    assert route_by_name["normal-route"].status == "reported"
    assert route_by_name["normal-route"].usage == 10.0
    assert route_by_name["normal-route"].limit == 100.0

    for malformed_name in (
        "huge-used",
        "huge-total",
        "huge-remaining",
        "huge-percentage",
        "huge-reset",
    ):
        assert malformed_name in route_by_name
        assert route_by_name[malformed_name].status == "unknown"
