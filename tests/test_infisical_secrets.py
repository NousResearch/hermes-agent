"""Hermetic tests for the Infisical secret-source integration."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.secret_sources import infisical as inf  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch):
    for key in (
        "NEW_KEY",
        "OPENAI_API_KEY",
        "INFISICAL_CLIENT_ID",
        "INFISICAL_CLIENT_SECRET",
        "INFISICAL_PROJECT_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    inf._reset_cache_for_tests()
    yield
    inf._reset_cache_for_tests()


def test_login_universal_auth_posts_client_credentials(monkeypatch):
    calls = []

    def fake_http(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return {"accessToken": "access-token", "expiresIn": 3600}

    monkeypatch.setattr(inf, "_http_json", fake_http)

    token, expires_in = inf.login_universal_auth(
        client_id="cid",
        client_secret="csecret",
        api_url="https://infisical.example.com/",
        organization_slug="acme",
    )

    assert token == "access-token"
    assert expires_in == 3600
    assert calls == [
        (
            "POST",
            "https://infisical.example.com/api/v1/auth/universal-auth/login",
            {
                "body": {
                    "clientId": "cid",
                    "clientSecret": "csecret",
                    "organizationSlug": "acme",
                },
            },
        )
    ]


def test_fetch_secrets_uses_v4_list_endpoint(monkeypatch):
    calls = []

    def fake_http(method, url, **kwargs):
        calls.append((method, url, kwargs))
        if method == "POST":
            return {"accessToken": "access-token"}
        return {
            "secrets": [
                {"secretKey": "OPENAI_API_KEY", "secretValue": "sk-test"},
                {"secretKey": "ANTHROPIC_API_KEY", "secretValue": "sk-ant"},
            ]
        }

    monkeypatch.setattr(inf, "_http_json", fake_http)

    secrets, warnings = inf.fetch_infisical_secrets(
        client_id="cid",
        client_secret="csecret",
        project_id="proj",
        environment="dev",
        secret_path="hermes",
        api_url="https://infisical.example.com",
        recursive=True,
        include_imports=False,
        expand_secret_references=False,
        use_cache=False,
    )

    assert secrets == {
        "OPENAI_API_KEY": "sk-test",
        "ANTHROPIC_API_KEY": "sk-ant",
    }
    assert warnings == []
    method, url, kwargs = calls[1]
    assert method == "GET"
    assert url == "https://infisical.example.com/api/v4/secrets"
    assert kwargs["token"] == "access-token"
    assert kwargs["params"] == {
        "projectId": "proj",
        "environment": "dev",
        "secretPath": "/hermes",
        "viewSecretValue": "true",
        "expandSecretReferences": "false",
        "recursive": "true",
        "includeImports": "false",
    }


def test_default_api_url_uses_current_us_cloud_host(monkeypatch):
    calls = []

    def fake_http(method, url, **kwargs):
        calls.append((method, url, kwargs))
        if method == "POST":
            return {"accessToken": "access-token"}
        return {"secrets": []}

    monkeypatch.setattr(inf, "_http_json", fake_http)

    inf.fetch_infisical_secrets(
        client_id="cid",
        client_secret="csecret",
        project_id="proj",
        use_cache=False,
    )

    assert calls[0][1] == "https://us.infisical.com/api/v1/auth/universal-auth/login"
    assert calls[1][1] == "https://us.infisical.com/api/v4/secrets"


def test_extract_includes_imports_with_lower_precedence():
    payload = {
        "imports": [
            {
                "secrets": [
                    {"secretKey": "SHARED_KEY", "secretValue": "imported"},
                    {"secretKey": "IMPORT_ONLY", "secretValue": "yes"},
                ]
            }
        ],
        "secrets": [
            {"secretKey": "SHARED_KEY", "secretValue": "local"},
            {"secretKey": "LOCAL_ONLY", "secretValue": "yes"},
        ],
    }

    secrets, warnings = inf._extract_secret_values(payload, include_imports=True)

    assert secrets == {
        "SHARED_KEY": "local",
        "IMPORT_ONLY": "yes",
        "LOCAL_ONLY": "yes",
    }
    assert any("Duplicate secret 'SHARED_KEY'" in warning for warning in warnings)


def test_fetch_skips_invalid_env_names(monkeypatch):
    def fake_http(method, url, **kwargs):
        if method == "POST":
            return {"accessToken": "access-token"}
        return {
            "secrets": [
                {"secretKey": "VALID_KEY", "secretValue": "v1"},
                {"secretKey": "1BAD", "secretValue": "v2"},
                {"secretKey": "HAS-DASH", "secretValue": "v3"},
            ]
        }

    monkeypatch.setattr(inf, "_http_json", fake_http)

    secrets, warnings = inf.fetch_infisical_secrets(
        client_id="cid",
        client_secret="csecret",
        project_id="proj",
        use_cache=False,
    )

    assert secrets == {"VALID_KEY": "v1"}
    assert len(warnings) == 2


def test_fetch_cache_hits(monkeypatch):
    call_count = {"n": 0}

    def fake_http(method, url, **kwargs):
        if method == "POST":
            return {"accessToken": "access-token"}
        call_count["n"] += 1
        return {"secrets": [{"secretKey": "KEY", "secretValue": "value"}]}

    monkeypatch.setattr(inf, "_http_json", fake_http)

    inf.fetch_infisical_secrets(
        client_id="cid",
        client_secret="csecret",
        project_id="proj",
        cache_ttl_seconds=60,
    )
    inf.fetch_infisical_secrets(
        client_id="cid",
        client_secret="csecret",
        project_id="proj",
        cache_ttl_seconds=60,
    )

    assert call_count["n"] == 1


def test_fetch_cache_returns_defensive_copy(monkeypatch):
    def fake_http(method, url, **kwargs):
        if method == "POST":
            return {"accessToken": "access-token"}
        return {"secrets": [{"secretKey": "KEY", "secretValue": "value"}]}

    monkeypatch.setattr(inf, "_http_json", fake_http)

    secrets, _warnings = inf.fetch_infisical_secrets(
        client_id="cid",
        client_secret="csecret",
        project_id="proj",
        cache_ttl_seconds=60,
    )
    secrets["KEY"] = "mutated"

    cached, _warnings = inf.fetch_infisical_secrets(
        client_id="cid",
        client_secret="csecret",
        project_id="proj",
        cache_ttl_seconds=60,
    )

    assert cached == {"KEY": "value"}


def test_http_json_wraps_os_errors(monkeypatch):
    def fake_urlopen(*_args, **_kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr(inf.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Infisical API request failed"):
        inf._http_json("GET", "https://infisical.example.com/api")


def test_apply_missing_bootstrap_credentials(monkeypatch):
    monkeypatch.delenv("INFISICAL_CLIENT_ID", raising=False)
    monkeypatch.delenv("INFISICAL_CLIENT_SECRET", raising=False)

    result = inf.apply_infisical_secrets(enabled=True, project_id="proj")

    assert not result.ok
    assert result.error is not None
    assert "INFISICAL_CLIENT_ID" in result.error


def test_apply_project_id_env_fallback(monkeypatch):
    monkeypatch.setenv("INFISICAL_CLIENT_ID", "cid")
    monkeypatch.setenv("INFISICAL_CLIENT_SECRET", "csecret")
    monkeypatch.setenv("INFISICAL_PROJECT_ID", "proj-from-env")
    captured = {}

    def fake_fetch(**kwargs):
        captured.update(kwargs)
        return {"NEW_KEY": "fresh"}, []

    monkeypatch.setattr(inf, "fetch_infisical_secrets", fake_fetch)

    result = inf.apply_infisical_secrets(enabled=True, project_id="")

    assert result.ok
    assert captured["project_id"] == "proj-from-env"
    assert os.environ["NEW_KEY"] == "fresh"
    assert "NEW_KEY" in result.applied


def test_apply_does_not_override_existing(monkeypatch):
    monkeypatch.setenv("INFISICAL_CLIENT_ID", "cid")
    monkeypatch.setenv("INFISICAL_CLIENT_SECRET", "csecret")
    monkeypatch.setenv("OPENAI_API_KEY", "existing")

    monkeypatch.setattr(
        inf,
        "fetch_infisical_secrets",
        lambda **_kwargs: (
            {"OPENAI_API_KEY": "fresh", "NEW_KEY": "new"},
            [],
        ),
    )

    result = inf.apply_infisical_secrets(
        enabled=True,
        project_id="proj",
        override_existing=False,
    )

    assert result.ok
    assert os.environ["OPENAI_API_KEY"] == "existing"
    assert os.environ["NEW_KEY"] == "new"
    assert "OPENAI_API_KEY" in result.skipped
    assert "NEW_KEY" in result.applied


def test_apply_never_overrides_bootstrap_credentials(monkeypatch):
    monkeypatch.setenv("INFISICAL_CLIENT_ID", "cid")
    monkeypatch.setenv("INFISICAL_CLIENT_SECRET", "original-secret")

    monkeypatch.setattr(
        inf,
        "fetch_infisical_secrets",
        lambda **_kwargs: (
            {"INFISICAL_CLIENT_SECRET": "malicious-replacement"},
            [],
        ),
    )

    result = inf.apply_infisical_secrets(
        enabled=True,
        project_id="proj",
        override_existing=True,
    )

    assert os.environ["INFISICAL_CLIENT_SECRET"] == "original-secret"
    assert "INFISICAL_CLIENT_SECRET" in result.skipped


def test_apply_swallows_fetch_errors(monkeypatch):
    monkeypatch.setenv("INFISICAL_CLIENT_ID", "cid")
    monkeypatch.setenv("INFISICAL_CLIENT_SECRET", "csecret")

    def fake_fetch(**_kwargs):
        raise RuntimeError("bad auth")

    monkeypatch.setattr(inf, "fetch_infisical_secrets", fake_fetch)

    result = inf.apply_infisical_secrets(enabled=True, project_id="proj")

    assert not result.ok
    assert result.error == "bad auth"
