from fastapi.testclient import TestClient

from hermes_cli.web_server import _SESSION_TOKEN, app


client = TestClient(app)
HEADERS = {"X-Hermes-Session-Token": _SESSION_TOKEN}


def _provider(provider_id: str) -> dict:
    response = client.get("/api/providers/credentials", headers=HEADERS)
    assert response.status_code == 200
    return next(row for row in response.json()["providers"] if row["id"] == provider_id)


def test_provider_credentials_expose_owner_roles_and_order():
    gemini = _provider("gemini")
    fields = {field["key"]: field for field in gemini["fields"]}

    assert fields["GOOGLE_API_KEY"]["role"] == "primary_secret"
    assert fields["GEMINI_API_KEY"]["role"] == "secret_alias"
    assert fields["GEMINI_BASE_URL"]["role"] == "endpoint"
    assert [field["order"] for field in gemini["fields"]] == sorted(
        field["order"] for field in gemini["fields"]
    )

    bedrock = _provider("bedrock")
    bedrock_fields = {field["key"]: field for field in bedrock["fields"]}
    assert bedrock_fields["AWS_REGION"]["role"] == "setting"
    assert bedrock_fields["AWS_PROFILE"]["role"] == "setting"


def test_provider_credential_mutation_is_bound_to_provider_identity():
    wrong = client.put(
        "/api/providers/credentials/xai/GOOGLE_API_KEY",
        json={"value": "secret"},
        headers=HEADERS,
    )
    assert wrong.status_code == 404

    saved = client.put(
        "/api/providers/credentials/gemini/GOOGLE_API_KEY",
        json={"value": "secret"},
        headers=HEADERS,
    )
    assert saved.status_code == 200
    assert next(
        field for field in _provider("gemini")["fields"] if field["key"] == "GOOGLE_API_KEY"
    )["is_set"] is True

    deleted = client.delete(
        "/api/providers/credentials/gemini/GOOGLE_API_KEY",
        headers=HEADERS,
    )
    assert deleted.status_code == 200
    assert next(
        field for field in _provider("gemini")["fields"] if field["key"] == "GOOGLE_API_KEY"
    )["is_set"] is False


def test_unavailable_validation_capability_is_explicit():
    response = client.post(
        "/api/providers/credentials/bedrock/AWS_PROFILE/validate",
        json={"value": "research"},
        headers=HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["status"] == "unsupported"


def test_account_status_has_safe_owner_display_label(monkeypatch):
    import hermes_cli.web_server as web_server

    monkeypatch.setattr(
        web_server,
        "_resolve_provider_status",
        lambda _provider_id, _status_fn: {
            "logged_in": True,
            "source": "auth_store",
            "source_label": "/home/private/.credentials.json",
        },
    )
    response = client.get("/api/providers/oauth", headers=HEADERS)
    assert response.status_code == 200
    for provider in response.json()["providers"]:
        assert provider["status"]["display_label"] == provider["name"]
        assert "/home/private" not in provider["status"]["display_label"]


def test_model_options_expose_owner_setup_surface(monkeypatch):
    import hermes_cli.inventory as inventory

    monkeypatch.setattr(inventory, "load_picker_context", lambda: object())
    monkeypatch.setattr(
        inventory,
        "build_model_options_payload",
        lambda *_args, **_kwargs: {
            "providers": [
                {"slug": "gemini", "name": "Gemini", "models": []},
                {"slug": "nous", "name": "Nous", "models": []},
                {"slug": "custom", "name": "Custom", "models": []},
            ]
        },
    )
    response = client.get(
        "/api/model/options?include_unconfigured=true&explicit_only=true",
        headers=HEADERS,
    )
    assert response.status_code == 200
    providers = {row["slug"]: row for row in response.json()["providers"]}
    assert providers["gemini"]["setup_kind"] == "credential"
    assert providers["nous"]["setup_kind"] == "account"
    assert providers["custom"]["setup_kind"] == "custom_endpoint"
