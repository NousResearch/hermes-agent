"""Azure Foundry in the /model picker (#27989).

Deployments are per-resource so the static catalog is intentionally empty; ``provider_model_ids``
must list the resource's ``/openai/deployments`` through the runtime credential resolver (API key OR
Entra ID token provider), fall back to ``/models`` only when that route is unavailable, and keep the
static ``[]`` on any failure. Only the HTTP leaf is stubbed; no real endpoint is contacted.
"""

from __future__ import annotations

from unittest.mock import patch

_BASE = "https://r.openai.azure.com/openai/v1"
_DEPLOYMENTS = {"data": [{"id": "gpt-5.6-sol", "status": "succeeded"}, {"id": "kimi-k2.6", "status": "succeeded"},
                         {"id": "o4-mini", "status": "provisioning"}]}
_CATALOG = {"object": "list", "data": [{"id": f"model-{i}"} for i in range(400)]}


def _http(routes):
    """``_http_get_json`` stub keyed by URL substring; records the auth each call carried."""
    seen = []

    def fake(url, api_key, timeout=6.0, *, token_provider=None):
        seen.append((url, api_key, token_provider))
        for needle, (status, body) in routes.items():
            if needle in url:
                return status, body
        return 404, None

    return fake, seen


class TestAzureFoundryPicker:
    def test_lists_the_resource_deployments_not_the_whole_catalog(self, monkeypatch):
        from hermes_cli.models import provider_model_ids

        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "az-secret")
        monkeypatch.setenv("AZURE_FOUNDRY_BASE_URL", _BASE)
        fake, seen = _http({"/openai/deployments?api-version=2023-03-15-preview": (200, _DEPLOYMENTS),
                            "/models": (200, _CATALOG)})

        with patch("hermes_cli.azure_detect._http_get_json", fake):
            ids = provider_model_ids("azure-foundry")

        # Succeeded first, provisioning after; the 400-id catalog is never consulted.
        assert ids == ["gpt-5.6-sol", "kimi-k2.6", "o4-mini"]
        assert [u for u, *_ in seen] == ["https://r.openai.azure.com/openai/deployments?api-version=2023-03-15-preview"]
        assert seen[0][1] == "az-secret"

    def test_falls_back_to_models_when_the_deployments_route_is_unavailable(self, monkeypatch):
        from hermes_cli.models import provider_model_ids

        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "az-secret")
        monkeypatch.setenv("AZURE_FOUNDRY_BASE_URL", "https://gateway.example/v1")
        fake, _ = _http({"/models": (200, {"data": [{"id": "gpt-4.1"}]})})

        with patch("hermes_cli.azure_detect._http_get_json", fake):
            assert provider_model_ids("azure-foundry") == ["gpt-4.1"]

    def test_entra_id_forwards_the_token_provider_instead_of_a_string_key(self, monkeypatch):
        from hermes_cli.models import provider_model_ids

        def sentinel_token_provider() -> str:
            return "fresh-entra-jwt"

        def _fake_runtime(*, requested_provider, model_cfg, **_kw):
            assert requested_provider == "azure-foundry"
            return {"provider": "azure-foundry", "api_mode": "chat_completions", "base_url": _BASE,
                    "api_key": sentinel_token_provider, "auth_mode": "entra_id", "source": "entra_id"}

        monkeypatch.setattr("hermes_cli.runtime_provider._resolve_azure_foundry_runtime", _fake_runtime)
        fake, seen = _http({"/openai/deployments": (200, _DEPLOYMENTS)})

        with patch("hermes_cli.azure_detect._http_get_json", fake):
            ids = provider_model_ids("azure-foundry")

        assert ids[:2] == ["gpt-5.6-sol", "kimi-k2.6"]
        assert seen[0][1] is sentinel_token_provider  # azure_detect mints the bearer from the callable

    def test_any_failure_keeps_the_static_empty_catalog(self, monkeypatch):
        from hermes_cli.models import provider_model_ids

        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "az-secret")
        monkeypatch.setenv("AZURE_FOUNDRY_BASE_URL", _BASE)

        with patch("hermes_cli.azure_detect._http_get_json", side_effect=RuntimeError("network down")):
            assert provider_model_ids("azure-foundry") == []

        monkeypatch.delenv("AZURE_FOUNDRY_API_KEY")
        assert provider_model_ids("azure-foundry") == []


def test_entra_only_azure_foundry_row_reaches_the_picker_without_an_api_key(monkeypatch, tmp_path):
    """Section 1 of the picker gates rows on an API-key env var; an Entra-configured Azure Foundry
    profile has none and must still surface with its discovered deployments."""
    import yaml

    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(yaml.safe_dump({"model": {
        "provider": "azure-foundry", "default": "gpt-5.6-sol", "base_url": _BASE, "auth_mode": "entra_id"}}))

    def _fake_runtime(*, requested_provider, model_cfg, **_kw):
        return {"provider": "azure-foundry", "api_mode": "chat_completions", "base_url": _BASE,
                "api_key": lambda: "fresh-entra-jwt", "auth_mode": "entra_id", "source": "entra_id"}

    monkeypatch.setattr("hermes_cli.runtime_provider._resolve_azure_foundry_runtime", _fake_runtime)
    fake, _ = _http({"/openai/deployments": (200, _DEPLOYMENTS)})
    from hermes_cli.model_switch_providers import list_authenticated_providers

    with patch("hermes_cli.azure_detect._http_get_json", fake), patch("agent.models_dev.fetch_models_dev", return_value={
            "azure": {"env": ["AZURE_FOUNDRY_API_KEY"], "name": "Azure", "models": {}}}):
        rows = [r for r in list_authenticated_providers(for_picker=True) if r["slug"] == "azure-foundry"]

    assert rows and rows[0]["models"][:2] == ["gpt-5.6-sol", "kimi-k2.6"]
