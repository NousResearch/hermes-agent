"""Deployment discovery must return deployed aliases, with profile-owned credentials."""

from types import SimpleNamespace

import pytest

from agent.azure_identity_adapter import EntraIdentityConfig
from hermes_cli import azure_foundry_deployments as deployments


def test_project_pages_return_deployment_names_without_requesting_arm_credentials(monkeypatch):
    requests = []
    def get(url, credential, **_):
        requests.append(url)
        assert credential() == "inference-token"
        if len(requests) == 1:
            return 200, {"value": [{"name": "custom-alias", "modelName": "catalog-model", "type": "ModelDeployment"}],
                         "nextLink": "?api-version=v1&skip=1"}
        return 200, {"value": [{"name": "another-alias", "type": "ModelDeployment"}]}
    monkeypatch.setattr(deployments, "_http_get_json", get)
    monkeypatch.setattr(deployments, "_probe_token", lambda *_args, **_kwargs: pytest.fail("Project read must not request ARM token"))
    result = deployments.discover_deployments("https://resource.services.ai.azure.com/api/projects/project/openai/v1", lambda: "inference-token", EntraIdentityConfig(credential_source="azure_cli"))
    assert result == ["another-alias", "custom-alias"]
    assert requests == ["https://resource.services.ai.azure.com/api/projects/project/deployments?api-version=v1",
                        "https://resource.services.ai.azure.com/api/projects/project/deployments?api-version=v1&skip=1"]


@pytest.mark.parametrize("source,selector", [("azure_cli", {"subscription": "Selected subscription"}), ("azure_cli", {"tenant_id": "selected-tenant"}), ("default", {})])
def test_resource_discovery_uses_same_identity_and_exact_resource(monkeypatch, source, selector):
    configs, requests = [], []
    sid = "11111111-1111-1111-1111-111111111111"
    rid = f"/subscriptions/{sid}/resourceGroups/group/providers/Microsoft.CognitiveServices/accounts/arm-resource-name"
    def provider(config, timeout):
        configs.append(config)
        assert 0 < timeout <= 18
        return {"token": SimpleNamespace(token="arm-token")}
    def get(url, credential, **_):
        requests.append(url)
        assert credential() == "arm-token"
        if "/subscriptions?" in url:
            return 200, {"value": [{"subscriptionId": sid, "displayName": "Selected subscription"}]}
        if "/accounts?" in url:
            return 200, {"value": [{"name": "resource", "id": rid.replace("arm-resource-name", "resource"), "properties": {"customSubDomainName": "different-resource"}},
                                  {"name": "arm-resource-name", "id": rid, "properties": {"customSubDomainName": "resource"}}]}
        assert url == f"https://management.azure.com{rid}/deployments?api-version=2024-10-01"
        return 200, {"value": [{"name": "deployed-alias", "properties": {"model": {"name": "catalog-model"}, "provisioningState": "Succeeded"}},
                              {"name": "failed-deployment", "properties": {"provisioningState": "Failed"}}]}
    monkeypatch.setattr(deployments, "_probe_token", provider)
    monkeypatch.setattr(deployments, "_http_get_json", get)
    config = EntraIdentityConfig(credential_source=source, **selector)
    assert deployments.discover_deployments("https://resource.services.ai.azure.com/openai/v1", lambda: "inference-token", config) == ["deployed-alias"]
    assert configs[0].credential_source == config.credential_source
    assert configs[0].tenant_id == config.tenant_id and configs[0].subscription == config.subscription
    assert configs[0].scope == "https://management.azure.com/.default"
    assert "/providers/Microsoft.CognitiveServices/accounts?" in requests[1]


@pytest.mark.parametrize("next_link", ["https://unrelated.example/deployments", "/different-project/deployments"])
def test_pagination_never_forwards_credentials_to_a_different_origin_or_path(monkeypatch, next_link):
    requests = []
    def get(url, *_args, **_kwargs):
        requests.append(url)
        return 200, {"value": [{"name": "partial", "type": "ModelDeployment"}], "nextLink": next_link}
    monkeypatch.setattr(deployments, "_http_get_json", get)
    assert deployments.discover_deployments("https://resource.services.ai.azure.com/api/projects/project/openai/v1", "api-key") == []
    assert len(requests) == 1


@pytest.mark.parametrize("status,body", [(403, None), (200, {"data": [{"id": "catalog-model"}]}), (200, {"value": []})])
def test_unavailable_listing_and_catalog_payloads_never_become_deployment_suggestions(monkeypatch, status, body):
    monkeypatch.setattr(deployments, "_http_get_json", lambda *_args, **_kwargs: (status, body))
    assert deployments.discover_deployments("https://resource.services.ai.azure.com/api/projects/project/openai/v1", "api-key") == []


def test_resource_api_key_does_not_borrow_an_ambient_entra_identity(monkeypatch):
    monkeypatch.setattr(deployments, "_probe_token", lambda *_args, **_kwargs: pytest.fail("API key cannot acquire an ARM credential"))
    monkeypatch.setattr(deployments, "_http_get_json", lambda *_args, **_kwargs: pytest.fail("No project endpoint was provided"))
    assert deployments.discover_deployments("https://resource.openai.azure.com/openai/v1", "api-key") == []


def test_arm_discovery_is_not_inferred_from_a_lookalike_or_proxy_hostname(monkeypatch):
    monkeypatch.setattr(deployments, "_probe_token", lambda *_args, **_kwargs: pytest.fail("Must not infer an ARM resource"))
    config = EntraIdentityConfig(credential_source="azure_cli")
    assert deployments.discover_deployments("https://resource.services.ai.azure.com.example.com/openai/v1", lambda: "token", config) == []


def test_page_and_time_budgets_bound_discovery(monkeypatch):
    requests = []
    def get(url, *_args, **_kwargs):
        requests.append(url)
        return 200, {"value": [], "nextLink": "?page=next"}
    monkeypatch.setattr(deployments, "_http_get_json", get)
    monkeypatch.setattr(deployments.time, "monotonic", lambda: 100.0)
    assert deployments._items("https://example.com/deployments", "token", 200.0) is None
    assert len(requests) == 20
    requests.clear()
    assert deployments._items("https://example.com/deployments", "token", 99.0) is None
    assert requests == []


def test_optional_arm_auth_failure_keeps_manual_inference_setup_available(monkeypatch):
    def unavailable(*_args):
        return {"error": "sensitive-auth-error"}
    monkeypatch.setattr(deployments, "_probe_token", unavailable)
    assert deployments.discover_deployments("https://resource.openai.azure.com/openai/v1", lambda: "inference-token", EntraIdentityConfig(credential_source="azure_cli")) == []


def test_exhausted_budget_does_not_start_an_arm_token_probe(monkeypatch):
    clock = iter([100.0, 119.0])
    monkeypatch.setattr(deployments.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(deployments, "_probe_token", lambda *_: pytest.fail("Budget already exhausted"))
    assert deployments.discover_deployments("https://resource.openai.azure.com/openai/v1", lambda: "token", EntraIdentityConfig(credential_source="azure_cli")) == []


def test_timed_out_arm_probe_does_not_start_resource_reads(monkeypatch):
    monkeypatch.setattr(deployments, "_probe_token", lambda *_: None)
    monkeypatch.setattr(deployments, "_http_get_json", lambda *_args, **_kwargs: pytest.fail("ARM token unavailable"))
    assert deployments.discover_deployments("https://resource.openai.azure.com/openai/v1", lambda: "token", EntraIdentityConfig(credential_source="azure_cli")) == []


def test_matching_resource_uses_endpoint_metadata_when_custom_subdomain_is_absent():
    assert deployments._matches_endpoint({"name": "different-name", "properties": {"endpoints": {"OpenAI": "https://resource.openai.azure.com/"}}}, "https://resource.openai.azure.com/openai/v1")
    assert not deployments._matches_endpoint({"name": "resource", "properties": {"endpoint": "https://different.openai.azure.com/"}}, "https://resource.openai.azure.com/openai/v1")
