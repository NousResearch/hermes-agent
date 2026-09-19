"""No-send regressions for catalog, reasoning, image, status, and doctor probes."""
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.routing_policy import RoutingPolicyError


def _denied(*_args, **_kwargs):
    raise RoutingPolicyError("denied")


def test_ollama_native_catalog_denial_prevents_urlopen():
    from hermes_cli.models_local import probe_ollama_local_models

    with patch("hermes_cli.routing_policy.check_outbound_route", side_effect=_denied), \
         patch("hermes_cli.models._urlopen_model_catalog_request") as send:
        with pytest.raises(RoutingPolicyError, match="denied"):
            probe_ollama_local_models("https://ollama.example/v1")
    send.assert_not_called()


def test_models_dev_registry_denial_prevents_http_get():
    from agent.models_dev import _fetch_models_dev_from_network

    with patch("hermes_cli.routing_policy.check_outbound_route", side_effect=_denied), \
         patch("agent.models_dev.requests.get") as send:
        with pytest.raises(RoutingPolicyError, match="denied"):
            _fetch_models_dev_from_network()
    send.assert_not_called()


def test_image_catalog_denial_prevents_http_get():
    from plugins.image_gen.openrouter import _get_catalog

    with patch("hermes_cli.routing_policy.check_outbound_route", side_effect=_denied), \
         patch("requests.get") as send:
        with pytest.raises(RoutingPolicyError, match="denied"):
            _get_catalog("https://openrouter.ai/api/v1", "/images/models", "test-key", 1)
    send.assert_not_called()


def test_image_generation_denial_prevents_final_post():
    from plugins.image_gen.openrouter import _build_providers

    provider = _build_providers()[0]
    with patch("hermes_cli.routing_policy.check_outbound_route", side_effect=_denied), \
         patch("plugins.image_gen.openrouter.post_json") as send:
        with pytest.raises(RoutingPolicyError, match="denied"):
            provider._generate_via_chat(
                model_id="openai/gpt-image-2", prompt="p", aspect="square",
                content=[{"type": "text", "text": "p"}], base_url="https://openrouter.ai/api/v1",
                headers={"Authorization": "Bearer test-key"},
            )
    send.assert_not_called()


def test_reasoning_catalog_denial_is_terminal_and_does_not_cache_failure():
    import hermes_cli.models_reasoning_caps as caps

    with patch("hermes_cli.routing_policy.check_outbound_route", side_effect=_denied), \
         patch("hermes_cli.models._urlopen_model_catalog_request") as send:
        with pytest.raises(RoutingPolicyError, match="denied"):
            caps._fetch_reasoning_caps_catalog("https://openrouter.ai/api/v1/models", 1)
    send.assert_not_called()


def test_manifest_denial_prevents_urlopen():
    from hermes_cli.model_catalog import _fetch_manifest

    with patch("hermes_cli.routing_policy.check_outbound_route", side_effect=_denied), \
         patch("hermes_cli.model_catalog.urllib.request.urlopen") as send:
        with pytest.raises(RoutingPolicyError, match="denied"):
            _fetch_manifest("https://openrouter.ai/catalog.json", 1)
    send.assert_not_called()


def test_doctor_openrouter_denial_prevents_http_get(monkeypatch):
    from hermes_cli.doctor_connectivity import _probe_openrouter

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with patch("hermes_cli.routing_policy.check_outbound_route", side_effect=_denied), \
         patch("httpx.get") as send:
        with pytest.raises(RoutingPolicyError, match="denied"):
            _probe_openrouter()
    send.assert_not_called()


def test_status_deep_openrouter_denial_prevents_http_get(monkeypatch):
    import hermes_cli.status as status

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    ctx = MagicMock(deep=True)
    with patch("hermes_cli.routing_policy.check_outbound_route", side_effect=_denied), \
         patch("httpx.get") as send:
        with pytest.raises(RoutingPolicyError, match="denied"):
            status._render_deep(ctx)
    send.assert_not_called()


def test_doctor_glm_denial_prevents_generic_models_get(monkeypatch):
    from hermes_cli.doctor_connectivity import _probe_apikey_provider

    monkeypatch.setenv("GLM_API_KEY", "test-key")
    with patch("hermes_cli.routing_policy.check_outbound_route", side_effect=_denied) as admit, \
         patch("httpx.get") as send:
        with pytest.raises(RoutingPolicyError, match="denied"):
            _probe_apikey_provider(
                "Z.AI / GLM", ("GLM_API_KEY",), "https://api.z.ai/api/paas/v4/models", "GLM_BASE_URL", True,
            )
    admit.assert_called_once_with(
        provider="zai", model="", base_url="https://api.z.ai/api/paas/v4/models",
    )
    send.assert_not_called()


def test_nous_image_catalog_and_final_post_denial_prevent_sends():
    from plugins.image_gen.openrouter import _build_providers

    provider = _build_providers()[1]
    runtime = {"provider": "nous", "api_key": "test-key", "base_url": "https://inference-api.nousresearch.com/v1"}
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=runtime), \
         patch("hermes_cli.routing_policy.check_outbound_route", side_effect=_denied), \
         patch("requests.get") as catalog_send, \
         patch("plugins.image_gen.openrouter.post_json") as post_send:
        with pytest.raises(RoutingPolicyError, match="denied"):
            provider._live_models()
        with pytest.raises(RoutingPolicyError, match="denied"):
            provider._generate_via_chat(
                model_id="nous-image", prompt="p", aspect="square", content=[{"type": "text", "text": "p"}],
                base_url=runtime["base_url"], headers={"Authorization": "Bearer test-key"},
            )
    catalog_send.assert_not_called()
    post_send.assert_not_called()


def test_doctor_anthropic_denial_prevents_initial_and_oauth_retry(monkeypatch):
    from hermes_cli.doctor_connectivity import _probe_anthropic

    monkeypatch.setattr("hermes_cli.auth.get_anthropic_key", lambda: "test-key")
    with patch("agent.anthropic_credentials._is_oauth_token", return_value=True), \
         patch("hermes_cli.routing_policy.check_outbound_route", side_effect=_denied) as admit, \
         patch("httpx.get") as send:
        with pytest.raises(RoutingPolicyError, match="denied"):
            _probe_anthropic()
    admit.assert_called_once_with(provider="anthropic", model="", base_url="https://api.anthropic.com/v1/models")
    send.assert_not_called()


def test_doctor_dashscope_retry_denial_prevents_fallback_get(monkeypatch):
    from hermes_cli.doctor_connectivity import _probe_apikey_provider

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)
    first = MagicMock(status_code=401)
    with patch("hermes_cli.doctor_connectivity._apikey_request", return_value=("", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models", {})), \
         patch("hermes_cli.routing_policy.check_outbound_route", side_effect=[None, RoutingPolicyError("denied")]) as admit, \
         patch("httpx.get", return_value=first) as send:
        with pytest.raises(RoutingPolicyError, match="denied"):
            _probe_apikey_provider(
                "Alibaba/DashScope", ("DASHSCOPE_API_KEY",),
                "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models", "DASHSCOPE_BASE_URL", True,
            )
    assert send.call_count == 1
    assert admit.call_args_list[-1].kwargs == {
        "provider": "alibaba", "model": "", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/models",
    }


def test_doctor_bedrock_denial_prevents_client_construction(monkeypatch):
    from hermes_cli import doctor_connectivity as dc

    monkeypatch.setattr("agent.bedrock_adapter.has_aws_credentials", lambda: True)
    monkeypatch.setattr("agent.bedrock_adapter.resolve_aws_auth_env_var", lambda: "AWS_ACCESS_KEY_ID")
    monkeypatch.setattr("agent.bedrock_adapter.resolve_bedrock_region", lambda: "us-east-1")
    with patch("hermes_cli.routing_policy.check_outbound_route", side_effect=_denied), \
         patch("boto3.client") as client:
        with pytest.raises(RoutingPolicyError, match="denied"):
            dc._probe_bedrock()
    client.assert_not_called()


def test_auto_detect_local_model_denial_prevents_models_get():
    from hermes_cli.runtime_provider import _auto_detect_local_model

    with patch("hermes_cli.routing_policy.check_outbound_route", side_effect=_denied), \
         patch("requests.get") as send:
        with pytest.raises(RoutingPolicyError, match="denied"):
            _auto_detect_local_model("http://127.0.0.1:1234", provider="lmstudio")
    send.assert_not_called()


def test_zai_probe_denial_prevents_candidate_post():
    from hermes_cli.auth_zai_kimi import _probe_single_zai_endpoint

    endpoint = ("global", "https://api.z.ai/api/paas/v4", ["glm-5"], "Global")
    with patch("hermes_cli.routing_policy.check_outbound_route", side_effect=_denied), \
         patch("hermes_cli.auth_zai_kimi.httpx.post") as send:
        with pytest.raises(RoutingPolicyError, match="denied"):
            _probe_single_zai_endpoint("test-key", endpoint, 1)
    send.assert_not_called()


def test_image_surface_catalog_uses_runtime_provider_and_profile_owner():
    from plugins.image_gen.openrouter import _CATALOG_CACHE, _select_surface

    _CATALOG_CACHE.clear()
    with patch("plugins.image_gen.openrouter._get_catalog", return_value=[("new/image", {})]) as catalog:
        assert _select_surface(
            "new/image", "https://inference.example/v1", "test-key", "nous",
            provider="nous", profile_home="/profiles/restrictive",
        ) == "images"
    assert catalog.call_args.kwargs == {"provider": "nous", "profile_home": "/profiles/restrictive"}
