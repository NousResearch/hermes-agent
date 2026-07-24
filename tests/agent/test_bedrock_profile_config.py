"""Regression coverage for named-profile Bedrock configuration and caches."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import agent.bedrock_adapter as bedrock
from agent.anthropic_adapter import build_anthropic_bedrock_client


def setup_function() -> None:
    bedrock.reset_client_cache()
    bedrock.reset_discovery_cache()


def test_runtime_client_cache_isolated_by_profile(monkeypatch) -> None:
    active_profile = {"value": "dev"}
    monkeypatch.setattr(
        bedrock,
        "_resolve_bedrock_profile",
        lambda: active_profile["value"],
    )

    dev_client = object()
    prod_client = object()
    dev_session = MagicMock()
    prod_session = MagicMock()
    dev_session.client.return_value = dev_client
    prod_session.client.return_value = prod_client
    boto3 = MagicMock()
    boto3.Session.side_effect = [dev_session, prod_session]
    monkeypatch.setattr(bedrock, "_require_boto3", lambda: boto3)

    assert bedrock._get_bedrock_runtime_client("us-east-1") is dev_client
    active_profile["value"] = "prod"
    assert bedrock._get_bedrock_runtime_client("us-east-1") is prod_client

    assert boto3.Session.call_args_list[0].kwargs == {"profile_name": "dev"}
    assert boto3.Session.call_args_list[1].kwargs == {"profile_name": "prod"}
    assert set(bedrock._bedrock_runtime_client_cache) == {
        ("us-east-1", "dev"),
        ("us-east-1", "prod"),
    }


def test_default_chain_preserves_region_only_cache_key(monkeypatch) -> None:
    monkeypatch.setattr(bedrock, "_resolve_bedrock_profile", lambda: "")
    client = object()
    boto3 = MagicMock()
    boto3.client.return_value = client
    monkeypatch.setattr(bedrock, "_require_boto3", lambda: boto3)

    assert bedrock._get_bedrock_control_client("eu-west-1") is client
    assert bedrock._get_bedrock_control_client("eu-west-1") is client
    boto3.client.assert_called_once_with("bedrock", region_name="eu-west-1")
    assert bedrock._bedrock_control_client_cache == {"eu-west-1": client}


def test_discovery_cache_isolated_when_profile_changes(monkeypatch) -> None:
    active_profile = {"value": "dev"}
    monkeypatch.setattr(
        bedrock,
        "_resolve_bedrock_profile",
        lambda: active_profile["value"],
    )

    client = MagicMock()
    client.list_foundation_models.side_effect = [
        {
            "modelSummaries": [
                {
                    "modelId": "dev.model",
                    "modelName": "Dev",
                    "providerName": "Dev",
                    "modelLifecycle": {"status": "ACTIVE"},
                    "responseStreamingSupported": True,
                    "outputModalities": ["TEXT"],
                }
            ]
        },
        {
            "modelSummaries": [
                {
                    "modelId": "prod.model",
                    "modelName": "Prod",
                    "providerName": "Prod",
                    "modelLifecycle": {"status": "ACTIVE"},
                    "responseStreamingSupported": True,
                    "outputModalities": ["TEXT"],
                }
            ]
        },
    ]
    client.list_inference_profiles.return_value = {}
    monkeypatch.setattr(bedrock, "_get_bedrock_control_client", lambda _region: client)

    assert [item["id"] for item in bedrock.discover_bedrock_models("us-east-1")] == [
        "dev.model"
    ]
    active_profile["value"] = "prod"
    assert [item["id"] for item in bedrock.discover_bedrock_models("us-east-1")] == [
        "prod.model"
    ]
    assert client.list_foundation_models.call_count == 2
    assert len(bedrock._discovery_cache) == 2


def test_anthropic_bedrock_preserves_retry_policy_and_profile(monkeypatch) -> None:
    sdk = SimpleNamespace(AnthropicBedrock=MagicMock())
    monkeypatch.setattr("agent.anthropic_adapter._get_anthropic_sdk", lambda: sdk)

    with patch("agent.bedrock_adapter._resolve_bedrock_profile", return_value="research"):
        build_anthropic_bedrock_client("us-west-2")

    kwargs = sdk.AnthropicBedrock.call_args.kwargs
    assert kwargs["aws_region"] == "us-west-2"
    assert kwargs["aws_profile"] == "research"
    assert kwargs["max_retries"] == 0
