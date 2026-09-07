"""Copilot routing contracts across credentials, catalog and SDK request boundaries."""

from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from agent.anthropic_adapter import build_anthropic_client
from agent.credential_pool import PooledCredential
from hermes_cli import models, runtime_provider


@pytest.mark.parametrize("credential_path", ["explicit", "normal", "pool"])
@pytest.mark.parametrize("default,saved,target,expected", [
    ("gpt-5.5", "codex_responses", "claude-sonnet-4.6", "anthropic_messages"),
    ("claude-sonnet-4.6", "anthropic_messages", "gpt-5.5", "codex_responses"),
])
def test_selected_model_controls_all_credential_paths(
    monkeypatch, credential_path, default, saved, target, expected,
):
    monkeypatch.setattr(runtime_provider, "_get_model_config", lambda: {
        "provider": "copilot", "default": default, "api_mode": saved,
    })
    monkeypatch.setattr(models, "fetch_github_model_catalog", lambda **kwargs: [
        {"id": "claude-sonnet-4.6", "supported_endpoints": ["/v1/messages", "/chat/completions"]},
    ])
    monkeypatch.setattr("hermes_cli.copilot_auth._try_gh_cli_token", lambda: "test-normal-token")
    kwargs = {"requested": "copilot", "target_model": target}
    pool = None
    if credential_path == "explicit":
        kwargs["explicit_api_key"] = "test-explicit-token"
    elif credential_path == "pool":
        entry = PooledCredential.from_dict("copilot", {
            "access_token": "test-pool-token", "base_url": "https://api.githubcopilot.com",
        })
        pool = SimpleNamespace(provider="copilot", has_credentials=lambda: True, select=lambda: entry)
        monkeypatch.setattr(runtime_provider, "load_pool", lambda provider: pool)
    else:
        monkeypatch.setattr(runtime_provider, "load_pool", lambda provider: None)
    result = runtime_provider.resolve_runtime_provider(**kwargs)
    assert result["api_mode"] == expected
    assert result["api_key"] == f"test-{credential_path}-token"
    if pool is not None:
        assert result["credential_pool"] is pool


@pytest.mark.parametrize("target", [None, "claude-sonnet-4.6", "copilot/claude-sonnet-4.6"])
def test_same_model_preserves_saved_mode_without_catalog(monkeypatch, target):
    fetch = Mock(side_effect=RuntimeError("Catalog unavailable"))
    monkeypatch.setattr(models, "fetch_github_model_catalog", fetch)
    result = runtime_provider._copilot_runtime_api_mode({
        "provider": "copilot", "default": "claude-sonnet-4.6", "api_mode": "anthropic_messages",
    }, "test-token", target_model=target)
    assert result == "anthropic_messages"
    fetch.assert_not_called()


@pytest.mark.parametrize("catalog,expected", [
    ([{"id": "grok-4.5", "supported_endpoints": [" /responses ", "/v1/messages"]}], "codex_responses"),
    ([{"id": "grok-4.5", "supported_endpoints": ["/chat/completions", "/v1/messages"]}], "anthropic_messages"),
    ([{"id": "grok-4.5", "supported_endpoints": ["/chat/completions"]}], "chat_completions"),
    ([{"id": "grok-4.5", "supported_endpoints": None}], "chat_completions"),
    ([{"id": "another-model", "supported_endpoints": ["/responses"]}], "chat_completions"),
    ([], "chat_completions"),
])
def test_catalog_endpoint_precedence(catalog, expected):
    assert models.copilot_model_api_mode("grok-4.5", catalog=catalog) == expected


def test_catalog_failure_falls_back_without_stale_mode(monkeypatch):
    fetch = Mock(side_effect=RuntimeError("Catalog unavailable"))
    monkeypatch.setattr(models, "fetch_github_model_catalog", fetch)
    result = runtime_provider._copilot_runtime_api_mode({
        "provider": "copilot", "default": "gpt-5.5", "api_mode": "codex_responses",
    }, "test-token", target_model="claude-sonnet-4.6")
    assert result == "chat_completions"
    fetch.assert_called()


@pytest.mark.parametrize("base_url", [
    "https://api.githubcopilot.com", "https://api.githubcopilot.com/v1",
    "https://enterprise.githubcopilot.com/v1/",
])
def test_real_sdk_sends_copilot_messages_with_bearer_only(monkeypatch, base_url):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-unrelated-anthropic-key")
    requests = []

    def capture(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "test-message", "type": "message", "role": "assistant",
            "model": "claude-sonnet-4.6", "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    # Retain the real SDK constructor and serialization; only HTTP transport is replaced.
    with build_anthropic_client("test-copilot-token", base_url=base_url) as client:
        with httpx.Client(transport=httpx.MockTransport(capture)) as transport, monkeypatch.context() as scoped:
            scoped.setattr(client, "_client", transport)
            response = client.messages.create(
                model="claude-sonnet-4.6", max_tokens=8,
                messages=[{"role": "user", "content": "hello"}],
            )
        assert response.content[0].text == "ok"
    assert len(requests) == 1
    request = requests[0]
    assert request.url.path == "/v1/messages"
    assert request.url.host == httpx.URL(base_url).host
    assert request.headers["authorization"] == "Bearer test-copilot-token"
    assert "x-api-key" not in request.headers


@pytest.mark.parametrize("base_url", [
    "https://githubcopilot.com.evil.example", "https://evil.example/githubcopilot.com",
])
def test_lookalike_hosts_do_not_get_copilot_bearer_auth(base_url):
    from agent.anthropic_endpoints import _requires_bearer_auth

    assert not _requires_bearer_auth(base_url)
