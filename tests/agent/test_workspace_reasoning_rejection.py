"""Workspace reasoning restrictions fail without retrying, mutating effort or switching models."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

from agent.error_classifier import FailoverReason, classify_api_error

BODY = {"error": {
    "type": "invalid_request_error", "code": "reasoning_effort_not_allowed",
    "message": "This reasoning level isn't available for the selected model in your workspace. "
               "Choose an available reasoning level or another model.",
}}


def rejection(status=403, shape="full"):
    body = BODY if shape == "full" else {"error": {shape: BODY["error"][shape]}}
    return openai.APIStatusError(
        "Rejected request", response=httpx.Response(status, request=httpx.Request("POST", "https://chatgpt.com")),
        body=body,
    )


@pytest.mark.parametrize("status", [400, 403])
@pytest.mark.parametrize("shape", ["full", "code", "message"])
def test_workspace_refusal_is_a_terminal_configuration_failure(status, shape):
    verdict = classify_api_error(rejection(status, shape), provider="openai-codex", model="gpt-6.1-terra")
    assert verdict.reason == FailoverReason.reasoning_configuration
    assert not any([verdict.retryable, verdict.should_compress, verdict.should_fallback, verdict.should_rotate_credential])
    assert not verdict.is_auth


@pytest.mark.parametrize("message,code,expected", [
    ("Invalid API key", None, FailoverReason.auth),
    ("This model isn't available in your workspace", None, FailoverReason.auth),
    ("spending limit exceeded; reasoning_effort_not_allowed", None, FailoverReason.billing),
    ("enable javascript and cookies to continue; reasoning_effort_not_allowed", None, FailoverReason.upstream_blocked),
    ("reasoning_effort_not_allowed", "upstream_unavailable", FailoverReason.overloaded),
])
def test_other_403_failures_keep_their_classification(message, code, expected):
    error = openai.PermissionDeniedError(
        "Rejected request", response=httpx.Response(403, request=httpx.Request("POST", "https://chatgpt.com")),
        body={"error": {"message": message, "code": code}},
    )
    assert classify_api_error(error).reason == expected


def test_configuration_failure_does_not_mutate_the_real_credential_pool_path():
    from agent.agent_runtime_helpers import recover_with_credential_pool

    pool = MagicMock()
    pool.provider = "openai-codex"
    agent = SimpleNamespace(provider="openai-codex", base_url="https://chatgpt.com/backend-api/codex",
                            model="gpt-6.1-terra", _credential_pool=pool)
    verdict = classify_api_error(rejection(), provider=agent.provider, model=agent.model)
    with patch("agent.agent_runtime_helpers.credential_pool_matches_provider", return_value=True), \
         patch("agent.agent_runtime_helpers._failed_credential_identity", return_value=("fake-key", "fake-id")):
        assert recover_with_credential_pool(
            agent, status_code=403, has_retried_429=False, classified_reason=verdict.reason,
            error_context=verdict.error_context,
        ) == (False, False)
    pool.try_refresh_matching.assert_not_called()
    pool.mark_exhausted_and_rotate.assert_not_called()


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_auxiliary_failure_is_not_repaired_and_a_configured_level_can_fix_it(async_mode, enabled):
    from agent import auxiliary_client as aux, auxiliary_reasoning_floor as floor

    floor._FLOORED_ROUTES.clear()
    client = MagicMock()
    client.base_url = "https://chatgpt.com/backend-api/codex"
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
    create = AsyncMock if async_mode else MagicMock
    client.chat.completions.create = create(side_effect=[rejection(), response])
    config = {"enabled": enabled, "effort": "high" if enabled else "none"}
    with patch.object(aux, "_resolve_task_provider_model", return_value=(
        "openai-codex", "gpt-6.1-terra", client.base_url, "fake-key", None,
    )), patch.object(aux, "_get_cached_client", return_value=(client, "gpt-6.1-terra")), \
         patch.object(aux, "_refresh_provider_credentials") as refresh, \
         patch.object(aux, "_try_payment_fallback") as fallback:
        kwargs = dict(task="title_generation", messages=[{"role": "user", "content": "hello"}], reasoning_config=config)
        with pytest.raises(openai.APIStatusError):
            if async_mode:
                asyncio.run(aux.async_call_llm(**kwargs))
            else:
                aux.call_llm(**kwargs)
        assert client.chat.completions.create.call_count == 1
        kwargs["reasoning_config"] = {"enabled": True, "effort": "medium"}
        result = asyncio.run(aux.async_call_llm(**kwargs)) if async_mode else aux.call_llm(**kwargs)
        assert result is response
    first, fixed = [call.kwargs for call in client.chat.completions.create.call_args_list]
    assert first["extra_body"]["reasoning"] == (config if enabled else {"enabled": False})
    assert fixed["extra_body"]["reasoning"] == {"enabled": True, "effort": "medium"}
    refresh.assert_not_called()
    fallback.assert_not_called()
    assert not floor._FLOORED_ROUTES


@pytest.mark.parametrize("enabled", [False, True])
def test_main_failure_keeps_settings_and_configured_fallback_unused(tmp_path, monkeypatch, enabled):
    from agent.error_surface import build_error_surface_from_result
    from run_agent import AIAgent
    from tests.fakes.fake_llm_provider import FakeLLMServer, Raw, Text, write_hermes_home

    config = {"enabled": enabled, "effort": "high" if enabled else "none"}
    with FakeLLMServer([Raw(body=json.dumps(BODY), status=403), Text("configured level works")]) as primary, \
         FakeLLMServer([Text("must not fall back")]) as fallback:
        home = write_hermes_home(tmp_path / ".hermes", primary.base_url)
        monkeypatch.setenv("HERMES_HOME", str(home))
        agent = AIAgent(model="test-model", provider="custom", base_url=primary.base_url, api_key="fake-key",
                        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=["memory"],
                        reasoning_config=config, fallback_model={"provider": "custom", "model": "fallback-model",
                        "base_url": fallback.base_url, "api_key": "fake-fallback-key"})
        with patch.object(agent, "_try_refresh_codex_client_credentials") as refresh:
            failed = agent.run_conversation("hello")
            assert failed["failed"] and not failed["completed"]
            assert "reasoning_effort" in failed["final_response"]
            surface = build_error_surface_from_result(failed, provider="openai-codex", model="gpt-6.1-terra")
            assert surface["layer"] != "auth" and not surface["retryable"]
            assert len(primary.main_requests()) == 1
            assert agent.reasoning_config == config
            assert not getattr(agent, "_reasoning_disable_rejected", False)
            assert not getattr(agent, "_reasoning_effort_rejected", False)
            agent.reasoning_config = {"enabled": True, "effort": "medium"}
            fixed = agent.run_conversation("try the configured level")
        assert fixed["final_response"] == "configured level works"
        assert [r["reasoning_effort"] for r in primary.main_requests()] == [config["effort"], "medium"]
        assert fallback.main_requests() == []
        refresh.assert_not_called()
