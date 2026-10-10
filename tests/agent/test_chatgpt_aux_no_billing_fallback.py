"""ChatGPT auxiliary transport failures must not spend another provider's API credit."""

import httpx
import openai
import pytest

from hermes_cli import auth, auth_chatgpt


@pytest.mark.parametrize("failure", [httpx.ConnectError, httpx.ReadTimeout])
def test_chatgpt_auxiliary_network_failure_stays_on_the_selected_plan(monkeypatch, tmp_path, failure):
    from agent import auxiliary_client as aux

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-main-key")
    (tmp_path / "config.yaml").write_text(
        "model:\n  provider: openrouter\n  default: vendor/model\n"
        "auxiliary:\n  transient_retries: 2\n  compression:\n"
        "    provider: openai-chatgpt\n    model: gpt-5.4\n")
    auth._save_auth_store({
        "version": 1, "providers": {auth_chatgpt.PROVIDER: {"active_credential_id": "selected"}},
        "credential_pool": {auth_chatgpt.PROVIDER: [{
            "id": "selected", "label": "Personal", "source": "manual:chatgpt",
            "auth_type": "oauth", "priority": 0, "access_token": "plan-access",
            "refresh_token": "plan-refresh", "expires_at_ms": 4102444800000,
            "chatgpt": {"client_id": "issued-client", "subject": "subject-one",
                        "scopes": [auth_chatgpt.DIRECT_SCOPE]},
        }]},
    })
    requests = []

    def send(request):
        requests.append(request)
        if request.url.host == "api.openai.com":
            raise failure("Temporary fixture transport failure", request=request)
        assert request.url.host == "openrouter.ai"
        return httpx.Response(200, json={
            "id": "fixture", "object": "chat.completion", "model": "vendor/model", "created": 0,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Paid fallback"},
                         "finish_reason": "stop"}],
        })

    monkeypatch.setattr("agent.process_bootstrap.build_keepalive_http_client",
                        lambda *args, **kwargs: httpx.Client(transport=httpx.MockTransport(send)))
    monkeypatch.setattr(aux, "_TRANSIENT_RETRY_BACKOFF_BASE", 0)
    aux._evict_cached_clients("openai-chatgpt")
    aux._evict_cached_clients("openrouter")
    try:
        with pytest.raises(openai.APIConnectionError):
            aux.call_llm(task="compression", messages=[{"role": "user", "content": "Summarize"}])
    finally:
        aux._evict_cached_clients("openai-chatgpt")
        aux._evict_cached_clients("openrouter")
        assert requests
        assert {request.url.host for request in requests} == {"api.openai.com"}
        assert len(requests) <= 3
