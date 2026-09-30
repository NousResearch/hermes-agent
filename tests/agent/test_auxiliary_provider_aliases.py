import pytest
from types import SimpleNamespace

@pytest.mark.parametrize("provider", ["nous", "nous-portal", "nousresearch"])
def test_all_nous_aliases_require_authorization_before_callback(monkeypatch, tmp_path, provider):
    import agent.auxiliary_client as module
    client = SimpleNamespace(base_url="https://inference-api.nousresearch.com/v1")
    monkeypatch.setattr(module, "get_hermes_home", lambda: tmp_path)
    requests = []
    def refuse(agent, kwargs, callback, *, route):
        assert route.provider == provider
        raise RuntimeError("synthetic authorization refusal")
    monkeypatch.setattr("agent.llm_egress_runtime.dispatch_authorized_agent_request", refuse)
    with pytest.raises(RuntimeError, match="synthetic authorization refusal"):
        module._authorize_auxiliary_request(
            client, {"model": "synthetic-model", "messages": []},
            lambda request: requests.append(request), provider=provider,
            api_mode="chat_completions", metadata=None,
        )
    assert requests == []
