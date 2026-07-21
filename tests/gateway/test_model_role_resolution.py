from gateway.run import GatewayRunner


def test_gateway_reuses_logical_role_runtime_tuple_for_session(monkeypatch):
    calls = []

    def resolve(model):
        calls.append(model)
        return "gpt-5.6-terra", {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "api_key": "provider-key",
            "base_url": "http://provider.example/v1",
        }

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_logical_model_runtime", resolve)
    runner = object.__new__(GatewayRunner)

    first_model, first_runtime = runner._resolve_session_logical_model_role(
        "session-1", "role:bob-main", {"provider": "stale", "api_mode": "stale"}
    )
    second_model, second_runtime = runner._resolve_session_logical_model_role(
        "session-1", "role:bob-main", {"provider": "other", "api_mode": "other"}
    )

    assert calls == ["role:bob-main"]
    assert (first_model, first_runtime["provider"], first_runtime["api_mode"]) == (
        "gpt-5.6-terra", "openai-codex", "codex_responses"
    )
    assert (second_model, second_runtime["provider"], second_runtime["api_mode"]) == (
        "gpt-5.6-terra", "openai-codex", "codex_responses"
    )
