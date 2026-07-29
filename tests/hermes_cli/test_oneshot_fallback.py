from hermes_cli import oneshot
from hermes_cli.auth import AuthError


def test_run_agent_uses_configured_fallback_when_primary_credentials_are_exhausted(monkeypatch):
    config = {
        "model": {"default": "k3", "provider": "kimi-coding"},
        "fallback_providers": [
            {
                "provider": "openai-codex",
                "model": "gpt-5.6-sol",
                "base_url": "https://chatgpt.com/backend-api/codex",
            }
        ],
    }
    calls = []

    def fake_resolve_runtime_provider(**kwargs):
        calls.append(kwargs)
        if kwargs.get("requested") in {None, "kimi-coding"}:
            raise AuthError("No usable credentials found for provider 'kimi-coding'.")
        return {
            "api_key": "fallback-token",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "credential_pool": None,
        }

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.suppress_status_output = False
            self.stream_delta_callback = object()
            self.tool_gen_callback = object()

        def run_conversation(self, prompt):
            assert prompt == "canary"
            assert self.kwargs["provider"] == "openai-codex"
            assert self.kwargs["model"] == "gpt-5.6-sol"
            return {"final_response": "ok"}

        def close(self):
            pass

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        fake_resolve_runtime_provider,
    )
    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    monkeypatch.setattr(oneshot, "_create_session_db_for_oneshot", lambda: None)

    response, result = oneshot._run_agent("canary", toolsets="safe")

    assert response == "ok"
    assert result["final_response"] == "ok"
    assert [call.get("requested") for call in calls] == [None, "openai-codex"]
