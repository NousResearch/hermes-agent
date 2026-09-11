"""End-to-end success evidence stays bound to the pool entry dispatched for the turn."""

from __future__ import annotations

from types import SimpleNamespace

from agent.credential_pool import AUTH_TYPE_OAUTH, CredentialPool, PooledCredential


class _Completions:
    def __init__(self):
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="synthetic success", reasoning=None, tool_calls=[]),
                finish_reason="stop",
            )],
            usage=None,
        )


class _Client:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_Completions())


def test_successful_turn_emits_the_real_synthetic_pool_entry_id(monkeypatch):
    """Exercise construction, dispatch, response intake, and hook delivery without auth I/O."""
    from run_agent import AIAgent

    entry = PooledCredential.from_dict("openai-codex", {
        "id": "synthetic-entry-a",
        "label": "synthetic",
        "provider": "openai-codex",
        "auth_type": AUTH_TYPE_OAUTH,
        "access_token": "synthetic-token-a",
        "refresh_token": "synthetic-refresh-a",
        "base_url": "https://chatgpt.com/backend-api/codex",
    })
    pool = CredentialPool(provider="openai-codex", entries=[entry])
    client = _Client()
    observed: list[dict] = []

    monkeypatch.setattr("agent.process_bootstrap.OpenAI", lambda **_kwargs: client)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("hermes_cli.auth.read_credential_pool", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("auth store read")))
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: name == "runtime_request_succeeded")

    def _capture_success(name, **kwargs):
        if name == "runtime_request_succeeded":
            observed.append({"name": name, **kwargs})
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", _capture_success)

    agent = AIAgent(
        model="gpt-5",
        provider="openai-codex",
        api_mode="chat_completions",
        api_key="synthetic-token-a",
        base_url="https://chatgpt.com/backend-api/codex",
        credential_pool=pool,
        platform="cli",
        max_iterations=1,
        quiet_mode=True,
        skip_memory=True,
    )

    result = agent.run_conversation("prove synthetic binding")

    assert result["final_response"].startswith("synthetic success")
    assert client.chat.completions.calls
    assert agent._credential_pool_entry_id == "synthetic-entry-a"
    assert len(observed) == 1
    assert observed[0]["name"] == "runtime_request_succeeded"
    assert observed[0]["credential_pool_entry_id"] == "synthetic-entry-a"
    assert observed[0]["provider"] == "openai-codex"
    assert observed[0]["model"] == "gpt-5"
    assert "token" not in observed[0]
    assert "refresh" not in observed[0]
