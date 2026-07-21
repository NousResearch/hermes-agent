"""Regression tests for stale direct-Codex session overrides.

When Hermes is configured to use a loopback OpenAI-compatible proxy, a
persisted ``/model`` override that still points at the direct ChatGPT Codex
endpoint must not resurrect that route after a gateway restart.
"""

from gateway.run import GatewayRunner


class _Store:
    def __init__(self, override):
        self.override = override
        self.cleared = []

    def get_model_override(self, _session_key):
        return self.override

    def set_model_override(self, session_key, override):
        self.cleared.append((session_key, override))


def test_rehydrate_clears_stale_direct_codex_override_for_local_proxy(
    monkeypatch,
):
    """A local-proxy primary must win over a persisted direct-Codex route."""

    session_key = "agent:main:telegram:dm:test-user"
    store = _Store(
        {
            "model": "gpt-5.6-sol",
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
        }
    )
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {}
    runner.session_store = store

    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:52533/v1")
    monkeypatch.setattr(
        "hermes_cli.runtime_provider._get_model_config",
        lambda: {"default": "gpt-5.6-sol", "provider": "openai-api"},
    )
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        lambda _provider: {
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "test-only",
            "api_mode": "codex_responses",
        },
    )

    runner._rehydrate_session_model_override(session_key)

    assert session_key not in runner._session_model_overrides
    assert store.cleared == [(session_key, None)]


def test_rehydrate_keeps_direct_codex_override_without_local_proxy(monkeypatch):
    """A direct-Codex choice remains valid when no local proxy is configured."""

    session_key = "agent:main:telegram:dm:test-user"
    store = _Store(
        {
            "model": "gpt-5.6-sol",
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
        }
    )
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {}
    runner.session_store = store

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider._get_model_config",
        lambda: {"default": "gpt-5.6-sol", "provider": "openai-codex"},
    )
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        lambda _provider: {
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "test-only",
            "api_mode": "codex_responses",
        },
    )

    runner._rehydrate_session_model_override(session_key)

    assert runner._session_model_overrides[session_key]["provider"] == "openai-codex"
    assert store.cleared == []


def test_rehydrate_keeps_non_codex_override_with_local_proxy(monkeypatch):
    """The local-proxy guard must not erase unrelated intentional overrides."""

    session_key = "agent:main:telegram:dm:test-user"
    store = _Store(
        {
            "model": "claude-sonnet-4.6",
            "provider": "anthropic",
            "base_url": "https://api.anthropic.com",
        }
    )
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {}
    runner.session_store = store

    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:52533/v1")
    monkeypatch.setattr(
        "hermes_cli.runtime_provider._get_model_config",
        lambda: {"default": "gpt-5.6-sol", "provider": "openai-api"},
    )
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        lambda _provider: {
            "provider": "anthropic",
            "base_url": "https://api.anthropic.com",
            "api_key": "test-only",
            "api_mode": "anthropic_messages",
        },
    )

    runner._rehydrate_session_model_override(session_key)

    assert runner._session_model_overrides[session_key]["provider"] == "anthropic"
    assert store.cleared == []
