"""Oneshot (-z) bootstrap must honour fallback_providers on primary AuthError.

Repro (#oneshot-codex-fallback): HERMES_HOME with model.provider=openai-codex,
every Codex credential exhausted/missing, and fallback_providers configured.
``hermes -z "..." --cli`` fails immediately with "No Codex credentials stored"
because ``hermes_cli.oneshot._run_agent`` resolves the provider via
``resolve_runtime_provider()`` BEFORE building the agent, and that bootstrap
call re-raises AuthError without consulting the fallback chain. The interactive
CLI (``_ensure_runtime_credentials``) and the cron scheduler both already fall
through to fallback_providers on the same failure; oneshot must match.
"""

from __future__ import annotations

import pytest

import hermes_cli.oneshot as oneshot
from hermes_cli.auth import AuthError

NVIDIA_RUNTIME = {
    "provider": "custom",
    "api_mode": "chat_completions",
    "base_url": "https://integrate.api.nvidia.com/v1",
    "api_key": "nvapi-test",
    "source": "custom-provider:nvidia",
}

NVIDIA2_RUNTIME = {
    "provider": "custom",
    "api_mode": "chat_completions",
    "base_url": "https://integrate2.api.nvidia.com/v1",
    "api_key": "nvapi-test-2",
    "source": "custom-provider:nvidia2",
}


def _codex_auth_error() -> AuthError:
    return AuthError(
        "No Codex credentials stored. Run `hermes auth` to authenticate.",
        provider="openai-codex",
        code="codex_auth_missing",
        relogin_required=True,
    )


class FakeAgent:
    """Stands in for run_agent.AIAgent; records construction kwargs."""

    instances: list["FakeAgent"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.suppress_status_output = False
        self.stream_delta_callback = None
        self.tool_gen_callback = None
        FakeAgent.instances.append(self)

    def run_conversation(self, prompt):
        return {"final_response": "ok"}


@pytest.fixture
def oneshot_env(monkeypatch):
    """Isolate _run_agent from config files, tools, sessions, and AIAgent."""
    FakeAgent.instances = []
    config = {
        "model": {"provider": "openai-codex", "model": "gpt-5.5-codex"},
        "fallback_providers": [
            {"provider": "custom:nvidia", "model": "moonshotai/kimi-k2.5"},
            {"provider": "custom:nvidia2", "model": "qwen/qwen3.5-coder"},
        ],
    }

    import hermes_cli.config as config_mod
    import hermes_cli.tools_config as tools_config_mod
    import run_agent as run_agent_mod

    monkeypatch.setattr(config_mod, "load_config", lambda *a, **k: config)
    monkeypatch.setattr(tools_config_mod, "_get_platform_tools", lambda *a, **k: set())
    monkeypatch.setattr(run_agent_mod, "AIAgent", FakeAgent)
    monkeypatch.setattr(oneshot, "_create_session_db_for_oneshot", lambda: None)
    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    return config


def _patch_resolver(monkeypatch, runtimes_by_provider):
    """resolve_runtime_provider stub: raise AuthError for the primary (codex)
    request, return a canned runtime for known fallback providers."""
    calls = []

    def fake_resolve(requested=None, **kwargs):
        calls.append(requested)
        if not requested or requested in ("openai-codex", "codex"):
            raise _codex_auth_error()
        runtime = runtimes_by_provider.get(requested)
        if runtime is None:
            raise AuthError(
                f"No credentials for {requested}.", provider=str(requested)
            )
        return dict(runtime)

    import hermes_cli.runtime_provider as rp_mod

    monkeypatch.setattr(rp_mod, "resolve_runtime_provider", fake_resolve)
    return calls


class TestOneshotBootstrapFallback:
    def test_primary_auth_error_falls_back_to_first_configured_provider(
        self, monkeypatch, oneshot_env
    ):
        calls = _patch_resolver(monkeypatch, {"custom:nvidia": NVIDIA_RUNTIME})

        response, result = oneshot._run_agent("hello")

        assert response == "ok"
        assert calls[0] in (None, "openai-codex")
        assert "custom:nvidia" in calls
        agent = FakeAgent.instances[-1]
        assert agent.kwargs["api_key"] == "nvapi-test"
        assert agent.kwargs["base_url"] == "https://integrate.api.nvidia.com/v1"
        assert agent.kwargs["provider"] == "custom"

    def test_fallback_switches_model_to_fallback_entry_model(
        self, monkeypatch, oneshot_env
    ):
        _patch_resolver(monkeypatch, {"custom:nvidia": NVIDIA_RUNTIME})

        oneshot._run_agent("hello")

        agent = FakeAgent.instances[-1]
        # The configured codex model can't run on the fallback provider —
        # the fallback entry's own model must be used.
        assert agent.kwargs["model"] == "moonshotai/kimi-k2.5"

    def test_fallback_chain_iterates_past_failing_entries(
        self, monkeypatch, oneshot_env
    ):
        # First fallback also has no credentials; second resolves.
        calls = _patch_resolver(monkeypatch, {"custom:nvidia2": NVIDIA2_RUNTIME})

        response, _ = oneshot._run_agent("hello")

        assert response == "ok"
        assert calls == [None, "custom:nvidia", "custom:nvidia2"] or calls == [
            "openai-codex",
            "custom:nvidia",
            "custom:nvidia2",
        ]
        agent = FakeAgent.instances[-1]
        assert agent.kwargs["api_key"] == "nvapi-test-2"
        assert agent.kwargs["model"] == "qwen/qwen3.5-coder"

    def test_auth_error_propagates_when_no_fallback_configured(
        self, monkeypatch, oneshot_env
    ):
        oneshot_env["fallback_providers"] = []
        _patch_resolver(monkeypatch, {})

        with pytest.raises(AuthError, match="No Codex credentials stored"):
            oneshot._run_agent("hello")
        assert FakeAgent.instances == []

    def test_original_auth_error_propagates_when_all_fallbacks_fail(
        self, monkeypatch, oneshot_env
    ):
        _patch_resolver(monkeypatch, {})  # every fallback raises AuthError too

        with pytest.raises(AuthError, match="No Codex credentials stored"):
            oneshot._run_agent("hello")
        assert FakeAgent.instances == []

    def test_non_auth_error_is_not_swallowed_by_fallback(
        self, monkeypatch, oneshot_env
    ):
        import hermes_cli.runtime_provider as rp_mod

        def broken_resolve(requested=None, **kwargs):
            raise ValueError("config parse exploded")

        monkeypatch.setattr(rp_mod, "resolve_runtime_provider", broken_resolve)

        with pytest.raises(ValueError, match="config parse exploded"):
            oneshot._run_agent("hello")
        assert FakeAgent.instances == []
