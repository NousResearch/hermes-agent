from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from hermes_cli import runtime_provider as rp
from hermes_cli.auth import AuthError
from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin


class _NoopDb:
    pass


class _RuntimeCLI(CLIAgentSetupMixin):
    def __init__(self) -> None:
        self.model = "deepseek-v4-flash"
        self.requested_provider = "opencode-go"
        self.provider = "opencode-go"
        self.api_key = None
        self.base_url = None
        self.api_mode = "chat_completions"
        self.acp_command = None
        self.acp_args = []
        self._credential_pool = None
        self._provider_source = None
        self._explicit_api_key = None
        self._explicit_base_url = None
        self._fallback_model = []
        self.agent = None
        self._active_agent_route_signature = None
        self.service_tier = None

    def _normalize_model_for_provider(self, _provider: str) -> bool:
        return False


def _runtime() -> dict:
    return {
        "provider": "opencode-go",
        "api_mode": "chat_completions",
        "base_url": "https://opencode.ai/zen/go/v1",
        "api_key": "test-key",
        "source": "test",
    }


def test_cli_primary_resolution_uses_effective_model(monkeypatch):
    calls: list[dict] = []

    def fake_resolver(**kwargs):
        calls.append(kwargs)
        return _runtime()

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolver)

    cli = _RuntimeCLI()
    assert cli._ensure_runtime_credentials() is True
    assert calls[0]["target_model"] == "deepseek-v4-flash"


def test_cli_fallback_resolution_uses_fallback_model(monkeypatch):
    calls: list[dict] = []

    def fake_resolver(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise AuthError("primary unavailable", provider="opencode-go")
        return _runtime()

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolver)

    cli = _RuntimeCLI()
    cli.model = "qwen3.7-plus"
    cli._fallback_model = [
        {"provider": "opencode-go", "model": "deepseek-v4-flash"}
    ]
    assert cli._ensure_runtime_credentials() is True
    assert calls[0]["target_model"] == "qwen3.7-plus"
    assert calls[1]["target_model"] == "deepseek-v4-flash"


def test_acp_resolution_uses_effective_model(monkeypatch):
    calls: list[dict] = []
    captured: dict = {}

    class CapturingAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    def resolver(**kwargs):
        calls.append(kwargs)
        return _runtime()

    def module(name: str, **attrs):
        result = ModuleType(name)
        for key, value in attrs.items():
            setattr(result, key, value)
        return result

    monkeypatch.setitem(sys.modules, "run_agent", module("run_agent", AIAgent=CapturingAgent))
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        module(
            "hermes_cli.config",
            load_config=lambda: {
                "model": {"default": "qwen3.7-plus", "provider": "opencode-go"}
            },
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        module("hermes_cli.runtime_provider", resolve_runtime_provider=resolver),
    )

    manager = SessionManager(db=_NoopDb())
    manager._make_agent(
        session_id="routing-test",
        cwd=".",
        model="deepseek-v4-flash",
        requested_provider="opencode-go",
    )

    assert calls[0]["target_model"] == "deepseek-v4-flash"
    assert captured["model"] == "deepseek-v4-flash"
    assert captured["api_mode"] == "chat_completions"
    assert captured["base_url"] == "https://opencode.ai/zen/go/v1"


def test_opencode_go_effective_model_override_drives_real_runtime_mode(monkeypatch):
    monkeypatch.setattr(rp, "resolve_provider", lambda *args, **kwargs: "opencode-go")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "opencode-go",
            "default": "qwen3.7-plus",
            "api_mode": "anthropic_messages",
        },
    )
    monkeypatch.setattr(rp, "load_pool", lambda _provider: None)
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-opencode-go-key")
    monkeypatch.delenv("OPENCODE_GO_BASE_URL", raising=False)

    resolved = rp.resolve_runtime_provider(
        requested="opencode-go",
        target_model="deepseek-v4-flash",
    )

    assert resolved["provider"] == "opencode-go"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["base_url"] == "https://opencode.ai/zen/go/v1"


@pytest.mark.parametrize(
    "provider",
    ["azure-foundry", "bedrock", "copilot", "nous", "opencode-go", "opencode-zen"],
)
@pytest.mark.asyncio
async def test_acp_model_switch_recomputes_model_dependent_transport(
    monkeypatch, provider
):
    captured: dict = {}
    old_agent = SimpleNamespace(
        provider=provider,
        base_url="https://opencode.ai/zen/go",
        api_mode="anthropic_messages",
    )
    state = SimpleNamespace(agent=old_agent, cwd=".", model="qwen3.7-plus")

    class Manager:
        def get_session(self, _session_id):
            return state

        def _make_agent(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(provider=provider)

        def save_session(self, _session_id):
            return None

    acp_agent = HermesACPAgent(session_manager=Manager())
    monkeypatch.setattr(
        acp_agent,
        "_resolve_model_selection",
        lambda _model_id, _provider: (provider, "deepseek-v4-flash"),
    )

    await acp_agent.set_session_model("deepseek-v4-flash", "routing-session")

    assert captured["model"] == "deepseek-v4-flash"
    assert "base_url" not in captured
    assert "api_mode" not in captured
