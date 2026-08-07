from __future__ import annotations

from types import SimpleNamespace

from hermes_cli.model_switch import ModelSwitchResult


class _FakeAgent:
    def __init__(self):
        self.calls = []
        self.model = "old/model"
        self.provider = "openrouter"

    def switch_model(self, **kwargs):
        self.calls.append(kwargs)
        self.model = kwargs["new_model"]
        self.provider = kwargs["new_provider"]


class _StubCLI:
    model = "old/model"
    provider = "openrouter"
    requested_provider = "openrouter"
    api_key = "sk-old"
    _explicit_api_key = "sk-old"
    base_url = "https://openrouter.ai/api/v1"
    _explicit_base_url = "https://openrouter.ai/api/v1"
    api_mode = "chat_completions"
    agent = None
    _pending_model_switch_note = None
    _pending_one_turn_model_restore = None

    def _confirm_expensive_model_switch(self, result):
        return True


def test_cli_model_once_records_restore_and_does_not_persist(monkeypatch):
    import cli as cli_mod

    stub = _StubCLI()
    stub.agent = _FakeAgent()
    stub._snapshot_model_runtime = cli_mod.HermesCLI._snapshot_model_runtime.__get__(stub)
    printed = []

    monkeypatch.setattr(cli_mod, "_cprint", lambda s, *a, **k: printed.append(str(s)))
    monkeypatch.setattr(cli_mod, "save_config_value", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not persist")))
    monkeypatch.setattr(
        "hermes_cli.inventory.load_picker_context",
        lambda: SimpleNamespace(
            user_providers=None,
            custom_providers=None,
            with_overrides=lambda **_: SimpleNamespace(user_providers=None, custom_providers=None),
        ),
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **_: ModelSwitchResult(
            success=True,
            new_model="claude-sonnet-4.6",
            target_provider="anthropic",
            api_key="sk-ant",
            base_url="https://api.anthropic.com",
            api_mode="anthropic_messages",
            provider_label="Anthropic",
        ),
    )
    monkeypatch.setattr("hermes_cli.model_switch.resolve_display_context_length", lambda *a, **k: None)

    cli_mod.HermesCLI._handle_model_switch(
        stub,
        "/model claude-sonnet-4.6 --provider anthropic --once",
    )

    assert stub.model == "claude-sonnet-4.6"
    assert stub.provider == "anthropic"
    assert stub.agent.calls[-1]["new_model"] == "claude-sonnet-4.6"
    assert stub._pending_one_turn_model_restore["model"] == "old/model"
    assert "next turn only" in printed[-1]


def test_cli_restore_model_runtime_snapshot_restores_agent():
    import cli as cli_mod

    stub = _StubCLI()
    stub.agent = _FakeAgent()
    snapshot = {
        "model": "old/model",
        "provider": "openrouter",
        "requested_provider": "openrouter",
        "api_key": "sk-old",
        "explicit_api_key": "sk-old",
        "base_url": "https://openrouter.ai/api/v1",
        "explicit_base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions",
    }

    cli_mod.HermesCLI._restore_model_runtime_snapshot(stub, snapshot)

    assert stub.model == "old/model"
    assert stub.provider == "openrouter"
    assert stub.agent.calls[-1]["new_model"] == "old/model"


def test_cli_restore_model_runtime_prefers_primary_runtime():
    import cli as cli_mod

    class Agent(_FakeAgent):
        _primary_runtime = None
        _rate_limited_until = 123

        def __init__(self):
            super().__init__()
            self.model = "temp/model"
            self.provider = "anthropic"

        def _restore_primary_runtime(self):
            self.model = self._primary_runtime["model"]
            self.provider = self._primary_runtime["provider"]
            return True

    stub = _StubCLI()
    stub.agent = Agent()
    snapshot = {
        "model": "old/model",
        "provider": "openrouter",
        "requested_provider": "openrouter",
        "api_key": "sk-old",
        "explicit_api_key": "sk-old",
        "base_url": "",
        "explicit_base_url": "",
        "api_mode": "chat_completions",
        "agent_primary_runtime": {
            "model": "old/model",
            "provider": "openrouter",
        },
    }

    cli_mod.HermesCLI._restore_model_runtime_snapshot(stub, snapshot)

    assert stub.agent.model == "old/model"
    assert stub.agent.provider == "openrouter"
    assert stub.agent.calls == []
