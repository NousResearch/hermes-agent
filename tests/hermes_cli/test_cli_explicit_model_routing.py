from __future__ import annotations

from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin


class DummyCLI(CLIAgentSetupMixin):
    def __init__(self, *, explicit_model: bool = False, explicit_provider: bool = False):
        self.api_key = "test-key"
        self.base_url = "https://chatgpt.com/backend-api/codex"
        self.provider = "openai-codex"
        self.api_mode = "codex_responses"
        self.acp_command = None
        self.acp_args = []
        self._credential_pool = None
        self.model = "gpt-5.3-codex-spark"
        self.service_tier = None
        self._explicit_model_override = explicit_model
        self._explicit_provider_override = explicit_provider


def test_explicit_model_override_bypasses_declarative_model_routing(monkeypatch):
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("explicit model override should not be re-routed")

    monkeypatch.setattr("hermes_cli.model_routing.resolve_turn_model_route", fail_if_called)

    route = DummyCLI(explicit_model=True)._resolve_turn_agent_config("What is Codex Spark?")

    assert called is False
    assert route["model"] == "gpt-5.3-codex-spark"
    assert route["runtime"]["provider"] == "openai-codex"
    assert route["routing"] is None


def test_explicit_provider_override_bypasses_declarative_model_routing(monkeypatch):
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("explicit provider override should not be re-routed")

    monkeypatch.setattr("hermes_cli.model_routing.resolve_turn_model_route", fail_if_called)

    route = DummyCLI(explicit_provider=True)._resolve_turn_agent_config("What is Codex Spark?")

    assert called is False
    assert route["model"] == "gpt-5.3-codex-spark"
    assert route["runtime"]["provider"] == "openai-codex"
    assert route["routing"] is None


def test_non_explicit_turn_still_uses_declarative_model_routing(monkeypatch):
    def fake_route(user_message, current_model, current_runtime):
        assert user_message == "normal work"
        assert current_model == "gpt-5.3-codex-spark"
        return (
            "deepseek-v4-flash",
            {**current_runtime, "provider": "opencode-go", "base_url": "https://opencode.ai/zen/go/v1", "api_mode": "chat_completions"},
            {"tier": "fast", "provider": "opencode-go", "model": "deepseek-v4-flash"},
        )

    monkeypatch.setattr("hermes_cli.model_routing.resolve_turn_model_route", fake_route)

    route = DummyCLI()._resolve_turn_agent_config("normal work")

    assert route["model"] == "deepseek-v4-flash"
    assert route["runtime"]["provider"] == "opencode-go"
    assert route["routing"] == {"tier": "fast", "provider": "opencode-go", "model": "deepseek-v4-flash"}
