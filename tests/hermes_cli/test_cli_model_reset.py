"""Tests for the CLI ``/model reset`` command (NS-563).

In the CLI a session-scoped ``/model`` switch lives directly in
``self.model``/``self.provider`` (no durable pin), so ``/model reset``
re-derives the config.yaml default and switches back to it session-scoped,
never persisting.  Parity with the gateway/TUI ``/model reset``.
"""

from __future__ import annotations

from types import SimpleNamespace

from hermes_cli.model_switch import ModelSwitchResult


class _FakeAgent:
    def __init__(self):
        self.calls = []
        self.model = "deepseek/deepseek-v4"
        self.provider = "openrouter"

    def switch_model(self, **kwargs):
        self.calls.append(kwargs)
        self.model = kwargs["new_model"]
        self.provider = kwargs["new_provider"]


class _StubCLI:
    model = "deepseek/deepseek-v4"
    provider = "openrouter"
    requested_provider = "openrouter"
    api_key = "sk-old"
    _explicit_api_key = "sk-old"
    base_url = "https://openrouter.ai/api/v1"
    _explicit_base_url = "https://openrouter.ai/api/v1"
    api_mode = "chat_completions"
    agent = None
    conversation_history: list = []
    _pending_model_switch_note = None
    _pending_one_turn_model_restore = None

    def _confirm_expensive_model_switch(self, result):
        return True


def _bind_real_methods(stub, cli_mod):
    """Bind the real reset + apply methods onto the stub instance."""
    for name in (
        "_handle_model_reset_command",
        "_apply_model_switch_result",
        "_clear_persisted_context_for_model_switch",
    ):
        method = getattr(cli_mod.HermesCLI, name, None)
        if method is not None:
            setattr(stub, name, method.__get__(stub))


def _patch_common(monkeypatch, cli_mod, printed, *, config_model):
    monkeypatch.setattr(cli_mod, "_cprint", lambda s, *a, **k: printed.append(str(s)))
    monkeypatch.setattr(
        cli_mod,
        "save_config_value",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("reset must never persist config")
        ),
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"model": config_model},
    )
    monkeypatch.setattr(
        "hermes_cli.inventory.load_picker_context",
        lambda: SimpleNamespace(
            user_providers=None,
            custom_providers=None,
            with_overrides=lambda **_: SimpleNamespace(
                user_providers=None, custom_providers=None
            ),
        ),
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.resolve_display_context_length",
        lambda *a, **k: None,
    )


def test_reset_switches_back_to_config_default(monkeypatch):
    import cli as cli_mod

    stub = _StubCLI()
    stub.agent = _FakeAgent()
    _bind_real_methods(stub, cli_mod)
    stub.model = "anthropic/claude-sonnet-4.6"
    stub.provider = "anthropic"
    stub.agent.model = "anthropic/claude-sonnet-4.6"
    stub.agent.provider = "anthropic"
    printed: list = []
    _patch_common(
        monkeypatch,
        cli_mod,
        printed,
        config_model={"default": "deepseek/deepseek-v4", "provider": "openrouter"},
    )
    captured = {}

    def fake_switch_model(**kwargs):
        captured.update(kwargs)
        return ModelSwitchResult(
            success=True,
            new_model="deepseek/deepseek-v4",
            target_provider="openrouter",
            api_key="sk-or",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
            provider_label="OpenRouter",
        )

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", fake_switch_model)

    cli_mod.HermesCLI._handle_model_switch(stub, "/model reset")

    assert stub.model == "deepseek/deepseek-v4"
    assert stub.provider == "openrouter"
    assert captured["raw_input"] == "deepseek/deepseek-v4"
    assert captured["explicit_provider"] == "openrouter"
    assert captured["is_global"] is False
    assert any("cleared" in line for line in printed)


def test_reset_clears_pending_one_turn_restore(monkeypatch):
    import cli as cli_mod

    stub = _StubCLI()
    stub.agent = _FakeAgent()
    _bind_real_methods(stub, cli_mod)
    stub._pending_one_turn_model_restore = {"model": "deepseek/deepseek-v4"}
    printed: list = []
    _patch_common(
        monkeypatch,
        cli_mod,
        printed,
        config_model={"default": "deepseek/deepseek-v4", "provider": "openrouter"},
    )

    cli_mod.HermesCLI._handle_model_switch(stub, "/model reset")

    assert stub._pending_one_turn_model_restore is None


def test_reset_noop_when_already_on_default(monkeypatch):
    import cli as cli_mod

    stub = _StubCLI()
    stub.agent = _FakeAgent()
    _bind_real_methods(stub, cli_mod)
    printed: list = []
    _patch_common(
        monkeypatch,
        cli_mod,
        printed,
        config_model={"default": "deepseek/deepseek-v4", "provider": "openrouter"},
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("no switch needed when already on default")
        ),
    )

    cli_mod.HermesCLI._handle_model_switch(stub, "/model reset")

    assert stub.model == "deepseek/deepseek-v4"
    assert any("No session model override" in line for line in printed)


def test_reset_without_config_default_errors(monkeypatch):
    import cli as cli_mod

    stub = _StubCLI()
    stub.agent = _FakeAgent()
    _bind_real_methods(stub, cli_mod)
    printed: list = []
    _patch_common(monkeypatch, cli_mod, printed, config_model={})

    cli_mod.HermesCLI._handle_model_switch(stub, "/model reset")

    assert any("No model.default configured" in line for line in printed)
    # Nothing mutated.
    assert stub.model == "deepseek/deepseek-v4"


def test_reset_with_flags_is_parse_error(monkeypatch):
    import cli as cli_mod

    stub = _StubCLI()
    stub.agent = _FakeAgent()
    _bind_real_methods(stub, cli_mod)
    printed: list = []
    _patch_common(
        monkeypatch,
        cli_mod,
        printed,
        config_model={"default": "deepseek/deepseek-v4", "provider": "openrouter"},
    )

    cli_mod.HermesCLI._handle_model_switch(stub, "/model reset --global")

    assert any("✗" in line for line in printed)
    assert stub.model == "deepseek/deepseek-v4"
