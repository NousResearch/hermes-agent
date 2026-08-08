from types import SimpleNamespace

import cli as cli_mod
from cli import HermesCLI
from hermes_cli import subagent_model
from hermes_cli.commands import resolve_command
from hermes_cli.model_switch import ModelSwitchResult


def _stub_cli():
    obj = HermesCLI.__new__(HermesCLI)
    obj.model = "primary-model"
    obj.provider = "primary-provider"
    obj.base_url = "https://primary.example/v1"
    obj.api_key = "primary-key"
    obj._app = None
    return obj


def test_subagent_command_is_classic_cli_only():
    command = resolve_command("/subagent")
    assert command is not None
    assert command.cli_only is True
    assert command.gateway_only is False


def test_classic_subagent_direct_model_uses_shared_core(monkeypatch):
    calls = []
    output = []
    expected = subagent_model.SubagentModelStatus(
        model="canonical-model",
        provider="target-provider",
        inherits_parent=False,
    )
    monkeypatch.setattr(
        subagent_model,
        "set_subagent_model",
        lambda model, provider=None: calls.append((model, provider)) or expected,
    )
    monkeypatch.setattr(cli_mod, "_cprint", output.append)

    cli = _stub_cli()
    cli._handle_subagent_command("/subagent model alias --provider target-provider")

    assert calls == [("alias", "target-provider")]
    assert cli.model == "primary-model"
    assert cli.provider == "primary-provider"
    assert any("canonical-model" in line for line in output)


def test_classic_provider_only_status_names_provider_default(monkeypatch):
    output = []
    monkeypatch.setattr(
        subagent_model,
        "get_subagent_model_status",
        lambda: subagent_model.SubagentModelStatus(None, "openrouter", False),
    )
    monkeypatch.setattr(
        subagent_model,
        "get_subagent_reasoning_status",
        lambda: subagent_model.SubagentReasoningStatus(None, True),
    )
    monkeypatch.setattr(cli_mod, "_cprint", output.append)

    _stub_cli()._handle_subagent_command("/subagent")

    assert any("provider default (openrouter)" in line for line in output)
    assert all("None" not in line for line in output)


def test_classic_subagent_reasoning_set_and_reset_are_independent(monkeypatch):
    calls = []
    output = []
    explicit = subagent_model.SubagentReasoningStatus(
        effort="high",
        inherits_parent=False,
    )
    inherited = subagent_model.SubagentReasoningStatus(
        effort=None,
        inherits_parent=True,
    )
    monkeypatch.setattr(
        subagent_model,
        "set_subagent_reasoning_effort",
        lambda effort: calls.append(("set", effort)) or explicit,
    )
    monkeypatch.setattr(
        subagent_model,
        "reset_subagent_reasoning_effort",
        lambda: calls.append(("reset", None)) or inherited,
    )
    monkeypatch.setattr(cli_mod, "_cprint", output.append)

    cli = _stub_cli()
    cli._handle_subagent_command("/subagent reasoning high")
    cli._handle_subagent_command("/subagent reasoning reset")

    assert calls == [("set", "high"), ("reset", None)]
    assert any("reasoning: high" in line for line in output)
    assert any("inherits parent" in line for line in output)


def test_classic_subagent_model_opens_existing_picker_for_subagent_target(monkeypatch):
    context = SimpleNamespace(
        current_model="parent-model",
        current_provider="parent-provider",
        user_providers={"target-provider": {}},
        custom_providers=[],
    )
    status = subagent_model.SubagentModelStatus(
        model="child-model",
        provider="target-provider",
        inherits_parent=False,
    )
    providers = [
        {
            "slug": "target-provider",
            "name": "Target Provider",
            "models": ["child-model", "other-model"],
            "total_models": 2,
            "is_current": False,
        }
    ]
    monkeypatch.setattr(
        "hermes_cli.inventory.load_picker_context",
        lambda: context,
    )
    monkeypatch.setattr(
        subagent_model,
        "get_subagent_model_status",
        lambda: status,
    )
    monkeypatch.setattr(
        subagent_model,
        "list_subagent_picker_providers",
        lambda refresh=False: providers,
    )

    captured = {}
    cli = _stub_cli()
    cli._open_model_picker = lambda *args, **kwargs: captured.update(
        args=args,
        kwargs=kwargs,
    )
    cli._handle_subagent_command("/subagent model")

    assert captured["args"][1:] == ("child-model", "target-provider")
    assert captured["kwargs"]["target"] == "subagent"
    assert captured["args"][0][0]["is_current"] is True
    assert cli.model == "primary-model"
    assert cli.provider == "primary-provider"


def test_classic_provider_only_opens_picker_on_requested_provider(monkeypatch):
    context = SimpleNamespace(
        current_model="parent-model",
        current_provider="parent-provider",
        user_providers={"requested-provider": {}},
        custom_providers=[],
    )
    status = subagent_model.SubagentModelStatus(
        model="old-child-model",
        provider="old-provider",
        inherits_parent=False,
    )
    providers = [
        {"slug": "old-provider", "models": ["old-child-model"]},
        {"slug": "requested-provider", "models": ["new-child-model"]},
    ]
    monkeypatch.setattr("hermes_cli.inventory.load_picker_context", lambda: context)
    monkeypatch.setattr(subagent_model, "get_subagent_model_status", lambda: status)
    monkeypatch.setattr(
        subagent_model, "list_subagent_picker_providers", lambda refresh=False: providers
    )

    captured = {}
    cli = _stub_cli()
    cli._open_model_picker = lambda *args, **kwargs: captured.update(
        args=args, kwargs=kwargs
    )
    cli._handle_subagent_command("/subagent model --provider requested-provider")

    assert captured["args"][2] == "requested-provider"
    assert captured["args"][1] == "unknown"
    assert captured["args"][0][1]["is_current"] is True
    assert captured["kwargs"]["target"] == "subagent"


def test_subagent_picker_opens_on_configured_delegation_model():
    cli = _stub_cli()
    cli._model_picker_state = {
        "stage": "provider",
        "selected": 0,
        "providers": [
            {
                "slug": "target-provider",
                "models": ["first-model", "child-model", "third-model"],
            }
        ],
        "current_model": "child-model",
        "current_provider": "target-provider",
        "target": "subagent",
    }
    cli._invalidate = lambda min_interval=0.25: None

    cli._handle_model_picker_selection()

    assert cli._model_picker_state["stage"] == "model"
    assert cli._model_picker_state["selected"] == 1


def test_model_picker_routes_subagent_target_without_main_switch(monkeypatch):
    result = ModelSwitchResult(
        success=True,
        new_model="child-model",
        target_provider="target-provider",
    )
    switch_kwargs = {}

    def fake_switch_model(**kwargs):
        switch_kwargs.update(kwargs)
        return result

    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        fake_switch_model,
    )

    routed = []
    cli = _stub_cli()
    cli._model_picker_state = {
        "stage": "model",
        "selected": 0,
        "provider_data": {"slug": "target-provider"},
        "model_list": ["child-model"],
        "current_model": "old-child-model",
        "current_provider": "target-provider",
        "custom_provs": [],
        "user_provs": {},
        "target": "subagent",
    }
    cli._close_model_picker = lambda: setattr(cli, "_model_picker_state", None)
    cli._confirm_and_apply_subagent_model_result = routed.append
    cli._confirm_and_apply_model_switch_result = lambda *_args: (_ for _ in ()).throw(
        AssertionError("subagent selection must not apply to the primary runtime")
    )

    cli._handle_model_picker_selection(persist_global=True)

    assert routed == [result]
    assert switch_kwargs["is_global"] is False
    assert cli.model == "primary-model"
    assert cli.provider == "primary-provider"


def test_model_picker_main_target_preserves_runtime_route(monkeypatch):
    result = ModelSwitchResult(
        success=True,
        new_model="next-primary-model",
        target_provider="primary-provider",
    )
    captured = {}

    def fake_switch_model(**kwargs):
        captured.update(kwargs)
        return result

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", fake_switch_model)

    routed = []
    cli = _stub_cli()
    cli._model_picker_state = {
        "stage": "model",
        "selected": 0,
        "provider_data": {"slug": "primary-provider"},
        "model_list": ["next-primary-model"],
        # The primary picker stores a human-facing label here. It must not be
        # passed back as the canonical current provider.
        "current_model": "primary-model",
        "current_provider": "Primary Provider",
        "custom_provs": [],
        "user_provs": {},
        "target": "main",
    }
    cli._close_model_picker = lambda: setattr(cli, "_model_picker_state", None)
    monkeypatch.setattr(
        cli,
        "_confirm_and_apply_model_switch_result",
        lambda value, persist, custom_providers=None: routed.append(
            (value, persist, custom_providers)
        ),
    )

    cli._handle_model_picker_selection(persist_global=True)

    assert captured["current_provider"] == "primary-provider"
    assert captured["current_model"] == "primary-model"
    assert routed == [(result, True, [])]


def test_subagent_picker_persists_override_without_mutating_primary(monkeypatch):
    result = ModelSwitchResult(
        success=True,
        new_model="child-model",
        target_provider="target-provider",
    )
    expected = subagent_model.SubagentModelStatus(
        model="child-model",
        provider="target-provider",
        inherits_parent=False,
    )
    calls = []
    output = []
    monkeypatch.setattr(
        subagent_model,
        "persist_subagent_switch_result",
        lambda value: calls.append(value) or expected,
    )
    monkeypatch.setattr(cli_mod, "_cprint", output.append)

    cli = _stub_cli()
    cli._confirm_expensive_model_switch = lambda _result, **_kwargs: True
    before = (cli.model, cli.provider, cli.base_url, cli.api_key)
    cli._confirm_and_apply_subagent_model_result(result)

    assert calls == [result]
    assert (cli.model, cli.provider, cli.base_url, cli.api_key) == before
    assert any("delegation.model/provider" in line for line in output)
