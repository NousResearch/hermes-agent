import importlib
import sys
import types


def _reset_modules():
    for name in list(sys.modules):
        if name == "cli" or name == "run_agent" or name == "tools" or name.startswith("tools."):
            sys.modules.pop(name, None)


def _import_cli():
    _reset_modules()
    sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
    sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
    sys.modules.setdefault("fal_client", types.SimpleNamespace())
    return importlib.import_module("cli")


def test_get_codex_cli_preferences_reads_reasoning_and_service_tier(tmp_path, monkeypatch):
    from hermes_cli.codex_models import get_codex_cli_preferences

    codex_home = tmp_path / "codex-home"
    codex_home.mkdir(parents=True, exist_ok=True)
    (codex_home / "config.toml").write_text(
        'model = "gpt-5.4"\nmodel_reasoning_effort = "xhigh"\nservice_tier = "fast"\n'
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    prefs = get_codex_cli_preferences()

    assert prefs == {
        "model": "gpt-5.4",
        "reasoning_effort": "xhigh",
        "service_tier": "fast",
    }


def test_model_command_uses_codex_cli_default_when_current_model_empty(monkeypatch):
    from hermes_cli.main import _model_flow_openai_codex

    captured = {}

    monkeypatch.setattr(
        "hermes_cli.auth.get_codex_auth_status",
        lambda: {"logged_in": True},
    )
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_codex_runtime_credentials",
        lambda *args, **kwargs: {"api_key": "codex-access-token"},
    )
    monkeypatch.setattr(
        "hermes_cli.codex_models.get_codex_model_ids",
        lambda access_token=None: ["gpt-5.4", "gpt-5.3-codex"],
    )
    monkeypatch.setattr(
        "hermes_cli.codex_models.get_codex_cli_preferences",
        lambda: {"model": "gpt-5.4"},
    )

    def _fake_prompt_model_selection(model_ids, current_model=""):
        captured["current_model"] = current_model
        return None

    monkeypatch.setattr(
        "hermes_cli.auth._prompt_model_selection",
        _fake_prompt_model_selection,
    )

    _model_flow_openai_codex({}, current_model="")

    assert captured["current_model"] == "gpt-5.4"


def test_cli_inherits_codex_cli_reasoning_and_fast_when_unset(monkeypatch):
    cli_mod = _import_cli()

    monkeypatch.setattr(cli_mod, "get_tool_definitions", lambda **kwargs: [])
    monkeypatch.setattr(
        cli_mod,
        "CLI_CONFIG",
        {
            "model": {
                "default": "",
                "base_url": "",
                "provider": "auto",
            },
            "display": {
                "compact": False,
                "tool_progress": "all",
                "resume_display": "full",
            },
            "agent": {
                "reasoning_effort": "",
                "service_tier": "",
            },
            "terminal": {
                "env_type": "local",
            },
        },
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "test-key",
            "source": "env/config",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.format_runtime_provider_error",
        lambda exc: str(exc),
    )
    monkeypatch.setattr(
        "hermes_cli.codex_models.get_codex_cli_preferences",
        lambda: {"reasoning_effort": "xhigh", "service_tier": "fast"},
    )
    monkeypatch.setattr(
        "hermes_cli.codex_models.get_codex_model_ids",
        lambda access_token=None: ["gpt-5.4"],
    )

    shell = cli_mod.HermesCLI(compact=True, max_turns=1)

    assert shell.reasoning_config is None
    assert shell.service_tier is None
    assert shell._ensure_runtime_credentials() is True
    assert shell.reasoning_config == {"enabled": True, "effort": "xhigh"}
    assert shell.service_tier == "priority"
