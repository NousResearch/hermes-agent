import importlib
import sys
import types
from contextlib import nullcontext
from types import SimpleNamespace

from hermes_cli.auth import AuthError
from hermes_cli import main as hermes_main


def _install_prompt_toolkit_stubs():
    class _Dummy:
        def __init__(self, *args, **kwargs):
            pass

    class _Condition:
        def __init__(self, func):
            self.func = func

        def __bool__(self):
            return bool(self.func())

    class _ANSI(str):
        pass

    root = types.ModuleType("prompt_toolkit")
    history = types.ModuleType("prompt_toolkit.history")
    styles = types.ModuleType("prompt_toolkit.styles")
    patch_stdout = types.ModuleType("prompt_toolkit.patch_stdout")
    application = types.ModuleType("prompt_toolkit.application")
    layout = types.ModuleType("prompt_toolkit.layout")
    processors = types.ModuleType("prompt_toolkit.layout.processors")
    filters = types.ModuleType("prompt_toolkit.filters")
    dimension = types.ModuleType("prompt_toolkit.layout.dimension")
    menus = types.ModuleType("prompt_toolkit.layout.menus")
    widgets = types.ModuleType("prompt_toolkit.widgets")
    key_binding = types.ModuleType("prompt_toolkit.key_binding")
    completion = types.ModuleType("prompt_toolkit.completion")
    formatted_text = types.ModuleType("prompt_toolkit.formatted_text")

    history.FileHistory = _Dummy
    styles.Style = _Dummy
    patch_stdout.patch_stdout = lambda *args, **kwargs: nullcontext()
    application.Application = _Dummy
    layout.Layout = _Dummy
    layout.HSplit = _Dummy
    layout.Window = _Dummy
    layout.FormattedTextControl = _Dummy
    layout.ConditionalContainer = _Dummy
    processors.Processor = _Dummy
    processors.Transformation = _Dummy
    processors.PasswordProcessor = _Dummy
    processors.ConditionalProcessor = _Dummy
    filters.Condition = _Condition
    dimension.Dimension = _Dummy
    menus.CompletionsMenu = _Dummy
    widgets.TextArea = _Dummy
    key_binding.KeyBindings = _Dummy
    completion.Completer = _Dummy
    completion.Completion = _Dummy
    formatted_text.ANSI = _ANSI
    root.print_formatted_text = lambda *args, **kwargs: None

    sys.modules.setdefault("prompt_toolkit", root)
    sys.modules.setdefault("prompt_toolkit.history", history)
    sys.modules.setdefault("prompt_toolkit.styles", styles)
    sys.modules.setdefault("prompt_toolkit.patch_stdout", patch_stdout)
    sys.modules.setdefault("prompt_toolkit.application", application)
    sys.modules.setdefault("prompt_toolkit.layout", layout)
    sys.modules.setdefault("prompt_toolkit.layout.processors", processors)
    sys.modules.setdefault("prompt_toolkit.filters", filters)
    sys.modules.setdefault("prompt_toolkit.layout.dimension", dimension)
    sys.modules.setdefault("prompt_toolkit.layout.menus", menus)
    sys.modules.setdefault("prompt_toolkit.widgets", widgets)
    sys.modules.setdefault("prompt_toolkit.key_binding", key_binding)
    sys.modules.setdefault("prompt_toolkit.completion", completion)
    sys.modules.setdefault("prompt_toolkit.formatted_text", formatted_text)


def _import_cli():
    try:
        importlib.import_module("prompt_toolkit")
    except ModuleNotFoundError:
        _install_prompt_toolkit_stubs()
    return importlib.import_module("cli")


def test_hermes_cli_init_does_not_eagerly_resolve_runtime_provider(monkeypatch):
    cli = _import_cli()
    calls = {"count": 0}

    def _unexpected_runtime_resolve(**kwargs):
        calls["count"] += 1
        raise AssertionError("resolve_runtime_provider should not be called in HermesCLI.__init__")

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", _unexpected_runtime_resolve)
    monkeypatch.setattr("hermes_cli.runtime_provider.format_runtime_provider_error", lambda exc: str(exc))

    shell = cli.HermesCLI(model="gpt-5", compact=True, max_turns=1)

    assert shell is not None
    assert calls["count"] == 0


def test_runtime_resolution_failure_is_not_sticky(monkeypatch):
    cli = _import_cli()
    calls = {"count": 0}

    def _runtime_resolve(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary auth failure")
        return {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "test-key",
            "source": "env/config",
        }

    class _DummyAgent:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", _runtime_resolve)
    monkeypatch.setattr("hermes_cli.runtime_provider.format_runtime_provider_error", lambda exc: str(exc))
    monkeypatch.setattr(cli, "AIAgent", _DummyAgent)

    shell = cli.HermesCLI(model="gpt-5", compact=True, max_turns=1)

    assert shell._init_agent() is False
    assert shell._init_agent() is True
    assert calls["count"] == 2
    assert shell.agent is not None


def test_runtime_resolution_rebuilds_agent_on_routing_change(monkeypatch):
    cli = _import_cli()

    def _runtime_resolve(**kwargs):
        return {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "base_url": "https://same-endpoint.example/v1",
            "api_key": "same-key",
            "source": "env/config",
        }

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", _runtime_resolve)
    monkeypatch.setattr("hermes_cli.runtime_provider.format_runtime_provider_error", lambda exc: str(exc))

    shell = cli.HermesCLI(model="gpt-5", compact=True, max_turns=1)
    shell.provider = "openrouter"
    shell.api_mode = "chat_completions"
    shell.base_url = "https://same-endpoint.example/v1"
    shell.api_key = "same-key"
    shell.agent = object()

    assert shell._ensure_runtime_credentials() is True
    assert shell.agent is None
    assert shell.provider == "openai-codex"
    assert shell.api_mode == "codex_responses"


def test_cmd_model_falls_back_to_auto_on_invalid_provider(monkeypatch, capsys):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"default": "gpt-5", "provider": "invalid-provider"}},
    )
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: None)
    monkeypatch.setattr("hermes_cli.config.get_env_value", lambda key: "")
    monkeypatch.setattr("hermes_cli.config.save_env_value", lambda key, value: None)

    def _resolve_provider(requested, **kwargs):
        if requested == "invalid-provider":
            raise AuthError("Unknown provider 'invalid-provider'.", code="invalid_provider")
        return "openrouter"

    monkeypatch.setattr("hermes_cli.auth.resolve_provider", _resolve_provider)
    monkeypatch.setattr(hermes_main, "_prompt_provider_choice", lambda choices: len(choices) - 1)

    hermes_main.cmd_model(SimpleNamespace())
    output = capsys.readouterr().out

    assert "Warning:" in output
    assert "falling back to auto provider detection" in output.lower()
    assert "No change." in output
