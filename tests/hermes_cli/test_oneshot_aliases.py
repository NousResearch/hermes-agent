import sys
from types import SimpleNamespace

from hermes_cli.model_switch import DirectAlias


class _DummyAgent:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    def chat(self, prompt):
        return f"ok:{prompt}"


def test_oneshot_direct_alias_with_explicit_provider(monkeypatch):
    """`hermes --provider X -m alias -z` should send the resolved model id."""
    import hermes_cli.model_switch as ms
    from hermes_cli import oneshot

    captured_runtime = {}
    monkeypatch.setattr(
        ms,
        "DIRECT_ALIASES",
        {
            "kimi26-cloud": DirectAlias(
                "kimi-k2.6:cloud",
                "ollama-cloud",
                "https://ollama.com/v1",
            )
        },
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"provider": "ollama-cloud", "default": "kimi-k2.6:cloud"}},
    )
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda cfg, platform: set())

    def _runtime(**kwargs):
        captured_runtime.update(kwargs)
        return {
            "api_key": "test-key",
            "base_url": "https://ollama.com/v1",
            "provider": "ollama-cloud",
            "api_mode": "chat_completions",
        }

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", _runtime)
    monkeypatch.setitem(sys.modules, "run_agent", SimpleNamespace(AIAgent=_DummyAgent))

    assert oneshot._run_agent("ping", model="kimi26-cloud", provider="ollama-cloud") == "ok:ping"
    assert captured_runtime["requested"] == "ollama-cloud"
    assert captured_runtime["target_model"] == "kimi-k2.6:cloud"
    assert captured_runtime["explicit_base_url"] == "https://ollama.com/v1"
    assert _DummyAgent.last_kwargs["model"] == "kimi-k2.6:cloud"


def test_oneshot_direct_alias_without_provider_uses_alias_route(monkeypatch):
    """Bare `hermes -m alias -z` should route through the alias provider."""
    import hermes_cli.model_switch as ms
    from hermes_cli import oneshot

    captured_runtime = {}
    monkeypatch.setattr(
        ms,
        "DIRECT_ALIASES",
        {
            "qwen35-cloud": DirectAlias(
                "qwen3.5:397b-cloud",
                "ollama-cloud",
                "https://ollama.com/v1",
            )
        },
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"provider": "openrouter", "default": "openai/gpt-5"}},
    )
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda cfg, platform: set())

    def _runtime(**kwargs):
        captured_runtime.update(kwargs)
        return {
            "api_key": "test-key",
            "base_url": "https://ollama.com/v1",
            "provider": "ollama-cloud",
            "api_mode": "chat_completions",
        }

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", _runtime)
    monkeypatch.setitem(sys.modules, "run_agent", SimpleNamespace(AIAgent=_DummyAgent))

    assert oneshot._run_agent("ping", model="qwen35-cloud") == "ok:ping"
    assert captured_runtime["requested"] == "ollama-cloud"
    assert captured_runtime["target_model"] == "qwen3.5:397b-cloud"
    assert captured_runtime["explicit_base_url"] == "https://ollama.com/v1"
    assert _DummyAgent.last_kwargs["model"] == "qwen3.5:397b-cloud"
