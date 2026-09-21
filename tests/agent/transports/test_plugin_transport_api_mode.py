"""A provider plugin's own ``api_mode`` reaches the runtime transport (#53054).

``register_transport(api_mode, cls)`` is the public seam for a plugin that speaks its own wire
dialect, and ``ProviderProfile.api_mode`` names it. Every gate between the profile and the agent
(``providers.determine_api_mode`` → ``runtime_provider`` → ``agent_init._resolve_api_mode`` →
the delegation resolver) validated the mode against a closed literal set, so the plugin's mode was
rewritten to ``chat_completions`` and its transport never selected — no error, prose-only turns.

Both tests drive a REAL plugin discovered from an isolated HERMES_HOME rather than the gates in
isolation, so they fail if any single gate regresses.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

_PLUGIN_SOURCE = '''\
from agent.transports import register_transport
from agent.transports.chat_completions import ChatCompletionsTransport
from providers import register_provider
from providers.base import ProviderProfile


class DialectTransport(ChatCompletionsTransport):
    api_mode = "__MODE__"


if "__REGISTER__" == "yes":
    register_transport("__MODE__", DialectTransport)
register_provider(ProviderProfile(name="__NAME__", auth_type="api_key", env_vars=("__ENV__",),
    base_url="https://relay.example.test/v1", api_mode="__MODE__", fallback_models=("example-model",)))
'''


@pytest.fixture
def install_dialect_plugin(tmp_path, monkeypatch):
    """Write a model-provider plugin (optionally registering its transport) under a temp HERMES_HOME."""
    installed: list[tuple[str, str]] = []

    def _install(name: str, mode: str, *, register: bool) -> None:
        env = f"{name.upper().replace('-', '_')}_API_KEY"
        plugin_dir = tmp_path / "hermes" / "plugins" / "model-providers" / name
        plugin_dir.mkdir(parents=True, exist_ok=True)
        (plugin_dir / "plugin.yaml").write_text(
            f"name: {name}\nkind: model-provider\nversion: 0.0.1\ndescription: dialect fixture\n", encoding="utf-8")
        source = (_PLUGIN_SOURCE.replace("__ENV__", env).replace("__NAME__", name).replace("__MODE__", mode)
                  .replace("__REGISTER__", "yes" if register else "no"))
        (plugin_dir / "__init__.py").write_text(source, encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        monkeypatch.setenv(env, "sk-fixture")
        import providers as _pkg
        _pkg._discovered = False
        for mod in [m for m in sys.modules if m.startswith("_hermes_user_provider")]:
            del sys.modules[mod]
        installed.append((name, mode))

    yield _install

    import providers as _pkg
    from agent.transports import _REGISTRY
    for name, mode in installed:
        _pkg._REGISTRY.pop(name, None)
        _REGISTRY.pop(mode, None)
        for alias, canonical in list(_pkg._ALIASES.items()):
            if canonical == name:
                _pkg._ALIASES.pop(alias, None)
    _pkg._PROVIDER_LIST_CACHE = None


def _agent_ladder_mode(provider: str, base_url: str, api_mode: str) -> str:
    from agent.agent_init import _resolve_api_mode
    agent = SimpleNamespace(provider=provider, model="example-model", _base_url_hostname="relay.example.test",
                            _base_url_lower=base_url.lower(), api_mode=None)
    _resolve_api_mode(agent, api_mode, provider, base_url)
    return agent.api_mode


def test_registered_plugin_dialect_reaches_every_gate(install_dialect_plugin):
    """profile.api_mode → determine_api_mode → resolve_runtime_provider → agent ladder → delegation."""
    from hermes_cli.providers import determine_api_mode
    from hermes_cli.runtime_provider import _parse_api_mode, resolve_runtime_provider
    from tools.delegate_tool_config import _direct_endpoint_credentials

    install_dialect_plugin("example-dialect", "example_dialect", register=True)

    assert determine_api_mode("example-dialect", "https://relay.example.test/v1") == "example_dialect"
    runtime = resolve_runtime_provider(requested="example-dialect", target_model="example-model")
    assert runtime["api_mode"] == "example_dialect"
    assert _parse_api_mode("example_dialect") == "example_dialect"
    assert _agent_ladder_mode(runtime["provider"], runtime["base_url"], runtime["api_mode"]) == "example_dialect"
    creds = _direct_endpoint_credentials(
        {"base_url": "https://relay.example.test/v1", "api_mode": "example_dialect", "provider": "", "model": "m",
         "api_key": None}, None)
    assert creds["api_mode"] == "example_dialect"


def test_unregistered_profile_mode_still_degrades_to_chat_completions(install_dialect_plugin):
    """The sets stay closed: a profile naming a transport nobody registered is not admitted."""
    from hermes_cli.providers import determine_api_mode
    from hermes_cli.runtime_provider import _parse_api_mode, resolve_runtime_provider

    install_dialect_plugin("example-bogus", "bogus_mode", register=False)

    assert determine_api_mode("example-bogus", "https://relay.example.test/v1") == "chat_completions"
    assert resolve_runtime_provider(requested="example-bogus", target_model="example-model")["api_mode"] == "chat_completions"
    assert _parse_api_mode("bogus_mode") is None
    assert _agent_ladder_mode("example-bogus", "https://relay.example.test/v1", "bogus_mode") == "chat_completions"
