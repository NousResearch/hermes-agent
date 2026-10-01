"""Provider-plugin API modes flow through the canonical route owner unchanged."""

from __future__ import annotations

import sys

import pytest

from providers.routing import InvocationRequest, resolve_invocation_route

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
    installed: list[tuple[str, str]] = []

    def _install(name: str, mode: str, *, register: bool) -> None:
        env = f"{name.upper().replace('-', '_')}_API_KEY"
        plugin_dir = tmp_path / "hermes" / "plugins" / "model-providers" / name
        plugin_dir.mkdir(parents=True, exist_ok=True)
        (plugin_dir / "plugin.yaml").write_text(
            f"name: {name}\nkind: model-provider\nversion: 0.0.1\ndescription: dialect fixture\n",
            encoding="utf-8",
        )
        source = (
            _PLUGIN_SOURCE.replace("__ENV__", env)
            .replace("__NAME__", name)
            .replace("__MODE__", mode)
            .replace("__REGISTER__", "yes" if register else "no")
        )
        (plugin_dir / "__init__.py").write_text(source, encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        monkeypatch.setenv(env, "sk-fixture")
        import providers as _pkg
        _pkg.discovery._discovered = False
        for mod in [m for m in sys.modules if m.startswith("_hermes_user_provider")]:
            del sys.modules[mod]
        installed.append((name, mode))

    yield _install

    import providers as _pkg
    from agent.transports import _REGISTRY
    for name, mode in installed:
        _pkg.registry._REGISTRY.pop(name, None)
        _REGISTRY.pop(mode, None)
        for alias, canonical in list(_pkg.registry._ALIASES.items()):
            if canonical == name:
                _pkg.registry._ALIASES.pop(alias, None)
    _pkg.registry._PROVIDER_LIST_CACHE = None


def test_registered_plugin_dialect_reaches_route_runtime_and_transport(install_dialect_plugin):
    from agent.transports import get_transport
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from tools.delegate_tool_config import _direct_endpoint_credentials

    install_dialect_plugin("example-dialect", "example_dialect", register=True)

    route = resolve_invocation_route(InvocationRequest(
        provider="example-dialect",
        model="example-model",
        base_url="https://relay.example.test/v1",
    ))
    assert route.api_mode == "example_dialect"

    runtime = resolve_runtime_provider(
        requested="example-dialect", target_model="example-model"
    )
    assert runtime["api_mode"] == "example_dialect"
    assert get_transport("example_dialect") is not None

    creds = _direct_endpoint_credentials(
        {
            "base_url": "https://relay.example.test/v1",
            "api_mode": "example_dialect",
            "provider": "",
            "model": "m",
            "api_key": None,
        },
        None,
    )
    assert creds["api_mode"] == "example_dialect"


def test_unregistered_profile_mode_is_preserved_but_has_no_transport(install_dialect_plugin):
    """Route identity stays extensible; transport registration is validated at construction."""
    from agent.transports import get_transport
    from hermes_cli.runtime_provider import resolve_runtime_provider

    install_dialect_plugin("example-bogus", "bogus_mode", register=False)

    route = resolve_invocation_route(InvocationRequest(
        provider="example-bogus",
        model="example-model",
        base_url="https://relay.example.test/v1",
    ))
    assert route.api_mode == "bogus_mode"
    assert resolve_runtime_provider(
        requested="example-bogus", target_model="example-model"
    )["api_mode"] == "bogus_mode"
    assert get_transport("bogus_mode") is None
