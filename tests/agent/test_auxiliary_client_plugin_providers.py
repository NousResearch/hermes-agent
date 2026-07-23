"""Plugin-registered auxiliary providers (``register_aux_provider``).

A plugin can contribute a whole auxiliary provider (e.g. a subprocess-backed
subscription pool) without editing this module: the registered name is valid
anywhere a provider name is accepted — explicit ``auxiliary.<task>.provider``,
aliases, and ``fallback_chain`` entries — and resolution degrades to
``(None, None)`` instead of hard-failing when the builder has no credentials.
"""
import os
from pathlib import Path

import pytest

from agent import auxiliary_client as aux
from hermes_cli.plugins import PluginContext


class _FakePoolClient:
    aux_async_passthrough = True

    def __init__(self, model):
        self.model = model
        self.base_url = "acp://test-pool"
        self.api_key = "test-pool"


def _good_builder(model=None, *, task=None):
    return _FakePoolClient(model or "pool-default"), model or "pool-default"


@pytest.fixture(autouse=True)
def _clean_registry():
    """Registration mutates both the registry and the alias map."""
    saved_providers = dict(aux._PLUGIN_AUX_PROVIDERS)
    saved_aliases = dict(aux._PROVIDER_ALIASES)
    try:
        yield
    finally:
        aux._PLUGIN_AUX_PROVIDERS.clear()
        aux._PLUGIN_AUX_PROVIDERS.update(saved_providers)
        aux._PROVIDER_ALIASES.clear()
        aux._PROVIDER_ALIASES.update(saved_aliases)


def test_reserved_and_invalid_registrations_are_rejected():
    with pytest.raises(ValueError):
        aux.register_aux_provider("openrouter", _good_builder)
    with pytest.raises(ValueError):
        aux.register_aux_provider("claude", _good_builder)  # anthropic alias
    with pytest.raises(ValueError):
        aux.register_aux_provider("test-pool", "not-callable")


def test_registered_provider_resolves_by_name_and_alias():
    aux.register_aux_provider("test-pool", _good_builder, aliases=("test-subs",))

    client, model = aux.resolve_provider_client("test-pool", model="m1")
    assert isinstance(client, _FakePoolClient) and model == "m1"

    client, model = aux.resolve_provider_client("test-subs", model="m2")
    assert isinstance(client, _FakePoolClient) and model == "m2"


def test_async_passthrough_client_is_returned_unwrapped():
    aux.register_aux_provider("test-pool", _good_builder)
    client, model = aux.resolve_provider_client(
        "test-pool", model="m3", async_mode=True)
    assert isinstance(client, _FakePoolClient) and model == "m3"


def test_builder_degradation_never_raises():
    aux.register_aux_provider(
        "empty-pool", lambda model=None, *, task=None: (None, None))
    assert aux.resolve_provider_client("empty-pool") == (None, None)

    def boom(model=None, *, task=None):
        raise RuntimeError("boom")

    aux.register_aux_provider("broken-pool", boom)
    assert aux.resolve_provider_client("broken-pool") == (None, None)


def test_fallback_chain_reaches_registered_provider():
    aux.register_aux_provider("test-pool", _good_builder)
    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text(
        "auxiliary:\n"
        "  compression:\n"
        "    fallback_chain:\n"
        "      - provider: test-pool\n"
        "        model: haiku\n"
    )
    client, model, label = aux._try_configured_fallback_chain(
        "compression", "openrouter")
    assert isinstance(client, _FakePoolClient)
    assert model == "haiku"


def test_plugin_context_exposes_registration():
    assert callable(getattr(PluginContext, "register_aux_provider", None))
