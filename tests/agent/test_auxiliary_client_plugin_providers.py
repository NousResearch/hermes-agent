"""Plugin-registered auxiliary providers (``register_aux_provider``).

A plugin can contribute a whole auxiliary provider (e.g. a subprocess-backed
subscription pool) without editing this module: the registered name is valid
anywhere a provider name is accepted — explicit ``auxiliary.<task>.provider``,
aliases, and ``fallback_chain`` entries — and resolution degrades to
``(None, None)`` instead of hard-failing when the builder has no credentials.

Registrations are owned by the plugin that made them: a second plugin cannot
take over a name or reroute one through an alias, and forced re-discovery
tears the registry down so a disabled plugin's provider stops resolving.
"""
import os
import textwrap
from pathlib import Path

import pytest

from agent import auxiliary_client as aux
from hermes_cli.plugins import PluginContext, PluginManager


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
    """Registration mutates the registry and both alias maps."""
    saved_providers = dict(aux._PLUGIN_AUX_PROVIDERS)
    saved_aliases = dict(aux._PROVIDER_ALIASES)
    saved_plugin_aliases = dict(aux._PLUGIN_AUX_ALIASES)
    try:
        yield
    finally:
        aux._PLUGIN_AUX_PROVIDERS.clear()
        aux._PLUGIN_AUX_PROVIDERS.update(saved_providers)
        aux._PROVIDER_ALIASES.clear()
        aux._PROVIDER_ALIASES.update(saved_aliases)
        aux._PLUGIN_AUX_ALIASES.clear()
        aux._PLUGIN_AUX_ALIASES.update(saved_plugin_aliases)


def test_reserved_and_invalid_registrations_are_rejected():
    with pytest.raises(ValueError):
        aux.register_aux_provider("openrouter", _good_builder)
    with pytest.raises(ValueError):
        aux.register_aux_provider("claude", _good_builder)  # anthropic alias
    with pytest.raises(ValueError):
        # Provider with no dedicated branch, resolved through the auth
        # PROVIDER_REGISTRY catch-all — shadowing it is just as breaking.
        aux.register_aux_provider("deepseek", _good_builder)
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


# ── Ownership ──────────────────────────────────────────────────────────────


def _other_builder(model=None, *, task=None):
    client = _FakePoolClient(model or "other-default")
    client.api_key = "other-pool"
    return client, model or "other-default"


def test_another_owner_cannot_take_over_a_registered_name():
    aux.register_aux_provider("shared-pool", _good_builder, owner="alpha")

    with pytest.raises(ValueError, match="alpha"):
        aux.register_aux_provider("shared-pool", _other_builder, owner="beta")

    client, _ = aux.resolve_provider_client("shared-pool", model="m")
    assert client.api_key == "test-pool"


def test_owner_may_re_register_its_own_provider():
    aux.register_aux_provider("shared-pool", _good_builder, owner="alpha",
                              aliases=("shared-subs", "shared-legacy"))
    aux.register_aux_provider("shared-pool", _other_builder, owner="alpha",
                              aliases=("shared-subs",))

    client, _ = aux.resolve_provider_client("shared-subs", model="m")
    assert client.api_key == "other-pool"
    # The dropped alias stops resolving instead of dangling on the old builder.
    assert "shared-legacy" not in aux._PROVIDER_ALIASES
    assert aux.resolve_provider_client("shared-legacy", model="m") == (None, None)


def test_alias_owned_by_another_plugin_is_rejected():
    aux.register_aux_provider("alpha-pool", _good_builder, owner="alpha",
                              aliases=("subs",))

    with pytest.raises(ValueError, match="alpha"):
        aux.register_aux_provider("beta-pool", _other_builder, owner="beta",
                                  aliases=("subs",))

    client, _ = aux.resolve_provider_client("subs", model="m")
    assert client.api_key == "test-pool"
    # A rejected registration leaves nothing behind — not even its name.
    assert "beta-pool" not in aux._PLUGIN_AUX_PROVIDERS


def test_alias_cannot_rewrite_an_existing_provider_name():
    aux.register_aux_provider("alpha-pool", _good_builder, owner="alpha")

    with pytest.raises(ValueError, match="alpha"):
        aux.register_aux_provider("beta-pool", _other_builder, owner="beta",
                                  aliases=("alpha-pool",))

    client, _ = aux.resolve_provider_client("alpha-pool", model="m")
    assert client.api_key == "test-pool"


# ── Plugin lifecycle ───────────────────────────────────────────────────────


_AUX_PLUGIN_SOURCE = textwrap.dedent('''\
    class _PluginClient:
        aux_async_passthrough = True
        api_key = "plugin-pool"


    def _build(model=None, *, task=None):
        return _PluginClient(), model or "plugin-default"


    def register(ctx):
        ctx.register_aux_provider("plugin-pool", _build, aliases=("plugin-subs",))
''')


def _install_aux_plugin(hermes_home: Path, *, enabled: bool) -> None:
    """Write a plugin that registers an aux provider, opted in or out."""
    plugin_dir = hermes_home / "plugins" / "aux_pool_plugin"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(
        "name: aux_pool_plugin\nversion: 0.1.0\ndescription: aux provider\n")
    (plugin_dir / "__init__.py").write_text(_AUX_PLUGIN_SOURCE)
    (hermes_home / "config.yaml").write_text(
        "plugins:\n  enabled: [aux_pool_plugin]\n" if enabled
        else "plugins:\n  enabled: []\n")


def test_forced_rediscovery_drops_a_disabled_plugins_provider():
    """A provider must stop routing once its plugin is gone.

    The registry is a module global rather than a PluginManager attribute, so
    without an explicit hand in the force-rediscover teardown the builder of a
    disabled plugin stays reachable for the life of the process.
    """
    home = Path(os.environ["HERMES_HOME"])
    _install_aux_plugin(home, enabled=True)

    manager = PluginManager()
    manager.discover_and_load()
    client, model = aux.resolve_provider_client("plugin-pool", model="m")
    assert client.api_key == "plugin-pool"
    assert model == "m"

    _install_aux_plugin(home, enabled=False)
    manager.discover_and_load(force=True)

    assert aux.resolve_provider_client("plugin-pool", model="m") == (None, None)
    assert aux.resolve_provider_client("plugin-subs", model="m") == (None, None)


def test_rediscovery_re_registers_a_still_enabled_plugin():
    """The teardown must not outlive the sweep that follows it."""
    home = Path(os.environ["HERMES_HOME"])
    _install_aux_plugin(home, enabled=True)

    manager = PluginManager()
    manager.discover_and_load()
    manager.discover_and_load(force=True)

    client, _ = aux.resolve_provider_client("plugin-subs", model="m")
    assert client.api_key == "plugin-pool"
