import pytest

from providers.base import ProviderProfile
import providers


@pytest.fixture(autouse=True)
def isolate_provider_registry():
    registry = providers._REGISTRY.copy()
    aliases = providers._ALIASES.copy()
    provider_list_cache = (
        None
        if providers._PROVIDER_LIST_CACHE is None
        else list(providers._PROVIDER_LIST_CACHE)
    )
    discovered = providers._discovered

    yield

    providers._REGISTRY.clear()
    providers._REGISTRY.update(registry)
    providers._ALIASES.clear()
    providers._ALIASES.update(aliases)
    providers._PROVIDER_LIST_CACHE = provider_list_cache
    providers._discovered = discovered


def _profile(name: str, *aliases: str) -> ProviderProfile:
    return ProviderProfile(name=name, aliases=aliases)


def _reset_registry() -> None:
    providers._REGISTRY.clear()
    providers._ALIASES.clear()
    providers._PROVIDER_LIST_CACHE = None
    providers._discovered = True


def test_list_providers_reuses_cached_snapshot_until_registration_changes():
    _reset_registry()
    first = _profile("alpha")
    providers.register_provider(first)

    listed = providers.list_providers()
    listed.clear()

    assert providers.list_providers() == [first]

    # Hit-path copy guard: mutating a CACHED return must not corrupt the
    # module-level snapshot for later callers (aliasing bug class).
    providers.list_providers().clear()
    assert providers.list_providers() == [first]

    second = _profile("beta")
    providers.register_provider(second)

    assert providers.list_providers() == [first, second]


def test_list_providers_dedupes_aliases_in_cached_snapshot():
    _reset_registry()
    profile = _profile("kimi", "moonshot", "kimi-k2")
    providers.register_provider(profile)

    assert providers.get_provider_profile("moonshot") is profile
    assert providers.list_providers() == [profile]


def test_provider_lookups_reuse_the_home_plugin_stamp_within_its_ttl(tmp_path, monkeypatch):
    """The model-picker hot path must not stat plugin directories per lookup (#119950)."""
    _reset_registry()
    profile = _profile("known")
    providers.register_provider(profile)
    stamp_calls = 0
    original_stamps = providers._plugin_dir_stamps

    def count_stamps(home):
        nonlocal stamp_calls
        stamp_calls += 1
        return original_stamps(home)

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr(providers, "_HOME_LAYERS", {})
    monkeypatch.setattr(providers, "_plugin_dir_stamps", count_stamps)
    monkeypatch.setattr(providers, "_scan_home_layer", lambda *_: None)

    assert providers.get_provider_profile("known") is profile
    assert providers.list_providers() == [profile]
    assert providers.get_provider_profile("known") is profile

    assert stamp_calls == 1
