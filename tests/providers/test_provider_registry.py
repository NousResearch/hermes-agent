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


def test_installed_provider_wins_over_same_named_user_dir(tmp_path, monkeypatch):
    """Step 2b (flat installed dir) must override step 2 (model-providers dir)
    on directory-name collision: both derive one module name each, so both
    import and register, and last-writer-wins keeps the later step."""
    import sys

    home = tmp_path / "hermes"
    mp_dir = home / "plugins" / "model-providers" / "dup"
    flat_dir = home / "plugins" / "dup"
    mp_dir.mkdir(parents=True)
    flat_dir.mkdir(parents=True)
    (mp_dir / "__init__.py").write_text(
        "from providers.base import ProviderProfile\n"
        "import providers\n"
        'providers.register_provider(ProviderProfile(name="dup", aliases=("from-mp-dir",)))\n',
        encoding="utf-8",
    )
    (flat_dir / "__init__.py").write_text(
        "from providers.base import ProviderProfile\n"
        "import providers\n"
        'providers.register_provider(ProviderProfile(name="dup", aliases=("from-flat-dir",)))\n',
        encoding="utf-8",
    )
    (flat_dir / "plugin.yaml").write_text("kind: model-provider\n", encoding="utf-8")

    recorded = []
    real_register = providers.register_provider

    def _recording_register(profile):
        recorded.append(profile)
        return real_register(profile)

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(providers, "_BUNDLED_PLUGINS_DIR", tmp_path / "empty-bundled")
    (tmp_path / "empty-bundled").mkdir()
    monkeypatch.setattr(providers, "register_provider", _recording_register)
    providers._discovered = False
    try:
        providers._discover_providers()
    finally:
        for mod in [m for m in sys.modules if m.startswith("_hermes_user_provider_")]:
            del sys.modules[mod]

    assert [p.aliases for p in recorded if p.name == "dup"] == [
        ("from-mp-dir",), ("from-flat-dir",),
    ]
    assert providers.get_provider_profile("dup").aliases == ("from-flat-dir",)
