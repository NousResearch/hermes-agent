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


def test_api_key_profiles_declare_their_registry_key_env():
    """Every api_key profile backed by an api_key ``PROVIDER_REGISTRY`` row declares that row's key
    variable first in ``env_vars``. Surfaces that read the profile (the provider-catalog e2e
    matrix, doctor, setup prompts) otherwise see a keyless provider. Runs on the default lane so
    the gap is caught even when the change classifier skips the e2e lane (#134320 -> #134385)."""
    from hermes_cli.auth import PROVIDER_REGISTRY

    missing = []
    for profile in providers.list_providers():
        row = PROVIDER_REGISTRY.get(profile.name)
        if profile.auth_type != "api_key" or row is None or row.auth_type != "api_key" or not row.api_key_env_vars:
            continue
        keys = [v for v in profile.env_vars if not v.endswith(("_BASE_URL", "_URL"))]
        if not keys or keys[0] not in row.api_key_env_vars:
            missing.append((profile.name, tuple(profile.env_vars), tuple(row.api_key_env_vars)))
    assert not missing, f"profiles whose env_vars miss their registry key: {missing}"


def test_plugin_load_failure_warns_once_per_source_and_name(tmp_path, caplog, monkeypatch):
    """Discovery retries failed plugin imports by design (a plugin landing mid-scan gets
    picked up) — the load-failure warning must not re-spam on every scan (PR review:
    18-19 duplicate "Failed to load" lines per `hermes doctor` run)."""
    _reset_registry()
    plugin_dir = tmp_path / "broken_plugin"
    plugin_dir.mkdir()
    (plugin_dir / "__init__.py").write_text("raise RuntimeError('boom')\n")
    providers._WARNED_PLUGIN_FAILURES.clear()

    with caplog.at_level("WARNING", logger="providers"):
        providers._import_plugin_dir(plugin_dir, "user", home_key=str(tmp_path))
        providers._import_plugin_dir(plugin_dir, "user", home_key=str(tmp_path))

    try:
        failures = [r for r in caplog.records if "Failed to load" in r.getMessage()]
        assert len(failures) == 1
        assert "boom" in failures[0].getMessage()
    finally:
        providers._WARNED_PLUGIN_FAILURES.clear()
