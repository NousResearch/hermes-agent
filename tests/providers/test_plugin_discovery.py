"""Tests for model-provider plugin discovery."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _clear_provider_caches():
    import providers as _pkg

    _pkg._REGISTRY.clear()
    _pkg._ALIASES.clear()
    _pkg._PROVIDER_LIST_CACHE = None
    _pkg._discovered = False
    _pkg._ACTIVATION_STATE = None
    _pkg._DISCOVERY_FINGERPRINT = None
    _pkg._IMPORTED_PROVIDER_MODULES.clear()
    for mod in list(sys.modules.keys()):
        if mod.startswith((
            "plugins.model_providers", "_hermes_user_provider", "_hermes_project_provider"
        )):
            del sys.modules[mod]


def _new_home(tmp_path: Path, name: str = "home") -> Path:
    home = tmp_path / name
    home.mkdir()
    return home


@pytest.fixture
def provider_home(tmp_path, monkeypatch):
    home = _new_home(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@contextmanager
def _provider_scope(home: Path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _run_python_json(home: Path, lines: tuple[str, ...], **extra_env):
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env.update(extra_env)
    completed = subprocess.run(
        [sys.executable, "-c", "\n".join(lines)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_bundled_plugins_discovered():
    """Every plugins/model-providers/<name>/ should contain a plugin.yaml + __init__.py."""
    plugins_dir = REPO_ROOT / "plugins" / "model-providers"
    assert plugins_dir.is_dir(), f"Missing {plugins_dir}"

    child_dirs = [c for c in plugins_dir.iterdir() if c.is_dir()]
    assert len(child_dirs) >= 28, f"Expected at least 28 provider plugins, found {len(child_dirs)}"

    for child in child_dirs:
        assert (child / "__init__.py").exists(), f"{child.name} missing __init__.py"
        assert (child / "plugin.yaml").exists(), f"{child.name} missing plugin.yaml"


def test_all_profiles_register():
    """After discovery, the registry must contain every bundled provider directory.

    This is an invariant — the number of profiles matches the number of plugin
    directories, not a hardcoded count. Counts shift when providers are
    added/removed; that's expected and shouldn't break CI.
    """
    _clear_provider_caches()
    from providers import list_providers

    plugins_dir = REPO_ROOT / "plugins" / "model-providers"
    plugin_dir_count = sum(1 for c in plugins_dir.iterdir() if c.is_dir())

    profiles = list_providers()
    names = sorted(p.name for p in profiles)
    # Some plugin __init__.py files register multiple profiles, so the registry
    # count is >= the directory count (never less).
    assert len(names) >= plugin_dir_count, (
        f"Expected at least {plugin_dir_count} profiles (one per plugin dir), got {len(names)}: {names}"
    )

    # Spot-check representative providers from different categories
    for required in (
        "openrouter", "anthropic", "custom", "bedrock", "openai-codex",
        "minimax-oauth", "gmi", "xiaomi", "alibaba-coding-plan", "fireworks",
    ):
        assert required in names, f"Missing profile: {required}"


def test_user_plugin_overrides_bundled(provider_home):
    _install_provider(
        provider_home,
        "gmi",
        base_url="https://user-override.example.com/v1",
        env_var="GMI_API_KEY",
        aliases=("gmi-user-override-test",),
    )
    _clear_provider_caches()
    from providers import get_provider_profile

    gmi = get_provider_profile("gmi")
    assert gmi is not None
    assert gmi.base_url == "https://user-override.example.com/v1"
    assert "gmi-user-override-test" in gmi.aliases
    _clear_provider_caches()


def _write_user_provider(
    home: Path,
    name: str,
    *,
    base_url: str,
    env_var: str = "TEST_PROVIDER_API_KEY",
    env_vars: tuple[str, ...] | None = None,
    aliases: tuple[str, ...] = (),
    marker: Path | None = None,
    fail_after_register: bool = False,
    provider_ids: tuple[str, ...] | None = None,
    profile_name: str | None = None,
) -> None:
    plugin_dir = _provider_dir(home, name)
    plugin_dir.mkdir(parents=True)
    registered_env_vars = env_vars or (env_var,)
    registered_name = profile_name or name
    marker_write = (
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')\n"
        if marker else ""
    )
    (plugin_dir / "__init__.py").write_text(
        marker_write
        + "from providers import register_provider\n"
        + "from providers.base import ProviderProfile\n"
        + "register_provider(ProviderProfile(\n"
        + f"    name={registered_name!r}, display_name={registered_name!r},\n"
        + f"    aliases={aliases!r}, env_vars={registered_env_vars!r},\n"
        + f"    base_url={base_url!r}, auth_type='api_key',\n"
        + "))\n"
        + ("raise RuntimeError('provider import failed')\n" if fail_after_register else ""),
        encoding="utf-8",
    )
    (plugin_dir / "plugin.yaml").write_text(
        f"name: {name}\n"
        "kind: model-provider\n"
        f"provider_ids: [{', '.join(provider_ids or (name,))}]\n",
        encoding="utf-8",
    )


def _write_activation(home: Path, *, enabled=(), disabled=()) -> None:
    (home / "config.yaml").write_text(
        "plugins:\n  enabled:\n"
        + "".join(f"    - {value}\n" for value in enabled)
        + "  disabled:\n"
        + "".join(f"    - {value}\n" for value in disabled),
        encoding="utf-8",
    )


def _provider_dir(home: Path, name: str) -> Path:
    return home / "plugins" / "model-providers" / name


def _install_provider(home: Path, name: str, *, state="enabled", **kwargs) -> str:
    _write_user_provider(home, name, **kwargs)
    key = f"model-providers/{name}"
    _write_activation(home, **({state: (key,)} if state else {}))
    return key


def _append_config(home: Path, text: str) -> None:
    path = home / "config.yaml"
    path.write_text(path.read_text(encoding="utf-8") + text, encoding="utf-8")


def _assert_provenance(providers, snapshot, provider_ids, *, current: bool) -> None:
    for provider_id in provider_ids:
        assert (provider_id in snapshot.current_canonical_provider_ids) is current
        assert provider_id in snapshot.observed_provider_canonical_ids
        assert provider_id in snapshot.observed_provider_aliases
        assert providers.get_provider_identity_provenance(provider_id) == "canonical"


def test_user_provider_is_not_imported_until_explicitly_enabled(tmp_path, provider_home):
    marker = tmp_path / "provider-imported.txt"
    _install_provider(
        provider_home,
        "opt-in-provider",
        state=None,
        base_url="https://opt-in.example/v1",
        marker=marker,
    )
    import providers

    _clear_provider_caches()
    assert providers.get_provider_profile("opt-in-provider") is None
    assert not marker.exists()

    _write_activation(provider_home, enabled=("model-providers/opt-in-provider",))
    providers.invalidate_provider_discovery()

    assert providers.get_provider_profile("opt-in-provider") is not None
    assert marker.read_text(encoding="utf-8") == "imported"


def test_provider_plugin_can_query_in_progress_registry_during_import(provider_home):
    _install_provider(
        provider_home,
        "openrouter",
        base_url="https://registry-reader.example/v1",
    )
    plugin_dir = provider_home / "plugins" / "model-providers" / "openrouter"
    (plugin_dir / "__init__.py").write_text(
        "from providers import (\n"
        "    get_provider_profile, list_providers, register_provider,\n"
        ")\n"
        "from providers.base import ProviderProfile\n"
        "assert get_provider_profile('openrouter') in list_providers()\n"
        "profile = ProviderProfile(\n"
        "    name='openrouter',\n"
        "    aliases=('registry-reader-alias',),\n"
        "    env_vars=('REGISTRY_READER_API_KEY',),\n"
        "    base_url='https://registry-reader.example/v1',\n"
        ")\n"
        "register_provider(profile)\n"
        "assert get_provider_profile('registry-reader-alias') is profile\n"
        "assert profile in list_providers()\n",
        encoding="utf-8",
    )
    import providers

    providers.invalidate_provider_discovery()

    assert (
        providers.get_provider_profile("registry-reader-alias").base_url
        == "https://registry-reader.example/v1"
    )


def test_provider_discovery_refreshes_when_profile_home_changes(tmp_path, monkeypatch):
    first_home = _new_home(tmp_path, "first")
    second_home = _new_home(tmp_path, "second")
    _install_provider(first_home, "profile-provider", base_url="https://first.example/v1")
    _install_provider(second_home, "profile-provider", base_url="https://second.example/v1")

    import providers

    monkeypatch.setenv("HERMES_HOME", str(first_home))
    providers.invalidate_provider_discovery()
    assert (
        providers.get_provider_profile("profile-provider").base_url
        == "https://first.example/v1"
    )

    monkeypatch.setenv("HERMES_HOME", str(second_home))
    assert (
        providers.get_provider_profile("profile-provider").base_url
        == "https://second.example/v1"
    )


def test_concurrent_profile_catalogs_do_not_mix_key_and_endpoint(tmp_path, monkeypatch):
    home_a = _new_home(tmp_path, "profile-a")
    home_b = _new_home(tmp_path, "profile-b")
    for home, endpoint, override, secret in (
        (home_a, "https://a.example/v1", "https://a-override.example/v1", "secret-a"),
        (home_b, "https://b.example/v1", "https://b-override.example/v1", "secret-b"),
    ):
        _install_provider(
            home,
            "profile-provider",
            base_url=endpoint,
            env_vars=("PROFILE_PROVIDER_API_KEY", "PROFILE_PROVIDER_BASE_URL"),
        )
        (home / ".env").write_text(
            f"PROFILE_PROVIDER_API_KEY={secret}\n"
            f"PROFILE_PROVIDER_BASE_URL={override}\n",
            encoding="utf-8",
        )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "process-home"))
    import providers
    from hermes_cli import auth
    a_discovered = threading.Event()
    b_published = threading.Event()

    def run_a():
        with _provider_scope(home_a):
            assert providers.get_provider_profile("profile-provider").base_url == "https://a.example/v1"
            a_discovered.set()
            assert b_published.wait(timeout=10)
            config = auth.PROVIDER_REGISTRY["profile-provider"]
            credentials = auth.resolve_api_key_provider_credentials("profile-provider")
            return config.inference_base_url, credentials

    def run_b():
        assert a_discovered.wait(timeout=10)
        with _provider_scope(home_b):
            try:
                config = auth.PROVIDER_REGISTRY["profile-provider"]
                credentials = auth.resolve_api_key_provider_credentials("profile-provider")
                return config.inference_base_url, credentials
            finally:
                b_published.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(run_a)
        future_b = executor.submit(run_b)
        endpoint_a, credentials_a = future_a.result(timeout=15)
        endpoint_b, credentials_b = future_b.result(timeout=15)

    assert (endpoint_a, credentials_a["api_key"], credentials_a["base_url"]) == (
        "https://a.example/v1", "secret-a", "https://a-override.example/v1"
    )
    assert (endpoint_b, credentials_b["api_key"], credentials_b["base_url"]) == (
        "https://b.example/v1", "secret-b", "https://b-override.example/v1"
    )


def test_cached_profile_switch_does_not_repeat_refresh_hooks(tmp_path, monkeypatch):
    home_a = _new_home(tmp_path, "profile-a")
    home_b = _new_home(tmp_path, "profile-b")
    for home in (home_a, home_b):
        _write_activation(home)

    import providers

    notifications: list[tuple[str, str]] = []
    monkeypatch.setattr(
        providers,
        "_PROVIDER_REFRESH_HOOKS",
        [lambda: notifications.append(providers.get_provider_scope_identity())],
    )

    for home in (home_a, home_b, home_a, home_b):
        monkeypatch.setenv("HERMES_HOME", str(home))
        snapshot = providers.get_provider_catalog_snapshot()
        assert snapshot.scope_identity == (str(home.resolve()), "")

    assert notifications == [
        (str(home_a.resolve()), ""),
        (str(home_b.resolve()), ""),
    ]


def test_inactive_external_manifest_cannot_suppress_static_provider(tmp_path, provider_home):
    marker = tmp_path / "openai-override-imported.txt"
    plugin_key = "model-providers/openai-api"
    _install_provider(
        provider_home,
        "openai-api",
        state=None,
        base_url="https://override-openai.example/v1",
        env_var="OVERRIDE_OPENAI_API_KEY",
        marker=marker,
    )

    import providers
    from hermes_cli import auth

    providers.invalidate_provider_discovery()
    assert not marker.exists()
    assert not providers.is_plugin_managed_provider_id("openai-api")
    assert auth.resolve_provider("openai-api") == "openai-api"
    assert (
        auth.PROVIDER_REGISTRY["openai-api"].inference_base_url
        == "https://api.openai.com/v1"
    )

    _write_activation(provider_home, enabled=(plugin_key,))
    providers.invalidate_provider_discovery()
    assert (
        providers.get_provider_profile("openai-api").base_url
        == "https://override-openai.example/v1"
    )
    assert marker.read_text(encoding="utf-8") == "imported"
    assert not providers.is_plugin_managed_provider_id("openai-api")

    _write_activation(provider_home, disabled=(plugin_key,))
    providers.invalidate_provider_discovery()

    assert providers.get_provider_profile("openai-api") is None
    assert not providers.is_plugin_managed_provider_id("openai-api")
    assert auth.resolve_provider("openai-api") == "openai-api"
    assert (
        auth.PROVIDER_REGISTRY["openai-api"].inference_base_url
        == "https://api.openai.com/v1"
    )


def test_profileless_core_identity_table_matches_runtime_catalogs():
    import providers
    from hermes_cli import auth, models
    from hermes_cli.providers import HERMES_OVERLAYS
    from providers._core_identities import (
        PROFILELESS_CORE_PROVIDER_ALIASES,
        PROFILELESS_CORE_PROVIDER_IDS,
    )

    snapshot = providers.get_provider_catalog_snapshot()
    virtual_ids = {
        provider_id
        for provider_id, overlay in HERMES_OVERLAYS.items()
        if overlay.auth_type == "virtual"
    }
    expected_ids = (
        set(auth._STATIC_PROVIDER_REGISTRY) | virtual_ids
    ) - set(snapshot.bundled_provider_ids)
    expected_aliases = {
        alias: canonical
        for alias, canonical in models._STATIC_PROVIDER_ALIASES.items()
        if canonical in expected_ids and alias != canonical
    }

    assert set(PROFILELESS_CORE_PROVIDER_IDS) == expected_ids
    assert PROFILELESS_CORE_PROVIDER_ALIASES == expected_aliases
    assert (
        snapshot.bundled_provider_ids
        <= snapshot.plugin_managed_provider_ids
    )
    assert "_core_identities" not in snapshot.known_provider_ids


def test_core_ownership_is_stable_across_models_then_auth_imports(provider_home):
    claimed_ids = ("openai-api", "lm-studio", "tokenhub", "x-ai-oauth", "moa")
    plugin_name = "early-core-collision"
    _install_provider(
        provider_home,
        plugin_name,
        state=None,
        base_url="https://inactive.invalid/v1",
        provider_ids=claimed_ids,
    )
    plugin_dir = _provider_dir(provider_home, plugin_name)
    (plugin_dir / "__init__.py").write_text(
        "raise AssertionError('inactive collision must not import')\n",
        encoding="utf-8",
    )
    result = _run_python_json(
        provider_home,
        (
            "import json",
            "from hermes_cli import models",
            "import providers",
            f"claimed = {claimed_ids!r}",
            "before = [v for v in claimed if providers.is_plugin_managed_provider_id(v)]",
            "before_moa = 'moa' in {e.slug for e in models.CANONICAL_PROVIDERS}",
            "from hermes_cli import auth",
            "after = [v for v in claimed if providers.is_plugin_managed_provider_id(v)]",
            "after_moa = 'moa' in {e.slug for e in models.CANONICAL_PROVIDERS}",
            "print(json.dumps([before, before_moa, after, after_moa, auth.resolve_provider('openai-api')]))",
        ),
    )
    assert result == [[], True, [], True, "openai-api"]


def test_disabled_external_manifest_provider_id_stays_activation_managed(
    tmp_path, provider_home, monkeypatch
):
    marker = tmp_path / "external-imported.txt"
    plugin_name = "fresh-external-gate"
    provider_id = "fresh-external-route"
    _install_provider(
        provider_home,
        plugin_name,
        state="disabled",
        base_url="https://external.invalid/v1",
        marker=marker,
        provider_ids=(provider_id,),
    )
    import providers
    from hermes_cli import runtime_provider

    providers.invalidate_provider_discovery()

    assert not marker.exists()
    assert providers.is_plugin_managed_provider_id(provider_id)
    monkeypatch.setattr(
        runtime_provider,
        "load_config",
        lambda: {
            "custom_providers": [
                {
                    "name": provider_id,
                    "base_url": "https://custom.invalid/v1",
                    "api_key": "custom-key",
                }
            ]
        },
    )
    with pytest.raises(runtime_provider.AuthError, match="disabled by plugin"):
        runtime_provider.resolve_runtime_provider(requested=provider_id)


def test_named_custom_alias_provenance_does_not_cross_profile_scopes(tmp_path, monkeypatch):
    home_a = _new_home(tmp_path, "profile-a")
    home_b = _new_home(tmp_path, "profile-b")
    _install_provider(
        home_a,
        "a-alias-owner",
        base_url="https://a.example/v1",
        aliases=("acme",),
    )
    _write_activation(home_b)

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "process-home"))
    import providers
    from hermes_cli import runtime_provider
    with _provider_scope(home_a):
        providers.invalidate_provider_discovery()
        assert providers.get_provider_profile("acme") is not None
        assert providers.get_provider_identity_provenance("acme") == "alias"

    with _provider_scope(home_b):
        providers.invalidate_provider_discovery()
        assert "acme" not in (
            providers.get_provider_catalog_snapshot().observed_provider_aliases
        )

        _install_provider(
            home_b,
            "b-alias-owner",
            base_url="https://b-alias.example/v1",
            aliases=("acme",),
        )
        providers.invalidate_provider_discovery()
        assert providers.get_provider_identity_provenance("acme") == "alias"

        _write_user_provider(
            home_b,
            "b-manifest-owner",
            base_url="https://b.example/v1",
            provider_ids=("acme",),
        )
        _write_activation(
            home_b,
            enabled=("model-providers/b-alias-owner",),
            disabled=("model-providers/b-manifest-owner",),
        )
        _append_config(
            home_b,
            "custom_providers:\n  - name: acme\n"
            "    base_url: https://custom-b.example/v1\n    api_key: custom-key\n",
        )
        providers.invalidate_provider_discovery()
        snapshot_b = providers.get_provider_catalog_snapshot()
        assert "acme" in snapshot_b.current_canonical_provider_ids
        assert "acme" in snapshot_b.observed_provider_aliases
        assert providers.get_provider_identity_provenance("acme") == "canonical"
        assert providers.is_provider_plugin_active("acme")
        assert not providers.is_provider_canonical_identity_active("acme")
        with pytest.raises(runtime_provider.AuthError, match="disabled by plugin"):
            runtime_provider.resolve_runtime_provider(requested="acme")


def test_scoped_canonical_history_outranks_alias_after_disable(
    provider_home,
):
    _write_user_provider(
        provider_home,
        "alias-owner",
        base_url="https://alias.example/v1",
        aliases=(" ACME ", " MANIFEST-ACME "),
    )
    _write_user_provider(
        provider_home,
        "manifest-profile-mismatch",
        base_url="https://canonical.example/v1",
        provider_ids=(" MANIFEST-ACME ",),
        profile_name=" ACME ",
    )
    plugin_keys = (
        "model-providers/alias-owner",
        "model-providers/manifest-profile-mismatch",
    )
    _write_activation(provider_home, enabled=plugin_keys)

    import providers
    from hermes_cli import runtime_provider

    providers.invalidate_provider_discovery()
    active = providers.get_provider_catalog_snapshot()
    colliding_ids = ("acme", "manifest-acme")
    _assert_provenance(providers, active, colliding_ids, current=True)
    for provider_id in colliding_ids:
        assert providers.get_provider_identity_provenance(f"  {provider_id.upper()}  ") == "canonical"

    manifest_path = _provider_dir(provider_home, "manifest-profile-mismatch") / "plugin.yaml"
    manifest_path.write_text(
        "name: manifest-profile-mismatch\n"
        "kind: model-provider\n"
        "provider_ids: [replacement-route]\n",
        encoding="utf-8",
    )

    _write_activation(provider_home, disabled=plugin_keys)
    _append_config(
        provider_home,
        "custom_providers:\n"
        "  - name: acme\n    base_url: https://custom.example/v1\n    api_key: custom-key\n"
        "  - name: manifest-acme\n"
        "    base_url: https://custom-manifest.example/v1\n    api_key: custom-key\n",
    )
    providers.invalidate_provider_discovery()

    disabled = providers.get_provider_catalog_snapshot()
    _assert_provenance(providers, disabled, colliding_ids, current=False)
    for provider_id in colliding_ids:
        assert providers.is_plugin_managed_provider_id(provider_id)
        assert not providers.is_provider_plugin_active(provider_id)
        with pytest.raises(
            runtime_provider.AuthError,
            match="disabled by plugin",
        ):
            runtime_provider.resolve_runtime_provider(requested=provider_id)


def test_provenance_growth_evicts_same_scope_activation_snapshots(provider_home, monkeypatch):
    plugin_key = _install_provider(
        provider_home,
        "history-owner",
        state="disabled",
        base_url="https://history.example/v1",
        provider_ids=("manifest-route",),
        profile_name="history-only-canonical",
    )

    import providers
    from hermes_cli.plugin_activation import PluginActivationState

    providers.invalidate_provider_discovery()
    disabled_snapshot = providers.get_provider_catalog_snapshot()
    assert (
        "history-only-canonical"
        not in disabled_snapshot.observed_provider_canonical_ids
    )

    state = {
        "value": PluginActivationState(
            enabled=frozenset({plugin_key}),
        )
    }
    monkeypatch.setattr(
        providers,
        "_current_activation_state",
        lambda: state["value"],
    )
    enabled_snapshot = providers.get_provider_catalog_snapshot()
    assert (
        "history-only-canonical"
        in enabled_snapshot.observed_provider_canonical_ids
    )

    state["value"] = PluginActivationState(
        enabled=frozenset(),
        disabled=frozenset({plugin_key}),
    )
    rebuilt_disabled = providers.get_provider_catalog_snapshot()

    assert rebuilt_disabled is not disabled_snapshot
    assert (
        "history-only-canonical"
        in rebuilt_disabled.observed_provider_canonical_ids
    )


def test_inactive_external_manifest_cannot_gate_runtime_registered_provider(
    provider_home, monkeypatch
):
    provider_id = "runtime-owned-provider"
    plugin_name = "runtime-collision"
    _install_provider(
        provider_home,
        plugin_name,
        state=None,
        base_url="https://inactive.invalid/v1",
        provider_ids=(provider_id,),
    )
    plugin_dir = _provider_dir(provider_home, plugin_name)
    (plugin_dir / "__init__.py").write_text(
        "raise AssertionError('inactive collision must not import')\n",
        encoding="utf-8",
    )

    import providers
    from providers.base import ProviderProfile

    runtime_profile = ProviderProfile(
        name=provider_id,
        aliases=("runtime-owned-alias",),
        env_vars=("RUNTIME_OWNED_API_KEY",),
        base_url="https://runtime-owned.example/v1",
    )
    monkeypatch.setattr(providers, "_RUNTIME_REGISTRY", {provider_id: runtime_profile})
    monkeypatch.setattr(
        providers,
        "_RUNTIME_ALIASES",
        {"runtime-owned-alias": provider_id},
    )
    monkeypatch.setattr(
        providers,
        "_RUNTIME_REGISTRATION_GENERATION",
        providers._RUNTIME_REGISTRATION_GENERATION + 1,
    )

    providers.invalidate_provider_discovery()

    assert providers.get_provider_profile(provider_id) is runtime_profile
    assert providers.get_provider_profile("runtime-owned-alias") is runtime_profile
    assert not providers.is_plugin_managed_provider_id(provider_id)
    assert not providers.is_plugin_managed_provider_id("runtime-owned-alias")


def test_disabling_provider_refreshes_active_provider_indexes(
    provider_home,
):
    _write_user_provider(
        provider_home,
        "refresh-provider",
        base_url="https://refresh.example/v1",
        env_var="REFRESH_PROVIDER_API_KEY",
    )
    _write_activation(
        provider_home,
        enabled=("model-providers/refresh-provider",),
    )
    import providers
    from hermes_cli import auth, models

    providers.invalidate_provider_discovery()
    assert "refresh-provider" in auth.PROVIDER_REGISTRY
    assert any(
        entry.slug == "refresh-provider"
        for entry in models.CANONICAL_PROVIDERS
    )
    assert "refresh-provider" in models._KNOWN_PROVIDER_NAMES

    _write_activation(
        provider_home,
        disabled=("model-providers/refresh-provider",),
    )
    providers.invalidate_provider_discovery()

    assert providers.get_provider_profile("refresh-provider") is None
    assert "refresh-provider" not in auth.PROVIDER_REGISTRY
    assert all(
        entry.slug != "refresh-provider"
        for entry in models.CANONICAL_PROVIDERS
    )
    assert "refresh-provider" not in models._KNOWN_PROVIDER_NAMES


def test_failed_provider_import_rolls_back_registry_aliases_and_modules(
    provider_home,
):
    plugin_name = "broken-provider"
    plugin_alias = "broken-provider-alias"
    _write_user_provider(
        provider_home,
        plugin_name,
        base_url="https://broken.example/v1",
        aliases=(plugin_alias,),
        fail_after_register=True,
    )
    plugin_dir = provider_home / "plugins" / "model-providers" / plugin_name
    (plugin_dir / "helper.py").write_text("LOADED = True\n", encoding="utf-8")
    init_path = plugin_dir / "__init__.py"
    init_path.write_text(
        "from . import helper\n" + init_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _write_activation(
        provider_home,
        enabled=(f"model-providers/{plugin_name}",),
    )
    import providers

    _clear_provider_caches()
    assert providers.get_provider_profile(plugin_name) is None
    assert providers.get_provider_profile(plugin_alias) is None
    assert plugin_name not in providers._REGISTRY
    assert plugin_alias not in providers._ALIASES
    module_prefix = "_hermes_user_provider_broken_provider"
    assert not any(
        module_name == module_prefix or module_name.startswith(f"{module_prefix}.")
        for module_name in sys.modules
    )


@pytest.mark.parametrize("source", ["user", "project"])
def test_provider_override_drives_auth_and_runtime_metadata(
    source,
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    home.mkdir()
    _write_activation(home, enabled=("model-providers/gmi",))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)

    if source == "user":
        plugin_home = home
    else:
        project_root = tmp_path / "project"
        project_root.mkdir()
        monkeypatch.chdir(project_root)
        monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "1")
        plugin_home = project_root / ".hermes"

    key_var = f"{source.upper()}_GMI_API_KEY"
    base_url_var = f"{source.upper()}_GMI_BASE_URL"
    profile_base_url = f"https://{source}-profile.example/v1"
    runtime_base_url = f"https://{source}-runtime.example/v1"
    _write_user_provider(
        plugin_home,
        "gmi",
        base_url=profile_base_url,
        env_vars=(key_var, base_url_var),
    )
    monkeypatch.setenv(key_var, f"{source}-secret")
    monkeypatch.setenv(base_url_var, runtime_base_url)

    import providers
    from hermes_cli import auth, runtime_provider

    providers.invalidate_provider_discovery()

    assert providers.get_provider_profile("gmi").base_url == profile_base_url
    provider_config = auth.PROVIDER_REGISTRY["gmi"]
    assert provider_config.inference_base_url == profile_base_url
    assert provider_config.api_key_env_vars == (key_var,)
    assert provider_config.base_url_env_var == base_url_var

    credentials = auth.resolve_api_key_provider_credentials("gmi")
    assert credentials["api_key"] == f"{source}-secret"
    assert credentials["base_url"] == runtime_base_url

    runtime = runtime_provider.resolve_runtime_provider(requested="gmi")
    assert runtime["provider"] == "gmi"
    assert runtime["api_key"] == f"{source}-secret"
    assert runtime["base_url"] == runtime_base_url


def test_third_party_provider_alias_follows_activation_refresh(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    home.mkdir()
    provider_name = "alias-lifecycle-provider"
    provider_alias = "alias-lifecycle-shortcut"
    _write_user_provider(
        home,
        provider_name,
        base_url="https://alias-lifecycle.example/v1",
        aliases=(provider_alias,),
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    import providers
    from hermes_cli import auth, models

    def assert_active() -> None:
        assert providers.get_provider_profile(provider_alias).name == provider_name
        assert models.normalize_provider(provider_alias) == provider_name
        assert provider_alias in models._KNOWN_PROVIDER_NAMES
        assert auth.resolve_provider(provider_alias) == provider_name

    _write_activation(
        home,
        enabled=(f"model-providers/{provider_name}",),
    )
    providers.invalidate_provider_discovery()
    assert_active()

    _write_activation(
        home,
        disabled=(f"model-providers/{provider_name}",),
    )
    providers.invalidate_provider_discovery()

    assert providers.get_provider_profile(provider_alias) is None
    assert models.normalize_provider(provider_alias) == provider_alias
    assert provider_alias not in models._KNOWN_PROVIDER_NAMES
    with pytest.raises(auth.AuthError):
        auth.resolve_provider(provider_alias)

    _write_activation(
        home,
        enabled=(f"model-providers/{provider_name}",),
    )
    providers.invalidate_provider_discovery()
    assert_active()


def test_auth_refresh_exception_preserves_lkg_and_empty_initial_state(monkeypatch):
    import providers
    from hermes_cli import auth

    original_registry = dict(auth.PROVIDER_REGISTRY)
    original_dynamic_keys = set(auth._DYNAMIC_PROVIDER_REGISTRY_KEYS)

    def fail_discovery():
        raise RuntimeError("activation refresh failed")

    monkeypatch.setattr(
        providers,
        "get_provider_catalog_snapshot",
        fail_discovery,
    )
    try:
        auth.PROVIDER_REGISTRY.replace({})
        auth._DYNAMIC_PROVIDER_REGISTRY_KEYS.clear()
        auth._refresh_provider_registry_from_plugins()
        assert dict(auth.PROVIDER_REGISTRY) == {}
        assert auth._DYNAMIC_PROVIDER_REGISTRY_KEYS == set()

        lkg_config = auth.ProviderConfig(
            id="last-known-good",
            name="Last Known Good",
            auth_type="api_key",
            inference_base_url="https://lkg.example/v1",
            api_key_env_vars=("LKG_API_KEY",),
        )
        auth.PROVIDER_REGISTRY.replace({"last-known-good": lkg_config})
        auth._DYNAMIC_PROVIDER_REGISTRY_KEYS.clear()
        auth._DYNAMIC_PROVIDER_REGISTRY_KEYS.add("last-known-good")

        auth._refresh_provider_registry_from_plugins()

        assert dict(auth.PROVIDER_REGISTRY) == {"last-known-good": lkg_config}
        assert auth._DYNAMIC_PROVIDER_REGISTRY_KEYS == {"last-known-good"}
    finally:
        auth.PROVIDER_REGISTRY.replace(original_registry)
        auth._DYNAMIC_PROVIDER_REGISTRY_KEYS.clear()
        auth._DYNAMIC_PROVIDER_REGISTRY_KEYS.update(original_dynamic_keys)


def test_auth_activation_check_fails_closed_on_discovery_error(monkeypatch):
    import providers
    from hermes_cli import auth

    def fail_discovery(_provider_id):
        raise RuntimeError("activation state unavailable")

    monkeypatch.setattr(providers, "is_plugin_managed_provider_id", fail_discovery)

    assert auth._provider_plugin_is_active("uncertain-provider") is False


def test_auth_lkg_never_crosses_activation_security_state(monkeypatch):
    """A refresh failure after disable/safe-mode must not revive old routes."""
    import providers
    from hermes_cli import auth, config
    from hermes_cli.plugin_activation import PluginActivationState

    scope = ("profile-home", "")
    enabled = PluginActivationState(
        enabled=frozenset({"model-providers/external"})
    )
    disabled = PluginActivationState(
        disabled=frozenset({"model-providers/external"})
    )
    external = auth.ProviderConfig(
        id="external",
        name="External",
        auth_type="api_key",
        inference_base_url="https://external.example/v1",
        api_key_env_vars=("EXTERNAL_API_KEY",),
    )

    original_lkg = dict(auth._PROVIDER_REGISTRY_LKG_BY_SECURITY_IDENTITY)
    auth._PROVIDER_REGISTRY_LKG_BY_SECURITY_IDENTITY.clear()
    auth._PROVIDER_REGISTRY_LKG_BY_SECURITY_IDENTITY[(scope, enabled)] = {
        "external": external
    }
    monkeypatch.setattr(
        auth,
        "_compute_provider_registry_snapshot",
        lambda: (_ for _ in ()).throw(RuntimeError("discovery failed")),
    )
    monkeypatch.setattr(providers, "get_provider_scope_identity", lambda: scope)

    try:
        monkeypatch.setattr(
            config,
            "load_plugin_activation_state",
            lambda: disabled,
        )
        assert auth._provider_registry_snapshot() == {}

        monkeypatch.setattr(
            config,
            "load_plugin_activation_state",
            lambda: enabled,
        )
        assert auth._provider_registry_snapshot() == {"external": external}
    finally:
        auth._PROVIDER_REGISTRY_LKG_BY_SECURITY_IDENTITY.clear()
        auth._PROVIDER_REGISTRY_LKG_BY_SECURITY_IDENTITY.update(original_lkg)


def test_legacy_single_file_provider_requires_explicit_opt_in(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    home.mkdir()
    legacy_root = tmp_path / "legacy-providers"
    legacy_root.mkdir()
    marker = tmp_path / "legacy-imported.txt"
    module_name = "legacy_opt_in_provider"
    (legacy_root / f"{module_name}.py").write_text(
        "from pathlib import Path\n"
        "from providers import register_provider\n"
        "from providers.base import ProviderProfile\n"
        f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')\n"
        "register_provider(ProviderProfile(\n"
        "    name='legacy-opt-in-provider',\n"
        "    env_vars=('LEGACY_OPT_IN_API_KEY',),\n"
        "    base_url='https://legacy.example/v1',\n"
        "))\n",
        encoding="utf-8",
    )
    _write_activation(home)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import providers

    monkeypatch.setattr(providers, "__path__", [str(legacy_root)])
    _clear_provider_caches()
    assert providers.get_provider_profile("legacy-opt-in-provider") is None
    assert not marker.exists()

    _write_activation(home, enabled=(f"model-providers/{module_name}",))
    providers.invalidate_provider_discovery()
    assert providers.get_provider_profile("legacy-opt-in-provider") is not None
    assert marker.read_text(encoding="utf-8") == "imported"

    _write_activation(home)
    providers.invalidate_provider_discovery()
    assert providers.get_provider_profile("legacy-opt-in-provider") is None
