"""Tests for the model-providers plugin discovery system.

Verifies that:
 1. All bundled providers at plugins/model-providers/<name>/ are discovered
 2. User plugins at $HERMES_HOME/plugins/model-providers/<name>/ override bundled
 3. plugin.yaml manifests with kind=model-provider are correctly categorized
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest



REPO_ROOT = Path(__file__).resolve().parents[2]
_RESERVED_OPERATIONAL_TEST_ENV = (
    "PATH",
    "HOME",
    "SHELL",
    "SystemRoot",
    "ComSpec",
)


def _clear_provider_caches():
    """Force providers/__init__.py to re-discover on next list_providers()."""
    import providers as _pkg
    _pkg._REGISTRY.clear()
    _pkg._ALIASES.clear()
    _pkg._PROVIDER_LIST_CACHE = None
    _pkg._discovered = False
    _pkg._ACTIVATION_STATE = None
    _pkg._DISCOVERY_FINGERPRINT = None
    _pkg._IMPORTED_PROVIDER_MODULES.clear()
    # Evict any cached plugin modules so the next import re-executes.
    for mod in list(sys.modules.keys()):
        if (
            mod.startswith("plugins.model_providers")
            or mod.startswith("_hermes_user_provider")
            or mod.startswith("_hermes_project_provider")
        ):
            del sys.modules[mod]


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


def test_user_plugin_overrides_bundled(tmp_path, monkeypatch):
    """A user plugin with the same name must override the bundled profile."""
    # Point HERMES_HOME at a fresh temp dir
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # get_hermes_home() may be module-cached depending on codebase; ensure the
    # env var is the source of truth. Most code paths re-read it each call.

    # Drop a user plugin that replaces 'gmi'
    user_gmi = hermes_home / "plugins" / "model-providers" / "gmi"
    user_gmi.mkdir(parents=True)
    (user_gmi / "__init__.py").write_text(
        "from providers import register_provider\n"
        "from providers.base import ProviderProfile\n"
        "\n"
        "custom_gmi = ProviderProfile(\n"
        '    name="gmi",\n'
        '    aliases=("gmi-user-override-test",),\n'
        '    env_vars=("GMI_API_KEY",),\n'
        '    base_url="https://user-override.example.com/v1",\n'
        '    auth_type="api_key",\n'
        ")\n"
        "register_provider(custom_gmi)\n"
    )
    (user_gmi / "plugin.yaml").write_text(
        "name: gmi-user-override\n"
        "kind: model-provider\n"
        "version: 0.0.1\n"
        "description: Test user override\n"
    )
    (hermes_home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - model-providers/gmi\n",
        encoding="utf-8",
    )

    _clear_provider_caches()
    from providers import get_provider_profile

    gmi = get_provider_profile("gmi")
    assert gmi is not None
    assert gmi.base_url == "https://user-override.example.com/v1", (
        f"User override not applied; got base_url={gmi.base_url!r}"
    )
    assert "gmi-user-override-test" in gmi.aliases

    # Clean up: reset discovery state so other tests see the bundled version
    _clear_provider_caches()


    # No import means the module must NOT be in the plugins list as a loaded one.
    # We check that the general loader didn't crash and didn't raise from the
    # broken __init__.py.


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
) -> None:
    plugin_dir = home / "plugins" / "model-providers" / name
    plugin_dir.mkdir(parents=True)
    marker_write = ""
    if marker is not None:
        marker_write = (
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')\n"
        )
    registered_env_vars = env_vars or (env_var,)
    failure = "raise RuntimeError('provider import failed')\n" if fail_after_register else ""
    (plugin_dir / "__init__.py").write_text(
        marker_write
        + "from providers import register_provider\n"
        + "from providers.base import ProviderProfile\n"
        + "register_provider(ProviderProfile(\n"
        + f"    name={name!r},\n"
        + f"    display_name={name!r},\n"
        + f"    aliases={aliases!r},\n"
        + f"    env_vars={registered_env_vars!r},\n"
        + f"    base_url={base_url!r},\n"
        + "    auth_type='api_key',\n"
        + "))\n"
        + failure,
        encoding="utf-8",
    )
    (plugin_dir / "plugin.yaml").write_text(
        f"name: {name}\n"
        "kind: model-provider\n"
        f"provider_ids: [{name}]\n",
        encoding="utf-8",
    )


def _write_activation(home: Path, *, enabled=(), disabled=()) -> None:
    enabled_lines = "".join(f"    - {value}\n" for value in enabled)
    disabled_lines = "".join(f"    - {value}\n" for value in disabled)
    (home / "config.yaml").write_text(
        "plugins:\n"
        "  enabled:\n"
        f"{enabled_lines}"
        "  disabled:\n"
        f"{disabled_lines}",
        encoding="utf-8",
    )


def test_user_provider_is_not_imported_until_explicitly_enabled(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    home.mkdir()
    marker = tmp_path / "provider-imported.txt"
    _write_user_provider(
        home,
        "opt-in-provider",
        base_url="https://opt-in.example/v1",
        marker=marker,
    )
    _write_activation(home)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import providers

    _clear_provider_caches()
    assert providers.get_provider_profile("opt-in-provider") is None
    assert not marker.exists()

    _write_activation(
        home,
        enabled=("model-providers/opt-in-provider",),
    )
    providers.invalidate_provider_discovery()

    assert providers.get_provider_profile("opt-in-provider") is not None
    assert marker.read_text(encoding="utf-8") == "imported"


def test_provider_discovery_refreshes_when_profile_home_changes(
    tmp_path,
    monkeypatch,
):
    first_home = tmp_path / "first"
    second_home = tmp_path / "second"
    first_home.mkdir()
    second_home.mkdir()
    _write_user_provider(
        first_home,
        "profile-provider",
        base_url="https://first.example/v1",
    )
    _write_user_provider(
        second_home,
        "profile-provider",
        base_url="https://second.example/v1",
    )
    for home in (first_home, second_home):
        _write_activation(
            home,
            enabled=("model-providers/profile-provider",),
        )

    import providers

    monkeypatch.setenv("HERMES_HOME", str(first_home))
    providers.invalidate_provider_discovery()
    assert (
        providers.get_provider_profile("profile-provider").base_url
        == "https://first.example/v1"
    )

    # Activation lists are identical; the discovery-root fingerprint must
    # still force a rebuild for the new profile.
    monkeypatch.setenv("HERMES_HOME", str(second_home))
    assert (
        providers.get_provider_profile("profile-provider").base_url
        == "https://second.example/v1"
    )


def test_concurrent_profile_catalogs_do_not_mix_key_and_endpoint(
    tmp_path,
    monkeypatch,
):
    """A profile key must never be paired with another profile's endpoint."""
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    home_a.mkdir()
    home_b.mkdir()
    for home, endpoint, override, secret in (
        (
            home_a,
            "https://a.example/v1",
            "https://a-override.example/v1",
            "secret-a",
        ),
        (
            home_b,
            "https://b.example/v1",
            "https://b-override.example/v1",
            "secret-b",
        ),
    ):
        _write_user_provider(
            home,
            "profile-provider",
            base_url=endpoint,
            env_vars=(
                "PROFILE_PROVIDER_API_KEY",
                "PROFILE_PROVIDER_BASE_URL",
            ),
        )
        _write_activation(
            home,
            enabled=("model-providers/profile-provider",),
        )
        (home / ".env").write_text(
            f"PROFILE_PROVIDER_API_KEY={secret}\n"
            f"PROFILE_PROVIDER_BASE_URL={override}\n",
            encoding="utf-8",
        )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "process-home"))
    import providers
    from hermes_cli import auth
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    a_discovered = threading.Event()
    b_published = threading.Event()

    def run_a():
        token = set_hermes_home_override(home_a)
        try:
            assert (
                providers.get_provider_profile("profile-provider").base_url
                == "https://a.example/v1"
            )
            a_discovered.set()
            assert b_published.wait(timeout=10)
            config = auth.PROVIDER_REGISTRY["profile-provider"]
            credentials = auth.resolve_api_key_provider_credentials(
                "profile-provider"
            )
            return config.inference_base_url, credentials
        finally:
            reset_hermes_home_override(token)

    def run_b():
        assert a_discovered.wait(timeout=10)
        token = set_hermes_home_override(home_b)
        try:
            config = auth.PROVIDER_REGISTRY["profile-provider"]
            credentials = auth.resolve_api_key_provider_credentials(
                "profile-provider"
            )
            return config.inference_base_url, credentials
        finally:
            b_published.set()
            reset_hermes_home_override(token)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(run_a)
        future_b = executor.submit(run_b)
        endpoint_a, credentials_a = future_a.result(timeout=15)
        endpoint_b, credentials_b = future_b.result(timeout=15)

    assert endpoint_a == "https://a.example/v1"
    assert credentials_a["api_key"] == "secret-a"
    assert credentials_a["base_url"] == "https://a-override.example/v1"
    assert endpoint_b == "https://b.example/v1"
    assert credentials_b["api_key"] == "secret-b"
    assert credentials_b["base_url"] == "https://b-override.example/v1"


def test_cached_profile_switch_does_not_repeat_refresh_hooks(
    tmp_path,
    monkeypatch,
):
    """A/B multiplex reads notify once per immutable catalog, not per switch."""
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    for home in (home_a, home_b):
        home.mkdir()
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


def test_inactive_external_manifest_cannot_suppress_static_provider(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    home.mkdir()
    marker = tmp_path / "takeover-imported.txt"
    _write_user_provider(
        home,
        "takeover-openai",
        base_url="https://takeover.invalid/v1",
        marker=marker,
    )
    manifest = home / "plugins" / "model-providers" / "takeover-openai" / "plugin.yaml"
    manifest.write_text(
        "name: takeover-openai\n"
        "kind: model-provider\n"
        "provider_ids: [openai-api]\n",
        encoding="utf-8",
    )
    _write_activation(home)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import providers
    from hermes_cli import auth

    providers.invalidate_provider_discovery()
    assert not marker.exists()
    assert "openai-api" in auth.PROVIDER_REGISTRY
    assert (
        auth.PROVIDER_REGISTRY["openai-api"].inference_base_url
        == "https://api.openai.com/v1"
    )


def test_observed_provider_secret_names_remain_blocked_after_disable(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    home.mkdir()
    _write_user_provider(
        home,
        "secret-provider",
        base_url="https://secret.example/v1",
        env_vars=("SECRET_PROVIDER_API_KEY", "SECRET_PROVIDER_BASE_URL"),
    )
    _write_activation(
        home,
        enabled=("model-providers/secret-provider",),
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    import providers
    from tools.environments.local import _HERMES_PROVIDER_ENV_BLOCKLIST

    providers.invalidate_provider_discovery()
    assert "SECRET_PROVIDER_API_KEY" in _HERMES_PROVIDER_ENV_BLOCKLIST
    assert "SECRET_PROVIDER_BASE_URL" in _HERMES_PROVIDER_ENV_BLOCKLIST

    _write_activation(
        home,
        disabled=("model-providers/secret-provider",),
    )
    providers.invalidate_provider_discovery()
    assert "SECRET_PROVIDER_API_KEY" in _HERMES_PROVIDER_ENV_BLOCKLIST
    assert "SECRET_PROVIDER_BASE_URL" in _HERMES_PROVIDER_ENV_BLOCKLIST


@pytest.mark.parametrize(
    ("activation", "requires_env"),
    (
        (
            "inactive",
            "  - PATH\n"
            "  - HOME\n"
            "  - SHELL\n"
            "  - SystemRoot\n"
            "  - ComSpec\n"
            "  - LC_VENDOR_ACCESS\n",
        ),
        (
            "disabled",
            "  - name: PATH\n"
            "  - name: HOME\n"
            "  - name: SHELL\n"
            "  - name: SystemRoot\n"
            "  - name: ComSpec\n"
            "  - name: LC_VENDOR_ACCESS\n"
            "    description: Vendor API credential\n"
            "    secret: true\n",
        ),
    ),
)
def test_inactive_manifest_credentials_are_blocked_after_process_restart(
    activation,
    requires_env,
    tmp_path,
):
    """A fresh process must scrub inactive credentials without importing code."""
    home = tmp_path / "home"
    home.mkdir()
    marker = tmp_path / "inactive-provider-imported.txt"
    provider_name = "inactive-secret-provider"
    _write_user_provider(
        home,
        provider_name,
        base_url="https://inactive.example/v1",
        env_var="LC_VENDOR_ACCESS",
        marker=marker,
    )
    manifest = home / "plugins" / "model-providers" / provider_name / "plugin.yaml"
    manifest.write_text(
        f"name: {provider_name}\n"
        "kind: model-provider\n"
        f"provider_ids: [{provider_name}]\n"
        "requires_env:\n"
        f"{requires_env}",
        encoding="utf-8",
    )
    _write_activation(
        home,
        disabled=(f"model-providers/{provider_name}",)
        if activation == "disabled"
        else (),
    )

    result = _probe_provider_security_in_fresh_process(home, provider_name)

    assert result == {
        "active": False,
        "observed": True,
        "observed_reserved": False,
        "profile_secret": False,
        "profile_reserved": False,
        "in_child": False,
        "in_code_child": False,
        "reserved_in_child": True,
        "reserved_in_code_child": True,
    }
    assert not marker.exists()


def _probe_provider_security_in_fresh_process(
    home: Path,
    provider_name: str,
) -> dict[str, bool]:
    reserved_env = {
        name: f"test-{name.lower()}"
        for name in _RESERVED_OPERATIONAL_TEST_ENV
    }
    probe = "\n".join(
        (
            "import json",
            "import providers",
            "from tools.code_execution_tool import _scrub_child_env",
            "from tools.environments.local import _sanitize_subprocess_env",
            f"reserved_env = {reserved_env!r}",
            f"profile = providers.get_provider_profile({provider_name!r})",
            "source_env = {**reserved_env, 'LC_VENDOR_ACCESS': 'test-secret'}",
            "sanitized = _sanitize_subprocess_env(source_env)",
            "scrubbed = _scrub_child_env(",
            "    source_env,",
            "    is_passthrough=lambda _name: True,",
            "    is_windows=True,",
            ")",
            "observed = providers.get_observed_provider_env_vars()",
            "print(json.dumps({",
            "    'active': profile is not None,",
            "    'observed': 'LC_VENDOR_ACCESS' in observed,",
            "    'observed_reserved': any(name in observed for name in reserved_env),",
            "    'profile_secret': bool(profile and 'LC_VENDOR_ACCESS' in profile.env_vars),",
            "    'profile_reserved': bool(profile and any(name in profile.env_vars for name in reserved_env)),",
            "    'in_child': 'LC_VENDOR_ACCESS' in sanitized,",
            "    'in_code_child': 'LC_VENDOR_ACCESS' in scrubbed,",
            "    'reserved_in_child': all(name in sanitized for name in reserved_env),",
            "    'reserved_in_code_child': all(name in scrubbed for name in reserved_env),",
            "}))",
        )
    )
    child_env = os.environ.copy()
    child_env["HERMES_HOME"] = str(home)
    child_env["LC_VENDOR_ACCESS"] = "test-secret"
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        env=child_env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_legacy_provider_observed_names_persist_across_disabled_restart(
    tmp_path,
):
    """Old manifests gain exact name-only protection after one active load."""
    home = tmp_path / "home"
    home.mkdir()
    marker = tmp_path / "legacy-provider-imported.txt"
    provider_name = "legacy-secret-provider"
    _write_user_provider(
        home,
        provider_name,
        base_url="https://legacy-secret.example/v1",
        env_vars=("LC_VENDOR_ACCESS", *_RESERVED_OPERATIONAL_TEST_ENV),
        marker=marker,
    )
    _write_activation(home, enabled=(f"model-providers/{provider_name}",))

    first = _probe_provider_security_in_fresh_process(home, provider_name)
    assert first == {
        "active": True,
        "observed": True,
        "observed_reserved": False,
        "profile_secret": True,
        "profile_reserved": False,
        "in_child": False,
        "in_code_child": False,
        "reserved_in_child": True,
        "reserved_in_code_child": True,
    }
    assert marker.exists()

    cache_files = list((home / "cache" / "provider-env-names").glob("*.json"))
    assert len(cache_files) == 1
    cache_text = cache_files[0].read_text(encoding="utf-8")
    assert "LC_VENDOR_ACCESS" in cache_text
    assert "test-secret" not in cache_text
    assert all(name not in cache_text for name in _RESERVED_OPERATIONAL_TEST_ENV)

    # A cache written by an older process may already contain poisoned names.
    # Cache parsing must re-apply the policy before publishing observations.
    cache_data = json.loads(cache_text)
    cache_data["env_vars"].extend(_RESERVED_OPERATIONAL_TEST_ENV)
    cache_files[0].write_text(json.dumps(cache_data), encoding="utf-8")

    marker.unlink()
    _write_activation(home, disabled=(f"model-providers/{provider_name}",))
    second = _probe_provider_security_in_fresh_process(home, provider_name)
    assert second == {
        "active": False,
        "observed": True,
        "observed_reserved": False,
        "profile_secret": False,
        "profile_reserved": False,
        "in_child": False,
        "in_code_child": False,
        "reserved_in_child": True,
        "reserved_in_code_child": True,
    }
    assert not marker.exists()
    healed_cache_text = cache_files[0].read_text(encoding="utf-8")
    assert "LC_VENDOR_ACCESS" in healed_cache_text
    assert all(
        name not in healed_cache_text
        for name in _RESERVED_OPERATIONAL_TEST_ENV
    )

    plugin_dir = home / "plugins" / "model-providers" / provider_name
    plugin_dir.rename(plugin_dir.with_name(f".{provider_name}-removed"))
    removed = _probe_provider_security_in_fresh_process(home, provider_name)
    assert removed == {
        "active": False,
        "observed": False,
        "observed_reserved": False,
        "profile_secret": False,
        "profile_reserved": False,
        "in_child": True,
        "in_code_child": True,
        "reserved_in_child": True,
        "reserved_in_code_child": True,
    }


def test_reserved_provider_names_are_exempt_after_in_memory_pollution():
    """Long-lived monotonic indexes must self-filter old poisoned names."""
    import providers
    from tools.code_execution_tool import (
        _WINDOWS_ESSENTIAL_ENV_VARS,
        _scrub_child_env,
    )
    from tools.environments import local

    secret_name = "LC_VENDOR_ACCESS"
    assert all(
        providers.is_reserved_provider_env_var(name)
        for name in (
            *_RESERVED_OPERATIONAL_TEST_ENV,
            *_WINDOWS_ESSENTIAL_ENV_VARS,
        )
    )
    assert not providers.is_reserved_provider_env_var(secret_name)

    injected_names = (*_RESERVED_OPERATIONAL_TEST_ENV, secret_name)
    with providers._DISCOVERY_LOCK:
        previously_observed = {
            name: name in providers._OBSERVED_PROVIDER_ENV_VARS
            for name in injected_names
        }
        providers._OBSERVED_PROVIDER_ENV_VARS.update(injected_names)
    with local._HERMES_PROVIDER_ENV_BLOCKLIST_LOCK:
        previously_blocked = {
            name: name in local._HERMES_PROVIDER_ENV_BLOCKLIST
            for name in injected_names
        }
        local._HERMES_PROVIDER_ENV_BLOCKLIST.update(injected_names)

    source_env = {
        **{
            name: f"test-{name.lower()}"
            for name in _RESERVED_OPERATIONAL_TEST_ENV
        },
        secret_name: "test-secret",
    }
    try:
        observed = providers.get_observed_provider_env_vars()
        assert secret_name in observed
        for name in _RESERVED_OPERATIONAL_TEST_ENV:
            assert name not in observed
            assert not providers.is_observed_provider_env_var(name)
            assert not local._is_provider_env_blocked(name)
        assert local._is_provider_env_blocked(secret_name)

        sanitized = local._sanitize_subprocess_env(source_env)
        scrubbed = _scrub_child_env(
            source_env,
            is_passthrough=lambda _name: True,
            is_windows=True,
        )
        assert secret_name not in sanitized
        assert secret_name not in scrubbed
        assert all(name in sanitized for name in _RESERVED_OPERATIONAL_TEST_ENV)
        assert all(name in scrubbed for name in _RESERVED_OPERATIONAL_TEST_ENV)
    finally:
        with providers._DISCOVERY_LOCK:
            for name, was_present in previously_observed.items():
                if not was_present:
                    providers._OBSERVED_PROVIDER_ENV_VARS.discard(name)
        with local._HERMES_PROVIDER_ENV_BLOCKLIST_LOCK:
            for name, was_present in previously_blocked.items():
                if was_present:
                    local._HERMES_PROVIDER_ENV_BLOCKLIST.add(name)
                else:
                    local._HERMES_PROVIDER_ENV_BLOCKLIST.discard(name)


def test_provider_secret_names_are_case_insensitive_on_windows(monkeypatch):
    """Mixed-case provider metadata must scrub any Windows key spelling."""
    import providers
    from tools.code_execution_tool import _scrub_child_env
    from tools.environments import local

    declared_name = "Lc_Vendor_Access"
    actual_name = "LC_VENDOR_ACCESS"
    monkeypatch.setattr(local, "_IS_WINDOWS", True)

    with providers._DISCOVERY_LOCK:
        previously_observed = declared_name in providers._OBSERVED_PROVIDER_ENV_VARS
        providers._OBSERVED_PROVIDER_ENV_VARS.add(declared_name)
    with local._HERMES_PROVIDER_ENV_BLOCKLIST_LOCK:
        previous_blocklist = set(local._HERMES_PROVIDER_ENV_BLOCKLIST)
        local._HERMES_PROVIDER_ENV_BLOCKLIST.add(declared_name)

    try:
        source_env = {actual_name: "test-secret"}
        assert local._is_provider_env_blocked(actual_name)
        assert actual_name not in local._sanitize_subprocess_env(source_env)
        assert actual_name not in _scrub_child_env(
            source_env,
            is_passthrough=lambda _name: True,
            is_windows=True,
        )
    finally:
        with providers._DISCOVERY_LOCK:
            if not previously_observed:
                providers._OBSERVED_PROVIDER_ENV_VARS.discard(declared_name)
        with local._HERMES_PROVIDER_ENV_BLOCKLIST_LOCK:
            local._HERMES_PROVIDER_ENV_BLOCKLIST.clear()
            local._HERMES_PROVIDER_ENV_BLOCKLIST.update(previous_blocklist)


def test_new_provider_secret_is_blocked_before_refresh_hooks_finish(
    tmp_path,
    monkeypatch,
):
    """Spawn-time checks must not depend on the later refresh-hook callback."""
    home = tmp_path / "home"
    home.mkdir()
    secret_name = "TOCTOU_PROVIDER_API_KEY"
    _write_user_provider(
        home,
        "toctou-provider",
        base_url="https://toctou.example/v1",
        env_var=secret_name,
    )
    _write_activation(
        home,
        enabled=("model-providers/toctou-provider",),
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    import providers
    from tools.code_execution_tool import _scrub_child_env
    from tools.env_passthrough import (
        clear_env_passthrough,
        is_env_passthrough,
        register_env_passthrough,
    )
    from tools.environments.local import (
        _HERMES_PROVIDER_ENV_BLOCKLIST,
        _HERMES_PROVIDER_ENV_BLOCKLIST_LOCK,
        _sanitize_subprocess_env,
    )

    # The skill arrives first, while the provider has not been discovered.
    clear_env_passthrough()
    with _HERMES_PROVIDER_ENV_BLOCKLIST_LOCK:
        _HERMES_PROVIDER_ENV_BLOCKLIST.discard(secret_name)
    register_env_passthrough([secret_name])
    assert is_env_passthrough(secret_name)

    callback_started = threading.Event()
    release_callback = threading.Event()
    errors: list[BaseException] = []

    def block_first_refresh_hook():
        callback_started.set()
        assert release_callback.wait(timeout=10)

    monkeypatch.setattr(
        providers,
        "_PROVIDER_REFRESH_HOOKS",
        [block_first_refresh_hook, *providers._PROVIDER_REFRESH_HOOKS],
    )

    def discover_provider():
        try:
            assert providers.get_provider_profile("toctou-provider") is not None
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=discover_provider, daemon=True)
    worker.start()
    try:
        assert callback_started.wait(timeout=10)
        # The ordinary blocklist callback is deliberately still queued.
        with _HERMES_PROVIDER_ENV_BLOCKLIST_LOCK:
            assert secret_name not in _HERMES_PROVIDER_ENV_BLOCKLIST

        # Provider-owned observed metadata plus execution-time refresh closes
        # the window and revokes the earlier passthrough entry immediately.
        assert not is_env_passthrough(secret_name)
        assert secret_name not in _sanitize_subprocess_env(
            {secret_name: "secret", "PATH": "/usr/bin"}
        )
        assert secret_name not in _scrub_child_env(
            {secret_name: "secret", "PATH": "/usr/bin"},
            is_passthrough=lambda _name: True,
            is_windows=False,
        )
    finally:
        release_callback.set()
        worker.join(timeout=10)
        clear_env_passthrough()

    assert not worker.is_alive()
    assert errors == []

def test_disabling_provider_refreshes_all_derived_surfaces(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    home.mkdir()
    _write_user_provider(
        home,
        "refresh-provider",
        base_url="https://refresh.example/v1",
        env_var="REFRESH_PROVIDER_API_KEY",
    )
    _write_activation(
        home,
        enabled=("model-providers/refresh-provider",),
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    import providers
    from hermes_cli import auth, config, models

    providers.invalidate_provider_discovery()
    assert "refresh-provider" in auth.PROVIDER_REGISTRY
    assert any(
        entry.slug == "refresh-provider"
        for entry in models.CANONICAL_PROVIDERS
    )
    assert "refresh-provider" in models._KNOWN_PROVIDER_NAMES
    assert "REFRESH_PROVIDER_API_KEY" in config.OPTIONAL_ENV_VARS

    _write_activation(
        home,
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
    # Observed secret names remain in the process-wide metadata/blocklist even
    # after this profile disables the plugin.  Routability is removed above;
    # retaining the name prevents a concurrent profile's key from becoming
    # inheritable by terminal subprocesses.
    assert "REFRESH_PROVIDER_API_KEY" in config.OPTIONAL_ENV_VARS


def test_failed_provider_import_rolls_back_registry_aliases_and_modules(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    home.mkdir()
    plugin_name = "broken-provider"
    plugin_alias = "broken-provider-alias"
    _write_user_provider(
        home,
        plugin_name,
        base_url="https://broken.example/v1",
        aliases=(plugin_alias,),
        fail_after_register=True,
    )
    plugin_dir = home / "plugins" / "model-providers" / plugin_name
    (plugin_dir / "helper.py").write_text("LOADED = True\n", encoding="utf-8")
    init_path = plugin_dir / "__init__.py"
    init_path.write_text(
        "from . import helper\n" + init_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _write_activation(
        home,
        enabled=(f"model-providers/{plugin_name}",),
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

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
