"""Regression coverage for live provider projection during and after discovery.

Plugins may import ``hermes_cli.auth`` while provider discovery is still walking plugin
directories. Later registrations and aliases must remain visible immediately from the
canonical registry without an auth-registry synchronization pass.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import providers
from hermes_cli.provider_auth import get_provider_config
from providers.base import ProviderProfile

REPO_ROOT = Path(__file__).resolve().parents[2]

_PLAIN_PROFILE = (
    "from providers import register_provider\n"
    "from providers.base import ProviderProfile\n"
    "register_provider(ProviderProfile(\n"
    "    name={name!r},\n"
    "    aliases=({alias!r},),\n"
    "    env_vars=(\'{env}\',),\n"
    "    base_url=\'https://{name}.example/v1\',\n"
    "    auth_type=\'api_key\',\n"
    "))\n"
)


def _write_plugin(root: Path, name: str, body: str) -> None:
    plugin_dir = root / "plugins" / "model-providers" / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text(body, encoding="utf-8")
    (plugin_dir / "plugin.yaml").write_text(
        f"name: {name}\nkind: model-provider\nversion: 0.0.1\ndescription: probe\n",
        encoding="utf-8",
    )


def _run_probe(hermes_home: Path, code: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)
    env.pop("HERMES_PROFILE", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_plugins_discovered_after_auth_import_resolve(tmp_path):
    hermes_home = tmp_path / ".hermes"
    _write_plugin(
        hermes_home,
        "aaa-early-probe",
        "import hermes_cli.auth  # simulate a core-importing plugin\n"
        + _PLAIN_PROFILE.format(
            name="aaa-early-probe", alias="aaa-alias", env="AAA_EARLY_PROBE_KEY"
        ),
    )
    _write_plugin(
        hermes_home,
        "zzz-late-probe",
        _PLAIN_PROFILE.format(
            name="zzz-late-probe", alias="zzz-alias", env="ZZZ_LATE_PROBE_KEY"
        ),
    )

    probe = _run_probe(
        hermes_home,
        "import providers\n"
        "names = {p.name for p in providers.list_providers()}\n"
        "assert {\'aaa-early-probe\', \'zzz-late-probe\'} <= names, names\n"
        "from hermes_cli.auth import resolve_provider\n"
        "from hermes_cli.provider_auth import get_provider_config\n"
        "assert get_provider_config(\'zzz-late-probe\') is not None\n"
        "assert resolve_provider(\'aaa-early-probe\') == \'aaa-early-probe\'\n"
        "assert resolve_provider(\'zzz-late-probe\') == \'zzz-late-probe\'\n"
        "assert resolve_provider(\'zzz-alias\') == \'zzz-late-probe\'\n"
        "cfg = get_provider_config(\'zzz-late-probe\')\n"
        "assert cfg.api_key_env_vars == (\'ZZZ_LATE_PROBE_KEY\',), cfg\n"
        "assert cfg.inference_base_url == \'https://zzz-late-probe.example/v1\', cfg\n",
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr


EARLY = "probe-102123-early"
LATE = "probe-102123-late"
LATE_ALIAS = "probe-102123-late-alias"


@pytest.fixture()
def _isolated_registries():
    """Snapshot canonical discovery state and restore it after each test."""
    saved_registry = dict(providers.registry._REGISTRY)
    saved_aliases = dict(providers.registry._ALIASES)
    saved_cache = providers.registry._PROVIDER_LIST_CACHE
    saved_discovered = providers.discovery._discovered
    saved_discovering = providers.discovery._discovering
    saved_source = providers.discovery._current_source
    saved_plugin_modules = {
        name for name in sys.modules if name.startswith("plugins.model_providers")
    }
    providers.registry._REGISTRY.clear()
    providers.registry._ALIASES.clear()
    providers.registry._PROVIDER_LIST_CACHE = None
    providers.discovery._discovered = False
    providers.discovery._discovering = False
    try:
        yield
    finally:
        for name in [
            name
            for name in sys.modules
            if name.startswith("plugins.model_providers") and name not in saved_plugin_modules
        ]:
            del sys.modules[name]
        providers.registry._REGISTRY.clear()
        providers.registry._REGISTRY.update(saved_registry)
        providers.registry._ALIASES.clear()
        providers.registry._ALIASES.update(saved_aliases)
        providers.registry._PROVIDER_LIST_CACHE = saved_cache
        providers.discovery._discovered = saved_discovered
        providers.discovery._discovering = saved_discovering
        providers.discovery._current_source = saved_source


def test_post_discovery_registration_is_live(_isolated_registries, monkeypatch, tmp_path):
    monkeypatch.setattr(providers.discovery, "_discover_entry_point_providers", lambda: None)
    monkeypatch.setattr(providers.discovery, "_BUNDLED_PLUGINS_DIR", tmp_path)
    monkeypatch.setattr(providers.discovery, "_user_plugins_dir", lambda: None)
    monkeypatch.setattr(providers.discovery, "_installed_plugins_dir", lambda: None)
    providers.discovery.ensure_process_discovered()

    providers.register_provider(
        ProviderProfile(
            name=LATE,
            display_name="Late",
            base_url="https://late.example/v1",
            env_vars=("PROBE_102123_LATE_KEY",),
            aliases=(LATE_ALIAS,),
        )
    )
    assert get_provider_config(LATE).id == LATE
    assert get_provider_config(LATE_ALIAS).id == LATE


def test_user_plugin_alias_repoints_and_display_name_follows(
    _isolated_registries, monkeypatch, tmp_path
):
    monkeypatch.setattr(providers.discovery, "_discover_entry_point_providers", lambda: None)
    monkeypatch.setattr(providers.discovery, "_BUNDLED_PLUGINS_DIR", tmp_path)
    monkeypatch.setattr(providers.discovery, "_user_plugins_dir", lambda: None)
    monkeypatch.setattr(providers.discovery, "_installed_plugins_dir", lambda: None)
    providers.discovery.ensure_process_discovered()

    taken_alias = "probe-116668-alias"
    monkeypatch.setattr(providers.discovery, "_current_source", "bundled")
    providers.register_provider(ProviderProfile(
        name="probe-116668-bundled", display_name="Bundled",
        base_url="https://bundled.example/v1",
        env_vars=("PROBE_116668_BUNDLED_KEY",), aliases=(taken_alias,)))
    assert get_provider_config(taken_alias).id == "probe-116668-bundled"

    providers.register_provider(ProviderProfile(
        name="probe-116668-other", display_name="Other",
        base_url="https://other.example/v1",
        env_vars=("PROBE_116668_OTHER_KEY",), aliases=(taken_alias,)))
    assert get_provider_config(taken_alias).id == "probe-116668-other"

    monkeypatch.setattr(providers.discovery, "_current_source", "user")
    providers.register_provider(ProviderProfile(
        name="probe-116668-user", display_name="Mine",
        base_url="https://mine.example/v1",
        env_vars=("PROBE_116668_USER_KEY",), aliases=(taken_alias,)))
    assert get_provider_config(taken_alias).id == "probe-116668-user"

    providers.register_provider(ProviderProfile(
        name="probe-116668-bundled", display_name="Bundled (mine)",
        base_url="https://mine.example/v2",
        env_vars=("PROBE_116668_BUNDLED_KEY",), aliases=(taken_alias,)))
    bundled = get_provider_config("probe-116668-bundled")
    assert bundled.name == "Bundled (mine)"
    assert bundled.inference_base_url == "https://mine.example/v2"
    assert get_provider_config(taken_alias).id == "probe-116668-bundled"
