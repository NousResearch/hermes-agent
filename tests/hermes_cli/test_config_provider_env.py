"""Provider-profile entries of OPTIONAL_ENV_VARS: config-free, same contents as before."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import config_provider_env as cpe
from hermes_cli.config_backend import ConfigBackendUnavailable
from hermes_cli.config_defaults_env import OptionalEnvVars


def _profile(name, *env_vars, auth_type="api_key"):
    return SimpleNamespace(name=name, display_name=name.title(), env_vars=env_vars, auth_type=auth_type,
                           signup_url=f"https://{name}.test/keys")


def test_every_api_key_provider_var_is_declared_in_core():
    import hermes_cli.config as cfg
    from providers import list_providers

    expected = {v for p in list_providers() if p.auth_type == "api_key" for v in p.env_vars}
    assert expected, "no api_key provider profiles registered"
    assert expected <= set(cfg.OPTIONAL_ENV_VARS)
    assert expected <= cfg.CORE_DECLARED_ENV_NAMES


def test_bundled_profiles_match_discovery_and_leave_the_registry_alone():
    import providers

    before = dict(providers._REGISTRY)
    bundled = {p.name: p for p in providers.bundled_profiles()}
    assert dict(providers._REGISTRY) == before
    discovered = {n for n in providers._REGISTRY if providers.provider_source(n) == "bundled"}
    assert discovered and set(bundled) >= discovered
    for name in discovered:
        assert bundled[name].env_vars == providers._REGISTRY[name].env_vars


def test_the_catalog_builds_under_a_backend_that_refuses_every_read(monkeypatch):
    import hermes_cli.config_backend as cb

    reads = []

    def refuse(*a, **k):
        reads.append(a)
        raise ConfigBackendUnavailable("this test's backend refuses every read")

    monkeypatch.setattr(cb, "_route", refuse)
    monkeypatch.setattr(cpe, "_CATALOG", None)
    entries = cpe._provider_entries()
    assert "ANTHROPIC_API_KEY" in entries or len(entries) > 10
    assert reads == []


@pytest.fixture
def table(monkeypatch):
    monkeypatch.setattr(cpe, "_STATIC_NAMES", frozenset({"CORE_KEY"}))
    monkeypatch.setattr(cpe, "_PROVIDER_NAMES", set())
    monkeypatch.setattr(cpe, "_CATALOG", None)
    t = OptionalEnvVars({"CORE_KEY": {"description": "core"}, "PLUG_KEY": {"description": "platform"}})
    t.arm(cpe._provider_entries, cpe._PROVIDER_NAMES.update)
    return t


def test_fill_runs_on_first_read_core_wins_provider_replaces_platform_first_provider_wins(table, monkeypatch):
    import providers

    calls = []
    profiles = [_profile("alpha", "CORE_KEY", "PLUG_KEY", "ALPHA_API_KEY", "ALPHA_BASE_URL"),
                _profile("beta", "ALPHA_API_KEY"), _profile("oauth", "OAUTH_TOKEN", auth_type="oauth")]
    monkeypatch.setattr(providers, "bundled_profiles", lambda: calls.append(1) or profiles)
    dict.__setitem__(table, "LATE", {})  # writes and raw dict access never fill
    assert calls == []
    assert table["CORE_KEY"] == {"description": "core"} and calls == [1]
    assert table["PLUG_KEY"]["description"] == "Alpha API key"
    assert table["ALPHA_API_KEY"]["description"] == "Alpha API key" and table["ALPHA_API_KEY"]["password"]
    assert table["ALPHA_BASE_URL"]["password"] is False and "OAUTH_TOKEN" not in table
    list(table.items())
    assert calls == [1]  # once


def test_discovery_completion_adds_registered_plugin_providers(table, monkeypatch):
    import providers

    from hermes_cli import config_defaults

    monkeypatch.setattr(providers, "bundled_profiles", lambda: [_profile("alpha", "ALPHA_API_KEY")])
    monkeypatch.setattr(config_defaults, "OPTIONAL_ENV_VARS", table)
    monkeypatch.setattr(providers, "_REGISTRY", {"alpha": _profile("alpha", "ALPHA_API_KEY"),
                                                 "plug": _profile("plug", "PLUG_API_KEY", "ALPHA_API_KEY")})
    monkeypatch.setattr(providers, "_SOURCES", {"alpha": "bundled", "plug": "runtime"})
    assert "ALPHA_API_KEY" in table
    cpe.sync_registered_providers()
    assert table["PLUG_API_KEY"]["description"] == "Plug API key"
    assert table["ALPHA_API_KEY"]["description"] == "Alpha API key"  # an earlier entry is kept
    assert "PLUG_API_KEY" in cpe._PROVIDER_NAMES


_ROOT = Path(__file__).resolve().parents[2]


def _cold(script: str, home: Path, *args: str) -> dict:
    """Run *script* in a fresh interpreter (nothing imported, nothing discovered) and return the
    JSON object it prints last."""
    env = {**os.environ, "HERMES_HOME": str(home), "PYTHONPATH": str(_ROOT)}
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(script), *args], cwd=_ROOT, env=env,
                            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def _marker_plugin(root: Path, name: str, marker: Path, env_var: str) -> None:
    """A provider plugin whose import writes *marker*: proof its code ran."""
    plugin = root / name
    plugin.mkdir(parents=True)
    plugin.joinpath("__init__.py").write_text(textwrap.dedent(f"""
        from pathlib import Path
        Path({str(marker)!r}).write_text("imported")
        from providers import register_provider
        from providers.base import ProviderProfile
        register_provider(ProviderProfile(name={name!r}, env_vars=({env_var!r},), auth_type="api_key"))
    """))


def test_cold_discovery_completion_reads_no_config_and_scans_no_home_plugins(tmp_path):
    """Round-5 blocker 1: the completion hook used ``provider_source()``, which scans the bound
    home's unscanned plugin tree (whose isolation check reads config). Under a backend that refuses
    every read it then raised before publishing the registered provider's env var."""
    home = tmp_path / "home"
    marker = tmp_path / "home-plugin-ran"
    _marker_plugin(home / "plugins" / "model-providers", "homeplug", marker, "HOMEPLUG_API_KEY")
    out = _cold("""
        import json, sys
        import hermes_cli.config as cfg
        import hermes_cli.config_backend as cb
        import providers
        from hermes_cli.config_backend import ConfigBackendUnavailable
        from providers.base import ProviderProfile

        reads = []
        def refuse(*a, **k):
            reads.append(repr(a)[:200])
            raise ConfigBackendUnavailable("this backend refuses every read")
        cb._route = refuse
        cfg.OPTIONAL_ENV_VARS.get("")  # the bundled catalog, filled before discovery

        def steps():  # an enabled entry point registering, then nothing else (no config needed)
            providers.register_provider(ProviderProfile(name="epplug", env_vars=("EPPLUG_API_KEY",),
                                                        auth_type="api_key"))
        providers._run_discovery_steps = steps
        hook_errors = []
        import hermes_cli.config_provider_env as cpe
        real = cpe.sync_registered_providers
        def hook():
            try:
                real()
            except BaseException as exc:
                hook_errors.append(repr(exc))
                raise
        cpe.sync_registered_providers = hook
        providers._discover_providers()
        print(json.dumps({"reads": reads, "hook_errors": hook_errors,
                          "ep": "EPPLUG_API_KEY" in cfg.OPTIONAL_ENV_VARS,
                          "home": "HOMEPLUG_API_KEY" in cfg.OPTIONAL_ENV_VARS,
                          "scanned": [k for k in sys.modules if "homeplug" in k]}))
    """, home)
    assert out["hook_errors"] == [] and out["reads"] == []
    assert out["ep"] and not out["home"]
    assert out["scanned"] == [] and not marker.exists()


_SAFE_CATALOG = """
    import json, sys
    from pathlib import Path
    from agent.safe_worker_policy import _bind_safe_worker_policy, _bind_worker_config
    if sys.argv[1] == "safe":
        _bind_safe_worker_policy(safe_mode=True, ignore_user_config=True, config={})
    else:
        _bind_worker_config({})
    import providers
    providers._BUNDLED_PLUGINS_DIR = Path(sys.argv[2])
    import hermes_cli.config as cfg
    present = "MARKPLUG_API_KEY" in cfg.OPTIONAL_ENV_VARS
    print(json.dumps({"present": present,
                      "catalog_modules": [k for k in sys.modules if k.startswith("_hermes_provider_catalog_")]}))
"""


@pytest.mark.parametrize("mode", ["safe", "ordinary"])
def test_cold_safe_worker_catalog_executes_no_plugin_code(tmp_path, mode):
    """Round-5 blocker 2: in a safe worker, building the provider env catalog (``bundled_profiles``)
    executes no plugin module, bundled included. The ordinary-worker run proves the marker plugin
    is really executed when the policy allows it, so the safe assertion is not vacuous."""
    bundled = tmp_path / "bundled"
    marker = tmp_path / "bundled-plugin-ran"
    _marker_plugin(bundled, "markplug", marker, "MARKPLUG_API_KEY")
    out = _cold(_SAFE_CATALOG, tmp_path / "home", mode, str(bundled))
    if mode == "safe":
        assert not marker.exists() and out["catalog_modules"] == [] and not out["present"]
    else:
        assert marker.exists() and out["present"]
