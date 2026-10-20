"""Gateway startup migrates only the homes it serves."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import secret_scope
from gateway import run as gateway_run
from gateway import run_bootstrap
from gateway.config import GatewayConfig
from hermes_cli import left_core_migration as migration
from hermes_constants import get_hermes_home
from hermes_yaml import safe_load
from tools.terminal_scope import get_terminal_scope


@pytest.fixture
def homes(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    homes = {"default": root, "A": root / "profiles/A", "B": root / "profiles/B"}
    for name, home in homes.items():
        (home / "plugins/homeassistant").mkdir(parents=True)
        (home / "config.yaml").write_text(
            "security:\n  allow_lazy_installs: false\n"
            "toolsets: [homeassistant]\n"
            "platform_toolsets:\n  telegram: [web]\n",
            encoding="utf-8",
        )
        (home / ".env").write_text(f"HASS_TOKEN=synthetic-{name}\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    monkeypatch.setattr(migration, "_attempted", set())
    monkeypatch.setattr(migration, "_undelivered", {})
    active = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(False)
    yield homes
    secret_scope.set_multiplex_active(active)
    from gateway import host_rendezvous
    from gateway.runtime_ownership import process_ownership

    process_ownership.close()
    host_rendezvous.release_host_lock(host_rendezvous.ROLE_GATEWAY)


def _files(home):
    return {
        str(path.relative_to(home)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in home.rglob("*")
        if path.is_file() and "profiles" not in path.relative_to(home).parts
    }


def _assert_migrated(home):
    config = safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    assert config["_left_core_installed"] == ["homeassistant"]
    assert config["_left_core_scoped"] == ["homeassistant"]
    assert config["known_plugin_toolsets"]["telegram"] == ["homeassistant"]


class _ReachedRunner(Exception):
    pass


@pytest.mark.parametrize(("mode", "force"), [
    ("standalone", False), ("standalone", True), ("injected", False),
    ("downgraded", False), ("multiplex", False),
])
def test_bootstrap_migrates_only_served_homes(homes, monkeypatch, mode, force):
    launch = homes["default"] if mode == "multiplex" else homes["A"]
    section = "gateway:\n  multiplex_profiles: true\n"
    if mode == "standalone":
        section += "  standalone: true\n"
    if mode in ("injected", "downgraded"):
        section = "gateway: {}\n"
    config_path = launch / "config.yaml"
    config_path.write_text(config_path.read_text(encoding="utf-8") + section, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(run_bootstrap, "_start_gateway_early_environment_probe", lambda: None)
    monkeypatch.setattr(run_bootstrap, "_start_gateway_configure_logging", lambda verbosity: None)
    monkeypatch.setattr("hermes_cli.resource_limits.apply_nofile_soft_limit", lambda: None)
    seen_modes = []

    def reached_runner(config):
        seen_modes.append(config.multiplex_profiles)
        raise _ReachedRunner

    runner_class = gateway_run.GatewayRunner
    monkeypatch.setattr(gateway_run, "GatewayRunner", reached_runner)
    before = {name: _files(home) for name, home in homes.items()}

    for attempt in range(2):
        migration._attempted.clear()
        config = None
        if mode in ("injected", "downgraded"):
            config = GatewayConfig(multiplex_profiles=mode == "downgraded")
        with pytest.raises(_ReachedRunner):
            asyncio.run(gateway_run.start_gateway(config, force=force))
        _assert_migrated(launch)
        assert get_hermes_home() == launch
        if mode != "multiplex":
            assert _files(homes["default"]) == before["default"]
            assert _files(homes["B"]) == before["B"]
        else:
            receiver = SimpleNamespace(
                _subscribe_plugin_rewire=lambda *a: None,
                _register_config_hooks=lambda *a, **kw: None,
                _snapshot_profile_busy_modes=lambda *a: None,
            )
            for name in ("A", "B"):
                asyncio.run(runner_class._load_secondary_profile_config(receiver, name, homes[name]))
                _assert_migrated(homes[name])
            assert get_hermes_home() == launch
        migrated = {name: _files(home) for name, home in homes.items()}
        if attempt == 0:
            first = migrated
        else:
            for name, files in migrated.items():
                assert files["config.yaml"] == first[name]["config.yaml"]
                assert files[".env"] == before[name][".env"]
    assert seen_modes == ([True, True] if mode == "multiplex" else [False, False])


@pytest.mark.parametrize("failure", ["write", "scope_entry"])
def test_foreign_recovery_restores_caller_scope_after_failure(homes, monkeypatch, failure):
    from gateway.run_startup import recover_left_core_in
    from hermes_cli import config as cli_config
    from hermes_cli import env_loader

    home_a, home_b = homes["A"], homes["B"]
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    before = _files(home_b)
    original_write = cli_config.atomic_config_write
    original_hydrate = env_loader.hydrate_profile_secret_sources

    def write(path, *args, **kwargs):
        if Path(path) == home_b / "config.yaml":
            raise OSError("synthetic write failure")
        return original_write(path, *args, **kwargs)

    def hydrate(home):
        if Path(home) == home_b:
            assert get_hermes_home() == home_b
            raise RuntimeError("synthetic scope failure")
        return original_hydrate(home)

    monkeypatch.setattr(cli_config, "atomic_config_write", write)
    if failure == "scope_entry":
        monkeypatch.setattr(env_loader, "hydrate_profile_secret_sources", hydrate)
    with gateway_run._profile_runtime_scope(home_a):
        terminal = get_terminal_scope()
        if failure == "scope_entry":
            with pytest.raises(RuntimeError, match="synthetic scope failure"):
                recover_left_core_in(home_b)
        else:
            recover_left_core_in(home_b)
        assert get_hermes_home() == home_a
        assert secret_scope.get_secret("HASS_TOKEN") == "synthetic-A"
        assert get_terminal_scope() == terminal
        migration.recover_at_startup()
        _assert_migrated(home_a)
    assert _files(home_b) == before
