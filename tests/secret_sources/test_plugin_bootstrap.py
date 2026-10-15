"""Integration coverage for startup-time SecretSource plugins."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from agent.secret_sources import registry
from agent.secret_sources.base import FetchResult, SecretSource
from hermes_cli import env_loader, plugins


_PLUGIN_SOURCE = """
import json
import sys

from agent.secret_sources.base import FetchResult, SecretSource, run_secret_cli


class RuntimeVaultSource(SecretSource):
    name = "runtime_vault"
    label = "Runtime Vault"
    shape = "mapped"
    scheme = "runtime-vault"

    def fetch(self, cfg, home_path):
        token_env = str(cfg.get("token_env", "PLUGIN_BOOTSTRAP_TOKEN"))
        proc = run_secret_cli(
            [
                sys.executable,
                "-c",
                "import json, os; print(json.dumps(sorted(os.environ)))",
            ],
            allow_env=[token_env],
        )
        child_keys = json.loads(proc.stdout)
        return FetchResult(
            secrets={
                "SHARED_KEY": str(cfg.get("value", "")),
                "PLUGIN_BOOTSTRAP_TOKEN": "must-not-replace-bootstrap",
                "CHILD_ENV_KEYS": ",".join(child_keys),
                "PROFILE_HOME": str(home_path.resolve()),
            }
        )

    def protected_env_vars(self, cfg):
        return frozenset({str(cfg.get("token_env", "PLUGIN_BOOTSTRAP_TOKEN"))})


def register(ctx):
    ctx.register_secret_source(RuntimeVaultSource())
"""


class _BulkPeerSource(SecretSource):
    name = "bulk_peer"
    label = "Bulk Peer"
    shape = "bulk"

    def fetch(self, cfg: dict, home_path: Path) -> FetchResult:
        return FetchResult(
            secrets={
                "SHARED_KEY": "from-bulk",
                "PLUGIN_BOOTSTRAP_TOKEN": "bulk-must-not-replace-bootstrap",
            }
        )


def _install_plugin(home: Path) -> None:
    plugin_dir = home / "plugins" / "runtime-vault"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({
            "name": "runtime-vault",
            "version": "1.0.0",
            "description": "Test runtime secret source",
        }),
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(_PLUGIN_SOURCE, encoding="utf-8")


def _write_config(home: Path, *, value: str, enable_plugin: bool = True) -> None:
    home.mkdir(parents=True, exist_ok=True)
    config = {
        "plugins": {"enabled": ["runtime-vault"] if enable_plugin else []},
        "secrets": {
            "sources": ["bulk_peer", "runtime_vault"],
            "bulk_peer": {"enabled": True},
            "runtime_vault": {
                "enabled": True,
                "override_existing": True,
                "token_env": "PLUGIN_BOOTSTRAP_TOKEN",
                "value": value,
            },
        },
    }
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _reset_plugin_state(monkeypatch, tmp_path: Path) -> None:
    registry._reset_registry_for_tests()
    env_loader._SECRET_SOURCES.clear()
    env_loader.reset_secret_source_cache()
    monkeypatch.setattr(plugins, "_plugin_manager", None)
    monkeypatch.setattr(
        plugins, "get_bundled_plugins_dir", lambda: tmp_path / "bundled"
    )
    monkeypatch.setattr(plugins.PluginManager, "_scan_entry_points", lambda self: [])


def test_first_env_load_discovers_plugin_before_secret_merge(
    tmp_path, monkeypatch, capsys
):
    home = tmp_path / "profile"
    _install_plugin(home)
    _write_config(home, value="from-plugin")
    (home / ".env").write_text(
        "PLUGIN_BOOTSTRAP_TOKEN=bootstrap-token\n"
        "UNRELATED_API_KEY=must-not-reach-child\n",
        encoding="utf-8",
    )
    _reset_plugin_state(monkeypatch, tmp_path)
    registry.register_source(_BulkPeerSource())
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in ("SHARED_KEY", "CHILD_ENV_KEYS", "PROFILE_HOME"):
        monkeypatch.delenv(key, raising=False)

    env_loader.load_hermes_dotenv(hermes_home=home)

    assert os.environ["SHARED_KEY"] == "from-plugin"
    assert env_loader.get_secret_source("SHARED_KEY") == "runtime_vault"
    assert os.environ["PLUGIN_BOOTSTRAP_TOKEN"] == "bootstrap-token"
    child_keys = set(os.environ["CHILD_ENV_KEYS"].split(","))
    assert "PLUGIN_BOOTSTRAP_TOKEN" in child_keys
    assert "UNRELATED_API_KEY" not in child_keys
    assert "kept value from runtime_vault" in capsys.readouterr().err


def test_plugin_discovery_and_apply_are_scoped_per_profile(tmp_path, monkeypatch):
    first = tmp_path / "profiles" / "first"
    second = tmp_path / "profiles" / "second"
    third = tmp_path / "profiles" / "third"
    _write_config(first, value="first", enable_plugin=False)
    _install_plugin(second)
    _write_config(second, value="second")
    _write_config(third, value="third", enable_plugin=False)
    _reset_plugin_state(monkeypatch, tmp_path)
    for key in ("SHARED_KEY", "CHILD_ENV_KEYS", "PROFILE_HOME"):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("HERMES_HOME", str(first))
    env_loader.load_hermes_dotenv(hermes_home=first)
    assert "PROFILE_HOME" not in os.environ

    monkeypatch.setenv("HERMES_HOME", str(second))
    env_loader.load_hermes_dotenv(hermes_home=second)
    assert os.environ["SHARED_KEY"] == "second"
    assert os.environ["PROFILE_HOME"] == str(second.resolve())

    for key in ("SHARED_KEY", "CHILD_ENV_KEYS", "PROFILE_HOME"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(third))
    env_loader.load_hermes_dotenv(hermes_home=third)
    assert "PROFILE_HOME" not in os.environ

    monkeypatch.setenv("HERMES_HOME", str(first))
    env_loader.load_hermes_dotenv(hermes_home=first)
    assert "PROFILE_HOME" not in os.environ
