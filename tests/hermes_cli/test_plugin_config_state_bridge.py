"""End-to-end coverage for the profile-scoped plugin config/state bridge (#64227)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _context(
    *, name: str = "fixture-plugin", key: str = "", namespace: str = ""
) -> PluginContext:
    return PluginContext(
        PluginManifest(name=name, key=key, skill_namespace=namespace),
        PluginManager(),
    )


def _in_home(home: Path, fn, *args):
    token = set_hermes_home_override(home)
    try:
        return fn(*args)
    finally:
        reset_hermes_home_override(token)


@pytest.fixture
def isolated_home(tmp_path: Path):
    home = tmp_path / "profile"
    token = set_hermes_home_override(home)
    try:
        yield home
    finally:
        reset_hermes_home_override(token)


def test_config_round_trip_uses_canonical_settings_namespace(
    isolated_home: Path,
) -> None:
    ctx = _context(key="category/fixture-plugin")

    assert ctx.get_config("api_url", default="unset") == "unset"
    ctx.set_config("api_url", r"C:\Users\Owner\Hermes 🚀")
    ctx.set_config("retry.policy", {"attempts": 3, "enabled": True})

    raw = yaml.safe_load((isolated_home / "config.yaml").read_text(encoding="utf-8"))
    settings = raw["plugins"]["entries"]["category/fixture-plugin"]["settings"]
    assert settings == {
        "api_url": r"C:\Users\Owner\Hermes 🚀",
        "retry": {"policy": {"attempts": 3, "enabled": True}},
    }
    assert ctx.get_config("api_url") == r"C:\Users\Owner\Hermes 🚀"
    assert ctx.get_config("retry.policy") == {"attempts": 3, "enabled": True}


def test_config_reads_legacy_config_namespace_until_canonical_value_is_set(
    isolated_home: Path,
) -> None:
    path = isolated_home / "config.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        yaml.safe_dump({
            "plugins": {
                "entries": {
                    "fixture-plugin": {
                        "config": {"endpoint": "legacy", "legacy_only": 7},
                        "settings": {"endpoint": "canonical"},
                    }
                }
            }
        }),
        encoding="utf-8",
    )
    ctx = _context()

    assert ctx.get_config("endpoint") == "canonical"
    assert ctx.get_config("legacy_only") == 7

    ctx.set_config("legacy_only", 8)
    assert ctx.get_config("legacy_only") == 8
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    entry = raw["plugins"]["entries"]["fixture-plugin"]
    assert entry["config"]["legacy_only"] == 7
    assert entry["settings"]["legacy_only"] == 8


@pytest.mark.parametrize(
    "key",
    [
        "../../security.approval_mode",
        r"..\..\model.provider",
        "security.approval_mode",
        "model.provider",
        "plugins.entries.other.settings.token",
        "settings.endpoint",
        "",
    ],
)
def test_config_rejects_global_cross_plugin_and_traversal_paths(
    isolated_home: Path, key: str, caplog: pytest.LogCaptureFixture
) -> None:
    ctx = _context()

    with pytest.raises(ValueError, match="plugin-relative config key"):
        ctx.set_config(key, "blocked")

    assert not (isolated_home / "config.yaml").exists()
    assert "Rejected config path" in caplog.text


def test_config_read_rejects_escape_instead_of_exposing_global_config(
    isolated_home: Path,
) -> None:
    path = isolated_home / "config.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("security:\n  approval_mode: always_allow\n", encoding="utf-8")
    ctx = _context()

    with pytest.raises(ValueError, match="plugin-relative config key"):
        ctx.get_config("security.approval_mode")


def test_config_parse_failure_is_not_silently_overwritten(isolated_home: Path) -> None:
    path = isolated_home / "config.yaml"
    path.parent.mkdir(parents=True)
    broken = "plugins:\n  entries: [unterminated\n"
    path.write_text(broken, encoding="utf-8")

    with pytest.raises(Exception, match="while parsing|expected"):
        _context().set_config("endpoint", "safe")

    assert path.read_text(encoding="utf-8") == broken


def test_concurrent_config_writes_do_not_drop_sibling_settings(
    isolated_home: Path,
) -> None:
    ctx = _context()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(
                lambda i: _in_home(isolated_home, ctx.set_config, f"worker_{i}", i),
                range(24),
            )
        )

    assert {f"worker_{i}": ctx.get_config(f"worker_{i}") for i in range(24)} == {
        f"worker_{i}": i for i in range(24)
    }


def test_config_cross_process_lock_preserves_every_setting(isolated_home: Path) -> None:
    script = """
import sys
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
ctx = PluginContext(PluginManifest(name='fixture-plugin'), PluginManager())
for i in range(int(sys.argv[1]), int(sys.argv[2])):
    ctx.set_config(f'process_{i}', i)
"""
    env = dict(os.environ, HERMES_HOME=str(isolated_home))
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(start), str(start + 20)],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
        )
        for start in (0, 20)
    ]
    assert [process.wait(timeout=30) for process in processes] == [0, 0]

    ctx = _context()
    assert {f"process_{i}": ctx.get_config(f"process_{i}") for i in range(40)} == {
        f"process_{i}": i for i in range(40)
    }


def test_state_round_trip_is_atomic_and_aligned_with_plugin_data(
    isolated_home: Path,
) -> None:
    ctx = _context(namespace="agent-plugin-fixture-a1b2c3d4")

    assert ctx.state.get("cursor", default="start") == "start"
    ctx.state.set("cursor", {"page": 2, "path": r"C:\Users\Owner"})

    state_path = (
        isolated_home / "plugin-data" / "agent-plugin-fixture-a1b2c3d4" / "state.json"
    )
    assert ctx.state.data_dir == state_path.parent
    assert json.loads(state_path.read_text(encoding="utf-8")) == {
        "cursor": {"page": 2, "path": r"C:\Users\Owner"}
    }
    assert ctx.state.get("cursor") == {"page": 2, "path": r"C:\Users\Owner"}
    assert not list(state_path.parent.glob("*.tmp"))


def test_native_state_namespace_is_windows_safe_and_cannot_traverse(
    isolated_home: Path,
) -> None:
    contexts = [_context(name="CON"), _context(key="../../other-plugin")]

    for index, ctx in enumerate(contexts):
        ctx.state.set("value", index)
        assert ctx.state.data_dir.parent == isolated_home / "plugin-data"
        assert ctx.state.data_dir.name.startswith("agent-plugin-")
        assert ctx.state.data_dir.name.upper() not in {"CON", "NUL", "COM1", "LPT1"}


def test_concurrent_state_updates_do_not_drop_keys(isolated_home: Path) -> None:
    ctx = _context()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(
                lambda i: _in_home(isolated_home, ctx.state.set, f"cursor_{i}", i),
                range(40),
            )
        )

    state = json.loads(ctx.state.path.read_text(encoding="utf-8"))
    assert state == {f"cursor_{i}": i for i in range(40)}


def test_state_cross_process_lock_preserves_every_update(isolated_home: Path) -> None:
    script = """
import sys
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
ctx = PluginContext(PluginManifest(name='fixture-plugin'), PluginManager())
for i in range(int(sys.argv[1]), int(sys.argv[2])):
    ctx.state.set(f'process_{i}', i)
"""
    env = dict(os.environ, HERMES_HOME=str(isolated_home))
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(start), str(start + 20)],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
        )
        for start in (0, 20)
    ]
    assert [process.wait(timeout=30) for process in processes] == [0, 0]

    state = json.loads(_context().state.path.read_text(encoding="utf-8"))
    assert state == {f"process_{i}": i for i in range(40)}


def test_state_quota_failure_preserves_previous_file(isolated_home: Path) -> None:
    ctx = _context()
    ctx.state.set("cursor", "safe")
    before = ctx.state.path.read_bytes()

    with pytest.raises(ValueError, match="quota"):
        ctx.state.set("oversized", "x" * (ctx.state.quota_bytes + 1))

    assert ctx.state.path.read_bytes() == before
    assert ctx.state.get("cursor") == "safe"


def test_corrupt_state_is_not_silently_overwritten(isolated_home: Path) -> None:
    ctx = _context()
    ctx.state.data_dir.mkdir(parents=True)
    ctx.state.path.write_text('{"cursor":', encoding="utf-8")

    with pytest.raises(RuntimeError, match="Cannot parse plugin state"):
        ctx.state.set("cursor", "replacement")

    assert ctx.state.path.read_text(encoding="utf-8") == '{"cursor":'


def test_config_and_state_follow_context_local_profile_scope(tmp_path: Path) -> None:
    homes = [tmp_path / "profiles" / "alpha", tmp_path / "profiles" / "beta"]
    ctx = _context(namespace="fixture-plugin-data")

    for index, home in enumerate(homes):
        home.mkdir(parents=True)
        token = set_hermes_home_override(home)
        try:
            ctx.set_config("profile_value", index)
            ctx.state.set("profile_value", index)
        finally:
            reset_hermes_home_override(token)

    # One globally-loaded plugin context follows each multiplexed turn's
    # context-local profile instead of pinning the startup profile.
    for index, home in enumerate(homes):
        token = set_hermes_home_override(home)
        try:
            assert ctx.get_config("profile_value") == index
            assert ctx.state.get("profile_value") == index
        finally:
            reset_hermes_home_override(token)


def test_fixture_plugin_round_trips_bridge_during_real_discovery(
    isolated_home: Path,
) -> None:
    plugin_dir = isolated_home / "plugins" / "bridge-fixture"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        "name: bridge-fixture\nversion: 1.0.0\n", encoding="utf-8"
    )
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        "    old = ctx.get_config('loads', default=0)\n"
        "    ctx.set_config('loads', old + 1)\n"
        "    ctx.state.set('registered', {'profile': ctx.profile_name})\n",
        encoding="utf-8",
    )
    (isolated_home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - bridge-fixture\n", encoding="utf-8"
    )

    manager = PluginManager()
    manager.discover_and_load()

    assert manager._plugins["bridge-fixture"].enabled is True
    ctx = _context(name="bridge-fixture")
    assert ctx.get_config("loads") == 1
    assert ctx.state.get("registered")["profile"] in {"custom", "default"}
