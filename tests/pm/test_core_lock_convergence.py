"""A core uv.lock bump reaches installs with plugin members (#91424, #91931).

The generation build seeds a plugin-enabled resolve from the previous generation's lock; uv keeps
every seeded pin that still resolves. These run the real ``pm.install.sync_venv`` (real uv,
offline fixture wheels, temp homes) through an update: core's lock moves a transitive and the new
generation must carry core's version, while pins only a plugin brings stay where they were.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import hermes_yaml as yaml

import pm.workspace as ws
from tests.pm import _fixtures

pytestmark = pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")


def _versions(lock: Path, name: str) -> list[str]:
    return [p["version"] for p in tomllib.loads(lock.read_text(encoding="utf-8"))["package"] if p["name"] == name]


@pytest.fixture
def update(tmp_path, monkeypatch):
    """Core needs pkga, which needs pkgb>=1.2,<2; core's lock starts at pkgb 1.2."""
    import pm.install
    import pm.paths

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _fixtures._wheel(wheels, "pkga", "1.0", ["pkgb>=1.2,<2"])
    _fixtures._wheel(wheels, "pkgb", "1.2")
    _fixtures._wheel(wheels, "pkgc", "1.0")
    core = tmp_path / "core"
    core.mkdir()
    (core / "pyproject.toml").write_text(
        '[project]\nname="core-proof"\nversion="1"\nrequires-python=">=3.11"\ndependencies=["pkga==1.0"]\n'
        f'[tool.uv]\npackage=false\nno-index=true\nfind-links=[{json.dumps(wheels.as_posix())}]\n',
        encoding="utf-8")
    uv = shutil.which("uv")
    subprocess.run([uv, "lock", "--offline"], cwd=core, check=True, capture_output=True, text=True, timeout=120)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    monkeypatch.setattr(pm.paths, "repo_root", lambda: core)
    monkeypatch.setattr(ws.paths, "repo_root", lambda: core)
    monkeypatch.setattr(pm.install, "lazy_installs_allowed", lambda: True)
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (Path(uv), Path(sys.executable)))

    def plugin(name: str, requirement: str) -> None:
        directory = home / "plugins" / name
        directory.mkdir(parents=True)
        (directory / "plugin.yaml").write_text(f"name: {name}\n", encoding="utf-8")
        (directory / "pyproject.toml").write_text(
            f'[project]\nname="{name}"\nversion="1"\nrequires-python=">=3.11"\n'
            f'dependencies=[{json.dumps(requirement)}]\n', encoding="utf-8")

    def enable(*names: str) -> None:
        (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": list(names)}}), encoding="utf-8")

    def sync() -> Path:
        from pm.environments import runtime_facts_path
        from pm.lock import Facts

        pm.install.sync_venv(explicit=True, evict_incompatible_plugins=True)
        return Path(Facts(runtime_facts_path(core), strict=True).get("venv")["resolved_lock"])

    def ship(name: str, version: str) -> None:
        """Upstream releases *name* *version* and core's lock picks it up."""
        _fixtures._wheel(wheels, name, version)
        subprocess.run([uv, "lock", "--offline", "-P", name], cwd=core, check=True, capture_output=True,
                       text=True, timeout=120)

    enable()
    return core, home, plugin, enable, sync, ship


def test_core_lock_bump_reaches_a_plugin_generation_and_leaves_plugin_pins(update):
    core, _home, plugin, enable, sync, ship = update
    plugin("unrelated", "pkgc>=1,<2")
    enable("unrelated")
    first = sync()
    assert _versions(first, "pkgb") == ["1.2"] and _versions(first, "pkgc") == ["1.0"]

    ship("pkgb", "1.3")
    # A newer pkgc is on the index too, but no core pin moved it: the plugin's seeded pin stays.
    _fixtures._wheel(core.parent / "wheels", "pkgc", "1.1")
    assert _versions(core / "uv.lock", "pkgb") == ["1.3"]
    second = sync()

    assert second != first
    assert _versions(second, "pkgb") == ["1.3"]
    assert _versions(second, "pkgc") == ["1.0"]


def test_plugin_pinning_the_old_version_is_disabled_and_the_update_completes(update):
    from pm.install import venv_is_current

    core, home, plugin, enable, sync, ship = update
    plugin("unrelated", "pkgc>=1,<2")
    plugin("pinned", "pkgb==1.2")
    enable("unrelated", "pinned")
    assert _versions(sync(), "pkgb") == ["1.2"]

    ship("pkgb", "1.3")
    lock = sync()

    assert _versions(lock, "pkgb") == ["1.3"]
    assert "pinned" not in (lock.parent / "pyproject.toml").read_text(encoding="utf-8")
    plugins = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))["plugins"]
    assert plugins["enabled"] == ["unrelated"] and plugins["disabled"] == ["pinned"]
    from pm.receipt import latest

    warnings = json.dumps(latest().get("warnings"))
    # The resolver's verdict, from the bisect that rebuilds the plugins one by one on core's pins.
    assert "Disabled plugin 'pinned'" in warnings and "no longer builds with it" in warnings
    assert "pkgb" in warnings
    assert venv_is_current(project_root=core) is True


def test_member_less_generation_still_installs_core_lock_frozen(update):
    core, _home, _plugin, _enable, sync, ship = update
    assert _versions(sync(), "pkgb") == ["1.2"]
    ship("pkgb", "1.3")
    lock = sync()
    assert lock.read_bytes() == (core / "uv.lock").read_bytes()


def test_a_bump_to_one_platform_fork_moves_that_fork(tmp_path):
    """core forks a package by platform; bumping only the lower fork must still move it."""
    def lock(path, packages):
        body = "".join(
            f'[[package]]\nname = "{n}"\nversion = "{v}"\nsource = {{ registry = "https://pypi.org/simple" }}\n'
            f'resolution-markers = ["{marker}"]\n\n'
            for n, v, marker in packages)
        path.write_text("version = 1\n\n" + body, encoding="utf-8")
        return path

    win, other = "sys_platform == 'win32'", "sys_platform != 'win32'"
    seed = lock(tmp_path / "seed.lock", [("numpy", "1.26.4", win), ("numpy", "2.1.0", other)])
    core = lock(tmp_path / "core.lock", [("numpy", "1.26.5", win), ("numpy", "2.1.0", other)])

    assert ws.core_lock_upgrades(core, seed) == ["numpy==1.26.5; (sys_platform == 'win32')"]
