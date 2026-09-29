"""Docker image swaps reselect the whole active plugin union before boot."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from pm import recovery


def _plugin(home: Path, name: str) -> Path:
    directory = home / "plugins" / name
    directory.mkdir(parents=True)
    (directory / "plugin.yaml").write_text(f"name: {name}\n", encoding="utf-8")
    (directory / "pyproject.toml").write_text(
        f'[project]\nname = "{name}"\nversion = "1.0.0"\n'
        'requires-python = ">=3.11"\ndependencies = []\n'
        '[build-system]\nrequires = ["setuptools"]\n'
        'build-backend = "setuptools.build_meta"\n', encoding="utf-8",
    )
    package = directory / name
    package.mkdir()
    (package / "__init__.py").write_text(f'IDENTITY = "{name}"\n', encoding="utf-8")
    return directory


def test_no_selected_dependencies_stays_on_base(tmp_path, monkeypatch):
    core = tmp_path / "core"
    core.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("memory:\n  provider: builtin\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pm.paths.repo_root", lambda: core)
    monkeypatch.setattr("pm.client.sync_venv", lambda **kw: pytest.fail("unexpected build"))
    assert recovery.refresh_dependencies(core) == "base"


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv required")
def test_cold_image_rebuilds_whole_profile_union_once(tmp_path, monkeypatch):
    uv = shutil.which("uv")
    assert uv is not None
    core = tmp_path / "core"
    core.mkdir()
    (core / "pyproject.toml").write_text(
        '[project]\nname = "fixture-core"\nversion = "1.0.0"\n'
        'requires-python = ">=3.11"\ndependencies = []\n', encoding="utf-8")
    subprocess.run([uv, "lock"], cwd=core, check=True,
                   capture_output=True, timeout=120)
    home = tmp_path / "home"
    home.mkdir()
    _plugin(home, "alpha")
    config = home / "config.yaml"
    config.write_text("memory:\n  provider: alpha\nmodel:\n  default: example/model\n", encoding="utf-8")
    profile = home / "profiles" / "work"
    profile.mkdir(parents=True)
    _plugin(profile, "beta")
    secondary = profile / "config.yaml"
    secondary.write_text("plugins:\n  enabled: [beta]\n", encoding="utf-8")
    original = (config.read_bytes(), secondary.read_bytes())
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("pm.paths.repo_root", lambda: core)
    monkeypatch.setattr("pm.workspace.paths.repo_root", lambda: core)
    monkeypatch.setattr("pm._uv._toolchain", lambda **kw: (Path(uv), Path(sys.executable)))
    from pm import install
    from pm.environments import selected_venv, runtime_facts_path
    from pm.lock import Facts
    monkeypatch.setattr("pm.client.sync_venv", lambda **kw: install.sync_venv(**kw))
    assert not runtime_facts_path(core).exists()
    assert recovery.refresh_dependencies(core) == "rebuilt"
    selected = selected_venv(core)
    fact = Facts(runtime_facts_path(core), strict=True).get("venv")
    assert fact is not None
    assert selected.is_dir() and fact["environment"] == str(selected)
    workspace = Path(fact["resolved_lock"]).parent
    assert "alpha" in (workspace / "pyproject.toml").read_text()
    assert "beta" in (workspace / "pyproject.toml").read_text()
    assert recovery.refresh_dependencies(core) == "current"
    assert (config.read_bytes(), secondary.read_bytes()) == original


def test_broken_secondary_config_aborts_before_partial_union(tmp_path, monkeypatch):
    core = tmp_path / "core"
    core.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    _plugin(home, "alpha")
    (home / "config.yaml").write_text("memory:\n  provider: alpha\n", encoding="utf-8")
    profile = home / "profiles" / "work"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("plugins: [malformed]\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pm.paths.repo_root", lambda: core)
    monkeypatch.setattr("pm.client.sync_venv", lambda **kw: pytest.fail("partial build"))
    with pytest.raises(ValueError, match="config.yaml"):
        recovery.refresh_dependencies(core)
    assert not (home / "installs").exists()


def test_selected_plugin_sync_failure_never_falls_back(tmp_path, monkeypatch):
    core = tmp_path / "core"
    core.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    _plugin(home, "alpha")
    (home / "config.yaml").write_text("memory:\n  provider: alpha\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pm.paths.repo_root", lambda: core)
    monkeypatch.setattr("pm.install.venv_is_current", lambda **kw: False)
    def reject(**kw):
        raise RuntimeError("impossible selection")
    monkeypatch.setattr("pm.client.sync_venv", reject)
    with pytest.raises(RuntimeError, match="impossible selection"):
        recovery.refresh_dependencies(core)
    assert not (home / "installs").exists()
