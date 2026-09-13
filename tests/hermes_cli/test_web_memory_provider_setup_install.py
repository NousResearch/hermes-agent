"""CLI/dashboard setup resolves declarations, not ambient importability."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from hermes_cli.web_routers import memory_providers as mp


@pytest.mark.parametrize("surface", ["dashboard", "cli"])
@pytest.mark.parametrize("declaration", ["pyproject", "python_dependencies", "pip_dependencies"])
def test_setup_admits_real_provider_union_and_keeps_selection_on_failure(tmp_path, monkeypatch, surface, declaration):
    import pm
    from hermes_constants import venv_python_path
    from hermes_cli.runtime_paths import selected_venv
    from tests.pm._fixtures import _wheel
    from hermes_cli import memory_setup
    from hermes_cli.web_server_memory import _memory_provider_setup_info

    uv = shutil.which("uv")
    assert uv, "real PM admission test requires uv"
    core, home, wheels = (tmp_path / name for name in ("core", "home", "wheels"))
    for directory in (core, home, wheels):
        directory.mkdir()
    for name in ("existing_dep", "provider_dep"):
        _wheel(wheels, name, "1.0")
    (core / "pyproject.toml").write_text(
        '[project]\nname="core"\nversion="1"\nrequires-python=">=3.14"\n'
        '[tool.uv]\npackage=false\nno-index=true\n'
        f'find-links=[{json.dumps(wheels.as_posix())}]\n')
    incumbent, candidate = (home / "plugins" / name for name in ("incumbent", "candidate"))
    for directory in (incumbent, candidate):
        directory.mkdir(parents=True)
    (incumbent / "plugin.yaml").write_text('name: incumbent\npython_dependencies: ["existing_dep==1.0"]\n')
    manifest = candidate / "plugin.yaml"
    def declare(requirements):
        if declaration == "pyproject":
            manifest.write_text('name: candidate\n')
            (candidate / "pyproject.toml").write_text(
                '[project]\nname="candidate"\nversion="1"\nrequires-python=">=3.14"\n'
                f'dependencies={json.dumps(requirements)}\n[tool.uv]\npackage=false\n')
        else:
            manifest.write_text(f'name: candidate\n{declaration}: {json.dumps(requirements)}\n')
    declare(["provider_dep==1.0", "provider_dep==2.0"])
    config = home / "config.yaml"
    config.write_text('plugins:\n  enabled: [incumbent]\n')
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    monkeypatch.setattr("pm.paths.repo_root", lambda: core)
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (Path(uv), Path(sys.executable)))
    monkeypatch.setattr("pm.client.is_runtime", lambda: True)
    monkeypatch.setattr("plugins.memory.find_provider_dir", lambda name: candidate)
    # An importable ambient module must not bypass admission or pin checks.
    (tmp_path / "provider_dep.py").write_text('VALUE="ambient, not admitted"\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "provider_dep", raising=False)
    import provider_dep
    assert provider_dep.VALUE == "ambient, not admitted"
    def prepare():
        if surface == "dashboard":
            return mp._install_memory_provider_python_dependencies("candidate")
        memory_setup._install_dependencies("candidate")
    pm.lock_project(core, explicit=True, offline=True)
    pm.sync_venv(explicit=True)
    original = selected_venv(core)
    original_config = config.read_bytes()
    before_listing = set(home.rglob("*"))
    info = _memory_provider_setup_info("candidate")
    assert info["python_dependencies_declared"] is True
    assert info["dependencies_installed"] is False
    assert set(home.rglob("*")) == before_listing, "listing must not prepare dependencies"

    if surface == "dashboard":
        failed = prepare()
        assert failed[0]["status"] == "failed", failed
    else:
        with pytest.raises(pm.InstallError):
            prepare()
    assert selected_venv(core) == original
    assert config.read_bytes() == original_config

    declare(["provider_dep==1.0"])
    success = prepare()
    if surface == "dashboard":
        assert success[0]["status"] == "restart_required", success
    python = venv_python_path(selected_venv(core))
    result = subprocess.run([str(python), "-I", "-c", "import existing_dep, provider_dep; print('both')"],
                            check=True, capture_output=True, text=True, timeout=30)
    assert result.stdout.strip() == "both"
    assert config.read_bytes() == original_config
    # Preparing an already-current union must not produce another generation.
    selected = selected_venv(core)
    prepare()
    assert selected_venv(core) == selected
    assert _memory_provider_setup_info("candidate")["dependencies_installed"] is True
    declare(["provider_dep==2.0"])
    assert _memory_provider_setup_info("candidate")["dependencies_installed"] is False
    assert selected_venv(core) == selected


@pytest.mark.parametrize("python_failure", [False, True])
def test_setup_reports_restart_and_preserves_external_steps(tmp_path, monkeypatch, python_failure):
    import shlex

    provider = tmp_path / "provider"
    provider.mkdir()
    (provider / "plugin.yaml").write_text(json.dumps({
        "name": "provider", "extra": "mem0", "external_dependencies": [
            {"name": "external-sidecar", "check": f'{shlex.quote(sys.executable)} -c "pass"'},
        ],
    }))
    monkeypatch.setattr("plugins.memory.find_provider_dir", lambda name: provider)
    monkeypatch.setattr(mp, "_load_memory_provider", lambda name: None)
    monkeypatch.setattr(mp, "_discover_memory_provider_statuses", lambda: [])
    # The resolver seam is isolated; real union behavior is exercised above.
    def sync(*args, **kwargs):
        if python_failure:
            raise RuntimeError("Python preparation refused")
    monkeypatch.setattr("pm.sync_venv", sync)
    result = mp._install_memory_provider_setup("provider")
    assert result["ok"] is not python_failure
    assert result["results"][0]["status"] == ("failed" if python_failure else "restart_required")
    assert result["results"][1]["name"] == "external-sidecar"
    assert result["results"][1]["status"] == "already_installed"
    assert all(row["status"] != "no_declared_steps" for row in result["results"])


def test_malformed_candidate_manifest_is_not_a_successful_noop(tmp_path, monkeypatch):
    (tmp_path / "plugin.yaml").write_text('name: [unterminated\n')
    monkeypatch.setattr("plugins.memory.find_provider_dir", lambda name: tmp_path)
    [row] = mp._install_memory_provider_python_dependencies("broken")
    assert row["status"] == "failed"