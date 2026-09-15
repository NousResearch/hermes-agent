"""Install-scoped dependency selection is readable before third-party imports."""
import json
from pathlib import Path

import pytest


def test_install_runtime_selection_is_scoped_and_read_only(tmp_path, monkeypatch):
    from hermes_cli import runtime_paths

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    first, second = tmp_path / "first", tmp_path / "second"
    for root in (first, second):
        (root / ".venv").mkdir(parents=True)
    assert runtime_paths.selected_venv(first) == first / ".venv"
    assert not home.exists()
    state = runtime_paths.install_state_dir(first)
    assert state != runtime_paths.install_state_dir(second)
    generation = state / "environments" / "candidate" / "venv"
    generation.mkdir(parents=True)
    (generation / "pyvenv.cfg").write_text("home = test\n")
    (state / "facts.json").write_text(json.dumps({
        "schema": 1, "packages": {"venv": {"environment": str(generation), "stamp": "verified"}},
    }))
    assert runtime_paths.selected_venv(first) == generation
    assert runtime_paths.selected_venv(second) == second / ".venv"
    monkeypatch.setenv("HERMES_HOME", str(home / "profiles" / "work"))
    assert runtime_paths.selected_venv(first) == generation


def test_boot_uses_one_selected_dependency_tree_in_fresh_process(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    from hermes_cli import runtime_paths

    root = tmp_path / "repo"
    base = root / "venv"
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    state = runtime_paths.install_state_dir(root)
    selected = state / "environments" / "new" / "venv"
    def site_of(venv):
        return venv / ("Lib/site-packages" if os.name == "nt" else f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
    for venv, version in [(base, "old"), (selected, "new")]:
        site = site_of(venv)
        site.mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("home = test")
        (site / "probe_package.py").write_text(f"version = {version!r}")
    (site_of(base) / "base_only.py").write_text("version = 'must-not-leak'")
    (state / "facts.json").write_text(json.dumps({"schema": 1, "packages": {
        "venv": {"environment": str(selected)}
    }}))
    code = (
        "import sys; from pathlib import Path; from hermes_cli.runtime_paths import activate_dependencies; "
        "sys.path.insert(0, sys.argv[2]); activate_dependencies(Path(sys.argv[1])); "
        "import probe_package, importlib.util; print(probe_package.version); "
        "print(importlib.util.find_spec('base_only') is None)"
    )
    process = subprocess.run([sys.executable, "-c", code, str(root), str(site_of(base))],
                             env=dict(os.environ), text=True, capture_output=True, timeout=30)
    assert process.returncode == 0, process.stderr
    assert process.stdout.splitlines() == ["new", "True"]


@pytest.mark.parametrize("command,allowed", [(["pm", "install", "--help"], True), (["pm", "doctor"], True),
    (["-p", "default", "pm", "repair"], True), (["chat"], False), (["chat", "pm", "install"], False)])
def test_broken_environment_keeps_explicit_repair_entry_reachable(tmp_path, monkeypatch, command, allowed):
    import os
    import subprocess
    import sys
    from hermes_cli.runtime_paths import runtime_facts_path

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    repo = Path(__file__).resolve().parents[2]
    record = runtime_facts_path(repo)
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"packages": {"venv": {"environment": str(tmp_path / "missing")}}}))
    code = "import sys; sys.argv = ['hermes', *sys.argv[1:]]; import hermes_bootstrap; print('bootstrap-ready')"
    result = subprocess.run([sys.executable, "-c", code, *command], env=dict(os.environ),
                            capture_output=True, text=True, timeout=30)
    assert (result.returncode == 0) is allowed, result.stderr
    if not allowed:
        assert "hermes pm repair" in result.stderr
        assert "Traceback" not in result.stderr


def test_manual_repair_bypasses_damaged_generation_activation(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    from hermes_cli.runtime_paths import install_state_dir, runtime_facts_path, site_packages

    repo = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    generation = install_state_dir(repo) / "environments" / "damaged"
    environment = generation / "venv"
    site_packages(environment).mkdir(parents=True)
    (environment / "pyvenv.cfg").write_text("home = test", encoding="utf-8")
    (generation / ".lease-managed").touch()
    (generation / ".leases").write_text("not a directory", encoding="utf-8")
    runtime_facts_path(repo).write_text(json.dumps({"schema": 1, "packages": {"venv": {
        "environment": str(environment), "extras": [], "stamp": "old",
    }}}), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(repo)}
    result = subprocess.run([sys.executable, "-S", "-m", "hermes_cli.main", "pm", "repair", "--help"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "hermes pm repair" in result.stdout


@pytest.mark.parametrize("data", [[], {"packages": []}, {"packages": {"venv": []}}])
def test_malformed_selection_has_actionable_error(tmp_path, monkeypatch, data):
    from hermes_cli import runtime_paths
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    record = runtime_paths.runtime_facts_path(tmp_path / "repo")
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="dependency environment"):
        runtime_paths.selected_venv(tmp_path / "repo")


@pytest.mark.parametrize("bad_path", ["outside", "missing"])
def test_invalid_selected_environment_never_silently_falls_back(tmp_path, monkeypatch, bad_path):
    from hermes_cli import runtime_paths

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    root = tmp_path / "repo"
    (root / "venv").mkdir(parents=True)
    state = runtime_paths.install_state_dir(root)
    state.mkdir(parents=True)
    candidate = tmp_path / "outside" if bad_path == "outside" else state / "environments" / "missing"
    if bad_path == "outside":
        candidate.mkdir()
        (candidate / "pyvenv.cfg").write_text("home = test\n")
    (state / "facts.json").write_text(json.dumps({
        "schema": 1, "packages": {"venv": {"environment": str(candidate)}},
    }))
    with pytest.raises(RuntimeError, match="environment"):
        runtime_paths.selected_venv(root)
