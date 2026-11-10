"""Plugin setup runs in the install's committed dependency environment, with its deps prepared first."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import plugins_cmd as cmd

DEP = "setup_fixture_only_dep"

SETUP = f'''from pathlib import Path

def describe(hermes_home):
    return {{"revision": "fixture-v1", "ready": (Path(hermes_home) / "runtime").exists(),
            "summary": "Install fixture runtime", "details": []}}

def run(hermes_home):
    import sys
    import {DEP}
    (Path(hermes_home) / "runtime").write_text({DEP}.VALUE + "\\n" + sys.executable)
'''


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    plugin = home / "plugins" / "env-fixture"
    plugin.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (plugin / "plugin.yaml").write_text("name: env-fixture\nsetup: {entrypoint: setup.py}\n")
    (plugin / "setup.py").write_text(SETUP)
    return home, plugin


def _commit_environment():
    """Commit a PM generation for this checkout (under the temp home) that alone carries DEP."""
    from pm.environments import install_state_dir, runtime_facts_path, site_packages, venv_python
    from pm.paths import repo_root

    root = repo_root()
    venv = install_state_dir(root) / "environments" / "gen1" / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True, timeout=120)
    packages = site_packages(venv)
    packages.mkdir(parents=True, exist_ok=True)
    (packages / f"{DEP}.py").write_text("VALUE = 'from the committed environment'\n")
    runtime_facts_path(root).write_text(json.dumps({"packages": {"venv": {"environment": str(venv)}}}))
    return venv_python(venv)


def _enable(consent=None):
    return cmd.dashboard_set_agent_plugin_enabled("env-fixture", enabled=True, setup_consent=consent,
                                                  _toggle_toolsets=False)


def test_setup_runs_on_the_committed_environment_interpreter(plugin, monkeypatch):
    home, _ = plugin
    with pytest.raises(ImportError):
        __import__(DEP)  # only the committed environment has it
    python = _commit_environment()
    admitted = []
    monkeypatch.setattr(cmd, "_activate_key", lambda key, **kw: admitted.append(key) or True)
    consent = _enable()["consent"]
    result = _enable(consent)
    assert result["ok"], result
    value, executable = (home / "runtime").read_text().splitlines()
    assert value == "from the committed environment"
    assert Path(executable) == python
    assert admitted == ["env-fixture"]


def test_declared_dependencies_are_prepared_before_setup_runs(plugin, monkeypatch):
    import pm.client
    from pm.plugin_inputs import Candidates, Selection

    home, plugin_dir = plugin
    (plugin_dir / "plugin.yaml").write_text(
        f"name: env-fixture\nsetup: {{entrypoint: setup.py}}\npython_dependencies:\n  - {DEP.replace('_', '-')}\n")
    calls = []

    def sync_venv(*args, plugins=None, **kwargs):
        calls.append(type(plugins).__name__)
        if isinstance(plugins, Candidates):
            assert [Path(d) for d in plugins.dirs] == [plugin_dir.absolute()]
            assert not (home / "runtime").exists(), "dependencies prepared after setup ran"
            _commit_environment()  # what PM publishes: a generation carrying the declared dep
        elif not isinstance(plugins, Selection):
            pytest.fail(f"unexpected sync: {plugins!r}")

    monkeypatch.setattr(pm.client, "sync_venv", sync_venv)
    consent = _enable()["consent"]
    assert calls == [], "reviewing setup must not install anything"
    result = _enable(consent)
    assert result["ok"], result
    assert (home / "runtime").read_text().splitlines()[0] == "from the committed environment"
    assert calls == ["Candidates", "Selection"]


def test_failed_dependency_preparation_skips_setup_and_keeps_enablement(plugin, monkeypatch):
    import pm.client

    home, plugin_dir = plugin
    (plugin_dir / "plugin.yaml").write_text(
        "name: env-fixture\nsetup: {entrypoint: setup.py}\npython_dependencies:\n  - missing-fixture-dep\n")

    def refuse(*args, **kwargs):
        raise RuntimeError("resolver refused missing-fixture-dep")

    monkeypatch.setattr(pm.client, "sync_venv", refuse)
    consent = _enable()["consent"]
    result = _enable(consent)
    assert result["status"] == "setup_failed", result
    assert "resolver refused missing-fixture-dep" in result["error"]
    assert "Enablement was not changed" in result["error"]
    assert not (home / "runtime").exists()
    assert "env-fixture" not in cmd._get_enabled_set()
