"""Plugin publication runs in PM's independent worker, not application callbacks."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.pm.test_worker import client, isolated_python, _current_environment  # noqa: F401
from tests.pm._fixtures import worker_toolchain


def test_worker_publishes_selection_even_when_dependencies_are_current(client, tmp_path, monkeypatch):
    from pm.environments import install_state_dir
    from pm import receipt

    home = tmp_path / "home"
    home.mkdir()
    config = home / "config.yaml"
    config.write_text('# keep my comment\nmodel: "untouched"\nplugins:\n  enabled: [old]\n')
    repo = _current_environment(tmp_path, monkeypatch, [])
    facts = (install_state_dir(repo) / "facts.json").read_bytes()
    with receipt.worker_context("selection-publication"):
        client.sync_venv(explicit=True, selection={
            "home": str(home), "enabled": ["plain"], "disabled": ["old"], "extra_dirs": [],
        })
        result = receipt.last_for_update("selection-publication", consume=True)
    from utils import fast_safe_load
    assert fast_safe_load(config.read_text())["plugins"] == {"enabled": ["plain"], "disabled": ["old"]}
    assert '# keep my comment' in config.read_text()
    assert 'model: "untouched"' in config.read_text()
    assert (install_state_dir(repo) / "facts.json").read_bytes() == facts
    assert not (install_state_dir(repo) / "publication.json").exists()
    assert result is not None
    assert result["outcome"] == "ok"
    assert result["venv_rebuild"]["ok"] is False
    assert json.loads((home / "logs/update_receipts/latest.json").read_text())["outcome"] == "ok"



@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("missing", [False, True], ids=["existing", "missing"])
def test_staged_plugin_publication_uses_installed_identity_and_local_dependencies(
    client, tmp_path, monkeypatch, isolated_python, active, missing,
):
    from pm.environments import install_state_dir, selected_venv
    from pm.store import tree_digest
    from tests.pm.test_environment_build import _wheel
    from pm import paths

    worker_toolchain(client, monkeypatch, isolated_python)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(paths, "repo_root", lambda: project)
    (project / "pyproject.toml").write_text(
        '[project]\nname="publication-core"\nversion="1"\nrequires-python=">=3.11"\n'
        '[tool.uv]\npackage=false\n')
    client.lock_project(project, offline=True, explicit=True)
    home = tmp_path / "home"
    target = home / "plugins" / "example"
    target.mkdir(parents=True)
    (home / "config.yaml").write_text('plugins:\n  enabled: ' + ('[example]' if active else '[]') + '\n')
    (target / "plugin.yaml").write_text("name: example\n")
    (target / "code.py").write_text('old code')
    metadata = target.parent / ".install-metadata.json"
    metadata.write_text('{"example":{"revision":"old"}}\n')
    client.sync_venv(explicit=True)
    previous = selected_venv(project)
    if missing:
        shutil.rmtree(target)
    staged = tmp_path / "staged"
    staged.mkdir()
    wheel = _wheel(tmp_path, "publication_dep")
    (staged / "plugin.yaml").write_text('name: example\npython_dependencies: ["publication-dep==1.0"]\n')
    metadata_path = project / "pyproject.toml"
    with metadata_path.open("a", encoding="utf-8") as stream:
        stream.write(f'no-index=true\nfind-links=[{json.dumps(wheel.parent.as_posix())}]\n')
    (staged / "code.py").write_text('new code')
    client.sync_venv(explicit=True, staged_plugin={
        "staged": str(staged), "target": str(target), "target_digest": tree_digest(target) if target.exists() else None,
        "old_metadata": {"example": {"revision": "old"}},
        "new_metadata": {"example": {"revision": "new"}},
    })
    assert (target / "code.py").read_text() == "new code"
    assert not staged.exists()
    assert json.loads(metadata.read_text())["example"]["revision"] == "new"
    selected = selected_venv(project)
    assert (selected != previous) is active
    if active:
        python = selected / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        probe = subprocess.run([str(python), "-I", "-c", "import publication_dep; print(publication_dep.__version__)"],
                               text=True, capture_output=True, timeout=30)
        assert probe.returncode == 0, probe.stderr
        assert probe.stdout.strip() == "1.0"
        facts = (install_state_dir(project) / "facts.json").read_bytes()
        client.sync_venv(explicit=True)
        assert (install_state_dir(project) / "facts.json").read_bytes() == facts
    assert not (install_state_dir(project) / "publication.json").exists()
    assert not list(target.parent.glob(".previous-*"))


@pytest.mark.parametrize("mutation", ["sibling", "active"])
def test_selection_refuses_config_edits_during_preparation(client, tmp_path, monkeypatch, isolated_python, mutation):
    from pm.environments import install_state_dir
    home = tmp_path / "home"
    home.mkdir()
    config = home / "config.yaml"
    config.write_text("plugins:\n  enabled: [old]\n")
    repo = _current_environment(tmp_path, monkeypatch, [])
    (repo / "uv.lock").write_text("version = 2\n")
    edited = config if mutation == "active" else home / "profiles/sibling/config.yaml"
    expected = "plugins:\n  enabled: [concurrent]\n"
    facts = (install_state_dir(repo) / "facts.json").read_bytes()
    injection = (
        "from pm.packages import Venv\n"
        "def apply(self, *args, **kwargs):\n"
        f"    p=Path({str(edited)!r}); p.parent.mkdir(parents=True, exist_ok=True); p.write_text({expected!r})\n"
        "    return {}\n"
        "Venv.apply=apply\n"
    )
    worker_toolchain(client, monkeypatch, isolated_python, injection)
    with pytest.raises(ValueError, match="changed"):
        client.sync_venv(explicit=True, selection={"home": str(home), "enabled": ["new"], "disabled": []})
    assert edited.read_text() == expected
    assert (install_state_dir(repo) / "facts.json").read_bytes() == facts
    assert not (install_state_dir(repo) / "publication.json").exists()


@pytest.mark.parametrize("invalid", ["name: [", "manifest_version: 999", "requires_hermes: '>=999'", "name: other"])
def test_worker_rejects_unloadable_staged_plugin_without_app_dependencies(client, tmp_path, monkeypatch, invalid):
    from pm.store import tree_digest
    _current_environment(tmp_path, monkeypatch, [])
    target = tmp_path / "home/plugins/example"
    target.mkdir(parents=True)
    (target / "plugin.yaml").write_text("name: example\n")
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "plugin.yaml").write_text(invalid)
    previous = tree_digest(target)
    with pytest.raises(ValueError):
        client.sync_venv(explicit=True, staged_plugin={
            "target": str(target), "staged": str(staged), "target_digest": previous,
            "old_metadata": {}, "new_metadata": {"example": {"revision": "new"}},
        })
    assert tree_digest(target) == previous
    assert staged.exists()


def test_additional_candidates_are_discovered_by_sync_and_passive_probe(client, tmp_path, monkeypatch):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "plugin.yaml").write_text("name: candidate\npython_dependencies: [fixture-dep==1]\n")
    _current_environment(tmp_path, monkeypatch, [candidate])
    assert client.venv_is_current(extra_plugin_dirs=[candidate])
    assert not client.venv_is_current()
    client.sync_venv(explicit=True, extra_plugin_dirs=[candidate])
    assert client.venv_is_current(extra_plugin_dirs=[candidate])


def test_memory_setup_sends_candidate_paths_instead_of_discovery_callbacks(tmp_path, monkeypatch):
    from hermes_cli.memory_setup import memory_provider_dependency_inputs
    candidate = tmp_path / "provider"
    candidate.mkdir()
    (candidate / "plugin.yaml").write_text("name: provider\npython_dependencies: [fixture-dep==1]\n")
    monkeypatch.setattr("plugins.memory.find_provider_dir", lambda name: candidate)
    _, inputs = memory_provider_dependency_inputs("provider")
    assert inputs == {"extras": [], "extra_plugin_dirs": [candidate]}


@pytest.mark.parametrize(("kind", "rebuild", "phase"), [
    (kind, rebuild, phase)
    for kind in ("selection", "tree") for rebuild in (False, True) for phase in ("journal", "payload", "commit")
] + [("tree", rebuild, phase) for rebuild in (False, True) for phase in ("backup", "tree")])
def test_worker_death_recovers_at_each_durable_publication_boundary(
    client, tmp_path, monkeypatch, isolated_python, kind, rebuild, phase,
):
    from pm.environments import install_state_dir, selected_venv
    from pm import paths
    from pm.package import InstallError
    from pm.store import tree_digest

    worker_toolchain(client, monkeypatch, isolated_python)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(paths, "repo_root", lambda: project)
    core = '[project]\nname="death-proof"\nversion="1"\nrequires-python=">=3.11"\n[tool.uv]\npackage=false\n'
    (project / "pyproject.toml").write_text(core)
    client.lock_project(project, offline=True, explicit=True)
    home = tmp_path / "home"
    target = home / "plugins/example"
    target.mkdir(parents=True)
    (target / "plugin.yaml").write_text("name: example\n")
    (target / "code.py").write_text("old")
    config = home / "config.yaml"
    config.write_text("# exact bytes\nplugins:\n  enabled: " + ("[example]" if kind == "tree" else "[]") + "\n")
    metadata = target.parent / ".install-metadata.json"
    metadata.write_text('{"example":{"revision":"old"}}\n')
    client.sync_venv(explicit=True)
    before_env = selected_venv(project)
    state = install_state_dir(project)
    facts = state / "facts.json"
    before = {p: p.read_bytes() for p in (config, metadata, facts)}
    before_tree = tree_digest(target)
    if kind == "selection":
        if rebuild:
            (target / "pyproject.toml").write_text(core.replace("death-proof", "example"))
        arguments = {"selection": {"home": str(home), "enabled": ["example"], "disabled": []}}
        payload = config
    else:
        staged = tmp_path / "staged"
        shutil.copytree(target, staged)
        (staged / "code.py").write_text("new")
        if rebuild:
            (staged / "pyproject.toml").write_text(core.replace("death-proof", "example"))
        arguments = {"staged_plugin": {"target": str(target), "staged": str(staged), "target_digest": before_tree,
                     "old_metadata": {"example": {"revision": "old"}}, "new_metadata": {"example": {"revision": "new"}}}}
        payload = metadata
    # Exit immediately after the real durable write, not a simulated publication.
    injection = (
        "import json\nimport pm.publication as publication\nimport hermes_cli.runtime_state as state\n"
        "original = state._atomic_bytes\n"
        "def write(path, data):\n    original(path, data)\n"
        f"    if {phase!r} == 'journal' and path.name == 'publication.json': os._exit(17)\n"
        f"    if {phase!r} == 'payload' and path == Path({str(payload)!r}): os._exit(17)\n"
        f"    if {phase!r} == 'commit' and path.name == 'publication.json' and json.loads(data).get('committed'): os._exit(17)\n"
        "publication._atomic_bytes = state._atomic_bytes = write\n"
        "original_replace = os.replace\ndef replace(source, target):\n    original_replace(source, target)\n"
        f"    if {phase!r} == 'backup' and Path(target).name.startswith('.previous-'): os._exit(17)\n"
        f"    if {phase!r} == 'tree' and Path(target) == Path({str(target)!r}): os._exit(17)\n"
        "os.replace = replace\n"
    )
    if phase == "commit" and rebuild:
        injection += ("from pm.lock import Facts\nrecord = Facts.record_state\n"
                      "def commit(self, *args, **kwargs):\n    record(self, *args, **kwargs)\n    os._exit(17)\n"
                      "Facts.record_state = commit\n")
    worker_toolchain(client, monkeypatch, isolated_python, injection)
    with pytest.raises(InstallError, match="worker exited without a result"):
        client.sync_venv(explicit=True, **arguments)
    assert (state / "publication.json").exists()
    source = Path(client.__file__).resolve().parent.parent
    program = (f"import sys; sys.path.insert(0, {str(source)!r}); from pathlib import Path; "
               "from pm.environments import activate_dependencies; "
               f"project = Path({str(project)!r})\n"
               "activate_dependencies(project)\nactivate_dependencies(project)\n")
    recovery = subprocess.run([sys.executable, "-I", "-S", "-c", program], capture_output=True, text=True,
                              env=dict(os.environ), timeout=30)
    assert recovery.returncode == 0, recovery.stderr
    committed = phase == "commit"
    assert (selected_venv(project) != before_env) is (committed and rebuild)
    if not committed:
        assert {p: p.read_bytes() for p in before} == before
    if kind == "tree":
        assert (target / "code.py").read_text() == ("new" if committed else "old")
        if not committed:
            assert tree_digest(target) == before_tree
    else:
        from utils import fast_safe_load
        assert fast_safe_load(config.read_text())["plugins"]["enabled"] == (["example"] if committed else [])
    assert not (state / "publication.json").exists()
    assert not list(target.parent.glob(".previous-*"))


@pytest.mark.parametrize("mutation", ["metadata", "target", "staged", "sibling-manifest"])
def test_staged_publication_refuses_concurrent_input_edits(client, tmp_path, monkeypatch, isolated_python, mutation):
    from pm.environments import install_state_dir
    from pm.store import tree_digest
    repo = _current_environment(tmp_path, monkeypatch, [])
    home = tmp_path / "home"
    target = home / "plugins/example"
    target.mkdir(parents=True)
    (target / "plugin.yaml").write_text("name: example\n")
    (target / "code.py").write_text("old code")
    sibling = target.parent / "sibling"
    sibling.mkdir()
    (sibling / "plugin.yaml").write_text("name: sibling\npython_dependencies: [fixture-dep==1]\n")
    (home / "config.yaml").write_text("plugins:\n  enabled: [example, sibling]\n")
    metadata = target.parent / ".install-metadata.json"
    metadata.write_text("{}\n")
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "plugin.yaml").write_text("name: example\npython_dependencies: [fixture-dep==1]\n")
    (staged / "code.py").write_text("new code")
    edited = {"metadata": metadata, "target": target / "code.py", "staged": staged / "code.py",
              "sibling-manifest": sibling / "plugin.yaml"}[mutation]
    proposed = "name: sibling\nversion: changed\npython_dependencies: [fixture-dep==2]\n" if mutation == "sibling-manifest" else "concurrent edit"
    worker_toolchain(client, monkeypatch, isolated_python,
        "from pm.packages import Venv\n"
        "def apply(*args, **kwargs):\n"
        f"    Path({str(edited)!r}).write_text({proposed!r})\n    return {{}}\n"
        "Venv.apply = apply\n")
    facts = (install_state_dir(repo) / "facts.json").read_bytes()
    with pytest.raises(ValueError, match="changed"):
        client.sync_venv(explicit=True, staged_plugin={
            "target": str(target), "staged": str(staged), "target_digest": tree_digest(target),
            "old_metadata": {}, "new_metadata": {"example": {"revision": "new"}},
        })
    assert edited.read_text() == proposed
    assert (target / "code.py").read_text() == (proposed if mutation == "target" else "old code")
    assert (install_state_dir(repo) / "facts.json").read_bytes() == facts
    assert not (install_state_dir(repo) / "publication.json").exists()


def test_inactive_portable_publication_does_not_inspect_unrelated_dependency_manifests(client, tmp_path, monkeypatch):
    from pm.store import tree_digest
    _current_environment(tmp_path, monkeypatch, [])
    home = tmp_path / "home"
    target = home / "plugins/inactive"
    target.mkdir(parents=True)
    sibling = home / "plugins/sibling"
    sibling.mkdir()
    (sibling / "plugin.yaml").write_bytes(b"\xff")
    (home / "config.yaml").write_text("plugins:\n  enabled: [sibling]\n")
    staged = tmp_path / "staged"
    staged.mkdir()
    from hermes_cli.agent_plugins import PLUGIN_SCHEMA_V1
    (staged / "plugin.json").write_text(json.dumps({"$schema": PLUGIN_SCHEMA_V1, "name": "inactive", "version": "1.0.0"}))
    client.sync_venv(explicit=True, staged_plugin={
        "target": str(target), "staged": str(staged), "target_digest": tree_digest(target),
        "old_metadata": {}, "new_metadata": {"inactive": {"revision": "new"}},
    })
    assert json.loads((target / "plugin.json").read_text())["name"] == "inactive"
    assert (sibling / "plugin.yaml").read_bytes() == b"\xff"


def test_selection_preserves_yaml11_values_and_quotes_plugin_names(client, tmp_path, monkeypatch):
    import hermes_yaml
    _current_environment(tmp_path, monkeypatch, [])
    home = tmp_path / "home"
    config = home / "config.yaml"
    config.write_text('feature: yes\nother: no\nlabel: "on"\nplugins: {enabled: []}\n')
    before = hermes_yaml.safe_load(config.read_bytes())
    client.sync_venv(explicit=True, selection={"home": str(home), "enabled": ["on", "yes", "no"], "disabled": []})
    after = hermes_yaml.safe_load(config.read_bytes())
    assert {key: after[key] for key in ("feature", "other", "label")} == {key: before[key] for key in ("feature", "other", "label")}
    assert set(after["plugins"]["enabled"]) == {"on", "yes", "no"}
    assert 'label: "on"' in config.read_text()


def test_explicit_publication_keeps_its_intent_through_tool_acquisition(client, tmp_path, monkeypatch, isolated_python):
    from pm import paths
    from pm.environments import selected_venv
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(paths, "repo_root", lambda: project)
    (project / "pyproject.toml").write_text('[project]\nname="explicit-proof"\nversion="1"\nrequires-python=">=3.11"\n[tool.uv]\npackage=false\n')
    worker_toolchain(client, monkeypatch, isolated_python)
    client.lock_project(project, offline=True, explicit=True)
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    worker_toolchain(client, monkeypatch, isolated_python,
        "from pm.package import InstallError\ntools = pm._uv._toolchain\n"
        "def acquire(*, explicit=False, **kwargs):\n"
        "    if not explicit: raise InstallError('tools', 'explicit intent was lost')\n"
        "    return tools(explicit=explicit, **kwargs)\npm._uv._toolchain = acquire\n")
    client.sync_venv(explicit=True, selection={"home": str(tmp_path / "home"), "enabled": [], "disabled": []})
    assert (selected_venv(project) / "pyvenv.cfg").is_file()


def test_stale_enablement_cannot_replace_a_newer_selection(client, tmp_path, monkeypatch):
    from hermes_cli import plugins_cmd as pc
    from hermes_cli.plugins_admission import AdmissionRefused
    from utils import fast_safe_load
    _current_environment(tmp_path, monkeypatch, [])
    config = tmp_path / "home/config.yaml"
    config.write_text("plugins: {enabled: [], disabled: []}\n")
    from pm import plugins_state

    read = plugins_state.read_home_selection
    def concurrent_commit(home):
        stale = read(home)
        config.write_text("plugins: {enabled: [first], disabled: []}\n")
        return stale
    monkeypatch.setattr(plugins_state, "read_home_selection", concurrent_commit)
    with pytest.raises(AdmissionRefused, match="changed"):
        pc._set_plugin_enabled("second", enable=True)
    assert fast_safe_load(config.read_bytes())["plugins"]["enabled"] == ["first"]
