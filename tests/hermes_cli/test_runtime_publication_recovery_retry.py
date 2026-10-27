"""A rollback interrupted between two configs is finished by the next recovery.

A plugin eviction journals every home it edits. If the second restore fails the journal must
survive untouched, so a retry restores both configs and only then removes it.
"""
import json
from pathlib import Path

import pytest

from hermes_cli import runtime_state
from pm import paths
from pm.environments import install_state_dir, runtime_facts_path
from pm.plugin_eviction import PluginEviction


def test_interrupted_multi_config_rollback_is_finished_by_retry(tmp_path, monkeypatch):
    user = tmp_path / "user"
    root = user / ".hermes"
    profile = root / "profiles" / "work"
    project = tmp_path / "repo"
    profile.mkdir(parents=True)
    project.mkdir()
    for var in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(var, str(user))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: user)
    monkeypatch.setattr(paths, "repo_root", lambda: project)

    root_config, profile_config = root / "config.yaml", profile / "config.yaml"
    root_config.write_bytes(b"# root comment\nplugins:\n  enabled: [alpha]\nmodel: root-model\n")
    profile_config.write_bytes(b"# profile comment\nplugins:\n  enabled: [beta]\nmodel: work-model\n")
    originals = {path: path.read_bytes() for path in (root_config, profile_config)}
    facts = runtime_facts_path(project)
    facts.parent.mkdir(parents=True)
    facts.write_bytes(b'{"synthetic": true}')

    entries = [(root / "plugins", "alpha", tmp_path / "src" / "alpha"),
               (profile / "plugins", "beta", tmp_path / "src" / "beta")]
    reasons = {plugin_dir.resolve(): "synthetic misfit" for _plugins_dir, _name, plugin_dir in entries}
    journal = install_state_dir(project) / "publication.json"

    with runtime_state.runtime_lock(project) as held:
        assert held
        PluginEviction(entries, reasons).publish(project)
    published = {path: path.read_bytes() for path in originals}
    journal_bytes = journal.read_bytes()
    assert all(published[path] != originals[path] for path in originals)
    assert {Path(entry["config"]) for entry in json.loads(journal_bytes)["configs"]} == set(originals)

    real_write = runtime_state._atomic_bytes

    def fail_second_config(path, data, *args, **kwargs):
        if Path(path) == profile_config:
            raise OSError("synthetic write failure")
        return real_write(path, data, *args, **kwargs)

    with monkeypatch.context() as failing:
        failing.setattr(runtime_state, "_atomic_bytes", fail_second_config)
        with runtime_state.runtime_lock(project) as held:
            assert held
            with pytest.raises(RuntimeError, match="cannot recover dependency publication"):
                runtime_state.recover_publication(project)

    assert root_config.read_bytes() == originals[root_config]
    assert profile_config.read_bytes() == published[profile_config]
    assert journal.read_bytes() == journal_bytes
    assert facts.read_bytes() == b'{"synthetic": true}'

    with runtime_state.runtime_lock(project) as held:
        assert held
        runtime_state.recover_publication(project)

    assert {path: path.read_bytes() for path in originals} == originals
    assert not journal.exists()
    assert facts.read_bytes() == b'{"synthetic": true}'

    with runtime_state.runtime_lock(project) as held:
        assert held
        runtime_state.recover_publication(project)
    assert {path: path.read_bytes() for path in originals} == originals
