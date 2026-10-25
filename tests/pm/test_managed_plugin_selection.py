"""Managed-only dependency selections must reach PM's member discovery (#140174)."""
import pytest

from pm.plugins_state import enabled_plugins_ordered, read_home_selection


@pytest.mark.parametrize("user_config", [None, "", "plugins:\n  enabled: [local-only]\n"])
def test_managed_selection_reaches_dependency_union(tmp_path, monkeypatch, user_config):
    import pm.plugins_state as state

    home = tmp_path / "profile"
    managed = tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setattr(state, "dependency_homes", lambda: [home])
    if user_config is not None:
        (home / "config.yaml").write_text(user_config, encoding="utf-8")
    (managed / "config.yaml").write_text(
        "plugins:\n  enabled: [managed-plugin]\nmemory:\n  provider: managed-memory\n",
        encoding="utf-8",
    )
    (home / "plugins" / "managed-memory").mkdir(parents=True)
    # The raw reader also feeds writers; administrator values must never leak into it.
    config = read_home_selection(home)
    assert config == (None if user_config is None else {} if not user_config else {
        "plugins": {"enabled": ["local-only"]}
    })
    assert enabled_plugins_ordered() == {
        home / "plugins": ["managed-plugin", "managed-memory"]
    }
    from pm.workspace import enabled_member_dirs
    for name in ("managed-plugin", "managed-memory"):
        directory = home / "plugins" / name
        directory.mkdir(exist_ok=True)
        (directory / "plugin.yaml").write_text(
            f"name: {name}\npython_dependencies: [fixture-dependency]\n", encoding="utf-8"
        )
    assert enabled_member_dirs() == [home / "plugins" / name for name in ("managed-plugin", "managed-memory")]


def test_managed_policy_is_part_of_publication_snapshot(tmp_path, monkeypatch):
    from pm.publication import selection_snapshot
    import pm.plugins_state as state

    home, managed = tmp_path / "home", tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    monkeypatch.setattr(state, "dependency_homes", lambda: [home])
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    before = selection_snapshot()
    (managed / "config.yaml").write_text("memory:\n  provider: changed\n", encoding="utf-8")
    assert selection_snapshot() != before, "managed selection changed during preparation"


def test_managed_provider_cannot_be_reported_evicted_by_raw_edit(tmp_path, monkeypatch):
    from pm.plugin_eviction import PluginEviction
    import pm.plugins_state as state

    home, managed = tmp_path / "home", tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    plugin = home / "plugins" / "pinned"
    plugin.mkdir(parents=True)
    monkeypatch.setattr(state, "dependency_homes", lambda: [home])
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    (managed / "config.yaml").write_text("memory:\n  provider: pinned\n", encoding="utf-8")
    with pytest.raises(ValueError, match="managed"):
        PluginEviction([(plugin.parent, "pinned", plugin)], {plugin.resolve(): "incompatible"})


def test_proposed_selection_preserves_managed_precedence_and_raw_config(tmp_path, monkeypatch):
    import pm.plugins_state as state
    home, managed = tmp_path / "home", tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    monkeypatch.setattr(state, "dependency_homes", lambda: [home])
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    path = home / "config.yaml"
    path.write_text("plugins:\n  enabled: [old]\n", encoding="utf-8")
    before = path.read_bytes()
    (managed / "config.yaml").write_text(
        "plugins:\n  enabled: [pinned, disabled]\n", encoding="utf-8"
    )
    assert enabled_plugins_ordered(proposed_home=home, enabled=["proposed"], disabled=["disabled"]) == {
        home / "plugins": ["pinned"]
    }
    assert path.read_bytes() == before


def test_pm_discovery_does_not_require_application_dependencies(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    home, managed = tmp_path / "home", tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    (managed / "config.yaml").write_text("plugins:\n  enabled: [pinned]\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(HERMES_HOME=str(home), HERMES_MANAGED_DIR=str(managed))
    repo = Path(__file__).resolve().parents[2]
    code = '''
import importlib.abc
import sys
from pathlib import Path
class NoApplicationDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'openai', 'dotenv', 'rich', 'pydantic', 'httpx', 'requests', 'prompt_toolkit'}:
            raise ImportError('PM runtime has no application dependency: ' + fullname)
sys.meta_path.insert(0, NoApplicationDependencies())
sys.path.insert(0, sys.argv[1])
from pm.plugins_state import enabled_plugins_ordered
assert list(enabled_plugins_ordered().values()) == [['pinned']]
'''
    result = subprocess.run([sys.executable, "-I", "-c", code, str(repo)], cwd=repo,
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_prepared_selection_refuses_managed_change_before_writing(tmp_path, monkeypatch):
    from pm.publication import PluginSelection
    import pm.plugins_state as state
    import pm.publication as publication

    home, managed = tmp_path / "home", tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    monkeypatch.setattr(state, "dependency_homes", lambda: [home])
    monkeypatch.setattr(publication, "dependency_home_root", lambda: home)
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    path = home / "config.yaml"
    path.write_text("plugins:\n  enabled: [old]\n", encoding="utf-8")
    before = path.read_bytes()
    change = PluginSelection({"home": str(home), "enabled": ["new"], "disabled": []})
    (managed / "config.yaml").write_text("plugins:\n  enabled: [pinned]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        change.publish(tmp_path / "project")
    assert path.read_bytes() == before


@pytest.mark.parametrize("managed_text", [
    "plugins:\n  enabled: wrong\n", "memory:\n  provider: []\n",
])
def test_invalid_managed_selection_refuses(tmp_path, monkeypatch, managed_text):
    import pm.plugins_state as state
    home, managed = tmp_path / "home", tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    monkeypatch.setattr(state, "dependency_homes", lambda: [home])
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    (managed / "config.yaml").write_text(managed_text, encoding="utf-8")
    with pytest.raises(ValueError):
        enabled_plugins_ordered()


def test_managed_enabled_eviction_keeps_admin_values_out_of_raw_write(tmp_path, monkeypatch):
    from pm.plugin_eviction import PluginEviction
    from utils import fast_safe_load
    import pm.plugins_state as state

    home, managed = tmp_path / "home", tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    plugin = home / "plugins" / "pinned"
    plugin.mkdir(parents=True)
    monkeypatch.setattr(state, "dependency_homes", lambda: [home])
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    (managed / "config.yaml").write_text("plugins:\n  enabled: [pinned]\n", encoding="utf-8")
    change = PluginEviction([(plugin.parent, "pinned", plugin)], {plugin.resolve(): "incompatible"})
    proposed = fast_safe_load(change.edits[0][2].decode())
    assert proposed == {"plugins": {"disabled": ["pinned"]}}


@pytest.mark.parametrize("incompatible", [False, True])
def test_eviction_refuses_policy_change_during_discovery(tmp_path, monkeypatch, incompatible):
    from types import SimpleNamespace
    import pm.plugin_eviction as eviction
    import pm.plugins_state as state
    import pm.workspace as workspace
    import pm.install as install
    from pm import receipt

    home, managed = tmp_path / "home", tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    plugin = home / "plugins" / "old"
    plugin.mkdir(parents=True)
    policy = managed / "config.yaml"
    policy.write_text("plugins:\n  enabled: [old]\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setattr(state, "dependency_homes", lambda: [home])
    def discovered(**kwargs):
        policy.write_text("plugins:\n  enabled: [new]\n", encoding="utf-8")
        return [(plugin.parent, "old", plugin)]
    monkeypatch.setattr(workspace, "enabled_plugin_entries", discovered)
    monkeypatch.setattr(workspace, "_is_member_candidate", lambda path: True)
    monkeypatch.setattr(eviction, "_interpreter_version", lambda: "3.14")
    monkeypatch.setattr(eviction, "static_verdicts", lambda *args: (
        {plugin.resolve(): "incompatible"} if incompatible else {}, {}
    ))
    monkeypatch.setattr(install, "_target_selection", lambda *a, **kw: ([], "stamp", kw["inputs"]))
    monkeypatch.setattr(install, "_runtime_state_matches", lambda *a: True)
    monkeypatch.setattr(receipt, "record_feature_list", lambda *a: None)
    committed = []
    monkeypatch.setattr(install, "_commit_selection", lambda *a, **kw: committed.append(kw))
    with pytest.raises(ValueError, match="changed"):
        eviction.sync_evicting(SimpleNamespace(), None, {}, extras=[], shipped=None, frozen=None, explicit=True)
    assert committed == []


def test_empty_eviction_still_fences_policy_without_writing(tmp_path, monkeypatch):
    from pm.plugin_eviction import PluginEviction
    import pm.plugins_state as state

    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setattr(state, "dependency_homes", lambda: [])
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    change = PluginEviction([], {})
    project = tmp_path / "not-created"
    change.publish(project)
    assert not project.exists()
    (managed / "config.yaml").write_text("plugins:\n  enabled: [new]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        change.publish(project)
    assert not project.exists()
