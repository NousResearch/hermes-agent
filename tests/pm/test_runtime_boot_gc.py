"""A real bootstrap reader pins its generation before GC can select victims."""
import json
import os
import subprocess
import sys


def test_bootstrap_lease_survives_selection_change(tmp_path, monkeypatch):
    from pm.environments import install_state_dir, runtime_facts_path, site_packages
    from hermes_cli.runtime_state import collect_generations

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    state = install_state_dir(repo)
    for name in ("first", "second", "unused"):
        venv = state / "environments" / name / "venv"
        venv.mkdir(parents=True)
        # pyvenv.cfg first: the layout keys its site-packages path off the recorded version.
        (venv / "pyvenv.cfg").write_text("version = 3.11")
        site_packages(venv).mkdir(parents=True)
        (venv.parent / ".lease-managed").touch()
    legacy = state / "environments" / "old-unleased"
    legacy.mkdir()

    def select(name):
        environment = state / "environments" / name / "venv"
        runtime_facts_path(repo).write_text(json.dumps({"packages": {"venv": {"environment": str(environment)}}}))
        return environment

    first = select("first")
    code = '''
import sys
from pathlib import Path
from pm.environments import activate_dependencies
activate_dependencies(Path(sys.argv[1]))
print("ready", flush=True)
sys.stdin.readline()
'''
    child = subprocess.Popen([sys.executable, "-c", code, str(repo)], env=dict(os.environ),
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        select("second")
        collect_generations(repo, min_age_seconds=0)
        assert first.is_dir()
        assert not (state / "environments" / "unused").exists()
        assert legacy.is_dir()
    finally:
        child.communicate("done\n", timeout=15)
    assert child.returncode == 0
    collect_generations(repo, min_age_seconds=0)
    assert not first.exists()
    assert (state / "environments" / "second").is_dir()
    assert legacy.is_dir()


def test_leased_generation_is_reported_instead_of_silently_kept(tmp_path, monkeypatch):
    """``pm gc`` printed "removed 0" while leased generations pinned the volume full (#137116):
    a generation skipped for a live lease must surface to the caller, not vanish from the report."""
    from pm.environments import install_state_dir, runtime_facts_path, site_packages
    from hermes_cli.runtime_state import collect_generations

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    state = install_state_dir(repo)
    for name in ("held", "selected", "unused"):
        venv = state / "environments" / name / "venv"
        venv.mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("version = 3.11")
        site_packages(venv).mkdir(parents=True)
        (venv.parent / ".lease-managed").touch()

    def select(name):
        environment = state / "environments" / name / "venv"
        runtime_facts_path(repo).write_text(json.dumps({"packages": {"venv": {"environment": str(environment)}}}))
        return environment

    select("held")
    code = '''
import sys
from pathlib import Path
from pm.environments import activate_dependencies
activate_dependencies(Path(sys.argv[1]))
print("ready", flush=True)
sys.stdin.readline()
'''
    child = subprocess.Popen([sys.executable, "-c", code, str(repo)], env=dict(os.environ),
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        select("selected")
        leased = []
        removed = collect_generations(repo, min_age_seconds=0, leased_generations=leased)
        assert [g.name for g in removed] == ["unused"]
        assert [g.name for g in leased] == ["held"], "a pinned generation must be reported, not silently kept"
        assert (state / "environments" / "held").is_dir()
    finally:
        child.communicate("done\n", timeout=15)
    assert child.returncode == 0
    later = []
    reclaimed = collect_generations(repo, min_age_seconds=0, leased_generations=later)
    assert [g.name for g in reclaimed] == ["held"], "once the reader exits the generation is reclaimed"
    assert later == [], "a reclaimed generation is not reported leased"


def test_cmd_gc_prints_leased_generation_skips(tmp_path, monkeypatch, capsys):
    """The summary names the pinned generations so a zero reclaim is diagnosable from the output alone."""
    from pm import cli as pm_cli

    def fake_collect(project, *, min_age_seconds=86400, leased_generations=None):
        if leased_generations is not None:
            leased_generations.append(tmp_path / "980a1ba1c0ffee")
        return [tmp_path / "gone"]

    monkeypatch.setattr("hermes_cli.runtime_state.collect_generations", fake_collect)
    monkeypatch.setattr("pm.runtime.collect_runtime_generations", lambda root: [object()])
    monkeypatch.setattr("pm.install_states.collect_orphan_install_states", lambda root: [])
    monkeypatch.setattr(pm_cli, "_gc_store", lambda store, facts: (1, 2))
    monkeypatch.setattr("pm.paths.writable_store_root", lambda: tmp_path / "store")
    monkeypatch.setattr("pm.lock.Facts", lambda root: None)
    monkeypatch.setattr("pm.paths.repo_root", lambda: tmp_path)
    monkeypatch.setattr("pm.environments.install_state_dir", lambda project: tmp_path / "state")
    monkeypatch.setattr("pm.environments.installs_root", lambda: tmp_path / "installs")

    assert pm_cli.cmd_gc(None) == 0

    out = capsys.readouterr().out
    assert "removed 1 dependency generations, 1 PM runtime generations" in out
    assert "gc: kept 1 leased generation(s) a running process still pins: 980a1ba1..." in out
