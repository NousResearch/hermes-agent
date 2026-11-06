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


def test_young_unleased_generations_are_capped(tmp_path, monkeypatch):
    """A launch loop commits a generation per launch; the day-long grace alone never bounds them."""
    import time
    from pm.environments import install_state_dir, runtime_facts_path
    from hermes_cli.runtime_state import collect_generations, lease_directory

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    generations = install_state_dir(repo) / "environments"
    now = time.time()

    def publish(name, age_seconds):
        generation = generations / name
        (generation / "venv").mkdir(parents=True)
        (generation / "venv" / "pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")
        marker = generation / ".lease-managed"
        marker.touch()
        os.utime(marker, (now - age_seconds, now - age_seconds))
        return generation

    selected = publish("selected", 0)
    runtime_facts_path(repo).write_text(
        json.dumps({"packages": {"venv": {"environment": str(selected / "venv")}}}))
    # young[0] is the newest; every one is far inside the default grace window.
    young = [publish(f"young-{index}", 60 * (index + 1)) for index in range(5)]
    leased = publish("leased", 60 * 10)
    legacy = generations / "legacy"  # pre-lease generation: no marker, never collected
    legacy.mkdir()
    release = lease_directory(leased)
    try:
        assert sorted(collect_generations(repo)) == sorted(young[2:])
        assert all(generation.is_dir() for generation in (selected, leased, legacy, *young[:2]))
        # The cap counts unleased generations only; a held lease is never a victim.
        assert collect_generations(repo, keep_recent=0) == young[:2]
        assert leased.is_dir()
    finally:
        release()
    assert collect_generations(repo, keep_recent=0) == [leased]
    assert selected.is_dir() and legacy.is_dir()
