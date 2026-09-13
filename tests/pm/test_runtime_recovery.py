"""Publication survives process death; live dependency generations survive GC."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("commit_facts", [False, True])
def test_killed_publication_recovers_before_boot(tmp_path, commit_facts):

    repo = tmp_path / "repo"
    repo.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    config = home / "config.yaml"
    config.write_text("plugins:\n  enabled: [old]\n")
    original = config.read_bytes()
    env = dict(os.environ, HERMES_HOME=str(home))
    code = '''
import os, sys
from pathlib import Path
from hermes_cli.plugins_admission import _config_commit
from hermes_cli.runtime_paths import runtime_facts_path
import pm.paths as paths
from pm.lock import Facts
repo, config = map(Path, sys.argv[1:3])
paths.repo_root = lambda: repo
change = _config_commit({"new"}, set())
if sys.argv[3] == "True":
    Facts(runtime_facts_path(repo)).record_state("venv", "new", [])
os._exit(17)
'''
    proc = subprocess.run([sys.executable, "-c", code, str(repo), str(config), str(commit_facts)], env=env, capture_output=True, text=True)
    assert proc.returncode == 17, proc.stderr
    boot = """
import sys
from pathlib import Path
from hermes_cli.runtime_paths import activate_dependencies, site_packages
repo, config = map(Path, sys.argv[1:3])
site_packages(repo / "venv").mkdir(parents=True, exist_ok=True)
activate_dependencies(repo)
print(config.read_text(), end="")
"""
    recovered = subprocess.run([sys.executable, "-c", boot, str(repo), str(config)],
                               env=env, capture_output=True, text=True, timeout=30)
    assert recovered.returncode == 0, recovered.stderr
    assert ("old" in recovered.stdout) is not commit_facts
    assert (config.read_bytes() == original) is not commit_facts


def test_generation_gc_keeps_selected_and_live_leases(tmp_path, monkeypatch):
    from hermes_cli.runtime_paths import install_state_dir
    from hermes_cli.runtime_state import collect_generations
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    state = install_state_dir(repo)
    gens = state / "environments"
    for name in ("selected", "live", "unused"):
        env = gens / name / "venv"
        env.mkdir(parents=True)
        (env / "pyvenv.cfg").write_text("version=3.11")
        (env.parent / ".lease-managed").touch()
    (gens / "old-unleased").mkdir()
    (state / "facts.json").write_text(json.dumps({"packages": {"venv": {"environment": str(gens / "selected" / "venv")}}}))
    code = '''
from pathlib import Path
from hermes_cli.runtime_state import lease_generation
import sys
lease_generation(Path(sys.argv[1]))
print("leased", flush=True)
sys.stdin.readline()
'''
    child = subprocess.Popen([sys.executable, "-c", code, str(gens / "live" / "venv")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "leased"
        collect_generations(repo, min_age_seconds=0)
        assert (gens / "selected").exists()
        assert (gens / "live").exists()
        assert not (gens / "unused").exists()
    finally:
        child.communicate("stop\n", timeout=15)
    assert child.returncode == 0
    collect_generations(repo, min_age_seconds=0)
    assert not (gens / "live").exists()
    assert (gens / "selected").exists()
    assert (gens / "old-unleased").is_dir()
