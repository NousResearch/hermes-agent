"""A backend binds even while another process holds the install lock.

Regression: ``runtime_lock`` waited forever, and the lock is held across a whole dependency
rebuild (tens of seconds on a bundle), so the second backend — the ``hermes-setup`` onboarding
profile — never reached its port: the renderer gave up at 40 s and Electron killed it at 90 s
while the holder ran a uv build it had started for a status probe. Deciding what a lost race
means is the caller's job now: readers skip, explicit installs still wait.
"""

import json
import os
import subprocess
import sys

import pytest

from pm.environments import install_state_dir, runtime_facts_path, site_packages

_HOLDER = """
import sys
from pathlib import Path
from hermes_cli.runtime_state import runtime_lock
with runtime_lock(Path(sys.argv[1]), timeout=0):
    print("locked", flush=True)
    sys.stdin.readline()
"""


@pytest.fixture
def install_tree(tmp_path, monkeypatch):
    """A real committed generation, ready for ``activate_dependencies`` to select."""
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    environment = install_state_dir(repo) / "environments" / "gen1" / "venv"
    site = site_packages(environment)
    site.mkdir(parents=True)
    (environment / "pyvenv.cfg").write_text("version = 3.14", encoding="utf-8")
    (environment.parent / ".lease-managed").touch()
    runtime_facts_path(repo).write_text(
        json.dumps({"packages": {"venv": {"environment": str(environment)}}}), encoding="utf-8")
    return repo, site


@pytest.fixture
def locked_install(install_tree):
    """A live second process holding ``.install.lock``, the way a sibling backend does."""
    repo, site = install_tree
    holder = subprocess.Popen([sys.executable, "-c", _HOLDER, str(repo)], env=dict(os.environ),
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
    assert holder.stdout is not None and holder.stdin is not None
    assert holder.stdout.readline().strip() == "locked"
    try:
        yield repo, site
    finally:
        holder.communicate("done\n", timeout=15)


def test_runtime_lock_reports_a_lost_race(locked_install):
    from hermes_cli.runtime_state import runtime_lock

    repo, _site = locked_install
    with runtime_lock(repo, timeout=0.2) as held:
        assert held is False
    # The blocked-install symptom: a bounded wait, then carry on — never the holder's rebuild.


def test_boot_activation_proceeds_while_the_install_is_locked(locked_install, monkeypatch):
    """The issue's symptom, inverted: the backend reaches its dependency environment and can bind
    while a sibling holds the lock. Recovery belongs to whoever holds it, so it is skipped."""
    import pm.environments as runtime_paths
    import hermes_cli.runtime_state as runtime_state

    repo, site = locked_install
    real_lock = runtime_state.runtime_lock
    monkeypatch.setattr(runtime_state, "runtime_lock",
                        lambda project, **kw: real_lock(project, timeout=0.2))
    recovered: list = []
    monkeypatch.setattr(runtime_state, "recover_publication", lambda project: recovered.append(project))

    saved_path = sys.path[:]
    try:
        runtime_paths.activate_dependencies(repo)
        assert str(site) in sys.path
    finally:
        sys.path[:] = saved_path
    assert recovered == []
