"""A reader that lost the install lock still ends up leasing the generation it imports from."""
import contextlib
import json
import os
import sys

import pytest


@pytest.fixture
def generations(tmp_path, monkeypatch):
    from pm.environments import install_state_dir, runtime_facts_path, site_packages

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    state = install_state_dir(repo)
    for name in ("first", "second"):
        venv = state / "environments" / name / "venv"
        venv.mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("version = 3.11", encoding="utf-8")
        site_packages(venv).mkdir(parents=True)
        (venv.parent / ".lease-managed").touch()

    def select(name):
        environment = state / "environments" / name / "venv"
        runtime_facts_path(repo).write_text(json.dumps({"packages": {"venv": {"environment": str(environment)}}}), encoding="utf-8")
        return environment.resolve()

    return repo, state, select


def test_unlocked_reader_releases_a_lease_the_installer_moved_away_from(generations, monkeypatch):
    from hermes_cli import runtime_state
    from pm.environments import activate_dependencies

    repo, state, select = generations
    first = select("first")
    real_lease = runtime_state.lease_generation
    leased = []

    def racing_lease(environment):
        # The installer commits between the reader's selection read and its lease.
        if not leased:
            select("second")
        leased.append(environment)
        return real_lease(environment)

    @contextlib.contextmanager
    def lost_lock(project, **kwargs):
        yield False

    monkeypatch.setattr(runtime_state, "runtime_lock", lost_lock)
    monkeypatch.setattr(runtime_state, "lease_generation", racing_lease)
    # Activation rewrites this process's import path and environment; keep it scoped.
    monkeypatch.setattr(sys, "path", list(sys.path))
    for key in ("PYTHONPATH", "PATH", "VIRTUAL_ENV"):
        monkeypatch.setenv(key, os.environ.get(key, ""))

    activate_dependencies(repo)

    assert leased[0] == first and leased[-1] == state / "environments" / "second" / "venv"
    assert not any((first.parent / ".leases").iterdir()), "the stale lease must be released"
    assert len(list((state / "environments" / "second" / ".leases").iterdir())) == 1


def test_release_removes_the_lease_file(generations):
    from hermes_cli.runtime_state import lease_generation

    _, state, select = generations
    environment = select("first")
    release = lease_generation(environment)
    assert list((environment.parent / ".leases").iterdir())
    release()
    assert not list((environment.parent / ".leases").iterdir())
