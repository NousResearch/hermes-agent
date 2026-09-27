"""The restart-safe cron worker must be spawned on an interpreter that can import the tree.

Regression: the worker was spawned on ``sys.executable`` -- the GATEWAY's interpreter. An
update can leave the gateway running a bare provisioned runtime whose site-packages holds pip
and nothing else, while the real dependency environment lives in the venv PM selected for the
install. The worker then died at import with ``ModuleNotFoundError`` (``ruamel.yaml``,
``dotenv``, ...) BEFORE its ownership acknowledgement, so every agent job was reported failed
while every systemd unit still read ``active`` -- and only jobs that import nothing
(``no_agent`` script jobs) kept working, which is what made the failure look partial.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from cron.scheduler_worker_env import worker_interpreter


@pytest.fixture
def fake_install(tmp_path, monkeypatch):
    """A temp Hermes root whose install has a real ``facts.json`` and venv generation."""
    import pm.environments as environments

    root = tmp_path / "repo"
    root.mkdir()
    # installs_root()/install_state_dir()/runtime_facts_path() all resolve through this.
    monkeypatch.setattr(environments, "dependency_home_root", lambda: tmp_path)

    def make(*, with_interpreter: bool = True) -> Path:
        state = environments.install_state_dir(root)
        venv = state / "environments" / "gen1" / "venv"
        (venv / "bin").mkdir(parents=True, exist_ok=True)
        # _recorded_venv() requires a real pyvenv.cfg and a path under environments/.
        (venv / "pyvenv.cfg").write_text("version = 3.14.7\n", encoding="utf-8")
        python = venv / "bin" / "python"
        if with_interpreter:
            python.write_text("#!/bin/sh\n", encoding="utf-8")
        facts = environments.runtime_facts_path(root)
        facts.parent.mkdir(parents=True, exist_ok=True)
        facts.write_text(
            json.dumps({"packages": {"venv": {"environment": str(venv)}}}),
            encoding="utf-8",
        )
        return python

    return root, make


def test_worker_uses_the_selected_environment_interpreter(fake_install):
    """The point of the fix: the worker runs on the environment PM selected, not the caller's."""
    root, make = fake_install
    expected = make()

    assert worker_interpreter(root) == str(expected)
    assert worker_interpreter(root) != sys.executable


def test_falls_back_to_the_caller_interpreter_without_a_dependency_record(tmp_path, monkeypatch):
    """Source checkout, wheel/pipx install, or a test: no facts.json -> old behaviour."""
    import pm.environments as environments

    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(environments, "dependency_home_root", lambda: tmp_path)

    assert not environments.runtime_facts_path(root).is_file()
    assert worker_interpreter(root) == sys.executable


def test_falls_back_when_the_selected_interpreter_is_missing(fake_install):
    """A half-removed generation must not wedge every job: fall back, never raise."""
    root, make = fake_install
    make(with_interpreter=False)

    assert worker_interpreter(root) == sys.executable


def test_broken_facts_record_falls_back_instead_of_raising(tmp_path, monkeypatch):
    """``selected_venv()`` raises on an invalid record; the spawn path must survive that."""
    import pm.environments as environments

    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(environments, "dependency_home_root", lambda: tmp_path)
    facts = environments.runtime_facts_path(root)
    facts.parent.mkdir(parents=True, exist_ok=True)
    facts.write_text("{ not json", encoding="utf-8")

    assert worker_interpreter(root) == sys.executable


def test_facts_record_pointing_outside_the_install_falls_back(tmp_path, monkeypatch):
    """``_recorded_venv`` rejects an environment outside this install; so must we."""
    import pm.environments as environments

    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(environments, "dependency_home_root", lambda: tmp_path)
    outside = tmp_path / "elsewhere" / "venv"
    (outside / "bin").mkdir(parents=True, exist_ok=True)
    (outside / "pyvenv.cfg").write_text("version = 3.14.7\n", encoding="utf-8")
    (outside / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    facts = environments.runtime_facts_path(root)
    facts.parent.mkdir(parents=True, exist_ok=True)
    facts.write_text(
        json.dumps({"packages": {"venv": {"environment": str(outside)}}}),
        encoding="utf-8",
    )

    assert worker_interpreter(root) == sys.executable
