"""The restart-safe cron worker's PYTHONPATH pin carries Hermes' own dependency graph.

A PM store-Python gateway activates its committed dependency generation in-process
(nothing lands in the environment), so a fresh ``sys.executable`` worker imports no
third-party packages at all — the pin must add that generation next to the checkout.
"""

import os

import pytest

from cron import scheduler_worker_env
from cron.scheduler_worker_env import pin_hermes_tree_on_pythonpath


@pytest.fixture()
def committed_generation(monkeypatch, tmp_path):
    """Fake a PM-committed generation venv whose site-packages really exists."""
    import pm.environments

    venv = tmp_path / "generations" / "g1" / "venv"
    venv.mkdir(parents=True)
    site_packages = pm.environments.site_packages(venv)
    site_packages.mkdir(parents=True)
    monkeypatch.setattr(pm.environments, "committed_venv", lambda root: venv)
    return site_packages


def test_pin_carries_checkout_then_committed_generation(tmp_path, committed_generation):
    env = pin_hermes_tree_on_pythonpath({"PYTHONPATH": ""}, tmp_path)
    assert env["PYTHONPATH"].split(os.pathsep) == [str(tmp_path), str(committed_generation)]


def test_pin_without_committed_generation_still_pins_checkout(monkeypatch, tmp_path):
    import pm.environments

    monkeypatch.setattr(pm.environments, "committed_venv", lambda root: None)
    env = pin_hermes_tree_on_pythonpath({"PYTHONPATH": ""}, tmp_path)
    assert env["PYTHONPATH"] == str(tmp_path)


def test_pin_does_not_duplicate_a_generation_already_on_the_path(tmp_path, committed_generation):
    env = pin_hermes_tree_on_pythonpath({"PYTHONPATH": str(committed_generation)}, tmp_path)
    entries = env["PYTHONPATH"].split(os.pathsep)
    assert entries.count(str(committed_generation)) == 1
    assert entries[0] == str(tmp_path)


def test_pin_leaves_the_caller_environ_alone(monkeypatch, tmp_path, committed_generation):
    monkeypatch.setenv("PYTHONPATH", "/do-not-touch")
    env = pin_hermes_tree_on_pythonpath({"PYTHONPATH": ""}, tmp_path)
    assert os.environ["PYTHONPATH"] == "/do-not-touch"
    assert str(tmp_path) in env["PYTHONPATH"]
    assert "/do-not-touch" not in env["PYTHONPATH"]


def test_pin_skips_entirely_for_a_purelib_install(monkeypatch, tmp_path):
    monkeypatch.setattr(scheduler_worker_env, "_installed_purelib", lambda: tmp_path.resolve())
    env = {"PYTHONPATH": "/kept"}
    assert pin_hermes_tree_on_pythonpath(env, tmp_path) == {"PYTHONPATH": "/kept"}
