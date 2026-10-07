"""The restart-safe cron worker's PYTHONPATH pin carries Hermes' own dependency graph.

A PM store-Python gateway activates its committed dependency generation in-process
(nothing lands in the environment), so a fresh ``sys.executable`` worker imports no
third-party packages at all — the pin must add that generation next to the checkout.
``PYTHONPATH`` does not process ``.pth`` files, so the spawn is additionally wrapped with
the ``site.addsitedir`` semantics ``pm.activate_dependencies`` gives the gateway itself.
"""

import os
import subprocess
import sys

import pytest

from cron import scheduler_worker_env
from cron.scheduler_worker_env import external_worker_argv, pin_hermes_tree_on_pythonpath


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


def test_worker_argv_wraps_dash_m_with_addsitedir_bootstrap(tmp_path, committed_generation):
    command = ["PY", "-m", "cron.scheduler", "--external-worker-file", "p", "--ack-file", "a"]
    argv = external_worker_argv(command)
    assert argv[0] == "PY"
    assert argv[1] == "-c"
    assert "site.addsitedir" in argv[2]
    assert "cron.scheduler" in argv[2]
    assert repr(str(committed_generation.resolve())) in argv[2]
    assert argv[3:] == command[2:]


def test_worker_argv_plain_without_a_committed_generation(monkeypatch):
    import pm.environments

    monkeypatch.setattr(pm.environments, "committed_venv", lambda root: None)
    command = ["PY", "-m", "cron.scheduler"]
    assert external_worker_argv(command) is command


def test_worker_argv_leaves_a_non_dash_m_command_alone(tmp_path, committed_generation):
    command = ["PY", "cron/scheduler.py", "--external-worker-file", "p"]
    assert external_worker_argv(command) is command


def test_worker_bootstrap_processes_pth_entries(tmp_path, committed_generation):
    """The review's probe, pinned as a regression test: a module reachable only through the
    generation's ``.pth`` (the pywin32 / ``__editable__`` shape) stays unimportable under a
    bare ``PYTHONPATH`` pin, but resolves under the generated bootstrap's addsitedir."""
    pth_only_dir = tmp_path / "pth-only"
    pth_only_dir.mkdir()
    (pth_only_dir / "zzpthprobe.py").write_text("print('pth-processed')\n", encoding="utf-8")
    (committed_generation / "zzprobe.pth").write_text(
        f"import sys; sys.path.insert(0, {str(pth_only_dir)!r})\n", encoding="utf-8"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), str(committed_generation)])}

    bare = subprocess.run(
        [sys.executable, "-c", "import zzpthprobe"],
        capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=60,
    )
    assert bare.returncode != 0  # negative control: PYTHONPATH does not process .pth

    bootstrap = scheduler_worker_env._WORKER_BOOTSTRAP_TEMPLATE.format(
        site_packages=str(committed_generation), module="zzpthprobe"
    )
    wrapped = subprocess.run(
        [sys.executable, "-c", bootstrap],
        capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=60,
    )
    assert wrapped.returncode == 0, wrapped.stderr
    assert "pth-processed" in wrapped.stdout
