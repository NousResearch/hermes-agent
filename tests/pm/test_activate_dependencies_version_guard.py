"""activate_dependencies must not inject a generation built for another interpreter.

Regression tests for #122395: a mismatched generation is refused by PM's bare
store Python (with the repair remedy) and never selected by a venv interpreter,
which keeps the packages it booted with.
"""

import sys
from contextlib import contextmanager

import pytest

import pm.environments as env


@contextmanager
def _runtime_lock(_project_root):
    yield True


def _mismatched_install(monkeypatch, tmp_path, *, venv_interpreter):
    """An install whose committed generation was built for Python 3.99."""
    venv = tmp_path / "generation"
    (venv / "lib" / "python3.99").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("version = 3.99.0\n", encoding="utf-8")
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setattr(env, "install_state_dir", lambda _root: state)
    monkeypatch.setattr(env, "committed_venv", lambda _root: venv)
    monkeypatch.setattr(env, "runtime_facts_path", lambda _root: tmp_path / "facts.json")

    import hermes_cli.runtime_state as runtime_state

    monkeypatch.setattr(runtime_state, "runtime_lock", _runtime_lock)
    monkeypatch.setattr(runtime_state, "recover_publication", lambda _root: None)
    monkeypatch.setattr(runtime_state, "lease_generation", lambda _environment: lambda: None)

    store_prefix = str(tmp_path / "store-python")
    monkeypatch.setattr(sys, "base_prefix", store_prefix)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv-python") if venv_interpreter else store_prefix)
    return venv


def test_store_python_refuses_generation_built_for_another_interpreter(monkeypatch, tmp_path):
    _mismatched_install(monkeypatch, tmp_path, venv_interpreter=False)
    with pytest.raises(RuntimeError, match=r"built for Python 3\.99 but this process runs"):
        env.activate_dependencies(tmp_path)


def test_venv_interpreter_keeps_its_own_packages(monkeypatch, tmp_path):
    _mismatched_install(monkeypatch, tmp_path, venv_interpreter=True)
    before = list(sys.path)
    env.activate_dependencies(tmp_path)  # no raise: keep the boot contract
    assert sys.path == before


def test_matching_generation_still_selects(monkeypatch, tmp_path):
    venv = tmp_path / "generation"
    running = f"python{sys.version_info.major}.{sys.version_info.minor}"
    (venv / "lib" / running).mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(
        f"version = {sys.version_info.major}.{sys.version_info.minor}.0\n", encoding="utf-8"
    )
    (venv / "lib" / running / "site-packages").mkdir()
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setattr(env, "install_state_dir", lambda _root: state)
    monkeypatch.setattr(env, "committed_venv", lambda _root: venv)
    monkeypatch.setattr(env, "runtime_facts_path", lambda _root: tmp_path / "facts.json")

    import hermes_cli.runtime_state as runtime_state

    monkeypatch.setattr(runtime_state, "runtime_lock", _runtime_lock)
    monkeypatch.setattr(runtime_state, "recover_publication", lambda _root: None)
    monkeypatch.setattr(runtime_state, "lease_generation", lambda _environment: lambda: None)

    before = list(sys.path)
    env.activate_dependencies(tmp_path)
    assert sys.path != before  # the generation was selected
