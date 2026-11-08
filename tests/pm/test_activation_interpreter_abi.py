"""PM activation never overlays a generation built by another Python (card t_e527db90).

A committed generation records the interpreter it was built for in its ``pyvenv.cfg``. Extension
modules are compiled per CPython minor version, so a 3.14 tree in a 3.11 process cannot supply
``pydantic_core``: the import dies with ``cannot import name 'ArgsKwargs'`` far from activation,
which reads like a product failure. Both cases here drive the real
``pm.environments.activate_dependencies`` and assert the resulting ``sys.path`` and ``PYTHONPATH``.

The contract: an interpreter only ever imports a tree built for its own ABI. A venv interpreter
keeps the packages it booted with when the committed generation is foreign (its own tree is
ABI-matched by construction, the generation's is not); PM's store Python carries none, so a
foreign generation is a broken install and activation refuses with the repair remedy. When the
recorded version matches, the committed generation stays authoritative -- including for a venv --
because that is the install's selection and no ABI is mixed.
"""

import json
import os
import sys

import pytest


def _running_version() -> tuple[int, int]:
    return sys.version_info[:2]


def _foreign_version() -> tuple[int, int]:
    major, minor = _running_version()
    return (major, minor + 1) if minor < 90 else (major + 1, 0)


def _generation(repo, version: tuple[int, int]):
    """Commit a generation for *repo* whose ``pyvenv.cfg`` records *version*."""
    from pm.environments import install_state_dir, runtime_facts_path, site_packages

    state = install_state_dir(repo)
    venv = state / "environments" / "generation" / "venv"
    venv.mkdir(parents=True)
    # pyvenv.cfg first: a POSIX layout keys its site-packages path off the recorded version.
    (venv / "pyvenv.cfg").write_text(f"version = {version[0]}.{version[1]}", encoding="utf-8")
    selected = site_packages(venv)
    selected.mkdir(parents=True)
    runtime_facts_path(repo).write_text(
        json.dumps({"packages": {"venv": {"environment": str(venv)}}}), encoding="utf-8")
    return venv.resolve(), selected.resolve()


@pytest.fixture
def scoped(monkeypatch):
    """Activation rewrites this process's import path and environment; keep every case local."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    for key in ("PYTHONPATH", "PATH", "VIRTUAL_ENV"):
        monkeypatch.setenv(key, os.environ.get(key, ""))


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    root = tmp_path / "repo"
    root.mkdir()
    return root


def test_a_venv_interpreter_keeps_its_own_packages_for_a_foreign_generation(
        repo, tmp_path, monkeypatch, scoped):
    from pm.environments import activate_dependencies

    venv, selected = _generation(repo, _foreign_version())
    own_site = tmp_path / "venv" / "Lib" / "site-packages"
    own_site.mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base"))
    kept = [str(repo.resolve()), str(own_site), *sys.path]
    sys.path[:] = kept
    monkeypatch.setenv("PYTHONPATH", str(repo.resolve()))

    activate_dependencies(repo)

    assert sys.path == kept, "the generation's tree must not enter a foreign interpreter's path"
    assert str(selected) not in sys.path
    assert os.environ["PYTHONPATH"] == str(repo.resolve()), "a foreign tree must not be exported"
    assert not (venv.parent / ".leases").exists(), "an unusable generation must not be leased"


def test_store_python_refuses_a_foreign_generation_with_the_repair_remedy(
        repo, tmp_path, monkeypatch, scoped):
    from pm.environments import activate_dependencies

    venv, selected = _generation(repo, _foreign_version())
    store = tmp_path / "tools" / "python-3.14"
    store.mkdir(parents=True)
    monkeypatch.setattr("pm.environments.store_root", lambda root: tmp_path / "tools")
    monkeypatch.setattr(sys, "prefix", str(store))
    monkeypatch.setattr(sys, "base_prefix", str(store))
    kept = [str(repo.resolve()), *sys.path]
    sys.path[:] = kept

    with pytest.raises(RuntimeError) as caught:
        activate_dependencies(repo)

    message = str(caught.value)
    assert f"built for Python {_foreign_version()[0]}.{_foreign_version()[1]}" in message
    assert "run `hermes pm repair`" in message
    assert sys.path == kept, "a refused selection must leave the import path untouched"
    assert str(selected) not in sys.path
    assert not (venv.parent / ".leases").exists()


def test_an_external_interpreter_keeps_its_own_packages_for_a_foreign_generation(
        repo, tmp_path, monkeypatch, scoped):
    """A Nix or system Python is not PM's store Python: it owns its original sys.path."""
    from pm.environments import activate_dependencies

    _, selected = _generation(repo, _foreign_version())
    external = tmp_path / "nix" / "python3"
    external.mkdir(parents=True)
    monkeypatch.setattr("pm.environments.store_root", lambda root: tmp_path / "tools")
    monkeypatch.setattr(sys, "prefix", str(external))
    monkeypatch.setattr(sys, "base_prefix", str(external))
    kept = [str(repo.resolve()), *sys.path]
    sys.path[:] = kept

    activate_dependencies(repo)

    assert sys.path == kept
    assert str(selected) not in sys.path


def test_store_python_still_selects_a_matching_generation(repo, tmp_path, monkeypatch, scoped):
    from pm.environments import activate_dependencies

    _, selected = _generation(repo, _running_version())
    store = tmp_path / "store-python"
    store.mkdir()
    monkeypatch.setattr(sys, "prefix", str(store))
    monkeypatch.setattr(sys, "base_prefix", str(store))
    boot_site = tmp_path / "boot" / "site-packages"
    boot_site.mkdir(parents=True)
    sys.path[:] = [str(repo.resolve()), str(boot_site), *sys.path]

    activate_dependencies(repo)

    assert sys.path[:2] == [str(repo.resolve()), str(selected)]
    assert str(boot_site) not in sys.path, "the tree it booted with is replaced, not extended"
    assert os.environ["PYTHONPATH"] == os.pathsep.join([str(repo.resolve()), str(selected)])


def test_a_venv_interpreter_still_selects_a_matching_generation(repo, tmp_path, monkeypatch, scoped):
    """The boundary: only the ABI decides, not ``sys.prefix`` alone."""
    from pm.environments import activate_dependencies

    _, selected = _generation(repo, _running_version())
    own_site = tmp_path / "venv" / "Lib" / "site-packages"
    own_site.mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base"))
    sys.path[:] = [str(repo.resolve()), str(own_site), *sys.path]

    activate_dependencies(repo)

    assert sys.path[:2] == [str(repo.resolve()), str(selected)]
    assert os.environ["PYTHONPATH"] == os.pathsep.join([str(repo.resolve()), str(selected)])
