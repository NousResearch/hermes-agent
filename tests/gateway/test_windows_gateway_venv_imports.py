"""Ownership gate for ``_ensure_windows_gateway_venv_imports`` (#122324).

The Windows gateway bootstrap used to prepend ``project_root/venv/Lib/site-packages`` to
``sys.path`` unconditionally. When a pre-PM in-tree 3.11 venv is still on disk while the
gateway runs from the managed store python (3.14), that put a cp311-built ``pydantic_core``
in front of the managed one — ``ImportError`` on the first extension import and a
hosted-room worker restart loop. ``sys.platform`` is faked so the gate's branch logic runs
on every host; all paths come from ``tmp_path``, so no real site-packages can leak in.
"""

import gateway.run as gateway_run


def _fake_venv(root, name="venv"):
    venv_dir = root / name
    site_packages = venv_dir / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)
    return venv_dir, site_packages


def _fake_project(tmp_path):
    project = tmp_path / "hermes-agent"
    (project / "gateway").mkdir(parents=True)
    return project


def test_own_venv_site_packages_still_injected(monkeypatch, tmp_path):
    project = _fake_project(tmp_path)
    venv_dir, site_packages = _fake_venv(project)

    monkeypatch.setattr(gateway_run.sys, "platform", "win32")
    monkeypatch.setattr(gateway_run.sys, "prefix", str(venv_dir))
    monkeypatch.setattr(gateway_run.sys, "path", ["existing"])
    monkeypatch.setattr(gateway_run, "__file__", str(project / "gateway" / "run.py"))
    monkeypatch.setenv("VIRTUAL_ENV", str(venv_dir))
    monkeypatch.setenv("PYTHONPATH", "already-there")

    gateway_run._ensure_windows_gateway_venv_imports()

    assert gateway_run.sys.path[:2] == [str(project), str(site_packages)]
    assert gateway_run.os.environ["VIRTUAL_ENV"] == str(venv_dir.resolve())
    pythonpath = gateway_run.os.environ["PYTHONPATH"].split(gateway_run.os.pathsep)
    assert pythonpath[:3] == [str(project), str(site_packages), "already-there"]


def test_foreign_venv_site_packages_not_injected(monkeypatch, tmp_path):
    """Regression: a launcher re-rendered onto the store python (no ``VIRTUAL_ENV``) still
    finds the leftover in-tree venv on disk — its site-packages must not land on the
    running interpreter's ``sys.path``."""
    project = _fake_project(tmp_path)
    _venv_dir, site_packages = _fake_venv(project)
    store_python = tmp_path / "store" / "python-3.14"
    store_python.mkdir(parents=True)

    monkeypatch.setattr(gateway_run.sys, "platform", "win32")
    monkeypatch.setattr(gateway_run.sys, "prefix", str(store_python))
    monkeypatch.setattr(gateway_run.sys, "path", ["existing"])
    monkeypatch.setattr(gateway_run, "__file__", str(project / "gateway" / "run.py"))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)

    gateway_run._ensure_windows_gateway_venv_imports()

    assert str(site_packages) not in gateway_run.sys.path
    assert gateway_run.sys.path == ["existing"]
    assert "VIRTUAL_ENV" not in gateway_run.os.environ
    assert "PYTHONPATH" not in gateway_run.os.environ


def test_own_project_venv_chosen_over_foreign_virtual_env(monkeypatch, tmp_path):
    """The ``VIRTUAL_ENV`` candidate is skipped the same way when it names another
    environment; the fallback in-tree venv is injected when it IS the running one."""
    project = _fake_project(tmp_path)
    own_venv, own_site_packages = _fake_venv(project)
    foreign_venv, foreign_site_packages = _fake_venv(tmp_path, name="other-env")

    monkeypatch.setattr(gateway_run.sys, "platform", "win32")
    monkeypatch.setattr(gateway_run.sys, "prefix", str(own_venv))
    monkeypatch.setattr(gateway_run.sys, "path", ["existing"])
    monkeypatch.setattr(gateway_run, "__file__", str(project / "gateway" / "run.py"))
    monkeypatch.setenv("VIRTUAL_ENV", str(foreign_venv))
    monkeypatch.delenv("PYTHONPATH", raising=False)

    gateway_run._ensure_windows_gateway_venv_imports()

    assert str(foreign_site_packages) not in gateway_run.sys.path
    assert str(own_site_packages) in gateway_run.sys.path
    assert gateway_run.os.environ["VIRTUAL_ENV"] == str(own_venv.resolve())
