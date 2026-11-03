"""Gateway Windows venv import guards (#122183, #57467, #139085)."""

import logging
import os
import sys

import pytest


def _venv(path, version):
    (path / "Lib" / "site-packages").mkdir(parents=True)
    (path / "pyvenv.cfg").write_text(f"home = C:\\Python\nversion_info = {version}\n", encoding="utf-8")
    return path


@pytest.mark.platforms("windows")
def test_committed_generation_blocks_the_legacy_venv_overlay(tmp_path, monkeypatch):
    """A PM-managed Windows gateway never overlays the leftover pre-PM venv.

    hermes_bootstrap already activated the store Python onto the committed
    generation; the late ``_ensure_windows_gateway_venv_imports`` overlay used
    to prepend ``<root>/venv`` (or ``VIRTUAL_ENV``) regardless, loading a cp311
    ``pydantic_core`` into 3.14.
    """
    import gateway.run as gateway_run

    root = tmp_path / "root"
    (root / "gateway").mkdir(parents=True)
    legacy = root / "venv"
    (legacy / "Lib" / "site-packages").mkdir(parents=True)
    generation = tmp_path / "gen"
    (generation / "Lib" / "site-packages").mkdir(parents=True)

    monkeypatch.setattr("pm.environments.committed_venv", lambda _root: generation)
    monkeypatch.setattr(gateway_run, "__file__", str(root / "gateway" / "run.py"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("VIRTUAL_ENV", str(legacy))
    monkeypatch.delenv("PYTHONPATH", raising=False)

    gateway_run._ensure_windows_gateway_venv_imports()

    legacy_site = str((legacy / "Lib" / "site-packages").resolve())
    assert legacy_site not in sys.path
    assert os.environ["VIRTUAL_ENV"] == str(legacy)
    assert "PYTHONPATH" not in os.environ


_OTHER_MINOR = 11 if sys.version_info.minor != 11 else 12


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("version_line", [
    pytest.param(f"version_info = {sys.version_info.major}.{_OTHER_MINOR}.9\n", id="foreign-abi"),
    pytest.param(None, id="no-version-metadata"),
])
def test_legacy_venv_of_unknown_or_foreign_abi_is_never_overlaid(tmp_path, monkeypatch, caplog, version_line):
    """#139085: with nothing committed and VIRTUAL_ENV unset (PM launchers and
    ``activate_dependencies`` pop it), the overlay fell back to ``<root>/venv`` and
    ``addsitedir``-ed a cp311 tree into a 3.14 gateway, which then died importing
    ``pydantic_core`` when the Group Chat worker imported ``tui_gateway.server``.
    A venv whose Python version can't be read fails closed the same way; both skips are logged."""
    import gateway.run as gateway_run

    root = tmp_path / "root"
    (root / "gateway").mkdir(parents=True)
    legacy = root / "venv"
    (legacy / "Lib" / "site-packages").mkdir(parents=True)
    if version_line is not None:
        (legacy / "pyvenv.cfg").write_text("home = C:\\Python\n" + version_line, encoding="utf-8")

    monkeypatch.setattr("pm.environments.committed_venv", lambda _root: None)
    monkeypatch.setattr(gateway_run, "__file__", str(root / "gateway" / "run.py"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)

    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        gateway_run._ensure_windows_gateway_venv_imports()

    legacy_site = legacy / "Lib" / "site-packages"
    assert not any(entry and os.path.samefile(entry, legacy_site)
                   for entry in sys.path if os.path.isdir(entry))
    assert "VIRTUAL_ENV" not in os.environ
    assert "Skipping venv overlay" in caplog.text


@pytest.mark.platforms("windows")
def test_same_abi_venv_is_still_overlaid(tmp_path, monkeypatch):
    """A pre-PM venv built for the running interpreter keeps its overlay."""
    import gateway.run as gateway_run

    root = tmp_path / "root"
    (root / "gateway").mkdir(parents=True)
    venv = _venv(root / "venv", "{}.{}.{}".format(*sys.version_info[:3]))

    monkeypatch.setattr("pm.environments.committed_venv", lambda _root: None)
    monkeypatch.setattr(gateway_run, "__file__", str(root / "gateway" / "run.py"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)

    gateway_run._ensure_windows_gateway_venv_imports()

    assert str((venv / "Lib" / "site-packages").resolve()) in sys.path
    assert os.environ["VIRTUAL_ENV"] == str(venv.resolve())


@pytest.mark.platforms("windows")
def test_venv_import_setup_scopes_to_sys_path_not_global_environ(
    tmp_path, monkeypatch
):
    """The venv import setup must not leak into the global environment (#57467).

    ``_ensure_windows_gateway_venv_imports`` patches the CURRENT process via
    ``sys.path``/``site.addsitedir`` only. Mutating ``os.environ["PYTHONPATH"]``
    there poisoned every child process spawned from the gateway session —
    non-Hermes Python tools resolved Hermes' venv packages ahead of their own.
    """
    import gateway.run as gateway_run

    root = tmp_path / "root"
    (root / "gateway").mkdir(parents=True)
    venv = _venv(root / "venv", "{}.{}.{}".format(*sys.version_info[:3]))
    site_packages = venv / "Lib" / "site-packages"

    monkeypatch.setattr("pm.environments.committed_venv", lambda _root: None)
    monkeypatch.setattr(gateway_run, "__file__", str(root / "gateway" / "run.py"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("VIRTUAL_ENV", str(venv))
    monkeypatch.setenv("PYTHONPATH", "already-there")

    gateway_run._ensure_windows_gateway_venv_imports()

    # In-process import surface: project root + venv site-packages on sys.path.
    assert str(root.resolve()) in sys.path
    assert str(site_packages) in sys.path
    # VIRTUAL_ENV stays — standard, scoped, no cross-process poisoning.
    assert os.environ["VIRTUAL_ENV"] == str(venv.resolve())
    # The inherited PYTHONPATH must be untouched — no Hermes paths prepended.
    assert os.environ["PYTHONPATH"] == "already-there"
    assert str(site_packages) not in os.environ["PYTHONPATH"]
    assert str(root) not in os.environ["PYTHONPATH"]
