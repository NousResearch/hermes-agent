"""Tests for the Windows gateway venv-injection guard (``_ensure_windows_gateway_venv_imports``).

Bug class (#122183, dup #122556/#122736/#122400): the Windows gateway injected the pre-PM
in-tree venv's site-packages (built for a different Python) into a managed-runtime process,
so the first late third-party import (``tui_gateway.server`` → ``pydantic_core``) crashed the
hosted_room_worker with ``ModuleNotFoundError: No module named 'pydantic_core._pydantic_core'``.

Contract after the fix:
1. A candidate venv whose ``pyvenv.cfg`` version does not match this interpreter is never
   injected (an ABI mismatch there is the crash itself).
2. The committed PM environment (``pm.environments.committed_venv``) is preferred over the
   ambient/in-tree candidates.
3. A matching-version venv is still injected — legacy behavior for installs without PM.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

import gateway.run as gateway_run

pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="_ensure_windows_gateway_venv_imports is a Windows-only startup path (#122183)",
)


@pytest.fixture
def outside_home_tmp():
    """A temp dir guaranteed OUTSIDE the real Hermes home.

    The suite's home_io_guard refuses any file I/O under the real home, and on some
    developer machines TMPDIR itself points into ``<home>/.hermes/cache`` — pytest's
    default ``tmp_path`` would then trip the guard on the first ``Path.exists()``.
    Falls back to the per-user local temp when the system temp is inside the home.
    """
    from hermes_constants import get_default_hermes_root

    base = Path(tempfile.gettempdir()).resolve()
    try:
        home = Path(get_default_hermes_root()).resolve()
        if base == home or home in base.parents:
            base = (Path(os.environ.get("LOCALAPPDATA") or home.parent) / "Temp").resolve()
            base.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    d = Path(tempfile.mkdtemp(prefix="venv-guard-test-", dir=base))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _make_venv(root: Path, version: str) -> Path:
    """A minimal fake venv: pyvenv.cfg + Lib/site-packages (or lib/pythonX.Y/site-packages)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyvenv.cfg").write_text(f"[config]\nversion = {version}\n", encoding="utf-8")
    if sys.platform == "win32":
        site = root / "Lib" / "site-packages"
    else:
        site = root / f"lib/python{version.rpartition('.')[0]}/site-packages"
    site.mkdir(parents=True, exist_ok=True)
    return site


def _call(monkeypatch, outside_home_tmp, *, venv_root=None, pm_env=None):
    """Run the injection with a controlled environment; return the sys.path entries it added."""
    monkeypatch.setenv("VIRTUAL_ENV", str(venv_root)) if venv_root else monkeypatch.delenv(
        "VIRTUAL_ENV", raising=False
    )
    monkeypatch.delenv("PYTHONPATH", raising=False)
    if pm_env is not None:
        monkeypatch.setattr(
            "pm.environments.committed_venv", lambda project_root: pm_env
        )
    else:
        monkeypatch.setattr("pm.environments.committed_venv", lambda project_root: None)

    saved = list(sys.path)
    # Stub ``site.addsitedir`` with the same sys.path-insertion contract but no I/O: the real
    # one stats EVERY sys.path entry (_init_pathinfo), and on default installs the interpreter's
    # own stdlib lives under the real Hermes home, which the suite's home_io_guard refuses —
    # environment-dependently (depends on how sys.prefix resolves). The unit under test is the
    # candidate-selection logic (which venvs are rejected/selected), not stdlib's addsitedir.
    added_by_stub: list[str] = []

    def _fake_addsitedir(sdir):
        entry = str(sdir)
        if entry not in sys.path:
            sys.path.append(entry)
        added_by_stub.append(entry)

    monkeypatch.setattr("site.addsitedir", _fake_addsitedir)
    try:
        gateway_run._ensure_windows_gateway_venv_imports()
        added = [entry for entry in sys.path if entry not in saved]
    finally:
        sys.path[:] = saved
    return added


def test_version_mismatched_venv_is_never_injected(monkeypatch, outside_home_tmp):
    site = _make_venv(outside_home_tmp / "old_venv", "9.9.99")
    added = _call(monkeypatch, outside_home_tmp, venv_root=outside_home_tmp / "old_venv")
    assert str(site) not in added, "a mismatched-ABI venv must not reach sys.path"


def test_matching_venv_still_injected(monkeypatch, outside_home_tmp):
    version = f"{sys.version_info[0]}.{sys.version_info[1]}.99"
    site = _make_venv(outside_home_tmp / "same_venv", version)
    added = _call(monkeypatch, outside_home_tmp, venv_root=outside_home_tmp / "same_venv")
    assert str(site) in added, "legacy single-venv installs keep their injection"


def test_pm_environment_wins_over_mismatched_ambient_venv(monkeypatch, outside_home_tmp):
    bad_venv = _make_venv(outside_home_tmp / "old_venv", "9.9.99")
    pm_site = _make_venv(outside_home_tmp / "pm_venv", f"{sys.version_info[0]}.{sys.version_info[1]}.0")
    added = _call(
        monkeypatch,
        outside_home_tmp,
        venv_root=outside_home_tmp / "old_venv",
        pm_env=outside_home_tmp / "pm_venv",
    )
    assert str(pm_site) in added, "the committed PM environment must be preferred"
    assert str(bad_venv) not in added, "the mismatched ambient venv must stay out"
