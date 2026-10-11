"""ABI gate for the Windows gateway venv-import shim (``gateway/run.py``).

``_ensure_windows_gateway_venv_imports`` exists so a detached Windows gateway run
still sees the Hermes packages when the launcher did not preserve PYTHONPATH. On a
pm install the store python already carries its own dependency generation, and the
legacy ``<repo>/venv`` belongs to a DIFFERENT interpreter: injecting it shadows the
correct tree, so pure-Python imports succeed and the ``cp3xx`` extension then dies
(``No module named 'pydantic_core._pydantic_core'``) — taking MCP discovery, the
hosted room worker and Group Chat with it.
"""

import os
import sys
from pathlib import Path

import pytest

from gateway.run import _ensure_windows_gateway_venv_imports, _venv_matches_running_python
import gateway.run as run_mod

_RUNNING = f"{sys.version_info[0]}.{sys.version_info[1]}"
_OTHER = f"{sys.version_info[0]}.{sys.version_info[1] + 1}"


def _fake_repo(tmp_path: Path, monkeypatch) -> Path:
    """Point the shim's ``project_root`` at a scratch tree.

    The shim also probes ``<repo>/venv``; on a developer box the real checkout's venv
    may legitimately share the running ABI, so the candidate list must not depend on
    it. Rewriting ``gateway.run.__file__`` keeps the assertions interpreter-agnostic.
    """
    repo = tmp_path / "repo"
    (repo / "gateway").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_mod, "__file__", str(repo / "gateway" / "run.py"))
    return repo


@pytest.fixture
def clean_path():
    """Restore ``sys.path`` in place — the shim mutates the same list object."""
    before = list(sys.path)
    yield
    sys.path[:] = before


def _fake_venv(root: Path, version_info: str | None) -> Path:
    venv = root / "fakevenv"
    (venv / "Lib" / "site-packages").mkdir(parents=True)
    if version_info is not None:
        (venv / "pyvenv.cfg").write_text(
            "home = C:\\python\n"
            f"version_info = {version_info}\n",
            encoding="utf-8",
        )
    return venv


def test_matching_abi_accepted(tmp_path):
    assert _venv_matches_running_python(_fake_venv(tmp_path, _RUNNING))


def test_foreign_abi_rejected(tmp_path):
    """A venv built for another minor version must never qualify."""
    assert not _venv_matches_running_python(_fake_venv(tmp_path, _OTHER))


def test_missing_cfg_rejected(tmp_path):
    assert not _venv_matches_running_python(_fake_venv(tmp_path, None))


def test_unversioned_cfg_rejected(tmp_path):
    venv = tmp_path / "nopevenv"
    (venv / "Lib" / "site-packages").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = C:\\python\n", encoding="utf-8")
    assert not _venv_matches_running_python(venv)


@pytest.mark.skipif(sys.platform != "win32", reason="windows-only shim")
def test_shim_skips_foreign_venv(tmp_path, monkeypatch, clean_path):
    """The whole regression: no foreign site-packages on sys.path, env untouched."""
    repo = _fake_repo(tmp_path, monkeypatch)
    venv = _fake_venv(tmp_path, _OTHER)
    site = venv / "Lib" / "site-packages"
    monkeypatch.setenv("VIRTUAL_ENV", str(venv))
    monkeypatch.delenv("PYTHONPATH", raising=False)

    _ensure_windows_gateway_venv_imports()

    assert str(site) not in sys.path
    assert os.environ["VIRTUAL_ENV"] == str(venv)  # not rewritten to <repo>/venv
    assert not os.environ.get("PYTHONPATH")
    assert repo.exists()  # the shim ran against the fake repo, not the real checkout


@pytest.mark.skipif(sys.platform != "win32", reason="windows-only shim")
def test_shim_injects_matching_venv(tmp_path, monkeypatch, clean_path):
    """A same-ABI venv keeps working — the shim's original purpose survives."""
    _fake_repo(tmp_path, monkeypatch)
    venv = _fake_venv(tmp_path, _RUNNING)
    site = venv / "Lib" / "site-packages"
    monkeypatch.setenv("VIRTUAL_ENV", str(venv))
    monkeypatch.delenv("PYTHONPATH", raising=False)

    _ensure_windows_gateway_venv_imports()

    assert str(site) in sys.path
    assert os.environ["VIRTUAL_ENV"] == str(venv)
    assert str(site) in os.environ.get("PYTHONPATH", "")
