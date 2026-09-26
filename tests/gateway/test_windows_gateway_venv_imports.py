"""Regression tests for the Windows gateway venv overlay interpreter guard (#122736).

``gateway/run.py::_ensure_windows_gateway_venv_imports()`` prepends the in-tree venv's
site-packages without checking which Python built that venv. After the PM migration the
gateway runs on the store Python (3.14) while the in-tree venv is 3.11: the overlay makes
every import resolve from the stale 3.11 tree and compiled extensions die on ABI mismatch
(pydantic_core._pydantic_core -> hosted_room_worker crash-loop) instead of falling through
to the committed dependency environment.
"""

import os
import sys
from pathlib import Path

import pytest

from gateway.run import _venv_matches_running_interpreter

_MAJOR, _MINOR = sys.version_info[:2]
_FOREIGN_MINOR = _MINOR + 1


def _make_venv(tmp_path: Path, cfg: str | None) -> Path:
    venv = tmp_path / "venv"
    (venv / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)
    if cfg is not None:
        (venv / "pyvenv.cfg").write_text(cfg, encoding="utf-8")
    return venv


def test_cpython_version_spelling_names_this_interpreter(tmp_path):
    venv = _make_venv(tmp_path, f"home = /py\nversion = {_MAJOR}.{_MINOR}.9\n")
    assert _venv_matches_running_interpreter(venv) is True


def test_uv_version_info_spelling_names_this_interpreter(tmp_path):
    venv = _make_venv(tmp_path, f"uv = 0.11.14\nversion_info = {_MAJOR}.{_MINOR}\n")
    assert _venv_matches_running_interpreter(venv) is True


def test_cpython_version_spelling_with_foreign_minor_is_rejected(tmp_path):
    venv = _make_venv(tmp_path, f"home = /py\nversion = {_MAJOR}.{_FOREIGN_MINOR}.2\n")
    assert _venv_matches_running_interpreter(venv) is False


def test_uv_version_info_spelling_with_foreign_minor_is_rejected(tmp_path):
    venv = _make_venv(tmp_path, f"uv = 0.11.14\nversion_info = {_MAJOR}.{_FOREIGN_MINOR}\n")
    assert _venv_matches_running_interpreter(venv) is False


def test_missing_pyvenv_cfg_keeps_the_historical_overlay(tmp_path):
    assert _venv_matches_running_interpreter(_make_venv(tmp_path, None)) is True


def test_pyvenv_cfg_without_version_key_keeps_the_historical_overlay(tmp_path):
    venv = _make_venv(tmp_path, "home = /py\ninclude-system-site-packages = false\n")
    assert _venv_matches_running_interpreter(venv) is True


@pytest.mark.platforms("windows")
def test_overlay_skips_venv_built_for_a_foreign_interpreter(tmp_path, monkeypatch):
    """A venv naming another major.minor must stay OFF sys.path; a matching one is overlaid.

    Windows-only: the overlay early-returns on other hosts and uses the Windows venv layout
    (``Lib/site-packages``). ``project_root`` is pinned hermetically through the module's
    ``__file__`` so the real checkout's venv cannot leak into the candidates.
    """
    import gateway.run as gateway_run

    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(gateway_run, "__file__", str(tmp_path / "gateway" / "run.py"))

    foreign = _make_venv(
        tmp_path, f"uv = 0.11.14\nversion_info = {_MAJOR}.{_FOREIGN_MINOR}\n"
    )
    matching = None  # created in the second phase under the same root

    sys_path = list(sys.path)
    environ = dict(os.environ)
    try:
        gateway_run._ensure_windows_gateway_venv_imports()
        assert str((foreign / "Lib" / "site-packages").resolve()) not in sys.path, (
            "a venv built for a foreign interpreter must not be overlaid (#122736)"
        )

        # Positive control under the same root: the guard, not the layout, is what skipped.
        import shutil

        shutil.rmtree(foreign)
        matching = _make_venv(
            tmp_path, f"home = /py\nversion = {_MAJOR}.{_MINOR}.4\n"
        )
        gateway_run._ensure_windows_gateway_venv_imports()
        assert str((matching / "Lib" / "site-packages").resolve()) in sys.path
    finally:
        sys.path[:] = sys_path
        os.environ.clear()
        os.environ.update(environ)
