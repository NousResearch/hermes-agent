"""Tests for ``_wait_for_interpreter_venv_ready`` in ``hermes_cli/main.py``.

During ``hermes update`` the managed-uv path can rebuild the project venv
(rmtree + ``uv venv``) before the desktop-rebuild and profile-skills-sync
steps spawn ``sys.executable``. If those children fire while the venv is
mid-rewrite, the interpreter launcher aborts with ``No pyvenv.cfg file`` and
the step spuriously "fails" on an otherwise-successful update. The helper
waits for the marker to settle first.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from hermes_cli.main import _wait_for_interpreter_venv_ready


def _make_fake_venv(tmp_path: Path, *, with_cfg: bool) -> Path:
    """Create a venv-shaped dir and return the interpreter path inside it."""
    bin_name = "Scripts" if os.name == "nt" else "bin"
    bin_dir = tmp_path / bin_name
    bin_dir.mkdir(parents=True)
    py = bin_dir / ("python.exe" if os.name == "nt" else "python")
    py.write_text("#!/bin/sh\n")
    if with_cfg:
        (tmp_path / "pyvenv.cfg").write_text("home = /usr\n")
    return py


class TestWaitForInterpreterVenvReady:
    def test_intact_venv_returns_immediately(self, tmp_path, monkeypatch):
        py = _make_fake_venv(tmp_path, with_cfg=True)
        monkeypatch.setattr("sys.executable", str(py))
        t0 = time.monotonic()
        assert _wait_for_interpreter_venv_ready(timeout=5) is True
        assert time.monotonic() - t0 < 0.5

    def test_non_venv_interpreter_returns_immediately(self, tmp_path, monkeypatch):
        # A bare interpreter whose parent.parent has no bin/Scripts marker
        # dir is not venv-hosted; pyvenv.cfg is irrelevant.
        sys_py = tmp_path / "usr" / "bin" / "python"
        sys_py.parent.mkdir(parents=True)
        sys_py.write_text("#!/bin/sh\n")
        # Ensure parent.parent (tmp_path/usr) has no bin sibling shaped like a venv
        monkeypatch.setattr("sys.executable", str(sys_py))
        # parent.parent == tmp_path/usr; its "bin" child IS tmp_path/usr/bin
        # which exists — so this would look venv-ish. Use a deeper layout
        # where parent.parent has no bin marker:
        deep = tmp_path / "opt" / "py3" / "real" / "python"
        deep.parent.mkdir(parents=True)
        deep.write_text("#!/bin/sh\n")
        monkeypatch.setattr("sys.executable", str(deep))
        t0 = time.monotonic()
        assert _wait_for_interpreter_venv_ready(timeout=5) is True
        assert time.monotonic() - t0 < 0.5

    def test_waits_for_cfg_to_appear(self, tmp_path, monkeypatch):
        py = _make_fake_venv(tmp_path, with_cfg=False)
        monkeypatch.setattr("sys.executable", str(py))

        def _write_cfg_later():
            time.sleep(0.6)
            (tmp_path / "pyvenv.cfg").write_text("home = /usr\n")

        th = threading.Thread(target=_write_cfg_later)
        th.start()
        try:
            t0 = time.monotonic()
            assert _wait_for_interpreter_venv_ready(timeout=5) is True
            elapsed = time.monotonic() - t0
        finally:
            th.join()
        assert 0.5 < elapsed < 2.0

    def test_returns_false_when_cfg_never_appears(self, tmp_path, monkeypatch):
        py = _make_fake_venv(tmp_path, with_cfg=False)
        monkeypatch.setattr("sys.executable", str(py))
        t0 = time.monotonic()
        assert _wait_for_interpreter_venv_ready(timeout=1) is False
        assert 0.9 < time.monotonic() - t0 < 1.6
