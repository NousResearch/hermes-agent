"""RED tests for the uv-target venv-python fallback (#ISS-A).

When ``ensure_uv`` succeeds, ``_default_venv_install_target`` returns
``(["<uv>", "pip"], {"VIRTUAL_ENV": PROJECT_ROOT/"venv"})``.  If the venv
python at that path does NOT exist (dev checkout using a different venv,
managed install, venv not yet created), ``_resolve_install_target_python``
must fall back to ``sys.executable`` so the import-probe recovery can still
run in-process and clear the lazy-refresh marker — otherwise the marker is
stuck "indeterminate" forever and the user sees the three "Import probes
unavailable" warnings on every launch.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import hermes_cli.main as m


def test_resolve_falls_back_to_sys_executable_when_uv_venv_missing(tmp_path):
    """uv prefix + missing venv python -> fall back to sys.executable, not None."""
    fake_uv = tmp_path / "uv"
    fake_uv.write_text("", encoding="utf-8")
    # PROJECT_ROOT/venv/bin/python does NOT exist.
    prefix = [str(fake_uv), "pip"]
    env = {"VIRTUAL_ENV": str(tmp_path / "venv")}
    resolved = m._resolve_install_target_python(prefix, env)
    assert resolved is not None, (
        "must fall back to sys.executable when uv-targeted venv python is missing"
    )
    assert Path(resolved) == Path(sys.executable)


def test_resolve_prefers_venv_python_when_present(tmp_path):
    """When the uv-targeted venv python exists, it takes precedence."""
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    venv_python = venv_bin / "python"
    venv_python.write_text("", encoding="utf-8")
    fake_uv = tmp_path / "uv"
    fake_uv.write_text("", encoding="utf-8")
    prefix = [str(fake_uv), "pip"]
    env = {"VIRTUAL_ENV": str(tmp_path / "venv")}
    resolved = m._resolve_install_target_python(prefix, env)
    assert Path(resolved) == venv_python


def test_detect_broken_imports_runs_when_uv_venv_missing(tmp_path, monkeypatch):
    """End-to-end: probes must run (return []), not None, when venv is absent.

    This is the path that clears the lazy-refresh marker instead of leaving
    it "indeterminate" forever.
    """
    fake_uv = tmp_path / "uv"
    fake_uv.write_text("", encoding="utf-8")
    env = {"VIRTUAL_ENV": str(tmp_path / "venv")}

    # Stub subprocess.run: probe succeeds, prints no broken modules.
    from unittest.mock import MagicMock

    def fake_run(cmd, **kwargs):
        result = MagicMock()
        result.stdout = ""
        result.stderr = ""
        result.returncode = 0
        return result

    monkeypatch.setattr(m.subprocess, "run", fake_run)
    broken = m._detect_broken_lazy_refresh_imports([str(fake_uv), "pip"], env=env)
    assert broken == [], "probes must run in-process via sys.executable fallback"


def test_repair_not_indeterminate_when_uv_venv_missing(tmp_path, monkeypatch, capsys):
    """The full marker-recovery path must report 'healthy', not 'indeterminate'."""
    fake_uv = tmp_path / "uv"
    fake_uv.write_text("", encoding="utf-8")
    env = {"VIRTUAL_ENV": str(tmp_path / "venv")}

    from unittest.mock import MagicMock

    def fake_run(cmd, **kwargs):
        result = MagicMock()
        result.stdout = ""
        result.stderr = ""
        result.returncode = 0
        return result

    monkeypatch.setattr(m.subprocess, "run", fake_run)
    status = m._repair_venv_via_import_probes([str(fake_uv), "pip"], env=env)
    out = capsys.readouterr().out
    assert status == "healthy", "must not be 'indeterminate' when probes can run"
    assert "Import probes unavailable" not in out
