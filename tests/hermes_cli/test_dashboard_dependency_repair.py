"""Tests for dashboard dependency self-healing on broken uvicorn installs."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli.main import (
    _attempt_dashboard_uvicorn_repair,
    _ensure_dashboard_runtime_deps,
)


def test_ensure_dashboard_runtime_deps_repairs_broken_uvicorn_install():
    calls = {"uvicorn": 0}
    real_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name == "fastapi":
            return object()
        if name == "uvicorn":
            calls["uvicorn"] += 1
            if calls["uvicorn"] == 1:
                raise ImportError("No module named 'uvicorn.supervisors'")
            return object()
        return real_import_module(name)

    with patch("importlib.import_module", side_effect=fake_import_module), patch(
        "subprocess.run",
        return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
    ) as mock_run:
        assert _ensure_dashboard_runtime_deps() is None

    assert calls["uvicorn"] == 2
    cmd = mock_run.call_args.args[0]
    assert cmd[1:] == [
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-cache-dir",
        "uvicorn[standard]==0.41.0",
    ]


def test_ensure_dashboard_runtime_deps_does_not_repair_missing_fastapi():
    real_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name == "fastapi":
            raise ImportError("No module named 'fastapi'")
        return real_import_module(name)

    with patch("importlib.import_module", side_effect=fake_import_module), patch(
        "subprocess.run"
    ) as mock_run:
        error = _ensure_dashboard_runtime_deps()

    assert isinstance(error, ImportError)
    assert "fastapi" in str(error)
    mock_run.assert_not_called()


def test_attempt_dashboard_uvicorn_repair_reports_failure(capsys):
    with patch(
        "subprocess.run",
        return_value=SimpleNamespace(returncode=1, stdout="", stderr="boom"),
    ):
        repaired = _attempt_dashboard_uvicorn_repair(
            ImportError("No module named 'uvicorn.supervisors'")
        )

    assert repaired is False
    out = capsys.readouterr().out
    assert "Automatic uvicorn repair failed." in out
    assert "boom" in out
