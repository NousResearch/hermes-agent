"""Workspace commands preserve PM's toolchain instead of consulting PATH."""
import importlib
import shutil
import subprocess

import pytest

from pm import workspace
from pm.package import InstallError


def test_workspace_uv_preserves_managed_tool_and_interpreter(monkeypatch, tmp_path):
    ensure = importlib.import_module("pm.ensure")
    monkeypatch.setattr(ensure, "uv", lambda **kwargs: ("managed/uv", {"UV_PYTHON": "managed/python", "UV_CACHE_DIR": str(tmp_path / "cache")}))
    monkeypatch.setattr(workspace, "_generate_pyproject", lambda *args, **kwargs: (tmp_path, False))
    def reject_path(*args, **kwargs):
        raise AssertionError("workspace commands must not resolve tools from PATH")
    monkeypatch.setattr(shutil, "which", reject_path)
    calls = []
    def run(cmd, **kwargs):
        calls.append((cmd, kwargs["env"]))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(workspace.subprocess, "run", run)
    workspace.lock_and_sync([], venv_dir=tmp_path / "venv", root=tmp_path)
    assert [cmd[1] for cmd, _ in calls] == ["lock", "sync"]
    assert all(cmd[0] == "managed/uv" and env["UV_PYTHON"] == "managed/python" for cmd, env in calls)


def test_workspace_uv_missing_toolchain_never_falls_back(monkeypatch, tmp_path):
    ensure = importlib.import_module("pm.ensure")
    monkeypatch.setattr(ensure, "uv", lambda **kwargs: (None, {}))
    monkeypatch.setattr(workspace, "_generate_pyproject", lambda *args, **kwargs: (tmp_path, False))
    monkeypatch.setattr(shutil, "which", lambda name: "developer/uv")
    with pytest.raises(InstallError, match="PM's uv and Python"):
        workspace.lock_and_sync([], venv_dir=tmp_path / "venv", root=tmp_path)
