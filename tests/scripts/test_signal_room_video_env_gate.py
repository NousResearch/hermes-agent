from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_video_env_gate.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_video_env_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_environment_gate_passes_when_blender_and_hyperframes_stack_are_available(monkeypatch) -> None:
    monkeypatch.delenv("SIGNAL_ROOM_MOHO_EXE", raising=False)
    monkeypatch.delenv("SIGNAL_ROOM_CAVALRY_EXE", raising=False)
    module = load_module()
    available = {
        "blender": "/usr/bin/blender",
        "npx": "/usr/bin/npx",
        "ffmpeg": "/usr/bin/ffmpeg",
    }

    result = module.evaluate_environment(lambda command: available.get(command))

    assert result["passed"] is True
    assert result["render_mode"] == "local_blender"
    assert result["blockers"] == []
    assert result["tools"]["blender"]["available"] is True
    assert result["tools"]["moho"]["available"] is False
    assert result["tools"]["hyperframes"]["available"] is True


def test_environment_gate_passes_with_configured_windows_moho_and_cavalry(monkeypatch) -> None:
    monkeypatch.setenv("SIGNAL_ROOM_MOHO_EXE", r"C:\Program Files\Moho 14\Moho.exe")
    monkeypatch.setenv("SIGNAL_ROOM_CAVALRY_EXE", r"C:\Program Files\Cavalry\Cavalry.exe")
    module = load_module()
    available = {
        "npx": "/usr/bin/npx",
        "ffmpeg": "/usr/bin/ffmpeg",
    }

    result = module.evaluate_environment(lambda command: available.get(command))

    assert result["passed"] is True
    assert result["render_mode"] == "local_moho"
    assert result["tools"]["moho"]["available"] is True
    assert result["tools"]["moho"]["source"] == "SIGNAL_ROOM_MOHO_EXE"
    assert result["tools"]["moho"]["automation_mode"] == "windows_scheduled_task_bridge"
    assert result["tools"]["cavalry"]["available"] is True
    assert result["tools"]["cavalry"]["source"] == "SIGNAL_ROOM_CAVALRY_EXE"
    assert result["tools"]["cavalry"]["role"] == "optional_motion_graphics"


def test_environment_gate_reports_external_pose_export_when_blender_and_moho_are_missing(monkeypatch) -> None:
    monkeypatch.delenv("SIGNAL_ROOM_MOHO_EXE", raising=False)
    monkeypatch.delenv("SIGNAL_ROOM_CAVALRY_EXE", raising=False)
    module = load_module()
    available = {
        "npx": "/usr/bin/npx",
        "ffmpeg": "/usr/bin/ffmpeg",
    }

    result = module.evaluate_environment(lambda command: available.get(command))

    assert result["passed"] is False
    assert result["render_mode"] == "external_pose_export_required"
    assert "Blender or Moho is required for local character pose rendering" in result["blockers"]
    assert result["tools"]["blender"]["available"] is False
    assert result["tools"]["moho"]["available"] is False


def test_environment_gate_reports_hyperframes_blockers_separately(monkeypatch) -> None:
    monkeypatch.delenv("SIGNAL_ROOM_MOHO_EXE", raising=False)
    monkeypatch.delenv("SIGNAL_ROOM_CAVALRY_EXE", raising=False)
    module = load_module()

    result = module.evaluate_environment(lambda command: None)

    assert result["passed"] is False
    assert "npx is required for HyperFrames preview/render commands" in result["blockers"]
    assert "ffmpeg is required for final video rendering" in result["blockers"]
