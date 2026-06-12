"""Behavior tests for the QuestFrame FH6VR Hermes plugin."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from plugins.questframe_fh6vr import core


@pytest.fixture
def plugin_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_status_without_launcher_configured(plugin_env):
    payload = json.loads(core.handle_status())
    assert payload["ok"] is True
    assert payload["plugin"] == "questframe-fh6vr"
    assert payload["launcher_exists"] is False
    assert "questframe_support_report" in payload["available_tools"]
    assert "questframe_depth_surface_selftest" in payload["available_tools"]
    assert "questframe_depth_reader_selftest" in payload["available_tools"]


def test_save_setup_values_writes_config(plugin_env, monkeypatch):
    launcher = plugin_env / "bin" / "FH6VR.Launcher.exe"
    launcher.parent.mkdir(parents=True)
    launcher.write_bytes(b"stub")

    saved = core.save_setup_values(
        {
            "launcher_exe": str(launcher),
            "vcc_project_roots": [str(plugin_env / "unity")],
        }
    )
    assert saved["ok"] is True
    assert "launcher_exe" in saved["saved"]

    status = core.status()
    assert status["launcher_exe"] == str(launcher)
    assert status["launcher_exists"] is True


def test_run_launcher_missing_path():
    result = core.run_launcher("preflight", launcher_exe=str(Path("/missing/FH6VR.Launcher.exe")))
    assert result["ok"] is False
    assert "does not exist" in result["error"]


def test_run_launcher_success_parses_json(tmp_path):
    launcher = tmp_path / "FH6VR.Launcher.exe"
    launcher.write_text("stub", encoding="utf-8")

    def fake_run(argv, **kwargs):
        assert argv[1] == "preflight"
        assert "--json" in argv
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout='{"overall":"Pass"}',
            stderr="",
        )

    with patch("plugins.questframe_fh6vr.core.subprocess.run", side_effect=fake_run):
        result = core.run_launcher(
            "preflight",
            launcher_exe=str(launcher),
            extra_args=["--json"],
        )

    assert result["ok"] is True
    assert result["json"]["overall"] == "Pass"


def test_support_report_uses_default_redacted_paths(tmp_path):
    launcher = tmp_path / "FH6VR.Launcher.exe"
    launcher.write_text("stub", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout='{"overall":"Pass"}', stderr="")

    with patch("plugins.questframe_fh6vr.core.subprocess.run", side_effect=fake_run):
        result = core.support_report(launcher_exe=str(launcher), include_live_openxr=True)

    assert result["ok"] is True
    assert result["redacted_by_default"] is True
    assert "--include-live-openxr" in calls[0]
    assert "--write-json" in calls[0]
    assert "--write-html" in calls[0]


def test_scan_unity_project_detects_vrchat_risks(tmp_path):
    project = tmp_path / "demo"
    (project / "Assets").mkdir(parents=True)
    packages = project / "Packages"
    packages.mkdir()
    (packages / "manifest.json").write_text(
        json.dumps(
            {
                "dependencies": {
                    "com.vrchat.avatars": "3.7.0",
                    "com.vrchat.worlds": "3.7.0",
                }
            }
        ),
        encoding="utf-8",
    )

    report = core.scan_unity_project(project)
    assert report["unity_project"] is True
    assert len(report["detected_packages"]) == 2
    assert "both Avatar and World SDK packages detected" in report["risks"]


def test_handle_slash_support_report_routes_to_handler():
    with patch(
        "plugins.questframe_fh6vr.core.handle_support_report",
        return_value='{"ok": true}',
    ) as handler:
        payload = core.handle_slash("support-report")
    handler.assert_called_once()
    assert '"ok": true' in payload


def test_handle_live_capture_selftest_passes_window_capture_flag(tmp_path):
    launcher = tmp_path / "FH6VR.Launcher.exe"
    launcher.write_text("stub", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout='{"Status":"Pass"}', stderr="")

    with patch("plugins.questframe_fh6vr.core.subprocess.run", side_effect=fake_run):
        payload = json.loads(
            core.handle_live_capture_selftest(
                {"launcher_exe": str(launcher), "attempt_window_capture": True}
            )
        )

    assert payload["ok"] is True
    assert "fh6-live-capture-selftest" in calls[0][1]
    assert "--attempt-window-capture" in calls[0]


def test_handle_depth_surface_selftest_runs_launcher_gate(tmp_path):
    launcher = tmp_path / "FH6VR.Launcher.exe"
    launcher.write_text("stub", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout='{"Status":"Pass"}', stderr="")

    with patch("plugins.questframe_fh6vr.core.subprocess.run", side_effect=fake_run):
        payload = json.loads(
            core.handle_depth_surface_selftest({"launcher_exe": str(launcher)})
        )

    assert payload["ok"] is True
    assert "fh6-depth-surface-selftest" in calls[0][1]
    assert "--json" in calls[0]


def test_handle_depth_reader_selftest_runs_fixture_gate(tmp_path):
    launcher = tmp_path / "FH6VR.Launcher.exe"
    launcher.write_text("stub", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout='{"Status":"Pass"}', stderr="")

    with patch("plugins.questframe_fh6vr.core.subprocess.run", side_effect=fake_run):
        payload = json.loads(
            core.handle_depth_reader_selftest(
                {"launcher_exe": str(launcher), "fixture": True}
            )
        )

    assert payload["ok"] is True
    assert "fh6-depth-reader-selftest" in calls[0][1]
    assert "--json" in calls[0]
    assert "--fixture" in calls[0]


def test_handle_slash_depth_reader_selftest_passes_fixture_flag():
    with patch(
        "plugins.questframe_fh6vr.core.handle_depth_reader_selftest",
        return_value='{"ok": true}',
    ) as handler:
        payload = core.handle_slash("depth-reader-selftest --fixture")

    handler.assert_called_once_with({"fixture": True})
    assert '"ok": true' in payload


def test_register_exposes_all_documented_tools():
    from plugins.questframe_fh6vr import _TOOLS

    tool_names = {name for name, *_rest in _TOOLS}
    expected = {
        "questframe_status",
        "questframe_setup",
        "questframe_fh6vr_preflight",
        "questframe_rtx3060_profiles",
        "questframe_rtx3060_selftest",
        "questframe_session_readiness",
        "questframe_graphics_session",
        "questframe_frame_loop",
        "questframe_dibr_swapchain",
        "questframe_fh6_capture_preflight",
        "questframe_live_capture_selftest",
        "questframe_depth_surface_selftest",
        "questframe_depth_reader_selftest",
        "questframe_support_report",
        "questframe_unity_scan",
    }
    assert expected.issubset(tool_names)
