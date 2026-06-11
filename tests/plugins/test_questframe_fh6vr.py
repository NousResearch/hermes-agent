from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from plugins.questframe_fh6vr import register
from plugins.questframe_fh6vr import core
from plugins.questframe_fh6vr import cli as questframe_cli


def test_run_launcher_parses_json(monkeypatch, tmp_path):
    launcher = tmp_path / "FH6VR.Launcher.exe"
    launcher.write_text("", encoding="utf-8")

    def fake_run(argv, **kwargs):
        assert argv == [str(launcher), "preflight", "--json"]
        assert kwargs["timeout"] == 60
        return SimpleNamespace(
            returncode=0,
            stdout='{"Overall":"Pass","Product":"FH6VR Launcher"}',
            stderr="",
        )

    monkeypatch.setattr(core, "resolve_launcher_path", lambda explicit=None: launcher)
    monkeypatch.setattr(core.subprocess, "run", fake_run)

    result = core.run_launcher("preflight", extra_args=["--json"])

    assert result["ok"] is True
    assert result["json"]["Overall"] == "Pass"


def test_graphics_session_handler_runs_launcher_probe(monkeypatch):
    seen = {}

    def fake_run_launcher(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return {
            "ok": True,
            "json": {
                "Status": "Pass",
                "SessionCreated": True,
                "SwapchainFormats": [{"Name": "DXGI_FORMAT_R8G8B8A8_UNORM_SRGB"}],
            },
        }

    monkeypatch.setattr(core, "run_launcher", fake_run_launcher)

    raw = core.handle_graphics_session({"timeout_seconds": 30})
    result = json.loads(raw)

    assert result["ok"] is True
    assert result["json"]["SessionCreated"] is True
    assert seen["command"] == "graphics-session-selftest"
    assert seen["kwargs"]["extra_args"] == ["--json"]
    assert seen["kwargs"]["timeout_seconds"] == 30


def test_capture_preflight_handler_runs_new_launcher_gate(monkeypatch):
    seen = {}

    def fake_run_launcher(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return {"ok": True, "json": {"Status": "Pass"}}

    monkeypatch.setattr(core, "run_launcher", fake_run_launcher)

    raw = core.handle_fh6_capture_preflight({"timeout_seconds": 45})
    result = json.loads(raw)

    assert result["ok"] is True
    assert seen["command"] == "fh6-capture-preflight"
    assert seen["kwargs"]["extra_args"] == ["--json"]
    assert seen["kwargs"]["timeout_seconds"] == 45


def test_rtx3060_selftest_handler_runs_profile_contract(monkeypatch):
    seen = {}

    def fake_run_launcher(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return {"ok": True, "json": {"Name": "RTX 3060 DIBR profile contract"}}

    monkeypatch.setattr(core, "run_launcher", fake_run_launcher)

    raw = core.handle_rtx3060_selftest({"timeout_seconds": 30})
    result = json.loads(raw)

    assert result["ok"] is True
    assert seen["command"] == "rtx3060-selftest"
    assert seen["kwargs"]["extra_args"] == ["--json"]
    assert seen["kwargs"]["timeout_seconds"] == 30


def test_rtx3060_profiles_handler_requests_json_profiles(monkeypatch):
    seen = {}

    def fake_run_launcher(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return {"ok": True, "json": [{"Name": "RTX3060_DIBR_Safe"}]}

    monkeypatch.setattr(core, "run_launcher", fake_run_launcher)

    raw = core.handle_rtx3060_profiles({})
    result = json.loads(raw)

    assert result["ok"] is True
    assert seen["command"] == "profiles"
    assert seen["kwargs"]["extra_args"] == ["--json"]


def test_support_report_builds_redacted_report_command(monkeypatch, tmp_path):
    seen = {}

    def fake_run_launcher(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return {"ok": True, "json": {"Product": "FH6VR Launcher"}}

    monkeypatch.setattr(core, "run_launcher", fake_run_launcher)
    json_path = tmp_path / "support.json"
    html_path = tmp_path / "support.html"

    result = core.support_report(
        json_path=str(json_path),
        html_path=str(html_path),
        include_live_openxr=True,
        timeout_seconds=120,
    )

    assert result["ok"] is True
    assert seen["command"] == "support-report"
    assert seen["kwargs"]["extra_args"] == [
        "--json",
        "--write-json",
        str(json_path),
        "--write-html",
        str(html_path),
        "--include-live-openxr",
    ]
    assert seen["kwargs"]["timeout_seconds"] == 120
    assert result["report_paths"]["json"] == str(json_path)
    assert result["redacted_by_default"] is True


def test_plugin_registers_full_questframe_tool_surface():
    calls = {"tools": [], "commands": [], "cli": []}

    class Ctx:
        def register_tool(self, **kwargs):
            calls["tools"].append(kwargs["name"])

        def register_command(self, name, **kwargs):
            calls["commands"].append(name)

        def register_cli_command(self, **kwargs):
            calls["cli"].append(kwargs["name"])

    register(Ctx())

    assert {
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
        "questframe_support_report",
        "questframe_unity_scan",
    }.issubset(set(calls["tools"]))
    assert calls["commands"] == ["questframe"]
    assert calls["cli"] == ["questframe"]


def test_cli_support_report_dispatch(monkeypatch, tmp_path):
    seen = {}

    def fake_support_report(**kwargs):
        seen.update(kwargs)
        return {"ok": True, "report_paths": {"json": kwargs["json_path"]}}

    monkeypatch.setattr(core, "support_report", fake_support_report)
    parser = questframe_cli.argparse.ArgumentParser()
    questframe_cli.register_cli(parser)
    json_path = tmp_path / "support.json"
    args = parser.parse_args(
        [
            "support-report",
            "--json-path",
            str(json_path),
            "--include-live-openxr",
            "--timeout-seconds",
            "90",
        ]
    )

    exit_code = questframe_cli.questframe_command(args)

    assert exit_code == 0
    assert seen["json_path"] == str(json_path)
    assert seen["include_live_openxr"] is True
    assert seen["timeout_seconds"] == 90


def test_cli_rtx3060_selftest_dispatch(monkeypatch):
    seen = {}

    def fake_run_launcher(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return {"ok": True}

    monkeypatch.setattr(core, "run_launcher", fake_run_launcher)
    parser = questframe_cli.argparse.ArgumentParser()
    questframe_cli.register_cli(parser)
    args = parser.parse_args(["rtx3060-selftest", "--timeout-seconds", "75"])

    exit_code = questframe_cli.questframe_command(args)

    assert exit_code == 0
    assert seen["command"] == "rtx3060-selftest"
    assert seen["kwargs"]["extra_args"] == ["--json"]
    assert seen["kwargs"]["timeout_seconds"] == 75


def test_unity_scan_detects_vrchat_packages(tmp_path):
    project = tmp_path / "AvatarProject"
    (project / "Assets").mkdir(parents=True)
    (project / "Packages").mkdir()
    (project / "ProjectSettings").mkdir()
    (project / "ProjectSettings" / "ProjectVersion.txt").write_text(
        "m_EditorVersion: 2022.3.22f1\n", encoding="utf-8"
    )
    (project / "Packages" / "manifest.json").write_text(
        json.dumps(
            {
                "dependencies": {
                    "com.vrchat.avatars": "3.10.3",
                    "nadena.dev.modular-avatar": "1.13.0",
                    "jp.lilxyzw.liltoon": "1.8.7",
                }
            }
        ),
        encoding="utf-8",
    )

    result = core.scan_unity_projects(project_path=str(project))

    assert result["ok"] is True
    assert result["project_count"] == 1
    first = result["projects"][0]
    detected = {pkg["id"] for pkg in first["detected_packages"]}
    assert "com.vrchat.avatars" in detected
    assert "nadena.dev.modular-avatar" in detected
    assert first["unity_version"] == "2022.3.22f1"
    assert "VRChat SDK package not detected" not in first["risks"]
