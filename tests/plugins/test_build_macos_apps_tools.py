"""Tool tests for the build-macos-apps plugin."""

import importlib.util
import json
import sys
import types
from pathlib import Path
from subprocess import CompletedProcess

import pytest


def _load_tools_module():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_dir = repo_root / "plugins" / "build-macos-apps"
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.build_macos_apps.tools",
        plugin_dir / "tools.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    pkg = sys.modules.get("hermes_plugins.build_macos_apps")
    if pkg is None:
        pkg = types.ModuleType("hermes_plugins.build_macos_apps")
        pkg.__path__ = [str(plugin_dir)]
        sys.modules["hermes_plugins.build_macos_apps"] = pkg

    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.build_macos_apps"
    sys.modules["hermes_plugins.build_macos_apps.tools"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def tools_mod():
    return _load_tools_module()


class TestCheckFn:
    def test_requires_macos_and_xcodebuild(self, tools_mod, monkeypatch):
        monkeypatch.setattr(tools_mod.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(tools_mod.shutil, "which", lambda name: "/usr/bin/xcodebuild")
        assert tools_mod.check_macos_dev_requirements() is True

        monkeypatch.setattr(tools_mod.platform, "system", lambda: "Linux")
        assert tools_mod.check_macos_dev_requirements() is False


class TestInspect:
    def test_detects_workspace_and_project(self, tools_mod, tmp_path):
        repo = tmp_path / "MyApp"
        (repo / "App.xcworkspace").mkdir(parents=True)
        (repo / "App.xcodeproj").mkdir()
        (repo / "Package.swift").write_text("// swift-tools-version: 5.9\n")

        result = json.loads(tools_mod.handle_macos_inspect_project({"path": str(repo)}))

        assert result["success"] is True
        assert result["recommended_container_type"] == "workspace"
        assert result["supports_xcodebuild_flow"] is True
        assert len(result["xcode_workspaces"]) == 1
        assert len(result["xcode_projects"]) == 1
        assert result["has_package_swift"] is True

    def test_errors_when_path_missing(self, tools_mod, tmp_path):
        result = json.loads(
            tools_mod.handle_macos_inspect_project({"path": str(tmp_path / "missing")})
        )
        assert result["success"] is False
        assert "Path not found" in result["error"]


class TestListSchemes:
    def test_lists_schemes_from_workspace(self, tools_mod, tmp_path, monkeypatch):
        repo = tmp_path / "WorkspaceRepo"
        workspace = repo / "App.xcworkspace"
        workspace.mkdir(parents=True)

        monkeypatch.setattr(
            tools_mod,
            "_run_xcodebuild",
            lambda command, cwd, timeout_seconds: CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    {
                        "project": {
                            "name": "App",
                            "schemes": ["App"],
                            "targets": ["App"],
                            "configurations": ["Debug", "Release"],
                        }
                    }
                ),
                stderr="",
            ),
        )

        result = json.loads(tools_mod.handle_macos_list_schemes({"path": str(repo)}))
        assert result["success"] is True
        assert result["container"]["type"] == "workspace"
        assert result["project"]["schemes"] == ["App"]

    def test_surfaces_xcodebuild_failure(self, tools_mod, tmp_path, monkeypatch):
        repo = tmp_path / "ProjectRepo"
        project = repo / "App.xcodeproj"
        project.mkdir(parents=True)

        monkeypatch.setattr(
            tools_mod,
            "_run_xcodebuild",
            lambda command, cwd, timeout_seconds: CompletedProcess(
                command,
                65,
                stdout="** BUILD FAILED **\n",
                stderr="xcodebuild: error: scheme App not found\n",
            ),
        )

        result = json.loads(tools_mod.handle_macos_list_schemes({"path": str(repo)}))
        assert result["success"] is False
        assert result["exit_code"] == 65
        assert result["output"]["highlights"]


class TestBuildProject:
    def test_build_includes_unsigned_flags(self, tools_mod, tmp_path, monkeypatch):
        repo = tmp_path / "BuildRepo"
        project = repo / "App.xcodeproj"
        project.mkdir(parents=True)
        captured = {}

        def fake_run(command, cwd, timeout_seconds):
            captured["command"] = command
            captured["cwd"] = cwd
            captured["timeout_seconds"] = timeout_seconds
            return CompletedProcess(command, 0, stdout="** BUILD SUCCEEDED **\n", stderr="")

        monkeypatch.setattr(tools_mod, "_run_xcodebuild", fake_run)

        result = json.loads(
            tools_mod.handle_macos_build_project(
                {"path": str(repo), "scheme": "App", "configuration": "Release"}
            )
        )

        assert result["success"] is True
        assert result["unsigned_build"] is True
        assert "CODE_SIGNING_ALLOWED=NO" in captured["command"]
        assert "CODE_SIGNING_REQUIRED=NO" in captured["command"]
        assert "CODE_SIGN_IDENTITY=" in captured["command"]
        assert result["configuration"] == "Release"

    def test_build_failure_returns_shaped_output(self, tools_mod, tmp_path, monkeypatch):
        repo = tmp_path / "BuildFailRepo"
        workspace = repo / "App.xcworkspace"
        workspace.mkdir(parents=True)

        monkeypatch.setattr(
            tools_mod,
            "_run_xcodebuild",
            lambda command, cwd, timeout_seconds: CompletedProcess(
                command,
                65,
                stdout="CompileSwift normal arm64 Foo.swift\n** BUILD FAILED **\n",
                stderr="Foo.swift:1:1: error: broken\n",
            ),
        )

        result = json.loads(
            tools_mod.handle_macos_build_project({"path": str(repo), "scheme": "App"})
        )

        assert result["success"] is False
        assert result["exit_code"] == 65
        assert result["output"]["highlights"]

    def test_build_requires_scheme(self, tools_mod, tmp_path):
        repo = tmp_path / "NoSchemeRepo"
        (repo / "App.xcodeproj").mkdir(parents=True)

        result = json.loads(tools_mod.handle_macos_build_project({"path": str(repo)}))
        assert result["success"] is False
        assert result["error"] == "scheme is required"


class TestTestProject:
    def test_test_includes_filters_and_result_bundle(self, tools_mod, tmp_path, monkeypatch):
        repo = tmp_path / "TestRepo"
        project = repo / "App.xcodeproj"
        project.mkdir(parents=True)
        captured = {}

        def fake_run(command, cwd, timeout_seconds):
            captured["command"] = command
            captured["cwd"] = cwd
            captured["timeout_seconds"] = timeout_seconds
            return CompletedProcess(command, 0, stdout="Test Suite 'All tests' passed\n", stderr="")

        monkeypatch.setattr(tools_mod, "_run_xcodebuild", fake_run)

        result = json.loads(
            tools_mod.handle_macos_test_project(
                {
                    "path": str(repo),
                    "scheme": "App",
                    "test_plan": "CI",
                    "only_testing": ["AppTests/FooTests"],
                    "skip_testing": ["AppUITests"],
                    "result_bundle_path": "artifacts/AppTests.xcresult",
                }
            )
        )

        assert result["success"] is True
        assert result["test_plan"] == "CI"
        assert result["only_testing"] == ["AppTests/FooTests"]
        assert result["skip_testing"] == ["AppUITests"]
        assert result["signing_disabled"] is True
        assert "-testPlan" in captured["command"]
        assert "-only-testing" in captured["command"]
        assert "-skip-testing" in captured["command"]
        assert "-resultBundlePath" in captured["command"]
        assert "CODE_SIGNING_ALLOWED=NO" in captured["command"]

    def test_test_failure_returns_shaped_output(self, tools_mod, tmp_path, monkeypatch):
        repo = tmp_path / "TestFailRepo"
        workspace = repo / "App.xcworkspace"
        workspace.mkdir(parents=True)

        monkeypatch.setattr(
            tools_mod,
            "_run_xcodebuild",
            lambda command, cwd, timeout_seconds: CompletedProcess(
                command,
                65,
                stdout="Test Case '-[AppTests FooTests testBar]' failed\n** TEST FAILED **\n",
                stderr="FooTests.swift:12: error: XCTAssertTrue failed\n",
            ),
        )

        result = json.loads(
            tools_mod.handle_macos_test_project({"path": str(repo), "scheme": "App"})
        )

        assert result["success"] is False
        assert result["exit_code"] == 65
        assert result["output"]["highlights"]

    def test_test_requires_scheme(self, tools_mod, tmp_path):
        repo = tmp_path / "NoTestSchemeRepo"
        (repo / "App.xcodeproj").mkdir(parents=True)

        result = json.loads(tools_mod.handle_macos_test_project({"path": str(repo)}))
        assert result["success"] is False
        assert result["error"] == "scheme is required"


class TestFindAppBundle:
    def test_prefers_build_products_match(self, tools_mod, tmp_path):
        repo = tmp_path / "RunRepo"
        primary = repo / "DerivedData" / "Build" / "Products" / "Debug" / "MyApp.app"
        fallback = repo / "dist" / "MyApp.app"
        primary.mkdir(parents=True)
        fallback.mkdir(parents=True)

        result = json.loads(
            tools_mod.handle_macos_find_app_bundle(
                {"path": str(repo), "app_name": "MyApp", "configuration": "Debug"}
            )
        )

        assert result["success"] is True
        assert result["recommended_app_bundle"] == str(primary)
        assert result["matches"][0]["path"] == str(primary)


class TestRunApp:
    def test_launches_discovered_bundle(self, tools_mod, tmp_path, monkeypatch):
        repo = tmp_path / "LaunchRepo"
        bundle = repo / "DerivedData" / "Build" / "Products" / "Debug" / "MyApp.app"
        bundle.mkdir(parents=True)

        launched = {}

        monkeypatch.setattr(
            tools_mod.subprocess,
            "run",
            lambda command, capture_output, text, check: (
                launched.setdefault("command", command),
                CompletedProcess(command, 0, stdout="", stderr=""),
            )[1],
        )
        monkeypatch.setattr(tools_mod, "_wait_for_app_state", lambda *args, **kwargs: [12345])

        result = json.loads(
            tools_mod.handle_macos_run_app(
                {
                    "path": str(repo),
                    "app_name": "MyApp",
                    "args": ["--demo"],
                    "new_instance": True,
                    "activate": False,
                }
            )
        )

        assert result["success"] is True
        assert result["is_running"] is True
        assert result["pids"] == [12345]
        assert launched["command"][:3] == ["open", "-n", "-g"]
        assert "--args" in launched["command"]


class TestStopApp:
    def test_stops_running_app_with_applescript(self, tools_mod, tmp_path, monkeypatch):
        repo = tmp_path / "StopRepo"
        bundle = repo / "DerivedData" / "Build" / "Products" / "Debug" / "MyApp.app"
        info_plist = bundle / "Contents" / "Info.plist"
        info_plist.parent.mkdir(parents=True)
        info_plist.write_bytes(
            tools_mod.plistlib.dumps({"CFBundleIdentifier": "com.example.MyApp"})
        )

        calls = []

        monkeypatch.setattr(tools_mod, "_pgrep_app", lambda app_name: [123] if not calls else [])

        def fake_run(command, capture_output, text, check):
            calls.append(command)
            return CompletedProcess(command, 0, stdout="", stderr="")

        monkeypatch.setattr(tools_mod.subprocess, "run", fake_run)
        monkeypatch.setattr(tools_mod, "_wait_for_app_state", lambda *args, **kwargs: [])

        result = json.loads(
            tools_mod.handle_macos_stop_app({"path": str(repo), "app_name": "MyApp"})
        )

        assert result["success"] is True
        assert result["stopped"] is True
        assert calls[0][:2] == ["osascript", "-e"]

    def test_returns_success_when_app_not_running(self, tools_mod, tmp_path, monkeypatch):
        repo = tmp_path / "StopIdleRepo"
        bundle = repo / "DerivedData" / "Build" / "Products" / "Debug" / "MyApp.app"
        bundle.mkdir(parents=True)
        monkeypatch.setattr(tools_mod, "_pgrep_app", lambda app_name: [])

        result = json.loads(
            tools_mod.handle_macos_stop_app({"path": str(repo), "app_name": "MyApp"})
        )

        assert result["success"] is True
        assert result["was_running"] is False


class TestReadRecentLogs:
    def test_reads_recent_logs_with_inferred_predicate(self, tools_mod, tmp_path, monkeypatch):
        repo = tmp_path / "LogsRepo"
        bundle = repo / "DerivedData" / "Build" / "Products" / "Debug" / "MyApp.app"
        info_plist = bundle / "Contents" / "Info.plist"
        info_plist.parent.mkdir(parents=True)
        info_plist.write_bytes(
            tools_mod.plistlib.dumps({"CFBundleIdentifier": "com.example.MyApp"})
        )
        captured = {}

        monkeypatch.setattr(
            tools_mod,
            "_run_system_command",
            lambda command, cwd=None, timeout_seconds=300: (
                captured.setdefault("command", command),
                CompletedProcess(command, 0, stdout="l1\nl2\nl3\n", stderr=""),
            )[1],
        )

        result = json.loads(
            tools_mod.handle_macos_read_recent_logs({"path": str(repo), "app_name": "MyApp", "limit": 2})
        )

        assert result["success"] is True
        assert result["line_count"] == 3
        assert result["truncated"] is True
        assert result["lines"] == ["l2", "l3"]
        assert captured["command"][:2] == ["log", "show"]


class TestCollectCrashReports:
    def test_collects_recent_crash_reports(self, tools_mod, tmp_path, monkeypatch):
        repo = tmp_path / "CrashRepo"
        bundle = repo / "DerivedData" / "Build" / "Products" / "Debug" / "MyApp.app"
        bundle.mkdir(parents=True)
        reports_dir = tmp_path / "Library" / "Logs" / "DiagnosticReports"
        reports_dir.mkdir(parents=True)
        report = reports_dir / "MyApp_2026-01-01-000000.crash"
        report.write_text("Header\nThread 0 Crashed:\n")
        monkeypatch.setattr(tools_mod.Path, "home", staticmethod(lambda: tmp_path))

        result = json.loads(
            tools_mod.handle_macos_collect_crash_reports({"path": str(repo), "app_name": "MyApp"})
        )

        assert result["success"] is True
        assert result["report_count"] == 1
        assert result["reports"][0]["filename"] == report.name


class TestShowBuildSettings:
    def test_parses_build_settings_output(self, tools_mod, tmp_path, monkeypatch):
        repo = tmp_path / "SettingsRepo"
        project = repo / "App.xcodeproj"
        project.mkdir(parents=True)

        monkeypatch.setattr(
            tools_mod,
            "_run_xcodebuild",
            lambda command, cwd, timeout_seconds: CompletedProcess(
                command,
                0,
                stdout=(
                    "Build settings for action build and target App:\n"
                    "    TARGET_NAME = App\n"
                    "    PRODUCT_NAME = App\n"
                    "    BUILT_PRODUCTS_DIR = /tmp/DerivedData/Build/Products/Debug\n"
                    "    PRODUCT_BUNDLE_IDENTIFIER = com.example.app\n"
                ),
                stderr="",
            ),
        )

        result = json.loads(
            tools_mod.handle_macos_show_build_settings({"path": str(repo), "scheme": "App"})
        )

        assert result["success"] is True
        assert result["section_count"] == 1
        assert result["interesting"]["TARGET_NAME"] == "App"
        assert "BUILT_PRODUCTS_DIR" in result["interesting"]

    def test_build_settings_requires_scheme(self, tools_mod, tmp_path):
        repo = tmp_path / "NoSettingsSchemeRepo"
        (repo / "App.xcodeproj").mkdir(parents=True)

        result = json.loads(tools_mod.handle_macos_show_build_settings({"path": str(repo)}))
        assert result["success"] is False
        assert result["error"] == "scheme is required"
