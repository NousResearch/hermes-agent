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
