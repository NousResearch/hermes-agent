"""Tests for Phase 2 Hermes Code Mode: CodeWorkspaceService."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch


# =============================================================================
# Helpers
# =============================================================================


def _make_node_project(root: Path, scripts: dict = None, deps: dict = None):
    pkg = {"name": "test-app", "version": "1.0.0"}
    if scripts:
        pkg["scripts"] = scripts
    if deps:
        pkg["dependencies"] = deps
    (root / "package.json").write_text(json.dumps(pkg))


# =============================================================================
# Stack detection
# =============================================================================


class TestDetectStack:
    def test_empty_dir_returns_empty(self, tmp_path):
        from hermes_cli.code.workspace_service import detect_stack

        assert detect_stack(str(tmp_path)) == []

    def test_node_detected_from_package_json(self, tmp_path):
        from hermes_cli.code.workspace_service import detect_stack

        _make_node_project(tmp_path)
        stack = detect_stack(str(tmp_path))
        assert "node" in stack

    def test_typescript_detected_from_tsconfig(self, tmp_path):
        from hermes_cli.code.workspace_service import detect_stack

        _make_node_project(tmp_path)
        (tmp_path / "tsconfig.json").write_text("{}")
        stack = detect_stack(str(tmp_path))
        assert "typescript" in stack

    def test_vite_detected_from_config_file(self, tmp_path):
        from hermes_cli.code.workspace_service import detect_stack

        _make_node_project(tmp_path)
        (tmp_path / "vite.config.ts").write_text("export default {}")
        stack = detect_stack(str(tmp_path))
        assert "vite" in stack

    def test_react_detected_from_dependency(self, tmp_path):
        from hermes_cli.code.workspace_service import detect_stack

        _make_node_project(tmp_path, deps={"react": "^18.0.0", "react-dom": "^18.0.0"})
        stack = detect_stack(str(tmp_path))
        assert "react" in stack

    def test_next_detected_from_config_file(self, tmp_path):
        from hermes_cli.code.workspace_service import detect_stack

        _make_node_project(tmp_path)
        (tmp_path / "next.config.js").write_text("module.exports = {}")
        stack = detect_stack(str(tmp_path))
        assert "next" in stack

    def test_tailwind_detected_from_config_file(self, tmp_path):
        from hermes_cli.code.workspace_service import detect_stack

        _make_node_project(tmp_path)
        (tmp_path / "tailwind.config.js").write_text("module.exports = {}")
        stack = detect_stack(str(tmp_path))
        assert "tailwind" in stack

    def test_go_detected_from_go_mod(self, tmp_path):
        from hermes_cli.code.workspace_service import detect_stack

        (tmp_path / "go.mod").write_text("module example.com/app\n\ngo 1.21\n")
        stack = detect_stack(str(tmp_path))
        assert "go" in stack

    def test_python_detected_from_pyproject_toml(self, tmp_path):
        from hermes_cli.code.workspace_service import detect_stack

        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'app'\n")
        stack = detect_stack(str(tmp_path))
        assert "python" in stack

    def test_python_fastapi_detected_from_requirements(self, tmp_path):
        from hermes_cli.code.workspace_service import detect_stack

        (tmp_path / "requirements.txt").write_text("fastapi==0.110.0\nuvicorn\n")
        stack = detect_stack(str(tmp_path))
        assert "python" in stack
        assert "fastapi" in stack

    def test_docker_detected_from_dockerfile(self, tmp_path):
        from hermes_cli.code.workspace_service import detect_stack

        (tmp_path / "Dockerfile").write_text("FROM python:3.11\n")
        stack = detect_stack(str(tmp_path))
        assert "docker" in stack

    def test_compose_detected_from_compose_file(self, tmp_path):
        from hermes_cli.code.workspace_service import detect_stack

        (tmp_path / "docker-compose.yml").write_text(
            "version: '3'\nservices:\n  app:\n"
        )
        stack = detect_stack(str(tmp_path))
        assert "compose" in stack

    def test_full_vite_react_ts_project(self, tmp_path):
        from hermes_cli.code.workspace_service import detect_stack

        _make_node_project(tmp_path, deps={"react": "^18.0.0"})
        (tmp_path / "tsconfig.json").write_text("{}")
        (tmp_path / "vite.config.ts").write_text("export default {}")
        stack = detect_stack(str(tmp_path))
        assert "node" in stack
        assert "typescript" in stack
        assert "react" in stack
        assert "vite" in stack


# =============================================================================
# Package manager detection
# =============================================================================


class TestDetectPackageManager:
    def test_pnpm_from_lockfile(self, tmp_path):
        from hermes_cli.code.workspace_service import detect_package_manager

        (tmp_path / "pnpm-lock.yaml").touch()
        assert detect_package_manager(str(tmp_path)) == "pnpm"

    def test_yarn_from_lockfile(self, tmp_path):
        from hermes_cli.code.workspace_service import detect_package_manager

        (tmp_path / "yarn.lock").touch()
        assert detect_package_manager(str(tmp_path)) == "yarn"

    def test_bun_from_lockfile(self, tmp_path):
        from hermes_cli.code.workspace_service import detect_package_manager

        (tmp_path / "bun.lock").touch()
        assert detect_package_manager(str(tmp_path)) == "bun"

    def test_npm_from_package_lock(self, tmp_path):
        from hermes_cli.code.workspace_service import detect_package_manager

        (tmp_path / "package-lock.json").write_text("{}")
        assert detect_package_manager(str(tmp_path)) == "npm"

    def test_go_from_go_mod(self, tmp_path):
        from hermes_cli.code.workspace_service import detect_package_manager

        (tmp_path / "go.mod").write_text("module example.com\n")
        assert detect_package_manager(str(tmp_path)) == "go"

    def test_none_for_empty_dir(self, tmp_path):
        from hermes_cli.code.workspace_service import detect_package_manager

        assert detect_package_manager(str(tmp_path)) is None

    def test_pnpm_takes_priority_over_npm(self, tmp_path):
        from hermes_cli.code.workspace_service import detect_package_manager

        (tmp_path / "pnpm-lock.yaml").touch()
        (tmp_path / "package-lock.json").write_text("{}")
        assert detect_package_manager(str(tmp_path)) == "pnpm"


# =============================================================================
# Command detection
# =============================================================================


class TestDetectCommands:
    def test_package_json_scripts_extracted(self, tmp_path):
        from hermes_cli.code.workspace_service import detect_commands

        _make_node_project(
            tmp_path,
            scripts={
                "build": "tsc",
                "test": "jest",
                "dev": "vite",
                "lint": "eslint .",
            },
        )
        (tmp_path / "package-lock.json").write_text("{}")
        cmds = detect_commands(str(tmp_path), "npm")
        names = [c["name"] for c in cmds]
        assert "build" in names
        assert "test" in names
        assert "dev" in names
        assert "lint" in names

    def test_command_kinds_classified_correctly(self, tmp_path):
        from hermes_cli.code.workspace_service import detect_commands

        _make_node_project(
            tmp_path, scripts={"build": "tsc", "test": "jest", "dev": "vite"}
        )
        cmds = {c["name"]: c for c in detect_commands(str(tmp_path), "npm")}
        assert cmds["build"]["kind"] == "build"
        assert cmds["test"]["kind"] == "test"
        assert cmds["dev"]["kind"] == "dev"

    def test_pnpm_prefix_used(self, tmp_path):
        from hermes_cli.code.workspace_service import detect_commands

        _make_node_project(tmp_path, scripts={"build": "tsc"})
        cmds = {c["name"]: c for c in detect_commands(str(tmp_path), "pnpm")}
        assert cmds["build"]["command"] == "pnpm run build"

    def test_go_commands_suggested_when_go_mod_exists(self, tmp_path):
        from hermes_cli.code.workspace_service import detect_commands

        (tmp_path / "go.mod").write_text("module example.com\n")
        cmds = {c["name"]: c for c in detect_commands(str(tmp_path))}
        assert "test" in cmds
        assert cmds["test"]["command"] == "go test ./..."
        assert cmds["test"]["source"] == "go"

    def test_makefile_targets_extracted(self, tmp_path):
        from hermes_cli.code.workspace_service import detect_commands

        (tmp_path / "Makefile").write_text(
            "build:\n\tgo build\n\ntest:\n\tgo test ./...\n"
        )
        cmds = {c["name"]: c for c in detect_commands(str(tmp_path))}
        assert "build" in cmds
        assert cmds["build"]["source"] == "makefile"
        assert cmds["build"]["command"] == "make build"


# =============================================================================
# Git detection
# =============================================================================


class TestDetectGitInfo:
    def test_non_git_dir_returns_false(self, tmp_path):
        from hermes_cli.code.workspace_service import detect_git_info

        result = detect_git_info(str(tmp_path))
        assert result["is_git_repo"] is False
        assert result["branch"] is None

    def test_git_dir_detected(self, tmp_path):
        from hermes_cli.code.workspace_service import detect_git_info

        (tmp_path / ".git").mkdir()
        # Mock subprocess so we don't need real git
        with patch("subprocess.run") as mock_run:
            mock_branch = mock_run.return_value
            mock_branch.returncode = 0
            mock_branch.stdout = "main\n"
            result = detect_git_info(str(tmp_path))
        assert result["is_git_repo"] is True

    def test_git_failure_returns_safe_defaults(self, tmp_path):
        from hermes_cli.code.workspace_service import detect_git_info

        (tmp_path / ".git").mkdir()
        with patch("subprocess.run", side_effect=Exception("git not found")):
            result = detect_git_info(str(tmp_path))
        assert result["is_git_repo"] is True  # .git dir exists
        assert result["branch"] is None


# =============================================================================
# CodeWorkspaceService
# =============================================================================


class TestCodeWorkspaceService:
    def test_inspect_path_invalid_raises_value_error(self):
        from hermes_cli.code.workspace_service import CodeWorkspaceService

        svc = CodeWorkspaceService()
        with pytest.raises(ValueError, match="does not exist"):
            svc.inspect_path("/nonexistent/path/that/cannot/exist/at/all")

    def test_inspect_path_file_raises_value_error(self, tmp_path):
        from hermes_cli.code.workspace_service import CodeWorkspaceService

        f = tmp_path / "file.txt"
        f.write_text("hi")
        svc = CodeWorkspaceService()
        with pytest.raises(ValueError, match="not a directory"):
            svc.inspect_path(str(f))

    def test_inspect_path_returns_metadata(self, tmp_path):
        from hermes_cli.code.workspace_service import CodeWorkspaceService

        _make_node_project(tmp_path, scripts={"build": "tsc"})
        (tmp_path / "tsconfig.json").write_text("{}")
        svc = CodeWorkspaceService()
        meta = svc.inspect_path(str(tmp_path))
        assert meta["name"] == tmp_path.name
        assert meta["path"] == str(tmp_path)
        assert "node" in meta["detected_stack"]
        assert "typescript" in meta["detected_stack"]

    def test_open_workspace_persists(self, tmp_path):
        from hermes_cli.code.workspace_service import CodeWorkspaceService
        from hermes_state import WorkspaceDB

        db_path = tmp_path / "ws_test.db"
        project = tmp_path / "myproject"
        project.mkdir()
        _make_node_project(project)

        svc = CodeWorkspaceService()
        with patch(
            "hermes_cli.code.workspace_service.CodeWorkspaceService.open_workspace"
        ) as mock_open:
            # Use real implementation with patched DB path
            pass

        # Use real DB with tmp_path
        db = WorkspaceDB(db_path=db_path)
        try:
            ws = db.upsert_workspace(
                path=str(project),
                name=project.name,
                is_git_repo=False,
                branch=None,
                repo_url=None,
                detected_stack=["node"],
                package_manager="npm",
                commands=[],
            )
            assert ws["id"]
            assert ws["name"] == project.name
            assert ws["detected_stack"] == ["node"]

            all_ws = db.list_workspaces()
            assert len(all_ws) == 1
            assert all_ws[0]["path"] == str(project)
        finally:
            db.close()

    def test_open_workspace_no_duplicate_on_same_path(self, tmp_path):
        from hermes_state import WorkspaceDB

        db_path = tmp_path / "ws_test.db"
        project = tmp_path / "proj"
        project.mkdir()

        db = WorkspaceDB(db_path=db_path)
        try:
            w1 = db.upsert_workspace(
                path=str(project), name="proj", detected_stack=["python"]
            )
            w2 = db.upsert_workspace(
                path=str(project), name="proj", detected_stack=["python", "fastapi"]
            )
            assert w1["id"] == w2["id"]
            assert len(db.list_workspaces()) == 1
            # Stack updated on second call
            assert "fastapi" in db.get_workspace(w1["id"])["detected_stack"]
        finally:
            db.close()

    def test_get_workspace_returns_none_for_unknown(self, tmp_path):
        from hermes_state import WorkspaceDB

        db = WorkspaceDB(db_path=tmp_path / "ws.db")
        try:
            assert db.get_workspace("nonexistent_id") is None
        finally:
            db.close()

    def test_list_workspaces_empty_initially(self, tmp_path):
        from hermes_state import WorkspaceDB

        db = WorkspaceDB(db_path=tmp_path / "ws.db")
        try:
            assert db.list_workspaces() == []
        finally:
            db.close()


# =============================================================================
# Schema
# =============================================================================


class TestWorkspaceSchema:
    def test_code_workspaces_table_exists(self, tmp_path):
        from hermes_state import WorkspaceDB

        db = WorkspaceDB(db_path=tmp_path / "ws.db")
        try:
            cursor = db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='code_workspaces'"
            )
            assert cursor.fetchone() is not None
        finally:
            db.close()

    def test_path_index_exists(self, tmp_path):
        from hermes_state import WorkspaceDB

        db = WorkspaceDB(db_path=tmp_path / "ws.db")
        try:
            cursor = db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_code_workspaces_path'"
            )
            assert cursor.fetchone() is not None
        finally:
            db.close()

    def test_schema_version_is_18(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            cursor = db._conn.execute("SELECT version FROM schema_version")
            assert cursor.fetchone()[0] == 18
        finally:
            db.close()

    def test_is_git_repo_deserialized_as_bool(self, tmp_path):
        from hermes_state import WorkspaceDB

        project = tmp_path / "proj"
        project.mkdir()
        db = WorkspaceDB(db_path=tmp_path / "ws.db")
        try:
            ws = db.upsert_workspace(path=str(project), name="proj", is_git_repo=True)
            assert ws["is_git_repo"] is True
            ws2 = db.upsert_workspace(
                path=str(tmp_path / "proj2"), name="proj2", is_git_repo=False
            )
            assert ws2["is_git_repo"] is False
        finally:
            db.close()


# =============================================================================
# REST endpoint tests
# =============================================================================


class TestCodeWorkspaceEndpoints:
    @pytest.fixture(autouse=True)
    def _setup(self):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("starlette not installed")

        from hermes_cli.web_server import app, _SESSION_TOKEN

        self.client = TestClient(app)
        self.client.headers["Authorization"] = f"Bearer {_SESSION_TOKEN}"

    def test_list_workspaces_returns_empty(self, monkeypatch):
        import hermes_cli.code.workspace_service as ws_mod

        monkeypatch.setattr(
            ws_mod.CodeWorkspaceService, "list_workspaces", lambda self: []
        )
        resp = self.client.get("/api/code/workspaces")
        assert resp.status_code == 200
        data = resp.json()
        assert data["workspaces"] == []
        assert data["total"] == 0

    def test_list_workspaces_requires_auth(self):
        import hermes_cli.web_server as web_server
        from starlette.testclient import TestClient

        client = TestClient(web_server.app)
        resp = client.get("/api/code/workspaces")
        assert resp.status_code == 401

    def test_open_workspace_invalid_path_returns_400(self, monkeypatch):
        import hermes_cli.code.workspace_service as ws_mod

        def _raise(self, path):
            raise ValueError(f"Path does not exist: {path}")

        monkeypatch.setattr(ws_mod.CodeWorkspaceService, "open_workspace", _raise)
        resp = self.client.post(
            "/api/code/workspaces/open", json={"path": "/nonexistent"}
        )
        assert resp.status_code == 400

    def test_open_workspace_returns_workspace(self, monkeypatch, tmp_path):
        import hermes_cli.code.workspace_service as ws_mod

        fake_ws = {
            "id": "abc123",
            "name": "myproject",
            "path": str(tmp_path),
            "is_git_repo": False,
            "branch": None,
            "repo_url": None,
            "detected_stack": ["python"],
            "package_manager": None,
            "commands": [],
            "created_at": "2026-04-24T00:00:00+00:00",
            "updated_at": "2026-04-24T00:00:00+00:00",
        }
        monkeypatch.setattr(
            ws_mod.CodeWorkspaceService, "open_workspace", lambda self, path: fake_ws
        )
        resp = self.client.post(
            "/api/code/workspaces/open", json={"path": str(tmp_path)}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["workspace"]["id"] == "abc123"
        assert data["workspace"]["detected_stack"] == ["python"]

    def test_get_workspace_not_found_returns_404(self, monkeypatch):
        import hermes_cli.code.workspace_service as ws_mod

        monkeypatch.setattr(
            ws_mod.CodeWorkspaceService, "get_workspace", lambda self, wid: None
        )
        resp = self.client.get("/api/code/workspaces/doesnotexist")
        assert resp.status_code == 404

    def test_get_workspace_returns_workspace(self, monkeypatch):
        import hermes_cli.code.workspace_service as ws_mod

        fake_ws = {
            "id": "ws1",
            "name": "proj",
            "path": "/some/path",
            "detected_stack": ["go"],
            "is_git_repo": True,
            "branch": "main",
            "repo_url": None,
            "package_manager": "go",
            "commands": [],
            "created_at": "2026-04-24T00:00:00+00:00",
            "updated_at": "2026-04-24T00:00:00+00:00",
        }
        monkeypatch.setattr(
            ws_mod.CodeWorkspaceService, "get_workspace", lambda self, wid: fake_ws
        )
        resp = self.client.get("/api/code/workspaces/ws1")
        assert resp.status_code == 200
        assert resp.json()["workspace"]["branch"] == "main"

    def test_refresh_workspace_not_found_returns_404(self, monkeypatch):
        import hermes_cli.code.workspace_service as ws_mod

        def _raise(self, wid):
            raise ValueError(f"Workspace not found: {wid}")

        monkeypatch.setattr(ws_mod.CodeWorkspaceService, "refresh_workspace", _raise)
        resp = self.client.post("/api/code/workspaces/bad_id/refresh")
        assert resp.status_code == 404

    def test_refresh_workspace_returns_updated(self, monkeypatch):
        import hermes_cli.code.workspace_service as ws_mod

        fake_ws = {
            "id": "ws1",
            "name": "proj",
            "path": "/some/path",
            "detected_stack": ["python", "fastapi"],
            "is_git_repo": True,
            "branch": "feature/x",
            "repo_url": None,
            "package_manager": None,
            "commands": [],
            "created_at": "2026-04-24T00:00:00+00:00",
            "updated_at": "2026-04-24T12:00:00+00:00",
        }
        monkeypatch.setattr(
            ws_mod.CodeWorkspaceService, "refresh_workspace", lambda self, wid: fake_ws
        )
        resp = self.client.post("/api/code/workspaces/ws1/refresh")
        assert resp.status_code == 200
        data = resp.json()
        assert "fastapi" in data["workspace"]["detected_stack"]
        assert data["workspace"]["branch"] == "feature/x"
