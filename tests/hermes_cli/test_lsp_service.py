"""Tests for Phase 7 Hermes Code Mode: CodeIntelligenceService / LSP."""

import json
import os
import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock


# =============================================================================
# Helpers
# =============================================================================


def _make_workspace(db, tmp_path, name="myproject", stack=None):
    project = tmp_path / name
    project.mkdir(exist_ok=True)
    return db.upsert_workspace(
        path=str(project),
        name=name,
        is_git_repo=True,
        branch="main",
        detected_stack=stack if stack is not None else ["typescript", "node"],
    )


def _make_session(db, wdb, tmp_path, stack=None):
    ws = _make_workspace(wdb, tmp_path, stack=stack)
    return db.create_session(workspace_id=ws["id"])


# =============================================================================
# Parsers — TypeScript
# =============================================================================


class TestParseTypescript:
    def test_standard_error(self):
        from hermes_cli.code.lsp_service import parse_typescript_output

        output = "src/App.tsx(10,5): error TS2322: Type 'string' is not assignable to type 'number'."
        diags = parse_typescript_output(output)
        assert len(diags) == 1
        d = diags[0]
        assert d["file"] == "src/App.tsx"
        assert d["line"] == 10
        assert d["column"] == 5
        assert d["severity"] == "error"
        assert d["source"] == "typescript"
        assert d["code"] == "TS2322"
        assert "string" in d["message"]

    def test_multiple_errors(self):
        from hermes_cli.code.lsp_service import parse_typescript_output

        output = (
            "src/a.ts(1,1): error TS1005: ';' expected.\n"
            "src/b.ts(20,3): error TS2304: Cannot find name 'foo'.\n"
        )
        diags = parse_typescript_output(output)
        assert len(diags) == 2
        assert diags[0]["file"] == "src/a.ts"
        assert diags[1]["file"] == "src/b.ts"
        assert diags[1]["code"] == "TS2304"

    def test_warning_severity(self):
        from hermes_cli.code.lsp_service import parse_typescript_output

        output = "src/c.ts(5,1): warning TS6133: 'x' is declared but its value is never read."
        diags = parse_typescript_output(output)
        assert len(diags) == 1
        assert diags[0]["severity"] == "warning"

    def test_empty_output(self):
        from hermes_cli.code.lsp_service import parse_typescript_output

        assert parse_typescript_output("") == []
        assert parse_typescript_output("No errors found") == []

    def test_mixed_noise_and_errors(self):
        from hermes_cli.code.lsp_service import parse_typescript_output

        output = (
            "Found 2 errors in 1 file.\n"
            "\n"
            "src/main.ts(3,7): error TS2345: Argument of type 'string' is not assignable.\n"
        )
        diags = parse_typescript_output(output)
        assert len(diags) == 1
        assert diags[0]["file"] == "src/main.ts"


# =============================================================================
# Parsers — Go
# =============================================================================


class TestParseGo:
    def test_vet_error(self):
        from hermes_cli.code.lsp_service import parse_go_output

        output = "pkg/handler.go:12:8: undefined: Foo"
        diags = parse_go_output(output)
        assert len(diags) == 1
        d = diags[0]
        assert d["file"] == "pkg/handler.go"
        assert d["line"] == 12
        assert d["column"] == 8
        assert d["severity"] == "error"
        assert d["source"] == "go"
        assert "undefined" in d["message"]

    def test_vet_warning(self):
        from hermes_cli.code.lsp_service import parse_go_output

        output = "main.go:42:2: should use return value from os.Open"
        diags = parse_go_output(output)
        assert len(diags) == 1
        assert diags[0]["severity"] == "warning"

    def test_skips_fail_headers(self):
        from hermes_cli.code.lsp_service import parse_go_output

        output = (
            "# github.com/example/pkg\n"
            "FAIL  github.com/example/pkg  0.001s\n"
            "--- FAIL: TestFoo (0.00s)\n"
        )
        diags = parse_go_output(output)
        assert len(diags) == 0

    def test_skips_ok_lines(self):
        from hermes_cli.code.lsp_service import parse_go_output

        output = "ok  	github.com/example/pkg	0.123s"
        diags = parse_go_output(output)
        assert len(diags) == 0

    def test_empty_output(self):
        from hermes_cli.code.lsp_service import parse_go_output

        assert parse_go_output("") == []

    def test_multiple_errors(self):
        from hermes_cli.code.lsp_service import parse_go_output

        output = (
            "cmd/main.go:5:1: expected declaration, found 'return'\n"
            "pkg/util.go:20:3: undefined: Bar\n"
        )
        diags = parse_go_output(output)
        assert len(diags) == 2


# =============================================================================
# Parsers — ESLint
# =============================================================================


class TestParseEslint:
    def test_compact_format(self):
        from hermes_cli.code.lsp_service import parse_eslint_output

        output = "/home/user/proj/src/index.ts: line 10, col 5, Error - Unexpected var (no-var)"
        diags = parse_eslint_output(output)
        assert len(diags) == 1
        d = diags[0]
        assert d["file"] == "/home/user/proj/src/index.ts"
        assert d["line"] == 10
        assert d["column"] == 5
        assert d["severity"] == "error"
        assert d["source"] == "eslint"
        assert d["code"] == "no-var"

    def test_compact_warning(self):
        from hermes_cli.code.lsp_service import parse_eslint_output

        output = "src/app.js: line 3, col 1, Warning - Unused variable (no-unused-vars)"
        diags = parse_eslint_output(output)
        assert len(diags) == 1
        assert diags[0]["severity"] == "warning"

    def test_raw_fallback(self):
        from hermes_cli.code.lsp_service import parse_eslint_output

        output = "Some completely unparseable ESLint output here"
        diags = parse_eslint_output(output)
        assert len(diags) == 1
        assert diags[0]["source"] == "eslint"
        assert diags[0]["raw"] is not None
        assert "unparseable" in diags[0]["message"]

    def test_empty_output(self):
        from hermes_cli.code.lsp_service import parse_eslint_output

        assert parse_eslint_output("") == []
        assert parse_eslint_output("   ") == []


# =============================================================================
# Language detection
# =============================================================================


class TestLanguageDetection:
    @pytest.fixture()
    def tmp_db(self, tmp_path):
        return tmp_path / "state.db"

    @pytest.fixture()
    def wdb(self, tmp_db):
        from hermes_state import WorkspaceDB

        d = WorkspaceDB(db_path=tmp_db)
        yield d
        d.close()

    @pytest.fixture()
    def svc(self, tmp_db):
        from hermes_cli.code.lsp_service import CodeIntelligenceService

        return CodeIntelligenceService(db_path=tmp_db)

    def test_typescript_detected(self, svc, wdb, tmp_path):
        ws = _make_workspace(wdb, tmp_path, stack=["typescript", "node"])
        langs = svc.get_supported_languages(ws["id"])
        assert "typescript" in langs

    def test_eslint_detected_with_lint_script(self, svc, wdb, tmp_path):
        ws = _make_workspace(wdb, tmp_path, stack=["typescript", "node"])
        project = Path(ws["path"])
        (project / "package.json").write_text(
            json.dumps({"scripts": {"lint": "eslint ."}})
        )
        langs = svc.get_supported_languages(ws["id"])
        assert "eslint" in langs

    def test_go_detected(self, svc, wdb, tmp_path):
        ws = _make_workspace(wdb, tmp_path, stack=["go"])
        langs = svc.get_supported_languages(ws["id"])
        assert "go" in langs

    def test_empty_stack(self, svc, wdb, tmp_path):
        ws = _make_workspace(wdb, tmp_path, stack=[])
        langs = svc.get_supported_languages(ws["id"])
        assert langs == []

    def test_unsupported_workspace(self, svc):
        with pytest.raises(ValueError, match="not found"):
            svc.get_supported_languages("nonexistent-id")


# =============================================================================
# Workspace diagnostics
# =============================================================================


class TestWorkspaceDiagnostics:
    @pytest.fixture()
    def tmp_db(self, tmp_path):
        return tmp_path / "state.db"

    @pytest.fixture()
    def wdb(self, tmp_db):
        from hermes_state import WorkspaceDB

        d = WorkspaceDB(db_path=tmp_db)
        yield d
        d.close()

    @pytest.fixture()
    def csdb(self, tmp_db):
        from hermes_state import CodeSessionDB

        d = CodeSessionDB(db_path=tmp_db)
        yield d
        d.close()

    @pytest.fixture()
    def svc(self, tmp_db):
        from hermes_cli.code.lsp_service import CodeIntelligenceService

        return CodeIntelligenceService(db_path=tmp_db)

    def _mock_run(self, rc=0, stdout="", stderr=""):
        """Return a mock _run_diagnostic_command function."""

        def fake_run(command, cwd, timeout=120):
            return rc, stdout, stderr

        return fake_run

    def test_unsupported_stack(self, svc, wdb, tmp_path):
        ws = _make_workspace(wdb, tmp_path, stack=[])
        result = svc.get_workspace_diagnostics(ws["id"])
        assert result["status"] == "unsupported"
        assert result["diagnostics"] == []
        assert result["summary"]["total"] == 0

    def test_no_workspace(self, svc):
        with pytest.raises(ValueError, match="not found"):
            svc.get_workspace_diagnostics("nonexistent-id")

    def test_typescript_with_errors(self, svc, wdb, tmp_path):
        ws = _make_workspace(wdb, tmp_path, stack=["typescript", "node"])
        project = Path(ws["path"])
        (project / "tsconfig.json").write_text("{}")
        (project / "package.json").write_text(
            json.dumps({"scripts": {"typecheck": "tsc --noEmit"}})
        )

        tsc_output = (
            "src/App.tsx(10,5): error TS2322: Type 'string' is not assignable to type 'number'.\n"
            "src/utils.ts(20,1): error TS1005: ';' expected.\n"
        )

        with patch.object(
            svc, "_run_diagnostic_command", self._mock_run(stdout=tsc_output)
        ):
            result = svc.get_workspace_diagnostics(ws["id"])

        assert result["status"] == "partial"
        assert len(result["diagnostics"]) == 2
        assert result["summary"]["errors"] == 2
        assert result["summary"]["warnings"] == 0
        assert result["summary"]["total"] == 2
        assert any("typecheck" in cmd for cmd in result["commands_run"])

    def test_typescript_clean(self, svc, wdb, tmp_path):
        ws = _make_workspace(wdb, tmp_path, stack=["typescript", "node"])
        project = Path(ws["path"])
        (project / "tsconfig.json").write_text("{}")
        (project / "package.json").write_text(
            json.dumps({"scripts": {"typecheck": "tsc --noEmit"}})
        )

        with patch.object(svc, "_run_diagnostic_command", self._mock_run(stdout="")):
            result = svc.get_workspace_diagnostics(ws["id"])

        assert result["status"] == "ok"
        assert result["summary"]["errors"] == 0

    def test_go_with_vet_errors(self, svc, wdb, tmp_path):
        ws = _make_workspace(wdb, tmp_path, stack=["go"])
        project = Path(ws["path"])
        (project / "go.mod").write_text("module example.com/test")

        vet_output = "pkg/handler.go:12:8: undefined: Foo\n"

        with patch.object(
            svc, "_run_diagnostic_command", self._mock_run(stderr=vet_output)
        ):
            result = svc.get_workspace_diagnostics(ws["id"])

        assert result["status"] == "partial"
        assert len(result["diagnostics"]) == 1
        assert result["diagnostics"][0]["source"] == "go"
        assert result["summary"]["errors"] == 1

    def test_go_clean(self, svc, wdb, tmp_path):
        ws = _make_workspace(wdb, tmp_path, stack=["go"])
        project = Path(ws["path"])
        (project / "go.mod").write_text("module example.com/test")

        with patch.object(svc, "_run_diagnostic_command", self._mock_run()):
            result = svc.get_workspace_diagnostics(ws["id"])

        assert result["status"] == "ok"
        assert result["summary"]["errors"] == 0

    def test_timeline_events_emitted(self, svc, wdb, csdb, tmp_path):
        ws = _make_workspace(wdb, tmp_path, stack=["go"])
        project = Path(ws["path"])
        (project / "go.mod").write_text("module example.com/test")
        session = csdb.create_session(workspace_id=ws["id"])

        with patch.object(svc, "_run_diagnostic_command", self._mock_run()):
            svc.get_workspace_diagnostics(ws["id"], code_session_id=session["id"])

        events = csdb.list_events(session["id"])
        types = [e["type"] for e in events]
        assert "diagnostics.started" in types
        assert "diagnostics.completed" in types

    def test_diagnostics_persisted(self, svc, wdb, tmp_path):
        from hermes_state import CodeDiagnosticsDB

        ws = _make_workspace(wdb, tmp_path, stack=["go"])
        project = Path(ws["path"])
        (project / "go.mod").write_text("module example.com/test")

        with patch.object(svc, "_run_diagnostic_command", self._mock_run()):
            svc.get_workspace_diagnostics(ws["id"])

        db = CodeDiagnosticsDB(db_path=tmp_path / "state.db")
        try:
            records = db.list_diagnostics(ws["id"])
            assert len(records) >= 1
            assert records[0]["workspace_id"] == ws["id"]
            assert records[0]["source"] == "workspace"
        finally:
            db.close()

    def test_duration_ms_present(self, svc, wdb, tmp_path):
        ws = _make_workspace(wdb, tmp_path, stack=["go"])
        project = Path(ws["path"])
        (project / "go.mod").write_text("module example.com/test")

        with patch.object(svc, "_run_diagnostic_command", self._mock_run()):
            result = svc.get_workspace_diagnostics(ws["id"])

        assert "duration_ms" in result
        assert isinstance(result["duration_ms"], int)

    def test_created_at_present(self, svc, wdb, tmp_path):
        ws = _make_workspace(wdb, tmp_path, stack=["go"])
        project = Path(ws["path"])
        (project / "go.mod").write_text("module example.com/test")

        with patch.object(svc, "_run_diagnostic_command", self._mock_run()):
            result = svc.get_workspace_diagnostics(ws["id"])

        assert "created_at" in result
        assert isinstance(result["created_at"], str)


# =============================================================================
# File diagnostics filtering
# =============================================================================


class TestFileDiagnostics:
    @pytest.fixture()
    def tmp_db(self, tmp_path):
        return tmp_path / "state.db"

    @pytest.fixture()
    def wdb(self, tmp_db):
        from hermes_state import WorkspaceDB

        d = WorkspaceDB(db_path=tmp_db)
        yield d
        d.close()

    @pytest.fixture()
    def svc(self, tmp_db):
        from hermes_cli.code.lsp_service import CodeIntelligenceService

        return CodeIntelligenceService(db_path=tmp_db)

    def test_filters_by_file(self, svc, wdb, tmp_path):
        ws = _make_workspace(wdb, tmp_path, stack=["typescript", "node"])
        project = Path(ws["path"])
        (project / "tsconfig.json").write_text("{}")
        (project / "package.json").write_text(
            json.dumps({"scripts": {"typecheck": "tsc --noEmit"}})
        )

        tsc_output = (
            "src/App.tsx(10,5): error TS2322: Type 'string' is not assignable to type 'number'.\n"
            "src/utils.ts(20,1): error TS1005: ';' expected.\n"
        )

        mock_fn = MagicMock(return_value=(0, tsc_output, ""))
        with patch.object(svc, "_run_diagnostic_command", mock_fn):
            result = svc.get_file_diagnostics(ws["id"], "src/App.tsx")

        assert result["file_path"] == "src/App.tsx"
        assert len(result["diagnostics"]) == 1
        assert result["diagnostics"][0]["file"] == "src/App.tsx"
        assert result["summary"]["total"] == 1

    def test_no_match_returns_empty(self, svc, wdb, tmp_path):
        ws = _make_workspace(wdb, tmp_path, stack=["typescript", "node"])
        project = Path(ws["path"])
        (project / "tsconfig.json").write_text("{}")
        (project / "package.json").write_text(
            json.dumps({"scripts": {"typecheck": "tsc --noEmit"}})
        )

        tsc_output = "src/App.tsx(10,5): error TS2322: bad type.\n"

        mock_fn = MagicMock(return_value=(0, tsc_output, ""))
        with patch.object(svc, "_run_diagnostic_command", mock_fn):
            result = svc.get_file_diagnostics(ws["id"], "src/NotFound.tsx")

        assert result["diagnostics"] == []
        assert result["summary"]["total"] == 0


# =============================================================================
# Restart language services
# =============================================================================


class TestRestartLanguageServices:
    @pytest.fixture()
    def tmp_db(self, tmp_path):
        return tmp_path / "state.db"

    @pytest.fixture()
    def wdb(self, tmp_db):
        from hermes_state import WorkspaceDB

        d = WorkspaceDB(db_path=tmp_db)
        yield d
        d.close()

    @pytest.fixture()
    def svc(self, tmp_db):
        from hermes_cli.code.lsp_service import CodeIntelligenceService

        return CodeIntelligenceService(db_path=tmp_db)

    def test_returns_noop(self, svc, wdb, tmp_path):
        ws = _make_workspace(wdb, tmp_path)
        result = svc.restart_language_services(ws["id"])
        assert result["status"] == "noop"
        assert result["workspace_id"] == ws["id"]

    def test_nonexistent_workspace(self, svc):
        with pytest.raises(ValueError, match="not found"):
            svc.restart_language_services("nonexistent-id")


# =============================================================================
# Command safety
# =============================================================================


class TestCommandSafety:
    @pytest.fixture()
    def tmp_db(self, tmp_path):
        return tmp_path / "state.db"

    @pytest.fixture()
    def svc(self, tmp_db):
        from hermes_cli.code.lsp_service import CodeIntelligenceService

        return CodeIntelligenceService(db_path=tmp_db)

    def test_dangerous_commands_not_used(self, svc, tmp_path):
        """Ensure the service never builds destructive commands."""
        root = tmp_path / "project"
        root.mkdir()

        # Create a package.json with a typecheck script
        (root / "package.json").write_text(
            json.dumps({"scripts": {"typecheck": "tsc --noEmit"}})
        )
        (root / "tsconfig.json").write_text("{}")

        # Build commands and verify none are destructive
        typecheck_cmd = svc._build_typecheck_command(root)
        assert typecheck_cmd is not None
        assert "rm " not in typecheck_cmd
        assert "install" not in typecheck_cmd
        assert ";" not in typecheck_cmd

        lint_cmd = svc._build_lint_command(root)
        assert lint_cmd is None  # no lint script

    def test_go_commands_are_safe(self, svc, tmp_path):
        root = tmp_path / "goproject"
        root.mkdir()
        (root / "go.mod").write_text("module example.com/test")

        vet_cmd = svc._build_go_vet_command(root)
        assert vet_cmd == "go vet ./..."

        test_cmd = svc._build_go_test_command(root)
        assert test_cmd == "go test ./..."

    def test_classify_blocks_npm_install(self, svc):
        safety = svc._classify_command_safety("npm install")
        from hermes_cli.code.command_runner import CommandSafety

        assert safety in (CommandSafety.BLOCKED, CommandSafety.NEEDS_APPROVAL)

    def test_classify_allows_go_vet(self, svc):
        from hermes_cli.code.command_runner import CommandSafety

        assert svc._classify_command_safety("go vet ./...") == CommandSafety.SAFE

    def test_classify_allows_npm_run_typecheck(self, svc):
        from hermes_cli.code.command_runner import CommandSafety

        assert svc._classify_command_safety("npm run typecheck") == CommandSafety.SAFE


# =============================================================================
# CodeDiagnosticsDB persistence
# =============================================================================


class TestCodeDiagnosticsDB:
    @pytest.fixture()
    def db_path(self, tmp_path):
        return tmp_path / "state.db"

    @pytest.fixture()
    def ddb(self, db_path):
        from hermes_state import CodeDiagnosticsDB

        d = CodeDiagnosticsDB(db_path=db_path)
        yield d
        d.close()

    @pytest.fixture()
    def wdb(self, db_path):
        from hermes_state import WorkspaceDB

        d = WorkspaceDB(db_path=db_path)
        yield d
        d.close()

    def test_save_and_get(self, ddb, wdb, tmp_path):
        ws = wdb.upsert_workspace(
            path=str(tmp_path / "proj"), name="proj", detected_stack=["go"]
        )
        saved = ddb.save_diagnostics(
            workspace_id=ws["id"],
            code_session_id=None,
            source="workspace",
            status="ok",
            diagnostics=[{"file": "main.go", "severity": "error", "message": "test"}],
            summary={"errors": 1, "warnings": 0, "info": 0, "hints": 0, "total": 1},
            commands=["go vet ./..."],
            duration_ms=500,
        )
        assert saved["id"]
        assert saved["workspace_id"] == ws["id"]
        assert saved["status"] == "ok"
        assert len(saved["diagnostics"]) == 1
        assert saved["diagnostics"][0]["file"] == "main.go"
        assert saved["duration_ms"] == 500

    def test_list_diagnostics(self, ddb, wdb, tmp_path):
        ws = wdb.upsert_workspace(
            path=str(tmp_path / "proj"), name="proj", detected_stack=["go"]
        )
        ddb.save_diagnostics(
            workspace_id=ws["id"],
            code_session_id=None,
            source="workspace",
            status="ok",
            diagnostics=[],
            summary={"errors": 0, "total": 0},
            commands=[],
            duration_ms=100,
        )
        ddb.save_diagnostics(
            workspace_id=ws["id"],
            code_session_id=None,
            source="workspace",
            status="error",
            diagnostics=[{"file": "a.go", "severity": "error"}],
            summary={"errors": 1, "total": 1},
            commands=["go vet"],
            duration_ms=200,
        )
        records = ddb.list_diagnostics(ws["id"])
        assert len(records) == 2
        statuses = {r["status"] for r in records}
        assert "ok" in statuses
        assert "error" in statuses

    def test_get_latest(self, ddb, wdb, tmp_path):
        ws = wdb.upsert_workspace(
            path=str(tmp_path / "proj"), name="proj", detected_stack=["go"]
        )
        ddb.save_diagnostics(
            workspace_id=ws["id"],
            code_session_id=None,
            source="workspace",
            status="ok",
            diagnostics=[],
            summary={},
            commands=[],
            duration_ms=50,
        )
        latest = ddb.get_latest_diagnostics(ws["id"])
        assert latest is not None
        assert latest["status"] == "ok"

    def test_get_nonexistent(self, ddb):
        assert ddb.get_diagnostics("nonexistent-id") is None

    def test_get_latest_empty(self, ddb, wdb, tmp_path):
        ws = wdb.upsert_workspace(
            path=str(tmp_path / "proj"), name="proj", detected_stack=["go"]
        )
        assert ddb.get_latest_diagnostics(ws["id"]) is None


# =============================================================================
# Schema version
# =============================================================================


class TestSchemaVersion:
    @pytest.fixture()
    def tmp_db(self, tmp_path):
        return tmp_path / "state.db"

    def test_schema_version_is_18(self, tmp_db):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_db)
        try:
            cursor = db._conn.execute("SELECT version FROM schema_version")
            row = cursor.fetchone()
            version = row["version"] if hasattr(row, "keys") else row[0]
            assert version == 18
        finally:
            db.close()

    def test_code_diagnostics_table_exists(self, tmp_db):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_db)
        try:
            cursor = db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='code_diagnostics'"
            )
            assert cursor.fetchone() is not None
        finally:
            db.close()


# =============================================================================
# REST endpoints
# =============================================================================


class TestDiagnosticsEndpoints:
    """Test the FastAPI REST endpoints using Starlette TestClient."""

    @pytest.fixture(autouse=True)
    def _setup_test_client(self):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        from hermes_cli.web_server import app, _SESSION_TOKEN

        self.client = TestClient(app)
        self.client.headers["Authorization"] = f"Bearer {_SESSION_TOKEN}"

    def _open_workspace(self, path: str) -> str:
        """Open a workspace and return its ID."""
        resp = self.client.post(
            "/api/code/workspaces/open",
            json={"path": path},
        )
        assert resp.status_code == 200
        return resp.json()["workspace"]["id"]

    def test_get_supported_languages(self, tmp_path):
        project = tmp_path / "tsproject"
        project.mkdir()
        (project / "package.json").write_text(
            json.dumps({"scripts": {"lint": "eslint ."}})
        )
        (project / "tsconfig.json").write_text("{}")

        ws_id = self._open_workspace(str(project))
        resp = self.client.get(f"/api/code/workspaces/{ws_id}/languages")
        assert resp.status_code == 200
        data = resp.json()
        assert "languages" in data
        assert "typescript" in data["languages"]

    def test_get_workspace_diagnostics_unsupported(self, tmp_path):
        project = tmp_path / "empty"
        project.mkdir()

        ws_id = self._open_workspace(str(project))
        resp = self.client.get(f"/api/code/workspaces/{ws_id}/diagnostics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["result"]["status"] == "unsupported"

    def test_restart_language_services(self, tmp_path):
        project = tmp_path / "tsproject"
        project.mkdir()
        (project / "tsconfig.json").write_text("{}")

        ws_id = self._open_workspace(str(project))
        resp = self.client.post(f"/api/code/workspaces/{ws_id}/lsp/restart")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "noop"

    def test_nonexistent_workspace_diagnostics(self):
        resp = self.client.get("/api/code/workspaces/nonexistent-id/diagnostics")
        assert resp.status_code == 404

    def test_nonexistent_workspace_languages(self):
        resp = self.client.get("/api/code/workspaces/nonexistent-id/languages")
        assert resp.status_code == 404

    def test_nonexistent_workspace_restart(self):
        resp = self.client.post("/api/code/workspaces/nonexistent-id/lsp/restart")
        assert resp.status_code == 404

    def test_nonexistent_workspace_file_diagnostics(self):
        resp = self.client.get(
            "/api/code/workspaces/nonexistent-id/diagnostics/file?path=src/app.ts"
        )
        assert resp.status_code == 404
