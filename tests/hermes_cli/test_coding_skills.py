"""Tests for Phase 9 Hermes Code Mode: CodingSkills."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def db_path(tmp_path):
    return tmp_path / "state.db"


@pytest.fixture()
def workspace(db_path, tmp_path):
    from hermes_state import WorkspaceDB

    project = tmp_path / "myproject"
    project.mkdir()
    wdb = WorkspaceDB(db_path=db_path)
    ws = wdb.upsert_workspace(
        path=str(project),
        name="myproject",
        is_git_repo=True,
        branch="main",
        detected_stack=["python"],
    )
    wdb.close()
    return ws


@pytest.fixture()
def code_session(db_path, workspace):
    from hermes_state import CodeSessionDB

    db = CodeSessionDB(db_path=db_path)
    session = db.create_session(
        workspace_id=workspace["id"],
        title="Test session",
        provider="anthropic",
        model="claude-sonnet-4-5",
    )
    db.close()
    return session


@pytest.fixture()
def svc(db_path):
    from hermes_cli.code.coding_skills import CodingSkillsService

    return CodingSkillsService(db_path=db_path)


# =============================================================================
# SkillRunDB
# =============================================================================


class TestSkillRunDB:
    @pytest.fixture()
    def db(self, db_path):
        from hermes_cli.code.coding_skills import SkillRunDB

        d = SkillRunDB(db_path=db_path)
        yield d
        d.close()

    def test_create_run(self, db, workspace):
        from hermes_cli.code.coding_skills import _utc_now

        now = _utc_now()
        run = db.create_run(
            run_id="run-1",
            skill_name="fix_build",
            workspace_id=workspace["id"],
            code_session_id=None,
            task_id=None,
            input_data={"commands": ["pytest"]},
            now=now,
        )
        assert run["id"] == "run-1"
        assert run["status"] == "created"
        assert run["skill_name"] == "fix_build"
        assert run["input"]["commands"] == ["pytest"]

    def test_get_run_not_found(self, db):
        assert db.get_run("nonexistent") is None

    def test_list_runs_empty(self, db, workspace):
        runs = db.list_runs(workspace_id=workspace["id"])
        assert runs == []

    def test_list_runs(self, db, workspace):
        from hermes_cli.code.coding_skills import _utc_now

        now = _utc_now()
        for i in range(3):
            db.create_run(
                run_id=f"run-{i}",
                skill_name="fix_build",
                workspace_id=workspace["id"],
                code_session_id=None,
                task_id=None,
                input_data={},
                now=now,
            )
        runs = db.list_runs(workspace_id=workspace["id"])
        assert len(runs) == 3

    def test_list_runs_filter_skill(self, db, workspace):
        from hermes_cli.code.coding_skills import _utc_now

        now = _utc_now()
        db.create_run(run_id="r1", skill_name="fix_build", workspace_id=workspace["id"],
                      code_session_id=None, task_id=None, input_data={}, now=now)
        db.create_run(run_id="r2", skill_name="review_diff", workspace_id=workspace["id"],
                      code_session_id=None, task_id=None, input_data={}, now=now)

        fix_runs = db.list_runs(skill_name="fix_build")
        assert len(fix_runs) == 1
        assert fix_runs[0]["skill_name"] == "fix_build"

    def test_update_run_status(self, db, workspace):
        from hermes_cli.code.coding_skills import _utc_now

        now = _utc_now()
        db.create_run(run_id="r1", skill_name="fix_build", workspace_id=workspace["id"],
                      code_session_id=None, task_id=None, input_data={}, now=now)
        updated = db.update_run("r1", status="running")
        assert updated["status"] == "running"

    def test_update_run_output(self, db, workspace):
        from hermes_cli.code.coding_skills import _utc_now

        now = _utc_now()
        db.create_run(run_id="r1", skill_name="fix_build", workspace_id=workspace["id"],
                      code_session_id=None, task_id=None, input_data={}, now=now)
        updated = db.update_run("r1", output={"errors": 2}, summary="2 errors found")
        assert updated["output"]["errors"] == 2
        assert updated["summary"] == "2 errors found"


# =============================================================================
# CodingSkillsService — core
# =============================================================================


class TestCodingSkillsService:
    def test_list_skills(self, svc):
        skills = svc.list_skills()
        names = [s["name"] for s in skills]
        assert "fix_build" in names
        assert "review_diff" in names
        assert "stabilize_hanging_task" in names
        assert "fix_runtime_error" in names
        assert "implement_feature" in names
        assert "refactor_react_page" in names
        assert "benchmark_provider" in names
        assert len(skills) == 7

    def test_create_run(self, svc, workspace):
        run = svc.create_run(
            skill_name="fix_build",
            workspace_id=workspace["id"],
        )
        assert run["id"]
        assert run["status"] == "created"
        assert run["skill_name"] == "fix_build"
        assert run["workspace_id"] == workspace["id"]

    def test_create_run_with_session(self, svc, workspace, code_session):
        run = svc.create_run(
            skill_name="review_diff",
            workspace_id=workspace["id"],
            code_session_id=code_session["id"],
        )
        assert run["code_session_id"] == code_session["id"]

    def test_create_run_unknown_skill(self, svc, workspace):
        with pytest.raises(ValueError, match="Unknown skill"):
            svc.create_run(skill_name="nonexistent_skill", workspace_id=workspace["id"])

    def test_create_run_invalid_workspace(self, svc):
        with pytest.raises(ValueError, match="Workspace not found"):
            svc.create_run(skill_name="fix_build", workspace_id="bad-id")

    def test_create_run_invalid_session(self, svc, workspace):
        with pytest.raises(ValueError, match="CodeSession not found"):
            svc.create_run(
                skill_name="fix_build",
                workspace_id=workspace["id"],
                code_session_id="bad-session",
            )

    def test_get_run(self, svc, workspace):
        run = svc.create_run(skill_name="fix_build", workspace_id=workspace["id"])
        retrieved = svc.get_run(run["id"])
        assert retrieved["id"] == run["id"]

    def test_get_run_not_found(self, svc):
        assert svc.get_run("nonexistent") is None

    def test_list_runs(self, svc, workspace):
        svc.create_run(skill_name="fix_build", workspace_id=workspace["id"])
        svc.create_run(skill_name="review_diff", workspace_id=workspace["id"])
        runs = svc.list_runs(workspace_id=workspace["id"])
        assert len(runs) == 2

    def test_list_runs_by_session(self, svc, workspace, code_session, db_path):
        from hermes_state import CodeSessionDB

        db = CodeSessionDB(db_path=db_path)
        session2 = db.create_session(workspace_id=workspace["id"], title="Session 2")
        db.close()

        svc.create_run(skill_name="fix_build", workspace_id=workspace["id"], code_session_id=code_session["id"])
        svc.create_run(skill_name="fix_build", workspace_id=workspace["id"], code_session_id=session2["id"])

        runs = svc.list_runs(code_session_id=code_session["id"])
        assert len(runs) == 1

    def test_cancel_run(self, svc, workspace):
        run = svc.create_run(skill_name="fix_build", workspace_id=workspace["id"])
        cancelled = svc.cancel_run(run["id"], reason="User cancelled")
        assert cancelled["status"] == "cancelled"
        assert cancelled["error"] == "User cancelled"

    def test_cancel_run_terminal(self, svc, workspace):
        run = svc.create_run(skill_name="fix_build", workspace_id=workspace["id"])
        svc.cancel_run(run["id"])
        with pytest.raises(ValueError, match="terminal state"):
            svc.cancel_run(run["id"])

    def test_cancel_run_not_found(self, svc):
        with pytest.raises(ValueError, match="Skill run not found"):
            svc.cancel_run("nonexistent")

    def test_resume_run_wrong_status(self, svc, workspace):
        run = svc.create_run(skill_name="fix_build", workspace_id=workspace["id"])
        with pytest.raises(ValueError, match="not waiting_approval"):
            svc.resume_run(run["id"])

    def test_run_skill_invalid_status(self, svc, workspace):
        run = svc.create_run(skill_name="fix_build", workspace_id=workspace["id"])
        svc.cancel_run(run["id"])
        with pytest.raises(ValueError, match="cannot be executed from status"):
            svc.run_skill(run["id"])


# =============================================================================
# fix_build
# =============================================================================


class TestFixBuild:
    def _mock_services(self, svc):
        mock_lsp = MagicMock()
        mock_lsp.run_diagnostics.return_value = {
            "id": "d1", "status": "ok", "summary": {"errors": 0, "warnings": 0}
        }

        mock_runner = MagicMock()
        mock_runner.classify_command.return_value = "safe"
        mock_runner.create_command.return_value = {"id": "cmd-1"}
        mock_runner.run_command_sync.return_value = {"exit_code": 0, "stdout": "passed", "stderr": ""}

        return mock_lsp, mock_runner

    def test_fix_build_runs_safe_commands(self, svc, workspace, code_session):
        mock_lsp, mock_runner = self._mock_services(svc)

        run = svc.create_run(
            skill_name="fix_build",
            workspace_id=workspace["id"],
            code_session_id=code_session["id"],
            input_data={"commands": ["python -m pytest"]},
        )

        with (
            patch.object(svc, "_lsp_service", return_value=mock_lsp),
            patch.object(svc, "_command_runner", return_value=mock_runner),
        ):
            result = svc.run_skill(run["id"])

        assert result["status"] == "completed"
        assert "passed" in result["summary"].lower() or "build" in result["summary"].lower()

    def test_fix_build_detects_errors(self, svc, workspace, code_session):
        mock_lsp = MagicMock()
        mock_lsp.run_diagnostics.return_value = {"id": "d1", "status": "ok", "summary": {"errors": 0}}

        mock_runner = MagicMock()
        mock_runner.classify_command.return_value = "safe"
        mock_runner.create_command.return_value = {"id": "cmd-1"}
        mock_runner.run_command_sync.return_value = {"exit_code": 1, "stdout": "", "stderr": "2 errors found"}

        run = svc.create_run(
            skill_name="fix_build",
            workspace_id=workspace["id"],
            code_session_id=code_session["id"],
            input_data={"commands": ["npm run typecheck"]},
        )

        with (
            patch.object(svc, "_lsp_service", return_value=mock_lsp),
            patch.object(svc, "_command_runner", return_value=mock_runner),
        ):
            result = svc.run_skill(run["id"])

        assert result["status"] == "completed"
        assert result["output"]["errors_found"]

    def test_fix_build_needs_approval_pauses(self, svc, workspace, code_session):
        """needs_approval command → waiting_approval."""
        mock_lsp = MagicMock()
        mock_lsp.run_diagnostics.return_value = {"id": "d1", "status": "ok", "summary": {}}

        mock_runner = MagicMock()
        mock_runner.classify_command.return_value = "needs_approval"

        run = svc.create_run(
            skill_name="fix_build",
            workspace_id=workspace["id"],
            code_session_id=code_session["id"],
            input_data={"commands": ["npm install"]},
        )

        with (
            patch.object(svc, "_lsp_service", return_value=mock_lsp),
            patch.object(svc, "_command_runner", return_value=mock_runner),
        ):
            result = svc.run_skill(run["id"])

        assert result["status"] == "waiting_approval"
        assert result["approval_id"] is not None

    def test_fix_build_blocked_command_skipped(self, svc, workspace, code_session):
        """blocked command skipped with warning."""
        mock_lsp = MagicMock()
        mock_lsp.run_diagnostics.return_value = {"id": "d1", "status": "ok", "summary": {}}

        mock_runner = MagicMock()
        mock_runner.classify_command.return_value = "blocked"

        run = svc.create_run(
            skill_name="fix_build",
            workspace_id=workspace["id"],
            code_session_id=code_session["id"],
            input_data={"commands": ["rm -rf /"]},
        )

        with (
            patch.object(svc, "_lsp_service", return_value=mock_lsp),
            patch.object(svc, "_command_runner", return_value=mock_runner),
        ):
            result = svc.run_skill(run["id"])

        assert result["status"] == "completed"
        # Blocked command skipped — no error fatal
        blocked = [c for c in result["commands"] if c.get("skipped")]
        assert len(blocked) == 1

    def test_fix_build_records_diagnostics(self, svc, workspace, code_session):
        mock_lsp = MagicMock()
        mock_lsp.run_diagnostics.return_value = {
            "id": "d1", "status": "ok", "summary": {"errors": 0, "warnings": 2}
        }

        mock_runner = MagicMock()
        mock_runner.classify_command.return_value = "safe"
        mock_runner.create_command.return_value = {"id": "cmd-1"}
        mock_runner.run_command_sync.return_value = {"exit_code": 0, "stdout": "ok", "stderr": ""}

        run = svc.create_run(
            skill_name="fix_build",
            workspace_id=workspace["id"],
            code_session_id=code_session["id"],
            input_data={"commands": ["python -m pytest"]},
        )

        with (
            patch.object(svc, "_lsp_service", return_value=mock_lsp),
            patch.object(svc, "_command_runner", return_value=mock_runner),
        ):
            result = svc.run_skill(run["id"])

        # diagnostics_before should be set
        assert result["diagnostics_before"] is not None
        assert result["diagnostics_before"]["summary"]["warnings"] == 2

    def test_fix_build_auto_creates_agent_flow(self, svc, workspace, code_session):
        """auto_fix=True + errors → agent flow created."""
        mock_lsp = MagicMock()
        mock_lsp.run_diagnostics.return_value = {"id": "d1", "status": "ok", "summary": {}}

        mock_runner = MagicMock()
        mock_runner.classify_command.return_value = "safe"
        mock_runner.create_command.return_value = {"id": "cmd-1"}
        mock_runner.run_command_sync.return_value = {"exit_code": 1, "stdout": "", "stderr": "error"}

        mock_agent = MagicMock()
        mock_agent.create_flow.return_value = {"id": "flow-abc"}

        run = svc.create_run(
            skill_name="fix_build",
            workspace_id=workspace["id"],
            code_session_id=code_session["id"],
            input_data={"commands": ["npm run typecheck"], "auto_fix": True},
        )

        with (
            patch.object(svc, "_lsp_service", return_value=mock_lsp),
            patch.object(svc, "_command_runner", return_value=mock_runner),
            patch.object(svc, "_agent_service", return_value=mock_agent),
        ):
            result = svc.run_skill(run["id"])

        assert result["agent_flow_id"] == "flow-abc"


# =============================================================================
# review_diff
# =============================================================================


class TestReviewDiff:
    def test_review_diff_no_changes(self, svc, workspace, code_session):
        mock_git = MagicMock()
        mock_git.get_status.return_value = {"branch": "main", "files": []}
        mock_git.get_diff.return_value = {"diff": ""}

        mock_lsp = MagicMock()
        mock_lsp.run_diagnostics.return_value = {"id": "d1", "status": "ok", "summary": {"errors": 0}}

        run = svc.create_run(
            skill_name="review_diff",
            workspace_id=workspace["id"],
            code_session_id=code_session["id"],
        )

        with (
            patch.object(svc, "_git_service", return_value=mock_git),
            patch.object(svc, "_lsp_service", return_value=mock_lsp),
        ):
            result = svc.run_skill(run["id"])

        assert result["status"] == "completed"
        assert result["output"]["decision"] == "approve"
        assert result["output"]["files_changed"] == []

    def test_review_diff_with_files(self, svc, workspace, code_session):
        mock_git = MagicMock()
        mock_git.get_status.return_value = {
            "branch": "main",
            "files": [{"path": "src/App.tsx"}, {"path": "src/utils.ts"}],
        }
        mock_git.get_diff.return_value = {"diff": "diff content here"}

        mock_lsp = MagicMock()
        mock_lsp.run_diagnostics.return_value = {"id": "d1", "status": "ok", "summary": {"errors": 0}}

        run = svc.create_run(
            skill_name="review_diff",
            workspace_id=workspace["id"],
            code_session_id=code_session["id"],
        )

        with (
            patch.object(svc, "_git_service", return_value=mock_git),
            patch.object(svc, "_lsp_service", return_value=mock_lsp),
        ):
            result = svc.run_skill(run["id"])

        assert result["status"] == "waiting_approval"
        assert result["approval_id"] is not None
        assert any("App.tsx" in f for f in result["output"]["files_changed"])

    def test_review_diff_detects_secret_risk(self, svc, workspace, code_session):
        mock_git = MagicMock()
        mock_git.get_status.return_value = {"branch": "main", "files": [{"path": ".env"}]}
        mock_git.get_diff.return_value = {"diff": "+API_KEY = secret123"}

        mock_lsp = MagicMock()
        mock_lsp.run_diagnostics.return_value = {"id": "d1", "status": "ok", "summary": {"errors": 0}}

        run = svc.create_run(
            skill_name="review_diff",
            workspace_id=workspace["id"],
            code_session_id=code_session["id"],
        )

        with (
            patch.object(svc, "_git_service", return_value=mock_git),
            patch.object(svc, "_lsp_service", return_value=mock_lsp),
        ):
            result = svc.run_skill(run["id"])

        assert any("secret" in r.lower() or "credential" in r.lower() for r in result["output"]["risks"])
        assert result["output"]["decision"] == "blocked"

    def test_review_diff_detects_dependency_risk(self, svc, workspace, code_session):
        mock_git = MagicMock()
        mock_git.get_status.return_value = {
            "branch": "main",
            "files": [{"path": "package.json"}, {"path": "package-lock.json"}],
        }
        mock_git.get_diff.return_value = {"diff": "dependency bump"}

        mock_lsp = MagicMock()
        mock_lsp.run_diagnostics.return_value = {"id": "d1", "status": "ok", "summary": {"errors": 0}}

        run = svc.create_run(
            skill_name="review_diff",
            workspace_id=workspace["id"],
            code_session_id=code_session["id"],
        )

        with (
            patch.object(svc, "_git_service", return_value=mock_git),
            patch.object(svc, "_lsp_service", return_value=mock_lsp),
        ):
            result = svc.run_skill(run["id"])

        assert any("dependency" in r.lower() for r in result["output"]["risks"])

    def test_review_diff_includes_diagnostics_summary(self, svc, workspace, code_session):
        mock_git = MagicMock()
        mock_git.get_status.return_value = {"branch": "main", "files": []}
        mock_git.get_diff.return_value = {"diff": ""}

        mock_lsp = MagicMock()
        mock_lsp.run_diagnostics.return_value = {
            "id": "d1", "status": "ok", "summary": {"errors": 3, "warnings": 1}
        }

        run = svc.create_run(
            skill_name="review_diff",
            workspace_id=workspace["id"],
            code_session_id=code_session["id"],
        )

        with (
            patch.object(svc, "_git_service", return_value=mock_git),
            patch.object(svc, "_lsp_service", return_value=mock_lsp),
        ):
            result = svc.run_skill(run["id"])

        assert result["output"]["diagnostics_summary"]["errors"] == 3
        # Errors detected → request_changes
        assert result["output"]["decision"] == "request_changes"


# =============================================================================
# stabilize_hanging_task
# =============================================================================


class TestStabilizeHangingTask:
    def test_stabilize_no_stuck_tasks(self, svc, workspace):
        run = svc.create_run(
            skill_name="stabilize_hanging_task",
            workspace_id=workspace["id"],
        )

        mock_agent = MagicMock()
        mock_agent.list_flows.return_value = []

        with (
            patch.object(svc, "_agent_service", return_value=mock_agent),
        ):
            result = svc.run_skill(run["id"])

        assert result["status"] == "completed"
        assert "no stuck" in result["summary"].lower()

    def test_stabilize_detects_stuck_flows(self, svc, workspace, code_session):
        run = svc.create_run(
            skill_name="stabilize_hanging_task",
            workspace_id=workspace["id"],
            code_session_id=code_session["id"],
        )

        mock_agent = MagicMock()
        mock_agent.list_flows.return_value = [
            {"id": "flow-1", "status": "coding"},
            {"id": "flow-2", "status": "running_tests"},
        ]

        with (
            patch.object(svc, "_agent_service", return_value=mock_agent),
        ):
            result = svc.run_skill(run["id"])

        assert result["status"] == "completed"
        assert len(result["output"]["stuck_flows"]) == 2

    def test_stabilize_cancels_safe_commands(self, svc, workspace, code_session):
        run = svc.create_run(
            skill_name="stabilize_hanging_task",
            workspace_id=workspace["id"],
            code_session_id=code_session["id"],
        )

        mock_agent = MagicMock()
        mock_agent.list_flows.return_value = []

        mock_runner = MagicMock()
        mock_runner.list_commands.return_value = [
            {"id": "cmd-1", "command": "python -m pytest", "status": "running"}
        ]
        mock_runner.classify_command.return_value = "safe"
        mock_runner.cancel_command.return_value = {"id": "cmd-1", "status": "cancelled"}

        with (
            patch.object(svc, "_agent_service", return_value=mock_agent),
            patch.object(svc, "_command_runner", return_value=mock_runner),
        ):
            result = svc.run_skill(run["id"])

        assert result["status"] == "completed"
        cancelled = [s for s in result["output"]["stabilized"] if s.get("action") == "cancelled"]
        assert len(cancelled) == 1

    def test_stabilize_requests_approval_for_sensitive_commands(self, svc, workspace, code_session):
        run = svc.create_run(
            skill_name="stabilize_hanging_task",
            workspace_id=workspace["id"],
            code_session_id=code_session["id"],
        )

        mock_agent = MagicMock()
        mock_agent.list_flows.return_value = []

        mock_runner = MagicMock()
        mock_runner.list_commands.return_value = [
            {"id": "cmd-1", "command": "npm install", "status": "running"}
        ]
        mock_runner.classify_command.return_value = "needs_approval"

        with (
            patch.object(svc, "_agent_service", return_value=mock_agent),
            patch.object(svc, "_command_runner", return_value=mock_runner),
        ):
            result = svc.run_skill(run["id"])

        assert result["status"] == "completed"
        approval_entries = [s for s in result["output"]["stabilized"] if s.get("action") == "approval_requested"]
        assert len(approval_entries) == 1


# =============================================================================
# implement_feature
# =============================================================================


class TestImplementFeature:
    def test_implement_feature_creates_agent_flow(self, svc, workspace, code_session):
        run = svc.create_run(
            skill_name="implement_feature",
            workspace_id=workspace["id"],
            code_session_id=code_session["id"],
            input_data={"description": "Add dark mode toggle to settings page"},
        )

        mock_agent = MagicMock()
        mock_agent.create_flow.return_value = {"id": "flow-xyz"}
        mock_agent.run_flow.return_value = {"id": "flow-xyz", "status": "completed"}

        with patch.object(svc, "_agent_service", return_value=mock_agent):
            result = svc.run_skill(run["id"])

        assert result["status"] == "completed"
        assert result["agent_flow_id"] == "flow-xyz"
        mock_agent.create_flow.assert_called_once()
        mock_agent.run_flow.assert_called_once()

    def test_implement_feature_requires_session(self, svc, workspace):
        run = svc.create_run(
            skill_name="implement_feature",
            workspace_id=workspace["id"],
            input_data={"description": "Add feature"},
        )
        result = svc.run_skill(run["id"])
        assert result["status"] == "failed"
        assert result["error"]

    def test_implement_feature_waiting_approval_propagates(self, svc, workspace, code_session):
        run = svc.create_run(
            skill_name="implement_feature",
            workspace_id=workspace["id"],
            code_session_id=code_session["id"],
            input_data={"description": "risky feature"},
        )

        mock_agent = MagicMock()
        mock_agent.create_flow.return_value = {"id": "flow-xyz"}
        mock_agent.run_flow.return_value = {"id": "flow-xyz", "status": "waiting_approval"}

        with patch.object(svc, "_agent_service", return_value=mock_agent):
            result = svc.run_skill(run["id"])

        assert result["status"] == "waiting_approval"


# =============================================================================
# benchmark_provider
# =============================================================================


class TestBenchmarkProvider:
    def test_benchmark_dry_run_default(self, svc, workspace, code_session):
        run = svc.create_run(
            skill_name="benchmark_provider",
            workspace_id=workspace["id"],
            code_session_id=code_session["id"],
            input_data={
                "task_prompt": "Implement auth",
                "providers": [{"provider": "anthropic", "model": "claude-sonnet-4-5"}],
                "dry_run": True,
            },
        )

        mock_agent = MagicMock()
        mock_agent.create_flow.return_value = {"id": "flow-1"}
        mock_agent._build_plan.return_value = {
            "steps": [{"role": "coder", "name": "s1"}],
            "test_commands": ["pytest"],
            "risks": [],
        }

        with patch.object(svc, "_agent_service", return_value=mock_agent):
            result = svc.run_skill(run["id"])

        assert result["status"] == "completed"
        assert result["output"]["dry_run"] is True
        assert len(result["output"]["results"]) == 1

    def test_benchmark_non_dry_run_requires_approval(self, svc, workspace, code_session):
        run = svc.create_run(
            skill_name="benchmark_provider",
            workspace_id=workspace["id"],
            code_session_id=code_session["id"],
            input_data={
                "task_prompt": "Implement feature",
                "providers": [{"provider": "anthropic", "model": "claude-sonnet-4-5"}],
                "dry_run": False,
            },
        )

        result = svc.run_skill(run["id"])

        assert result["status"] == "waiting_approval"
        assert result["approval_id"] is not None


# =============================================================================
# refactor_react_page
# =============================================================================


class TestRefactorReactPage:
    def test_refactor_creates_approval(self, svc, workspace, code_session):
        mock_lsp = MagicMock()
        mock_lsp.run_diagnostics.return_value = {"id": "d1", "status": "ok", "summary": {"errors": 0}}

        run = svc.create_run(
            skill_name="refactor_react_page",
            workspace_id=workspace["id"],
            code_session_id=code_session["id"],
            input_data={"page_path": "src/features/Chat/ChatPage.tsx", "goal": "Extract sidebar component"},
        )

        with patch.object(svc, "_lsp_service", return_value=mock_lsp):
            result = svc.run_skill(run["id"])

        assert result["status"] == "waiting_approval"
        assert result["approval_id"] is not None

    def test_refactor_preserves_hermesweb_principles(self, svc, workspace, code_session):
        mock_lsp = MagicMock()
        mock_lsp.run_diagnostics.return_value = {"id": "d1", "status": "ok", "summary": {"errors": 0}}

        run = svc.create_run(
            skill_name="refactor_react_page",
            workspace_id=workspace["id"],
            code_session_id=code_session["id"],
            input_data={"page_path": "src/Page.tsx", "goal": "Refactor layout"},
        )

        with patch.object(svc, "_lsp_service", return_value=mock_lsp):
            result = svc.run_skill(run["id"])

        principles = result["output"].get("hermesWeb_principles", [])
        assert any("dark mode" in p.lower() for p in principles)
        assert any("animation" in p.lower() for p in principles)


# =============================================================================
# fix_runtime_error
# =============================================================================


class TestFixRuntimeError:
    def test_fix_runtime_error_extracts_files(self, svc, workspace, code_session):
        mock_lsp = MagicMock()
        mock_lsp.run_diagnostics.return_value = {"id": "d1", "status": "ok", "summary": {"errors": 2}}

        run = svc.create_run(
            skill_name="fix_runtime_error",
            workspace_id=workspace["id"],
            code_session_id=code_session["id"],
            input_data={
                "error_message": "TypeError: Cannot read property 'map' of undefined",
                "stack_trace": "at ChatPage.tsx:42:10\nat App.tsx:15:5",
            },
        )

        with patch.object(svc, "_lsp_service", return_value=mock_lsp):
            result = svc.run_skill(run["id"])

        assert result["status"] == "completed"
        assert result["output"]["file_refs"]
        assert result["output"]["diagnostics_summary"]["errors"] == 2

    def test_fix_runtime_error_with_file_hint(self, svc, workspace, code_session):
        mock_lsp = MagicMock()
        mock_lsp.run_diagnostics.return_value = {"id": "d1", "status": "ok", "summary": {"errors": 0}}

        run = svc.create_run(
            skill_name="fix_runtime_error",
            workspace_id=workspace["id"],
            code_session_id=code_session["id"],
            input_data={
                "error_message": "build error",
                "file_hint": "src/components/Button.tsx",
            },
        )

        with patch.object(svc, "_lsp_service", return_value=mock_lsp):
            result = svc.run_skill(run["id"])

        assert "src/components/Button.tsx" in result["output"]["file_refs"]


# =============================================================================
# Schema version
# =============================================================================


class TestSchemaVersion:
    def test_schema_version_is_18(self):
        import hermes_state

        assert hermes_state.SCHEMA_VERSION == 18

    def test_schema_contains_code_skill_runs(self):
        import hermes_state

        assert "code_skill_runs" in hermes_state.SCHEMA_SQL

    def test_migration_creates_skill_runs_table(self, tmp_path):
        from hermes_cli.code.coding_skills import SkillRunDB

        db_path = tmp_path / "migration_test.db"
        db = SkillRunDB(db_path=db_path)
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='code_skill_runs'"
        )
        assert cursor.fetchone() is not None
        db.close()


# =============================================================================
# REST endpoints
# =============================================================================


class TestSkillEndpoints:
    @pytest.fixture()
    def client(self):
        from starlette.testclient import TestClient
        from hermes_cli.web_server import app, _SESSION_TOKEN

        c = TestClient(app)
        c.headers["Authorization"] = f"Bearer {_SESSION_TOKEN}"
        return c

    def test_list_skills(self, client):
        response = client.get("/api/code/skills")
        assert response.status_code == 200
        data = response.json()
        assert "skills" in data
        assert len(data["skills"]) == 7

    def test_list_skill_runs(self, client):
        response = client.get("/api/code/skill-runs")
        assert response.status_code == 200
        data = response.json()
        assert "runs" in data
        assert isinstance(data["runs"], list)

    def test_create_skill_run_missing_workspace(self, client):
        response = client.post(
            "/api/code/skill-runs",
            json={"skill_name": "fix_build", "workspace_id": "nonexistent"},
        )
        assert response.status_code == 404

    def test_create_skill_run_unknown_skill(self, client):
        response = client.post(
            "/api/code/skill-runs",
            json={"skill_name": "unknown_skill", "workspace_id": "some-ws"},
        )
        assert response.status_code == 404

    def test_get_skill_run_not_found(self, client):
        response = client.get("/api/code/skill-runs/nonexistent")
        assert response.status_code == 404

    def test_run_skill_not_found(self, client):
        response = client.post("/api/code/skill-runs/nonexistent/run")
        assert response.status_code == 400

    def test_cancel_skill_run_not_found(self, client):
        response = client.post("/api/code/skill-runs/nonexistent/cancel")
        assert response.status_code == 400

    def test_session_skill_runs_endpoint(self, client):
        response = client.get("/api/code/sessions/some-session/skill-runs")
        assert response.status_code == 200
        data = response.json()
        assert "runs" in data

    def test_skill_shortcut_endpoint_unknown_skill(self, client):
        response = client.post(
            "/api/code/skills/unknown_skill/run",
            json={"skill_name": "unknown_skill", "workspace_id": "bad"},
        )
        assert response.status_code == 400

    def test_detect_build_commands_python(self, svc, workspace, db_path):
        from hermes_state import WorkspaceDB

        wdb = WorkspaceDB(db_path=db_path)
        wdb.upsert_workspace(
            path=str(workspace["path"]),
            name="myproject",
            detected_stack=["python"],
        )
        wdb.close()

        commands = svc._detect_build_commands(workspace["id"])
        assert any("pytest" in c for c in commands)

    def test_detect_build_commands_node(self, svc, workspace, db_path):
        from hermes_state import WorkspaceDB

        wdb = WorkspaceDB(db_path=db_path)
        wdb.upsert_workspace(
            path=str(workspace["path"]),
            name="myproject",
            detected_stack=["typescript", "node"],
        )
        wdb.close()

        commands = svc._detect_build_commands(workspace["id"])
        assert any("typecheck" in c for c in commands)

    def test_events_registered_on_create(self, svc, workspace, code_session, db_path):
        from hermes_state import CodeSessionDB

        svc.create_run(
            skill_name="fix_build",
            workspace_id=workspace["id"],
            code_session_id=code_session["id"],
        )

        db = CodeSessionDB(db_path=db_path)
        events = db.list_events(code_session["id"])
        db.close()

        assert any(e["type"] == "skill.started" for e in events)
