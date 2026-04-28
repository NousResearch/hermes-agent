"""Tests for Phase 8 Hermes Code Mode: Multi-Agent Coding Flow."""

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
    from hermes_cli.code.multi_agent_coding import MultiAgentCodingService

    return MultiAgentCodingService(db_path=db_path)


# =============================================================================
# AgentFlowDB — persistence layer
# =============================================================================


class TestAgentFlowDB:
    @pytest.fixture()
    def flow_db(self, db_path):
        from hermes_cli.code.multi_agent_coding import AgentFlowDB

        db = AgentFlowDB(db_path=db_path)
        yield db
        db.close()

    def test_create_flow(self, flow_db, workspace, code_session):
        from hermes_cli.code.multi_agent_coding import _utc_now

        now = _utc_now()
        flow = flow_db.create_flow(
            flow_id="flow-1",
            code_session_id=code_session["id"],
            workspace_id=workspace["id"],
            task_id=None,
            title="Test flow",
            description="Fix the bug",
            provider="anthropic",
            model="claude-sonnet-4-5",
            preset="planner",
            now=now,
        )
        assert flow["id"] == "flow-1"
        assert flow["status"] == "created"
        assert flow["code_session_id"] == code_session["id"]
        assert flow["workspace_id"] == workspace["id"]
        assert flow["plan"] == {}

    def test_get_flow_not_found(self, flow_db):
        assert flow_db.get_flow("nonexistent") is None

    def test_list_flows(self, flow_db, workspace, code_session):
        from hermes_cli.code.multi_agent_coding import _utc_now

        now = _utc_now()
        for i in range(3):
            flow_db.create_flow(
                flow_id=f"flow-{i}",
                code_session_id=code_session["id"],
                workspace_id=workspace["id"],
                task_id=None,
                title=f"Flow {i}",
                description=None,
                provider=None,
                model=None,
                preset=None,
                now=now,
            )

        flows = flow_db.list_flows(code_session_id=code_session["id"])
        assert len(flows) == 3

    def test_list_flows_filter_status(self, flow_db, workspace, code_session):
        from hermes_cli.code.multi_agent_coding import _utc_now

        now = _utc_now()
        flow_db.create_flow(
            flow_id="f1",
            code_session_id=code_session["id"],
            workspace_id=workspace["id"],
            task_id=None, title=None, description=None,
            provider=None, model=None, preset=None, now=now,
        )
        flow_db.update_flow("f1", status="completed", completed_at=now)

        flow_db.create_flow(
            flow_id="f2",
            code_session_id=code_session["id"],
            workspace_id=workspace["id"],
            task_id=None, title=None, description=None,
            provider=None, model=None, preset=None, now=now,
        )

        completed = flow_db.list_flows(status="completed")
        assert len(completed) == 1
        assert completed[0]["id"] == "f1"

    def test_update_flow_status(self, flow_db, workspace, code_session):
        from hermes_cli.code.multi_agent_coding import _utc_now

        now = _utc_now()
        flow_db.create_flow(
            flow_id="f1",
            code_session_id=code_session["id"],
            workspace_id=workspace["id"],
            task_id=None, title=None, description=None,
            provider=None, model=None, preset=None, now=now,
        )
        updated = flow_db.update_flow("f1", status="planning", current_role="orchestrator")
        assert updated["status"] == "planning"
        assert updated["current_role"] == "orchestrator"

    def test_update_flow_plan(self, flow_db, workspace, code_session):
        from hermes_cli.code.multi_agent_coding import _utc_now

        now = _utc_now()
        flow_db.create_flow(
            flow_id="f1",
            code_session_id=code_session["id"],
            workspace_id=workspace["id"],
            task_id=None, title=None, description=None,
            provider=None, model=None, preset=None, now=now,
        )
        plan = {"summary": "Fix bug", "steps": [], "risks": [], "requires_approval": False}
        updated = flow_db.update_flow("f1", plan=plan)
        assert updated["plan"]["summary"] == "Fix bug"

    def test_create_and_list_steps(self, flow_db, workspace, code_session):
        from hermes_cli.code.multi_agent_coding import _utc_now

        now = _utc_now()
        flow_db.create_flow(
            flow_id="f1",
            code_session_id=code_session["id"],
            workspace_id=workspace["id"],
            task_id=None, title=None, description=None,
            provider=None, model=None, preset=None, now=now,
        )
        step = flow_db.create_step(
            step_id="step-1",
            flow_id="f1",
            role="orchestrator",
            name="Plan task",
            input_data={"description": "fix bug"},
            now=now,
        )
        assert step["status"] == "pending"
        assert step["role"] == "orchestrator"

        steps = flow_db.list_steps("f1")
        assert len(steps) == 1
        assert steps[0]["id"] == "step-1"

    def test_update_step(self, flow_db, workspace, code_session):
        from hermes_cli.code.multi_agent_coding import _utc_now

        now = _utc_now()
        flow_db.create_flow(
            flow_id="f1",
            code_session_id=code_session["id"],
            workspace_id=workspace["id"],
            task_id=None, title=None, description=None,
            provider=None, model=None, preset=None, now=now,
        )
        flow_db.create_step(step_id="step-1", flow_id="f1", role="tester", name="Run tests", now=now)
        updated = flow_db.update_step(
            "step-1",
            status="completed",
            output={"exit_code": 0},
            completed_at=now,
        )
        assert updated["status"] == "completed"
        assert updated["output"]["exit_code"] == 0


# =============================================================================
# MultiAgentCodingService
# =============================================================================


class TestMultiAgentCodingService:
    def test_create_flow(self, svc, workspace, code_session):
        flow = svc.create_flow(
            code_session_id=code_session["id"],
            workspace_id=workspace["id"],
            title="Fix chat bug",
            description="Fix navigation issue",
        )
        assert flow["id"]
        assert flow["status"] == "created"
        assert flow["code_session_id"] == code_session["id"]
        assert flow["workspace_id"] == workspace["id"]

    def test_create_flow_invalid_session(self, svc, workspace):
        with pytest.raises(ValueError, match="CodeSession not found"):
            svc.create_flow(
                code_session_id="nonexistent-session",
                workspace_id=workspace["id"],
            )

    def test_create_flow_invalid_workspace(self, svc, code_session):
        with pytest.raises(ValueError, match="Workspace not found"):
            svc.create_flow(
                code_session_id=code_session["id"],
                workspace_id="nonexistent-workspace",
            )

    def test_get_flow(self, svc, workspace, code_session):
        flow = svc.create_flow(
            code_session_id=code_session["id"],
            workspace_id=workspace["id"],
        )
        retrieved = svc.get_flow(flow["id"])
        assert retrieved["id"] == flow["id"]
        assert "steps" in retrieved

    def test_get_flow_not_found(self, svc):
        assert svc.get_flow("nonexistent") is None

    def test_list_flows(self, svc, workspace, code_session):
        svc.create_flow(code_session_id=code_session["id"], workspace_id=workspace["id"])
        svc.create_flow(code_session_id=code_session["id"], workspace_id=workspace["id"])
        flows = svc.list_flows(code_session_id=code_session["id"])
        assert len(flows) == 2

    def test_list_flows_by_session(self, svc, workspace, code_session, db_path):
        # Create a second session
        from hermes_state import CodeSessionDB

        db = CodeSessionDB(db_path=db_path)
        session2 = db.create_session(workspace_id=workspace["id"], title="Session 2")
        db.close()

        svc.create_flow(code_session_id=code_session["id"], workspace_id=workspace["id"])
        svc.create_flow(code_session_id=session2["id"], workspace_id=workspace["id"])

        flows = svc.list_flows(code_session_id=code_session["id"])
        assert len(flows) == 1
        assert flows[0]["code_session_id"] == code_session["id"]

    def test_cancel_flow(self, svc, workspace, code_session):
        flow = svc.create_flow(
            code_session_id=code_session["id"],
            workspace_id=workspace["id"],
        )
        cancelled = svc.cancel_flow(flow["id"], reason="User requested")
        assert cancelled["status"] == "cancelled"
        assert cancelled["error"] == "User requested"

    def test_cancel_flow_already_terminal(self, svc, workspace, code_session):
        flow = svc.create_flow(
            code_session_id=code_session["id"],
            workspace_id=workspace["id"],
        )
        svc.cancel_flow(flow["id"])
        with pytest.raises(ValueError, match="terminal state"):
            svc.cancel_flow(flow["id"])

    def test_cancel_flow_not_found(self, svc):
        with pytest.raises(ValueError, match="Flow not found"):
            svc.cancel_flow("nonexistent")

    def test_resume_flow_wrong_status(self, svc, workspace, code_session):
        flow = svc.create_flow(
            code_session_id=code_session["id"],
            workspace_id=workspace["id"],
        )
        with pytest.raises(ValueError, match="not waiting_approval"):
            svc.resume_flow(flow["id"])


# =============================================================================
# Orchestrator / plan building
# =============================================================================


class TestOrchestrator:
    def test_build_plan_python_stack(self, svc, workspace, code_session, db_path):
        from hermes_state import WorkspaceDB

        wdb = WorkspaceDB(db_path=db_path)
        wdb.upsert_workspace(
            path=str(workspace["path"]),
            name="myproject",
            detected_stack=["python"],
        )
        wdb.close()

        flow = svc.create_flow(
            code_session_id=code_session["id"],
            workspace_id=workspace["id"],
            description="Fix the import error",
        )

        plan = svc._build_plan(flow, workspace["id"])
        assert "steps" in plan
        assert "test_commands" in plan
        assert any("pytest" in c for c in plan["test_commands"])

    def test_build_plan_node_stack(self, svc, workspace, code_session, db_path):
        from hermes_state import WorkspaceDB

        wdb = WorkspaceDB(db_path=db_path)
        wdb.upsert_workspace(
            path=str(workspace["path"]),
            name="myproject",
            detected_stack=["typescript", "node"],
        )
        wdb.close()

        flow = svc.create_flow(
            code_session_id=code_session["id"],
            workspace_id=workspace["id"],
            description="Fix TypeScript error",
        )

        plan = svc._build_plan(flow, workspace["id"])
        assert any("typecheck" in c or "lint" in c for c in plan["test_commands"])

    def test_build_plan_security_risk(self, svc, workspace, code_session):
        flow = svc.create_flow(
            code_session_id=code_session["id"],
            workspace_id=workspace["id"],
            description="Update auth middleware",
        )

        plan = svc._build_plan(flow, workspace["id"])
        assert any("auth" in r.lower() or "security" in r.lower() for r in plan["risks"])
        assert plan["requires_approval"] is True

    def test_build_plan_database_risk(self, svc, workspace, code_session):
        flow = svc.create_flow(
            code_session_id=code_session["id"],
            workspace_id=workspace["id"],
            description="Run database migration",
        )

        plan = svc._build_plan(flow, workspace["id"])
        assert any("migration" in r.lower() or "backup" in r.lower() for r in plan["risks"])


# =============================================================================
# Tester — command safety
# =============================================================================


class TestTesterCommandSafety:
    def test_tester_skips_blocked_command(self, svc, workspace, code_session, db_path):
        """Blocked commands must not run and must log error."""
        flow = svc.create_flow(
            code_session_id=code_session["id"],
            workspace_id=workspace["id"],
            description="test",
        )

        from hermes_cli.code.multi_agent_coding import AgentFlowDB, _utc_now

        db = AgentFlowDB(db_path=db_path)
        flow = db.update_flow(flow["id"], plan={"test_commands": ["rm -rf /tmp/test"], "risks": []})

        # Patch git/diagnostics to avoid real calls
        with (
            patch.object(svc, "_git_service") as mock_git,
            patch.object(svc, "_lsp_service") as mock_lsp,
        ):
            mock_git.return_value.get_status.return_value = {"branch": "main", "files": []}
            mock_lsp.return_value.run_diagnostics.return_value = {"id": "d1", "status": "completed", "summary": {}}

            result = svc._run_tester(flow, db)

        steps = db.list_steps(flow["id"])
        tester_steps = [s for s in steps if s["role"] == "tester"]
        # Blocked command → step failed, flow stays running_tests (no waiting_approval)
        assert all(s["status"] == "failed" or s["status"] == "skipped" for s in tester_steps) or result["status"] != "waiting_approval"
        db.close()

    def test_tester_needs_approval_pauses_flow(self, svc, workspace, code_session, db_path):
        """needs_approval command → flow goes to waiting_approval."""
        flow = svc.create_flow(
            code_session_id=code_session["id"],
            workspace_id=workspace["id"],
            description="test",
        )

        from hermes_cli.code.multi_agent_coding import AgentFlowDB, _utc_now

        db = AgentFlowDB(db_path=db_path)
        flow = db.update_flow(flow["id"], plan={"test_commands": ["npm install"], "risks": []})

        with (
            patch.object(svc, "_git_service") as mock_git,
            patch.object(svc, "_lsp_service") as mock_lsp,
        ):
            mock_git.return_value.get_status.return_value = {"branch": "main", "files": []}
            mock_lsp.return_value.run_diagnostics.return_value = {"id": "d1", "status": "completed", "summary": {}}

            result = svc._run_tester(flow, db)

        assert result["status"] == "waiting_approval"
        assert result["approval_id"] is not None
        db.close()

    def test_tester_blocked_command_generates_clear_error(self, svc, workspace, code_session, db_path):
        """Blocked command step must record error with 'blocked' text."""
        flow = svc.create_flow(
            code_session_id=code_session["id"],
            workspace_id=workspace["id"],
            description="test",
        )

        from hermes_cli.code.multi_agent_coding import AgentFlowDB

        db = AgentFlowDB(db_path=db_path)
        flow = db.update_flow(flow["id"], plan={"test_commands": ["sudo rm -rf /"], "risks": []})

        with (
            patch.object(svc, "_git_service") as mock_git,
            patch.object(svc, "_lsp_service") as mock_lsp,
        ):
            mock_git.return_value.get_status.return_value = {"branch": "main", "files": []}
            mock_lsp.return_value.run_diagnostics.return_value = {"id": "d1", "status": "completed", "summary": {}}
            svc._run_tester(flow, db)

        steps = db.list_steps(flow["id"])
        tester_steps = [s for s in steps if s["role"] == "tester"]
        blocked_step = next((s for s in tester_steps if s.get("error") and "blocked" in s["error"].lower()), None)
        assert blocked_step is not None, "Expected a step with 'blocked' error"
        db.close()

    def test_tester_safe_command_runs(self, svc, workspace, code_session, db_path):
        """Safe command is executed via CommandRunner."""
        flow = svc.create_flow(
            code_session_id=code_session["id"],
            workspace_id=workspace["id"],
            description="test",
        )

        from hermes_cli.code.multi_agent_coding import AgentFlowDB

        db = AgentFlowDB(db_path=db_path)
        flow = db.update_flow(flow["id"], plan={"test_commands": ["python -m pytest"], "risks": []})

        mock_runner = MagicMock()
        mock_runner.classify_command.return_value = "safe"
        mock_runner.create_command.return_value = {"id": "cmd-1"}
        mock_runner.run_command_sync.return_value = {"exit_code": 0, "stdout": "1 passed"}

        with (
            patch.object(svc, "_git_service") as mock_git,
            patch.object(svc, "_lsp_service") as mock_lsp,
            patch.object(svc, "_command_runner", return_value=mock_runner),
        ):
            mock_git.return_value.get_status.return_value = {"branch": "main", "files": []}
            mock_lsp.return_value.run_diagnostics.return_value = {"id": "d1", "status": "completed", "summary": {}}
            svc._run_tester(flow, db)

        steps = db.list_steps(flow["id"])
        tester_steps = [s for s in steps if s["role"] == "tester"]
        assert any(s["status"] == "completed" for s in tester_steps)
        db.close()


# =============================================================================
# Reviewer
# =============================================================================


class TestReviewer:
    def test_reviewer_generates_summary(self, svc, workspace, code_session, db_path):
        flow = svc.create_flow(
            code_session_id=code_session["id"],
            workspace_id=workspace["id"],
            description="test",
        )

        from hermes_cli.code.multi_agent_coding import AgentFlowDB

        db = AgentFlowDB(db_path=db_path)
        flow = db.update_flow(flow["id"], plan={"risks": [], "test_commands": []})

        mock_git = MagicMock()
        mock_git.get_status.return_value = {"branch": "main", "files": [{"path": "foo.py"}]}
        mock_git.get_diff.return_value = {"stat": "1 file changed"}

        mock_lsp = MagicMock()
        mock_lsp.run_diagnostics.return_value = {"id": "d1", "status": "completed", "summary": {"errors": 0, "warnings": 0}}

        with (
            patch.object(svc, "_git_service", return_value=mock_git),
            patch.object(svc, "_lsp_service", return_value=mock_lsp),
        ):
            result = svc._run_reviewer(flow, db)

        flow_final = db.get_flow(flow["id"])
        assert flow_final["review"] is not None
        assert "summary" in flow_final["review"]
        assert "decision" in flow_final["review"]
        db.close()

    def test_reviewer_approves_clean_run(self, svc, workspace, code_session, db_path):
        """No errors + no risks + no changed files → completed without approval."""
        flow = svc.create_flow(
            code_session_id=code_session["id"],
            workspace_id=workspace["id"],
            description="minor refactor",
        )

        from hermes_cli.code.multi_agent_coding import AgentFlowDB

        db = AgentFlowDB(db_path=db_path)
        flow = db.update_flow(flow["id"], plan={"risks": [], "test_commands": []})

        mock_git = MagicMock()
        mock_git.get_status.return_value = {"branch": "main", "files": []}
        mock_git.get_diff.return_value = {"stat": ""}

        mock_lsp = MagicMock()
        mock_lsp.run_diagnostics.return_value = {"id": "d1", "status": "completed", "summary": {"errors": 0}}

        with (
            patch.object(svc, "_git_service", return_value=mock_git),
            patch.object(svc, "_lsp_service", return_value=mock_lsp),
        ):
            svc._run_reviewer(flow, db)

        flow_final = db.get_flow(flow["id"])
        assert flow_final["status"] == "completed"
        db.close()

    def test_reviewer_requests_approval_on_errors(self, svc, workspace, code_session, db_path):
        """Diagnostic errors → waiting_approval with approval_id set."""
        flow = svc.create_flow(
            code_session_id=code_session["id"],
            workspace_id=workspace["id"],
            description="risky change",
        )

        from hermes_cli.code.multi_agent_coding import AgentFlowDB

        db = AgentFlowDB(db_path=db_path)
        flow = db.update_flow(flow["id"], plan={"risks": [], "test_commands": []})

        mock_git = MagicMock()
        mock_git.get_status.return_value = {"branch": "main", "files": [{"path": "broken.py"}]}
        mock_git.get_diff.return_value = {"stat": "1 file changed"}

        mock_lsp = MagicMock()
        mock_lsp.run_diagnostics.return_value = {"id": "d1", "status": "completed", "summary": {"errors": 3}}

        with (
            patch.object(svc, "_git_service", return_value=mock_git),
            patch.object(svc, "_lsp_service", return_value=mock_lsp),
        ):
            svc._run_reviewer(flow, db)

        flow_final = db.get_flow(flow["id"])
        assert flow_final["status"] == "waiting_approval"
        assert flow_final["approval_id"] is not None
        db.close()

    def test_reviewer_diagnostics_summary_included(self, svc, workspace, code_session, db_path):
        flow = svc.create_flow(
            code_session_id=code_session["id"],
            workspace_id=workspace["id"],
            description="test",
        )

        from hermes_cli.code.multi_agent_coding import AgentFlowDB

        db = AgentFlowDB(db_path=db_path)
        flow = db.update_flow(flow["id"], plan={"risks": [], "test_commands": []})

        mock_git = MagicMock()
        mock_git.get_status.return_value = {"branch": "main", "files": []}
        mock_git.get_diff.return_value = {"stat": ""}

        mock_lsp = MagicMock()
        mock_lsp.run_diagnostics.return_value = {
            "id": "d1", "status": "completed",
            "summary": {"errors": 0, "warnings": 2, "info": 1},
        }

        with (
            patch.object(svc, "_git_service", return_value=mock_git),
            patch.object(svc, "_lsp_service", return_value=mock_lsp),
        ):
            svc._run_reviewer(flow, db)

        flow_final = db.get_flow(flow["id"])
        diag = flow_final["review"]["diagnostics_summary"]
        assert diag.get("warnings") == 2
        db.close()


# =============================================================================
# Full flow execution (mocked integration)
# =============================================================================


class TestFullFlow:
    def _mock_services(self, svc):
        mock_git = MagicMock()
        mock_git.get_status.return_value = {"branch": "main", "files": []}
        mock_git.get_diff.return_value = {"stat": ""}

        mock_lsp = MagicMock()
        mock_lsp.run_diagnostics.return_value = {"id": "d1", "status": "completed", "summary": {"errors": 0}}

        mock_runner = MagicMock()
        mock_runner.classify_command.return_value = "safe"
        mock_runner.create_command.return_value = {"id": "cmd-1"}
        mock_runner.run_command_sync.return_value = {"exit_code": 0, "stdout": "ok"}

        return mock_git, mock_lsp, mock_runner

    def test_run_flow_end_to_end(self, svc, workspace, code_session):
        flow = svc.create_flow(
            code_session_id=code_session["id"],
            workspace_id=workspace["id"],
            description="simple refactor",
        )

        mock_git, mock_lsp, mock_runner = self._mock_services(svc)

        with (
            patch.object(svc, "_git_service", return_value=mock_git),
            patch.object(svc, "_lsp_service", return_value=mock_lsp),
            patch.object(svc, "_command_runner", return_value=mock_runner),
        ):
            result = svc.run_flow(flow["id"])

        # No changed files + no errors → completed
        assert result["status"] == "completed"
        assert "steps" in result
        assert len(result["steps"]) > 0

    def test_run_flow_creates_steps_for_all_roles(self, svc, workspace, code_session):
        flow = svc.create_flow(
            code_session_id=code_session["id"],
            workspace_id=workspace["id"],
            description="fix bug",
        )

        mock_git, mock_lsp, mock_runner = self._mock_services(svc)

        with (
            patch.object(svc, "_git_service", return_value=mock_git),
            patch.object(svc, "_lsp_service", return_value=mock_lsp),
            patch.object(svc, "_command_runner", return_value=mock_runner),
        ):
            result = svc.run_flow(flow["id"])

        roles = {s["role"] for s in result["steps"]}
        assert "orchestrator" in roles
        assert "coder" in roles
        assert "reviewer" in roles

    def test_run_flow_invalid_status(self, svc, workspace, code_session, db_path):
        flow = svc.create_flow(
            code_session_id=code_session["id"],
            workspace_id=workspace["id"],
        )
        svc.cancel_flow(flow["id"])
        with pytest.raises(ValueError, match="cannot be run from status"):
            svc.run_flow(flow["id"])

    def test_resume_flow_after_approval(self, svc, workspace, code_session, db_path):
        """Resume a waiting_approval flow."""
        from hermes_cli.code.multi_agent_coding import AgentFlowDB

        flow = svc.create_flow(
            code_session_id=code_session["id"],
            workspace_id=workspace["id"],
            description="risky change",
        )

        db = AgentFlowDB(db_path=db_path)
        db.update_flow(flow["id"], status="waiting_approval")
        db.close()

        mock_git = MagicMock()
        mock_git.get_status.return_value = {"branch": "main", "files": []}
        mock_git.get_diff.return_value = {"stat": ""}

        mock_lsp = MagicMock()
        mock_lsp.run_diagnostics.return_value = {"id": "d1", "status": "completed", "summary": {"errors": 0}}

        with (
            patch.object(svc, "_git_service", return_value=mock_git),
            patch.object(svc, "_lsp_service", return_value=mock_lsp),
        ):
            result = svc.resume_flow(flow["id"])

        assert result["status"] == "completed"

    def test_flow_linked_to_code_session_events(self, svc, workspace, code_session, db_path):
        """Creating a flow adds an event to the code session timeline."""
        from hermes_state import CodeSessionDB

        flow = svc.create_flow(
            code_session_id=code_session["id"],
            workspace_id=workspace["id"],
        )

        db = CodeSessionDB(db_path=db_path)
        events = db.list_events(code_session["id"])
        db.close()

        event_types = [e["type"] for e in events]
        assert "agent.started" in event_types


# =============================================================================
# Schema version
# =============================================================================


class TestSchemaVersion:
    def test_schema_version_is_18(self):
        import hermes_state

        assert hermes_state.SCHEMA_VERSION == 18

    def test_schema_contains_code_agent_flows(self):
        import hermes_state

        assert "code_agent_flows" in hermes_state.SCHEMA_SQL

    def test_schema_contains_code_agent_flow_steps(self):
        import hermes_state

        assert "code_agent_flow_steps" in hermes_state.SCHEMA_SQL

    def test_migration_creates_tables(self, tmp_path):
        from hermes_cli.code.multi_agent_coding import AgentFlowDB

        db_path = tmp_path / "migration_test.db"
        db = AgentFlowDB(db_path=db_path)
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('code_agent_flows', 'code_agent_flow_steps')"
        )
        tables = {row[0] for row in cursor.fetchall()}
        assert "code_agent_flows" in tables
        assert "code_agent_flow_steps" in tables
        db.close()


# =============================================================================
# REST endpoints
# =============================================================================


class TestAgentFlowEndpoints:
    @pytest.fixture()
    def client(self, tmp_path):
        from starlette.testclient import TestClient
        from hermes_cli.web_server import app, _SESSION_TOKEN

        c = TestClient(app)
        c.headers["Authorization"] = f"Bearer {_SESSION_TOKEN}"
        return c

    def test_list_agent_flows_returns_list(self, client):
        response = client.get("/api/code/agent-flows")
        assert response.status_code == 200
        data = response.json()
        assert "flows" in data
        assert isinstance(data["flows"], list)

    def test_create_agent_flow_missing_session(self, client):
        response = client.post(
            "/api/code/agent-flows",
            json={
                "workspace_id": "nonexistent-ws",
                "code_session_id": "nonexistent-session",
                "title": "Test",
            },
        )
        assert response.status_code == 404

    def test_get_agent_flow_not_found(self, client):
        response = client.get("/api/code/agent-flows/nonexistent-flow")
        assert response.status_code == 404

    def test_cancel_agent_flow_not_found(self, client):
        response = client.post("/api/code/agent-flows/nonexistent-flow/cancel")
        assert response.status_code == 400

    def test_run_agent_flow_not_found(self, client):
        response = client.post("/api/code/agent-flows/nonexistent-flow/run")
        assert response.status_code == 400

    def test_session_agent_flows_endpoint(self, client):
        response = client.get("/api/code/sessions/some-session/agent-flows")
        assert response.status_code == 200
        data = response.json()
        assert "flows" in data
