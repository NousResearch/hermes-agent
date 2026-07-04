"""Tests for the Context Governor."""

import sys
import tempfile
import shutil
import os
import importlib
from pathlib import Path

sys.path.insert(0, '/home/orchestrator/.hermes/hermes-agent')

from agent.context_governor import (
    ContextGovernor,
    TaskStateLedger,
    ToolOutputReducer,
    get_context_governor,
    reset_context_governor,
)
from hermes_state import SessionDB
from hermes_constants import get_hermes_home


class TestContextGovernor:
    def _temp_home(self):
        home = Path(tempfile.mkdtemp())
        os.environ['HERMES_HOME'] = str(home)
        importlib.reload(sys.modules['hermes_constants'])
        return home

    def test_default_config(self):
        reset_context_governor()
        g = get_context_governor()
        assert g.raw_tool_window == 5
        assert g.summary_window == 3
        assert g.max_state_summary_words == 700
        assert g.max_raw_log_lines == 200
        assert g.verification_required is True

    def test_ledger_initial_state(self):
        g = ContextGovernor()
        assert g.ledger.repo == ""
        assert g.ledger.objective == ""
        assert g.ledger.completed_actions == []

    def test_update_ledger(self):
        g = ContextGovernor()
        g.update_ledger(
            repo="marcusgoll/cfipros",
            objective="Set up admin email notifications",
            current_branch="feature/admin-email",
            completed_actions=["Inspected email transport code"],
        )
        assert g.ledger.repo == "marcusgoll/cfipros"
        assert g.ledger.objective == "Set up admin email notifications"
        assert g.ledger.current_branch == "feature/admin-email"
        assert "Inspected email transport code" in g.ledger.completed_actions

    def test_record_tool_call_pruning(self):
        g = ContextGovernor(raw_tool_window=3, summary_window=2)
        for i in range(5):
            g.record_tool_call(
                tool_name="read_file",
                tool_args={"path": f"/tmp/file{i}.txt"},
                tool_result=f"content {i}",
                turn_index=i,
            )
        # Raw window should keep last 3
        assert len(g.raw_tool_calls) == 3
        assert g.raw_tool_calls[0].tool_args["path"] == "/tmp/file2.txt"
        assert g.raw_tool_calls[-1].tool_args["path"] == "/tmp/file4.txt"
        # Summarized window compacts adjacent summaries with the same objective
        assert len(g.summarized_window) == 1
        assert "/tmp/file0.txt" in g.summarized_window[0].files_or_resources_touched
        assert "/tmp/file1.txt" in g.summarized_window[0].files_or_resources_touched

    def test_context_for_model_includes_ledger(self):
        g = ContextGovernor()
        g.update_ledger(
            repo="marcusgoll/cfipros",
            objective="Set up admin email notifications",
            completed_actions=["Inspected email transport code"],
        )
        ctx = g.get_context_for_model()
        assert "marcusgoll/cfipros" in ctx
        assert "Set up admin email notifications" in ctx
        assert "Inspected email transport code" in ctx

    def test_session_lifecycle(self):
        g = ContextGovernor()
        g.on_session_start("session-123")
        assert g._session_id == "session-123"
        g.record_tool_call("terminal", {"command": "echo hi"}, "hi", 0)
        g.on_session_end("session-123", [])
        assert len(g.raw_tool_calls) == 0
        assert len(g.summarized_window) == 0
        g.on_session_reset()
        assert g._session_id is None
        assert g.ledger.objective == ""

    def test_reset_global_governor(self):
        reset_context_governor()
        g1 = get_context_governor()
        g1.update_ledger(objective="task 1")
        reset_context_governor()
        g2 = get_context_governor()
        assert g1 is not g2
        assert g2.ledger.objective == ""

    def test_verifier_unknown_task_type(self):
        g = ContextGovernor()
        result = g.verify_completion("unknown_task")
        assert result["verified"] is False
        assert "Unknown task type" in result["error"]

    def test_persistence_across_new_process(self):
        """Context Governor ledger + raw tool window should survive a new process."""
        home = self._temp_home()
        try:
            reset_context_governor()
            g = get_context_governor()

            db = SessionDB(get_hermes_home() / "sessions.db")
            db.create_session("persist-test", "cli", model="test-model")
            db.close()

            g.on_session_start("persist-test")
            g.update_ledger(
                repo="marcusgoll/cfipros",
                objective="Set up admin email notifications",
                current_branch="feature/admin-email",
                completed_actions=["Inspected email transport code"],
            )
            for i in range(10):
                g.record_tool_call("read_file", {"path": f"/tmp/file{i}.txt"}, f"content {i}", i)

            g.on_session_end("persist-test", [])

            # Simulate new process: reload module, reset singleton, then load
            import agent.context_governor as cg_module
            importlib.reload(cg_module)
            from agent.context_governor import get_context_governor as get_gov2

            g2 = get_gov2()
            g2.on_session_start("persist-test")
            assert g2.ledger.objective == "Set up admin email notifications"
            assert g2.ledger.repo == "marcusgoll/cfipros"
            assert g2.ledger.current_branch == "feature/admin-email"
            assert len(g2.raw_tool_calls) == 5
            # Summaries are compacted by objective, so only one remains
            assert len(g2.summarized_window) == 1
            assert "/tmp/file4.txt" in g2.summarized_window[0].files_or_resources_touched
            assert g2.raw_tool_calls[0].tool_name == "read_file"
        finally:
            shutil.rmtree(home, ignore_errors=True)
            # Reset environment
            if 'HERMES_HOME' in os.environ:
                del os.environ['HERMES_HOME']

    def test_new_session_after_reset(self):
        """After /new, a new session should not inherit old ledger state."""
        home = self._temp_home()
        try:
            reset_context_governor()
            g = get_context_governor()

            db = SessionDB(get_hermes_home() / "sessions.db")
            db.create_session("old-session", "cli", model="test-model")
            db.close()

            g.on_session_start("old-session")
            g.update_ledger(objective="Old objective")
            g.on_session_end("old-session", [])

            reset_context_governor()
            g2 = get_context_governor()
            db = SessionDB(get_hermes_home() / "sessions.db")
            db.create_session("new-session", "cli", model="test-model")
            db.close()

            g2.on_session_start("new-session")
            assert g2.ledger.objective == ""
            assert len(g2.raw_tool_calls) == 0
            assert len(g2.summarized_window) == 0
        finally:
            shutil.rmtree(home, ignore_errors=True)
            if 'HERMES_HOME' in os.environ:
                del os.environ['HERMES_HOME']
