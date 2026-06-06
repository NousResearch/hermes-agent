#!/usr/bin/env python3
"""
Tests for executors/bridge_cli.py — fixture-based acceptance CLI + D1a-local
review/qa stubs.

Scope:
  - _stub_review_report: produces ReviewReport with correct shape; detects
    hardcoded-secret / SQL-injection / no-tests / no-issues heuristics
  - _stub_qa_report: produces QAReport with correct shape
  - _fixture_happy_path / _fixture_failed_path: produce a non-empty list of
    RunEvents each
  - _run_fixture: routes fixture through RunBridge and returns a RunResult
  - cmd_accept: prints pass/fail for happy-path and failed scenarios
  - cmd_logs: prints logs and tool calls
  - cmd_changed_files: prints changed files
  - cmd_diff: prints diff
  - cmd_ipc: trigger-review, trigger-qa, continue, retry, unknown action
  - handle_bridge_command: dispatches all subcommands
  - Guard: importing bridge_cli does NOT pull in review_handler, review_cli,
    or cli (D1a boundary protection)
  - Guard: no subprocess / git / DB calls anywhere in bridge_cli

Strictly no subprocess, no real files, no DB, no model, no real project writes.
Uses tmp_path and capsys only.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from executors import bridge_cli
from executors.bridge_cli import (
    _fixture_failed_path,
    _fixture_happy_path,
    _run_fixture,
    _stub_qa_report,
    _stub_review_report,
    cmd_accept,
    cmd_changed_files,
    cmd_diff,
    cmd_ipc,
    cmd_logs,
    handle_bridge_command,
)
from executors.types import (
    FindingCategory,
    QAReport,
    QAStatus,
    ReviewReport,
    ReviewStatus,
    RunEvent,
    RunEventType,
    RunStatus,
    Severity,
)


# ---------------------------------------------------------------------------
# Async helper
# ---------------------------------------------------------------------------


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# 1. _stub_review_report
# ---------------------------------------------------------------------------


class TestStubReviewReport:
    def test_basic_shape(self) -> None:
        report = _stub_review_report("main-1", "")
        assert isinstance(report, ReviewReport)
        assert report.review_run_id == "review-main-1-stub"
        assert report.executor == "stub"
        assert report.total_findings >= 1
        # Empty diff should produce at least an info finding
        assert any(
            f.severity == Severity.INFO for f in report.findings
        )

    def test_detects_hardcoded_secret(self) -> None:
        report = _stub_review_report("m", "password = 'hunter2'")
        secret_findings = [
            f for f in report.findings if "secret" in f.title.lower()
        ]
        assert len(secret_findings) == 1
        assert secret_findings[0].severity == Severity.HIGH
        assert secret_findings[0].category == FindingCategory.SECURITY

    def test_detects_sql_injection(self) -> None:
        report = _stub_review_report("m", "SELECT * FROM users")
        sqli = [f for f in report.findings if "SQL" in f.title]
        assert len(sqli) == 1
        assert sqli[0].severity == Severity.CRITICAL

    def test_detects_fstring_injection(self) -> None:
        report = _stub_review_report("m", 'query = f"SELECT * FROM {table}"')
        sqli = [f for f in report.findings if "SQL" in f.title]
        assert len(sqli) == 1
        assert sqli[0].severity == Severity.CRITICAL

    def test_detects_missing_tests(self) -> None:
        # Two source files changed, no test file → missing-tests finding.
        patch = (
            "diff --git a/a.py b/a.py\n"
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "+x = 1\n"
            "diff --git a/b.py b/b.py\n"
            "--- a/b.py\n"
            "+++ b/b.py\n"
            "+y = 2\n"
        )
        report = _stub_review_report("m", patch)
        no_test_findings = [
            f for f in report.findings
            if "test" in f.title.lower() and "coverage" in f.title.lower()
        ]
        assert len(no_test_findings) == 1
        assert no_test_findings[0].category == FindingCategory.TEST_COVERAGE

    def test_no_test_finding_when_tests_present(self) -> None:
        patch = (
            "diff --git a/a.py b/a.py\n"
            "diff --git a/test_a.py b/test_a.py\n"
        )
        report = _stub_review_report("m", patch)
        no_test_findings = [
            f for f in report.findings
            if "test" in f.title.lower() and "coverage" in f.title.lower()
        ]
        assert no_test_findings == []

    def test_severity_counts_match_findings(self) -> None:
        report = _stub_review_report("m", "password = 'x'")  # 1 HIGH finding
        assert report.high_count == 1
        assert report.total_findings == len(report.findings)

    def test_status_completed(self) -> None:
        report = _stub_review_report("m", "anything")
        assert report.status == ReviewStatus.COMPLETED

    def test_findings_have_required_ids(self) -> None:
        report = _stub_review_report("m", "password = 'x'")
        for f in report.findings:
            assert f.id
            assert f.run_id == "review-m-stub"
            assert f.title


# ---------------------------------------------------------------------------
# 2. _stub_qa_report
# ---------------------------------------------------------------------------


class TestStubQAReport:
    def test_basic_shape(self) -> None:
        report = _stub_qa_report("main-1", ["a.py", "b.py", "c.py"])
        assert isinstance(report, QAReport)
        assert report.qa_run_id == "qa-main-1-stub"
        assert report.executor == "stub"
        assert report.test_passed == 3
        assert report.test_failed == 0
        assert report.test_skipped == 0
        assert "3" in report.test_output
        assert report.risks == []

    def test_empty_changed_files(self) -> None:
        report = _stub_qa_report("m", [])
        assert report.test_passed == 0
        assert "0" in report.test_output

    def test_status_completed(self) -> None:
        report = _stub_qa_report("m", ["a.py"])
        assert report.status == QAStatus.COMPLETED


# ---------------------------------------------------------------------------
# 3. Fixture event generators
# ---------------------------------------------------------------------------


class TestFixtures:
    def test_happy_path_non_empty(self) -> None:
        events = _fixture_happy_path()
        assert isinstance(events, list)
        assert len(events) > 5
        # Must end with a COMPLETED event
        assert events[-1].type == RunEventType.COMPLETED

    def test_happy_path_contains_diff(self) -> None:
        events = _fixture_happy_path()
        diffs = [e for e in events if e.type == RunEventType.DIFF]
        assert len(diffs) >= 1
        assert "diff --git" in diffs[0].payload["patch"]

    def test_failed_path_non_empty(self) -> None:
        events = _fixture_failed_path()
        assert isinstance(events, list)
        assert len(events) >= 3
        # Must end with a FAILED event
        assert events[-1].type == RunEventType.FAILED
        assert events[-1].payload.get("error_summary")


# ---------------------------------------------------------------------------
# 4. _run_fixture
# ---------------------------------------------------------------------------


class TestRunFixture:
    def test_happy_path_yields_completed_result(self) -> None:
        result = _run_fixture("happy-path")
        assert isinstance(result, type(_run_fixture("happy-path")))
        assert result.status == RunStatus.COMPLETED
        assert len(result.changed_files) >= 1

    def test_failed_yields_failed_result(self) -> None:
        result = _run_fixture("failed")
        assert result.status == RunStatus.FAILED
        assert result.error_summary

    def test_unknown_scenario_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            _run_fixture("does-not-exist")
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# 5. cmd_accept
# ---------------------------------------------------------------------------


class TestCmdAccept:
    def test_happy_path_passes(self, capsys) -> None:
        _run(cmd_accept("happy-path"))
        captured = capsys.readouterr()
        # Should print check marks for each check
        assert "passed" in captured.out
        assert "Acceptance:" in captured.out

    def test_failed_passes(self, capsys) -> None:
        _run(cmd_accept("failed"))
        captured = capsys.readouterr()
        assert "Acceptance:" in captured.out

    def test_unknown_scenario_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            _run(cmd_accept("no-such-scenario"))
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# 6. cmd_logs / cmd_changed_files / cmd_diff
# ---------------------------------------------------------------------------


class TestCmdLogs:
    def test_prints_logs_header(self, capsys) -> None:
        _run(cmd_logs("happy-path"))
        captured = capsys.readouterr()
        assert "Logs" in captured.out
        assert "Tool Calls" in captured.out


class TestCmdChangedFiles:
    def test_prints_changed_files(self, capsys) -> None:
        _run(cmd_changed_files("happy-path"))
        captured = capsys.readouterr()
        assert "Changed Files" in captured.out

    def test_empty_for_failed(self, capsys) -> None:
        _run(cmd_changed_files("failed"))
        captured = capsys.readouterr()
        assert "No files changed" in captured.out


class TestCmdDiff:
    def test_prints_diff(self, capsys) -> None:
        _run(cmd_diff("happy-path"))
        captured = capsys.readouterr()
        assert "Diff" in captured.out
        assert "diff --git" in captured.out

    def test_empty_for_failed(self, capsys) -> None:
        _run(cmd_diff("failed"))
        captured = capsys.readouterr()
        assert "No diff available" in captured.out


# ---------------------------------------------------------------------------
# 7. cmd_ipc
# ---------------------------------------------------------------------------


class TestCmdIPC:
    def test_trigger_review_json_shape(self, capsys) -> None:
        _run(cmd_ipc("trigger-review"))
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["action"] == "review:trigger"
        assert data["run_id"].startswith("review-run-main-001-stub")
        assert data["status"] == "completed"
        assert data["executor"] == "stub"
        assert isinstance(data["total_findings"], int)
        assert isinstance(data["findings"], list)
        for f in data["findings"]:
            assert "severity" in f
            assert "category" in f
            assert "title" in f

    def test_trigger_qa_json_shape(self, capsys) -> None:
        _run(cmd_ipc("trigger-qa"))
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["action"] == "qa:trigger"
        assert data["run_id"].startswith("qa-run-main-001-stub")
        assert data["status"] == "completed"
        assert data["executor"] == "stub"
        assert data["test_passed"] >= 0
        assert "test_failed" in data
        assert "test_skipped" in data

    def test_continue_json_shape(self, capsys) -> None:
        _run(cmd_ipc("continue", thread_id="t-99"))
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["action"] == "run:continue"
        assert data["request"]["thread_id"] == "t-99"
        assert data["response"]["run_id"].startswith("run-continue-t-99")

    def test_retry_json_shape(self, capsys) -> None:
        _run(cmd_ipc("retry", thread_id="t-42"))
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["action"] == "run:retry"
        assert data["request"]["thread_id"] == "t-42"
        assert data["response"]["run_id"].startswith("run-retry-t-42")
        assert data["response"]["run_seq"] == 2

    def test_unknown_action_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            _run(cmd_ipc("not-a-real-action"))
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# 8. handle_bridge_command dispatcher
# ---------------------------------------------------------------------------


def _args(**kwargs) -> argparse.Namespace:
    """Build a minimal Namespace for handle_bridge_command tests."""
    base = {
        "bridge_subcommand": None,
        "scenario": "happy-path",
        "fixture": "happy-path",
        "ipc_action": "continue",
        "main_run_id": "run-1",
        "thread_id": "thread-1",
    }
    base.update(kwargs)
    return argparse.Namespace(**base)


class TestHandleBridgeCommand:
    def test_no_subcommand_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            _run(handle_bridge_command(_args()))
        assert exc_info.value.code == 1

    def test_accept_dispatches(self, capsys) -> None:
        _run(handle_bridge_command(_args(bridge_subcommand="accept", scenario="happy-path")))
        assert "Acceptance:" in capsys.readouterr().out

    def test_logs_dispatches(self, capsys) -> None:
        _run(handle_bridge_command(_args(bridge_subcommand="logs", fixture="happy-path")))
        assert "Logs" in capsys.readouterr().out

    def test_changed_files_dispatches(self, capsys) -> None:
        _run(handle_bridge_command(_args(bridge_subcommand="changed-files", fixture="happy-path")))
        assert "Changed Files" in capsys.readouterr().out

    def test_diff_dispatches(self, capsys) -> None:
        _run(handle_bridge_command(_args(bridge_subcommand="diff", fixture="happy-path")))
        assert "Diff" in capsys.readouterr().out

    def test_ipc_dispatches(self, capsys) -> None:
        _run(handle_bridge_command(_args(bridge_subcommand="ipc", ipc_action="continue", thread_id="t")))
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["action"] == "run:continue"

    def test_unknown_subcommand_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            _run(handle_bridge_command(_args(bridge_subcommand="not-real")))
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# 9. Boundary guards — bridge_cli must NOT pull in forbidden modules
# ---------------------------------------------------------------------------


class TestBoundaryGuards:
    def test_importing_bridge_cli_does_not_pull_review_handler(self) -> None:
        # Clear any cached version, then import fresh and inspect sys.modules.
        for mod in list(sys.modules):
            if mod == "executors.review_handler" or mod.startswith("executors.review_handler."):
                del sys.modules[mod]
        importlib.reload(bridge_cli)
        assert "executors.review_handler" not in sys.modules
        # And no submodule under review_handler was pulled in either
        review_handler_subs = [
            m for m in sys.modules
            if m == "executors.review_handler"
            or m.startswith("executors.review_handler.")
        ]
        assert review_handler_subs == []

    def test_importing_bridge_cli_does_not_pull_review_cli(self) -> None:
        for mod in list(sys.modules):
            if mod == "executors.review_cli" or mod.startswith("executors.review_cli."):
                del sys.modules[mod]
        importlib.reload(bridge_cli)
        assert "executors.review_cli" not in sys.modules

    def test_importing_bridge_cli_does_not_pull_cli(self) -> None:
        # `executors.cli` is the top-level CLI; this is the integration commit
        # we explicitly defer. bridge_cli must not import it.
        for mod in list(sys.modules):
            if mod == "executors.cli" or mod.startswith("executors.cli."):
                del sys.modules[mod]
        importlib.reload(bridge_cli)
        assert "executors.cli" not in sys.modules

    def test_no_subprocess_in_bridge_cli(self) -> None:
        # bridge_cli must not import subprocess
        src = Path(bridge_cli.__file__).read_text()
        assert "subprocess" not in src, "bridge_cli.py must not import subprocess"

    def test_no_git_invocations_in_bridge_cli(self) -> None:
        # No calls to git, no "git " patterns in code.
        src = Path(bridge_cli.__file__).read_text()
        # 'subprocess.run' is the canonical executor boundary; assert not present
        assert "subprocess.run" not in src
        assert "subprocess.Popen" not in src
        # Inline checks for "git " invocations in any string literal
        assert "git diff" not in src
        assert "git rev-parse" not in src

    def test_no_db_in_bridge_cli(self) -> None:
        src = Path(bridge_cli.__file__).read_text()
        assert "sqlite3" not in src
        assert "kanban_db" not in src
        assert "task_events" not in src

    def test_no_model_invocation(self) -> None:
        src = Path(bridge_cli.__file__).read_text()
        # No direct call to any model binary
        assert "opencode" not in src  # would spawn subprocess; forbidden in D1a
        assert "claude" not in src.lower() or "claude_code" not in src  # no adapter invocation
        # Note: docstring / comments may mention "claude-code" as a string
        # (e.g. _stub_review_report has no model invocation), so we only
        # check for known binary names that would imply subprocess.
