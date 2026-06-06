#!/usr/bin/env python3
"""
Tests for executors/bridge.py — RunBridge aggregation + result_to_ipc_events.

Scope:
  - RunBridge.ingest: each RunEventType routes to the right internal handler
  - RunResult helpers: full_message, has_changes, has_diff, to_summary
  - _parse_diff_patch: "added" / "modified" / "deleted" status detection,
    multiple file blocks
  - _extract_files_from_content: heuristic extraction from tool output
  - Tool-call / tool-result pairing by name
  - result_to_ipc_events: produces tool.log, diff, changed_file, run.completed /
    run.failed events with correct payload shapes
  - events_by_type counting
  - Unknown event type is logged but does not crash

Strictly no subprocess, no git, no real files, no model, no DB.
"""
from __future__ import annotations

import datetime
from typing import List

import pytest

from executors.bridge import (
    ChangedFile,
    IPCChangedFile,
    IPCEvent,
    RunBridge,
    RunResult,
    result_to_ipc_events,
)
from executors.types import (
    RunEvent,
    RunEventType,
    RunStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ev(t: RunEventType, payload: dict, seq: int = 0) -> RunEvent:
    return RunEvent(type=t, payload=payload, seq=seq)


# ---------------------------------------------------------------------------
# 1. RunBridge.ingest — basic routing
# ---------------------------------------------------------------------------


class TestIngestRouting:
    def test_message_event_appends_to_parts(self) -> None:
        b = RunBridge()
        b.ingest(_ev(RunEventType.MESSAGE, {"content": "hello"}))
        b.ingest(_ev(RunEventType.MESSAGE, {"content": " world"}))
        result = b.finalize()
        assert result.message_parts == ["hello", " world"]
        assert result.full_message() == "hello world"

    def test_message_with_empty_content_is_skipped(self) -> None:
        b = RunBridge()
        b.ingest(_ev(RunEventType.MESSAGE, {"content": ""}))
        b.ingest(_ev(RunEventType.MESSAGE, {}))
        assert b.finalize().message_parts == []

    def test_reasoning_event(self) -> None:
        b = RunBridge()
        b.ingest(_ev(RunEventType.REASONING, {"content": "thinking..."}))
        assert b.finalize().reasoning_blocks == ["thinking..."]

    def test_log_event(self) -> None:
        b = RunBridge()
        b.ingest(_ev(RunEventType.LOG, {"tool": "bash", "message": "ok", "level": "info"}))
        result = b.finalize()
        assert len(result.logs) == 1
        assert result.logs[0].tool == "bash"
        assert result.logs[0].level == "info"

    def test_completed_event_sets_status(self) -> None:
        b = RunBridge()
        b.ingest(_ev(RunEventType.COMPLETED, {}))
        assert b.finalize().status == RunStatus.COMPLETED

    def test_failed_event_sets_status_and_error(self) -> None:
        b = RunBridge()
        b.ingest(_ev(RunEventType.FAILED, {"error_summary": "boom"}))
        result = b.finalize()
        assert result.status == RunStatus.FAILED
        assert result.error_summary == "boom"

    def test_unknown_event_type_is_logged_not_crashed(self) -> None:
        # Patch the dispatch so it sees an unrecognized type.value, then
        # verify RunBridge logs a warn entry instead of crashing.
        b = RunBridge()
        ev = RunEvent(type=RunEventType.MESSAGE, payload={"content": "x"})
        # Bypass the enum's __setattr__ guard so we can set type.value
        # to a name that doesn't match any _handle_* method.
        original_value = ev.type.value
        try:
            object.__setattr__(ev.type, "_value_", "totally_unknown")
            b.ingest(ev)
        finally:
            object.__setattr__(ev.type, "_value_", original_value)
        result = b.finalize()
        # Should have logged a warn-level entry
        assert any("Unknown event" in log.message for log in result.logs)

    def test_total_events_and_events_by_type(self) -> None:
        b = RunBridge()
        b.ingest(_ev(RunEventType.MESSAGE, {"content": "a"}, seq=0))
        b.ingest(_ev(RunEventType.MESSAGE, {"content": "b"}, seq=1))
        b.ingest(_ev(RunEventType.COMPLETED, {}, seq=2))
        result = b.finalize()
        assert result.total_events == 3
        assert result.events_by_type.get("message") == 2
        assert result.events_by_type.get("completed") == 1


# ---------------------------------------------------------------------------
# 2. Tool call / tool result pairing
# ---------------------------------------------------------------------------


class TestToolPairing:
    def test_pending_tool_call_paired_with_result(self) -> None:
        b = RunBridge()
        b.ingest(_ev(RunEventType.TOOL_CALL, {"tool_name": "read_file", "arguments": {"path": "a.py"}}, seq=1))
        b.ingest(_ev(RunEventType.TOOL_RESULT, {
            "tool_name": "read_file",
            "tool_call_id": "tc-1",
            "content": "file contents",
            "stdout": "out",
            "duration": 0.5,
        }, seq=2))
        result = b.finalize()
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.tool_name == "read_file"
        assert tc.stdout == "out"
        assert tc.duration == 0.5
        # The pending tool should be drained
        assert b._pending_tools == {}

    def test_tool_result_without_pending_call_still_recorded(self) -> None:
        b = RunBridge()
        b.ingest(_ev(RunEventType.TOOL_RESULT, {
            "tool_name": "read_file",
            "content": "data",
        }, seq=1))
        result = b.finalize()
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "read_file"

    def test_error_flag_propagates_to_tool_call(self) -> None:
        b = RunBridge()
        b.ingest(_ev(RunEventType.TOOL_CALL, {"tool_name": "bash"}, seq=1))
        b.ingest(_ev(RunEventType.TOOL_RESULT, {
            "tool_name": "bash",
            "content": "fail",
            "is_error": True,
            "stderr": "boom",
        }, seq=2))
        result = b.finalize()
        assert result.tool_calls[0].error is True
        # Logs should include an error entry
        assert any(log.level == "error" for log in result.logs)

    def test_stdout_lines_become_log_entries(self) -> None:
        b = RunBridge()
        b.ingest(_ev(RunEventType.TOOL_CALL, {"tool_name": "bash"}, seq=1))
        b.ingest(_ev(RunEventType.TOOL_RESULT, {
            "tool_name": "bash",
            "stdout": "line one\nline two",
        }, seq=2))
        result = b.finalize()
        log_msgs = [log.message for log in result.logs if log.tool == "bash"]
        assert "line one" in log_msgs
        assert "line two" in log_msgs


# ---------------------------------------------------------------------------
# 3. Diff parsing
# ---------------------------------------------------------------------------


class TestParseDiffPatch:
    def test_added_file(self) -> None:
        b = RunBridge()
        patch = (
            "diff --git a/new.py b/new.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/new.py\n"
            "@@ -0,0 +1,3 @@\n"
            "+a = 1\n"
            "+b = 2\n"
            "+c = 3\n"
        )
        b.ingest(_ev(RunEventType.DIFF, {"patch": patch}))
        result = b.finalize()
        assert len(result.changed_files) == 1
        cf = result.changed_files[0]
        assert cf.path == "new.py"
        assert cf.status == "added"
        assert cf.additions == 3
        assert cf.deletions == 0
        assert len(result.diff_patches) == 1

    def test_modified_file(self) -> None:
        b = RunBridge()
        patch = (
            "diff --git a/x.py b/x.py\n"
            "--- a/x.py\n"
            "+++ b/x.py\n"
            "@@ -1,3 +1,4 @@\n"
            " a\n"
            "-b\n"
            "+b2\n"
            "+c\n"
        )
        b.ingest(_ev(RunEventType.DIFF, {"patch": patch}))
        result = b.finalize()
        cf = result.changed_files[0]
        assert cf.path == "x.py"
        assert cf.status == "modified"
        assert cf.additions == 2
        assert cf.deletions == 1

    def test_multiple_files_in_one_patch(self) -> None:
        b = RunBridge()
        patch = (
            "diff --git a/a.py b/a.py\n"
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ -1 +1 @@\n"
            "-x\n"
            "+y\n"
            "diff --git a/b.py b/b.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/b.py\n"
            "@@ -0,0 +1,2 @@\n"
            "+p\n"
            "+q\n"
        )
        b.ingest(_ev(RunEventType.DIFF, {"patch": patch}))
        result = b.finalize()
        assert len(result.changed_files) == 2
        paths = {cf.path for cf in result.changed_files}
        assert paths == {"a.py", "b.py"}
        statuses = {cf.path: cf.status for cf in result.changed_files}
        assert statuses["a.py"] == "modified"
        assert statuses["b.py"] == "added"

    def test_empty_diff_patch_no_files(self) -> None:
        b = RunBridge()
        b.ingest(_ev(RunEventType.DIFF, {"patch": ""}))
        result = b.finalize()
        assert result.changed_files == []
        assert result.diff_patches == []


# ---------------------------------------------------------------------------
# 4. Heuristic file extraction from tool content
# ---------------------------------------------------------------------------


class TestExtractFilesFromContent:
    def test_wrote_to_pattern(self) -> None:
        b = RunBridge()
        b.ingest(_ev(RunEventType.TOOL_RESULT, {
            "tool_name": "write_file",
            "content": "Wrote to /repo/services/auth.py successfully",
        }))
        result = b.finalize()
        assert any(cf.path == "/repo/services/auth.py" for cf in result.changed_files)

    def test_created_file_pattern(self) -> None:
        b = RunBridge()
        b.ingest(_ev(RunEventType.TOOL_RESULT, {
            "tool_name": "write_file",
            "content": "Created /repo/utils/helpers.ts",
        }))
        result = b.finalize()
        assert any(cf.path == "/repo/utils/helpers.ts" for cf in result.changed_files)

    def test_no_match_no_files(self) -> None:
        b = RunBridge()
        b.ingest(_ev(RunEventType.TOOL_RESULT, {
            "tool_name": "write_file",
            "content": "just some prose with no file references",
        }))
        result = b.finalize()
        # Should not crash, no files extracted
        assert result.changed_files == []

    def test_no_duplicate_files(self) -> None:
        b = RunBridge()
        b.ingest(_ev(RunEventType.TOOL_RESULT, {
            "tool_name": "write_file",
            "content": "Wrote to /a/b.py\nWrote to /a/b.py",
        }))
        result = b.finalize()
        matching = [cf for cf in result.changed_files if cf.path == "/a/b.py"]
        assert len(matching) == 1


# ---------------------------------------------------------------------------
# 5. RunResult helpers
# ---------------------------------------------------------------------------


class TestRunResultHelpers:
    def test_full_message_concatenates(self) -> None:
        r = RunResult(run_id="r", message_parts=["a", "b", "c"])
        assert r.full_message() == "abc"

    def test_has_changes(self) -> None:
        r1 = RunResult(run_id="r")
        r2 = RunResult(run_id="r", changed_files=[ChangedFile(path="x", status="added")])
        assert not r1.has_changes()
        assert r2.has_changes()

    def test_has_diff(self) -> None:
        r1 = RunResult(run_id="r")
        r2 = RunResult(run_id="r", diff_patches=["patch"])
        assert not r1.has_diff()
        assert r2.has_diff()

    def test_to_summary_running(self) -> None:
        r = RunResult(
            run_id="r",
            status=RunStatus.RUNNING,
            message_parts=["a", "b"],
            tool_calls=[{"x": 1}],  # type: ignore[list-item]
        )
        # The summary only references len(...) so the tool_calls can be anything.
        s = r.to_summary()
        assert "Running" in s
        assert "2 message parts" in s

    def test_to_summary_completed_no_changes(self) -> None:
        r = RunResult(
            run_id="r",
            status=RunStatus.COMPLETED,
            total_events=5,
        )
        s = r.to_summary()
        assert "Completed" in s
        assert "5 events" in s
        assert "files changed" not in s

    def test_to_summary_completed_with_changes(self) -> None:
        r = RunResult(
            run_id="r",
            status=RunStatus.COMPLETED,
            total_events=5,
            changed_files=[ChangedFile(path="a", status="modified")],
        )
        s = r.to_summary()
        assert "1 files changed" in s

    def test_to_summary_failed(self) -> None:
        r = RunResult(run_id="r", status=RunStatus.FAILED, error_summary="x")
        s = r.to_summary()
        assert "Failed" in s
        assert "x" in s


# ---------------------------------------------------------------------------
# 6. result_to_ipc_events
# ---------------------------------------------------------------------------


class TestResultToIPCEvents:
    def test_completed_run_produces_status_event(self) -> None:
        result = RunResult(run_id="r-1", status=RunStatus.COMPLETED)
        events = result_to_ipc_events(result)
        status_events = [e for e in events if e.event in ("run.completed", "run.failed")]
        assert len(status_events) == 1
        assert status_events[0].event == "run.completed"
        assert status_events[0].payload["status"] == "completed"

    def test_failed_run_produces_failed_status(self) -> None:
        result = RunResult(
            run_id="r-1", status=RunStatus.FAILED, error_summary="boom"
        )
        events = result_to_ipc_events(result)
        status_events = [e for e in events if e.event in ("run.completed", "run.failed")]
        assert len(status_events) == 1
        assert status_events[0].event == "run.failed"
        assert status_events[0].payload["error_summary"] == "boom"

    def test_logs_become_tool_log_events(self) -> None:
        from executors.bridge import _LogEntry  # type: ignore[attr-defined]
        result = RunResult(
            run_id="r",
            logs=[_LogEntry(tool="bash", message="hello", level="info")],  # type: ignore[list-item]
        )
        events = result_to_ipc_events(result)
        tool_logs = [e for e in events if e.event == "tool.log"]
        assert len(tool_logs) == 1
        assert tool_logs[0].payload["tool"] == "bash"
        assert tool_logs[0].payload["message"] == "hello"

    def test_diff_patches_become_diff_events(self) -> None:
        result = RunResult(run_id="r", diff_patches=["patch-1", "patch-2"])
        events = result_to_ipc_events(result)
        diff_events = [e for e in events if e.event == "diff"]
        assert len(diff_events) == 2
        assert diff_events[0].payload["patch"] == "patch-1"

    def test_changed_files_become_changed_file_events(self) -> None:
        result = RunResult(
            run_id="r",
            changed_files=[ChangedFile(path="a.py", status="added", additions=5, deletions=0)],
        )
        events = result_to_ipc_events(result)
        cf_events = [e for e in events if e.event == "changed_file"]
        assert len(cf_events) == 1
        assert cf_events[0].payload["path"] == "a.py"
        assert cf_events[0].payload["additions"] == 5

    def test_all_events_have_run_id_and_timestamp(self) -> None:
        result = RunResult(run_id="r-9", status=RunStatus.COMPLETED)
        events = result_to_ipc_events(result)
        assert all(e.run_id == "r-9" for e in events)
        assert all(isinstance(e.timestamp, float) for e in events)


# ---------------------------------------------------------------------------
# 7. ChangedFile / IPCEvent / IPCChangedFile dataclass shape
# ---------------------------------------------------------------------------


class TestDataclassShapes:
    def test_changed_file_defaults(self) -> None:
        cf = ChangedFile(path="a", status="added")
        assert cf.additions == 0
        assert cf.deletions == 0
        assert cf.diff_patch == ""

    def test_ipc_event_defaults(self) -> None:
        ev = IPCEvent(event="x", run_id="r", timestamp=1.0)
        assert ev.payload == {}

    def test_ipc_changed_file_defaults(self) -> None:
        cf = IPCChangedFile(path="a", status="added")
        assert cf.additions == 0
        assert cf.deletions == 0
        assert cf.absolute_path == ""


# ---------------------------------------------------------------------------
# 8. run_id is captured from first event
# ---------------------------------------------------------------------------


class TestRunIdCapture:
    def test_run_id_from_event_payload(self) -> None:
        b = RunBridge()
        b.ingest(RunEvent(
            type=RunEventType.MESSAGE,
            payload={"content": "hi", "run_id": "from-event"},
        ))
        result = b.finalize()
        assert result.run_id == "from-event"

    def test_run_id_falls_back_to_constructor_arg(self) -> None:
        b = RunBridge(run_id="from-ctor")
        b.ingest(_ev(RunEventType.MESSAGE, {"content": "hi"}))
        result = b.finalize()
        assert result.run_id == "from-ctor"
