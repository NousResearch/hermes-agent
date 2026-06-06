#!/usr/bin/env python3
"""
Tests for executors/ipc.py — IPC protocol dataclasses + channel catalog.

Scope:
  - All Request / Response dataclasses instantiate with required + default fields
  - Default values match the docstring contract
  - Serialization shape (dataclasses.asdict) is JSON-friendly
  - HermesAPI class is a documentation stub (inert, non-instantiable by contract)
  - IPC_CHANNELS catalog covers every Request dataclass

Strictly no subprocess, no real network, no DB, no real files. Pure stdlib + pytest.
"""
from __future__ import annotations

import dataclasses
import json
from typing import get_type_hints

import pytest

from executors import ipc


# ---------------------------------------------------------------------------
# 1. Run-lifecycle dataclasses
# ---------------------------------------------------------------------------


class TestCreateRun:
    def test_minimal(self) -> None:
        req = ipc.CreateRunRequest(thread_id="t-1", prompt="hi", executor_type="hermes-local")
        assert req.thread_id == "t-1"
        assert req.prompt == "hi"
        assert req.executor_type == "hermes-local"
        # Defaults
        assert req.run_type == "main"
        assert req.project_root == "."

    def test_all_fields(self) -> None:
        req = ipc.CreateRunRequest(
            thread_id="t-2",
            prompt="build",
            executor_type="claude-code",
            run_type="review",
            project_root="/tmp/proj",
        )
        assert req.run_type == "review"
        assert req.project_root == "/tmp/proj"

    def test_json_serializable(self) -> None:
        req = ipc.CreateRunRequest(thread_id="t", prompt="p", executor_type="e")
        encoded = json.dumps(dataclasses.asdict(req))
        decoded = json.loads(encoded)
        assert decoded["thread_id"] == "t"
        assert decoded["run_type"] == "main"


class TestCreateRunResponse:
    def test_defaults(self) -> None:
        resp = ipc.CreateRunResponse(run_id="run-001", status="created")
        assert resp.run_id == "run-001"
        assert resp.status == "created"


class TestStopRun:
    def test_only_run_id(self) -> None:
        req = ipc.StopRunRequest(run_id="run-007")
        assert req.run_id == "run-007"
        d = dataclasses.asdict(req)
        assert d == {"run_id": "run-007"}


class TestContinueRun:
    def test_defaults(self) -> None:
        req = ipc.ContinueRunRequest(
            thread_id="t", prompt="p", previous_run_id="run-1"
        )
        assert req.executor_type == ""  # default empty
        d = dataclasses.asdict(req)
        assert d["executor_type"] == ""

    def test_response(self) -> None:
        resp = ipc.ContinueRunResponse(run_id="run-2")
        assert resp.run_id == "run-2"


class TestRetryRun:
    def test_defaults(self) -> None:
        req = ipc.RetryRunRequest(thread_id="t", prompt="p")
        assert req.executor_type == ""

    def test_response(self) -> None:
        resp = ipc.RetryRunResponse(run_id="r-3", run_seq=2)
        assert resp.run_seq == 2


# ---------------------------------------------------------------------------
# 2. Review / QA dataclasses
# ---------------------------------------------------------------------------


class TestTriggerReview:
    def test_defaults(self) -> None:
        req = ipc.TriggerReviewRequest(
            main_run_id="main-1", worktree_path="/tmp/wt"
        )
        assert req.diff_patch == ""
        assert req.task_goal == ""
        assert req.changed_files == []
        assert req.executor_type == ""

    def test_with_changed_files(self) -> None:
        req = ipc.TriggerReviewRequest(
            main_run_id="m",
            worktree_path="/wt",
            changed_files=["a.py", "b.ts"],
            executor_type="claude-code",
        )
        assert len(req.changed_files) == 2
        assert req.executor_type == "claude-code"


class TestTriggerQA:
    def test_defaults(self) -> None:
        req = ipc.TriggerQARequest(main_run_id="m", worktree_path="/wt")
        assert req.test_commands == []
        assert req.task_goal == ""
        assert req.executor_type == ""

    def test_with_commands(self) -> None:
        req = ipc.TriggerQARequest(
            main_run_id="m",
            worktree_path="/wt",
            test_commands=["pytest -q", "ruff check ."],
        )
        assert len(req.test_commands) == 2


# ---------------------------------------------------------------------------
# 3. Data-read dataclasses
# ---------------------------------------------------------------------------


class TestGetChangedFiles:
    def test_request(self) -> None:
        req = ipc.GetChangedFilesRequest(run_id="r-1")
        assert req.run_id == "r-1"

    def test_response_default_empty(self) -> None:
        resp = ipc.GetChangedFilesResponse(run_id="r-1")
        assert resp.files == []

    def test_response_with_files(self) -> None:
        resp = ipc.GetChangedFilesResponse(
            run_id="r-1",
            files=[{"path": "a.py", "status": "added", "additions": 10, "deletions": 0, "absolute_path": "/x/a.py"}],
        )
        assert len(resp.files) == 1
        assert resp.files[0]["path"] == "a.py"


class TestGetGatewayStatus:
    def test_defaults(self) -> None:
        resp = ipc.GetGatewayStatusResponse(connected=True)
        assert resp.connected is True
        assert resp.model == ""
        assert resp.error is None

    def test_with_error(self) -> None:
        resp = ipc.GetGatewayStatusResponse(
            connected=False, model="", error="not connected"
        )
        assert resp.error == "not connected"


# ---------------------------------------------------------------------------
# 4. Event streaming
# ---------------------------------------------------------------------------


class TestRawRunEvent:
    def test_minimal(self) -> None:
        ev = ipc.RawRunEvent(event="tool.completed", run_id="r-1", timestamp=100.0)
        assert ev.payload == {}
        assert ev.timestamp == 100.0

    def test_with_payload(self) -> None:
        ev = ipc.RawRunEvent(
            event="diff",
            run_id="r-1",
            timestamp=200.5,
            payload={"patch": "@@ -1 +1 @@\n-old\n+new\n", "files": 1},
        )
        assert ev.payload["files"] == 1

    def test_json_serializable(self) -> None:
        ev = ipc.RawRunEvent(event="e", run_id="r", timestamp=1.0, payload={"k": "v"})
        s = json.dumps(dataclasses.asdict(ev))
        assert json.loads(s)["payload"]["k"] == "v"


# ---------------------------------------------------------------------------
# 5. Approval
# ---------------------------------------------------------------------------


class TestResolveApproval:
    def test_required_only(self) -> None:
        req = ipc.ResolveApprovalRequest(run_id="r-1", decision="accept")
        assert req.comment is None

    def test_with_comment(self) -> None:
        req = ipc.ResolveApprovalRequest(
            run_id="r-1", decision="reject", comment="not yet ready"
        )
        assert req.decision == "reject"
        assert req.comment == "not yet ready"


# ---------------------------------------------------------------------------
# 6. HermesAPI (documentation stub)
# ---------------------------------------------------------------------------


class TestHermesAPIDocumentation:
    def test_class_exists(self) -> None:
        assert ipc.HermesAPI is not None

    def test_methods_have_docstring_or_signature(self) -> None:
        # Every method declared on HermesAPI must be callable-shaped.
        for name in (
            "createRun", "stopRun", "continueRun", "retryRun",
            "streamRunEvents", "resolveApproval",
            "triggerReview", "triggerQA",
            "getTaskThreads", "getChangedFiles", "getGatewayStatus",
        ):
            assert hasattr(ipc.HermesAPI, name), f"missing method: {name}"


# ---------------------------------------------------------------------------
# 7. IPC_CHANNELS catalog
# ---------------------------------------------------------------------------


class TestIPCChannelsCatalog:
    def test_is_dict(self) -> None:
        assert isinstance(ipc.IPC_CHANNELS, dict)
        assert len(ipc.IPC_CHANNELS) > 0

    def test_covers_lifecycle(self) -> None:
        for channel in (
            "run:create", "run:stop", "run:continue", "run:retry",
            "review:trigger", "qa:trigger",
            "approval:resolve",
            "data:changed-files", "data:gateway-status",
            "run:events:subscribe", "run:events:unsubscribe",
        ):
            assert channel in ipc.IPC_CHANNELS, f"missing channel: {channel}"

    def test_values_are_descriptive_strings(self) -> None:
        for k, v in ipc.IPC_CHANNELS.items():
            assert isinstance(v, str) and v, f"channel {k} has empty desc"


# ---------------------------------------------------------------------------
# 8. Round-trip: every Request is asdict()-able
# ---------------------------------------------------------------------------


class TestRoundTrip:
    @pytest.mark.parametrize("cls_name,kwargs", [
        ("CreateRunRequest", {"thread_id": "t", "prompt": "p", "executor_type": "e"}),
        ("CreateRunResponse", {"run_id": "r", "status": "created"}),
        ("StopRunRequest", {"run_id": "r"}),
        ("ContinueRunRequest", {"thread_id": "t", "prompt": "p", "previous_run_id": "r"}),
        ("ContinueRunResponse", {"run_id": "r"}),
        ("RetryRunRequest", {"thread_id": "t", "prompt": "p"}),
        ("RetryRunResponse", {"run_id": "r", "run_seq": 1}),
        ("TriggerReviewRequest", {"main_run_id": "m", "worktree_path": "/w"}),
        ("TriggerReviewResponse", {"review_run_id": "rv", "status": "ok"}),
        ("TriggerQARequest", {"main_run_id": "m", "worktree_path": "/w"}),
        ("TriggerQAResponse", {"qa_run_id": "q", "status": "ok"}),
        ("GetChangedFilesRequest", {"run_id": "r"}),
        ("GetChangedFilesResponse", {"run_id": "r"}),
        ("GetGatewayStatusResponse", {"connected": True}),
        ("RawRunEvent", {"event": "e", "run_id": "r", "timestamp": 0.0}),
        ("ResolveApprovalRequest", {"run_id": "r", "decision": "accept"}),
    ])
    def test_dataclass_json_roundtrip(self, cls_name: str, kwargs: dict) -> None:
        cls = getattr(ipc, cls_name)
        obj = cls(**kwargs)
        d = dataclasses.asdict(obj)
        # asdict must be JSON-encodable
        encoded = json.dumps(d)
        decoded = json.loads(encoded)
        assert isinstance(decoded, dict)
