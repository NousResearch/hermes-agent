"""Tests for /v1/runs endpoints: start, status, events, and stop.

Covers:
- POST /v1/runs — start a run (202)
- GET /v1/runs/{run_id} — poll run status
- GET /v1/runs/{run_id}/events — SSE event stream
- POST /v1/runs/{run_id}/stop — interrupt a running agent
- Auth, error handling, and cleanup
"""

import asyncio
import json
import threading
import time as _time
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)
from gateway.dev_execution import DevExecutionStore, _extract_completion_summary, supervisor_loop_tick
from gateway.dev_control.acceptance_criteria import acceptance_criteria_to_strings
from gateway.dev_control.clarifications import DevClarificationStore
from gateway.dev_control.plan_artifacts import DevPlanArtifactStore
from gateway.dev_control.acceptance_verification import DevVerificationStore
from gateway.dev_control.production_signals import DevProductionSignalStore
from gateway.dev_control.reliability import DevReliabilityStore
from gateway.dev_control.lab_loop import DevLabLoopStore
from gateway.dev_control.repo_grounding import collect_repo_grounding
from gateway.dev_control.read_models import build_agent_board_rows
from gateway.subagent_events import SubagentEventStore
from tools.ao_bridge import AOSession
from tools.openhands_bridge import OpenHandsSession


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    """Create an adapter with optional API key."""
    extra = {}
    if api_key:
        extra["key"] = api_key
    config = PlatformConfig(enabled=True, extra=extra)
    adapter = APIServerAdapter(config)
    return adapter


def _fake_text_response(content: str):
    message = type("Message", (), {"content": content})()
    choice = type("Choice", (), {"message": message})()
    return type("Response", (), {"choices": [choice]})()


def _fake_clarification_response():
    payload = {
        "questions": [
            {
                "question_id": "q1",
                "prompt": "What planning flow should this prioritize first?",
                "recommended_option_id": "a",
                "allow_freeform": True,
                "reason": "Scope controls implementation risk.",
                "options": [
                    {"option_id": "a", "label": "Guided planning", "description": "Use it to turn vague briefs into structured plans."},
                    {"option_id": "b", "label": "Dev questions", "description": "Use it mainly when Dev needs clarification."},
                ],
            },
            {
                "question_id": "q2",
                "prompt": "Where should the first version appear?",
                "recommended_option_id": "a",
                "allow_freeform": True,
                "reason": "The entry point affects state and UI scope.",
                "options": [
                    {"option_id": "a", "label": "Composer chip", "description": "Start from the main composer in Plan mode."},
                    {"option_id": "b", "label": "Separate panel", "description": "Keep planning separate from chat."},
                ],
            },
            {
                "question_id": "q3",
                "prompt": "What should completion produce?",
                "recommended_option_id": "a",
                "allow_freeform": True,
                "reason": "Output scope prevents accidental execution.",
                "options": [
                    {"option_id": "a", "label": "Clarified brief", "description": "Produce a structured brief only."},
                    {"option_id": "b", "label": "Execution plan", "description": "Draft a plan immediately."},
                ],
            },
        ]
    }
    return _fake_text_response(json.dumps(payload))


def _fake_plan_artifact_response(title: str = "Planning Mode Artifact"):
    payload = {
        "title": title,
        "overview": "Create a durable planning artifact from the clarified vision.",
        "product_intent": "Give Felipe a reviewable plan before build work starts.",
        "scope": ["Render the artifact in the planning panel."],
        "non_goals": ["Do not create or launch Dev execution plans."],
        "assumptions": ["The clarification answers are the source of truth."],
        "user_workflow": ["Felipe answers questions, reviews the artifact, then approves when ready."],
        "implementation_slices": [
            {"title": "Artifact storage", "description": "Persist generated plan artifacts and versions."},
            {"title": "Planning panel", "description": "Show the latest artifact in the right-side panel."},
        ],
        "validation_slices": [
            {"title": "No execution", "description": "Confirm no workers or Dev plans are created."},
        ],
        "acceptance_criteria": [
            {
                "statement": "Artifact can be retrieved and approved without execution.",
                "verification_method": "test",
                "verification_detail": "scripts/run_tests.sh tests/gateway/test_api_server_runs.py -- -k dev_plan_artifact",
                "machine_checkable": True,
            }
        ],
        "risks": ["The first artifact may still need human revision."],
        "open_questions": [],
        "recommended_next_action": "Review, revise if needed, then approve for Phase 29.",
    }
    return _fake_text_response(json.dumps(payload))


def _fake_clarified_brief_response():
    payload = {
        "refined_vision": "Build an interactive planning mode for vague feature ideas.",
        "goals": ["Clarify scope before creating a plan artifact."],
        "non_goals": ["Do not create or launch a Dev execution plan from this clarification alone."],
        "constraints": [],
        "assumptions": ["Project: Oryn Workspace"],
        "acceptance_criteria": [
            {
                "statement": "The clarified brief contains a verifiable definition of done.",
                "verification_method": "test",
                "verification_detail": "scripts/run_tests.sh tests/gateway/test_api_server_runs.py -- -k dev_clarification",
                "machine_checkable": True,
            }
        ],
        "risk_notes": ["Technical feasibility still needs codebase validation."],
        "open_questions": [],
        "suggested_next_action": "Review the clarified brief, then draft a Dev execution plan if the direction is correct.",
    }
    return _fake_text_response(json.dumps(payload))


def _fake_artifact_execution_tasks_response(project_id: str = "OrynWorkspace"):
    payload = {
        "tasks": [
            {
                "goal": "Implement artifact storage",
                "prompt": "Implement the approved artifact storage slice with validation evidence.",
                "profile_id": "workspace.implement",
                "project_id": project_id,
                "permissions": "edit",
                "dependencies": [],
                "acceptance_criteria": ["Artifact storage works."],
            },
            {
                "goal": "Validate planning panel",
                "prompt": "Inspect the planning panel and report whether the artifact appears.",
                "profile_id": "workspace.inspect",
                "project_id": project_id,
                "permissions": "read_only",
                "dependencies": ["Implement artifact storage"],
                "acceptance_criteria": ["Panel evidence is reported."],
            },
        ]
    }
    return _fake_text_response(json.dumps(payload))


class _FakeRuntimeRouter:
    def __init__(self):
        self.spawned = []

    def spawn(self, runtime: str, **kwargs):
        index = len(self.spawned) + 1
        session = AOSession(
            id=f"oryn-workspace-test-{index}",
            project_id=kwargs.get("project_id"),
            status="running",
            branch=f"session/oryn-workspace-test-{index}",
            workspace_path=f"/tmp/oryn-workspace-test-{index}",
            tmux_name=f"tmux-oryn-workspace-test-{index}",
            agent=kwargs.get("agent") or "codex",
            model=kwargs.get("model") or "gpt-5.5",
            reasoning_effort=kwargs.get("reasoning_effort"),
            open_command=f"tmux attach -t tmux-oryn-workspace-test-{index}",
        )
        self.spawned.append({"runtime": runtime, "kwargs": kwargs, "session": session})
        return session


def _create_runs_app(adapter: APIServerAdapter) -> web.Application:
    """Create an aiohttp app with /v1/runs routes registered."""
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    app.router.add_get("/v1/runs/{run_id}/subagents/events", adapter._handle_run_subagent_events)
    app.router.add_get("/v1/subagents/board", adapter._handle_subagent_board)
    app.router.add_get("/v1/subagents/events", adapter._handle_subagent_events)
    app.router.add_get("/v1/oryn/project-dashboard", adapter._handle_oryn_project_dashboard)
    app.router.add_get("/v1/dev/launch-profiles", adapter._handle_dev_launch_profiles)
    app.router.add_get("/v1/dev/runtimes", adapter._handle_dev_worker_runtimes)
    app.router.add_post("/v1/dev/runtime-selection", adapter._handle_dev_runtime_selection)
    app.router.add_get("/v1/dev/harness/components", adapter._handle_dev_harness_components)
    app.router.add_post("/v1/dev/harness/report", adapter._handle_dev_harness_report)
    app.router.add_get("/v1/dev/harness/recommendations", adapter._handle_dev_harness_recommendations)
    app.router.add_post("/v1/dev/harness/recommendations", adapter._handle_dev_harness_recommendations)
    app.router.add_get("/v1/dev/harness/recommendations/{recommendation_run_id}", adapter._handle_dev_harness_recommendation_detail)
    app.router.add_get("/v1/dev/harness/benchmarks", adapter._handle_dev_harness_benchmarks)
    app.router.add_post("/v1/dev/harness/benchmarks", adapter._handle_dev_harness_benchmarks)
    app.router.add_get("/v1/dev/harness/benchmarks/{benchmark_run_id}", adapter._handle_dev_harness_benchmark_detail)
    app.router.add_get("/v1/dev/clarifications", adapter._handle_dev_clarifications)
    app.router.add_post("/v1/dev/clarifications", adapter._handle_dev_clarifications)
    app.router.add_get("/v1/dev/clarifications/{clarification_id}", adapter._handle_dev_clarification_detail)
    app.router.add_post("/v1/dev/clarifications/{clarification_id}/answer", adapter._handle_dev_clarification_answer)
    app.router.add_post("/v1/dev/clarifications/{clarification_id}/complete", adapter._handle_dev_clarification_complete)
    app.router.add_post("/v1/dev/clarifications/{clarification_id}/cancel", adapter._handle_dev_clarification_cancel)
    app.router.add_get("/v1/dev/plan-artifacts", adapter._handle_dev_plan_artifacts)
    app.router.add_post("/v1/dev/plan-artifacts", adapter._handle_dev_plan_artifacts)
    app.router.add_get("/v1/dev/plan-artifacts/{plan_artifact_id}", adapter._handle_dev_plan_artifact_detail)
    app.router.add_post("/v1/dev/plan-artifacts/{plan_artifact_id}/revise", adapter._handle_dev_plan_artifact_revise)
    app.router.add_post("/v1/dev/plan-artifacts/{plan_artifact_id}/approve", adapter._handle_dev_plan_artifact_approve)
    app.router.add_post("/v1/dev/plan-artifacts/{plan_artifact_id}/cancel", adapter._handle_dev_plan_artifact_cancel)
    app.router.add_post("/v1/dev/plan-artifacts/{plan_artifact_id}/create-execution-plan", adapter._handle_dev_plan_artifact_create_execution_plan)
    app.router.add_get("/v1/dev/plan-artifacts/{plan_artifact_id}/builds", adapter._handle_dev_plan_artifact_builds)
    app.router.add_get("/v1/dev/runtimes/openhands/server", adapter._handle_dev_openhands_server_status)
    app.router.add_post("/v1/dev/runtimes/openhands/server/start", adapter._handle_dev_openhands_server_start)
    app.router.add_post("/v1/dev/runtimes/openhands/server/stop", adapter._handle_dev_openhands_server_stop)
    app.router.add_get("/v1/dev/execution-plans", adapter._handle_dev_execution_plans)
    app.router.add_post("/v1/dev/execution-plans", adapter._handle_dev_execution_plans)
    app.router.add_get("/v1/dev/verification-runs", adapter._handle_dev_verification_runs)
    app.router.add_post("/v1/dev/verification-runs", adapter._handle_dev_verification_runs)
    app.router.add_get("/v1/dev/verification-runs/{verification_run_id}", adapter._handle_dev_verification_run_detail)
    app.router.add_get("/v1/dev/signal-reports", adapter._handle_dev_signal_reports)
    app.router.add_post("/v1/dev/signal-reports", adapter._handle_dev_signal_reports)
    app.router.add_get("/v1/dev/signal-reports/{report_id}", adapter._handle_dev_signal_report_detail)
    app.router.add_get("/v1/dev/backlog-proposals", adapter._handle_dev_backlog_proposals)
    app.router.add_post("/v1/dev/backlog-proposals/{proposal_id}/{action}", adapter._handle_dev_backlog_proposal_action)
    app.router.add_get("/v1/dev/signal-health", adapter._handle_dev_signal_health)
    app.router.add_get("/v1/dev/reliability", adapter._handle_dev_reliability)
    app.router.add_post("/v1/dev/reliability/recompute", adapter._handle_dev_reliability_recompute)
    app.router.add_get("/v1/dev/reliability/weakest", adapter._handle_dev_reliability_weakest)
    app.router.add_get("/v1/dev/reliability/{category:.+}", adapter._handle_dev_reliability_category)
    app.router.add_get("/v1/dev/lab-loop/health", adapter._handle_dev_lab_loop_health)
    app.router.add_post("/v1/dev/execution-plans/supervise", adapter._handle_dev_execution_plans_supervise)
    app.router.add_get("/v1/dev/supervisor/loop", adapter._handle_dev_supervisor_loop)
    app.router.add_post("/v1/dev/supervisor/loop", adapter._handle_dev_supervisor_loop)
    app.router.add_get("/v1/dev/runbooks", adapter._handle_dev_runbooks)
    app.router.add_post("/v1/dev/runbooks", adapter._handle_dev_runbooks)
    app.router.add_get("/v1/dev/runbooks/{runbook_id}", adapter._handle_dev_runbook_detail)
    app.router.add_post("/v1/dev/runbooks/{runbook_id}", adapter._handle_dev_runbook_detail)
    app.router.add_get("/v1/dev/supervisor/approvals", adapter._handle_dev_supervisor_approvals)
    app.router.add_get("/v1/dev/supervisor/approvals/{approval_id}", adapter._handle_dev_supervisor_approval_detail)
    app.router.add_post("/v1/dev/supervisor/approvals/{approval_id}/approve", adapter._handle_dev_supervisor_approval_approve)
    app.router.add_post("/v1/dev/supervisor/approvals/{approval_id}/deny", adapter._handle_dev_supervisor_approval_deny)
    app.router.add_post("/v1/dev/supervisor/approvals/{approval_id}/apply", adapter._handle_dev_supervisor_approval_apply)
    app.router.add_get("/v1/dev/execution-plans/{plan_id}/draft-review", adapter._handle_dev_execution_plan_draft_review)
    app.router.add_post("/v1/dev/execution-plans/{plan_id}/revise-draft", adapter._handle_dev_execution_plan_revise_draft)
    app.router.add_post("/v1/dev/execution-plans/{plan_id}/approve-draft", adapter._handle_dev_execution_plan_approve_draft)
    app.router.add_post("/v1/dev/execution-plans/{plan_id}/cancel-draft", adapter._handle_dev_execution_plan_cancel_draft)
    app.router.add_get("/v1/dev/execution-plans/{plan_id}", adapter._handle_dev_execution_plan_detail)
    app.router.add_get("/v1/dev/execution-plans/{plan_id}/status", adapter._handle_dev_execution_plan_status)
    app.router.add_post("/v1/dev/execution-plans/{plan_id}/synthesize", adapter._handle_dev_execution_plan_synthesize)
    app.router.add_get("/v1/dev/execution-plans/{plan_id}/review", adapter._handle_dev_execution_plan_review)
    app.router.add_post("/v1/dev/execution-plans/{plan_id}/review", adapter._handle_dev_execution_plan_review)
    app.router.add_post("/v1/dev/execution-plans/{plan_id}/apply-review", adapter._handle_dev_execution_plan_apply_review)
    app.router.add_get("/v1/dev/execution-plans/{plan_id}/next-step", adapter._handle_dev_execution_plan_next_step)
    app.router.add_post("/v1/dev/execution-plans/{plan_id}/next-step", adapter._handle_dev_execution_plan_next_step)
    app.router.add_post("/v1/dev/execution-plans/{plan_id}/test-state", adapter._handle_dev_execution_plan_test_state)
    app.router.add_post("/v1/dev/execution-plans/{plan_id}/launch", adapter._handle_dev_execution_plan_launch)
    app.router.add_post("/v1/runs/{run_id}/approval", adapter._handle_run_approval)
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    app.router.add_get("/v1/ao/sessions", adapter._handle_ao_sessions)
    app.router.add_get("/v1/ao/sessions/{session_id}", adapter._handle_ao_session_detail)
    app.router.add_post("/v1/ao/sessions/{session_id}/stop", adapter._handle_ao_session_stop)
    app.router.add_post("/v1/ao/sessions/{session_id}/open", adapter._handle_ao_session_open)
    app.router.add_get("/v1/ao/sessions/{session_id}/diagnostics", adapter._handle_ao_session_diagnostics)
    app.router.add_post("/v1/ao/sessions/{session_id}/follow-up", adapter._handle_ao_session_follow_up)
    app.router.add_post("/v1/ao/sessions/{session_id}/resume", adapter._handle_ao_session_resume)
    app.router.add_post("/v1/ao/sessions/{session_id}/retry", adapter._handle_ao_session_retry)
    app.router.add_post("/v1/ao/sessions/{session_id}/repair-retry", adapter._handle_ao_session_repair_retry)
    app.router.add_post("/v1/ao/sessions/{session_id}/reassign", adapter._handle_ao_session_reassign)
    return app


def _persist_runtime_policy_benchmark(
    db_path,
    *,
    benchmark_run_id: str = "devbench-policy",
    ao: dict | None = None,
    openhands: dict | None = None,
) -> None:
    from gateway.dev_control.harness_benchmarks import DevHarnessBenchmarkStore

    now = _time.time()
    defaults = {
        "ao": {
            "runtime": "ao",
            "case_count": 6,
            "iteration_count": 2,
            "median_score": 0.887,
            "median_task_quality_score": 0.75,
            "median_contract_compliance_score": 1.0,
            "marker_pass_rate": 1.0,
            "required_evidence_pass_rate": 1.0,
            "delivery_failure_rate": 0.0,
            "average_duration_seconds": 35.0,
            "token_sample_count": 6,
            "cost_sample_count": 0,
            "total_tokens": 1000,
            "total_cost_usd": 0.0,
        },
        "openhands": {
            "runtime": "openhands",
            "case_count": 6,
            "iteration_count": 2,
            "median_score": 0.879,
            "median_task_quality_score": 0.737,
            "median_contract_compliance_score": 1.0,
            "marker_pass_rate": 1.0,
            "required_evidence_pass_rate": 1.0,
            "delivery_failure_rate": 0.0,
            "average_duration_seconds": 14.0,
            "token_sample_count": 6,
            "cost_sample_count": 6,
            "total_tokens": 800,
            "total_cost_usd": 0.01,
        },
    }
    defaults["ao"].update(ao or {})
    defaults["openhands"].update(openhands or {})
    store = DevHarnessBenchmarkStore(db_path)
    try:
        store.persist_run({
            "ok": True,
            "object": "hermes.dev_harness_benchmark_run",
            "benchmark_run_id": benchmark_run_id,
            "mode": "live",
            "live": True,
            "created_at": now,
            "completed_at": now,
            "runtime_results": [defaults["ao"], defaults["openhands"]],
            "case_results": [],
            "cases": [],
            "summary": {},
        })
    finally:
        store.close()


def _make_slow_agent(**kwargs):
    """Create a mock agent that blocks in run_conversation until interrupted.

    Returns (mock_agent, agent_ready_event, interrupt_event) where
    agent_ready_event is set once run_conversation starts, and
    interrupt_event is set when interrupt() is called.
    """
    ready = threading.Event()
    interrupted = threading.Event()

    mock_agent = MagicMock()

    def _do_interrupt(message=None):
        interrupted.set()

    mock_agent.interrupt = MagicMock(side_effect=_do_interrupt)

    def _slow_run(user_message=None, conversation_history=None, task_id=None):
        ready.set()
        # Block until interrupt() is called
        interrupted.wait(timeout=10)
        return {"final_response": "interrupted"}

    mock_agent.run_conversation.side_effect = _slow_run
    mock_agent.session_prompt_tokens = 0
    mock_agent.session_completion_tokens = 0
    mock_agent.session_total_tokens = 0

    return mock_agent, ready, interrupted


class _FakeAOBridge:
    def __init__(self):
        self.sent_messages = []
        self.spawn_kwargs = None
        self.codex_exec_kwargs = None
        self.killed_sessions = []
        self.session = AOSession(
            id="oryn-workspace-9",
            project_id="OrynWorkspace",
            status="working",
            activity="active",
            branch="feat/retry",
            workspace_path="/tmp/oryn-workspace-9",
            tmux_name="tmux-oryn-workspace-9",
            agent="codex",
            model="gpt-5.5",
            reasoning_effort="medium",
            open_command="tmux attach -t tmux-oryn-workspace-9",
        )

    def send(self, session_id, message):
        self.sent_messages.append((session_id, message))
        return self.session

    def status(self, session_id):
        return self.session

    def spawn(self, **kwargs):
        self.spawn_kwargs = kwargs
        return self.session

    def list(self, project_id=None):
        if project_id and project_id != self.session.project_id:
            return []
        return [self.session]

    def runtime_health(self, session):
        return {"runtime_health": "ok", "runtime_warning": None}

    def capture_output(self, session, lines=40):
        return "line one\nline two\nlatest worker output"

    def kill(self, session_id, **_kwargs):
        self.killed_sessions.append(session_id)
        self.session.status = "killed"

    def run_codex_exec_benchmark(self, **kwargs):
        self.codex_exec_kwargs = kwargs
        return {
            "session_id": "codex-exec-test",
            "status": "completed",
            "summary": (
                "BENCHMARK_RESULT\n"
                "marker: BENCH_AGENT_BOARD_METADATA_DONE\n"
                "finding_1: WorkspaceAgentBoardView renders runtime metadata.\n"
                "finding_2: WorkspaceSubagentActivity exposes agent and model metadata.\n"
                "FINAL_MARKER: BENCH_AGENT_BOARD_METADATA_DONE"
            ),
            "output_tail": (
                "BENCHMARK_RESULT\n"
                "finding_1: WorkspaceAgentBoardView renders runtime metadata.\n"
                "finding_2: WorkspaceSubagentActivity exposes agent and model metadata.\n"
                "FINAL_MARKER: BENCH_AGENT_BOARD_METADATA_DONE"
            ),
            "duration_seconds": 1.2,
            "workspace_path": "/tmp/oryn-workspace",
            "agent": "codex",
            "model": "gpt-5.5",
            "reasoning_effort": "medium",
            "token_total": 1234,
        }


class _FakeOpenHandsBridge:
    def __init__(self, *, launch_supported=True, spawn_error: Exception | None = None):
        self.launch_supported = launch_supported
        self.spawn_error = spawn_error
        self.sent_messages = []
        self.spawn_kwargs = None
        self.session = OpenHandsSession(
            id="oh-conv-1",
            project_id="OrynWorkspace",
            status="running",
            workspace_path="/tmp/oh-conv-1",
            branch="session/oh-conv-1",
            agent="openhands",
            model="gpt-5.5",
            reasoning_effort="medium",
            output_tail="OpenHands output tail",
            open_command="http://127.0.0.1:3000/conversations/oh-conv-1",
        )

    def discovery(self):
        if self.launch_supported:
            return {
                "available": True,
                "launch_supported": True,
                "configured_mode": "server",
                "setup_warning": None,
                "server_url": "http://127.0.0.1:3000",
                "sdk_available": False,
                "command": None,
            }
        return {
            "available": False,
            "launch_supported": False,
            "configured_mode": "missing",
            "setup_warning": "OpenHands is not installed or configured.",
            "server_url": None,
            "sdk_available": False,
            "command": None,
        }

    def spawn(self, **kwargs):
        self.spawn_kwargs = kwargs
        if self.spawn_error:
            raise self.spawn_error
        return self.session

    def status(self, session_id):
        return self.session if session_id == self.session.id else None

    def list(self, project_id=None):
        if project_id and project_id != self.session.project_id:
            return []
        return [self.session]

    def send(self, session_id, message):
        self.sent_messages.append((session_id, message))
        return self.session

    def kill(self, session_id, **_kwargs):
        self.session.status = "killed"

    def capture_output(self, session, lines=40):
        return session.output_tail or ""

    def runtime_health(self, session):
        return {"runtime_health": "ok", "runtime_warning": None, "configured_mode": "server"}


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


# ---------------------------------------------------------------------------
# POST /v1/runs — start a run
# ---------------------------------------------------------------------------


class TestStartRun:
    @pytest.mark.asyncio
    async def test_start_returns_202(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 10
                mock_agent.session_completion_tokens = 5
                mock_agent.session_total_tokens = 15
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                assert data["status"] == "started"
                assert data["run_id"].startswith("run_")

                status_resp = await cli.get(f"/v1/runs/{data['run_id']}")
                assert status_resp.status == 200
                status = await status_resp.json()
                assert status["run_id"] == data["run_id"]
                assert status["status"] in {"queued", "running", "completed"}
                assert status["object"] == "hermes.run"

    @pytest.mark.asyncio
    async def test_start_invalid_json_returns_400(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_start_missing_input_returns_400(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json={"model": "test"})
            assert resp.status == 400
            data = await resp.json()
            assert "input" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_start_empty_input_returns_400(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json={"input": ""})
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_start_invalid_history_does_not_allocate_run(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs",
                json={"input": "hello", "conversation_history": {"role": "user"}},
            )
        assert resp.status == 400
        assert adapter._run_streams == {}
        assert adapter._run_statuses == {}

    @pytest.mark.asyncio
    async def test_start_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json={"input": "hello"})
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_start_with_valid_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "ok"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                assert resp.status == 202

    @pytest.mark.asyncio
    async def test_run_cleans_up_temporary_agent(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 10
                mock_agent.session_completion_tokens = 5
                mock_agent.session_total_tokens = 15
                mock_agent._session_messages = [{"role": "user", "content": "hello"}]
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(20):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

                assert status["status"] == "completed"
                mock_agent.shutdown_memory_provider.assert_called_once_with(mock_agent._session_messages)
                mock_agent.close.assert_called_once()


# ---------------------------------------------------------------------------
# GET /v1/runs/{run_id} — poll run status
# ---------------------------------------------------------------------------


class TestRunStatus:
    @pytest.mark.asyncio
    async def test_status_completed_run_includes_output_and_usage(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 4
                mock_agent.session_completion_tokens = 2
                mock_agent.session_total_tokens = 6
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(20):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    assert status_resp.status == 200
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

                assert status["status"] == "completed"
                assert status["output"] == "done"
                assert status["usage"]["total_tokens"] == 6
                assert status["last_event"] == "run.completed"

    @pytest.mark.asyncio
    async def test_status_reflects_explicit_session_id(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "session_id": "space-session"},
                )
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(20):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

                mock_agent.run_conversation.assert_called_once()
                assert mock_agent.run_conversation.call_args.kwargs["task_id"] == "space-session"
                assert status["session_id"] == "space-session"

    @pytest.mark.asyncio
    async def test_status_not_found_returns_404(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_nonexistent")
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_status_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_any")
        assert resp.status == 401


# ---------------------------------------------------------------------------
# GET /v1/runs/{run_id}/events — SSE event stream
# ---------------------------------------------------------------------------


class TestRunEvents:
    @pytest.mark.asyncio
    async def test_run_event_callback_forwards_subagent_events(self, adapter, tmp_path):
        run_id = "run_subagents"
        q = asyncio.Queue()
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        adapter._run_streams[run_id] = q
        adapter._run_statuses[run_id] = {"run_id": run_id, "status": "running"}

        callback = adapter._make_run_event_callback(run_id, asyncio.get_running_loop(), session_id="session-1")
        for event_type in (
            "subagent.start",
            "subagent.tool",
            "subagent.progress",
            "subagent.thinking",
            "subagent.complete",
        ):
            callback(
                event_type,
                tool_name="terminal" if event_type == "subagent.tool" else None,
                preview="latest activity",
                subagent_id="child-1",
                parent_id="parent-1",
                depth=1,
                goal="Inspect the workspace",
                status="completed" if event_type == "subagent.complete" else "running",
                summary="Found the relevant files" if event_type == "subagent.complete" else None,
            )

            event = await asyncio.wait_for(q.get(), timeout=1.0)
            assert event["event"] == event_type
            assert event["event_id"] > 0
            assert event["schema_version"] == 1
            assert event["session_id"] == "session-1"
            assert event["run_id"] == run_id
            assert event["subagent_id"] == "child-1"
            assert event["parent_id"] == "parent-1"
            assert event["depth"] == 1
            assert event["goal"] == "Inspect the workspace"
            assert event["status"] in {"running", "completed"}

        assert event["summary"] == "Found the relevant files"
        stored = adapter._subagent_event_store.list_events(session_id="session-1")
        assert [item["event"] for item in stored] == [
            "subagent.start",
            "subagent.tool",
            "subagent.progress",
            "subagent.thinking",
            "subagent.complete",
        ]
        assert {item["schema_version"] for item in stored} == {1}

    @pytest.mark.asyncio
    async def test_run_subagent_events_replay_endpoint(self, adapter, tmp_path):
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        adapter._subagent_event_store.append_event({
            "event": "subagent.start",
            "run_id": "run_replay",
            "session_id": "session-replay",
            "subagent_id": "child-1",
            "depth": 0,
            "goal": "Replay me",
            "status": "running",
        })
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_replay/subagents/events")
            assert resp.status == 200
            data = await resp.json()

        assert data["total"] == 1
        assert data["data"][0]["event"] == "subagent.start"
        assert data["data"][0]["subagent_id"] == "child-1"

    @pytest.mark.asyncio
    async def test_run_subagent_events_replay_accepts_legacy_events_without_schema_version(self, adapter, tmp_path):
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        legacy_payload = {
            "event": "subagent.complete",
            "run_id": "run_legacy",
            "session_id": "session-legacy",
            "subagent_id": "child-legacy",
            "runtime": "hermes",
            "status": "completed",
            "summary": "Legacy event before schema version.",
            "created_at": 123.0,
        }
        with adapter._subagent_event_store._conn:
            adapter._subagent_event_store._conn.execute(
                """
                INSERT INTO subagent_events (
                    created_at, session_id, run_id, subagent_id, parent_id,
                    runtime, ao_session_id, event_type, status, goal, summary, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    123.0,
                    "session-legacy",
                    "run_legacy",
                    "child-legacy",
                    None,
                    "hermes",
                    None,
                    "subagent.complete",
                    "completed",
                    None,
                    "Legacy event before schema version.",
                    json.dumps(legacy_payload),
                ),
            )
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_legacy/subagents/events")
            assert resp.status == 200
            data = await resp.json()

        assert data["total"] == 1
        assert data["data"][0]["event"] == "subagent.complete"
        assert "schema_version" not in data["data"][0]

    @pytest.mark.asyncio
    async def test_global_subagent_events_endpoint_filters_events(self, adapter, tmp_path):
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        adapter._subagent_event_store.append_event({
            "event": "subagent.start",
            "run_id": "run-1",
            "session_id": "session-1",
            "subagent_id": "native-1",
            "runtime": "hermes",
            "depth": 0,
            "goal": "Native work",
            "status": "running",
        })
        adapter._subagent_event_store.append_event({
            "event": "subagent.complete",
            "run_id": "run-2",
            "session_id": "session-2",
            "subagent_id": "ao:oryn-workspace-9",
            "ao_session_id": "oryn-workspace-9",
            "runtime": "ao",
            "depth": 0,
            "goal": "AO work",
            "status": "completed",
        })
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/subagents/events?runtime=ao")
            assert resp.status == 200
            data = await resp.json()

        assert data["total"] == 1
        assert data["data"][0]["ao_session_id"] == "oryn-workspace-9"

    @pytest.mark.asyncio
    async def test_dev_launch_profiles_endpoint_returns_defaults(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/dev/launch-profiles")
            assert resp.status == 200
            data = await resp.json()

        profile_ids = {profile["id"] for profile in data["data"]}
        assert "workspace.inspect" in profile_ids
        assert "platform.implement" in profile_ids
        workspace_profile = next(profile for profile in data["data"] if profile["id"] == "workspace.inspect")
        assert workspace_profile["project_id"] == "OrynWorkspace"
        assert workspace_profile["agent"] == "codex"
        assert workspace_profile["model"] == "gpt-5.5"
        assert workspace_profile["reasoning_effort"] == "medium"
        assert workspace_profile["runtime"] == "ao"

    @pytest.mark.asyncio
    async def test_dev_worker_runtimes_endpoint_returns_ao_fixture_and_openhands(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/dev/runtimes")
            assert resp.status == 200
            data = await resp.json()

        runtimes = {runtime["id"]: runtime for runtime in data["data"]}
        assert runtimes["ao"]["available"] is True
        assert runtimes["ao"]["launch_supported"] is True
        assert runtimes["ao"]["test_only"] is False
        assert "spawn" in runtimes["ao"]["supported_actions"]
        assert runtimes["ao"]["capabilities"]["can_spawn"] is True
        assert runtimes["ao"]["can_stop"] is True
        assert runtimes["ao"]["supports_terminal"] is True
        assert runtimes["fixture"]["available"] is True
        assert runtimes["fixture"]["launch_supported"] is False
        assert runtimes["fixture"]["test_only"] is True
        assert runtimes["fixture"]["capabilities"]["test_only"] is True
        assert runtimes["fixture"]["can_spawn"] is False
        assert runtimes["openhands"]["label"] == "OpenHands"
        assert runtimes["openhands"]["test_only"] is False
        assert runtimes["openhands"]["configured_mode"] in {"missing", "sdk", "cli", "server"}
        assert "runtime_health" in runtimes["openhands"]["supported_actions"]
        assert "can_capture_output" in runtimes["openhands"]["capabilities"]

    @pytest.mark.asyncio
    async def test_dev_runtime_selection_prefers_openhands_for_auto_read_only(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)
        openhands = _FakeOpenHandsBridge()

        async with TestClient(TestServer(app)) as cli:
            with patch("tools.openhands_bridge.OpenHandsBridge", return_value=openhands):
                resp = await cli.post("/v1/dev/runtime-selection", json={
                    "runtime": "auto",
                    "goal": "Inspect Agent Board state",
                    "prompt": "Inspect the files read-only and report one finding. Do not edit files.",
                    "permissions": "read_only",
                    "project_id": "OrynWorkspace",
                })
                assert resp.status == 200
                data = await resp.json()

        assert data["object"] == "hermes.dev_runtime_selection"
        assert data["selected_runtime"] == "openhands"
        assert data["selection_mode"] == "auto"
        assert data["fallback_runtime"] == "ao"
        assert "can_capture_output" in data["required_capabilities"]
        assert data["runtime_policy_status"] == "insufficient_evidence"
        assert "No live AO/OpenHands benchmark evidence" in data["runtime_policy_evidence"]["reason"]

    @pytest.mark.asyncio
    async def test_dev_runtime_selection_uses_healthy_benchmark_evidence_for_auto_read_only(self, adapter, tmp_path):
        db_path = tmp_path / "state.db"
        adapter._dev_execution_store = DevExecutionStore(db_path)
        _persist_runtime_policy_benchmark(db_path, benchmark_run_id="devbench-healthy")
        app = _create_runs_app(adapter)
        openhands = _FakeOpenHandsBridge()

        async with TestClient(TestServer(app)) as cli:
            with patch("tools.openhands_bridge.OpenHandsBridge", return_value=openhands):
                resp = await cli.post("/v1/dev/runtime-selection", json={
                    "runtime": "auto",
                    "goal": "Inspect Agent Board state",
                    "prompt": "Inspect the files read-only and report one finding.",
                    "permissions": "read_only",
                })
                assert resp.status == 200
                data = await resp.json()

        assert data["selected_runtime"] == "openhands"
        assert data["runtime_policy_status"] == "healthy"
        assert data["runtime_policy_evidence"]["benchmark_run_id"] == "devbench-healthy"
        assert data["runtime_policy_evidence"]["runtimes"]["openhands"]["sample_count"] == 6

    @pytest.mark.asyncio
    async def test_dev_clarification_lifecycle_persists_answers_and_brief(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            project_context = {
                "project_id": "OrynWorkspace",
                "project_name": "Oryn Workspace",
                "vision": "Make Oryn a self-running development system.",
                "coordinator_profile": "dev",
                "repositories": [{"label": "Workspace", "path": "/Users/felipe/projects/Oryn"}],
                "work_items": ["Thread project planning context"],
            }
            with patch("gateway.dev_control.clarifications.call_llm", return_value=_fake_clarification_response()) as call_mock:
                start = await cli.post("/v1/dev/clarifications", json={
                    "vision_brief": "Build an interactive planning mode for vague feature ideas.",
                    "project_id": "OrynWorkspace",
                    "session_id": "session-1",
                    "project_context": project_context,
                })
            assert start.status == 200
            data = await start.json()

            assert data["object"] == "hermes.dev_clarification"
            assert data["status"] == "active"
            assert data["project_id"] == "OrynWorkspace"
            assert data["project_context"]["project_name"] == "Oryn Workspace"
            assert data["project_context"]["repositories"][0]["path"] == "/Users/felipe/projects/Oryn"
            assert len(data["questions"]) == 3
            assert data["current_question"]["question_id"] == "q1"
            assert "Oryn Workspace" in call_mock.call_args.kwargs["messages"][1]["content"]

            first = await cli.post(f"/v1/dev/clarifications/{data['clarification_id']}/answer", json={
                "question_id": "q1",
                "option_id": "a",
            })
            assert first.status == 200
            first_data = await first.json()
            assert first_data["current_question_index"] == 1
            assert first_data["answers"][0]["option_label"] == "Guided planning"

            second = await cli.post(f"/v1/dev/clarifications/{data['clarification_id']}/answer", json={
                "question_id": "q2",
                "answer_text": "Use it for Dev planning first, then general agent questions.",
            })
            assert second.status == 200

            with patch("gateway.dev_control.clarifications.call_llm", return_value=_fake_clarified_brief_response()) as synthesis_mock:
                complete = await cli.post(f"/v1/dev/clarifications/{data['clarification_id']}/complete", json={})
            assert complete.status == 200
            completed = await complete.json()

            assert completed["status"] == "completed"
            assert completed["clarified_brief"]["refined_vision"].startswith("Build an interactive")
            assert "Do not create or launch a Dev execution plan" in completed["clarified_brief"]["non_goals"][0]
            assert "Project: Oryn Workspace" in completed["clarified_brief"]["assumptions"]
            assert completed["clarified_brief"]["acceptance_criteria"][0]["machine_checkable"] is True
            assert synthesis_mock.call_args.kwargs["task"] == "dev_clarification_synthesis"
            assert adapter._dev_execution_store.list_plans(limit=10) == []

            listed = await cli.get("/v1/dev/clarifications?status=completed")
            assert listed.status == 200
            listed_data = await listed.json()
            assert listed_data["total"] == 1
            assert listed_data["data"][0]["clarification_id"] == data["clarification_id"]
            project_listed = await cli.get("/v1/dev/clarifications?project_id=OrynWorkspace")
            other_project_listed = await cli.get("/v1/dev/clarifications?project_id=OtherProject")
            assert project_listed.status == 200
            assert other_project_listed.status == 200
            assert (await project_listed.json())["total"] == 1
            assert (await other_project_listed.json())["total"] == 0

    @pytest.mark.asyncio
    async def test_dev_clarification_resolves_project_id_from_project_context_only(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch("gateway.dev_control.clarifications.call_llm", return_value=_fake_clarification_response()):
                start = await cli.post("/v1/dev/clarifications", json={
                    "vision_brief": "Scope planning to a custom Hermes project id.",
                    "project_context": {
                        "project_id": "FelipeSideProject",
                        "project_name": "Felipe Side Project",
                        "vision": "Keep this work isolated from OrynWorkspace defaults.",
                    },
                })
            assert start.status == 200
            data = await start.json()
            assert data["project_id"] == "FelipeSideProject"
            assert data["project_context"]["project_id"] == "FelipeSideProject"

            listed = await cli.get("/v1/dev/clarifications?project_id=FelipeSideProject")
            other = await cli.get("/v1/dev/clarifications?project_id=OrynWorkspace")
            assert (await listed.json())["total"] == 1
            assert (await other.json())["total"] == 0

    @pytest.mark.asyncio
    async def test_dev_clarification_persists_bounded_repo_grounding(self, adapter, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "AGENTS.md").write_text("Use focused changes and run tests.\n", encoding="utf-8")
        (repo / "feedback_loop.py").write_text("def runtime_error_to_backlog():\n    return 'backlog'\n", encoding="utf-8")
        before = sorted(path.relative_to(repo) for path in repo.rglob("*"))
        monkeypatch.setenv("ORYN_REPO_GROUNDING_ROOTS", str(tmp_path))
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch("gateway.dev_control.clarifications.call_llm", return_value=_fake_clarification_response()) as call_mock:
                resp = await cli.post("/v1/dev/clarifications", json={
                    "vision_brief": "Turn runtime errors into reviewed backlog items.",
                    "project_context": {
                        "repositories": [{"label": "Fixture", "path": str(repo)}],
                    },
                })
            assert resp.status == 200
            data = await resp.json()

        after = sorted(path.relative_to(repo) for path in repo.rglob("*"))
        assert before == after
        assert data["grounding"]["time_budget_seconds"] == 8.0
        assert data["grounding"]["file_budget"] == 60
        assert data["grounding"]["byte_budget"] == 204800
        assert data["grounding_provenance"]
        assert any(path.endswith("AGENTS.md") for path in data["grounding_provenance"])
        assert "Read-only repository grounding" in call_mock.call_args.kwargs["messages"][1]["content"]

    @pytest.mark.asyncio
    async def test_dev_clarification_grounding_invalid_path_warns_without_raising(self, adapter, tmp_path, monkeypatch):
        monkeypatch.setenv("ORYN_REPO_GROUNDING_ROOTS", str(tmp_path))
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch("gateway.dev_control.clarifications.call_llm", return_value=_fake_clarification_response()):
                resp = await cli.post("/v1/dev/clarifications", json={
                    "vision_brief": "Clarify work with an unavailable repository.",
                    "project_context": {
                        "repositories": [{"label": "Missing", "path": str(tmp_path / "missing")}],
                    },
                })
            assert resp.status == 200
            data = await resp.json()

        assert data["grounding_provenance"] == []
        assert data["grounding_warnings"]
        assert "does not exist" in data["grounding_warnings"][0]

    def test_repo_grounding_caps_time_budget_and_skips_unsafe_roots(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "README.md").write_text("Repo readme\n", encoding="utf-8")

        result = collect_repo_grounding(
            repositories=[{"path": str(repo)}],
            vision_brief="Read a repo",
            time_budget_seconds=99,
            allowed_roots=[tmp_path / "other"],
        )

        assert result["grounding"]["time_budget_seconds"] == 10.0
        assert result["provenance"] == []
        assert result["warnings"]
        assert "outside allowed grounding roots" in result["warnings"][0]

    @pytest.mark.asyncio
    async def test_dev_clarification_synthesis_fallback_marks_criteria_manual(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch("gateway.dev_control.clarifications.call_llm", return_value=_fake_clarification_response()):
                start = await cli.post("/v1/dev/clarifications", json={
                    "vision_brief": "Clarify a fallback synthesis path.",
                    "project_context": {"project_name": "Oryn Workspace"},
                })
            clarification = await start.json()
            for question in clarification["questions"]:
                await cli.post(f"/v1/dev/clarifications/{clarification['clarification_id']}/answer", json={
                    "question_id": question["question_id"],
                    "option_id": question["recommended_option_id"],
                })
            with patch("gateway.dev_control.clarifications.call_llm", side_effect=RuntimeError("synthesis unavailable")):
                complete = await cli.post(f"/v1/dev/clarifications/{clarification['clarification_id']}/complete", json={})
            assert complete.status == 200
            completed = await complete.json()

        assert "LLM clarified brief synthesis failed" in completed["warning"]
        criteria = completed["clarified_brief"]["acceptance_criteria"]
        assert criteria
        assert all(item["machine_checkable"] is False for item in criteria)
        assert all(item["verification_method"] == "manual" for item in criteria)

    @pytest.mark.asyncio
    async def test_dev_plan_artifact_lifecycle_versions_and_approves_without_execution_plan(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch("gateway.dev_control.clarifications.call_llm", return_value=_fake_clarification_response()):
                start = await cli.post("/v1/dev/clarifications", json={
                    "vision_brief": "Build a durable planning artifact review flow.",
                })
            assert start.status == 200
            clarification = await start.json()
            for question in clarification["questions"]:
                answer = await cli.post(f"/v1/dev/clarifications/{clarification['clarification_id']}/answer", json={
                    "question_id": question["question_id"],
                    "option_id": question["recommended_option_id"],
                })
                assert answer.status == 200

            with patch("gateway.dev_control.clarifications.call_llm", return_value=_fake_clarified_brief_response()):
                complete = await cli.post(f"/v1/dev/clarifications/{clarification['clarification_id']}/complete", json={})
            assert complete.status == 200

            with patch("gateway.dev_control.plan_artifacts.call_llm", return_value=_fake_plan_artifact_response()):
                created_resp = await cli.post("/v1/dev/plan-artifacts", json={
                    "clarification_id": clarification["clarification_id"],
                })
            assert created_resp.status == 200
            created = await created_resp.json()

            assert created["object"] == "hermes.dev_plan_artifact"
            assert created["status"] == "draft"
            assert created["version"] == 1
            assert created["source"] == "llm"
            assert created["payload"]["implementation_slices"]
            assert created["payload"]["acceptance_criteria"][0]["statement"]
            assert created["payload"]["acceptance_criteria"][0]["machine_checkable"] is True

            with patch("gateway.dev_control.plan_artifacts.call_llm", return_value=_fake_plan_artifact_response("Revised Planning Artifact")):
                revised_resp = await cli.post(f"/v1/dev/plan-artifacts/{created['plan_artifact_id']}/revise", json={
                    "feedback_instruction": "Make the panel workflow more explicit.",
                })
            assert revised_resp.status == 200
            revised = await revised_resp.json()
            assert revised["version"] == 2
            assert revised["title"] == "Revised Planning Artifact"

            old_resp = await cli.get(f"/v1/dev/plan-artifacts/{created['plan_artifact_id']}")
            assert old_resp.status == 200
            old = await old_resp.json()
            assert old["status"] == "superseded"
            assert old["superseded_by"] == revised["plan_artifact_id"]

            approved_resp = await cli.post(f"/v1/dev/plan-artifacts/{revised['plan_artifact_id']}/approve", json={})
            assert approved_resp.status == 200
            approved = await approved_resp.json()
            assert approved["status"] == "approved"
            assert approved["approved_at"] is not None
            assert adapter._dev_execution_store.list_plans(limit=10) == []

    @pytest.mark.asyncio
    async def test_dev_plan_artifact_create_execution_plan_requires_approval(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch("gateway.dev_control.clarifications.call_llm", return_value=_fake_clarification_response()):
                start = await cli.post("/v1/dev/clarifications", json={
                    "vision_brief": "Build artifact to Dev plan conversion.",
                })
            clarification = await start.json()
            for question in clarification["questions"]:
                await cli.post(f"/v1/dev/clarifications/{clarification['clarification_id']}/answer", json={
                    "question_id": question["question_id"],
                    "option_id": question["recommended_option_id"],
                })
            with patch("gateway.dev_control.clarifications.call_llm", return_value=_fake_clarified_brief_response()):
                await cli.post(f"/v1/dev/clarifications/{clarification['clarification_id']}/complete", json={})

            with patch("gateway.dev_control.plan_artifacts.call_llm", return_value=_fake_plan_artifact_response()):
                artifact_resp = await cli.post("/v1/dev/plan-artifacts", json={
                    "clarification_id": clarification["clarification_id"],
                })
            artifact = await artifact_resp.json()

            build_resp = await cli.post(f"/v1/dev/plan-artifacts/{artifact['plan_artifact_id']}/create-execution-plan", json={})

        assert build_resp.status == 400
        assert adapter._dev_execution_store.list_plans(limit=10) == []

    @pytest.mark.asyncio
    async def test_dev_plan_artifact_create_execution_plan_creates_planned_plan_and_build_record(self, adapter, tmp_path):
        db_path = tmp_path / "state.db"
        adapter._dev_execution_store = DevExecutionStore(db_path)
        adapter._subagent_event_store = SubagentEventStore(db_path)
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch("gateway.dev_control.clarifications.call_llm", return_value=_fake_clarification_response()):
                start = await cli.post("/v1/dev/clarifications", json={
                    "vision_brief": "Build artifact to Dev plan conversion.",
                    "project_id": "OrynPlatform",
                    "project_context": {
                        "project_id": "OrynPlatform",
                        "project_name": "Oryn Platform",
                        "vision": "Keep platform work isolated from workspace work.",
                    },
                })
            clarification = await start.json()
            for question in clarification["questions"]:
                await cli.post(f"/v1/dev/clarifications/{clarification['clarification_id']}/answer", json={
                    "question_id": question["question_id"],
                    "option_id": question["recommended_option_id"],
                })
            with patch("gateway.dev_control.clarifications.call_llm", return_value=_fake_clarified_brief_response()):
                await cli.post(f"/v1/dev/clarifications/{clarification['clarification_id']}/complete", json={})

            with patch("gateway.dev_control.plan_artifacts.call_llm", return_value=_fake_plan_artifact_response()):
                artifact_resp = await cli.post("/v1/dev/plan-artifacts", json={
                    "clarification_id": clarification["clarification_id"],
                })
            artifact = await artifact_resp.json()
            approved_resp = await cli.post(f"/v1/dev/plan-artifacts/{artifact['plan_artifact_id']}/approve", json={})
            approved = await approved_resp.json()

            with patch("gateway.dev_control.plan_artifacts.call_llm", return_value=_fake_artifact_execution_tasks_response("OrynPlatform")):
                build_resp = await cli.post(f"/v1/dev/plan-artifacts/{approved['plan_artifact_id']}/create-execution-plan", json={})
            assert build_resp.status == 200
            build = await build_resp.json()

            builds_resp = await cli.get(f"/v1/dev/plan-artifacts/{approved['plan_artifact_id']}/builds")
            assert builds_resp.status == 200
            builds = await builds_resp.json()
            artifact_list_resp = await cli.get("/v1/dev/plan-artifacts?project_id=OrynPlatform")
            other_artifact_list_resp = await cli.get("/v1/dev/plan-artifacts?project_id=OrynWorkspace")
            plans_for_project_resp = await cli.get("/v1/dev/execution-plans?project_id=OrynPlatform")
            plans_for_other_project_resp = await cli.get("/v1/dev/execution-plans?project_id=OrynWorkspace")
            artifact_list = await artifact_list_resp.json()
            other_artifact_list = await other_artifact_list_resp.json()
            plans_for_project = await plans_for_project_resp.json()
            plans_for_other_project = await plans_for_other_project_resp.json()
            draft_resp = await cli.get(f"/v1/dev/execution-plans/{build['plan_id']}/draft-review")
            assert draft_resp.status == 200
            draft = await draft_resp.json()
            launch_resp = await cli.post(f"/v1/dev/execution-plans/{build['plan_id']}/launch", json={})
            draft_approve_resp = await cli.post(f"/v1/dev/execution-plans/{build['plan_id']}/approve-draft", json={})
            assert draft_approve_resp.status == 200
            fake_router = _FakeRuntimeRouter()
            with patch("gateway.dev_execution._ensure_runtime_router", return_value=fake_router):
                approved_launch_resp = await cli.post(f"/v1/dev/execution-plans/{build['plan_id']}/launch", json={})
            assert approved_launch_resp.status == 200
            approved_launch = await approved_launch_resp.json()
            launched_draft_resp = await cli.get(f"/v1/dev/execution-plans/{build['plan_id']}/draft-review")
            assert launched_draft_resp.status == 200
            launched_draft = await launched_draft_resp.json()

        assert build["object"] == "hermes.dev_plan_artifact_build"
        assert build["plan_artifact_id"] == approved["plan_artifact_id"]
        assert build["status"] == "created"
        assert build["source"] == "llm"
        assert build["task_count"] == 2
        assert approved["project_id"] == "OrynPlatform"
        assert build["plan"]["status"] == "planned"
        assert build["plan"]["plan_id"] == build["plan_id"]
        assert len(build["plan"]["tasks"]) == 2
        assert {task["project_id"] for task in build["plan"]["tasks"]} == {"OrynPlatform"}
        assert build["plan"]["tasks"][0]["status"] == "planned"
        assert build["plan"]["tasks"][0]["profile_id"] == "workspace.implement"
        assert build["plan"]["tasks"][1]["profile_id"] == "workspace.inspect"
        assert build["plan"]["tasks"][0].get("ao_session_id") is None
        assert build["plan"]["tasks"][0].get("runtime_session_id") is None
        assert "DEV_WORKER_EVIDENCE" in build["plan"]["tasks"][0]["prompt"]
        assert isinstance(build["plan"]["tasks"][0]["acceptance_criteria"][0], str)
        assert builds["total"] == 1
        assert builds["data"][0]["plan_id"] == build["plan_id"]
        assert artifact_list["total"] == 1
        assert other_artifact_list["total"] == 0
        assert plans_for_project["total"] == 1
        assert plans_for_other_project["total"] == 0
        assert draft["draft_status"] == "draft"
        assert draft["version"] == 1
        assert draft["plan_artifact_id"] == approved["plan_artifact_id"]
        assert draft["build_id"] == build["build_id"]
        assert launch_resp.status == 400
        assert approved_launch["ok"] is True
        assert len(approved_launch["launched"]) == 2
        assert approved_launch["launched"][0]["runtime"] == "ao"
        assert approved_launch["launched"][0]["runtime_session_id"] == "oryn-workspace-test-1"
        assert approved_launch["launch_record"]["plan_id"] == build["plan_id"]
        assert approved_launch["launch_record"]["launch_scope"] == "all"
        assert approved_launch["launch_record"]["requested_task_ids"] == []
        assert approved_launch["launch_record"]["launched_count"] == 2
        assert approved_launch["launch_record"]["failure_count"] == 0
        assert approved_launch["plan"]["status"] == "launched"
        assert approved_launch["plan"]["tasks"][0]["status"] == "launched"
        assert launched_draft["launch_records"][0]["launch_id"] == approved_launch["launch_record"]["launch_id"]
        events = SubagentEventStore(db_path).list_events(limit=10)
        assert [event["event"] for event in events].count("subagent.start") == 2

    @pytest.mark.asyncio
    async def test_dev_execution_plan_draft_launch_records_smoke_subset(self, adapter, tmp_path):
        db_path = tmp_path / "state.db"
        adapter._dev_execution_store = DevExecutionStore(db_path)
        adapter._subagent_event_store = SubagentEventStore(db_path)
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch("gateway.dev_control.clarifications.call_llm", return_value=_fake_clarification_response()):
                start = await cli.post("/v1/dev/clarifications", json={
                    "vision_brief": "Build artifact smoke launch record.",
                })
            clarification = await start.json()
            for question in clarification["questions"]:
                await cli.post(f"/v1/dev/clarifications/{clarification['clarification_id']}/answer", json={
                    "question_id": question["question_id"],
                    "option_id": question["recommended_option_id"],
                })
            with patch("gateway.dev_control.clarifications.call_llm", return_value=_fake_clarified_brief_response()):
                await cli.post(f"/v1/dev/clarifications/{clarification['clarification_id']}/complete", json={})

            with patch("gateway.dev_control.plan_artifacts.call_llm", return_value=_fake_plan_artifact_response()):
                artifact_resp = await cli.post("/v1/dev/plan-artifacts", json={
                    "clarification_id": clarification["clarification_id"],
                })
            artifact = await artifact_resp.json()
            approved_resp = await cli.post(f"/v1/dev/plan-artifacts/{artifact['plan_artifact_id']}/approve", json={})
            approved = await approved_resp.json()

            with patch("gateway.dev_control.plan_artifacts.call_llm", return_value=_fake_artifact_execution_tasks_response()):
                build_resp = await cli.post(f"/v1/dev/plan-artifacts/{approved['plan_artifact_id']}/create-execution-plan", json={})
            build = await build_resp.json()
            task_id = build["plan"]["tasks"][0]["task_id"]

            approve_resp = await cli.post(f"/v1/dev/execution-plans/{build['plan_id']}/approve-draft", json={})
            assert approve_resp.status == 200
            fake_router = _FakeRuntimeRouter()
            with patch("gateway.dev_execution._ensure_runtime_router", return_value=fake_router):
                launch_resp = await cli.post(f"/v1/dev/execution-plans/{build['plan_id']}/launch", json={
                    "task_ids": [task_id],
                })
            assert launch_resp.status == 200
            launch = await launch_resp.json()
            draft_resp = await cli.get(f"/v1/dev/execution-plans/{build['plan_id']}/draft-review")
            draft = await draft_resp.json()

        assert launch["launch_record"]["launch_scope"] == "smoke"
        assert launch["launch_record"]["requested_task_ids"] == [task_id]
        assert launch["launch_record"]["launched_task_ids"] == [task_id]
        assert launch["launch_record"]["failed_task_ids"] == []
        assert launch["launch_record"]["launched_count"] == 1
        assert launch["launch_record"]["failure_count"] == 0
        assert draft["launch_records"][0]["launch_id"] == launch["launch_record"]["launch_id"]
        assert draft["plan"]["tasks"][0]["status"] == "launched"
        assert draft["plan"]["tasks"][1]["status"] == "planned"

    @pytest.mark.asyncio
    async def test_dev_verification_runs_api_launches_verify_worker_and_surfaces_read_model(self, adapter, tmp_path):
        db_path = tmp_path / "state.db"
        adapter._dev_execution_store = DevExecutionStore(db_path)
        adapter._subagent_event_store = SubagentEventStore(db_path)
        adapter._dev_verification_store = DevVerificationStore(db_path)
        criteria = acceptance_criteria_to_strings([{
            "statement": "Verification API can run a worker.",
            "verification_method": "test",
            "verification_detail": "scripts/run_tests.sh tests/gateway/test_acceptance_verification.py",
            "machine_checkable": True,
        }])
        plan = adapter._dev_execution_store.create_plan(
            title="Verify via API",
            vision_brief=None,
            tasks=[{
                "goal": "Implemented feature",
                "prompt": "Implement the feature.",
                "profile_id": "workspace.implement",
                "project_id": "OrynWorkspace",
                "permissions": "edit",
                "acceptance_criteria": criteria,
            }],
        )
        task_id = plan["tasks"][0]["task_id"]
        from gateway.dev_execution import set_execution_plan_test_state

        set_execution_plan_test_state(
            store=adapter._dev_execution_store,
            plan_id=plan["plan_id"],
            task_id=task_id,
            state="completed_ok",
            event_store=adapter._subagent_event_store,
            ao_session_id="implemented-api-session",
            files_written=["gateway/dev_control/example.py"],
        )
        fake_router = _FakeRuntimeRouter()
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch("gateway.dev_control.acceptance_verification._runtime_router", return_value=fake_router):
                start_resp = await cli.post("/v1/dev/verification-runs", json={
                    "plan_id": plan["plan_id"],
                    "task_id": task_id,
                })
            assert start_resp.status == 200
            started = await start_resp.json()
            run_id = started["verification_run_id"]
            adapter._subagent_event_store.append_event({
                "event": "subagent.complete",
                "subagent_id": f"ao:{started['verification_session_id']}",
                "ao_session_id": started["verification_session_id"],
                "status": "completed",
                "summary": (
                    "Verification finished\n"
                    "```json DEV_VERIFICATION_RESULTS\n"
                    + json.dumps({
                        "object": "hermes.dev_verification_results",
                        "results": [{
                            "criterion_id": "crit-1",
                            "command_run": started["executable_commands"][0]["command"],
                            "exit_code": 0,
                            "output_excerpt": "1 passed in 0.1s",
                            "notes": "",
                        }],
                    })
                    + "\n```"
                ),
            })
            detail_resp = await cli.get(f"/v1/dev/verification-runs/{run_id}")
            plan_resp = await cli.get(f"/v1/dev/execution-plans/{plan['plan_id']}")
            detail = await detail_resp.json()
            plan_payload = await plan_resp.json()

        assert started["worker_launch_profile_id"] == "workspace.test"
        assert started["status"] == "launched"
        assert fake_router.spawned[0]["runtime"] == "ao"
        assert "DEV_VERIFICATION_RESULTS" in fake_router.spawned[0]["kwargs"]["prompt"]
        assert detail["status"] == "completed"
        assert detail["verdict"] == "verified"
        task = plan_payload["plan"]["tasks"][0]
        assert task["acceptance_verification"]["verdict"] == "verified"
        assert plan_payload["plan"]["acceptance_verification"]["counts"]["passed"] == 1

    @pytest.mark.asyncio
    async def test_dev_execution_plan_draft_can_revise_and_approve_without_launch(self, adapter, tmp_path):
        db_path = tmp_path / "state.db"
        adapter._dev_execution_store = DevExecutionStore(db_path)
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch("gateway.dev_control.clarifications.call_llm", return_value=_fake_clarification_response()):
                start = await cli.post("/v1/dev/clarifications", json={
                    "vision_brief": "Build artifact draft review.",
                })
            clarification = await start.json()
            for question in clarification["questions"]:
                await cli.post(f"/v1/dev/clarifications/{clarification['clarification_id']}/answer", json={
                    "question_id": question["question_id"],
                    "option_id": question["recommended_option_id"],
                })
            with patch("gateway.dev_control.clarifications.call_llm", return_value=_fake_clarified_brief_response()):
                await cli.post(f"/v1/dev/clarifications/{clarification['clarification_id']}/complete", json={})
            with patch("gateway.dev_control.plan_artifacts.call_llm", return_value=_fake_plan_artifact_response()):
                artifact_resp = await cli.post("/v1/dev/plan-artifacts", json={
                    "clarification_id": clarification["clarification_id"],
                })
            artifact = await artifact_resp.json()
            approved_resp = await cli.post(f"/v1/dev/plan-artifacts/{artifact['plan_artifact_id']}/approve", json={})
            approved = await approved_resp.json()
            with patch("gateway.dev_control.plan_artifacts.call_llm", return_value=_fake_artifact_execution_tasks_response()):
                build_resp = await cli.post(f"/v1/dev/plan-artifacts/{approved['plan_artifact_id']}/create-execution-plan", json={})
            build = await build_resp.json()

            revised_payload = {
                "tasks": [
                    {
                        "goal": "Implement only the review gate",
                        "prompt": "Reduce the draft to the review gate only.",
                        "profile_id": "workspace.implement",
                        "project_id": "OrynWorkspace",
                        "permissions": "edit",
                        "dependencies": [],
                        "acceptance_criteria": ["Review gate is visible."],
                    }
                ]
            }
            with patch("gateway.dev_control.plan_artifacts.call_llm", return_value=_fake_text_response(json.dumps(revised_payload))):
                revise_resp = await cli.post(f"/v1/dev/execution-plans/{build['plan_id']}/revise-draft", json={
                    "feedback_instruction": "Reduce to one task focused on the review gate.",
                })
            assert revise_resp.status == 200
            revised = await revise_resp.json()
            approve_resp = await cli.post(f"/v1/dev/execution-plans/{build['plan_id']}/approve-draft", json={})
            assert approve_resp.status == 200
            draft_approved = await approve_resp.json()

        assert revised["draft_status"] == "revision_requested"
        assert revised["version"] == 2
        assert len(revised["revision_history"]) == 1
        assert len(revised["plan"]["tasks"]) == 1
        assert revised["plan"]["tasks"][0]["goal"] == "Implement only the review gate"
        assert "DEV_WORKER_EVIDENCE" in revised["plan"]["tasks"][0]["prompt"]
        assert draft_approved["draft_status"] == "approved_for_launch"
        assert draft_approved["approved_at"] is not None
        assert SubagentEventStore(db_path).list_events(limit=10) == []

    @pytest.mark.asyncio
    async def test_dev_execution_plan_draft_revise_falls_back_when_llm_invalid(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch("gateway.dev_control.clarifications.call_llm", return_value=_fake_clarification_response()):
                start = await cli.post("/v1/dev/clarifications", json={
                    "vision_brief": "Build artifact draft review fallback.",
                })
            clarification = await start.json()
            for question in clarification["questions"]:
                await cli.post(f"/v1/dev/clarifications/{clarification['clarification_id']}/answer", json={
                    "question_id": question["question_id"],
                    "option_id": question["recommended_option_id"],
                })
            with patch("gateway.dev_control.clarifications.call_llm", return_value=_fake_clarified_brief_response()):
                await cli.post(f"/v1/dev/clarifications/{clarification['clarification_id']}/complete", json={})
            with patch("gateway.dev_control.plan_artifacts.call_llm", return_value=_fake_plan_artifact_response()):
                artifact_resp = await cli.post("/v1/dev/plan-artifacts", json={
                    "clarification_id": clarification["clarification_id"],
                })
            artifact = await artifact_resp.json()
            approved_resp = await cli.post(f"/v1/dev/plan-artifacts/{artifact['plan_artifact_id']}/approve", json={})
            approved = await approved_resp.json()
            with patch("gateway.dev_control.plan_artifacts.call_llm", return_value=_fake_artifact_execution_tasks_response()):
                build_resp = await cli.post(f"/v1/dev/plan-artifacts/{approved['plan_artifact_id']}/create-execution-plan", json={})
            build = await build_resp.json()

            with patch("gateway.dev_control.plan_artifacts.call_llm", side_effect=[
                _fake_text_response("not json"),
                _fake_text_response('{"tasks":[{"goal":"broken"'),
            ]):
                revise_resp = await cli.post(f"/v1/dev/execution-plans/{build['plan_id']}/revise-draft", json={
                    "feedback": "Make fallback deterministic.",
                })
            assert revise_resp.status == 200
            revised = await revise_resp.json()

        assert revised["source"] == "fallback"
        assert revised["version"] == 2
        assert "LLM task drafting failed" in revised["payload"]["draft_warning"]
        assert len(revised["plan"]["tasks"]) == 3

    @pytest.mark.asyncio
    async def test_dev_plan_artifact_create_execution_plan_falls_back_when_task_llm_invalid(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch("gateway.dev_control.clarifications.call_llm", return_value=_fake_clarification_response()):
                start = await cli.post("/v1/dev/clarifications", json={
                    "vision_brief": "Build artifact to Dev plan conversion with fallback.",
                })
            clarification = await start.json()
            for question in clarification["questions"]:
                await cli.post(f"/v1/dev/clarifications/{clarification['clarification_id']}/answer", json={
                    "question_id": question["question_id"],
                    "option_id": question["recommended_option_id"],
                })
            with patch("gateway.dev_control.clarifications.call_llm", return_value=_fake_clarified_brief_response()):
                await cli.post(f"/v1/dev/clarifications/{clarification['clarification_id']}/complete", json={})

            with patch("gateway.dev_control.plan_artifacts.call_llm", return_value=_fake_plan_artifact_response()):
                artifact_resp = await cli.post("/v1/dev/plan-artifacts", json={
                    "clarification_id": clarification["clarification_id"],
                })
            artifact = await artifact_resp.json()
            approved_resp = await cli.post(f"/v1/dev/plan-artifacts/{artifact['plan_artifact_id']}/approve", json={})
            approved = await approved_resp.json()

            with patch("gateway.dev_control.plan_artifacts.call_llm", side_effect=[
                _fake_text_response("not json"),
                _fake_text_response('{"tasks":[{"goal":"broken"'),
            ]):
                build_resp = await cli.post(f"/v1/dev/plan-artifacts/{approved['plan_artifact_id']}/create-execution-plan", json={})
            assert build_resp.status == 200
            build = await build_resp.json()

        assert build["source"] == "fallback"
        assert "LLM task drafting failed" in build["payload"]["draft_warning"]
        assert build["task_count"] == 3
        assert [task["profile_id"] for task in build["plan"]["tasks"]] == [
            "workspace.implement",
            "workspace.implement",
            "workspace.inspect",
        ]

    @pytest.mark.asyncio
    async def test_dev_plan_artifact_rejects_incomplete_clarification(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch("gateway.dev_control.clarifications.call_llm", return_value=_fake_clarification_response()):
                start = await cli.post("/v1/dev/clarifications", json={
                    "vision_brief": "Build a planning artifact.",
                })
            clarification = await start.json()
            created_resp = await cli.post("/v1/dev/plan-artifacts", json={
                "clarification_id": clarification["clarification_id"],
            })

        assert created_resp.status == 400

    @pytest.mark.asyncio
    async def test_dev_plan_artifact_falls_back_when_llm_output_invalid(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch("gateway.dev_control.clarifications.call_llm", return_value=_fake_clarification_response()):
                start = await cli.post("/v1/dev/clarifications", json={
                    "vision_brief": "Build a planning artifact fallback.",
                })
            clarification = await start.json()
            for question in clarification["questions"]:
                await cli.post(f"/v1/dev/clarifications/{clarification['clarification_id']}/answer", json={
                    "question_id": question["question_id"],
                    "option_id": question["recommended_option_id"],
                })
            with patch("gateway.dev_control.clarifications.call_llm", return_value=_fake_clarified_brief_response()):
                await cli.post(f"/v1/dev/clarifications/{clarification['clarification_id']}/complete", json={})

            with patch("gateway.dev_control.plan_artifacts.call_llm", side_effect=[
                _fake_text_response("not json"),
                _fake_text_response('{"title":"broken"'),
            ]):
                created_resp = await cli.post("/v1/dev/plan-artifacts", json={
                    "clarification_id": clarification["clarification_id"],
                })
            assert created_resp.status == 200
            created = await created_resp.json()

        assert created["source"] == "fallback"
        assert "LLM plan artifact generation failed" in created["warning"]
        assert created["payload"]["implementation_slices"]

    @pytest.mark.asyncio
    async def test_dev_plan_artifact_derives_missing_llm_title(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch("gateway.dev_control.clarifications.call_llm", return_value=_fake_clarification_response()):
                start = await cli.post("/v1/dev/clarifications", json={
                    "vision_brief": "Build a planning artifact with a recovered title.",
                })
            clarification = await start.json()
            for question in clarification["questions"]:
                await cli.post(f"/v1/dev/clarifications/{clarification['clarification_id']}/answer", json={
                    "question_id": question["question_id"],
                    "option_id": question["recommended_option_id"],
                })
            with patch("gateway.dev_control.clarifications.call_llm", return_value=_fake_clarified_brief_response()):
                await cli.post(f"/v1/dev/clarifications/{clarification['clarification_id']}/complete", json={})

            payload = json.loads(_fake_plan_artifact_response("Ignored").choices[0].message.content)
            payload.pop("title", None)
            payload["overview"] = "Create a release gate that only offers publish when checks pass."
            with patch("gateway.dev_control.plan_artifacts.call_llm", return_value=_fake_text_response(json.dumps(payload))):
                created_resp = await cli.post("/v1/dev/plan-artifacts", json={
                    "clarification_id": clarification["clarification_id"],
                })
            assert created_resp.status == 200
            created = await created_resp.json()

        assert created["source"] == "llm"
        assert created.get("warning") is None
        assert created["title"] == "a release gate that only offers publish when checks pass"
        assert created["payload"]["title"] == created["title"]

    @pytest.mark.asyncio
    async def test_dev_clarification_invalid_llm_json_retries_once(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch("gateway.dev_control.clarifications.call_llm", side_effect=[
                _fake_text_response("not json"),
                _fake_clarification_response(),
            ]) as call_mock:
                resp = await cli.post("/v1/dev/clarifications", json={
                    "vision_brief": "Clarify a benchmark improvement idea.",
                })
            assert resp.status == 200
            data = await resp.json()

        assert data["status"] == "active"
        assert len(data["questions"]) == 3
        assert call_mock.call_args_list[0].kwargs["extra_body"]["response_format"]["type"] == "json_schema"

    @pytest.mark.asyncio
    async def test_dev_clarification_retries_without_schema_mode_when_provider_rejects_it(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch("gateway.dev_control.clarifications.call_llm", side_effect=[
                RuntimeError("unsupported response_format json_schema"),
                _fake_clarification_response(),
            ]) as call_mock:
                resp = await cli.post("/v1/dev/clarifications", json={
                    "vision_brief": "Clarify a planning-mode improvement idea.",
                })
            assert resp.status == 200
            data = await resp.json()

        assert data["status"] == "active"
        assert data["generation_mode"] == "llm"
        assert call_mock.call_count == 2
        assert "extra_body" in call_mock.call_args_list[0].kwargs
        assert "extra_body" not in call_mock.call_args_list[1].kwargs

    @pytest.mark.asyncio
    async def test_dev_clarification_uses_fallback_questions_when_llm_json_remains_invalid(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch("gateway.dev_control.clarifications.call_llm", side_effect=[
                _fake_text_response("not json"),
                _fake_text_response('{"questions":[{"question_id":"q1","prompt":"broken"'),
            ]):
                resp = await cli.post("/v1/dev/clarifications", json={
                    "vision_brief": "Clarify a planning-mode improvement idea.",
                })
            assert resp.status == 200
            data = await resp.json()

        assert data["status"] == "active"
        assert data["generation_mode"] == "fallback"
        assert "LLM question generation failed" in data["warning"]
        assert len(data["questions"]) == 5
        assert data["current_question"]["question_id"] == "q1"

    @pytest.mark.asyncio
    async def test_dev_runtime_selection_falls_back_when_openhands_benchmark_degraded(self, adapter, tmp_path):
        db_path = tmp_path / "state.db"
        adapter._dev_execution_store = DevExecutionStore(db_path)
        _persist_runtime_policy_benchmark(
            db_path,
            benchmark_run_id="devbench-degraded",
            openhands={"marker_pass_rate": 0.5},
        )
        app = _create_runs_app(adapter)
        openhands = _FakeOpenHandsBridge()

        async with TestClient(TestServer(app)) as cli:
            with patch("tools.openhands_bridge.OpenHandsBridge", return_value=openhands):
                resp = await cli.post("/v1/dev/runtime-selection", json={
                    "runtime": "auto",
                    "goal": "Inspect Agent Board state",
                    "prompt": "Inspect read-only.",
                    "permissions": "read_only",
                })
                assert resp.status == 200
                data = await resp.json()

        assert data["selected_runtime"] == "ao"
        assert data["selection_mode"] == "fallback"
        assert data["runtime_policy_status"] == "degraded"
        assert "marker pass rate" in data["runtime_policy_reason"]
        assert "benchmark evidence degraded" in data["runtime_fallback_reason"]

    @pytest.mark.asyncio
    async def test_dev_runtime_selection_falls_back_to_ao_when_openhands_unavailable(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)
        openhands = _FakeOpenHandsBridge(launch_supported=False)

        async with TestClient(TestServer(app)) as cli:
            with patch("tools.openhands_bridge.OpenHandsBridge", return_value=openhands):
                resp = await cli.post("/v1/dev/runtime-selection", json={
                    "runtime": "auto",
                    "goal": "Inspect Agent Board state",
                    "prompt": "Inspect read-only.",
                    "permissions": "read_only",
                })
                assert resp.status == 200
                data = await resp.json()

        assert data["selected_runtime"] == "ao"
        assert data["selection_mode"] == "fallback"
        assert data["runtime_fallback_reason"] == "OpenHands is not installed or configured."
        assert data["runtime_policy_status"] == "fallback"

    @pytest.mark.asyncio
    async def test_dev_runtime_selection_uses_ao_for_auto_implementation(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)
        openhands = _FakeOpenHandsBridge()

        async with TestClient(TestServer(app)) as cli:
            with patch("tools.openhands_bridge.OpenHandsBridge", return_value=openhands):
                resp = await cli.post("/v1/dev/runtime-selection", json={
                    "runtime": "auto",
                    "goal": "Implement Agent Board changes",
                    "prompt": "Modify the SwiftUI view and add tests.",
                    "permissions": "edit",
                })
                assert resp.status == 200
                data = await resp.json()

        assert data["selected_runtime"] == "ao"
        assert data["selection_mode"] == "auto"
        assert "Implementation" in data["reason"]
        assert data["runtime_policy_status"] == "not_applicable"

    @pytest.mark.asyncio
    async def test_dev_harness_components_endpoint_returns_active_component_hashes(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/dev/harness/components")
            assert resp.status == 200
            data = await resp.json()

        component_ids = {component["component_id"] for component in data["data"]}
        assert "runtime-selection-policy" in component_ids
        assert "launch-profiles" in component_ids
        assert "runtime-adapters" in component_ids
        assert "summary-quality-classifier" in component_ids
        for component in data["data"]:
            assert component["version_hash"]
            assert component["kind"]
            assert component["source"]

    @pytest.mark.asyncio
    async def test_dev_harness_report_summarizes_weak_fixture_and_persists(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Harness weak fixture",
                "tasks": [{"goal": "Fixture weak", "prompt": "Return unclear.", "project_id": "OrynWorkspace"}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            state_resp = await cli.post(f"/v1/dev/execution-plans/{plan_id}/test-state", json={
                "task_id": task_id,
                "state": "completed_weak",
                "summary": "unclear",
            })
            assert state_resp.status == 200
            report_resp = await cli.post("/v1/dev/harness/report", json={"plan_ids": [plan_id]})
            assert report_resp.status == 200
            report = await report_resp.json()

        assert report["object"] == "hermes.dev_harness_report"
        assert report["summary"]["plan_count"] == 1
        assert report["summary"]["task_count"] == 1
        assert report["summary"]["weak_summary_count"] == 1
        assert report["summary"]["by_runtime"]["fixture"] == 1
        assert report["plan_observations"][0]["plan_id"] == plan_id
        assert report["plan_observations"][0]["tasks"][0]["summary_quality"] == "warning"
        assert any(pattern["pattern"] == "weak_or_missing_summary" for pattern in report["failure_patterns"])
        assert any(item["plan_id"] == plan_id and item["task_id"] == task_id for item in report["evidence"])

        row = adapter._dev_execution_store._conn.execute(
            "SELECT payload FROM dev_harness_reports WHERE report_id = ?",
            (report["report_id"],),
        ).fetchone()
        assert row is not None
        persisted = json.loads(row["payload"])
        assert persisted["report_id"] == report["report_id"]

    @pytest.mark.asyncio
    async def test_dev_harness_report_tracks_runtime_fallback_reason(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        openhands = _FakeOpenHandsBridge(spawn_error=RuntimeError("server refused launch"))
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Harness fallback",
                "tasks": [{
                    "goal": "Inspect runtime fallback",
                    "prompt": "Inspect read-only and report.",
                    "runtime": "auto",
                    "project_id": "OrynWorkspace",
                    "permissions": "read_only",
                }],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]

            with patch("tools.ao_bridge.AOBridge", return_value=bridge), \
                 patch("tools.openhands_bridge.OpenHandsBridge", return_value=openhands):
                launch_resp = await cli.post(f"/v1/dev/execution-plans/{plan_id}/launch")
                assert launch_resp.status == 200
                launch_data = await launch_resp.json()
                report_resp = await cli.post("/v1/dev/harness/report", json={"plan_ids": [plan_id]})
                assert report_resp.status == 200
                report = await report_resp.json()

        assert launch_data["launched"][0]["runtime"] == "ao"
        assert "OpenHands launch failed" in launch_data["launched"][0]["runtime_fallback_reason"]
        assert report["summary"]["fallback_count"] == 1
        assert any(pattern["pattern"] == "runtime_fallback" for pattern in report["failure_patterns"])
        ao_runtime = next(item for item in report["runtime_observations"] if item["runtime"] == "ao")
        assert ao_runtime["fallbacks"] == 1

    @pytest.mark.asyncio
    async def test_dev_harness_recommendations_empty_for_clean_report(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Clean harness recommendation source",
                "tasks": [{"goal": "Return PHASE23_CLEAN_DONE", "prompt": "Return PHASE23_CLEAN_DONE."}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            state_resp = await cli.post(f"/v1/dev/execution-plans/{plan_id}/test-state", json={
                "task_id": task_id,
                "state": "completed_ok",
                "summary": "PHASE23_CLEAN_DONE Verified clean completion with evidence.",
            })
            assert state_resp.status == 200
            report_resp = await cli.post("/v1/dev/harness/report", json={"plan_ids": [plan_id]})
            report = await report_resp.json()
            rec_resp = await cli.post("/v1/dev/harness/recommendations", json={"report_id": report["report_id"]})
            assert rec_resp.status == 200
            rec_data = await rec_resp.json()

        assert rec_data["object"] == "hermes.dev_harness_recommendation_run"
        assert rec_data["report_id"] == report["report_id"]
        assert rec_data["recommendations"] == []
        assert rec_data["summary"]["recommendation_count"] == 0

    @pytest.mark.asyncio
    async def test_dev_harness_recommendations_prompt_echo_and_marker_are_advisory(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Prompt recommendation source",
                "tasks": [
                    {
                        "goal": "Inspect board and end with PHASE23_MARKER_DONE",
                        "prompt": "Inspect board and end with PHASE23_MARKER_DONE.",
                    },
                    {"goal": "Inspect prompt echo one", "prompt": "Inspect files."},
                    {"goal": "Inspect prompt echo two", "prompt": "Inspect files."},
                ],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            tasks = create_data["plan"]["tasks"]
            for idx, task in enumerate(tasks):
                session_id = f"fixture-phase23-{idx}"
                adapter._dev_execution_store.update_task_launch(
                    plan_id=plan_id,
                    task_id=task["task_id"],
                    ao_session_id=session_id,
                )
                summary = (
                    "Inspection completed successfully without the literal requested marker."
                    if idx == 0
                    else "## Hermes AO Delegation Contract\nTask Brief: inspect files and summarize."
                )
                adapter._subagent_event_store.append_event({
                    "event": "subagent.complete",
                    "subagent_id": f"fixture:{session_id}",
                    "ao_session_id": session_id,
                    "runtime": "fixture",
                    "status": "completed",
                    "summary": summary,
                    "launch_plan_id": plan_id,
                    "launch_task_id": task["task_id"],
                    "goal": task["goal"],
                })
            report_resp = await cli.post("/v1/dev/harness/report", json={"plan_ids": [plan_id]})
            report = await report_resp.json()
            rec_resp = await cli.post("/v1/dev/harness/recommendations", json={"report_id": report["report_id"]})
            assert rec_resp.status == 200
            rec_data = await rec_resp.json()

        recommendations = {item["title"]: item for item in rec_data["recommendations"]}
        echo = recommendations["Separate worker contract from task brief to reduce prompt echo summaries"]
        marker = recommendations["Make completion marker requirements harder for workers to paraphrase"]
        assert echo["category"] == "prompt_template"
        assert echo["priority"] == "high"
        assert echo["status"] == "proposed"
        assert "Do not disable any runtime" in echo["non_goals"][0]
        assert marker["category"] == "prompt_template"
        assert marker["priority"] == "low"
        assert rec_data["summary"]["by_category"]["prompt_template"] == 2

        listed_resp_data = None
        async with TestClient(TestServer(app)) as cli:
            listed_resp = await cli.get(f"/v1/dev/harness/recommendations?report_id={report['report_id']}")
            listed_resp_data = await listed_resp.json()
            detail_resp = await cli.get(f"/v1/dev/harness/recommendations/{rec_data['recommendation_run_id']}")
            detail_data = await detail_resp.json()
        assert listed_resp_data["total"] == 1
        assert detail_data["recommendation_run_id"] == rec_data["recommendation_run_id"]

    @pytest.mark.asyncio
    async def test_dev_harness_recommendations_runtime_policy_requires_minimum_sample(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            small_plan = adapter._dev_execution_store.create_plan(
                title="Small fallback sample",
                vision_brief=None,
                tasks=[{"goal": "Fallback", "prompt": "Inspect."}],
            )
            small_task = small_plan["tasks"][0]
            adapter._dev_execution_store.update_task_launch(
                plan_id=small_plan["plan_id"],
                task_id=small_task["task_id"],
                ao_session_id="fixture-small-fallback",
            )
            adapter._subagent_event_store.append_event({
                "event": "subagent.complete",
                "subagent_id": "fixture:small",
                "ao_session_id": "fixture-small-fallback",
                "runtime": "fixture",
                "status": "completed",
                "summary": "PHASE23_SMALL_FALLBACK_DONE clean completion.",
                "runtime_fallback_reason": "OpenHands launch failed, falling back to AO.",
                "launch_plan_id": small_plan["plan_id"],
                "launch_task_id": small_task["task_id"],
            })
            small_report_resp = await cli.post("/v1/dev/harness/report", json={"plan_ids": [small_plan["plan_id"]]})
            small_report = await small_report_resp.json()
            small_rec_resp = await cli.post("/v1/dev/harness/recommendations", json={"report_id": small_report["report_id"]})
            small_rec = await small_rec_resp.json()

            large_plan = adapter._dev_execution_store.create_plan(
                title="Large fallback sample",
                vision_brief=None,
                tasks=[{"goal": f"Task {idx}", "prompt": "Inspect."} for idx in range(5)],
            )
            for idx, task in enumerate(large_plan["tasks"]):
                session_id = f"fixture-large-fallback-{idx}"
                adapter._dev_execution_store.update_task_launch(
                    plan_id=large_plan["plan_id"],
                    task_id=task["task_id"],
                    ao_session_id=session_id,
                )
                adapter._subagent_event_store.append_event({
                    "event": "subagent.complete",
                    "subagent_id": f"fixture:{idx}",
                    "ao_session_id": session_id,
                    "runtime": "fixture",
                    "status": "completed",
                    "summary": f"PHASE23_LARGE_FALLBACK_{idx}_DONE clean completion.",
                    "runtime_fallback_reason": "OpenHands launch failed, falling back to AO." if idx == 0 else None,
                    "launch_plan_id": large_plan["plan_id"],
                    "launch_task_id": task["task_id"],
                })
            large_report_resp = await cli.post("/v1/dev/harness/report", json={"plan_ids": [large_plan["plan_id"]]})
            large_report = await large_report_resp.json()
            large_rec_resp = await cli.post("/v1/dev/harness/recommendations", json={"report_id": large_report["report_id"]})
            large_rec = await large_rec_resp.json()

        assert all(item["category"] != "runtime_policy" for item in small_rec["recommendations"])
        runtime_recs = [item for item in large_rec["recommendations"] if item["category"] == "runtime_policy"]
        assert len(runtime_recs) == 1
        assert runtime_recs[0]["priority"] == "low"
        assert "Do not disable a runtime" in runtime_recs[0]["non_goals"][0]

    def test_dev_harness_recommendation_tools_match_api_shape(self, adapter, tmp_path):
        from tools.dev_execution_tools import (
            _handle_dev_harness_recommendation_runs,
            _handle_dev_harness_recommendations,
        )

        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        plan = adapter._dev_execution_store.create_plan(
            title="Tool recommendation source",
            vision_brief=None,
            tasks=[{"goal": "Return unclear", "prompt": "Return unclear."}],
        )
        task = plan["tasks"][0]
        adapter._dev_execution_store.update_task_launch(
            plan_id=plan["plan_id"],
            task_id=task["task_id"],
            ao_session_id="fixture-tool-rec",
        )
        adapter._subagent_event_store.append_event({
            "event": "subagent.complete",
            "subagent_id": "fixture:tool",
            "ao_session_id": "fixture-tool-rec",
            "runtime": "fixture",
            "status": "completed",
            "summary": "unclear",
            "launch_plan_id": plan["plan_id"],
            "launch_task_id": task["task_id"],
        })
        with patch("tools.dev_execution_tools.DevExecutionStore", return_value=adapter._dev_execution_store), \
             patch("tools.dev_execution_tools.SubagentEventStore", return_value=adapter._subagent_event_store):
            generated = json.loads(_handle_dev_harness_recommendations({"plan_ids": [plan["plan_id"]]}))
            listed = json.loads(_handle_dev_harness_recommendation_runs({}))
            fetched = json.loads(_handle_dev_harness_recommendation_runs({
                "recommendation_run_id": generated["recommendation_run_id"],
            }))

        assert generated["object"] == "hermes.dev_harness_recommendation_run"
        assert generated["summary"]["recommendation_count"] == 1
        assert listed["object"] == "list"
        assert listed["total"] == 1
        assert fetched["recommendation_run_id"] == generated["recommendation_run_id"]

    @pytest.mark.asyncio
    async def test_dev_harness_benchmark_dry_run_persists_without_spawning(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/dev/harness/benchmarks", json={
                "runtimes": ["ao"],
                "max_cases": 1,
                "persist": True,
            })
            assert resp.status == 200
            data = await resp.json()
            list_resp = await cli.get("/v1/dev/harness/benchmarks")
            listed = await list_resp.json()
            detail_resp = await cli.get(f"/v1/dev/harness/benchmarks/{data['benchmark_run_id']}")
            detail = await detail_resp.json()

        assert data["object"] == "hermes.dev_harness_benchmark_run"
        assert data["mode"] == "dry_run"
        assert data["live"] is False
        assert data["case_results"][0]["status"] == "validated"
        assert data["case_results"][0]["overall_score"] is None
        assert adapter._dev_execution_store.list_plans() == []
        assert adapter._subagent_event_store.list_events() == []
        assert listed["total"] == 1
        assert detail["benchmark_run_id"] == data["benchmark_run_id"]

    @pytest.mark.asyncio
    async def test_dev_harness_benchmark_fixture_scores_ao_and_openhands(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/dev/harness/benchmarks", json={
                "mode": "fixture",
                "runtimes": ["ao", "openhands"],
                "max_cases": 1,
                "iterations": 2,
            })
            assert resp.status == 200
            data = await resp.json()

        assert data["mode"] == "fixture"
        assert data["live"] is False
        assert data["iterations"] == 2
        assert data["summary"]["iteration_count"] == 2
        assert len(data["case_results"]) == 4
        ao_results = [item for item in data["case_results"] if item["runtime"] == "ao"]
        openhands_results = [item for item in data["case_results"] if item["runtime"] == "openhands"]
        assert {item["iteration"] for item in ao_results} == {1, 2}
        ao = ao_results[0]
        openhands = openhands_results[0]
        ao_runtime = next(item for item in data["runtime_results"] if item["runtime"] == "ao")
        openhands_runtime = next(item for item in data["runtime_results"] if item["runtime"] == "openhands")
        assert ao["status"] == "completed"
        assert ao["marker_score"] == 1.0
        assert ao["delivery_score"] == 1.0
        assert ao["contract_compliance_score"] == 0.333
        assert ao["output_contract_score"] == 0.0
        assert "WorkspaceAgentBoardView" in ao["evidence_terms_matched"]
        assert "WorkspaceAgentBoardView" in ao["required_evidence_terms_matched"]
        assert ao["required_evidence_score"] == 1.0
        assert "WorkspaceAgentBoardView" in ao["strong_evidence_terms_matched"]
        assert ao["file_symbol_reference_count"] >= 2
        assert ao["specificity_score"] > openhands["specificity_score"]
        assert openhands["status"] == "needs_review"
        assert openhands["marker_score"] == 0.0
        assert openhands["delivery_score"] == 1.0
        assert openhands["task_quality_score"] > openhands["contract_compliance_score"]
        assert openhands["strong_evidence_terms_matched"] == []
        assert openhands["required_evidence_score"] == 0.0
        assert openhands["specificity_score"] <= 0.55
        assert openhands["generic_finding_penalty"] > 0
        assert ao_runtime["iteration_count"] == 2
        assert ao_runtime["median_score"] == ao["overall_score"]
        assert ao_runtime["marker_pass_rate"] == 1.0
        assert openhands_runtime["required_evidence_pass_rate"] == 0.0
        assert adapter._dev_execution_store.list_plans() == []
        assert adapter._subagent_event_store.list_events() == []

    def test_dev_harness_benchmark_marker_scoring_requires_exact_final_marker_line(self):
        from gateway.dev_control.harness_benchmarks import (
            _cost_usd,
            _detect_prompt_delivery_failure,
            _openhands_usage_from_text,
            _score_result,
            _token_total,
        )

        echoed_prompt = _score_result({
            "runtime": "ao",
            "marker": "BENCH_AGENT_BOARD_METADATA_DONE",
            "status": "completed",
            "summary": (
                "Task Brief: End with FINAL_MARKER: BENCH_AGENT_BOARD_METADATA_DONE.\n"
                "The worker completed the inspection."
            ),
            "files_read_count": 2,
        })
        transcript_only = _score_result({
            "runtime": "ao",
            "marker": "BENCH_AGENT_BOARD_METADATA_DONE",
            "status": "needs_review",
            "summary": "The persisted summary omitted the exact marker.",
            "output_tail": (
                "The worker transcript contains the final answer.\n"
                "FINAL_MARKER: BENCH_AGENT_BOARD_METADATA_DONE"
            ),
            "files_read_count": 2,
        })
        structured_without_marker = _score_result({
            "runtime": "ao",
            "marker": "BENCH_AGENT_BOARD_METADATA_DONE",
            "status": "needs_review",
            "summary": (
                "BENCHMARK_RESULT\n"
                "marker: BENCH_AGENT_BOARD_METADATA_DONE\n"
                "finding_1: Runtime metadata is shown on the board.\n"
                "finding_2: Session metadata is shown in the drawer."
            ),
            "expected_evidence_terms": ["WorkspaceAgentBoardView", "runtime_session_id", "agent", "model"],
            "required_evidence_terms": ["WorkspaceAgentBoardView"],
            "files_read_count": 2,
        })
        specific_without_marker = _score_result({
            "runtime": "ao",
            "marker": "BENCH_AGENT_BOARD_METADATA_DONE",
            "status": "needs_review",
            "summary": (
                "BENCHMARK_RESULT\n"
                "marker: BENCH_AGENT_BOARD_METADATA_DONE\n"
                "finding_1: WorkspaceAgentBoardView includes runtime_session_id in the card metadata.\n"
                "finding_2: WorkspaceSubagentActivity carries agent and model fields for the drawer."
            ),
            "expected_evidence_terms": ["WorkspaceAgentBoardView", "WorkspaceSubagentActivity", "runtime_session_id", "agent", "model"],
            "required_evidence_terms": ["WorkspaceAgentBoardView"],
            "files_read_count": 2,
        })
        exact_line = _score_result({
            "runtime": "ao",
            "marker": "BENCH_AGENT_BOARD_METADATA_DONE",
            "status": "completed",
            "summary": (
                "The worker completed the inspection with two concrete findings.\n"
                "FINAL_MARKER: BENCH_AGENT_BOARD_METADATA_DONE"
            ),
            "files_read_count": 2,
        })

        assert echoed_prompt["marker_present"] is False
        assert echoed_prompt["marker_score"] == 0.0
        assert transcript_only["marker_present"] is True
        assert transcript_only["marker_in_summary"] is False
        assert transcript_only["marker_in_output_tail"] is True
        assert transcript_only["marker_score"] == 1.0
        assert structured_without_marker["marker_present"] is False
        assert structured_without_marker["structured_result_present"] is True
        assert structured_without_marker["findings_score"] == 1.0
        assert structured_without_marker["generic_finding_penalty"] > 0
        assert structured_without_marker["strong_evidence_terms_matched"] == []
        assert structured_without_marker["required_evidence_score"] == 0.0
        assert structured_without_marker["specificity_score"] <= 0.55
        assert structured_without_marker["contract_compliance_score"] == 0.333
        assert structured_without_marker["output_contract_score"] == 0.0
        assert specific_without_marker["specificity_score"] > structured_without_marker["specificity_score"]
        assert specific_without_marker["required_evidence_score"] == 1.0
        assert "WorkspaceAgentBoardView" in specific_without_marker["required_evidence_terms_matched"]
        assert specific_without_marker["file_symbol_reference_count"] >= 2
        assert "WorkspaceSubagentActivity" in specific_without_marker["strong_evidence_terms_matched"]
        assert specific_without_marker["task_quality_score"] > structured_without_marker["task_quality_score"]
        assert specific_without_marker["overall_score"] > structured_without_marker["overall_score"]
        assert "WorkspaceAgentBoardView" in specific_without_marker["evidence_terms_matched"]
        assert exact_line["marker_present"] is True
        assert exact_line["marker_in_summary"] is True
        assert exact_line["marker_score"] == 1.0

        echoed_contract_tail = (
            "codex --approval-mode full-auto --model 'gpt-5.5'\n"
            "Output contract:\n"
            "Return only this shape:\n"
            "BENCHMARK_RESULT\n"
            "marker: BENCH_AGENT_BOARD_METADATA_DONE\n"
            "finding_1: <one concrete finding>\n"
            "finding_2: <one concrete finding>\n"
            "FINAL_MARKER: BENCH_AGENT_BOARD_METADATA_DONE\n"
        )
        assert "echoed prompt/placeholder contract" in _detect_prompt_delivery_failure(
            runtime="ao",
            output_tail=echoed_contract_tail,
            marker="BENCH_AGENT_BOARD_METADATA_DONE",
        )

        failed_delivery = _score_result({
            "runtime": "ao",
            "marker": "BENCH_AGENT_BOARD_METADATA_DONE",
            "status": "runtime_delivery_failed",
            "runtime_delivery_failed": True,
            "summary": echoed_contract_tail,
            "output_tail": echoed_contract_tail,
            "files_read_count": 0,
        })
        assert failed_delivery["marker_present"] is False
        assert failed_delivery["structured_result_present"] is False
        assert failed_delivery["delivery_score"] == 0.0

        openhands_tail = (
            "stats: {'usage_to_metrics': {'default': {'model_name': 'openrouter/openai/gpt-4o-mini', "
            "'accumulated_cost': 0.00176895, 'accumulated_token_usage': {'prompt_tokens': 18601, "
            "'completion_tokens': 474, 'cache_read_tokens': 17408, 'cache_write_tokens': 0}}}}"
        )
        usage = _openhands_usage_from_text(openhands_tail)
        assert usage["cost_usd"] == 0.00176895
        assert usage["token_total"] == 19075
        assert usage["prompt_tokens"] == 18601
        assert usage["completion_tokens"] == 474
        assert usage["cache_read_tokens"] == 17408
        assert _token_total({"output_tail": openhands_tail}) == 19075
        assert _cost_usd({"output_tail": openhands_tail}) == 0.00176895

    @pytest.mark.asyncio
    async def test_dev_harness_benchmark_live_ao_uses_codex_exec(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                resp = await cli.post("/v1/dev/harness/benchmarks", json={
                    "live": True,
                    "runtimes": ["ao"],
                    "max_cases": 1,
                    "timeout_seconds": 1,
                })
                assert resp.status == 200
                data = await resp.json()

        assert data["mode"] == "live"
        assert data["live"] is True
        assert adapter._dev_execution_store.list_plans() == []
        result = data["case_results"][0]
        assert result["runtime"] == "ao"
        assert result["plan_id"] is None
        assert result["runtime_session_id"] == "codex-exec-test"
        assert result["benchmark_execution_mode"] == "codex_exec"
        assert result["output_tail_captured"] is True
        assert "BENCHMARK_RESULT" in result["output_tail"]
        assert result["marker_present"] is True
        assert result["token_total"] == 1234
        assert bridge.spawn_kwargs is None
        assert bridge.codex_exec_kwargs["project_id"] == "OrynWorkspace"
        assert "Do not edit files" in bridge.codex_exec_kwargs["prompt"]
        assert "## Hermes AO Delegation Contract" not in bridge.codex_exec_kwargs["prompt"]
        assert "## Dev Launch Profile" not in bridge.codex_exec_kwargs["prompt"]
        assert "Return only this shape:" in bridge.codex_exec_kwargs["prompt"]
        assert "BENCHMARK_RESULT" in bridge.codex_exec_kwargs["prompt"]
        assert "Required evidence terms:" in bridge.codex_exec_kwargs["prompt"]
        assert "WorkspaceAgentBoardView" in bridge.codex_exec_kwargs["prompt"]
        assert "The final line must be exactly:" in bridge.codex_exec_kwargs["prompt"]
        assert "FINAL_MARKER: BENCH_AGENT_BOARD_METADATA_DONE" in bridge.codex_exec_kwargs["prompt"]
        assert adapter._subagent_event_store.list_events() == []

    @pytest.mark.asyncio
    async def test_dev_harness_benchmark_detects_ao_prompt_delivery_failure(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.run_codex_exec_benchmark = lambda **kwargs: {
            "session_id": "codex-exec-failed",
            "status": "completed",
            "summary": (
                "BENCHMARK_RESULT\n"
                "marker: BENCH_AGENT_BOARD_METADATA_DONE\n"
                "finding_1: <one concrete finding>\n"
                "finding_2: <one concrete finding>\n"
                "FINAL_MARKER: BENCH_AGENT_BOARD_METADATA_DONE"
            ),
            "output_tail": (
                "BENCHMARK_RESULT\n"
                "marker: BENCH_AGENT_BOARD_METADATA_DONE\n"
                "finding_1: <one concrete finding>\n"
                "finding_2: <one concrete finding>\n"
                "FINAL_MARKER: BENCH_AGENT_BOARD_METADATA_DONE"
            ),
            "duration_seconds": 1.0,
        }
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                resp = await cli.post("/v1/dev/harness/benchmarks", json={
                    "live": True,
                    "runtimes": ["ao"],
                    "max_cases": 1,
                    "timeout_seconds": 1,
                })
                assert resp.status == 200
                data = await resp.json()

        result = data["case_results"][0]
        runtime_result = data["runtime_results"][0]
        assert result["status"] == "runtime_delivery_failed"
        assert result["runtime_delivery_failed"] is True
        assert result["delivery_status"] == "failed"
        assert "echoed prompt/placeholder contract" in result["delivery_failure_reason"]
        assert result["delivery_cleanup"] is None
        assert result["delivery_score"] == 0.0
        assert result["task_quality_score"] < 1.0
        assert result["marker_present"] is False
        assert result["marker_in_output_tail"] is False
        assert bridge.killed_sessions == []
        assert runtime_result["runtime_delivery_failed"] == 1
        assert data["summary"]["runtime_delivery_failed_count"] == 1

    def test_dev_harness_benchmark_tools_match_api_shape(self, adapter, tmp_path):
        from tools.dev_execution_tools import (
            _handle_dev_harness_benchmark,
            _handle_dev_harness_benchmark_runs,
        )

        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        with patch("tools.dev_execution_tools.DevExecutionStore", return_value=adapter._dev_execution_store), \
             patch("tools.dev_execution_tools.SubagentEventStore", return_value=adapter._subagent_event_store):
            generated = json.loads(_handle_dev_harness_benchmark({
                "mode": "fixture",
                "runtimes": ["ao"],
                "max_cases": 1,
            }))
            listed = json.loads(_handle_dev_harness_benchmark_runs({}))
            fetched = json.loads(_handle_dev_harness_benchmark_runs({
                "benchmark_run_id": generated["benchmark_run_id"],
            }))

        assert generated["object"] == "hermes.dev_harness_benchmark_run"
        assert generated["summary"]["case_result_count"] == 1
        assert listed["object"] == "list"
        assert listed["total"] == 1
        assert fetched["benchmark_run_id"] == generated["benchmark_run_id"]

    @pytest.mark.asyncio
    async def test_dev_harness_benchmark_live_iterations_are_capped(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                resp = await cli.post("/v1/dev/harness/benchmarks", json={
                    "live": True,
                    "runtimes": ["ao"],
                    "max_cases": 1,
                    "iterations": 9,
                    "timeout_seconds": 1,
                })
                assert resp.status == 200
                data = await resp.json()

        assert data["iterations"] == 3
        assert len(data["case_results"]) == 3
        assert adapter._dev_execution_store.list_plans() == []
        assert data["summary"]["iteration_count"] == 3
        runtime = data["runtime_results"][0]
        assert runtime["iteration_count"] == 3
        assert runtime["median_delivery_score"] == 1.0
        assert all(result["benchmark_execution_mode"] == "codex_exec" for result in data["case_results"])

    @pytest.mark.asyncio
    async def test_dev_harness_recommendations_can_reference_benchmark_run(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            bench_resp = await cli.post("/v1/dev/harness/benchmarks", json={
                "mode": "fixture",
                "runtimes": ["ao", "openhands"],
                "max_cases": 1,
            })
            benchmark = await bench_resp.json()
            report_resp = await cli.post("/v1/dev/harness/report", json={"limit": 1, "persist": True})
            report = await report_resp.json()
            rec_resp = await cli.post("/v1/dev/harness/recommendations", json={
                "report_id": report["report_id"],
                "benchmark_run_id": benchmark["benchmark_run_id"],
            })
            rec = await rec_resp.json()

        assert rec["benchmark_run_id"] == benchmark["benchmark_run_id"]
        assert rec["benchmark_snapshot"]["benchmark_run_id"] == benchmark["benchmark_run_id"]
        assert rec["summary"]["benchmark_run_id"] == benchmark["benchmark_run_id"]
        runtime_recs = [
            item for item in rec["recommendations"]
            if item["category"] == "runtime_policy"
            and "controlled benchmark evidence" in item["reason"]
        ]
        assert runtime_recs
        reason = runtime_recs[0]["reason"]
        assert "median score" in reason
        assert "task quality" in reason
        assert "contract compliance" in reason
        assert "marker pass rate" in reason
        assert "required evidence pass rate" in reason
        assert "delivery failure rate" in reason
        assert "avg duration" in reason
        assert "not an automatic policy change" in reason
        assert "Keep AO preferred for read-only inspection" in runtime_recs[0]["suggested_change"]
        assert "Require more benchmark samples" in runtime_recs[0]["implementation_brief"]
        assert "Do not mutate runtime policy" in runtime_recs[0]["non_goals"][0]
        output_contract_recs = [
            item for item in rec["recommendations"]
            if item["category"] == "prompt_template"
            and "Worker Output Contract v2" in item["title"]
        ]
        assert output_contract_recs
        assert "median output contract score" in output_contract_recs[0]["reason"]
        assert "Do not mutate runtime policy" in output_contract_recs[0]["non_goals"][0]

    def test_benchmark_policy_posture_can_recommend_read_only_openhands_watch(self):
        from gateway.dev_control.harness_recommendations import _benchmark_runtime_policy_posture

        posture = _benchmark_runtime_policy_posture([
            {
                "runtime": "ao",
                "median_score": 0.884,
                "median_task_quality_score": 0.964,
                "median_contract_compliance_score": 0.5,
                "delivery_failure_rate": 0.0,
                "average_duration_seconds": 1.679,
            },
            {
                "runtime": "openhands",
                "median_score": 0.887,
                "median_task_quality_score": 0.75,
                "median_contract_compliance_score": 1.0,
                "delivery_failure_rate": 0.0,
                "average_duration_seconds": 9.708,
            },
        ])

        assert "AO as the production default for write/retry/recovery" in posture
        assert "OpenHands for read-only inspection" in posture
        assert "despite higher latency" in posture
        assert "AO fallback unchanged" in posture

    def test_dev_create_execution_plan_tool_schema_accepts_runtime_selection_inputs(self):
        from tools.dev_execution_tools import DEV_CREATE_EXECUTION_PLAN_SCHEMA

        task_properties = DEV_CREATE_EXECUTION_PLAN_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]
        assert "runtime" in task_properties
        assert "permissions" in task_properties
        assert "agent" in task_properties
        assert "model" in task_properties
        assert "reasoning_effort" in task_properties

    @pytest.mark.asyncio
    async def test_dev_openhands_server_status_reports_missing_cli(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch("tools.openhands_bridge.shutil.which", return_value=None), \
                 patch("tools.openhands_bridge.read_openhands_server_metadata", return_value={}):
                resp = await cli.get("/v1/dev/runtimes/openhands/server")
                assert resp.status == 200
                data = await resp.json()

        assert data["status"] == "missing_cli"
        assert data["ok"] is False
        assert "uv tool install openhands --python 3.12" in data["install_instruction"]

    @pytest.mark.asyncio
    async def test_dev_execution_plan_create_persists_without_spawning(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 8 smoke",
                "vision_brief": "Let Dev launch workers.",
                "tasks": [{
                    "goal": "Inspect board launch UI",
                    "prompt": "Inspect the Agent Board launch UI.",
                    "profile_id": "workspace.inspect",
                    "acceptance_criteria": ["Report whether the launch action exists."],
                }],
            })
            assert resp.status == 200
            data = await resp.json()

        plan = data["plan"]
        assert plan["status"] == "planned"
        assert plan["tasks"][0]["status"] == "planned"
        assert plan["tasks"][0]["ao_session_id"] is None
        assert adapter._subagent_event_store.list_events() == []

    @pytest.mark.asyncio
    async def test_dev_execution_plan_launch_spawns_and_links_board_metadata(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 8 launch",
                "tasks": [{
                    "goal": "Implement launch metadata",
                    "prompt": "Inspect the metadata path.",
                    "profile_id": "workspace.implement",
                    "acceptance_criteria": ["Board row includes launch metadata."],
                }],
            })
            assert create_resp.status == 200
            plan_id = (await create_resp.json())["plan"]["plan_id"]

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                launch_resp = await cli.post(f"/v1/dev/execution-plans/{plan_id}/launch", json={})
                assert launch_resp.status == 200
                launch_data = await launch_resp.json()
                board_resp = await cli.get("/v1/subagents/board")
                board_data = await board_resp.json()

        assert launch_data["ok"] is True
        assert launch_data["launched"][0]["ao_session_id"] == "oryn-workspace-9"
        assert launch_data["launched"][0]["runtime"] == "ao"
        assert launch_data["launched"][0]["runtime_session_id"] == "oryn-workspace-9"
        assert bridge.spawn_kwargs["project_id"] == "OrynWorkspace"
        assert bridge.spawn_kwargs["model"] == "gpt-5.5"
        assert bridge.spawn_kwargs["reasoning_effort"] == "high"
        assert "Permissions: edit" in bridge.spawn_kwargs["prompt"]
        row = next(item for item in board_data["data"] if item["id"] == "ao:oryn-workspace-9")
        assert row["launch_profile_id"] == "workspace.implement"
        assert row["runtime"] == "ao"
        assert row["runtime_session_id"] == "oryn-workspace-9"
        assert row["runtime_project_id"] == "OrynWorkspace"
        assert row["launch_plan_id"] == plan_id
        assert row["permissions"] == "edit"
        assert row["acceptance_criteria"] == ["Board row includes launch metadata."]

    @pytest.mark.asyncio
    async def test_dev_execution_plan_launch_rejects_unavailable_openhands_without_mutating_task(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        openhands = _FakeOpenHandsBridge(launch_supported=False)
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 18 unavailable OpenHands runtime",
                "tasks": [{
                    "goal": "Try OpenHands runtime",
                    "prompt": "Do not launch.",
                    "runtime": "openhands",
                }],
            })
            assert create_resp.status == 200
            plan_id = (await create_resp.json())["plan"]["plan_id"]
            with patch("tools.openhands_bridge.OpenHandsBridge", return_value=openhands):
                launch_resp = await cli.post(f"/v1/dev/execution-plans/{plan_id}/launch", json={})
            assert launch_resp.status == 200
            launch_data = await launch_resp.json()
            status_resp = await cli.get(f"/v1/dev/execution-plans/{plan_id}/status")
            status_data = await status_resp.json()

        assert launch_data["ok"] is False
        assert launch_data["failures"][0]["error"] == "OpenHands is not installed or configured."
        assert status_data["status"] == "planned"
        assert status_data["tasks"][0]["ao_session_id"] is None
        assert adapter._subagent_event_store.list_events() == []

    @pytest.mark.asyncio
    async def test_dev_execution_plan_launch_spawns_openhands_runtime_without_ao_event_metadata(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        openhands = _FakeOpenHandsBridge()
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 18 OpenHands launch",
                "tasks": [{
                    "goal": "Inspect with OpenHands",
                    "prompt": "Inspect the workspace read-only.",
                    "profile_id": "workspace.openhands.inspect",
                    "acceptance_criteria": ["Report one concrete finding."],
                }],
            })
            assert create_resp.status == 200
            plan_id = (await create_resp.json())["plan"]["plan_id"]

            with patch("tools.openhands_bridge.OpenHandsBridge", return_value=openhands):
                launch_resp = await cli.post(f"/v1/dev/execution-plans/{plan_id}/launch", json={})
                assert launch_resp.status == 200
                launch_data = await launch_resp.json()
                status_resp = await cli.get(f"/v1/dev/execution-plans/{plan_id}/status")
                status_data = await status_resp.json()
                board_resp = await cli.get("/v1/subagents/board")
                board_data = await board_resp.json()

        assert launch_data["ok"] is True
        assert launch_data["launched"][0]["runtime"] == "openhands"
        assert launch_data["launched"][0]["runtime_session_id"] == "oh-conv-1"
        assert openhands.spawn_kwargs["project_id"] == "OrynWorkspace"
        assert "minimal_worker_prompt" not in openhands.spawn_kwargs
        assert "Hermes AO Delegation Contract" not in openhands.spawn_kwargs["prompt"]
        task = status_data["tasks"][0]
        assert task["runtime"] == "openhands"
        assert task["runtime_session_id"] == "oh-conv-1"
        events = adapter._subagent_event_store.list_events()
        assert events[0]["runtime"] == "openhands"
        assert events[0]["runtime_session_id"] == "oh-conv-1"
        assert events[0].get("ao_session_id") is None
        row = next(item for item in board_data["data"] if item["id"] == "openhands:oh-conv-1")
        assert row["runtime"] == "openhands"
        assert row["runtime_session_id"] == "oh-conv-1"
        assert row["ao_session_id"] is None
        assert row["can_open"] is False
        assert row["action_unavailable_reason"] == "AO controls are only available for AO-backed workers."

    @pytest.mark.asyncio
    async def test_dev_execution_plan_launch_auto_read_only_selects_openhands(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        openhands = _FakeOpenHandsBridge()
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 21 auto OpenHands",
                "tasks": [{
                    "goal": "Inspect board read-only",
                    "prompt": "Inspect the Agent Board read-only and report one finding. Do not edit files.",
                    "runtime": "auto",
                    "permissions": "read_only",
                }],
            })
            assert create_resp.status == 200
            plan_id = (await create_resp.json())["plan"]["plan_id"]

            with patch("tools.openhands_bridge.OpenHandsBridge", return_value=openhands):
                launch_resp = await cli.post(f"/v1/dev/execution-plans/{plan_id}/launch", json={})
                assert launch_resp.status == 200
                launch_data = await launch_resp.json()
                status_resp = await cli.get(f"/v1/dev/execution-plans/{plan_id}/status")
                status_data = await status_resp.json()

        assert launch_data["ok"] is True
        assert launch_data["launched"][0]["runtime"] == "openhands"
        assert launch_data["launched"][0]["runtime_selection"]["selection_mode"] == "auto"
        assert launch_data["launched"][0]["runtime_selection_reason"] == "Read-only inspection can run on healthy OpenHands with output capture."
        assert launch_data["launched"][0]["runtime_policy_status"] == "insufficient_evidence"
        task = status_data["tasks"][0]
        assert task["runtime"] == "openhands"
        assert task["selected_runtime"] == "openhands"
        assert task["runtime_selection"]["selection_mode"] == "auto"
        assert task["runtime_policy_status"] == "insufficient_evidence"
        events = adapter._subagent_event_store.list_events()
        assert events[0]["runtime"] == "openhands"
        assert events[0]["selected_runtime"] == "openhands"
        assert events[0]["runtime_policy_status"] == "insufficient_evidence"

    @pytest.mark.asyncio
    async def test_dev_execution_plan_launch_auto_openhands_spawn_failure_falls_back_to_ao(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        openhands = _FakeOpenHandsBridge(spawn_error=RuntimeError("server rejected session"))
        bridge = _FakeAOBridge()
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 21 auto fallback",
                "tasks": [{
                    "goal": "Inspect with fallback",
                    "prompt": "Inspect read-only and report one finding.",
                    "runtime": "auto",
                    "permissions": "read_only",
                }],
            })
            assert create_resp.status == 200
            plan_id = (await create_resp.json())["plan"]["plan_id"]

            with patch("tools.openhands_bridge.OpenHandsBridge", return_value=openhands), \
                 patch("tools.ao_bridge.AOBridge", return_value=bridge):
                launch_resp = await cli.post(f"/v1/dev/execution-plans/{plan_id}/launch", json={})
                assert launch_resp.status == 200
                launch_data = await launch_resp.json()
                status_resp = await cli.get(f"/v1/dev/execution-plans/{plan_id}/status")
                status_data = await status_resp.json()

        assert launch_data["ok"] is True
        assert launch_data["launched"][0]["runtime"] == "ao"
        assert launch_data["launched"][0]["runtime_selection"]["selection_mode"] == "fallback"
        assert "server rejected session" in launch_data["launched"][0]["runtime_fallback_reason"]
        assert launch_data["launched"][0]["runtime_policy_status"] == "fallback"
        assert bridge.spawn_kwargs["project_id"] == "OrynWorkspace"
        assert "Hermes AO Delegation Contract" in bridge.spawn_kwargs["prompt"]
        task = status_data["tasks"][0]
        assert task["runtime"] == "ao"
        assert task["selected_runtime"] == "ao"
        assert "server rejected session" in task["runtime_fallback_reason"]
        events = adapter._subagent_event_store.list_events()
        assert events[0]["runtime"] == "ao"
        assert events[0]["runtime_selection"]["selection_mode"] == "fallback"

    @pytest.mark.asyncio
    async def test_dev_execution_plan_status_reports_planned_without_launched_tasks(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 9 planned",
                "tasks": [{"goal": "Plan only", "prompt": "Do not launch yet."}],
            })
            assert create_resp.status == 200
            plan_id = (await create_resp.json())["plan"]["plan_id"]

            resp = await cli.get(f"/v1/dev/execution-plans/{plan_id}/status")
            assert resp.status == 200
            data = await resp.json()

        assert data["status"] == "planned"
        assert data["tasks"][0]["status"] == "planned"
        assert data["tasks"][0]["status_reason"] == "Task has not been launched."

    @pytest.mark.asyncio
    async def test_dev_execution_plan_status_derives_running_from_ao_session(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 9 running",
                "tasks": [{"goal": "Running worker", "prompt": "Keep working."}],
            })
            plan_id = (await create_resp.json())["plan"]["plan_id"]
            task_id = (await (await cli.get(f"/v1/dev/execution-plans/{plan_id}")).json())["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                resp = await cli.get(f"/v1/dev/execution-plans/{plan_id}/status")
                assert resp.status == 200
                data = await resp.json()

        assert data["status"] == "running"
        assert data["tasks"][0]["status"] == "running"
        assert data["tasks"][0]["ao_session_id"] == "oryn-workspace-9"
        assert data["tasks"][0]["runtime"] == "ao"
        assert data["tasks"][0]["runtime_session_id"] == "oryn-workspace-9"
        assert data["tasks"][0]["runtime_project_id"] == "OrynWorkspace"
        assert data["tasks"][0]["model"] == "gpt-5.5"

    @pytest.mark.asyncio
    async def test_dev_execution_plan_status_reports_completed_with_good_summary(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "done"
        bridge.session.summary = "BOARD_DONE Verified the requested implementation path and found no unresolved gaps."
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 9 completed",
                "tasks": [{"goal": "Verify result BOARD_DONE", "prompt": "Return BOARD_DONE when complete."}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                resp = await cli.get(f"/v1/dev/execution-plans/{plan_id}/status")
                assert resp.status == 200
                data = await resp.json()

        assert data["status"] == "completed"
        assert data["tasks"][0]["status"] == "completed"
        assert data["tasks"][0]["summary_quality"] == "ok"

    @pytest.mark.asyncio
    async def test_dev_execution_plan_status_marks_stale_ao_session_failed(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.runtime_health = lambda session: {
            "runtime_health": "stale",
            "runtime_warning": "AO reports this worker as running, but its tmux/process runtime is gone.",
        }
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 9 stale",
                "tasks": [{"goal": "Stale worker", "prompt": "Keep working."}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                resp = await cli.get(f"/v1/dev/execution-plans/{plan_id}/status")
                assert resp.status == 200
                data = await resp.json()

        assert data["status"] == "failed"
        assert data["tasks"][0]["status"] == "failed"
        assert "runtime is gone" in data["tasks"][0]["status_reason"]

    @pytest.mark.asyncio
    async def test_dev_execution_plan_synthesis_marks_weak_completed_summary_needs_review(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "done"
        bridge.session.summary = "unclear"
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 9 synthesis",
                "tasks": [{
                    "goal": "Verify worker result BOARD_DONE",
                    "prompt": "Return BOARD_DONE when complete.",
                }],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                resp = await cli.post(f"/v1/dev/execution-plans/{plan_id}/synthesize", json={})
                assert resp.status == 200
                data = await resp.json()

        assert data["status"] == "needs_review"
        assert data["tasks"][0]["status"] == "needs_review"
        assert data["tasks"][0]["summary_quality"] == "warning"
        assert data["unresolved_gaps"]
        assert "Phase 9 synthesis: needs_review" in data["report"]

    @pytest.mark.asyncio
    async def test_dev_execution_plan_review_running_is_not_ready(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 10 running",
                "tasks": [{"goal": "Keep working", "prompt": "Stay running."}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                resp = await cli.get(f"/v1/dev/execution-plans/{plan_id}/review?include_synthesis=false")
                assert resp.status == 200
                data = await resp.json()

        assert data["status"] == "running"
        assert data["review_status"] == "not_ready"
        assert data["recommended_action"] == "none"
        assert data["target_task_ids"] == [task_id]
        assert data["synthesis"] is None

    @pytest.mark.asyncio
    async def test_dev_execution_plan_review_accepts_clean_completed_plan(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "done"
        bridge.session.summary = "PHASE10_DONE Verified the implementation and tests with no unresolved gaps."
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 10 clean",
                "tasks": [{"goal": "Return PHASE10_DONE", "prompt": "Return PHASE10_DONE when finished."}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                resp = await cli.post(f"/v1/dev/execution-plans/{plan_id}/review", json={})
                assert resp.status == 200
                data = await resp.json()

        assert data["status"] == "completed"
        assert data["review_status"] == "accepted"
        assert data["recommended_action"] == "accept"
        assert data["target_task_ids"] == []
        assert "PHASE10_DONE" in data["synthesis"]["report"]

    @pytest.mark.asyncio
    async def test_dev_execution_plan_review_weak_summary_recommends_follow_up_when_available(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "done"
        bridge.session.summary = "unclear"
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 10 weak",
                "tasks": [{"goal": "Return PHASE10_DONE", "prompt": "Return PHASE10_DONE when finished."}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                resp = await cli.get(f"/v1/dev/execution-plans/{plan_id}/review")
                assert resp.status == 200
                data = await resp.json()

        assert data["status"] == "needs_review"
        assert data["review_status"] == "needs_follow_up"
        assert data["recommended_action"] == "follow_up"
        assert data["target_task_ids"] == [task_id]
        assert "final implementation summary" in data["suggested_message"]

    @pytest.mark.asyncio
    async def test_dev_execution_plan_review_rejects_prompt_echo_even_with_required_marker(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "done"
        bridge.session.summary = (
            "## Task Brief\n"
            "## Dev Launch Profile\n"
            "Permissions: read_only\n"
            "## Required Actions\n"
            "Read WorkspaceAgentBoardView.swift and WorkspaceSubagentActivity.swift.\n"
            "## Final Summary Requirement\n"
            "Your final summary must contain concrete findings and end with PHASE10_REVIEW_DONE."
        )
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 10 prompt echo",
                "tasks": [{"goal": "Return PHASE10_REVIEW_DONE", "prompt": "Return PHASE10_REVIEW_DONE when finished."}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                status_resp = await cli.get(f"/v1/dev/execution-plans/{plan_id}/status")
                assert status_resp.status == 200
                status_data = await status_resp.json()
                review_resp = await cli.get(f"/v1/dev/execution-plans/{plan_id}/review")
                assert review_resp.status == 200
                review_data = await review_resp.json()

        assert status_data["status"] == "needs_review"
        assert status_data["tasks"][0]["status"] == "needs_review"
        assert "echo the prompt" in status_data["tasks"][0]["summary_warning"]
        assert review_data["review_status"] == "needs_follow_up"
        assert review_data["recommended_action"] == "follow_up"
        assert review_data["target_task_ids"] == [task_id]

    @pytest.mark.asyncio
    async def test_dev_execution_plan_status_repairs_prompt_echo_summary_from_transcript(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "done"
        bridge.session.summary = (
            "## Task Brief\n"
            "Permissions: read_only\n"
            "## Final Summary Requirement\n"
            "End with PHASE10_REVIEW_DONE."
        )
        bridge.capture_output = lambda session, lines=160: (
            "• Read WorkspaceAgentBoardView.swift and WorkspaceSubagentActivity.swift\n"
            "────────────────────────────────────────────────────────────────────────────────\n"
            "WorkspaceAgentBoardView.swift displays Dev review labels from plan.reviewStatus.\n"
            "WorkspaceSubagentActivity.swift decodes review_status into reviewStatus plus recommended_action metadata.\n"
            "Dev review labels are decoded and displayed with file-backed evidence.\n\n"
            "PHASE10_REVIEW_DONE\n"
            "────────────────────────────────────────────────────────────────────────────────\n"
        )
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 10 transcript repair",
                "tasks": [{"goal": "Return PHASE10_REVIEW_DONE", "prompt": "Read files and return PHASE10_REVIEW_DONE."}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )
            adapter._subagent_event_store.append_event({
                "event": "subagent.complete",
                "subagent_id": "ao:oryn-workspace-9",
                "ao_session_id": "oryn-workspace-9",
                "runtime": "ao",
                "status": "completed",
                "goal": "Return PHASE10_REVIEW_DONE",
                "summary": bridge.session.summary,
                "message": bridge.session.summary,
                "launch_plan_id": plan_id,
                "launch_task_id": task_id,
            })

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                status_resp = await cli.get(f"/v1/dev/execution-plans/{plan_id}/status")
                assert status_resp.status == 200
                status_data = await status_resp.json()
                review_resp = await cli.get(f"/v1/dev/execution-plans/{plan_id}/review?include_synthesis=false")
                assert review_resp.status == 200
                review_data = await review_resp.json()

        task_data = status_data["tasks"][0]
        assert status_data["status"] == "completed"
        assert task_data["status"] == "completed"
        assert task_data["summary_quality"] == "ok"
        assert "WorkspaceAgentBoardView.swift displays Dev review labels" in task_data["summary"]
        assert review_data["review_status"] == "accepted"
        assert review_data["recommended_action"] == "accept"
        complete_events = [
            event for event in adapter._subagent_event_store.list_events(ao_session_id="oryn-workspace-9")
            if event["event"] == "subagent.complete"
        ]
        assert any(event.get("transcript_corrected") is True for event in complete_events)

    @pytest.mark.asyncio
    async def test_dev_execution_plan_synthesis_preserves_transcript_findings_and_marker(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "done"
        bridge.session.summary = (
            "## Task Brief\n"
            "Permissions: read_only\n"
            "## Final Summary Requirement\n"
            "End with PHASE10_TRANSCRIPT_REPAIR_DONE."
        )
        final_answer = (
            "• ## FINDING 1: WorkspaceAgentBoardView.swift\n\n"
            "  devPlansStrip renders review labels by calling devPlanReviewLabel(plan.reviewStatus).\n\n"
            "  ## FINDING 2: WorkspaceSubagentActivity.swift\n\n"
            "  WorkspaceDevExecutionPlan decodes review_status into reviewStatus and carries recommendedAction metadata.\n\n"
            "  ## DEV REVIEW LABELS\n\n"
            "  Yes. The model decodes review_status and the board displays it in a pill.\n\n"
            "  PHASE10_TRANSCRIPT_REPAIR_DONE\n"
        )
        bridge.capture_output = lambda session, lines=160: (
            "prompt echo before answer\n"
            "────────────────────────────────────────────────────────────────────────────────\n"
            f"{final_answer}"
            "─ Worked for 1m 11s ────────────────────────────────────────────────────────────\n"
        )
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 10 synthesis detail",
                "tasks": [{"goal": "Return PHASE10_TRANSCRIPT_REPAIR_DONE", "prompt": "Read files and return PHASE10_TRANSCRIPT_REPAIR_DONE."}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )
            adapter._subagent_event_store.append_event({
                "event": "subagent.complete",
                "subagent_id": "ao:oryn-workspace-9",
                "ao_session_id": "oryn-workspace-9",
                "runtime": "ao",
                "status": "completed",
                "goal": "Return PHASE10_TRANSCRIPT_REPAIR_DONE",
                "summary": bridge.session.summary,
                "launch_plan_id": plan_id,
                "launch_task_id": task_id,
            })

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                resp = await cli.post(f"/v1/dev/execution-plans/{plan_id}/synthesize", json={})
                assert resp.status == 200
                data = await resp.json()

        task_summary = data["tasks"][0]["summary"]
        assert "devPlansStrip renders review labels" in task_summary
        assert "WorkspaceDevExecutionPlan decodes review_status" in task_summary
        assert "PHASE10_TRANSCRIPT_REPAIR_DONE" in task_summary
        assert "devPlansStrip renders review labels" in data["report"]
        assert "WorkspaceDevExecutionPlan decodes review_status" in data["report"]
        assert "PHASE10_TRANSCRIPT_REPAIR_DONE" in data["report"]

    @pytest.mark.asyncio
    async def test_dev_execution_plan_review_failed_with_prompt_metadata_recommends_repair_retry(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "errored"
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 10 failed",
                "tasks": [{"goal": "Fix failure", "prompt": "Try the implementation."}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                resp = await cli.get(f"/v1/dev/execution-plans/{plan_id}/review?include_synthesis=false")
                assert resp.status == 200
                data = await resp.json()

        assert data["status"] == "failed"
        assert data["review_status"] == "retry_recommended"
        assert data["recommended_action"] == "repair_retry"
        assert data["target_task_ids"] == [task_id]

    @pytest.mark.asyncio
    async def test_dev_execution_plan_review_failed_without_prompt_metadata_requires_human(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "errored"
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 10 failed without prompt",
                "tasks": [{"goal": "Fix failure", "prompt": "Original prompt."}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )
            adapter._dev_execution_store._conn.execute(
                "UPDATE dev_execution_plan_tasks SET prompt = '', payload = '{}' WHERE task_id = ?",
                (task_id,),
            )
            adapter._dev_execution_store._conn.commit()

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                resp = await cli.get(f"/v1/dev/execution-plans/{plan_id}/review?include_synthesis=false")
                assert resp.status == 200
                data = await resp.json()

        assert data["status"] == "failed"
        assert data["review_status"] == "human_review_required"
        assert data["recommended_action"] == "human_review"
        assert data["target_task_ids"] == [task_id]

    @pytest.mark.asyncio
    async def test_dev_execution_plan_apply_review_accept_persists_action_event(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "done"
        bridge.session.summary = "PHASE11_ACCEPT_DONE Verified implementation with no unresolved gaps."
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 11 accept",
                "tasks": [{"goal": "Return PHASE11_ACCEPT_DONE", "prompt": "Return PHASE11_ACCEPT_DONE when finished."}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                resp = await cli.post(f"/v1/dev/execution-plans/{plan_id}/apply-review", json={"include_synthesis": False})
                assert resp.status == 200
                data = await resp.json()

        assert data["object"] == "hermes.dev_execution_plan_review_application"
        assert data["applied_action"] == "accept"
        assert data["status"] == "applied"
        assert data["results"][0]["task_id"] == task_id
        assert data["results"][0]["action"] == "accept"
        assert data["results"][0]["status"] == "accepted"
        events = adapter._subagent_event_store.list_events(ao_session_id="oryn-workspace-9")
        assert events[-1]["event"] == "subagent.action"
        assert events[-1]["action"] == "accept"
        assert events[-1]["launch_plan_id"] == plan_id
        assert events[-1]["launch_task_id"] == task_id

    @pytest.mark.asyncio
    async def test_dev_execution_plan_apply_review_follow_up_sends_message(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "done"
        bridge.session.summary = "unclear"
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 11 follow-up",
                "tasks": [{"goal": "Return PHASE11_FOLLOW_UP_DONE", "prompt": "Return PHASE11_FOLLOW_UP_DONE when finished."}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                resp = await cli.post(
                    f"/v1/dev/execution-plans/{plan_id}/apply-review",
                    json={"include_synthesis": False, "message": "Send concrete findings now."},
                )
                assert resp.status == 200
                data = await resp.json()

        assert data["applied_action"] == "follow_up"
        assert data["review"]["review_status"] == "needs_follow_up"
        assert data["results"][0]["action"] == "follow-up"
        assert bridge.sent_messages == [("oryn-workspace-9", "Send concrete findings now.")]
        events = adapter._subagent_event_store.list_events(ao_session_id="oryn-workspace-9")
        assert events[-1]["event"] == "subagent.action"
        assert events[-1]["action"] == "follow-up"
        assert events[-1]["sent_message"] == "Send concrete findings now."
        assert events[-1]["launch_task_id"] == task_id

    @pytest.mark.asyncio
    async def test_dev_execution_plan_apply_review_infers_weak_completion_from_idle_transcript(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "working"
        bridge.session.summary = None
        bridge.capture_output = lambda session, lines=160: (
            "Some setup output\n"
            "──────────────────────────────────────────────────────────────────────────────\n"
            "unclear\n\n"
            "─ Worked for 2s ─────────────────────────────────────────────────────────────\n"
        )
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 11 inferred weak",
                "tasks": [{"goal": "Produce weak summary", "prompt": "Finish with only: unclear"}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                review_resp = await cli.get(f"/v1/dev/execution-plans/{plan_id}/review?include_synthesis=false")
                assert review_resp.status == 200
                review_data = await review_resp.json()
                apply_resp = await cli.post(
                    f"/v1/dev/execution-plans/{plan_id}/apply-review",
                    json={"include_synthesis": False, "message": "Please provide concrete findings."},
                )
                assert apply_resp.status == 200
                apply_data = await apply_resp.json()

        assert review_data["status"] == "needs_review"
        assert review_data["status_payload"]["tasks"][0]["status"] == "needs_review"
        assert review_data["status_payload"]["tasks"][0]["summary"] == "unclear"
        assert review_data["review_status"] == "needs_follow_up"
        assert review_data["recommended_action"] == "follow_up"
        assert apply_data["applied_action"] == "follow_up"
        assert apply_data["results"][0]["action"] == "follow-up"
        assert bridge.sent_messages == [("oryn-workspace-9", "Please provide concrete findings.")]
        events = adapter._subagent_event_store.list_events(ao_session_id="oryn-workspace-9")
        assert any(event.get("transcript_inferred_completion") is True for event in events)
        assert events[-1]["event"] == "subagent.action"
        assert events[-1]["action"] == "follow-up"
        assert events[-1]["launch_task_id"] == task_id

    @pytest.mark.asyncio
    async def test_dev_execution_plan_apply_review_infers_weak_completion_from_codex_idle_prompt(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "spawning"
        bridge.session.summary = None
        bridge.capture_output = lambda session, lines=160: (
            "## Task Brief\n"
            "Do not inspect files. Finish quickly with only this final summary: unclear\n\n"
            "• unclear\n\n\n"
            "› Improve documentation in @filename\n\n"
            "  gpt-5.5 medium · ~/.worktrees/OrynWorkspace/oryn-workspace-20"
        )
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 11 codex idle",
                "tasks": [{"goal": "Produce weak summary", "prompt": "Finish with only: unclear"}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                review_resp = await cli.get(f"/v1/dev/execution-plans/{plan_id}/review?include_synthesis=false")
                assert review_resp.status == 200
                review_data = await review_resp.json()
                apply_resp = await cli.post(
                    f"/v1/dev/execution-plans/{plan_id}/apply-review",
                    json={"include_synthesis": False, "message": "Please provide concrete findings."},
                )
                assert apply_resp.status == 200
                apply_data = await apply_resp.json()

        assert review_data["status"] == "needs_review"
        assert review_data["status_payload"]["tasks"][0]["status"] == "needs_review"
        assert review_data["status_payload"]["tasks"][0]["summary"] == "unclear"
        assert review_data["review_status"] == "needs_follow_up"
        assert review_data["recommended_action"] == "follow_up"
        assert apply_data["applied_action"] == "follow_up"
        assert apply_data["results"][0]["action"] == "follow-up"
        assert bridge.sent_messages == [("oryn-workspace-9", "Please provide concrete findings.")]
        events = adapter._subagent_event_store.list_events(ao_session_id="oryn-workspace-9")
        assert any(event.get("transcript_inferred_completion") is True for event in events)
        assert events[-1]["event"] == "subagent.action"
        assert events[-1]["action"] == "follow-up"
        assert events[-1]["launch_task_id"] == task_id

    @pytest.mark.asyncio
    async def test_dev_execution_plan_ignores_codex_status_line_as_completion_summary(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "spawning"
        bridge.session.summary = "Working (6s • esc to interrupt)"
        bridge.capture_output = lambda session, lines=160: (
            "## Task Brief\n"
            "Finish quickly with only this final summary: unclear\n\n"
            "Working (6s • esc to interrupt)\n"
            "──────────────────────────────────────────────────────────────────────────────\n"
            "unclear\n\n"
            "─ Worked for 6s ─────────────────────────────────────────────────────────────\n"
        )
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 14 status-line weak summary",
                "tasks": [{"goal": "Produce weak summary", "prompt": "Finish with only: unclear"}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                review_resp = await cli.get(f"/v1/dev/execution-plans/{plan_id}/review?include_synthesis=false")
                assert review_resp.status == 200
                review_data = await review_resp.json()

        task_data = review_data["status_payload"]["tasks"][0]
        assert review_data["status"] == "needs_review"
        assert task_data["status"] == "needs_review"
        assert task_data["summary"] == "unclear"
        assert task_data["summary_quality"] == "warning"
        assert "too short" in task_data["summary_warning"]
        assert review_data["review_status"] == "needs_follow_up"
        assert review_data["recommended_action"] == "follow_up"
        events = adapter._subagent_event_store.list_events(ao_session_id="oryn-workspace-9")
        assert any(event.get("transcript_inferred_completion") is True for event in events)
        assert events[-1]["summary"] == "unclear"
        assert events[-1]["launch_task_id"] == task_id

    @pytest.mark.asyncio
    async def test_dev_execution_plan_apply_review_failed_repair_retry_spawns_replacement(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "errored"
        target_session = AOSession(
            id="oryn-workspace-10",
            project_id="OrynWorkspace",
            status="working",
            branch="repair/phase-11",
            workspace_path="/tmp/oryn-workspace-10",
            tmux_name="tmux-oryn-workspace-10",
            agent="codex",
            model="gpt-5.5",
            reasoning_effort="high",
            open_command="tmux attach -t tmux-oryn-workspace-10",
        )

        def _spawn(**kwargs):
            bridge.spawn_kwargs = kwargs
            return target_session

        bridge.spawn = _spawn
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 11 repair",
                "tasks": [{"goal": "Repair failed task", "prompt": "Implement the fix."}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                resp = await cli.post(
                    f"/v1/dev/execution-plans/{plan_id}/apply-review",
                    json={"include_synthesis": False, "instruction": "Avoid the previous failure."},
                )
                assert resp.status == 200
                data = await resp.json()

        assert data["applied_action"] == "repair_retry"
        assert data["results"][0]["action"] == "repair-retry"
        assert data["results"][0]["target_ao_session_id"] == "oryn-workspace-10"
        assert "Recovery diagnostics:" in bridge.spawn_kwargs["prompt"]
        assert "latest worker output" in bridge.spawn_kwargs["prompt"]
        assert adapter._dev_execution_store.get_plan(plan_id)["tasks"][0]["ao_session_id"] == "oryn-workspace-10"
        prompt_meta = adapter._subagent_event_store.get_ao_prompt("oryn-workspace-10")
        assert prompt_meta["launch_plan_id"] == plan_id
        assert prompt_meta["launch_task_id"] == task_id
        events = adapter._subagent_event_store.list_events(ao_session_id="oryn-workspace-9")
        assert events[-1]["event"] == "subagent.action"
        assert events[-1]["action"] == "repair-retry"
        assert events[-1]["target_ao_session_id"] == "oryn-workspace-10"

    @pytest.mark.asyncio
    async def test_dev_execution_plan_apply_review_not_ready_is_no_op(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 11 running",
                "tasks": [{"goal": "Keep running", "prompt": "Keep working."}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                resp = await cli.post(f"/v1/dev/execution-plans/{plan_id}/apply-review", json={"include_synthesis": False})
                assert resp.status == 200
                data = await resp.json()

        assert data["applied_action"] == "none"
        assert data["status"] == "no_op"
        assert data["results"] == []
        assert data["skipped"][0]["target_task_ids"] == [task_id]
        assert bridge.sent_messages == []
        assert bridge.spawn_kwargs is None
        assert adapter._subagent_event_store.list_events(ao_session_id="oryn-workspace-9") == []

    @pytest.mark.asyncio
    async def test_dev_apply_execution_plan_review_tool_matches_api_shape(self, adapter, tmp_path):
        from tools.dev_execution_tools import _handle_dev_apply_execution_plan_review

        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "done"
        bridge.session.summary = "PHASE11_TOOL_DONE Verified the tool path with no unresolved gaps."
        plan = adapter._dev_execution_store.create_plan(
            title="Phase 11 tool",
            vision_brief=None,
            tasks=[{"goal": "Return PHASE11_TOOL_DONE", "prompt": "Return PHASE11_TOOL_DONE when finished."}],
        )
        plan_id = plan["plan_id"]
        task_id = plan["tasks"][0]["task_id"]
        adapter._dev_execution_store.update_task_launch(
            plan_id=plan_id,
            task_id=task_id,
            ao_session_id="oryn-workspace-9",
        )

        with patch("tools.dev_execution_tools.DevExecutionStore", return_value=adapter._dev_execution_store), \
             patch("tools.dev_execution_tools.SubagentEventStore", return_value=adapter._subagent_event_store), \
             patch("tools.ao_bridge.AOBridge", return_value=bridge):
            raw = _handle_dev_apply_execution_plan_review({
                "plan_id": plan_id,
                "include_synthesis": False,
            })
            data = json.loads(raw)

        assert data["object"] == "hermes.dev_execution_plan_review_application"
        assert data["applied_action"] == "accept"
        assert data["results"][0]["task_id"] == task_id
        assert data["results"][0]["action"] == "accept"

    @pytest.mark.asyncio
    async def test_dev_execution_plan_supervise_accept_applies_and_records_history(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "done"
        bridge.session.summary = "PHASE12_SUPERVISOR_ACCEPT_DONE Verified clean result with no unresolved gaps."
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 12 accepted",
                "tasks": [{"goal": "Return PHASE12_SUPERVISOR_ACCEPT_DONE", "prompt": "Return PHASE12_SUPERVISOR_ACCEPT_DONE."}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                resp = await cli.post("/v1/dev/execution-plans/supervise", json={
                    "plan_ids": [plan_id],
                    "include_synthesis": False,
                })
                assert resp.status == 200
                data = await resp.json()
                plans_resp = await cli.get("/v1/dev/execution-plans")
                plans_data = await plans_resp.json()

        assert data["object"] == "hermes.dev_execution_plan_supervision_run"
        assert data["plans"][0]["review_status"] == "accepted"
        assert data["plans"][0]["supervisor_status"] == "applied"
        assert data["applied"][0]["action"] == "accept"
        events = adapter._subagent_event_store.list_events(ao_session_id="oryn-workspace-9")
        assert events[-1]["event"] == "subagent.action"
        assert events[-1]["action"] == "accept"
        plan = next(item for item in plans_data["data"] if item["plan_id"] == plan_id)
        assert plan["supervisor_status"] == "applied"
        assert plan["supervisor_last_action"] == "accept"
        assert plan["supervisor_last_run_id"] == data["run_id"]

    @pytest.mark.asyncio
    async def test_dev_execution_plan_supervise_follow_up_applies_when_available(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "done"
        bridge.session.summary = "unclear"
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 12 follow-up",
                "tasks": [{"goal": "Return PHASE12_FOLLOW_UP_DONE", "prompt": "Return PHASE12_FOLLOW_UP_DONE."}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                resp = await cli.post("/v1/dev/execution-plans/supervise", json={"plan_ids": [plan_id]})
                assert resp.status == 200
                data = await resp.json()

        assert data["plans"][0]["review_status"] == "needs_follow_up"
        assert data["plans"][0]["supervisor_status"] == "applied"
        assert data["applied"][0]["action"] == "follow_up"
        assert bridge.sent_messages == [(
            "oryn-workspace-9",
            "Please provide a concise final implementation summary with verification evidence and unresolved gaps.",
        )]
        events = adapter._subagent_event_store.list_events(ao_session_id="oryn-workspace-9")
        assert events[-1]["event"] == "subagent.action"
        assert events[-1]["action"] == "follow-up"
        assert events[-1]["launch_task_id"] == task_id

    @pytest.mark.asyncio
    async def test_dev_execution_plan_supervise_does_not_repeat_follow_up(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "done"
        bridge.session.summary = "unclear"
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 12 follow-up idempotency",
                "tasks": [{"goal": "Return PHASE12_FOLLOW_UP_REPEAT_DONE", "prompt": "Return PHASE12_FOLLOW_UP_REPEAT_DONE."}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                first = await cli.post("/v1/dev/execution-plans/supervise", json={"plan_ids": [plan_id]})
                assert first.status == 200
                second = await cli.post("/v1/dev/execution-plans/supervise", json={"plan_ids": [plan_id]})
                assert second.status == 200
                data = await second.json()

        assert len(bridge.sent_messages) == 1
        assert data["plans"][0]["supervisor_status"] == "skipped"
        assert data["plans"][0]["supervisor_last_message"] == "Follow-up limit has been reached."
        assert data["plans"][0]["max_follow_ups_reached"] is True
        events = adapter._subagent_event_store.list_events(ao_session_id="oryn-workspace-9")
        follow_up_events = [
            event for event in events
            if event.get("event") == "subagent.action" and event.get("action") == "follow-up"
        ]
        assert len(follow_up_events) == 1
        assert follow_up_events[0]["launch_task_id"] == task_id

    @pytest.mark.asyncio
    async def test_dev_execution_plan_supervise_repair_retry_requires_manual_approval(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "errored"
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 12 failed",
                "tasks": [{"goal": "Repair failed worker", "prompt": "Implement the fix."}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                resp = await cli.post("/v1/dev/execution-plans/supervise", json={"plan_ids": [plan_id]})
                assert resp.status == 200
                data = await resp.json()

        assert data["plans"][0]["review_status"] == "retry_recommended"
        assert data["plans"][0]["recommended_action"] == "repair_retry"
        assert data["plans"][0]["supervisor_status"] == "skipped"
        assert data["plans"][0]["supervisor_approval_status"] == "pending"
        assert data["skipped"][0]["message"] == "repair-retry requires manual approval. Approval required."
        assert data["skipped"][0]["approval_status"] == "pending"
        approval_id = data["skipped"][0]["approval_id"]
        approval = adapter._dev_execution_store.get_supervisor_approval(approval_id)
        assert approval["plan_id"] == plan_id
        assert approval["recommended_action"] == "repair_retry"
        assert approval["task_ids"] == [task_id]
        assert bridge.spawn_kwargs is None
        assert adapter._subagent_event_store.list_events(ao_session_id="oryn-workspace-9") == []

    @pytest.mark.asyncio
    async def test_dev_execution_plan_supervise_reuses_pending_repair_retry_approval(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "errored"
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 13 reuse approval",
                "tasks": [{"goal": "Repair failed worker", "prompt": "Implement the fix."}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                first = await cli.post("/v1/dev/execution-plans/supervise", json={"plan_ids": [plan_id]})
                assert first.status == 200
                first_data = await first.json()
                second = await cli.post("/v1/dev/execution-plans/supervise", json={"plan_ids": [plan_id]})
                assert second.status == 200
                second_data = await second.json()

        first_approval_id = first_data["skipped"][0]["approval_id"]
        second_approval_id = second_data["skipped"][0]["approval_id"]
        assert second_approval_id == first_approval_id
        approvals = adapter._dev_execution_store.list_supervisor_approvals(plan_id=plan_id)
        assert len(approvals) == 1
        assert approvals[0]["task_ids"] == [task_id]

    @pytest.mark.asyncio
    async def test_dev_supervisor_approval_approve_then_apply_consumes_and_spawns_repair_retry(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "errored"
        target_session = AOSession(
            id="oryn-workspace-10",
            project_id="OrynWorkspace",
            status="working",
            branch="repair/phase-13",
            workspace_path="/tmp/oryn-workspace-10",
            tmux_name="tmux-oryn-workspace-10",
            agent="codex",
            model="gpt-5.5",
            reasoning_effort="high",
            open_command="tmux attach -t tmux-oryn-workspace-10",
        )

        def _spawn(**kwargs):
            bridge.spawn_kwargs = kwargs
            return target_session

        bridge.spawn = _spawn
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 13 apply approval",
                "tasks": [{"goal": "Repair failed worker", "prompt": "Implement the fix."}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                supervise_resp = await cli.post("/v1/dev/execution-plans/supervise", json={"plan_ids": [plan_id]})
                assert supervise_resp.status == 200
                supervise_data = await supervise_resp.json()
                approval_id = supervise_data["skipped"][0]["approval_id"]

                approve_resp = await cli.post(
                    f"/v1/dev/supervisor/approvals/{approval_id}/approve",
                    json={"resolved_by": "dev-test", "instruction": "Use the transcript tail."},
                )
                assert approve_resp.status == 200
                approve_data = await approve_resp.json()

                apply_resp = await cli.post(
                    f"/v1/dev/supervisor/approvals/{approval_id}/apply",
                    json={"include_synthesis": False},
                )
                assert apply_resp.status == 200
                apply_data = await apply_resp.json()

        assert approve_data["approval"]["status"] == "approved"
        assert apply_data["status"] == "applied"
        assert apply_data["approval"]["status"] == "consumed"
        assert apply_data["application"]["applied_action"] == "repair_retry"
        assert apply_data["application"]["results"][0]["target_ao_session_id"] == "oryn-workspace-10"
        assert bridge.spawn_kwargs is not None
        assert "Use the transcript tail." in bridge.spawn_kwargs["prompt"]
        assert adapter._dev_execution_store.get_plan(plan_id)["tasks"][0]["ao_session_id"] == "oryn-workspace-10"
        events = adapter._subagent_event_store.list_events(ao_session_id="oryn-workspace-9")
        assert events[-1]["event"] == "subagent.action"
        assert events[-1]["action"] == "repair-retry"
        assert events[-1]["launch_task_id"] == task_id

    @pytest.mark.asyncio
    async def test_dev_supervisor_denied_approval_cannot_be_applied(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "errored"
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 13 deny approval",
                "tasks": [{"goal": "Repair failed worker", "prompt": "Implement the fix."}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                supervise_resp = await cli.post("/v1/dev/execution-plans/supervise", json={"plan_ids": [plan_id]})
                approval_id = (await supervise_resp.json())["skipped"][0]["approval_id"]
                deny_resp = await cli.post(f"/v1/dev/supervisor/approvals/{approval_id}/deny", json={"message": "Not now."})
                assert deny_resp.status == 200
                apply_resp = await cli.post(f"/v1/dev/supervisor/approvals/{approval_id}/apply", json={"include_synthesis": False})
                assert apply_resp.status == 409
                apply_data = await apply_resp.json()

        assert apply_data["status"] == "rejected"
        assert apply_data["approval"]["status"] == "denied"
        assert bridge.spawn_kwargs is None
        assert adapter._subagent_event_store.list_events(ao_session_id="oryn-workspace-9") == []

    @pytest.mark.asyncio
    async def test_dev_supervisor_expired_approval_cannot_be_approved_or_applied(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "errored"
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 13 expired approval",
                "tasks": [{"goal": "Repair failed worker", "prompt": "Implement the fix."}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                supervise_resp = await cli.post("/v1/dev/execution-plans/supervise", json={"plan_ids": [plan_id]})
                approval_id = (await supervise_resp.json())["skipped"][0]["approval_id"]

                with adapter._dev_execution_store._lock, adapter._dev_execution_store._conn:
                    adapter._dev_execution_store._conn.execute(
                        "UPDATE dev_execution_supervisor_approvals SET expires_at = ? WHERE approval_id = ?",
                        (_time.time() - 1, approval_id),
                    )

                approve_resp = await cli.post(f"/v1/dev/supervisor/approvals/{approval_id}/approve", json={})
                assert approve_resp.status == 409
                apply_resp = await cli.post(f"/v1/dev/supervisor/approvals/{approval_id}/apply", json={"include_synthesis": False})
                assert apply_resp.status == 409
                apply_data = await apply_resp.json()

        assert apply_data["status"] == "rejected"
        assert apply_data["approval"]["status"] == "expired"
        assert bridge.spawn_kwargs is None

    @pytest.mark.asyncio
    async def test_dev_supervisor_approval_apply_is_single_use(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "errored"
        target_session = AOSession(
            id="oryn-workspace-10",
            project_id="OrynWorkspace",
            status="working",
            branch="repair/phase-13",
            workspace_path="/tmp/oryn-workspace-10",
            tmux_name="tmux-oryn-workspace-10",
            agent="codex",
            model="gpt-5.5",
            reasoning_effort="medium",
            open_command="tmux attach -t tmux-oryn-workspace-10",
        )
        spawn_calls = []

        def _spawn(**kwargs):
            spawn_calls.append(kwargs)
            return target_session

        bridge.spawn = _spawn
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 13 single use",
                "tasks": [{"goal": "Repair failed worker", "prompt": "Implement the fix."}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                supervise_resp = await cli.post("/v1/dev/execution-plans/supervise", json={"plan_ids": [plan_id]})
                approval_id = (await supervise_resp.json())["skipped"][0]["approval_id"]
                approve_resp = await cli.post(f"/v1/dev/supervisor/approvals/{approval_id}/approve", json={})
                assert approve_resp.status == 200
                first_apply = await cli.post(f"/v1/dev/supervisor/approvals/{approval_id}/apply", json={"include_synthesis": False})
                assert first_apply.status == 200
                second_apply = await cli.post(f"/v1/dev/supervisor/approvals/{approval_id}/apply", json={"include_synthesis": False})
                assert second_apply.status == 409
                second_data = await second_apply.json()

        assert len(spawn_calls) == 1
        assert second_data["status"] == "rejected"
        assert second_data["approval"]["status"] == "consumed"

    @pytest.mark.asyncio
    async def test_dev_execution_plan_supervise_running_is_not_ready_no_op(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 12 running",
                "tasks": [{"goal": "Keep running", "prompt": "Keep running."}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                resp = await cli.post("/v1/dev/execution-plans/supervise", json={"plan_ids": [plan_id]})
                assert resp.status == 200
                data = await resp.json()

        assert data["plans"][0]["review_status"] == "not_ready"
        assert data["plans"][0]["recommended_action"] == "none"
        assert data["plans"][0]["supervisor_status"] == "skipped"
        assert data["skipped"][0]["message"] == "Execution plan still has tasks that are not terminal."
        assert bridge.sent_messages == []
        assert adapter._subagent_event_store.list_events(ao_session_id="oryn-workspace-9") == []

    @pytest.mark.asyncio
    async def test_dev_supervisor_loop_is_project_opt_in(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "done"
        bridge.session.summary = "unclear"
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 15 opt in",
                "tasks": [{"goal": "Return unclear", "prompt": "Return unclear.", "project_id": "OrynWorkspace"}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                disabled_tick = supervisor_loop_tick(
                    store=adapter._dev_execution_store,
                    event_store=adapter._subagent_event_store,
                    now=1_000,
                )
                enable_resp = await cli.post("/v1/dev/supervisor/loop", json={
                    "project_id": "OrynWorkspace",
                    "supervisor_enabled": True,
                    "supervisor_interval_seconds": 60,
                    "supervisor_limit": 10,
                })
                assert enable_resp.status == 200
                enabled_data = await enable_resp.json()
                enabled_tick = supervisor_loop_tick(
                    store=adapter._dev_execution_store,
                    event_store=adapter._subagent_event_store,
                    now=2_000,
                )

        assert disabled_tick["tick_count"] == 0
        assert enabled_data["loop"]["supervisor_enabled"] is True
        assert enabled_tick["tick_count"] == 1
        assert enabled_tick["ticks"][0]["result"]["applied"][0]["action"] == "follow_up"
        assert bridge.sent_messages
        plan = adapter._dev_execution_store.get_plan(plan_id)
        assert plan["supervisor_enabled"] is True
        assert plan["supervisor_loop_status"] == "completed"
        events = adapter._subagent_event_store.list_events(ao_session_id="oryn-workspace-9")
        assert events[-1]["event"] == "subagent.action"
        assert events[-1]["action"] == "follow-up"
        assert events[-1]["launch_task_id"] == task_id

    @pytest.mark.asyncio
    async def test_dev_supervisor_loop_creates_retry_approval_without_spawning(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "errored"
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            runbook_resp = await cli.post("/v1/dev/runbooks", json={
                "project_id": "OrynWorkspace",
                "policy_profile": "standard",
                "supervisor_enabled": True,
            })
            assert runbook_resp.status == 200
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 15 approval",
                "tasks": [{"goal": "Repair failed worker", "prompt": "Implement the fix.", "project_id": "OrynWorkspace"}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                first_tick = supervisor_loop_tick(
                    store=adapter._dev_execution_store,
                    event_store=adapter._subagent_event_store,
                    now=3_000,
                )
                second_tick = supervisor_loop_tick(
                    store=adapter._dev_execution_store,
                    event_store=adapter._subagent_event_store,
                    now=3_061,
                )

        assert first_tick["ticks"][0]["result"]["skipped"][0]["approval_status"] == "pending"
        assert first_tick["ticks"][0]["result"]["skipped"][0]["action"] == "repair_retry"
        assert bridge.spawn_kwargs is None
        approvals = adapter._dev_execution_store.list_supervisor_approvals(plan_id=plan_id)
        assert len(approvals) == 1
        assert approvals[0]["task_ids"] == [task_id]
        assert second_tick["ticks"][0]["result"]["skipped"][0]["approval_id"] == approvals[0]["approval_id"]
        assert adapter._dev_execution_store.get_plan(plan_id)["supervisor_approval_status"] == "pending"

    @pytest.mark.asyncio
    async def test_dev_execution_plan_test_state_completed_weak_gets_bounded_follow_up(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.send = MagicMock(side_effect=AssertionError("fixture follow-up must not call AO"))
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            runbook_resp = await cli.post("/v1/dev/runbooks", json={
                "project_id": "OrynWorkspace",
                "policy_profile": "standard",
                "supervisor_enabled": True,
            })
            assert runbook_resp.status == 200
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 16 weak fixture",
                "tasks": [{"goal": "Fixture weak", "prompt": "Return weak.", "project_id": "OrynWorkspace"}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            state_resp = await cli.post(f"/v1/dev/execution-plans/{plan_id}/test-state", json={
                "task_id": task_id,
                "state": "completed_weak",
            })
            assert state_resp.status == 200
            state_data = await state_resp.json()

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                first_tick = supervisor_loop_tick(
                    store=adapter._dev_execution_store,
                    event_store=adapter._subagent_event_store,
                    now=4_000,
                )
                second_tick = supervisor_loop_tick(
                    store=adapter._dev_execution_store,
                    event_store=adapter._subagent_event_store,
                    now=4_061,
                )
                status_resp = await cli.get(f"/v1/dev/execution-plans/{plan_id}/status")
                status_data = await status_resp.json()

        assert state_data["runtime"] == "fixture"
        assert state_data["ao_session_id"] == f"fixture-{task_id}"
        assert status_data["status"] == "needs_review"
        assert status_data["tasks"][0]["summary_quality"] == "warning"
        assert first_tick["ticks"][0]["result"]["applied"][0]["action"] == "follow_up"
        assert second_tick["ticks"][0]["result"]["skipped"][0]["message"] == "Follow-up limit has been reached."
        follow_up_events = [
            event
            for event in adapter._subagent_event_store.list_events(ao_session_id=f"fixture-{task_id}")
            if event.get("action") == "follow-up"
        ]
        assert len(follow_up_events) == 1
        assert follow_up_events[0]["runtime"] == "fixture"
        assert follow_up_events[0]["fixture"] is True
        assert follow_up_events[0]["action_status"] == "succeeded"
        bridge.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_dev_execution_plan_test_state_completed_ok_auto_accepts(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 16 ok fixture",
                "tasks": [{"goal": "Return PHASE16_FIXTURE_OK_DONE", "prompt": "Return PHASE16_FIXTURE_OK_DONE.", "project_id": "OrynWorkspace"}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            state_resp = await cli.post(f"/v1/dev/execution-plans/{plan_id}/test-state", json={
                "task_id": task_id,
                "state": "completed_ok",
            })
            assert state_resp.status == 200

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                review_resp = await cli.get(f"/v1/dev/execution-plans/{plan_id}/review?include_synthesis=false")
                review_data = await review_resp.json()
                supervise_resp = await cli.post("/v1/dev/execution-plans/supervise", json={"plan_ids": [plan_id]})
                supervise_data = await supervise_resp.json()

        assert review_data["review_status"] == "accepted"
        assert review_data["recommended_action"] == "accept"
        assert supervise_data["applied"][0]["action"] == "accept"
        events = adapter._subagent_event_store.list_events(ao_session_id=f"fixture-{task_id}")
        assert events[-1]["event"] == "subagent.action"
        assert events[-1]["action"] == "accept"

    @pytest.mark.asyncio
    async def test_dev_execution_plan_test_state_failed_repairable_creates_approval(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            runbook_resp = await cli.post("/v1/dev/runbooks", json={
                "project_id": "OrynWorkspace",
                "policy_profile": "standard",
                "supervisor_enabled": True,
            })
            assert runbook_resp.status == 200
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 16 repairable fixture",
                "tasks": [{"goal": "Fixture failed", "prompt": "Do work.", "project_id": "OrynWorkspace"}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            state_resp = await cli.post(f"/v1/dev/execution-plans/{plan_id}/test-state", json={
                "task_id": task_id,
                "state": "failed_repairable",
            })
            assert state_resp.status == 200

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                tick = supervisor_loop_tick(
                    store=adapter._dev_execution_store,
                    event_store=adapter._subagent_event_store,
                    now=5_000,
                )

        assert tick["ticks"][0]["result"]["skipped"][0]["approval_status"] == "pending"
        assert tick["ticks"][0]["result"]["skipped"][0]["action"] == "repair_retry"
        assert bridge.spawn_kwargs is None
        approvals = adapter._dev_execution_store.list_supervisor_approvals(plan_id=plan_id)
        assert len(approvals) == 1
        assert approvals[0]["task_ids"] == [task_id]

    @pytest.mark.asyncio
    async def test_dev_execution_plan_test_state_failed_unrepairable_requires_human_review(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 16 unrepairable fixture",
                "tasks": [{"goal": "Fixture failed", "prompt": "Do work.", "project_id": "OrynWorkspace"}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            state_resp = await cli.post(f"/v1/dev/execution-plans/{plan_id}/test-state", json={
                "task_id": task_id,
                "state": "failed_unrepairable",
            })
            assert state_resp.status == 200

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                review_resp = await cli.get(f"/v1/dev/execution-plans/{plan_id}/review?include_synthesis=false")
                review_data = await review_resp.json()
                supervise_resp = await cli.post("/v1/dev/execution-plans/supervise", json={"plan_ids": [plan_id]})
                supervise_data = await supervise_resp.json()

        assert review_data["review_status"] == "human_review_required"
        assert review_data["recommended_action"] == "human_review"
        assert supervise_data["skipped"][0]["action"] == "human_review"
        assert adapter._dev_execution_store.list_supervisor_approvals(plan_id=plan_id) == []

    @pytest.mark.asyncio
    async def test_dev_execution_plan_test_state_running_is_not_ready_and_board_decodes_fixture(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 16 running fixture",
                "tasks": [{"goal": "Fixture running", "prompt": "Keep running.", "project_id": "OrynWorkspace"}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            state_resp = await cli.post(f"/v1/dev/execution-plans/{plan_id}/test-state", json={
                "task_id": task_id,
                "state": "running",
            })
            assert state_resp.status == 200

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                review_resp = await cli.get(f"/v1/dev/execution-plans/{plan_id}/review?include_synthesis=false")
                review_data = await review_resp.json()
                board_resp = await cli.get("/v1/subagents/board?runtime=fixture")
                board_data = await board_resp.json()

        assert review_data["review_status"] == "not_ready"
        assert review_data["recommended_action"] == "none"
        assert board_data["data"][0]["runtime"] == "fixture"
        assert board_data["data"][0]["can_open"] is False
        assert board_data["data"][0]["can_stop"] is False

    @pytest.mark.asyncio
    async def test_dev_execution_plan_supervise_dry_run_does_not_append_actions(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "done"
        bridge.session.summary = "PHASE12_SUPERVISOR_DRY_RUN_DONE Verified dry run with no unresolved gaps."
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 12 dry run",
                "tasks": [{"goal": "Return PHASE12_SUPERVISOR_DRY_RUN_DONE", "prompt": "Return PHASE12_SUPERVISOR_DRY_RUN_DONE."}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                resp = await cli.post("/v1/dev/execution-plans/supervise", json={
                    "plan_ids": [plan_id],
                    "apply_guarded_actions": False,
                })
                assert resp.status == 200
                data = await resp.json()

        assert data["plans"][0]["review_status"] == "accepted"
        assert data["plans"][0]["supervisor_status"] == "observed"
        assert data["skipped"][0]["status"] == "observed"
        assert adapter._subagent_event_store.list_events(ao_session_id="oryn-workspace-9") == []

    @pytest.mark.asyncio
    async def test_dev_execution_plan_status_includes_default_standard_runbook_and_next_step(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 14 default policy",
                "tasks": [{"goal": "Keep running", "prompt": "Keep running.", "project_id": "OrynWorkspace"}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                resp = await cli.get(f"/v1/dev/execution-plans/{plan_id}/status")
                assert resp.status == 200
                data = await resp.json()
                next_resp = await cli.get(f"/v1/dev/execution-plans/{plan_id}/next-step")
                assert next_resp.status == 200
                next_data = await next_resp.json()

        assert data["policy_profile"] == "standard"
        assert data["policy_source"] == "global"
        assert data["max_follow_ups_per_task"] == 1
        assert data["next_step"] == "wait"
        assert data["plan"]["policy_profile"] == "standard"
        assert data["plan"]["follow_up_count"] == 0
        assert next_data["next_step"] == "wait"
        assert next_data["runbook"]["policy_profile"] == "standard"

    @pytest.mark.asyncio
    async def test_dev_execution_plan_supervise_standard_policy_follow_up_limit_one(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "done"
        bridge.session.summary = "unclear"
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 14 standard follow-up limit",
                "tasks": [{"goal": "Return PHASE14_FOLLOW_UP_LIMIT_DONE", "prompt": "Return unclear.", "project_id": "OrynWorkspace"}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                first = await cli.post("/v1/dev/execution-plans/supervise", json={"plan_ids": [plan_id]})
                assert first.status == 200
                first_data = await first.json()
                second = await cli.post("/v1/dev/execution-plans/supervise", json={"plan_ids": [plan_id]})
                assert second.status == 200
                second_data = await second.json()

        assert first_data["plans"][0]["policy_profile"] == "standard"
        assert first_data["plans"][0]["supervisor_status"] == "applied"
        assert first_data["plans"][0]["follow_up_count"] == 0
        assert second_data["plans"][0]["supervisor_status"] == "skipped"
        assert second_data["plans"][0]["supervisor_last_message"] == "Follow-up limit has been reached."
        assert second_data["plans"][0]["max_follow_ups_reached"] is True
        assert len(bridge.sent_messages) == 1
        events = adapter._subagent_event_store.list_events(ao_session_id="oryn-workspace-9")
        follow_up_events = [
            event for event in events
            if event.get("event") == "subagent.action" and event.get("action") == "follow-up"
        ]
        assert len(follow_up_events) == 1
        assert follow_up_events[0]["launch_task_id"] == task_id

    @pytest.mark.asyncio
    async def test_dev_execution_plan_supervise_conservative_policy_does_not_auto_follow_up(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "done"
        bridge.session.summary = "unclear"
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            runbook_resp = await cli.post("/v1/dev/runbooks", json={
                "project_id": "OrynWorkspace",
                "policy_profile": "conservative",
            })
            assert runbook_resp.status == 200
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 14 conservative follow-up",
                "tasks": [{"goal": "Return unclear", "prompt": "Return unclear.", "project_id": "OrynWorkspace"}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                resp = await cli.post("/v1/dev/execution-plans/supervise", json={"plan_ids": [plan_id]})
                assert resp.status == 200
                data = await resp.json()
                next_resp = await cli.get(f"/v1/dev/execution-plans/{plan_id}/next-step")
                next_data = await next_resp.json()

        assert data["plans"][0]["policy_profile"] == "conservative"
        assert data["plans"][0]["policy_source"] == "project"
        assert data["plans"][0]["supervisor_status"] == "skipped"
        assert data["plans"][0]["supervisor_last_message"] == "Current runbook does not auto-send follow-ups."
        assert bridge.sent_messages == []
        assert next_data["next_step"] == "ask_human"

    @pytest.mark.asyncio
    async def test_dev_next_execution_step_reports_pending_and_approved_approval_states(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "errored"
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/execution-plans", json={
                "title": "Phase 14 next approval",
                "tasks": [{"goal": "Repair failed worker", "prompt": "Implement the fix.", "project_id": "OrynWorkspace"}],
            })
            create_data = await create_resp.json()
            plan_id = create_data["plan"]["plan_id"]
            task_id = create_data["plan"]["tasks"][0]["task_id"]
            adapter._dev_execution_store.update_task_launch(
                plan_id=plan_id,
                task_id=task_id,
                ao_session_id="oryn-workspace-9",
            )

            with patch("tools.ao_bridge.AOBridge", return_value=bridge):
                supervise_resp = await cli.post("/v1/dev/execution-plans/supervise", json={"plan_ids": [plan_id]})
                approval_id = (await supervise_resp.json())["skipped"][0]["approval_id"]
                pending_resp = await cli.get(f"/v1/dev/execution-plans/{plan_id}/next-step")
                pending_data = await pending_resp.json()
                approve_resp = await cli.post(f"/v1/dev/supervisor/approvals/{approval_id}/approve", json={})
                assert approve_resp.status == 200
                approved_resp = await cli.get(f"/v1/dev/execution-plans/{plan_id}/next-step")
                approved_data = await approved_resp.json()

        assert pending_data["next_step"] == "approve"
        assert pending_data["approval_id"] == approval_id
        assert pending_data["target_task_ids"] == [task_id]
        assert approved_data["next_step"] == "apply_approval"
        assert approved_data["approval_id"] == approval_id

    @pytest.mark.asyncio
    async def test_dev_runbook_tools_and_next_step_tool_match_api_shape(self, adapter, tmp_path):
        from tools.dev_execution_tools import (
            _handle_dev_next_execution_step,
            _handle_dev_runbooks,
            _handle_dev_set_project_runbook,
        )

        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        plan = adapter._dev_execution_store.create_plan(
            title="Phase 14 tool",
            vision_brief=None,
            tasks=[{"goal": "Keep running", "prompt": "Keep running.", "project_id": "OrynWorkspace"}],
        )
        plan_id = plan["plan_id"]
        task_id = plan["tasks"][0]["task_id"]
        adapter._dev_execution_store.update_task_launch(
            plan_id=plan_id,
            task_id=task_id,
            ao_session_id="oryn-workspace-9",
        )

        with patch("tools.dev_execution_tools.DevExecutionStore", return_value=adapter._dev_execution_store), \
             patch("tools.dev_execution_tools.SubagentEventStore", return_value=adapter._subagent_event_store), \
             patch("tools.ao_bridge.AOBridge", return_value=bridge):
            set_raw = _handle_dev_set_project_runbook({
                "project_id": "OrynWorkspace",
                "policy_profile": "aggressive",
            })
            listed = json.loads(_handle_dev_runbooks({"project_id": "OrynWorkspace"}))
            next_data = json.loads(_handle_dev_next_execution_step({"plan_id": plan_id}))

        set_data = json.loads(set_raw)
        assert set_data["runbook"]["policy_profile"] == "aggressive"
        assert listed["data"][0]["project_id"] == "OrynWorkspace"
        assert next_data["object"] == "hermes.dev_execution_plan_next_step"
        assert next_data["next_step"] == "wait"
        assert next_data["runbook"]["policy_profile"] == "aggressive"

    @pytest.mark.asyncio
    async def test_dev_supervise_execution_plans_tool_matches_api_shape(self, adapter, tmp_path):
        from tools.dev_execution_tools import _handle_dev_supervise_execution_plans

        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "done"
        bridge.session.summary = "PHASE12_SUPERVISOR_TOOL_DONE Verified tool supervision with no unresolved gaps."
        plan = adapter._dev_execution_store.create_plan(
            title="Phase 12 tool",
            vision_brief=None,
            tasks=[{"goal": "Return PHASE12_SUPERVISOR_TOOL_DONE", "prompt": "Return PHASE12_SUPERVISOR_TOOL_DONE."}],
        )
        plan_id = plan["plan_id"]
        task_id = plan["tasks"][0]["task_id"]
        adapter._dev_execution_store.update_task_launch(
            plan_id=plan_id,
            task_id=task_id,
            ao_session_id="oryn-workspace-9",
        )

        with patch("tools.dev_execution_tools.DevExecutionStore", return_value=adapter._dev_execution_store), \
             patch("tools.dev_execution_tools.SubagentEventStore", return_value=adapter._subagent_event_store), \
             patch("tools.ao_bridge.AOBridge", return_value=bridge):
            raw = _handle_dev_supervise_execution_plans({"plan_ids": [plan_id]})
            data = json.loads(raw)

        assert data["object"] == "hermes.dev_execution_plan_supervision_run"
        assert data["plans"][0]["plan_id"] == plan_id
        assert data["applied"][0]["action"] == "accept"

    @pytest.mark.asyncio
    async def test_dev_supervisor_approval_tools_match_api_shape(self, adapter, tmp_path):
        from tools.dev_execution_tools import (
            _handle_dev_approve_supervisor_action,
            _handle_dev_apply_supervisor_approval,
            _handle_dev_supervise_execution_plans,
            _handle_dev_supervisor_approvals,
        )

        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "errored"
        target_session = AOSession(
            id="oryn-workspace-10",
            project_id="OrynWorkspace",
            status="working",
            branch="repair/phase-13-tool",
            workspace_path="/tmp/oryn-workspace-10",
            tmux_name="tmux-oryn-workspace-10",
            agent="codex",
            model="gpt-5.5",
            reasoning_effort="medium",
            open_command="tmux attach -t tmux-oryn-workspace-10",
        )

        def _spawn(**kwargs):
            bridge.spawn_kwargs = kwargs
            return target_session

        bridge.spawn = _spawn
        plan = adapter._dev_execution_store.create_plan(
            title="Phase 13 approval tool",
            vision_brief=None,
            tasks=[{"goal": "Repair failed worker", "prompt": "Implement the fix."}],
        )
        plan_id = plan["plan_id"]
        task_id = plan["tasks"][0]["task_id"]
        adapter._dev_execution_store.update_task_launch(
            plan_id=plan_id,
            task_id=task_id,
            ao_session_id="oryn-workspace-9",
        )

        with patch("tools.dev_execution_tools.DevExecutionStore", return_value=adapter._dev_execution_store), \
             patch("tools.dev_execution_tools.SubagentEventStore", return_value=adapter._subagent_event_store), \
             patch("tools.ao_bridge.AOBridge", return_value=bridge):
            supervise = json.loads(_handle_dev_supervise_execution_plans({"plan_ids": [plan_id]}))
            approval_id = supervise["skipped"][0]["approval_id"]
            listed = json.loads(_handle_dev_supervisor_approvals({"plan_id": plan_id}))
            approved = json.loads(_handle_dev_approve_supervisor_action({
                "approval_id": approval_id,
                "instruction": "Use the tool approval.",
            }))
            applied = json.loads(_handle_dev_apply_supervisor_approval({
                "approval_id": approval_id,
                "include_synthesis": False,
            }))

        assert listed["object"] == "list"
        assert listed["data"][0]["approval_id"] == approval_id
        assert approved["approval"]["status"] == "approved"
        assert applied["object"] == "hermes.dev_supervisor_approval_application"
        assert applied["status"] == "applied"
        assert applied["approval"]["status"] == "consumed"
        assert applied["application"]["results"][0]["target_ao_session_id"] == "oryn-workspace-10"
        assert adapter._dev_execution_store.get_plan(plan_id)["tasks"][0]["ao_session_id"] == "oryn-workspace-10"
        events = adapter._subagent_event_store.list_events(ao_session_id="oryn-workspace-9")
        assert events[-1]["launch_task_id"] == task_id

    @pytest.mark.asyncio
    async def test_dev_execution_plan_status_prefers_marker_bullet_over_contract_transcript(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.session.status = "spawning"
        bridge.session.summary = None
        bridge.capture_output = lambda session, lines=160: (
            "Workspace Agent Rules\n\n"
            "## Dev Worker Contract\n"
            "Do not edit files. Return this exact final answer:\n"
            "PHASE11_ACCEPT_BRANCH_DONE Verified clean completion with no unresolved gaps.\n\n"
            "• PHASE11_ACCEPT_BRANCH_DONE Verified clean completion with no unresolved gaps.\n\n"
            "› Improve documentation in @filename\n\n"
            "  gpt-5.5 medium · ~/.worktrees/OrynWorkspace/oryn-workspace-21"
        )
        plan = adapter._dev_execution_store.create_plan(
            title="Phase 11 accept marker extraction",
            vision_brief=None,
            tasks=[{
                "goal": "Produce clean accepted Phase 11 result",
                "prompt": (
                    "Do not edit files. Return this exact final answer:\n"
                    "PHASE11_ACCEPT_BRANCH_DONE Verified clean completion with no unresolved gaps."
                ),
            }],
        )
        task_id = plan["tasks"][0]["task_id"]
        adapter._dev_execution_store.update_task_launch(
            plan_id=plan["plan_id"],
            task_id=task_id,
            ao_session_id="oryn-workspace-9",
        )
        app = _create_runs_app(adapter)

        with patch("tools.ao_bridge.AOBridge", return_value=bridge):
            async with TestClient(TestServer(app)) as cli:
                status_resp = await cli.get(f"/v1/dev/execution-plans/{plan['plan_id']}/status")
                assert status_resp.status == 200
                status_data = await status_resp.json()
                review_resp = await cli.get(f"/v1/dev/execution-plans/{plan['plan_id']}/review?include_synthesis=false")
                assert review_resp.status == 200
                review_data = await review_resp.json()

        assert status_data["status"] == "completed"
        assert status_data["tasks"][0]["status"] == "completed"
        assert status_data["tasks"][0]["summary"] == "PHASE11_ACCEPT_BRANCH_DONE Verified clean completion with no unresolved gaps."
        assert status_data["tasks"][0]["summary_quality"] == "ok"
        assert review_data["review_status"] == "accepted"
        assert review_data["recommended_action"] == "accept"

    def test_dev_execution_plan_marker_extraction_prefers_exact_marker_line(self):
        transcript = (
            "## Dev Launch Profile\n"
            "Prompt says to reply with PHASE19_OPENHANDS_DEV_PLAN_DONE.\n"
            "execution_status: running\n"
            "PHASE19_OPENHANDS_DEV_PLAN_DONE\n"
            "execution_status: finished\n"
            "Please provide a concise final implementation summary.\n"
        )

        assert _extract_completion_summary(
            transcript,
            ["PHASE19_OPENHANDS_DEV_PLAN_DONE"],
        ) == "PHASE19_OPENHANDS_DEV_PLAN_DONE"

    @pytest.mark.asyncio
    async def test_dev_execution_plan_status_scopes_reused_ao_session_events_to_task(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        old_plan = adapter._dev_execution_store.create_plan(
            title="Old plan",
            vision_brief=None,
            tasks=[{"goal": "Old work", "prompt": "Old worker prompt."}],
        )
        new_plan = adapter._dev_execution_store.create_plan(
            title="New plan",
            vision_brief=None,
            tasks=[{"goal": "New work", "prompt": "New worker prompt."}],
        )
        old_task = old_plan["tasks"][0]
        new_task = new_plan["tasks"][0]
        adapter._dev_execution_store.update_task_launch(
            plan_id=old_plan["plan_id"],
            task_id=old_task["task_id"],
            ao_session_id="oryn-workspace-9",
        )
        adapter._subagent_event_store.append_event({
            "event": "subagent.start",
            "subagent_id": "ao:oryn-workspace-9",
            "ao_session_id": "oryn-workspace-9",
            "runtime": "ao",
            "status": "running",
            "goal": "Old work",
            "launch_plan_id": old_plan["plan_id"],
            "launch_task_id": old_task["task_id"],
            "created_at": 100,
        })
        adapter._subagent_event_store.append_event({
            "event": "subagent.action",
            "subagent_id": "ao:oryn-workspace-9",
            "ao_session_id": "oryn-workspace-9",
            "runtime": "ao",
            "status": "killed",
            "action": "stop",
            "action_status": "killed",
            "message": "AO worker stopped by user.",
            "launch_plan_id": old_plan["plan_id"],
            "launch_task_id": old_task["task_id"],
            "created_at": 110,
        })
        adapter._subagent_event_store.append_event({
            "event": "subagent.complete",
            "subagent_id": "ao:oryn-workspace-9",
            "ao_session_id": "oryn-workspace-9",
            "runtime": "ao",
            "status": "killed",
            "summary": "AO worker stopped by user.",
            "launch_plan_id": old_plan["plan_id"],
            "launch_task_id": old_task["task_id"],
            "created_at": 111,
        })
        adapter._dev_execution_store.update_task_launch(
            plan_id=new_plan["plan_id"],
            task_id=new_task["task_id"],
            ao_session_id="oryn-workspace-9",
        )
        adapter._subagent_event_store.append_event({
            "event": "subagent.start",
            "subagent_id": "ao:oryn-workspace-9",
            "ao_session_id": "oryn-workspace-9",
            "runtime": "ao",
            "status": "running",
            "goal": "New work",
            "launch_plan_id": new_plan["plan_id"],
            "launch_task_id": new_task["task_id"],
            "created_at": 200,
        })
        bridge = _FakeAOBridge()
        app = _create_runs_app(adapter)

        with patch("tools.ao_bridge.AOBridge", return_value=bridge):
            async with TestClient(TestServer(app)) as cli:
                old_resp = await cli.get(f"/v1/dev/execution-plans/{old_plan['plan_id']}/status")
                assert old_resp.status == 200
                old_data = await old_resp.json()
                new_resp = await cli.get(f"/v1/dev/execution-plans/{new_plan['plan_id']}/status")
                assert new_resp.status == 200
                new_data = await new_resp.json()

        assert old_data["status"] == "failed"
        assert old_data["tasks"][0]["recent_action"] == "stop"
        assert new_data["status"] == "running"
        assert new_data["tasks"][0]["recent_action"] is None
        assert new_data["tasks"][0]["last_event"]["launch_plan_id"] == new_plan["plan_id"]

    @pytest.mark.asyncio
    async def test_dev_execution_plan_status_syncs_completion_from_transcript_marker(self, adapter, tmp_path):
        adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        plan = adapter._dev_execution_store.create_plan(
            title="Marker completion",
            vision_brief=None,
            tasks=[{
                "goal": "Validate marker completion",
                "prompt": "Inspect without edits. End with PHASE9_CLEAN_DONE.",
                "profile_id": "workspace.inspect",
            }],
        )
        task = plan["tasks"][0]
        adapter._dev_execution_store.update_task_launch(
            plan_id=plan["plan_id"],
            task_id=task["task_id"],
            ao_session_id="oryn-workspace-9",
        )
        adapter._subagent_event_store.append_event({
            "event": "subagent.start",
            "subagent_id": "ao:oryn-workspace-9",
            "ao_session_id": "oryn-workspace-9",
            "runtime": "ao",
            "status": "running",
            "goal": "Validate marker completion",
            "launch_plan_id": plan["plan_id"],
            "launch_task_id": task["task_id"],
        })
        bridge = _FakeAOBridge()
        bridge.capture_output = lambda session, lines=160: (
            "summary:\n"
            "The Dev Plans strip displays derived status from Hermes.\n"
            "PHASE9_CLEAN_DONE\n\n"
            "› Summarize recent commits"
        )
        app = _create_runs_app(adapter)

        with patch("tools.ao_bridge.AOBridge", return_value=bridge):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get(f"/v1/dev/execution-plans/{plan['plan_id']}/status")
                assert resp.status == 200
                data = await resp.json()
                repeat_resp = await cli.get(f"/v1/dev/execution-plans/{plan['plan_id']}/status")
                assert repeat_resp.status == 200

        assert data["status"] == "completed"
        task_data = data["tasks"][0]
        assert task_data["status"] == "completed"
        assert task_data["summary_quality"] == "ok"
        assert "PHASE9_CLEAN_DONE" in task_data["summary"]
        events = adapter._subagent_event_store.list_events(ao_session_id="oryn-workspace-9")
        complete_events = [event for event in events if event["event"] == "subagent.complete"]
        assert len(complete_events) == 1
        assert complete_events[0]["launch_plan_id"] == plan["plan_id"]
        assert complete_events[0]["launch_task_id"] == task["task_id"]

    @pytest.mark.asyncio
    async def test_oryn_project_dashboard_returns_bundled_read_model_with_etag(self, adapter, tmp_path):
        db_path = tmp_path / "state.db"
        adapter._dev_clarification_store = DevClarificationStore(db_path)
        adapter._dev_execution_store = DevExecutionStore(db_path)
        adapter._dev_plan_artifact_store = DevPlanArtifactStore(db_path)
        adapter._subagent_event_store = SubagentEventStore(db_path)
        app = _create_runs_app(adapter)

        bridge = MagicMock()
        bridge.list.return_value = []
        with patch("tools.ao_bridge.AOBridge", return_value=bridge):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/oryn/project-dashboard?project_id=OrynWorkspace")
                assert resp.status == 200
                etag = resp.headers.get("ETag")
                assert etag
                data = await resp.json()
                assert data["object"] == "hermes.oryn.project_dashboard"
                assert data["project_id"] == "OrynWorkspace"
                assert data["clarifications"] == []
                assert data["plan_artifacts"] == []
                assert data["dev_plans"] == []
                assert data["subagent_board"]["data"] == []

                repeat = await cli.get(
                    "/v1/oryn/project-dashboard?project_id=OrynWorkspace",
                    headers={"If-None-Match": etag},
                )
                assert repeat.status == 304

    @pytest.mark.asyncio
    async def test_subagent_board_merges_ao_sessions_and_maps_lanes(self, adapter, tmp_path):
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        adapter._subagent_event_store.upsert_ao_prompt(
            ao_session_id="oryn-workspace-9",
            project_id="OrynWorkspace",
            prompt="Inspect the workspace",
            goal="Inspect with AO",
            issue_id=None,
            branch="feat/board",
            agent="codex",
            model="gpt-5.5",
            reasoning_effort="high",
        )
        adapter._subagent_event_store.append_event({
            "event": "subagent.start",
            "run_id": "run-1",
            "session_id": "session-1",
            "subagent_id": "native-1",
            "runtime": "hermes",
            "depth": 0,
            "goal": "Native work",
            "status": "running",
            "message": "Reading files",
            "context_usage": {
                "session": {"total_tokens": 42},
                "categories": [{"key": "tools", "label": "Tools", "tokens": 12}],
            },
        })
        adapter._subagent_event_store.append_event({
            "event": "subagent.complete",
            "run_id": "run-2",
            "session_id": "session-2",
            "subagent_id": "ao:oryn-workspace-9",
            "ao_session_id": "oryn-workspace-9",
            "runtime": "ao",
            "depth": 0,
            "goal": "Inspect with AO",
            "status": "killed",
            "summary": "Stopped",
        })
        bridge = _FakeAOBridge()
        app = _create_runs_app(adapter)

        with patch("tools.ao_bridge.AOBridge", return_value=bridge):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/subagents/board")
                assert resp.status == 200
                data = await resp.json()

        rows = {item["id"]: item for item in data["data"]}
        with patch("tools.ao_bridge.AOBridge", return_value=bridge):
            direct_rows = {
                item["id"]: item
                for item in build_agent_board_rows(store=adapter._subagent_event_store, params={}, limit=250)
            }
        assert rows["native-1"]["lane"] == "running"
        assert direct_rows["native-1"]["lane"] == rows["native-1"]["lane"]
        assert rows["native-1"]["token_total"] == 42
        assert rows["native-1"]["context_usage_categories"][0]["key"] == "tools"
        assert rows["native-1"]["can_open"] is False
        assert rows["ao:oryn-workspace-9"]["lane"] == "failed"
        assert direct_rows["ao:oryn-workspace-9"]["lane"] == rows["ao:oryn-workspace-9"]["lane"]
        assert rows["ao:oryn-workspace-9"]["lane_reason"] == "Worker was stopped before completing."
        assert rows["ao:oryn-workspace-9"]["attention_level"] == "high"
        assert rows["ao:oryn-workspace-9"]["group_key"] == "project:OrynWorkspace"
        assert rows["ao:oryn-workspace-9"]["group_kind"] == "project"
        assert rows["ao:oryn-workspace-9"]["has_prompt_metadata"] is True
        assert rows["ao:oryn-workspace-9"]["agent"] == "codex"
        assert rows["ao:oryn-workspace-9"]["model"] == "gpt-5.5"
        assert rows["ao:oryn-workspace-9"]["reasoning_effort"] == "high"
        assert rows["ao:oryn-workspace-9"]["can_open"] is True
        assert rows["ao:oryn-workspace-9"]["can_retry"] is True
        assert data["lanes"]["running"] == 1
        assert data["lanes"]["failed"] == 1
        assert data["attention_count"] == 1
        groups = {group["key"]: group for group in data["groups"]}
        assert groups["project:OrynWorkspace"]["attention_count"] == 1

    @pytest.mark.asyncio
    async def test_subagent_board_flat_file_ao_rows_include_stable_arrays(self, adapter, tmp_path):
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        app = _create_runs_app(adapter)

        with patch("tools.ao_bridge.AOBridge", return_value=bridge):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/subagents/board")
                assert resp.status == 200
                data = await resp.json()

        row = next(item for item in data["data"] if item["id"] == "ao:oryn-workspace-9")
        assert row["files_read"] == []
        assert row["files_written"] == []
        assert row["output_tail"] == []
        assert row["lane_reason"] == "Worker is active and reporting progress."
        assert row["attention_level"] == "none"
        assert row["group_key"] == "project:OrynWorkspace"
        assert row["can_open"] is True
        assert row["can_stop"] is True
        assert row["can_follow_up"] is True

    @pytest.mark.asyncio
    async def test_subagent_board_marks_stale_running_ao_runtime(self, adapter, tmp_path):
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        bridge.runtime_health = lambda session: {
            "runtime_health": "stale",
            "runtime_warning": "AO reports this worker as running, but its tmux/process runtime is gone.",
        }
        app = _create_runs_app(adapter)

        with patch("tools.ao_bridge.AOBridge", return_value=bridge):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/subagents/board")
                assert resp.status == 200
                data = await resp.json()

        row = next(item for item in data["data"] if item["id"] == "ao:oryn-workspace-9")
        assert row["lane"] == "failed"
        assert row["status"] == "terminated"
        assert row["runtime_health"] == "stale"
        assert "runtime is gone" in row["runtime_warning"]
        assert row["can_stop"] is False
        assert row["can_follow_up"] is False

    @pytest.mark.asyncio
    async def test_ao_follow_up_endpoint_sends_message_and_records_event(self, adapter, tmp_path):
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        app = _create_runs_app(adapter)

        with patch("tools.ao_bridge.AOBridge", return_value=bridge):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/ao/sessions/oryn-workspace-9/follow-up",
                    json={"message": "Please inspect the failing test"},
                )
                assert resp.status == 200
                data = await resp.json()

        assert data["ok"] is True
        assert data["mode"] == "follow-up"
        assert data["source_session_id"] == "oryn-workspace-9"
        assert data["message"] == "Follow-up sent"
        assert data["action"] == "follow-up"
        assert data["action_event"]["event"] == "subagent.action"
        assert data["action_event"]["action"] == "follow-up"
        assert data["session"]["has_prompt_metadata"] is False
        assert bridge.sent_messages == [("oryn-workspace-9", "Please inspect the failing test")]
        events = adapter._subagent_event_store.list_events(ao_session_id="oryn-workspace-9")
        assert events[-1]["event"] == "subagent.action"
        assert events[-1]["message"] == "Follow-up sent"

    @pytest.mark.asyncio
    async def test_ao_diagnostics_endpoint_returns_runtime_and_transcript_tail(self, adapter, tmp_path):
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        adapter._subagent_event_store.append_event({
            "event": "subagent.progress",
            "subagent_id": "ao:oryn-workspace-9",
            "ao_session_id": "oryn-workspace-9",
            "runtime": "ao",
            "status": "running",
            "message": "Inspecting files",
        })
        bridge = _FakeAOBridge()
        app = _create_runs_app(adapter)

        with patch("tools.ao_bridge.AOBridge", return_value=bridge):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/ao/sessions/oryn-workspace-9/diagnostics?lines=240")
                assert resp.status == 200
                data = await resp.json()

        assert data["object"] == "hermes.ao_session_diagnostics"
        assert data["runtime_health"] == "ok"
        assert data["diagnostic_status"] == "running"
        assert data["transcript_available"] is True
        assert "latest worker output" in data["transcript_tail"]
        assert data["last_event"]["message"] == "Inspecting files"
        assert data["can_resume"] is True

    @pytest.mark.asyncio
    async def test_ao_diagnostics_ignores_action_from_previous_lifecycle(self, adapter, tmp_path):
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        adapter._subagent_event_store.append_event({
            "event": "subagent.start",
            "subagent_id": "ao:oryn-workspace-9",
            "ao_session_id": "oryn-workspace-9",
            "runtime": "ao",
            "status": "running",
            "created_at": 100,
        })
        adapter._subagent_event_store.append_event({
            "event": "subagent.action",
            "subagent_id": "ao:oryn-workspace-9",
            "ao_session_id": "oryn-workspace-9",
            "runtime": "ao",
            "status": "killed",
            "action": "stop",
            "action_status": "killed",
            "message": "AO worker stopped by user.",
            "created_at": 110,
        })
        adapter._subagent_event_store.append_event({
            "event": "subagent.start",
            "subagent_id": "ao:oryn-workspace-9",
            "ao_session_id": "oryn-workspace-9",
            "runtime": "ao",
            "status": "running",
            "created_at": 120,
        })
        bridge = _FakeAOBridge()
        app = _create_runs_app(adapter)

        with patch("tools.ao_bridge.AOBridge", return_value=bridge):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/ao/sessions/oryn-workspace-9/diagnostics")
                assert resp.status == 200
                data = await resp.json()

        assert data["diagnostic_status"] == "running"
        assert data["last_action"] is None
        assert data["last_event"]["event"] == "subagent.start"

    @pytest.mark.asyncio
    async def test_ao_resume_endpoint_sends_message_and_records_action(self, adapter, tmp_path):
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        bridge = _FakeAOBridge()
        app = _create_runs_app(adapter)

        with patch("tools.ao_bridge.AOBridge", return_value=bridge):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/ao/sessions/oryn-workspace-9/resume",
                    json={"message": "Continue and report blockers"},
                )
                assert resp.status == 200
                data = await resp.json()

        assert data["ok"] is True
        assert data["action"] == "resume"
        assert data["message"] == "Resume sent"
        assert bridge.sent_messages == [("oryn-workspace-9", "Continue and report blockers")]
        events = adapter._subagent_event_store.list_events(ao_session_id="oryn-workspace-9")
        assert events[-1]["event"] == "subagent.action"
        assert events[-1]["action"] == "resume"

    @pytest.mark.asyncio
    async def test_ao_stop_endpoint_kills_session_and_records_terminal_event(self, adapter, tmp_path):
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        adapter._subagent_event_store.append_event({
            "event": "subagent.progress",
            "session_id": "session-1",
            "run_id": "run-1",
            "subagent_id": "ao:oryn-workspace-9",
            "ao_session_id": "oryn-workspace-9",
            "runtime": "ao",
            "goal": "Stop smoke test",
            "status": "running",
        })
        bridge = _FakeAOBridge()
        app = _create_runs_app(adapter)

        with patch("tools.ao_bridge.AOBridge", return_value=bridge):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post("/v1/ao/sessions/oryn-workspace-9/stop")
                assert resp.status == 200
                data = await resp.json()

        assert data["ok"] is True
        assert data["mode"] == "stop"
        assert data["action"] == "stop"
        assert data["source_session_id"] == "oryn-workspace-9"
        assert data["message"] == "AO worker stopped by user."
        assert data["event"]["event"] == "subagent.action"
        assert data["event"]["action"] == "stop"
        assert data["event"]["status"] == "killed"
        assert data["event"]["session_id"] == "session-1"
        assert bridge.killed_sessions == ["oryn-workspace-9"]
        events = adapter._subagent_event_store.list_events(ao_session_id="oryn-workspace-9")
        assert events[-1]["event"] == "subagent.action"
        assert events[-1]["status"] == "killed"
        assert any(event["event"] == "subagent.complete" for event in events)

        with patch("tools.ao_bridge.AOBridge", return_value=bridge):
            async with TestClient(TestServer(app)) as cli:
                repeat_resp = await cli.post("/v1/ao/sessions/oryn-workspace-9/stop")
                assert repeat_resp.status == 200
                repeat_data = await repeat_resp.json()

        assert repeat_data["ok"] is True
        assert repeat_data["action"] == "stop"
        assert repeat_data["status"] == "already_stopped"
        assert repeat_data["message"] == "AO worker is already stopped."
        assert bridge.killed_sessions == ["oryn-workspace-9"]
        repeated_events = adapter._subagent_event_store.list_events(ao_session_id="oryn-workspace-9")
        assert len(repeated_events) == len(events)

    @pytest.mark.asyncio
    async def test_ao_retry_endpoint_spawns_from_stored_prompt(self, adapter, tmp_path):
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        adapter._subagent_event_store.upsert_ao_prompt(
            ao_session_id="oryn-workspace-1",
            project_id="OrynWorkspace",
            prompt="Original task prompt",
            goal="Original goal",
            issue_id="DEV-1",
            branch="feat/original",
            agent="codex",
            model="gpt-5.5",
            reasoning_effort="medium",
        )
        adapter._subagent_event_store.append_event({
            "event": "subagent.complete",
            "subagent_id": "ao:oryn-workspace-1",
            "ao_session_id": "oryn-workspace-1",
            "runtime": "ao",
            "summary": "Previous worker failed on tests",
            "status": "failed",
        })
        bridge = _FakeAOBridge()
        app = _create_runs_app(adapter)

        with patch("tools.ao_bridge.AOBridge", return_value=bridge):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/ao/sessions/oryn-workspace-1/retry",
                    json={"instruction": "Try a smaller patch"},
                )
                assert resp.status == 200
                data = await resp.json()

        assert data["ok"] is True
        assert data["mode"] == "retry"
        assert data["action"] == "retry"
        assert data["target_ao_session_id"] == "oryn-workspace-9"
        assert data["message"] == "AO retry session spawned"
        assert "## Hermes AO Delegation Contract" in bridge.spawn_kwargs["prompt"]
        assert "Original task prompt" in bridge.spawn_kwargs["prompt"]
        assert "Previous worker failed on tests" in bridge.spawn_kwargs["prompt"]
        assert "Try a smaller patch" in bridge.spawn_kwargs["prompt"]
        assert data["event"]["event"] == "subagent.action"
        assert data["event"]["action"] == "retry"
        events = adapter._subagent_event_store.list_events(ao_session_id="oryn-workspace-9")
        assert any(event["event"] == "subagent.start" for event in events)
        start_event = next(event for event in events if event["event"] == "subagent.start")
        assert start_event["agent"] == "codex"
        assert start_event["model"] == "gpt-5.5"
        assert start_event["reasoning_effort"] == "medium"
        assert adapter._subagent_event_store.get_ao_prompt("oryn-workspace-9")["reasoning_effort"] == "medium"

    @pytest.mark.asyncio
    async def test_ao_repair_retry_endpoint_includes_diagnostics_in_prompt(self, adapter, tmp_path):
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        adapter._subagent_event_store.upsert_ao_prompt(
            ao_session_id="oryn-workspace-1",
            project_id="OrynWorkspace",
            prompt="Original task prompt",
            goal="Original goal",
            issue_id=None,
            branch=None,
        )
        adapter._subagent_event_store.append_event({
            "event": "subagent.complete",
            "subagent_id": "ao:oryn-workspace-1",
            "ao_session_id": "oryn-workspace-1",
            "runtime": "ao",
            "summary": "Worker stopped before finishing",
            "status": "failed",
        })
        bridge = _FakeAOBridge()
        bridge.session.id = "oryn-workspace-1"
        app = _create_runs_app(adapter)

        with patch("tools.ao_bridge.AOBridge", return_value=bridge):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/ao/sessions/oryn-workspace-1/repair-retry",
                    json={"instruction": "Avoid the previous failure"},
                )
                assert resp.status == 200
                data = await resp.json()

        assert data["ok"] is True
        assert data["action"] == "repair-retry"
        assert "## Hermes AO Delegation Contract" in bridge.spawn_kwargs["prompt"]
        assert "Recovery diagnostics:" in bridge.spawn_kwargs["prompt"]
        assert "latest worker output" in bridge.spawn_kwargs["prompt"]

    @pytest.mark.asyncio
    async def test_subagent_board_includes_recent_action_and_summary_quality(self, adapter, tmp_path):
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        adapter._subagent_event_store.append_event({
            "event": "subagent.complete",
            "subagent_id": "ao:oryn-workspace-9",
            "ao_session_id": "oryn-workspace-9",
            "runtime": "ao",
            "goal": "Return BOARD_DONE after inspection",
            "status": "completed",
            "summary": "did not produce a clear BOARD_DONE conclusion",
        })
        adapter._subagent_event_store.append_event({
            "event": "subagent.action",
            "subagent_id": "ao:oryn-workspace-9",
            "ao_session_id": "oryn-workspace-9",
            "runtime": "ao",
            "status": "completed",
            "action": "follow-up",
            "action_status": "succeeded",
            "message": "Follow-up sent",
        })
        app = _create_runs_app(adapter)

        with patch("tools.ao_bridge.AOBridge") as mock_bridge_cls:
            mock_bridge = MagicMock()
            mock_bridge.list.return_value = []
            mock_bridge_cls.return_value = mock_bridge
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/subagents/board")
                assert resp.status == 200
                data = await resp.json()

        item = data["data"][0]
        assert item["recent_action"] == "follow-up"
        assert item["recent_action_status"] == "succeeded"
        assert item["recent_action_message"] == "Follow-up sent"
        assert item["summary_quality"] == "warning"
        assert "incomplete" in item["summary_warning"]

    @pytest.mark.asyncio
    async def test_subagent_board_ignores_action_from_previous_ao_lifecycle(self, adapter, tmp_path):
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        adapter._subagent_event_store.append_event({
            "event": "subagent.start",
            "subagent_id": "ao:oryn-workspace-9",
            "ao_session_id": "oryn-workspace-9",
            "runtime": "ao",
            "status": "running",
            "created_at": 100,
        })
        adapter._subagent_event_store.append_event({
            "event": "subagent.action",
            "subagent_id": "ao:oryn-workspace-9",
            "ao_session_id": "oryn-workspace-9",
            "runtime": "ao",
            "status": "killed",
            "action": "stop",
            "action_status": "killed",
            "message": "AO worker stopped by user.",
            "created_at": 110,
        })
        adapter._subagent_event_store.append_event({
            "event": "subagent.start",
            "subagent_id": "ao:oryn-workspace-9",
            "ao_session_id": "oryn-workspace-9",
            "runtime": "ao",
            "status": "running",
            "created_at": 120,
        })
        bridge = _FakeAOBridge()
        app = _create_runs_app(adapter)

        with patch("tools.ao_bridge.AOBridge", return_value=bridge):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/subagents/board")
                assert resp.status == 200
                data = await resp.json()

        item = next(row for row in data["data"] if row["id"] == "ao:oryn-workspace-9")
        assert item["lane"] == "running"
        assert item["status"] == "running"
        assert item["recent_action"] is None
        assert item["recent_action_status"] is None

    @pytest.mark.asyncio
    async def test_run_events_stream_emits_named_sse_for_subagent_events(self, adapter):
        app = _create_runs_app(adapter)
        run_id = "run_subagent_stream"
        adapter._run_streams[run_id] = asyncio.Queue()
        adapter._run_statuses[run_id] = {"run_id": run_id, "status": "running"}
        await adapter._run_streams[run_id].put({
            "event": "subagent.start",
            "run_id": run_id,
            "subagent_id": "child-1",
            "parent_id": None,
            "depth": 0,
            "goal": "Read files",
            "status": "running",
        })
        await adapter._run_streams[run_id].put(None)

        async with TestClient(TestServer(app)) as cli:
            events_resp = await cli.get(f"/v1/runs/{run_id}/events")
            assert events_resp.status == 200
            body = await events_resp.text()

        assert "event: subagent.start" in body
        assert '"subagent_id": "child-1"' in body
        assert '"run_id": "run_subagent_stream"' in body

    @pytest.mark.asyncio
    async def test_events_stream_returns_completed(self, adapter):
        """Events stream should receive run.completed when agent finishes."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "Hello!"}
                mock_agent.session_prompt_tokens = 10
                mock_agent.session_completion_tokens = 5
                mock_agent.session_total_tokens = 15
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                # Subscribe to events
                events_resp = await cli.get(f"/v1/runs/{run_id}/events")
                assert events_resp.status == 200
                body = await events_resp.text()

                # Should contain run.completed
                assert "run.completed" in body
                assert "Hello!" in body



    @pytest.mark.asyncio
    async def test_approval_response_without_pending_returns_409(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                data = await resp.json()
                run_id = data["run_id"]

                approval_resp = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once"},
                )
                assert approval_resp.status == 409
                approval_data = await approval_resp.json()
                assert approval_data["error"]["code"] in {
                    "approval_not_active",
                    "approval_not_pending",
                }

    @pytest.mark.asyncio
    async def test_approval_string_false_does_not_resolve_all(self, adapter):
        """Quoted false must not fan out approval resolution across the queue."""
        app = _create_runs_app(adapter)
        run_id = "run_bool_parse"
        adapter._run_statuses[run_id] = {"run_id": run_id, "status": "running"}
        adapter._run_approval_sessions[run_id] = "session-123"

        async with TestClient(TestServer(app)) as cli:
            with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
                approval_resp = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once", "all": "false"},
                )

        assert approval_resp.status == 200
        mock_resolve.assert_called_once_with(
            "session-123",
            "once",
            resolve_all=False,
        )

    @pytest.mark.asyncio
    async def test_events_not_found_returns_404(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_nonexistent/events")
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_events_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_any/events")
        assert resp.status == 401


# ---------------------------------------------------------------------------
# POST /v1/runs/{run_id}/stop — interrupt a running agent
# ---------------------------------------------------------------------------


class TestStopRun:
    @pytest.mark.asyncio
    async def test_stop_running_agent(self, adapter):
        """Stop should interrupt the agent and cancel the task."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                # Wait for agent to start running in the thread
                agent_ready.wait(timeout=3.0)
                await asyncio.sleep(0.1)

                # Verify agent ref is stored
                assert run_id in adapter._active_run_agents

                # Stop the run
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                stop_data = await stop_resp.json()
                assert stop_data["run_id"] == run_id
                assert stop_data["status"] == "stopping"

                # Agent interrupt should have been called
                mock_agent.interrupt.assert_called_once_with("Stop requested via API")

                status_resp = await cli.get(f"/v1/runs/{run_id}")
                assert status_resp.status == 200
                status_data = await status_resp.json()
                assert status_data["status"] in {"stopping", "cancelled"}

                # Refs should be cleaned up
                await asyncio.sleep(0.5)
                assert run_id not in adapter._active_run_agents
                assert run_id not in adapter._active_run_tasks

    @pytest.mark.asyncio
    async def test_stop_nonexistent_run_returns_404(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_nonexistent/stop")
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_stop_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_any/stop")
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_stop_already_completed_run_returns_404(self, adapter):
        """Stopping a run that already finished should return 404 (refs cleaned up)."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                # Start and wait for completion
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                await asyncio.sleep(0.3)

                # Run should be done, refs cleaned up
                assert run_id not in adapter._active_run_agents

                # Stop should return 404
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 404

    @pytest.mark.asyncio
    async def test_stop_interrupt_exception_does_not_crash(self, adapter):
        """If agent.interrupt() raises, stop should still succeed."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, interrupted = _make_slow_agent()

                # Override the interrupt side_effect to raise. Still trip
                # ``interrupted`` so the slow_run thread unblocks at teardown
                # — without this the agent thread blocks the full 10s
                # timeout and the test teardown waits the same amount.
                def _raising_interrupt(message=None):
                    interrupted.set()
                    raise RuntimeError("interrupt failed")

                mock_agent.interrupt = MagicMock(side_effect=_raising_interrupt)
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                agent_ready.wait(timeout=3.0)
                await asyncio.sleep(0.1)

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                stop_data = await stop_resp.json()
                assert stop_data["status"] == "stopping"

    @pytest.mark.asyncio
    async def test_stop_sends_sentinel_to_events_stream(self, adapter):
        """After stop, the events stream should close."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                agent_ready.wait(timeout=3.0)
                await asyncio.sleep(0.1)

                # Subscribe to events in background
                events_task = asyncio.ensure_future(
                    cli.get(f"/v1/runs/{run_id}/events")
                )

                await asyncio.sleep(0.1)

                # Stop the run
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200

                # Events stream should close
                events_resp = await asyncio.wait_for(events_task, timeout=5.0)
                assert events_resp.status == 200
                body = await events_resp.text()
                # Stream should have received run.failed and closed
                assert "run.failed" in body or "stream closed" in body


class TestAOSessionControls:
    @pytest.mark.asyncio
    async def test_stop_ao_session_calls_bridge(self, adapter, tmp_path):
        adapter._subagent_event_store = SubagentEventStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch("tools.ao_bridge.AOBridge") as mock_bridge_cls:
                mock_bridge = MagicMock()
                mock_bridge_cls.return_value = mock_bridge

                resp = await cli.post("/v1/ao/sessions/oryn-workspace-1/stop")

                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "killed"
                mock_bridge.kill.assert_called_once_with("oryn-workspace-1", session=mock_bridge.status.return_value)

    @pytest.mark.asyncio
    async def test_open_ao_session_returns_attach_info(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch("tools.ao_bridge.AOBridge") as mock_bridge_cls:
                mock_bridge = MagicMock()
                mock_bridge.open_session.return_value = {
                    "ok": True,
                    "opened": True,
                    "session": {
                        "runtime": "ao",
                        "ao_session_id": "oryn-workspace-1",
                        "workspace_path": "/tmp/worktree",
                        "open_command": "tmux attach -t abc-oryn-workspace-1",
                    },
                }
                mock_bridge_cls.return_value = mock_bridge

                resp = await cli.post("/v1/ao/sessions/oryn-workspace-1/open")

                assert resp.status == 200
                data = await resp.json()
                assert data["session"]["runtime"] == "ao"
                assert data["session"]["ao_session_id"] == "oryn-workspace-1"
                mock_bridge.open_session.assert_called_once_with("oryn-workspace-1")


class TestDevProductionSignalAPI:
    @pytest.mark.asyncio
    async def test_signal_report_health_and_proposal_action_routes(self, adapter, tmp_path):
        db_path = tmp_path / "state.db"
        adapter._subagent_event_store = SubagentEventStore(db_path)
        adapter._dev_signal_store = DevProductionSignalStore(db_path)
        for _ in range(3):
            adapter._subagent_event_store.append_event({
                "runtime": "fixture",
                "subagent_id": "worker-api",
                "event": "subagent.completed",
                "status": "failed",
                "summary": "api production signal",
                "launch_plan_id": "plan-api",
                "launch_task_id": "task-api",
                "worker_confidence": 0.3,
                "output_contract_score": 0.4,
                "created_at": _time.time(),
            })

        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            create_resp = await cli.post("/v1/dev/signal-reports", json={"window_days": 7, "digest": True})
            assert create_resp.status == 200
            report = await create_resp.json()
            assert report["status"] == "completed_with_clusters"
            assert report["counts"]["proposal_count"] >= 1

            list_resp = await cli.get("/v1/dev/signal-reports")
            assert list_resp.status == 200
            assert (await list_resp.json())["total"] == 1

            proposals_resp = await cli.get("/v1/dev/backlog-proposals")
            assert proposals_resp.status == 200
            proposal = (await proposals_resp.json())["data"][0]
            assert proposal["payload"]["source"] == "production_signal"

            approve_resp = await cli.post(f"/v1/dev/backlog-proposals/{proposal['proposal_id']}/approve")
            assert approve_resp.status == 200
            assert (await approve_resp.json())["status"] == "approved"

            health_resp = await cli.get("/v1/dev/signal-health")
            assert health_resp.status == 200
            health = await health_resp.json()
            assert health["last_analysis_status"] == "completed_with_clusters"
            assert health["proposals_by_status"]["approved"] == 1


class TestDevReliabilityAPI:
    @pytest.mark.asyncio
    async def test_reliability_routes_are_advisory_and_empty_safe(self, adapter, tmp_path):
        db_path = tmp_path / "state.db"
        adapter._dev_reliability_store = DevReliabilityStore(db_path)
        adapter._dev_reliability_store.upsert_outcome({
            "plan_id": "plan-api",
            "task_id": "task-api",
            "category": "workspace.implement/high",
            "profile_id": "workspace.implement",
            "risk_level": "high",
            "terminal_status": "completed",
            "merged": True,
            "verification_verdict": "verified",
            "ci_state": "success",
            "code_review_verdict": "approved",
            "output_contract_score": 0.95,
            "completed_at": _time.time(),
        })

        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            score_resp = await cli.get("/v1/dev/reliability")
            assert score_resp.status == 200
            score = await score_resp.json()
            assert score["advisory_only"] is True
            assert score["categories"][0]["category"] == "workspace.implement/high"
            assert score["categories"][0]["tier"] == "unproven"

            weak_resp = await cli.get("/v1/dev/reliability/weakest")
            assert weak_resp.status == 200
            weak = await weak_resp.json()
            assert weak["data"][0]["category"] == "workspace.implement/high"

            detail_resp = await cli.get("/v1/dev/reliability/workspace.implement/high")
            assert detail_resp.status == 200
            detail = await detail_resp.json()
            assert detail["category"] == "workspace.implement/high"
            assert detail["outcomes"][0]["task_id"] == "task-api"

    @pytest.mark.asyncio
    async def test_lab_loop_health_route_is_empty_safe(self, adapter, tmp_path):
        db_path = tmp_path / "state.db"
        adapter._dev_lab_loop_store = DevLabLoopStore(db_path)

        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/dev/lab-loop/health")
            assert resp.status == 200
            data = await resp.json()
            assert data["object"] == "hermes.dev_lab_loop_health"
            assert data["state"]["status"] == "idle"
            assert data["real_outcome_count"] == 0
