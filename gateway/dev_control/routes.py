"""Oryn Dev control API route registrations and handlers.

This module is the Oryn-owned route boundary for the Dev control plane.  It
keeps /v1/dev/* handlers out of the upstream-maintained API server module while
preserving the existing adapter method shape and auth/middleware behavior.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from typing import Any, Dict, Optional

from aiohttp import web

from gateway.dev_execution import (
    DevExecutionStore,
    apply_execution_plan_review,
    apply_supervisor_approval,
    approve_supervisor_approval,
    deny_supervisor_approval,
    derive_execution_plan_status,
    get_runbook,
    get_supervisor_approval,
    launch_execution_plan,
    list_launch_profiles,
    list_runbooks,
    list_supervisor_approvals,
    list_supervisor_loop_status,
    list_worker_runtimes,
    next_execution_step,
    review_execution_plan,
    select_execution_runtime,
    set_execution_plan_test_state,
    set_project_runbook,
    set_supervisor_loop,
    supervise_execution_plans,
    supervisor_loop_tick,
    synthesize_execution_plan,
)
from gateway.dev_control.acceptance_verification import (
    DevVerificationStore,
    launch_verification_run,
    list_verification_runs,
    refresh_verification_run,
)
from gateway.dev_control.ci_status import fetch_ci_status
from gateway.dev_control.clarifications import (
    DevClarificationStore,
    answer_clarification,
    cancel_clarification,
    complete_clarification,
    get_clarification,
    list_clarifications,
    start_clarification,
)
from gateway.dev_control.harness_benchmarks import (
    get_harness_benchmark_run,
    list_harness_benchmark_runs,
    run_harness_benchmark,
)
from gateway.dev_control.harness_observability import (
    generate_harness_report,
    list_harness_components,
)
from gateway.dev_control.harness_recommendations import (
    generate_harness_recommendations,
    get_harness_recommendation_run,
    list_harness_recommendation_runs,
)
from gateway.dev_control.github_pr_automation import (
    DevGitHubPRAutomationStore,
    automation_summary,
    process_github_webhook,
    run_manual_pr_automation_action,
    verify_github_signature,
)
from gateway.dev_control.incidents import (
    DevIncidentStore,
    acknowledge_incident,
    detect_incidents,
    resolve_incident,
)
from gateway.dev_control.lab_loop import DevLabLoopStore, loop_health
from gateway.dev_control.plan_artifacts import (
    DevPlanArtifactStore,
    approve_execution_plan_draft,
    approve_plan_artifact,
    cancel_execution_plan_draft,
    cancel_plan_artifact,
    create_execution_plan_from_artifact,
    create_plan_artifact,
    get_execution_plan_draft_review,
    get_plan_artifact,
    list_plan_artifact_builds,
    list_plan_artifacts,
    revise_execution_plan_draft,
    revise_plan_artifact,
)
from gateway.dev_control.project_scope import project_id_from_payload, resolve_project_id
from gateway.dev_control.product_events import DevProductEventStore
from gateway.dev_control.production_signals import (
    DevProductionSignalStore,
    generate_signal_report,
    list_backlog_proposals,
    list_signal_reports,
    measure_proposal_outcome,
    run_signal_digest,
    signal_health,
    transition_backlog_proposal,
)
from gateway.dev_control.read_models import (
    build_agent_board_response,
    build_agent_board_rows,
    build_dev_plans_response,
)
from gateway.dev_control.reliability import (
    DevReliabilityStore,
    recompute_reliability_outcomes,
    scorecard as reliability_scorecard,
    weakest_categories,
)
from gateway.dev_control.scm_lifecycle import (
    DevSCMLifecycleStore,
    build_code_review_prompt,
    compose_merge_readiness,
    execute_merge,
    fetch_pr_state,
    parse_code_review_result,
    request_merge_approval,
)
from tools.openhands_bridge import (
    openhands_server_status,
    start_openhands_server,
    stop_openhands_server,
)

logger = logging.getLogger(__name__)

_TRUE_REQUEST_BOOL_STRINGS = frozenset({"1", "true", "yes", "on"})
_FALSE_REQUEST_BOOL_STRINGS = frozenset({"0", "false", "no", "off"})


def _coerce_request_bool(value: Any, default: bool = False) -> bool:
    """Normalize boolean-like API payload values."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_REQUEST_BOOL_STRINGS:
            return True
        if normalized in _FALSE_REQUEST_BOOL_STRINGS:
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _fetch_ci_status(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Fetch CI status while preserving legacy api_server monkeypatch points."""
    api_server_module = sys.modules.get("gateway.platforms.api_server")
    patched = getattr(api_server_module, "fetch_ci_status", None)
    if patched is not None and patched is not fetch_ci_status:
        return patched(*args, **kwargs)
    return fetch_ci_status(*args, **kwargs)


def _openai_error(
    message: str,
    err_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> Dict[str, Any]:
    """OpenAI-style error envelope."""
    return {
        "error": {
            "message": message,
            "type": err_type,
            "param": param,
            "code": code,
        }
    }


def dev_control_capabilities() -> dict[str, dict[str, str]]:
    """Return machine-readable capability entries for the Dev control plane."""
    return {
        "dev_worker_runtimes": {"method": "GET", "path": "/v1/dev/runtimes"},
        "dev_runtime_selection": {"method": "POST", "path": "/v1/dev/runtime-selection"},
        "dev_harness_components": {"method": "GET", "path": "/v1/dev/harness/components"},
        "dev_harness_report": {"method": "POST", "path": "/v1/dev/harness/report"},
        "dev_harness_recommendations": {"method": "POST", "path": "/v1/dev/harness/recommendations"},
        "dev_harness_recommendation_runs": {"method": "GET", "path": "/v1/dev/harness/recommendations"},
        "dev_harness_benchmark": {"method": "POST", "path": "/v1/dev/harness/benchmarks"},
        "dev_harness_benchmark_runs": {"method": "GET", "path": "/v1/dev/harness/benchmarks"},
        "dev_clarifications": {"method": "GET", "path": "/v1/dev/clarifications"},
        "dev_start_clarification": {"method": "POST", "path": "/v1/dev/clarifications"},
        "dev_answer_clarification": {"method": "POST", "path": "/v1/dev/clarifications/{clarification_id}/answer"},
        "dev_complete_clarification": {"method": "POST", "path": "/v1/dev/clarifications/{clarification_id}/complete"},
        "dev_cancel_clarification": {"method": "POST", "path": "/v1/dev/clarifications/{clarification_id}/cancel"},
        "dev_plan_artifacts": {"method": "GET", "path": "/v1/dev/plan-artifacts"},
        "dev_create_plan_artifact": {"method": "POST", "path": "/v1/dev/plan-artifacts"},
        "dev_revise_plan_artifact": {"method": "POST", "path": "/v1/dev/plan-artifacts/{plan_artifact_id}/revise"},
        "dev_approve_plan_artifact": {"method": "POST", "path": "/v1/dev/plan-artifacts/{plan_artifact_id}/approve"},
        "dev_cancel_plan_artifact": {"method": "POST", "path": "/v1/dev/plan-artifacts/{plan_artifact_id}/cancel"},
        "dev_create_execution_plan_from_artifact": {"method": "POST", "path": "/v1/dev/plan-artifacts/{plan_artifact_id}/create-execution-plan"},
        "dev_plan_artifact_builds": {"method": "GET", "path": "/v1/dev/plan-artifacts/{plan_artifact_id}/builds"},
        "dev_execution_plan_draft_review": {"method": "GET", "path": "/v1/dev/execution-plans/{plan_id}/draft-review"},
        "dev_revise_execution_plan_draft": {"method": "POST", "path": "/v1/dev/execution-plans/{plan_id}/revise-draft"},
        "dev_approve_execution_plan_draft": {"method": "POST", "path": "/v1/dev/execution-plans/{plan_id}/approve-draft"},
        "dev_cancel_execution_plan_draft": {"method": "POST", "path": "/v1/dev/execution-plans/{plan_id}/cancel-draft"},
        "dev_verification_runs": {"method": "GET", "path": "/v1/dev/verification-runs"},
        "dev_start_verification_run": {"method": "POST", "path": "/v1/dev/verification-runs"},
        "dev_verification_run_detail": {"method": "GET", "path": "/v1/dev/verification-runs/{verification_run_id}"},
        "dev_signal_reports": {"method": "GET", "path": "/v1/dev/signal-reports"},
        "dev_create_signal_report": {"method": "POST", "path": "/v1/dev/signal-reports"},
        "dev_signal_report_detail": {"method": "GET", "path": "/v1/dev/signal-reports/{report_id}"},
        "dev_ci_status": {"method": "GET", "path": "/v1/dev/ci-status"},
        "dev_product_events": {"method": "GET", "path": "/v1/dev/product-events"},
        "dev_ingest_product_events": {"method": "POST", "path": "/v1/dev/product-events"},
        "dev_detect_incidents": {"method": "POST", "path": "/v1/dev/incidents/detect"},
        "dev_incidents": {"method": "GET", "path": "/v1/dev/incidents"},
        "dev_incident_detail": {"method": "GET", "path": "/v1/dev/incidents/{incident_id}"},
        "dev_incident_action": {"method": "POST", "path": "/v1/dev/incidents/{incident_id}/{action}"},
        "dev_pr_state": {"method": "GET", "path": "/v1/dev/pr-state"},
        "dev_code_review_runs": {"method": "GET", "path": "/v1/dev/code-review-runs"},
        "dev_start_code_review_run": {"method": "POST", "path": "/v1/dev/code-review-runs"},
        "dev_merge_readiness": {"method": "GET", "path": "/v1/dev/merge-readiness"},
        "dev_merge_approval_request": {"method": "POST", "path": "/v1/dev/merge-approvals"},
        "dev_merge_approval_approve": {"method": "POST", "path": "/v1/dev/merge-approvals/{approval_id}/approve"},
        "dev_merge_execute": {"method": "POST", "path": "/v1/dev/merge"},
        "dev_github_webhooks": {"method": "POST", "path": "/v1/dev/github-webhooks"},
        "dev_pr_automation": {"method": "GET", "path": "/v1/dev/pr-automation"},
        "dev_pr_automation_actions": {"method": "POST", "path": "/v1/dev/pr-automation/actions"},
        "dev_backlog_proposals": {"method": "GET", "path": "/v1/dev/backlog-proposals"},
        "dev_backlog_proposal_action": {"method": "POST", "path": "/v1/dev/backlog-proposals/{proposal_id}/{action}"},
        "dev_signal_health": {"method": "GET", "path": "/v1/dev/signal-health"},
        "dev_reliability": {"method": "GET", "path": "/v1/dev/reliability"},
        "dev_reliability_recompute": {"method": "POST", "path": "/v1/dev/reliability/recompute"},
        "dev_reliability_weakest": {"method": "GET", "path": "/v1/dev/reliability/weakest"},
        "dev_reliability_category": {"method": "GET", "path": "/v1/dev/reliability/{category}"},
        "dev_lab_loop_health": {"method": "GET", "path": "/v1/dev/lab-loop/health"},
    }


def register_dev_control_routes(app: web.Application, adapter: Any) -> None:
    """Register Oryn Dev control routes on the shared API server app."""
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
    app.router.add_get("/v1/dev/ci-status", adapter._handle_dev_ci_status)
    app.router.add_get("/v1/dev/pr-state", adapter._handle_dev_pr_state)
    app.router.add_get("/v1/dev/code-review-runs", adapter._handle_dev_code_review_runs)
    app.router.add_post("/v1/dev/code-review-runs", adapter._handle_dev_code_review_runs)
    app.router.add_get("/v1/dev/merge-readiness", adapter._handle_dev_merge_readiness)
    app.router.add_post("/v1/dev/merge-approvals", adapter._handle_dev_merge_approvals)
    app.router.add_post("/v1/dev/merge-approvals/{approval_id}/approve", adapter._handle_dev_merge_approval_approve)
    app.router.add_post("/v1/dev/merge", adapter._handle_dev_merge_execute)
    app.router.add_post("/v1/dev/github-webhooks", adapter._handle_dev_github_webhooks)
    app.router.add_get("/v1/dev/pr-automation", adapter._handle_dev_pr_automation)
    app.router.add_post("/v1/dev/pr-automation/actions", adapter._handle_dev_pr_automation_actions)
    app.router.add_get("/v1/dev/product-events", adapter._handle_dev_product_events)
    app.router.add_post("/v1/dev/product-events", adapter._handle_dev_product_events)
    app.router.add_post("/v1/dev/incidents/detect", adapter._handle_dev_incident_detect)
    app.router.add_get("/v1/dev/incidents", adapter._handle_dev_incidents)
    app.router.add_get("/v1/dev/incidents/{incident_id}", adapter._handle_dev_incident_detail)
    app.router.add_post("/v1/dev/incidents/{incident_id}/{action}", adapter._handle_dev_incident_action)
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
    logger.info("Registered /v1/dev/* gateway routes")


class DevControlRouteMixin:
    """Adapter mixin containing Oryn Dev control route handlers."""

    async def _handle_oryn_project_dashboard(self, request: "web.Request") -> "web.Response":
        """GET /v1/oryn/project-dashboard — bundled project dashboard read model."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        project_id = request.rel_url.query.get("project_id") or None
        try:
            default_plan_limit = int(os.getenv("ORYN_PROJECT_DASHBOARD_PLAN_LIMIT", "12"))
        except ValueError:
            default_plan_limit = 12
        try:
            plan_limit = int(request.rel_url.query.get("plan_limit", str(default_plan_limit)))
        except ValueError:
            plan_limit = default_plan_limit
        plan_limit = max(1, min(plan_limit, 50))
        derive_plans = str(request.rel_url.query.get("derive_plans") or "").lower() in {"1", "true", "yes"}
        clarification_store = self._ensure_dev_clarification_store()
        artifact_store = self._ensure_dev_plan_artifact_store()
        execution_store = self._ensure_dev_execution_store()
        verification_store = self._ensure_dev_verification_store()
        signal_store = self._ensure_dev_signal_store()
        incident_store = self._ensure_dev_incident_store()
        scm_store = self._ensure_dev_scm_store()
        pr_automation_store = self._ensure_dev_github_pr_automation_store()
        reliability_store = self._ensure_dev_reliability_store()
        lab_loop_store = self._ensure_dev_lab_loop_store()
        event_store = self._ensure_subagent_event_store()
        if clarification_store is None or artifact_store is None or execution_store is None or event_store is None:
            return web.json_response(_openai_error("Oryn project dashboard stores unavailable"), status=503)

        fingerprint = self._project_dashboard_fingerprint(
            project_id,
            plan_limit=plan_limit,
            derive_plans=derive_plans,
        )

        def _compute_payload() -> Dict[str, Any]:
            clarifications = list_clarifications(
                store=clarification_store,
                project_id=project_id,
                session_id=None,
                status=None,
                limit=5,
            )
            artifacts = list_plan_artifacts(
                store=artifact_store,
                clarification_id=None,
                project_id=project_id,
                status=None,
                limit=10,
            )
            raw_plans = execution_store.list_plans(limit=plan_limit, project_id=project_id)
            plans = list(raw_plans)
            if derive_plans:
                try:
                    from tools.ao_bridge import AOBridge

                    bridge = AOBridge()
                except Exception:
                    bridge = None
                if bridge is not None:
                    derived_plans = []
                    for plan in raw_plans:
                        try:
                            derived_plans.append(derive_execution_plan_status(
                                store=execution_store,
                                plan_id=plan["plan_id"],
                                bridge=bridge,
                                event_store=event_store,
                                verification_store=verification_store,
                            )["plan"])
                        except Exception:
                            derived_plans.append(plan)
                    plans = derived_plans

            board_params = {
                "project_id": project_id,
                "limit": "100",
            }
            board_rows = build_agent_board_rows(
                store=event_store,
                params=board_params,
                limit=100,
                ao_snapshot_cache=self._ao_snapshot_cache,
            )
            latest_artifact = next(
                (
                    artifact for artifact in artifacts.get("data", [])
                    if artifact.get("status") not in {"cancelled", "superseded"}
                ),
                None,
            )
            latest_build = None
            latest_draft_review = None
            if latest_artifact:
                builds = list_plan_artifact_builds(
                    store=artifact_store,
                    plan_artifact_id=latest_artifact["plan_artifact_id"],
                    limit=1,
                ).get("data", [])
                latest_build = builds[0] if builds else None
                if latest_build and latest_build.get("plan_id"):
                    try:
                        latest_draft_review = get_execution_plan_draft_review(
                            execution_store=execution_store,
                            plan_id=latest_build["plan_id"],
                        )
                    except Exception:
                        latest_draft_review = None

            return {
                "object": "hermes.oryn.project_dashboard",
                "project_id": project_id,
                "clarifications": clarifications.get("data", []),
                "plan_artifacts": artifacts.get("data", []),
                "subagent_board": build_agent_board_response(board_rows),
                "dev_plans": plans,
                "latest_plan_artifact_build": latest_build,
                "latest_draft_review": latest_draft_review,
                "dev_signal_health": signal_health(signal_store=signal_store, event_store=event_store) if signal_store else None,
                "dev_backlog_proposals": (list_backlog_proposals(signal_store=signal_store, limit=10).get("data", []) if signal_store else []),
                "dev_incidents": (incident_store.list_incidents(limit=5) if incident_store else []),
                "dev_merge_readiness": (scm_store.latest_readiness(limit=5) if scm_store else []),
                "dev_pr_automation": (
                    automation_summary(store=pr_automation_store, scm_store=scm_store, limit=5).get("data", [])
                    if pr_automation_store
                    else []
                ),
                "dev_reliability": (reliability_scorecard(reliability_store.list_outcomes(limit=500)) if reliability_store else None),
                "dev_lab_loop_health": (loop_health(db_path=lab_loop_store.db_path) if lab_loop_store else None),
            }

        cached = await self._read_model_cache.get_or_compute(
            key=f"project-dashboard:{project_id or 'default'}:{plan_limit}:{derive_plans}",
            fingerprint=fingerprint,
            compute=_compute_payload,
        )
        return self._cached_read_model_response(
            request,
            cached.payload,
            cached.fingerprint,
            cached=cached,
            model_name="project.dashboard",
        )

    def _ensure_dev_execution_store(self) -> Optional[DevExecutionStore]:
        """Lazily initialise persistent Dev execution plan storage."""
        if self._dev_execution_store is None:
            try:
                self._dev_execution_store = DevExecutionStore()
            except Exception as exc:
                logger.warning("Dev execution store unavailable: %s", exc)
        return self._dev_execution_store

    async def _handle_dev_launch_profiles(self, request: "web.Request") -> "web.Response":
        """GET /v1/dev/launch-profiles — list Dev worker launch profiles."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        profiles = list_launch_profiles()
        return web.json_response({"object": "list", "data": profiles, "total": len(profiles)})

    async def _handle_dev_worker_runtimes(self, request: "web.Request") -> "web.Response":
        """GET /v1/dev/runtimes — list Dev worker runtimes."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        runtimes = list_worker_runtimes()
        return web.json_response({"object": "list", "data": runtimes, "total": len(runtimes)})

    async def _handle_dev_runtime_selection(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/runtime-selection — dry-run Dev worker runtime selection."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            body = await request.json()
        except Exception:
            body = {}
        result = select_execution_runtime(
            goal=body.get("goal"),
            prompt=body.get("prompt"),
            profile_id=body.get("profile_id") or body.get("launch_profile_id"),
            runtime=body.get("runtime"),
            project_id=body.get("project_id"),
            permissions=body.get("permissions"),
            db_path=self._ensure_dev_execution_store().db_path,
        )
        return web.json_response({
            "ok": True,
            "object": "hermes.dev_runtime_selection",
            **result,
        })

    def _ensure_dev_clarification_store(self) -> Optional[DevClarificationStore]:
        """Create the Dev clarification store on the same state.db as execution plans."""
        if self._dev_clarification_store is not None:
            return self._dev_clarification_store
        execution_store = self._ensure_dev_execution_store()
        if execution_store is None:
            return None
        try:
            self._dev_clarification_store = DevClarificationStore(db_path=execution_store.db_path)
        except Exception as exc:
            logger.warning("Dev clarification store unavailable: %s", exc)
            return None
        return self._dev_clarification_store

    def _ensure_dev_plan_artifact_store(self) -> Optional[DevPlanArtifactStore]:
        """Create the Dev plan artifact store on the same state.db as execution plans."""
        if self._dev_plan_artifact_store is not None:
            return self._dev_plan_artifact_store
        execution_store = self._ensure_dev_execution_store()
        if execution_store is None:
            return None
        try:
            self._dev_plan_artifact_store = DevPlanArtifactStore(db_path=execution_store.db_path)
        except Exception as exc:
            logger.warning("Dev plan artifact store unavailable: %s", exc)
            return None
        return self._dev_plan_artifact_store

    def _ensure_dev_verification_store(self) -> Optional[DevVerificationStore]:
        """Create the Dev verification store on the same state.db as execution plans."""
        if self._dev_verification_store is not None:
            return self._dev_verification_store
        execution_store = self._ensure_dev_execution_store()
        if execution_store is None:
            return None
        try:
            self._dev_verification_store = DevVerificationStore(db_path=execution_store.db_path)
        except Exception as exc:
            logger.warning("Dev verification store unavailable: %s", exc)
            return None
        return self._dev_verification_store

    def _ensure_dev_signal_store(self) -> Optional[DevProductionSignalStore]:
        """Create the Dev production signal store on the same state.db as execution plans."""
        if self._dev_signal_store is not None:
            return self._dev_signal_store
        execution_store = self._ensure_dev_execution_store()
        if execution_store is None:
            return None
        try:
            self._dev_signal_store = DevProductionSignalStore(db_path=execution_store.db_path)
        except Exception as exc:
            logger.warning("Dev production signal store unavailable: %s", exc)
            return None
        return self._dev_signal_store

    def _ensure_dev_product_event_store(self) -> Optional[DevProductEventStore]:
        """Create the product-event store on the same state.db as execution plans."""
        if self._dev_product_event_store is not None:
            return self._dev_product_event_store
        execution_store = self._ensure_dev_execution_store()
        if execution_store is None:
            return None
        try:
            self._dev_product_event_store = DevProductEventStore(db_path=execution_store.db_path)
        except Exception as exc:
            logger.warning("Dev product event store unavailable: %s", exc)
            return None
        return self._dev_product_event_store

    def _ensure_dev_incident_store(self) -> Optional[DevIncidentStore]:
        """Create the Dev incident store on the same state.db as execution plans."""
        if self._dev_incident_store is not None:
            return self._dev_incident_store
        execution_store = self._ensure_dev_execution_store()
        if execution_store is None:
            return None
        try:
            self._dev_incident_store = DevIncidentStore(db_path=execution_store.db_path)
        except Exception as exc:
            logger.warning("Dev incident store unavailable: %s", exc)
            return None
        return self._dev_incident_store

    def _ensure_dev_scm_store(self) -> Optional[DevSCMLifecycleStore]:
        """Create the SCM lifecycle store on the same state.db as execution plans."""
        if self._dev_scm_store is not None:
            return self._dev_scm_store
        execution_store = self._ensure_dev_execution_store()
        if execution_store is None:
            return None
        try:
            self._dev_scm_store = DevSCMLifecycleStore(db_path=execution_store.db_path)
        except Exception as exc:
            logger.warning("Dev SCM lifecycle store unavailable: %s", exc)
            return None
        return self._dev_scm_store

    def _ensure_dev_github_pr_automation_store(self) -> Optional[DevGitHubPRAutomationStore]:
        """Create the GitHub PR automation store on the same state.db as execution plans."""
        if getattr(self, "_dev_github_pr_automation_store", None) is not None:
            return self._dev_github_pr_automation_store
        execution_store = self._ensure_dev_execution_store()
        if execution_store is None:
            return None
        try:
            self._dev_github_pr_automation_store = DevGitHubPRAutomationStore(db_path=execution_store.db_path)
        except Exception as exc:
            logger.warning("Dev GitHub PR automation store unavailable: %s", exc)
            return None
        return self._dev_github_pr_automation_store

    def _ensure_dev_reliability_store(self) -> Optional[DevReliabilityStore]:
        """Create the Dev reliability store on the same state.db as execution plans."""
        if getattr(self, "_dev_reliability_store", None) is not None:
            return self._dev_reliability_store
        execution_store = self._ensure_dev_execution_store()
        if execution_store is None:
            return None
        try:
            self._dev_reliability_store = DevReliabilityStore(db_path=execution_store.db_path)
        except Exception as exc:
            logger.warning("Dev reliability store unavailable: %s", exc)
            return None
        return self._dev_reliability_store

    def _ensure_dev_lab_loop_store(self) -> Optional[DevLabLoopStore]:
        """Create the Dev Lab loop store on the same state.db as execution plans."""
        if getattr(self, "_dev_lab_loop_store", None) is not None:
            return self._dev_lab_loop_store
        execution_store = self._ensure_dev_execution_store()
        if execution_store is None:
            return None
        try:
            self._dev_lab_loop_store = DevLabLoopStore(db_path=execution_store.db_path)
        except Exception as exc:
            logger.warning("Dev Lab loop store unavailable: %s", exc)
            return None
        return self._dev_lab_loop_store

    async def _handle_dev_clarifications(self, request: "web.Request") -> "web.Response":
        """GET/POST /v1/dev/clarifications — list or start durable clarification sessions."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_clarification_store()
        if store is None:
            return web.json_response({"error": {"message": "Dev clarification store unavailable"}}, status=503)
        if request.method == "GET":
            result = list_clarifications(
                store=store,
                project_id=request.rel_url.query.get("project_id") or None,
                session_id=request.rel_url.query.get("session_id") or None,
                status=request.rel_url.query.get("status") or None,
                limit=request.rel_url.query.get("limit") or 50,
            )
            return web.json_response(result)
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            result = start_clarification(
                store=store,
                vision_brief=body.get("vision_brief") or body.get("brief") or "",
                project_id=project_id_from_payload(body),
                session_id=body.get("session_id"),
                project_context=body.get("project_context"),
                max_questions=body.get("max_questions") or 5,
            )
        except ValueError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=400)
        except Exception as exc:
            return web.json_response({"error": {"message": f"Dev clarification start failed: {exc}"}}, status=500)
        return web.json_response(result)

    async def _handle_dev_clarification_detail(self, request: "web.Request") -> "web.Response":
        """GET /v1/dev/clarifications/{clarification_id}."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_clarification_store()
        if store is None:
            return web.json_response({"error": {"message": "Dev clarification store unavailable"}}, status=503)
        try:
            result = get_clarification(store=store, clarification_id=request.match_info["clarification_id"])
        except KeyError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=404)
        return web.json_response(result)

    async def _handle_dev_clarification_answer(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/clarifications/{clarification_id}/answer."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_clarification_store()
        if store is None:
            return web.json_response({"error": {"message": "Dev clarification store unavailable"}}, status=503)
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            result = answer_clarification(
                store=store,
                clarification_id=request.match_info["clarification_id"],
                question_id=body.get("question_id"),
                option_id=body.get("option_id"),
                answer_text=body.get("answer_text"),
                skipped=_coerce_request_bool(body.get("skipped"), default=False),
                back=_coerce_request_bool(body.get("back"), default=False),
            )
        except KeyError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=404)
        except ValueError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=400)
        return web.json_response(result)

    async def _handle_dev_clarification_complete(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/clarifications/{clarification_id}/complete."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_clarification_store()
        if store is None:
            return web.json_response({"error": {"message": "Dev clarification store unavailable"}}, status=503)
        try:
            result = complete_clarification(store=store, clarification_id=request.match_info["clarification_id"])
        except KeyError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=404)
        except ValueError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=400)
        return web.json_response(result)

    async def _handle_dev_clarification_cancel(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/clarifications/{clarification_id}/cancel."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_clarification_store()
        if store is None:
            return web.json_response({"error": {"message": "Dev clarification store unavailable"}}, status=503)
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            result = cancel_clarification(
                store=store,
                clarification_id=request.match_info["clarification_id"],
                reason=body.get("reason"),
            )
        except KeyError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=404)
        except ValueError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=400)
        return web.json_response(result)

    async def _handle_dev_plan_artifacts(self, request: "web.Request") -> "web.Response":
        """GET/POST /v1/dev/plan-artifacts — list or create durable planning artifacts."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_plan_artifact_store()
        clarification_store = self._ensure_dev_clarification_store()
        if store is None or clarification_store is None:
            return web.json_response({"error": {"message": "Dev plan artifact store unavailable"}}, status=503)
        if request.method == "GET":
            result = list_plan_artifacts(
                store=store,
                clarification_id=request.rel_url.query.get("clarification_id") or None,
                project_id=request.rel_url.query.get("project_id") or None,
                status=request.rel_url.query.get("status") or None,
                limit=request.rel_url.query.get("limit") or 50,
            )
            return web.json_response(result)
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            result = create_plan_artifact(
                store=store,
                clarification_store=clarification_store,
                clarification_id=body.get("clarification_id") or "",
            )
        except KeyError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=404)
        except ValueError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=400)
        except Exception as exc:
            return web.json_response({"error": {"message": f"Dev plan artifact creation failed: {exc}"}}, status=500)
        return web.json_response(result)

    async def _handle_dev_plan_artifact_detail(self, request: "web.Request") -> "web.Response":
        """GET /v1/dev/plan-artifacts/{plan_artifact_id}."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_plan_artifact_store()
        if store is None:
            return web.json_response({"error": {"message": "Dev plan artifact store unavailable"}}, status=503)
        try:
            result = get_plan_artifact(store=store, plan_artifact_id=request.match_info["plan_artifact_id"])
        except KeyError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=404)
        return web.json_response(result)

    async def _handle_dev_plan_artifact_revise(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/plan-artifacts/{plan_artifact_id}/revise."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_plan_artifact_store()
        clarification_store = self._ensure_dev_clarification_store()
        if store is None or clarification_store is None:
            return web.json_response({"error": {"message": "Dev plan artifact store unavailable"}}, status=503)
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            result = revise_plan_artifact(
                store=store,
                clarification_store=clarification_store,
                plan_artifact_id=request.match_info["plan_artifact_id"],
                feedback=body.get("feedback_instruction") or body.get("feedback") or "",
            )
        except KeyError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=404)
        except ValueError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=400)
        except Exception as exc:
            return web.json_response({"error": {"message": f"Dev plan artifact revision failed: {exc}"}}, status=500)
        return web.json_response(result)

    async def _handle_dev_plan_artifact_approve(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/plan-artifacts/{plan_artifact_id}/approve."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_plan_artifact_store()
        if store is None:
            return web.json_response({"error": {"message": "Dev plan artifact store unavailable"}}, status=503)
        try:
            result = approve_plan_artifact(store=store, plan_artifact_id=request.match_info["plan_artifact_id"])
        except KeyError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=404)
        except ValueError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=400)
        return web.json_response(result)

    async def _handle_dev_plan_artifact_cancel(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/plan-artifacts/{plan_artifact_id}/cancel."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_plan_artifact_store()
        if store is None:
            return web.json_response({"error": {"message": "Dev plan artifact store unavailable"}}, status=503)
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            result = cancel_plan_artifact(
                store=store,
                plan_artifact_id=request.match_info["plan_artifact_id"],
                reason=body.get("reason"),
            )
        except KeyError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=404)
        except ValueError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=400)
        return web.json_response(result)

    async def _handle_dev_plan_artifact_create_execution_plan(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/plan-artifacts/{plan_artifact_id}/create-execution-plan."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        artifact_store = self._ensure_dev_plan_artifact_store()
        execution_store = self._ensure_dev_execution_store()
        if artifact_store is None or execution_store is None:
            return web.json_response({"error": {"message": "Dev plan artifact store unavailable"}}, status=503)
        try:
            result = create_execution_plan_from_artifact(
                artifact_store=artifact_store,
                execution_store=execution_store,
                plan_artifact_id=request.match_info["plan_artifact_id"],
            )
        except KeyError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=404)
        except ValueError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=400)
        except Exception as exc:
            return web.json_response({"error": {"message": f"Dev plan artifact build failed: {exc}"}}, status=500)
        return web.json_response(result)

    async def _handle_dev_plan_artifact_builds(self, request: "web.Request") -> "web.Response":
        """GET /v1/dev/plan-artifacts/{plan_artifact_id}/builds."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_plan_artifact_store()
        if store is None:
            return web.json_response({"error": {"message": "Dev plan artifact store unavailable"}}, status=503)
        try:
            result = list_plan_artifact_builds(
                store=store,
                plan_artifact_id=request.match_info["plan_artifact_id"],
                limit=request.rel_url.query.get("limit") or 25,
            )
        except KeyError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=404)
        return web.json_response(result)

    async def _handle_dev_execution_plan_draft_review(self, request: "web.Request") -> "web.Response":
        """GET /v1/dev/execution-plans/{plan_id}/draft-review."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_execution_store()
        if store is None:
            return web.json_response({"error": {"message": "Dev execution store unavailable"}}, status=503)
        try:
            result = get_execution_plan_draft_review(
                execution_store=store,
                plan_id=request.match_info["plan_id"],
            )
        except KeyError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=404)
        return web.json_response(result)

    async def _handle_dev_execution_plan_revise_draft(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/execution-plans/{plan_id}/revise-draft."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        artifact_store = self._ensure_dev_plan_artifact_store()
        execution_store = self._ensure_dev_execution_store()
        if artifact_store is None or execution_store is None:
            return web.json_response({"error": {"message": "Dev draft review store unavailable"}}, status=503)
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            result = revise_execution_plan_draft(
                artifact_store=artifact_store,
                execution_store=execution_store,
                plan_id=request.match_info["plan_id"],
                feedback=body.get("feedback_instruction") or body.get("feedback") or "",
            )
        except KeyError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=404)
        except ValueError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=400)
        except Exception as exc:
            return web.json_response({"error": {"message": f"Dev draft revision failed: {exc}"}}, status=500)
        return web.json_response(result)

    async def _handle_dev_execution_plan_approve_draft(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/execution-plans/{plan_id}/approve-draft."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_execution_store()
        if store is None:
            return web.json_response({"error": {"message": "Dev execution store unavailable"}}, status=503)
        try:
            result = approve_execution_plan_draft(
                execution_store=store,
                plan_id=request.match_info["plan_id"],
            )
        except KeyError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=404)
        except ValueError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=400)
        return web.json_response(result)

    async def _handle_dev_execution_plan_cancel_draft(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/execution-plans/{plan_id}/cancel-draft."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_execution_store()
        if store is None:
            return web.json_response({"error": {"message": "Dev execution store unavailable"}}, status=503)
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            result = cancel_execution_plan_draft(
                execution_store=store,
                plan_id=request.match_info["plan_id"],
                reason=body.get("reason"),
            )
        except KeyError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=404)
        except ValueError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=400)
        return web.json_response(result)

    async def _handle_dev_harness_components(self, request: "web.Request") -> "web.Response":
        """GET /v1/dev/harness/components — list active Dev harness components."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        components = list_harness_components(store=self._ensure_dev_execution_store())
        return web.json_response({"object": "list", "data": components, "total": len(components)})

    async def _handle_dev_harness_report(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/harness/report — build an observe-only harness experience report."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            body = await request.json()
        except Exception:
            body = {}
        store = self._ensure_dev_execution_store()
        event_store = self._ensure_subagent_event_store()
        if store is None or event_store is None:
            return web.json_response({"error": {"message": "Dev harness stores are unavailable"}}, status=503)
        try:
            report = generate_harness_report(
                store=store,
                event_store=event_store,
                plan_ids=body.get("plan_ids"),
                project_id=body.get("project_id"),
                limit=body.get("limit") or 25,
                since=body.get("since"),
                persist=_coerce_request_bool(body.get("persist"), default=True),
            )
        except KeyError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=404)
        except Exception as exc:
            return web.json_response({"error": {"message": f"Dev harness report failed: {exc}"}}, status=500)
        return web.json_response(report)

    async def _handle_dev_harness_recommendations(self, request: "web.Request") -> "web.Response":
        """GET/POST /v1/dev/harness/recommendations — list or generate recommendation runs."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_execution_store()
        event_store = self._ensure_subagent_event_store()
        if store is None or event_store is None:
            return web.json_response({"error": {"message": "Dev harness stores are unavailable"}}, status=503)
        if request.method == "GET":
            result = list_harness_recommendation_runs(
                store=store,
                report_id=request.rel_url.query.get("report_id") or None,
                limit=request.rel_url.query.get("limit") or 50,
            )
            return web.json_response(result)
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            result = generate_harness_recommendations(
                store=store,
                event_store=event_store,
                report_id=body.get("report_id"),
                plan_ids=body.get("plan_ids"),
                project_id=body.get("project_id"),
                limit=body.get("limit") or 25,
                since=body.get("since"),
                benchmark_run_id=body.get("benchmark_run_id"),
                persist=_coerce_request_bool(body.get("persist"), default=True),
            )
        except KeyError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=404)
        except Exception as exc:
            return web.json_response({"error": {"message": f"Dev harness recommendations failed: {exc}"}}, status=500)
        return web.json_response(result)

    async def _handle_dev_harness_recommendation_detail(self, request: "web.Request") -> "web.Response":
        """GET /v1/dev/harness/recommendations/{recommendation_run_id}."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_execution_store()
        if store is None:
            return web.json_response({"error": {"message": "Dev execution store unavailable"}}, status=503)
        try:
            result = get_harness_recommendation_run(
                store=store,
                recommendation_run_id=request.match_info["recommendation_run_id"],
            )
        except KeyError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=404)
        except Exception as exc:
            return web.json_response({"error": {"message": f"Dev harness recommendation lookup failed: {exc}"}}, status=500)
        return web.json_response(result)

    async def _handle_dev_harness_benchmarks(self, request: "web.Request") -> "web.Response":
        """GET/POST /v1/dev/harness/benchmarks — list or create benchmark runs."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_execution_store()
        event_store = self._ensure_subagent_event_store()
        if store is None or event_store is None:
            return web.json_response({"error": {"message": "Dev harness stores are unavailable"}}, status=503)
        if request.method == "GET":
            return web.json_response(list_harness_benchmark_runs(
                store=store,
                limit=request.rel_url.query.get("limit") or 50,
            ))
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            result = run_harness_benchmark(
                store=store,
                event_store=event_store,
                runtimes=body.get("runtimes"),
                cases=body.get("cases"),
                mode=body.get("mode"),
                live=_coerce_request_bool(body.get("live"), default=False),
                project_id=resolve_project_id(body.get("project_id")),
                max_cases=body.get("max_cases") or 3,
                iterations=body.get("iterations") or 1,
                timeout_seconds=body.get("timeout_seconds") or 180,
                persist=_coerce_request_bool(body.get("persist"), default=True),
            )
        except Exception as exc:
            return web.json_response({"error": {"message": f"Dev harness benchmark failed: {exc}"}}, status=500)
        return web.json_response(result)

    async def _handle_dev_harness_benchmark_detail(self, request: "web.Request") -> "web.Response":
        """GET /v1/dev/harness/benchmarks/{benchmark_run_id}."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_execution_store()
        if store is None:
            return web.json_response({"error": {"message": "Dev execution store unavailable"}}, status=503)
        try:
            result = get_harness_benchmark_run(
                store=store,
                benchmark_run_id=request.match_info["benchmark_run_id"],
            )
        except KeyError as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=404)
        except Exception as exc:
            return web.json_response({"error": {"message": f"Dev harness benchmark lookup failed: {exc}"}}, status=500)
        return web.json_response(result)

    async def _handle_dev_openhands_server_status(self, request: "web.Request") -> "web.Response":
        """GET /v1/dev/runtimes/openhands/server — inspect local OpenHands server."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        return web.json_response(openhands_server_status())

    async def _handle_dev_openhands_server_start(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/runtimes/openhands/server/start — start local OpenHands server."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            body = await request.json()
        except Exception:
            body = {}
        result = start_openhands_server(
            cwd=body.get("cwd"),
            server_url=body.get("server_url"),
            wait_seconds=body.get("wait_seconds") or 5.0,
        )
        return web.json_response(result)

    async def _handle_dev_openhands_server_stop(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/runtimes/openhands/server/stop — stop local OpenHands server."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        return web.json_response(stop_openhands_server())

    async def _handle_dev_execution_plans(self, request: "web.Request") -> "web.Response":
        """GET/POST /v1/dev/execution-plans — list or create Dev plans."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_execution_store()
        if store is None:
            return web.json_response(_openai_error("Dev execution store unavailable"), status=503)
        if request.method == "GET":
            limit = self._bounded_query_limit(request, default=50)
            project_id = request.rel_url.query.get("project_id") or None
            event_store = self._ensure_subagent_event_store()
            raw_plans = store.list_plans(limit=limit, project_id=project_id)
            plans = []
            try:
                from tools.ao_bridge import AOBridge

                bridge = AOBridge()
            except Exception:
                bridge = None
            for plan in raw_plans:
                if bridge is None:
                    plans.append(plan)
                    continue
                try:
                    plans.append(derive_execution_plan_status(
                        store=store,
                        plan_id=plan["plan_id"],
                        bridge=bridge,
                        event_store=event_store,
                        verification_store=self._ensure_dev_verification_store(),
                    )["plan"])
                except Exception:
                    plans.append(plan)
            return web.json_response(build_dev_plans_response(plans))
        try:
            body = await request.json()
        except Exception:
            return web.json_response(_openai_error("Invalid JSON"), status=400)
        try:
            plan = store.create_plan(
                title=body.get("title") or "Dev execution plan",
                vision_brief=body.get("vision_brief"),
                tasks=body.get("tasks") or [],
                runbook_id=body.get("runbook_id"),
                policy_profile=body.get("policy_profile"),
            )
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=400)
        return web.json_response({"ok": True, "plan": plan})

    async def _handle_dev_execution_plan_detail(self, request: "web.Request") -> "web.Response":
        """GET /v1/dev/execution-plans/{plan_id} — fetch one Dev plan."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_execution_store()
        if store is None:
            return web.json_response(_openai_error("Dev execution store unavailable"), status=503)
        plan_id = str(request.match_info.get("plan_id") or "").strip()
        try:
            result = derive_execution_plan_status(
                store=store,
                plan_id=plan_id,
                event_store=self._ensure_subagent_event_store(),
                verification_store=self._ensure_dev_verification_store(),
            )
        except KeyError:
            return web.json_response(_openai_error(f"Dev execution plan not found: {plan_id}"), status=404)
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=500)
        return web.json_response({"ok": True, "plan": result["plan"], "status": result["status"], "summary": result["summary"]})

    async def _handle_dev_execution_plan_status(self, request: "web.Request") -> "web.Response":
        """GET /v1/dev/execution-plans/{plan_id}/status — derived Dev plan status."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_execution_store()
        if store is None:
            return web.json_response(_openai_error("Dev execution store unavailable"), status=503)
        plan_id = str(request.match_info.get("plan_id") or "").strip()
        try:
            result = derive_execution_plan_status(
                store=store,
                plan_id=plan_id,
                event_store=self._ensure_subagent_event_store(),
                verification_store=self._ensure_dev_verification_store(),
            )
        except KeyError as exc:
            return web.json_response(_openai_error(str(exc)), status=404)
        except ValueError as exc:
            return web.json_response(_openai_error(str(exc)), status=400)
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=500)
        return web.json_response(result)

    async def _handle_dev_execution_plan_synthesize(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/execution-plans/{plan_id}/synthesize — compact Dev report."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_execution_store()
        if store is None:
            return web.json_response(_openai_error("Dev execution store unavailable"), status=503)
        plan_id = str(request.match_info.get("plan_id") or "").strip()
        try:
            result = synthesize_execution_plan(
                store=store,
                plan_id=plan_id,
                event_store=self._ensure_subagent_event_store(),
            )
        except KeyError as exc:
            return web.json_response(_openai_error(str(exc)), status=404)
        except ValueError as exc:
            return web.json_response(_openai_error(str(exc)), status=400)
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=500)
        return web.json_response(result)

    async def _handle_dev_execution_plan_review(self, request: "web.Request") -> "web.Response":
        """GET/POST /v1/dev/execution-plans/{plan_id}/review — Dev recommendation."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_execution_store()
        if store is None:
            return web.json_response(_openai_error("Dev execution store unavailable"), status=503)
        plan_id = str(request.match_info.get("plan_id") or "").strip()
        include_synthesis = True
        if request.method == "GET":
            include_synthesis = str(request.query.get("include_synthesis", "true")).lower() not in {"0", "false", "no"}
        else:
            try:
                body = await request.json()
            except Exception:
                body = {}
            if isinstance(body, dict) and "include_synthesis" in body:
                include_synthesis = bool(body.get("include_synthesis"))
        try:
            result = review_execution_plan(
                store=store,
                plan_id=plan_id,
                event_store=self._ensure_subagent_event_store(),
                include_synthesis=include_synthesis,
            )
        except KeyError as exc:
            return web.json_response(_openai_error(str(exc)), status=404)
        except ValueError as exc:
            return web.json_response(_openai_error(str(exc)), status=400)
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=500)
        return web.json_response(result)

    async def _handle_dev_execution_plan_apply_review(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/execution-plans/{plan_id}/apply-review — apply current Dev recommendation."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_execution_store()
        if store is None:
            return web.json_response(_openai_error("Dev execution store unavailable"), status=503)
        plan_id = str(request.match_info.get("plan_id") or "").strip()
        try:
            body = await request.json()
        except Exception:
            body = {}
        body = body if isinstance(body, dict) else {}
        try:
            result = apply_execution_plan_review(
                store=store,
                plan_id=plan_id,
                event_store=self._ensure_subagent_event_store(),
                include_synthesis=bool(body.get("include_synthesis", True)),
                message=body.get("message"),
                instruction=body.get("instruction"),
                project_id=body.get("project_id"),
                agent=body.get("agent"),
                model=body.get("model"),
                reasoning_effort=body.get("reasoning_effort"),
            )
        except KeyError as exc:
            return web.json_response(_openai_error(str(exc)), status=404)
        except ValueError as exc:
            return web.json_response(_openai_error(str(exc)), status=409)
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=500)
        return web.json_response(result)

    async def _handle_dev_execution_plans_supervise(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/execution-plans/supervise — run guarded Dev supervision."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_execution_store()
        if store is None:
            return web.json_response(_openai_error("Dev execution store unavailable"), status=503)
        try:
            body = await request.json()
        except Exception:
            body = {}
        body = body if isinstance(body, dict) else {}
        try:
            result = supervise_execution_plans(
                store=store,
                plan_ids=body.get("plan_ids") or None,
                limit=int(body.get("limit") or 20),
                project_id=body.get("project_id") or None,
                reviewable_only=bool(body.get("reviewable_only", False)),
                event_store=self._ensure_subagent_event_store(),
                apply_guarded_actions=bool(body.get("apply_guarded_actions", True)),
                include_synthesis=bool(body.get("include_synthesis", False)),
            )
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=500)
        return web.json_response(result)

    async def _handle_dev_execution_plan_next_step(self, request: "web.Request") -> "web.Response":
        """GET/POST /v1/dev/execution-plans/{plan_id}/next-step — recommendation-only next step."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_execution_store()
        if store is None:
            return web.json_response(_openai_error("Dev execution store unavailable"), status=503)
        plan_id = str(request.match_info.get("plan_id") or "").strip()
        include_synthesis = False
        if request.method == "GET":
            include_synthesis = str(request.query.get("include_synthesis", "false")).lower() in {"1", "true", "yes"}
        else:
            try:
                body = await request.json()
            except Exception:
                body = {}
            if isinstance(body, dict) and "include_synthesis" in body:
                include_synthesis = bool(body.get("include_synthesis"))
        try:
            result = next_execution_step(
                store=store,
                plan_id=plan_id,
                event_store=self._ensure_subagent_event_store(),
                include_synthesis=include_synthesis,
            )
        except KeyError as exc:
            return web.json_response(_openai_error(str(exc)), status=404)
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=500)
        return web.json_response(result)

    async def _handle_dev_execution_plan_test_state(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/execution-plans/{plan_id}/test-state — inject deterministic Dev fixture state."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_execution_store()
        if store is None:
            return web.json_response(_openai_error("Dev execution store unavailable"), status=503)
        plan_id = str(request.match_info.get("plan_id") or "").strip()
        try:
            body = await request.json()
        except Exception:
            body = {}
        body = body if isinstance(body, dict) else {}
        try:
            result = set_execution_plan_test_state(
                store=store,
                plan_id=plan_id,
                task_id=body.get("task_id") or "",
                state=body.get("state") or "",
                event_store=self._ensure_subagent_event_store(),
                summary=body.get("summary"),
                status_reason=body.get("status_reason"),
                ao_session_id=body.get("ao_session_id"),
                runtime=body.get("runtime"),
                project_id=body.get("project_id"),
                files_read=body.get("files_read") if isinstance(body.get("files_read"), list) else None,
                files_written=body.get("files_written") if isinstance(body.get("files_written"), list) else None,
                verification_evidence=body.get("verification_evidence") if isinstance(body.get("verification_evidence"), list) else None,
            )
        except KeyError as exc:
            return web.json_response(_openai_error(str(exc)), status=404)
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=400)
        return web.json_response(result)

    async def _handle_dev_runbooks(self, request: "web.Request") -> "web.Response":
        """GET/POST /v1/dev/runbooks — list or upsert project Dev runbooks."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_execution_store()
        if store is None:
            return web.json_response(_openai_error("Dev execution store unavailable"), status=503)
        if request.method == "GET":
            try:
                result = list_runbooks(
                    store=store,
                    project_id=request.query.get("project_id") or None,
                    limit=self._bounded_query_limit(request, default=100),
                )
            except Exception as exc:
                return web.json_response(_openai_error(str(exc)), status=500)
            return web.json_response(result)
        try:
            body = await request.json()
        except Exception:
            body = {}
        body = body if isinstance(body, dict) else {}
        try:
            result = set_project_runbook(
                store=store,
                project_id=body.get("project_id") or "",
                policy_profile=body.get("policy_profile") or "standard",
                max_follow_ups_per_task=body.get("max_follow_ups_per_task"),
                max_retries_per_task=body.get("max_retries_per_task"),
                supervisor_enabled=body.get("supervisor_enabled"),
                supervisor_interval_seconds=body.get("supervisor_interval_seconds"),
                supervisor_limit=body.get("supervisor_limit"),
                supervisor_include_synthesis=body.get("supervisor_include_synthesis"),
                supervisor_apply_guarded_actions=body.get("supervisor_apply_guarded_actions"),
            )
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=400)
        return web.json_response(result)

    async def _handle_dev_supervisor_loop(self, request: "web.Request") -> "web.Response":
        """GET/POST /v1/dev/supervisor/loop — inspect or configure guarded supervisor loop."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_execution_store()
        if store is None:
            return web.json_response(_openai_error("Dev execution store unavailable"), status=503)
        if request.method == "GET":
            try:
                result = list_supervisor_loop_status(
                    store=store,
                    project_id=request.query.get("project_id") or None,
                )
            except Exception as exc:
                return web.json_response(_openai_error(str(exc)), status=500)
            return web.json_response(result)
        try:
            body = await request.json()
        except Exception:
            body = {}
        body = body if isinstance(body, dict) else {}
        try:
            result = set_supervisor_loop(
                store=store,
                project_id=body.get("project_id") or "",
                supervisor_enabled=body.get("supervisor_enabled"),
                supervisor_interval_seconds=body.get("supervisor_interval_seconds"),
                supervisor_limit=body.get("supervisor_limit"),
                supervisor_include_synthesis=body.get("supervisor_include_synthesis"),
                supervisor_apply_guarded_actions=body.get("supervisor_apply_guarded_actions"),
            )
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=400)
        return web.json_response(result)

    async def _handle_dev_runbook_detail(self, request: "web.Request") -> "web.Response":
        """GET/POST /v1/dev/runbooks/{runbook_id} — fetch or update one Dev runbook."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_execution_store()
        if store is None:
            return web.json_response(_openai_error("Dev execution store unavailable"), status=503)
        runbook_id = str(request.match_info.get("runbook_id") or "").strip()
        if request.method == "GET":
            try:
                result = get_runbook(store=store, runbook_id=runbook_id)
            except KeyError as exc:
                return web.json_response(_openai_error(str(exc)), status=404)
            except Exception as exc:
                return web.json_response(_openai_error(str(exc)), status=500)
            return web.json_response(result)
        try:
            body = await request.json()
        except Exception:
            body = {}
        body = body if isinstance(body, dict) else {}
        existing = store.get_runbook(runbook_id)
        if not existing:
            return web.json_response(_openai_error(f"Dev runbook not found: {runbook_id}"), status=404)
        try:
            result = set_project_runbook(
                store=store,
                project_id=body.get("project_id") or existing.get("project_id") or "",
                policy_profile=body.get("policy_profile") or existing.get("policy_profile") or "standard",
                max_follow_ups_per_task=body.get("max_follow_ups_per_task", existing.get("max_follow_ups_per_task")),
                max_retries_per_task=body.get("max_retries_per_task", existing.get("max_retries_per_task")),
                supervisor_enabled=body.get("supervisor_enabled", existing.get("supervisor_enabled")),
                supervisor_interval_seconds=body.get("supervisor_interval_seconds", existing.get("supervisor_interval_seconds")),
                supervisor_limit=body.get("supervisor_limit", existing.get("supervisor_limit")),
                supervisor_include_synthesis=body.get("supervisor_include_synthesis", existing.get("supervisor_include_synthesis")),
                supervisor_apply_guarded_actions=body.get("supervisor_apply_guarded_actions", existing.get("supervisor_apply_guarded_actions")),
            )
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=400)
        return web.json_response(result)

    async def _handle_dev_supervisor_approvals(self, request: "web.Request") -> "web.Response":
        """GET /v1/dev/supervisor/approvals — list Dev supervisor approvals."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_execution_store()
        if store is None:
            return web.json_response(_openai_error("Dev execution store unavailable"), status=503)
        params = request.rel_url.query
        try:
            result = list_supervisor_approvals(
                store=store,
                status=params.get("status") or None,
                plan_id=params.get("plan_id") or None,
                limit=self._bounded_query_limit(request, default=50),
            )
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=500)
        return web.json_response(result)

    async def _handle_dev_supervisor_approval_detail(self, request: "web.Request") -> "web.Response":
        """GET /v1/dev/supervisor/approvals/{approval_id} — fetch one approval."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_execution_store()
        if store is None:
            return web.json_response(_openai_error("Dev execution store unavailable"), status=503)
        approval_id = str(request.match_info.get("approval_id") or "").strip()
        try:
            result = get_supervisor_approval(store=store, approval_id=approval_id)
        except KeyError as exc:
            return web.json_response(_openai_error(str(exc)), status=404)
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=500)
        return web.json_response(result)

    async def _handle_dev_supervisor_approval_approve(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/supervisor/approvals/{approval_id}/approve — approve one request."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_execution_store()
        if store is None:
            return web.json_response(_openai_error("Dev execution store unavailable"), status=503)
        approval_id = str(request.match_info.get("approval_id") or "").strip()
        try:
            body = await request.json()
        except Exception:
            body = {}
        body = body if isinstance(body, dict) else {}
        try:
            result = approve_supervisor_approval(
                store=store,
                approval_id=approval_id,
                resolved_by=body.get("resolved_by"),
                message=body.get("message"),
                instruction=body.get("instruction"),
                project_id=body.get("project_id"),
                agent=body.get("agent"),
                model=body.get("model"),
                reasoning_effort=body.get("reasoning_effort"),
            )
        except KeyError as exc:
            return web.json_response(_openai_error(str(exc)), status=404)
        except ValueError as exc:
            return web.json_response(_openai_error(str(exc)), status=409)
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=500)
        return web.json_response(result)

    async def _handle_dev_supervisor_approval_deny(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/supervisor/approvals/{approval_id}/deny — deny one request."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_execution_store()
        if store is None:
            return web.json_response(_openai_error("Dev execution store unavailable"), status=503)
        approval_id = str(request.match_info.get("approval_id") or "").strip()
        try:
            body = await request.json()
        except Exception:
            body = {}
        body = body if isinstance(body, dict) else {}
        try:
            result = deny_supervisor_approval(
                store=store,
                approval_id=approval_id,
                resolved_by=body.get("resolved_by"),
                message=body.get("message"),
            )
        except KeyError as exc:
            return web.json_response(_openai_error(str(exc)), status=404)
        except ValueError as exc:
            return web.json_response(_openai_error(str(exc)), status=409)
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=500)
        return web.json_response(result)

    async def _handle_dev_supervisor_approval_apply(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/supervisor/approvals/{approval_id}/apply — consume and apply approval."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_execution_store()
        if store is None:
            return web.json_response(_openai_error("Dev execution store unavailable"), status=503)
        approval_id = str(request.match_info.get("approval_id") or "").strip()
        try:
            body = await request.json()
        except Exception:
            body = {}
        body = body if isinstance(body, dict) else {}
        try:
            result = apply_supervisor_approval(
                store=store,
                approval_id=approval_id,
                event_store=self._ensure_subagent_event_store(),
                include_synthesis=bool(body.get("include_synthesis", True)),
            )
        except KeyError as exc:
            return web.json_response(_openai_error(str(exc)), status=404)
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=500)
        return web.json_response(result, status=200 if result.get("ok") else 409)

    async def _handle_dev_execution_plan_launch(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/execution-plans/{plan_id}/launch — launch plan tasks."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        store = self._ensure_dev_execution_store()
        event_store = self._ensure_subagent_event_store()
        if store is None:
            return web.json_response(_openai_error("Dev execution store unavailable"), status=503)
        plan_id = str(request.match_info.get("plan_id") or "").strip()
        try:
            body = await request.json()
        except Exception:
            body = {}
        task_ids = body.get("task_ids") if isinstance(body, dict) else None
        try:
            result = launch_execution_plan(
                store=store,
                plan_id=plan_id,
                task_ids=task_ids,
                event_store=event_store,
            )
        except KeyError as exc:
            return web.json_response(_openai_error(str(exc)), status=404)
        except ValueError as exc:
            return web.json_response(_openai_error(str(exc)), status=400)
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=500)
        return web.json_response(result)

    async def _handle_dev_verification_runs(self, request: "web.Request") -> "web.Response":
        """GET/POST /v1/dev/verification-runs — list or launch advisory verification."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        execution_store = self._ensure_dev_execution_store()
        verification_store = self._ensure_dev_verification_store()
        event_store = self._ensure_subagent_event_store()
        if execution_store is None or verification_store is None:
            return web.json_response(_openai_error("Dev verification store unavailable"), status=503)
        if request.method == "GET":
            try:
                result = list_verification_runs(
                    verification_store=verification_store,
                    plan_id=request.query.get("plan_id") or None,
                    task_id=request.query.get("task_id") or None,
                    limit=self._bounded_query_limit(request, default=50),
                    event_store=event_store,
                )
            except Exception as exc:
                return web.json_response(_openai_error(str(exc)), status=500)
            return web.json_response(result)
        try:
            body = await request.json()
        except Exception:
            body = {}
        body = body if isinstance(body, dict) else {}
        plan_id = str(body.get("plan_id") or "").strip()
        task_id = str(body.get("task_id") or "").strip() or None
        if not plan_id:
            return web.json_response(_openai_error("plan_id is required"), status=400)
        try:
            result = launch_verification_run(
                execution_store=execution_store,
                verification_store=verification_store,
                plan_id=plan_id,
                task_id=task_id,
                event_store=event_store,
            )
        except KeyError as exc:
            return web.json_response(_openai_error(str(exc)), status=404)
        except ValueError as exc:
            return web.json_response(_openai_error(str(exc)), status=409)
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=500)
        return web.json_response(result)

    async def _handle_dev_verification_run_detail(self, request: "web.Request") -> "web.Response":
        """GET /v1/dev/verification-runs/{verification_run_id}."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        verification_store = self._ensure_dev_verification_store()
        if verification_store is None:
            return web.json_response(_openai_error("Dev verification store unavailable"), status=503)
        run_id = str(request.match_info.get("verification_run_id") or "").strip()
        try:
            result = refresh_verification_run(
                verification_store=verification_store,
                verification_run_id=run_id,
                event_store=self._ensure_subagent_event_store(),
            )
        except KeyError as exc:
            return web.json_response(_openai_error(str(exc)), status=404)
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=500)
        return web.json_response(result)

    async def _handle_dev_signal_reports(self, request: "web.Request") -> "web.Response":
        """GET/POST /v1/dev/signal-reports — list or run production-signal digest."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        signal_store = self._ensure_dev_signal_store()
        event_store = self._ensure_subagent_event_store()
        if signal_store is None or event_store is None:
            return web.json_response(_openai_error("Dev signal stores unavailable"), status=503)
        if request.method == "GET":
            try:
                return web.json_response(list_signal_reports(
                    signal_store=signal_store,
                    limit=self._bounded_query_limit(request, default=50),
                ))
            except Exception as exc:
                return web.json_response(_openai_error(str(exc)), status=500)
        try:
            body = await request.json()
        except Exception:
            body = {}
        body = body if isinstance(body, dict) else {}
        try:
            if _coerce_request_bool(body.get("digest"), default=False):
                result = run_signal_digest(
                    signal_store=signal_store,
                    event_store=event_store,
                    product_event_store=self._ensure_dev_product_event_store() if str(body.get("source") or "").lower() == "product" else None,
                    reliability_store=self._ensure_dev_reliability_store() if str(body.get("source") or "").lower() == "reliability" else None,
                    source=body.get("source") or "deterministic",
                    window_days=body.get("window_days"),
                    filters=body.get("filters") if isinstance(body.get("filters"), dict) else {},
                    persist=_coerce_request_bool(body.get("persist"), default=True),
                )
            else:
                result = generate_signal_report(
                    signal_store=signal_store,
                    event_store=event_store,
                    product_event_store=self._ensure_dev_product_event_store() if str(body.get("source") or "").lower() == "product" else None,
                    reliability_store=self._ensure_dev_reliability_store() if str(body.get("source") or "").lower() == "reliability" else None,
                    source=body.get("source") or "deterministic",
                    window_days=body.get("window_days"),
                    filters=body.get("filters") if isinstance(body.get("filters"), dict) else {},
                    persist=_coerce_request_bool(body.get("persist"), default=True),
                    create_proposals=_coerce_request_bool(body.get("create_proposals"), default=True),
                )
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=500)
        return web.json_response(result)

    async def _handle_dev_ci_status(self, request: "web.Request") -> "web.Response":
        """GET /v1/dev/ci-status — fail-open GitHub CI state for a repo/ref."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        repo = request.rel_url.query.get("repo") or ""
        ref = request.rel_url.query.get("ref") or ""
        result = _fetch_ci_status(repo=repo, ref=ref)
        return web.json_response(result)

    def _dev_merge_readiness_snapshot(
        self,
        *,
        repo: str,
        pr_number: int,
        plan_id: Optional[str],
        task_id: Optional[str],
    ) -> Dict[str, Any]:
        scm_store = self._ensure_dev_scm_store()
        execution_store = self._ensure_dev_execution_store()
        verification_store = self._ensure_dev_verification_store()
        if scm_store is None:
            raise RuntimeError("Dev SCM lifecycle store unavailable")
        pr_state = fetch_pr_state(repo=repo, pr_number=pr_number, plan_id=plan_id, task_id=task_id)
        if pr_state.get("head_sha") or pr_state.get("warnings"):
            try:
                scm_store.upsert_pr_state(pr_state)
            except Exception as exc:
                pr_state.setdefault("warnings", []).append(f"PR state was not persisted: {exc}")
        plan = execution_store.get_plan(plan_id) if execution_store is not None and plan_id else None
        draft_status = (plan or {}).get("draft_status")
        verification = {}
        if verification_store is not None and plan_id and task_id:
            try:
                verification = verification_store.latest_for_task(plan_id=plan_id, task_id=task_id) or {}
            except Exception:
                verification = {}
        code_review = scm_store.latest_code_review(
            repo=repo,
            pr_number=int(pr_number),
            head_sha=pr_state.get("head_sha"),
        ) or scm_store.latest_code_review(repo=repo, pr_number=int(pr_number)) or {}
        readiness = compose_merge_readiness(
            repo=repo,
            pr_number=int(pr_number),
            pr_state=pr_state,
            draft_status=draft_status,
            verification=verification,
            code_review=code_review,
            plan_id=plan_id,
            task_id=task_id,
        )
        return scm_store.record_readiness(readiness)

    async def _handle_dev_pr_state(self, request: "web.Request") -> "web.Response":
        """GET /v1/dev/pr-state — fetch and persist fail-open PR state."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        scm_store = self._ensure_dev_scm_store()
        if scm_store is None:
            return web.json_response(_openai_error("Dev SCM lifecycle store unavailable"), status=503)
        try:
            result = fetch_pr_state(
                repo=request.rel_url.query.get("repo") or "",
                pr_number=int(request.rel_url.query.get("pr_number") or "0"),
                plan_id=request.rel_url.query.get("plan_id") or None,
                task_id=request.rel_url.query.get("task_id") or None,
            )
            if result.get("pr_number"):
                result = scm_store.upsert_pr_state(result)
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=400)
        return web.json_response(result)

    async def _handle_dev_code_review_runs(self, request: "web.Request") -> "web.Response":
        """GET/POST /v1/dev/code-review-runs — list or record an independent PR review."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        scm_store = self._ensure_dev_scm_store()
        execution_store = self._ensure_dev_execution_store()
        if scm_store is None:
            return web.json_response(_openai_error("Dev SCM lifecycle store unavailable"), status=503)
        if request.method == "GET":
            pr_number = request.rel_url.query.get("pr_number")
            try:
                return web.json_response({
                    "ok": True,
                    "object": "list",
                    "data": scm_store.list_code_reviews(
                        repo=request.rel_url.query.get("repo") or None,
                        pr_number=int(pr_number) if pr_number else None,
                        limit=self._bounded_query_limit(request, default=50),
                    ),
                })
            except Exception as exc:
                return web.json_response(_openai_error(str(exc)), status=400)
        try:
            body = await request.json()
        except Exception:
            body = {}
        body = body if isinstance(body, dict) else {}
        try:
            repo = str(body.get("repo") or "").strip()
            pr_number = int(body.get("pr_number") or 0)
            plan_id = str(body.get("plan_id") or "").strip() or None
            task_id = str(body.get("task_id") or "").strip() or None
            pr_state = body.get("pr_state") if isinstance(body.get("pr_state"), dict) else None
            if pr_state is None:
                pr_state = fetch_pr_state(repo=repo, pr_number=pr_number, plan_id=plan_id, task_id=task_id)
            plan = execution_store.get_plan(plan_id) if execution_store is not None and plan_id else {}
            parsed = parse_code_review_result(body.get("result") if body.get("result") is not None else body)
            prompt = build_code_review_prompt(plan=plan or {}, pr_state=pr_state)
            review = scm_store.record_code_review({
                **parsed,
                "repo": repo,
                "plan_id": plan_id,
                "task_id": task_id,
                "pr_number": pr_number,
                "head_sha": pr_state.get("head_sha"),
                "status": "completed" if body.get("result") is not None or parsed.get("verdict") != "unknown" else "requested",
                "profile_id": "review",
                "permissions": "review_only",
                "prompt": prompt,
            })
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=400)
        return web.json_response(review)

    async def _handle_dev_merge_readiness(self, request: "web.Request") -> "web.Response":
        """GET /v1/dev/merge-readiness — compose draft, CI, verification, review, and mergeability gates."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            result = self._dev_merge_readiness_snapshot(
                repo=request.rel_url.query.get("repo") or "",
                pr_number=int(request.rel_url.query.get("pr_number") or "0"),
                plan_id=request.rel_url.query.get("plan_id") or None,
                task_id=request.rel_url.query.get("task_id") or None,
            )
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=400)
        return web.json_response(result)

    async def _handle_dev_merge_approvals(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/merge-approvals — request a single-use head_sha-bound merge approval."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        scm_store = self._ensure_dev_scm_store()
        if scm_store is None:
            return web.json_response(_openai_error("Dev SCM lifecycle store unavailable"), status=503)
        try:
            body = await request.json()
        except Exception:
            body = {}
        body = body if isinstance(body, dict) else {}
        try:
            readiness = body.get("readiness") if isinstance(body.get("readiness"), dict) else None
            if readiness is None:
                readiness = self._dev_merge_readiness_snapshot(
                    repo=str(body.get("repo") or ""),
                    pr_number=int(body.get("pr_number") or 0),
                    plan_id=body.get("plan_id"),
                    task_id=body.get("task_id"),
                )
            result = request_merge_approval(
                store=scm_store,
                readiness=readiness,
                requested_by=body.get("requested_by") or "felipe",
            )
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=400)
        return web.json_response(result)

    async def _handle_dev_merge_approval_approve(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/merge-approvals/{approval_id}/approve — approve a requested merge."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        scm_store = self._ensure_dev_scm_store()
        if scm_store is None:
            return web.json_response(_openai_error("Dev SCM lifecycle store unavailable"), status=503)
        try:
            body = await request.json()
        except Exception:
            body = {}
        body = body if isinstance(body, dict) else {}
        try:
            approval = scm_store.approve_merge_approval(
                request.match_info["approval_id"],
                approved_by=body.get("approved_by") or "felipe",
                message=body.get("message"),
            )
        except KeyError as exc:
            return web.json_response(_openai_error(str(exc)), status=404)
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=400)
        return web.json_response({"ok": True, "object": "hermes.dev_merge_approval_resolution", "approval": approval})

    async def _handle_dev_merge_execute(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/merge — execute only with a live single-use approval and green gates."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        scm_store = self._ensure_dev_scm_store()
        if scm_store is None:
            return web.json_response(_openai_error("Dev SCM lifecycle store unavailable"), status=503)
        try:
            body = await request.json()
        except Exception:
            body = {}
        body = body if isinstance(body, dict) else {}
        try:
            approval_id = str(body.get("approval_id") or "").strip()
            approval = scm_store.get_merge_approval(approval_id)
            if not approval:
                return web.json_response(_openai_error("Merge approval not found"), status=404)
            live_readiness = body.get("live_readiness") if isinstance(body.get("live_readiness"), dict) else None
            if live_readiness is None:
                live_readiness = self._dev_merge_readiness_snapshot(
                    repo=approval["repo"],
                    pr_number=int(approval["pr_number"]),
                    plan_id=approval.get("plan_id"),
                    task_id=approval.get("task_id"),
                )
            result = execute_merge(
                store=scm_store,
                approval_id=approval_id,
                live_readiness=live_readiness,
                merge_method=str(body.get("merge_method") or "squash"),
            )
        except KeyError as exc:
            return web.json_response(_openai_error(str(exc)), status=404)
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=400)
        return web.json_response(result)

    async def _handle_dev_github_webhooks(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/github-webhooks — verified GitHub webhook entrypoint."""
        automation_store = self._ensure_dev_github_pr_automation_store()
        scm_store = self._ensure_dev_scm_store()
        if automation_store is None or scm_store is None:
            return web.json_response(_openai_error("Dev GitHub PR automation store unavailable"), status=503)
        secret = os.getenv("GITHUB_WEBHOOK_SECRET") or os.getenv("HERMES_GITHUB_WEBHOOK_SECRET") or ""
        if not secret.strip():
            return web.json_response(_openai_error("GITHUB_WEBHOOK_SECRET is not configured"), status=503)
        body = await request.read()
        signature = request.headers.get("X-Hub-Signature-256", "")
        if not verify_github_signature(body=body, signature_header=signature, secret=secret):
            return web.json_response(_openai_error("Invalid GitHub webhook signature"), status=401)
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            return web.json_response(_openai_error("Invalid GitHub webhook JSON"), status=400)
        if not isinstance(payload, dict):
            return web.json_response(_openai_error("GitHub webhook payload must be an object"), status=400)
        delivery_id = str(request.headers.get("X-GitHub-Delivery") or "").strip()
        event_type = str(request.headers.get("X-GitHub-Event") or "").strip()
        if not delivery_id or not event_type:
            return web.json_response(_openai_error("GitHub delivery and event headers are required"), status=400)
        try:
            result = process_github_webhook(
                store=automation_store,
                scm_store=scm_store,
                delivery_id=delivery_id,
                event_type=event_type,
                payload=payload,
            )
        except Exception as exc:
            logger.exception("[api_server] GitHub webhook processing failed")
            return web.json_response(_openai_error(str(exc)), status=500)
        return web.json_response(result)

    async def _handle_dev_pr_automation(self, request: "web.Request") -> "web.Response":
        """GET /v1/dev/pr-automation — list recent PR automation decisions."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        automation_store = self._ensure_dev_github_pr_automation_store()
        scm_store = self._ensure_dev_scm_store()
        if automation_store is None:
            return web.json_response(_openai_error("Dev GitHub PR automation store unavailable"), status=503)
        pr_number = request.rel_url.query.get("pr_number")
        try:
            result = automation_summary(
                store=automation_store,
                scm_store=scm_store,
                repo=request.rel_url.query.get("repo") or None,
                pr_number=int(pr_number) if pr_number else None,
                limit=self._bounded_query_limit(request, default=25),
            )
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=400)
        return web.json_response(result)

    async def _handle_dev_pr_automation_actions(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/pr-automation/actions — manually trigger one PR automation action."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        automation_store = self._ensure_dev_github_pr_automation_store()
        scm_store = self._ensure_dev_scm_store()
        if automation_store is None or scm_store is None:
            return web.json_response(_openai_error("Dev GitHub PR automation store unavailable"), status=503)
        try:
            body = await request.json()
        except Exception:
            body = {}
        body = body if isinstance(body, dict) else {}
        try:
            result = run_manual_pr_automation_action(
                action=str(body.get("action") or "").strip(),
                repo=str(body.get("repo") or "").strip(),
                pr_number=int(body.get("pr_number") or 0),
                store=automation_store,
                scm_store=scm_store,
            )
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=400)
        return web.json_response(result)

    async def _handle_dev_product_events(self, request: "web.Request") -> "web.Response":
        """GET/POST /v1/dev/product-events — inspect or ingest shipped-product error events."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        product_store = self._ensure_dev_product_event_store()
        if product_store is None:
            return web.json_response(_openai_error("Dev product event store unavailable"), status=503)
        if request.method == "GET":
            try:
                return web.json_response({
                    "ok": True,
                    "object": "list",
                    "data": product_store.list_events(
                        event_type=request.rel_url.query.get("type") or None,
                        limit=self._bounded_query_limit(request, default=100),
                    ),
                })
            except Exception as exc:
                return web.json_response(_openai_error(str(exc)), status=500)
        try:
            body = await request.json()
        except Exception:
            return web.json_response(_openai_error("Invalid JSON body"), status=400)
        try:
            return web.json_response(product_store.ingest_batch(body))
        except ValueError as exc:
            return web.json_response(_openai_error(str(exc)), status=400)
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=500)

    async def _handle_dev_incident_detect(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/incidents/detect — classify advisory incidents."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        incident_store = self._ensure_dev_incident_store()
        product_store = self._ensure_dev_product_event_store()
        signal_store = self._ensure_dev_signal_store()
        if incident_store is None or product_store is None:
            return web.json_response(_openai_error("Dev incident stores unavailable"), status=503)
        try:
            body = await request.json()
        except Exception:
            body = {}
        body = body if isinstance(body, dict) else {}
        try:
            result = detect_incidents(
                incident_store=incident_store,
                product_event_store=product_store,
                signal_store=signal_store,
                current_release=body.get("current_release") if isinstance(body.get("current_release"), dict) else None,
                releases=body.get("releases") if isinstance(body.get("releases"), list) else [],
                repo=str(body.get("repo") or ""),
                window_days=body.get("window_days"),
                filters=body.get("filters") if isinstance(body.get("filters"), dict) else {},
                persist=_coerce_request_bool(body.get("persist"), default=True),
            )
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=500)
        return web.json_response(result)

    async def _handle_dev_incidents(self, request: "web.Request") -> "web.Response":
        """GET /v1/dev/incidents."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        incident_store = self._ensure_dev_incident_store()
        if incident_store is None:
            return web.json_response(_openai_error("Dev incident store unavailable"), status=503)
        data = incident_store.list_incidents(
            status=request.rel_url.query.get("status") or None,
            limit=self._bounded_query_limit(request, default=50),
        )
        return web.json_response({"ok": True, "object": "list", "data": data, "total": len(data)})

    async def _handle_dev_incident_detail(self, request: "web.Request") -> "web.Response":
        """GET /v1/dev/incidents/{incident_id}."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        incident_store = self._ensure_dev_incident_store()
        if incident_store is None:
            return web.json_response(_openai_error("Dev incident store unavailable"), status=503)
        incident = incident_store.get_incident(request.match_info["incident_id"])
        if not incident:
            return web.json_response(_openai_error("Dev incident not found"), status=404)
        return web.json_response(incident)

    async def _handle_dev_incident_action(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/incidents/{incident_id}/{action}."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        incident_store = self._ensure_dev_incident_store()
        signal_store = self._ensure_dev_signal_store()
        if incident_store is None:
            return web.json_response(_openai_error("Dev incident store unavailable"), status=503)
        action = str(request.match_info.get("action") or "").strip()
        try:
            body = await request.json()
        except Exception:
            body = {}
        body = body if isinstance(body, dict) else {}
        try:
            if action == "acknowledge":
                result = acknowledge_incident(
                    incident_store=incident_store,
                    incident_id=request.match_info["incident_id"],
                )
            elif action == "resolve":
                if signal_store is None:
                    return web.json_response(_openai_error("Dev signal store unavailable"), status=503)
                result = resolve_incident(
                    incident_store=incident_store,
                    signal_store=signal_store,
                    incident_id=request.match_info["incident_id"],
                    postmortem=body.get("postmortem") if isinstance(body.get("postmortem"), dict) else body,
                )
            else:
                return web.json_response(_openai_error(f"Unsupported incident action: {action}"), status=400)
        except KeyError as exc:
            return web.json_response(_openai_error(str(exc)), status=404)
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=500)
        return web.json_response(result)

    async def _handle_dev_signal_report_detail(self, request: "web.Request") -> "web.Response":
        """GET /v1/dev/signal-reports/{report_id}."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        signal_store = self._ensure_dev_signal_store()
        if signal_store is None:
            return web.json_response(_openai_error("Dev signal store unavailable"), status=503)
        report = signal_store.get_report(request.match_info["report_id"])
        if not report:
            return web.json_response(_openai_error("Dev signal report not found"), status=404)
        return web.json_response(report)

    async def _handle_dev_backlog_proposals(self, request: "web.Request") -> "web.Response":
        """GET /v1/dev/backlog-proposals."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        signal_store = self._ensure_dev_signal_store()
        if signal_store is None:
            return web.json_response(_openai_error("Dev signal store unavailable"), status=503)
        try:
            return web.json_response(list_backlog_proposals(
                signal_store=signal_store,
                status=request.rel_url.query.get("status") or None,
                limit=self._bounded_query_limit(request, default=50),
            ))
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=500)

    async def _handle_dev_backlog_proposal_action(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/backlog-proposals/{proposal_id}/{action}."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        signal_store = self._ensure_dev_signal_store()
        event_store = self._ensure_subagent_event_store()
        if signal_store is None:
            return web.json_response(_openai_error("Dev signal store unavailable"), status=503)
        action = str(request.match_info.get("action") or "").strip()
        try:
            body = await request.json()
        except Exception:
            body = {}
        body = body if isinstance(body, dict) else {}
        try:
            if action == "measure-outcome":
                if event_store is None:
                    return web.json_response(_openai_error("Subagent event store unavailable"), status=503)
                result = measure_proposal_outcome(
                    signal_store=signal_store,
                    event_store=event_store,
                    product_event_store=self._ensure_dev_product_event_store() if str(body.get("source") or "").lower() == "product" else None,
                    reliability_store=self._ensure_dev_reliability_store() if str(body.get("source") or "").lower() == "reliability" else None,
                    proposal_id=request.match_info["proposal_id"],
                    window_days=body.get("window_days"),
                    source=body.get("source") or "deterministic",
                )
            else:
                result = transition_backlog_proposal(
                    signal_store=signal_store,
                    clarification_store=self._ensure_dev_clarification_store(),
                    proposal_id=request.match_info["proposal_id"],
                    action=action,
                    project_id=body.get("project_id"),
                )
        except KeyError as exc:
            return web.json_response(_openai_error(str(exc)), status=404)
        except ValueError as exc:
            return web.json_response(_openai_error(str(exc)), status=400)
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=500)
        return web.json_response(result)

    async def _handle_dev_signal_health(self, request: "web.Request") -> "web.Response":
        """GET /v1/dev/signal-health."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        signal_store = self._ensure_dev_signal_store()
        if signal_store is None:
            return web.json_response(_openai_error("Dev signal store unavailable"), status=503)
        return web.json_response(signal_health(
            signal_store=signal_store,
            event_store=self._ensure_subagent_event_store(),
        ))

    async def _handle_dev_reliability(self, request: "web.Request") -> "web.Response":
        """GET /v1/dev/reliability — advisory scorecard by task category."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        reliability_store = self._ensure_dev_reliability_store()
        if reliability_store is None:
            return web.json_response(_openai_error("Dev reliability store unavailable"), status=503)
        outcomes = reliability_store.list_outcomes(limit=5000)
        return web.json_response(reliability_scorecard(outcomes))

    async def _handle_dev_reliability_recompute(self, request: "web.Request") -> "web.Response":
        """POST /v1/dev/reliability/recompute — rebuild outcomes from gate stores."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            body = await request.json()
        except Exception:
            body = {}
        reliability_store = self._ensure_dev_reliability_store()
        execution_store = self._ensure_dev_execution_store()
        if reliability_store is None or execution_store is None:
            return web.json_response(_openai_error("Dev reliability stores unavailable"), status=503)
        result = recompute_reliability_outcomes(
            reliability_store=reliability_store,
            execution_store=execution_store,
            verification_store=self._ensure_dev_verification_store(),
            scm_store=self._ensure_dev_scm_store(),
            incident_store=self._ensure_dev_incident_store(),
            product_event_store=self._ensure_dev_product_event_store(),
            project_id=body.get("project_id"),
            limit=int(body.get("limit") or 200),
        )
        return web.json_response({
            **result,
            "scorecard": reliability_scorecard(reliability_store.list_outcomes(limit=5000)),
            "advisory_only": True,
        })

    async def _handle_dev_reliability_weakest(self, request: "web.Request") -> "web.Response":
        """GET /v1/dev/reliability/weakest — categories for self-improvement targeting."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        reliability_store = self._ensure_dev_reliability_store()
        if reliability_store is None:
            return web.json_response(_openai_error("Dev reliability store unavailable"), status=503)
        try:
            limit = int(request.rel_url.query.get("limit", "5"))
        except ValueError:
            limit = 5
        card = reliability_scorecard(reliability_store.list_outcomes(limit=5000))
        rows = weakest_categories(card.get("categories") or [], limit=limit)
        return web.json_response({
            "ok": True,
            "object": "list",
            "data": rows,
            "total": len(rows),
            "advisory_only": True,
        })

    async def _handle_dev_reliability_category(self, request: "web.Request") -> "web.Response":
        """GET /v1/dev/reliability/{category} — detail for one category."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        reliability_store = self._ensure_dev_reliability_store()
        if reliability_store is None:
            return web.json_response(_openai_error("Dev reliability store unavailable"), status=503)
        category = str(request.match_info.get("category") or "").strip()
        outcomes = reliability_store.list_outcomes(category=category, limit=1000)
        card = reliability_scorecard(outcomes)
        category_rows = card.get("categories") or []
        detail = category_rows[0] if category_rows else {
            "object": "hermes.dev_reliability_category",
            "category": category,
            "sample_count": 0,
            "success_rate": None,
            "tier": "unproven",
            "trend": "flat",
            "tier_reasons": ["No samples in window."],
            "advisory_only": True,
        }
        return web.json_response({
            **detail,
            "outcomes": outcomes,
            "measurements": reliability_store.list_improvement_measurements(category=category),
        })

    async def _handle_dev_lab_loop_health(self, request: "web.Request") -> "web.Response":
        """GET /v1/dev/lab-loop/health — advisory lab observe-loop health."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        loop_store = self._ensure_dev_lab_loop_store()
        if loop_store is None:
            return web.json_response(_openai_error("Dev Lab loop store unavailable"), status=503)
        return web.json_response(loop_health(db_path=loop_store.db_path))

    async def _run_dev_supervisor_loop(self) -> None:
        """Run the project-opt-in Dev supervisor loop while the gateway is alive."""
        while True:
            await asyncio.sleep(10)
            try:
                store = self._ensure_dev_execution_store()
                if store is None:
                    continue
                supervisor_loop_tick(
                    store=store,
                    event_store=self._ensure_subagent_event_store(),
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[api_server] Dev supervisor loop tick failed")
