from __future__ import annotations

from typing import Any

import httpx

from cory_runtime.control_plane import ControlPlaneClient
from cory_runtime.models import ClaimedJobEnvelope, InterpretationSubmission


def _claim_payload(claimed: bool = True) -> dict[str, Any]:
    return {
        "ok": True,
        "claimed": claimed,
        "job": None
        if not claimed
        else {"id": "job-1", "requestId": "request-1", "status": "running"},
        "request": None
        if not claimed
        else {
            "id": "request-1",
            "title": "Need architecture options",
            "sourceType": "slack",
            "sourcePayload": {"text": "compare architecture options"},
            "requestType": "advisory_discussion",
            "workflowState": "interpretation_pending",
        },
        "interpretations": [],
        "clarifications": [],
        "linkedTask": None,
        "harness": None
        if not claimed
        else {
            "version": "2026-05-02",
            "runtime": "hermes_agent",
            "agent": "cory",
            "mode": "request_interpretation",
            "scenario": "collaboration",
            "preferredResponseLanguage": "zh-TW",
            "artifactKeyLanguage": "en",
            "contextSummary": {
                "sourceType": "slack",
                "requestType": "advisory_discussion",
                "workflowState": "interpretation_pending",
                "interpretationCount": 0,
                "openClarificationCount": 0,
                "linkedExecutionTask": False,
            },
            "skills": [],
            "prompt": {
                "systemIntent": [],
                "operatingRules": [],
                "taskPrompt": "Interpret the request.",
                "deliverables": [],
            },
            "guardrails": [],
            "outputContract": {
                "completeEndpoint": "/api/internal/request-interpretation-jobs/job-1/complete",
                "failEndpoint": "/api/internal/request-interpretation-jobs/job-1/fail",
                "responseLanguage": "zh-TW",
                "artifactKeyLanguage": "en",
                "interpretationFields": [],
                "nextWorkflowStateOptions": [
                    {"state": "needs_clarification", "useWhen": "ambiguity remains"},
                    {"state": "draft_ready", "useWhen": "draft is reviewable"},
                ],
                "notes": [],
            },
        },
    }


def _submission() -> InterpretationSubmission:
    return InterpretationSubmission.model_validate(
        {
            "interpretation": {
                "producedBy": "cory_hermes",
                "interpretationStatus": "completed",
                "summary": "整理成 reviewable draft。",
                "proposedRequestType": "advisory_discussion",
                "proposedClarifications": [],
                "rawResponse": {"decision": "analysis"},
            },
            "nextWorkflowState": "draft_ready",
        },
    )


def test_claim_next_job_returns_none_when_queue_empty() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=_claim_payload(claimed=False)),
    )

    with ControlPlaneClient(
        base_url="http://control-plane.test",
        token="secret-token",
        transport=transport,
    ) as client:
        assert client.claim_next_job() is None


def test_complete_job_posts_bearer_token_and_submission_payload() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/claim"):
            return httpx.Response(200, json=_claim_payload(claimed=True))

        captured["path"] = request.url.path
        captured["authorization"] = request.headers.get("authorization")
        captured["json"] = request.read().decode("utf-8")
        return httpx.Response(200, json={"ok": True})

    with ControlPlaneClient(
        base_url="http://control-plane.test",
        token="secret-token",
        transport=httpx.MockTransport(handler),
    ) as client:
        claim = client.claim_next_job()
        assert isinstance(claim, ClaimedJobEnvelope)
        client.complete_job(claim, _submission())

    assert captured["path"] == "/api/internal/request-interpretation-jobs/job-1/complete"
    assert captured["authorization"] == "Bearer secret-token"
    assert '"nextWorkflowState":"draft_ready"' in captured["json"]
