from __future__ import annotations

import pytest

from cory_runtime.models import ClaimedJobEnvelope, InterpretationSubmission
from cory_runtime.worker import CoryControlPlaneWorker, WorkerOutcome


class FakeClient:
    def __init__(self, claim: ClaimedJobEnvelope | None) -> None:
        self.claim = claim
        self.completed: list[InterpretationSubmission] = []
        self.failed: list[str] = []

    def claim_next_job(self) -> ClaimedJobEnvelope | None:
        claim, self.claim = self.claim, None
        return claim

    def complete_job(
        self,
        claim: ClaimedJobEnvelope,
        submission: InterpretationSubmission,
    ) -> dict[str, object]:
        self.completed.append(submission)
        return {"ok": True}

    def fail_job(self, claim: ClaimedJobEnvelope, error_message: str) -> dict[str, object]:
        self.failed.append(error_message)
        return {"ok": True}


class FakeExecutor:
    def __init__(self, result: InterpretationSubmission | None = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error

    def run(self, claim: ClaimedJobEnvelope) -> InterpretationSubmission:
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _claim() -> ClaimedJobEnvelope:
    return ClaimedJobEnvelope.model_validate(
        {
            "ok": True,
            "claimed": True,
            "job": {"id": "job-1", "requestId": "request-1", "status": "running"},
            "request": {
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
            "harness": {
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
        },
    )


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


def test_worker_run_once_completes_successfully() -> None:
    client = FakeClient(_claim())
    worker = CoryControlPlaneWorker(
        client=client,
        executor=FakeExecutor(result=_submission()),
        sleep_fn=lambda _seconds: None,
    )

    outcome = worker.run_once()

    assert outcome == WorkerOutcome.COMPLETED
    assert len(client.completed) == 1
    assert client.failed == []


def test_worker_run_once_reports_runtime_failure() -> None:
    client = FakeClient(_claim())
    worker = CoryControlPlaneWorker(
        client=client,
        executor=FakeExecutor(error=RuntimeError("provider timeout")),
        sleep_fn=lambda _seconds: None,
    )

    outcome = worker.run_once()

    assert outcome == WorkerOutcome.FAILED
    assert client.completed == []
    assert client.failed == ["provider timeout"]


def test_worker_run_once_returns_no_job_when_queue_empty() -> None:
    client = FakeClient(None)
    worker = CoryControlPlaneWorker(
        client=client,
        executor=FakeExecutor(result=_submission()),
        sleep_fn=lambda _seconds: None,
    )

    assert worker.run_once() == WorkerOutcome.NO_JOB
    assert client.completed == []
    assert client.failed == []
