from __future__ import annotations

from cory_runtime.models import ClaimedJobEnvelope
from cory_runtime.prompting import build_system_message, build_user_message


def _claim() -> ClaimedJobEnvelope:
    return ClaimedJobEnvelope.model_validate(
        {
            "ok": True,
            "claimed": True,
            "job": {
                "id": "job-1",
                "requestId": "request-1",
                "status": "running",
            },
            "request": {
                "id": "request-1",
                "title": "Need architecture options",
                "routingText": "Please compare options.",
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
                "skills": [
                    {
                        "id": "discussion-interpretation",
                        "required": True,
                        "why": "Interpret PM discussion before locking workflow.",
                    },
                    {
                        "id": "technical-option-analysis",
                        "required": False,
                        "why": "Compare candidate approaches.",
                    },
                ],
                "prompt": {
                    "systemIntent": ["You are Cory."],
                    "operatingRules": ["Do not start execution."],
                    "taskPrompt": "Interpret the collaboration request.",
                    "deliverables": ["Return valid JSON."],
                },
                "guardrails": ["Do not invent project ids."],
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


def test_build_system_message_includes_skill_content() -> None:
    message = build_system_message(_claim())

    assert "discussion-interpretation (required)" in message
    assert "technical-option-analysis (optional)" in message
    assert "Do not start execution." in message
    assert "Allowed nextWorkflowState values: needs_clarification, draft_ready" in message


def test_build_user_message_includes_request_and_contract_shape() -> None:
    message = build_user_message(_claim())

    assert '"jobId": "job-1"' in message
    assert '"sourceType": "slack"' in message
    assert '"proposedRequestType": "advisory_discussion"' in message
    assert "Interpret the collaboration request." in message
