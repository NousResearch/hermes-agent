from __future__ import annotations

import pytest

from cory_runtime.models import CoryHermesHarness
from cory_runtime.parser import OutputValidationError, parse_submission


@pytest.fixture
def harness() -> CoryHermesHarness:
    return CoryHermesHarness.model_validate(
        {
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
    )


def test_parse_submission_extracts_fenced_json_and_normalizes_defaults(
    harness: CoryHermesHarness,
) -> None:
    submission = parse_submission(
        """```json
        {
          "interpretation": {
            "interpretationStatus": "completed",
            "summary": "以 API 契約為主軸整理需求。",
            "proposedRequestType": "advisory_discussion",
            "proposedClarifications": [],
            "rawResponse": {
              "decision": "discussion",
              "evidence": ["slack discussion"]
            }
          }
        }
        ```""",
        harness,
    )

    assert submission.interpretation.producedBy == "cory_hermes"
    assert submission.nextWorkflowState == "draft_ready"


def test_parse_submission_requires_clarifications_for_clarification_state(
    harness: CoryHermesHarness,
) -> None:
    with pytest.raises(OutputValidationError):
        parse_submission(
            """
            {
              "interpretation": {
                "interpretationStatus": "needs_human_review",
                "summary": "需要更多資訊",
                "proposedRequestType": "advisory_discussion",
                "proposedClarifications": [],
                "rawResponse": {"decision": "clarify"}
              },
              "nextWorkflowState": "needs_clarification"
            }
            """,
            harness,
        )
