from __future__ import annotations

from cory_runtime.executor import HermesCoryExecutor
from cory_runtime.models import ClaimedJobEnvelope


class FakeAgent:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls: list[dict[str, object]] = []

    def run_conversation(
        self,
        user_message: str,
        system_message: str | None = None,
        conversation_history: list[dict[str, object]] | None = None,
        task_id: str | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "user_message": user_message,
                "system_message": system_message,
                "conversation_history": conversation_history,
                "task_id": task_id,
            },
        )
        return {
            "final_response": self._responses.pop(0),
            "messages": [{"role": "assistant", "content": "stub"}],
        }


def _claim() -> ClaimedJobEnvelope:
    return ClaimedJobEnvelope.model_validate(
        {
            "ok": True,
            "claimed": True,
            "job": {"id": "job-1", "requestId": "request-1", "status": "running"},
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
                "skills": [],
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


def test_executor_retries_invalid_output_and_returns_normalized_submission() -> None:
    agent = FakeAgent(
        [
            "not json at all",
            """
            {
              "interpretation": {
                "interpretationStatus": "completed",
                "summary": "先整理成可 review 的分析。",
                "proposedRequestType": "advisory_discussion",
                "proposedClarifications": [],
                "rawResponse": {"decision": "analysis"}
              }
            }
            """,
        ],
    )
    executor = HermesCoryExecutor(
        max_completion_attempts=2,
        agent_factory=lambda: agent,
    )

    submission = executor.run(_claim())

    assert submission.interpretation.producedBy == "cory_hermes"
    assert submission.nextWorkflowState == "draft_ready"
    assert len(agent.calls) == 2
    assert "Validation error:" in str(agent.calls[1]["user_message"])
