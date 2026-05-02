from __future__ import annotations

import json
from typing import Any

from .models import ClaimedJobEnvelope
from .skill_loader import load_skill_bundle

_MAX_CONTEXT_CHARS = 8000


def _bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- none"


def _truncate(text: str, limit: int = _MAX_CONTEXT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[: int(limit * 0.7)].rstrip()
    tail = text[-int(limit * 0.2) :].lstrip()
    return f"{head}\n...\n[truncated]\n...\n{tail}"


def _pretty_json(data: Any) -> str:
    try:
        return _truncate(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))
    except TypeError:
        return _truncate(repr(data))


def _render_skills(claim: ClaimedJobEnvelope) -> str:
    rendered: list[str] = []
    for skill in load_skill_bundle(claim.harness.skills):
        requirement = "required" if skill.required else "optional"
        rendered.append(
            "\n".join(
                [
                    f"### {skill.id} ({requirement})",
                    f"Why it matters: {skill.why}",
                    skill.content,
                ],
            ),
        )
    return "\n\n".join(rendered)


def build_system_message(claim: ClaimedJobEnvelope) -> str:
    assert claim.harness is not None
    allowed_states = [option.state for option in claim.harness.outputContract.nextWorkflowStateOptions]

    return "\n\n".join(
        [
            "# Cory Request Interpretation Runtime",
            f"Harness version: {claim.harness.version}",
            "## Identity",
            _bullet_list(claim.harness.prompt.systemIntent),
            "## Operating Rules",
            _bullet_list(claim.harness.prompt.operatingRules),
            "## Guardrails",
            _bullet_list(claim.harness.guardrails),
            "## Deliverables",
            _bullet_list(claim.harness.prompt.deliverables),
            "## Output Contract",
            "\n".join(
                [
                    "Return only a single JSON object. Do not wrap it in markdown fences. Do not add commentary before or after the JSON.",
                    'Set `interpretation.producedBy` to `"cory_hermes"`.',
                    "Allowed interpretationStatus values: completed, needs_human_review.",
                    "Allowed proposedRequestType values: advisory_discussion, governed_change_request, execution_task, km_candidate.",
                    f"Allowed nextWorkflowState values: {', '.join(allowed_states) if allowed_states else 'none'}; you may also use null.",
                    "Use zh-TW for human-facing strings like summary, scope, non-goals, and clarification prompts.",
                    "Keep machine-facing keys, enums, and identifiers in English.",
                    "Do not claim that approval has happened or that execution has started.",
                    "Do not call tools or attempt code execution in this runtime step.",
                    "rawResponse must be an object with machine-readable rationale, evidence, and decision notes.",
                ],
            ),
            "## Cory Skills",
            _render_skills(claim),
        ],
    )


def build_user_message(claim: ClaimedJobEnvelope) -> str:
    assert claim.job is not None
    assert claim.request is not None
    assert claim.harness is not None

    interpretations = [
        {
            "id": item.id,
            "producedBy": item.producedBy,
            "interpretationStatus": item.interpretationStatus,
            "summary": item.summary,
            "proposedRequestType": item.proposedRequestType,
            "proposedScope": item.proposedScope,
            "proposedNonGoals": item.proposedNonGoals,
            "proposedClarifications": item.proposedClarifications,
        }
        for item in claim.interpretations
    ]
    clarifications = [
        {
            "id": item.id,
            "prompt": item.prompt,
            "status": item.status,
            "response": item.response,
            "answeredBy": item.answeredBy,
        }
        for item in claim.clarifications
    ]

    return "\n\n".join(
        [
            "Interpret the following control-plane request using the supplied harness and return only valid JSON.",
            "## Task Prompt",
            claim.harness.prompt.taskPrompt,
            "## Request",
            _pretty_json(
                {
                    "jobId": claim.job.id,
                    "requestId": claim.request.id,
                    "title": claim.request.title,
                    "routingText": claim.request.routingText,
                    "sourceType": claim.request.sourceType,
                    "sourceUrl": claim.request.sourceUrl,
                    "sourceEventId": claim.request.sourceEventId,
                    "requestType": claim.request.requestType,
                    "workflowState": claim.request.workflowState,
                    "requestedBy": claim.request.requestedBy,
                    "projectId": claim.request.projectId,
                    "repoId": claim.request.repoId,
                },
            ),
            "## Source Payload",
            _pretty_json(claim.request.sourcePayload),
            "## Prior Interpretations",
            _pretty_json(interpretations),
            "## Clarifications",
            _pretty_json(clarifications),
            "## Linked Execution Task",
            _pretty_json(
                None
                if claim.linkedTask is None
                else {
                    "id": claim.linkedTask.id,
                    "title": claim.linkedTask.title,
                    "status": claim.linkedTask.status,
                },
            ),
            "## Required JSON Shape",
            _pretty_json(
                {
                    "interpretation": {
                        "producedBy": "cory_hermes",
                        "interpretationStatus": "completed",
                        "summary": "zh-TW summary",
                        "proposedRequestType": "advisory_discussion",
                        "proposedProjectId": None,
                        "proposedRepoId": None,
                        "proposedScope": None,
                        "proposedNonGoals": None,
                        "proposedClarifications": [],
                        "proposedBrief": None,
                        "rawResponse": {
                            "scenario": claim.harness.scenario,
                            "decision": "short English artifact key or enum-safe label",
                            "evidence": [],
                            "notes": [],
                        },
                    },
                    "nextWorkflowState": claim.harness.outputContract.nextWorkflowStateOptions[0].state
                    if claim.harness.outputContract.nextWorkflowStateOptions
                    else None,
                },
            ),
        ],
    )


def build_repair_prompt(claim: ClaimedJobEnvelope, error_message: str) -> str:
    assert claim.harness is not None
    allowed_states = [option.state for option in claim.harness.outputContract.nextWorkflowStateOptions]
    return "\n".join(
        [
            "Your previous response did not satisfy the Cory interpretation contract.",
            f"Validation error: {error_message}",
            "Return a corrected JSON object only. No markdown fences. No extra prose.",
            "Required top-level key: interpretation.",
            "Allowed interpretationStatus values: completed, needs_human_review.",
            f"Allowed nextWorkflowState values: {', '.join(allowed_states) if allowed_states else 'none'}; you may also use null.",
        ],
    )
