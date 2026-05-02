from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from .models import CoryHermesHarness, InterpretationSubmission


class OutputValidationError(ValueError):
    """Raised when the Hermes response cannot be normalized into the runtime contract."""


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise OutputValidationError("model returned an empty response")

    candidates = [stripped]
    fenced = _FENCED_JSON_RE.findall(stripped)
    candidates.extend(item.strip() for item in fenced)

    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

    raise OutputValidationError("model response did not contain a valid JSON object")


def normalize_submission(
    payload: dict[str, Any],
    harness: CoryHermesHarness,
) -> InterpretationSubmission:
    envelope = payload
    if "interpretation" not in envelope:
        envelope = {"interpretation": payload}

    interpretation = envelope.setdefault("interpretation", {})
    if not isinstance(interpretation, dict):
        raise OutputValidationError("interpretation must be an object")

    interpretation.setdefault("producedBy", "cory_hermes")

    allowed_states = [option.state for option in harness.outputContract.nextWorkflowStateOptions]
    next_state = envelope.get("nextWorkflowState")
    clarifications = interpretation.get("proposedClarifications") or []
    if next_state is None:
        if clarifications and "needs_clarification" in allowed_states:
            envelope["nextWorkflowState"] = "needs_clarification"
        elif "draft_ready" in allowed_states:
            envelope["nextWorkflowState"] = "draft_ready"
        elif len(allowed_states) == 1:
            envelope["nextWorkflowState"] = allowed_states[0]

    try:
        submission = InterpretationSubmission.model_validate(envelope)
    except ValidationError as exc:
        raise OutputValidationError(exc.errors()[0]["msg"]) from exc

    if submission.nextWorkflowState and submission.nextWorkflowState not in allowed_states:
        raise OutputValidationError(
            f"nextWorkflowState must be one of: {', '.join(allowed_states)}",
        )

    if (
        submission.nextWorkflowState == "needs_clarification"
        and not submission.interpretation.proposedClarifications
    ):
        raise OutputValidationError(
            "nextWorkflowState=needs_clarification requires proposedClarifications",
        )

    return submission


def parse_submission(text: str, harness: CoryHermesHarness) -> InterpretationSubmission:
    payload = extract_json_object(text)
    return normalize_submission(payload, harness)
