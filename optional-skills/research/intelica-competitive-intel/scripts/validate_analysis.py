"""Validate the shape of an Intelica /intel analysis response.

Pure logic, no network calls — used both by the agent (to sanity-check a
response before summarizing it) and by the skill's test suite.
"""
from __future__ import annotations

VALID_ACTIONS = {"enter", "avoid", "monitor", "acquire", "partner"}
VALID_CONFIDENCE = {"low", "medium", "high"}


def validate_analysis(data: dict) -> tuple[bool, str]:
    """Return (is_valid, reason). reason is empty when is_valid is True."""
    if not isinstance(data, dict):
        return False, "response is not a JSON object"

    imi = data.get("intelica_moat_index")
    if not isinstance(imi, (int, float)) or not (0 <= imi <= 1):
        return False, f"intelica_moat_index must be a number 0-1, got {imi!r}"

    decision = data.get("decision_recommendation")
    if not isinstance(decision, dict):
        return False, "decision_recommendation must be an object"

    action = decision.get("action")
    if action not in VALID_ACTIONS:
        return False, f"decision_recommendation.action must be one of {sorted(VALID_ACTIONS)}, got {action!r}"

    confidence = data.get("confidence")
    if confidence is not None and confidence not in VALID_CONFIDENCE:
        return False, f"confidence must be one of {sorted(VALID_CONFIDENCE)} or absent, got {confidence!r}"

    competitors = data.get("detected_competitors")
    if competitors is not None and not isinstance(competitors, list):
        return False, "detected_competitors must be a list when present"

    return True, ""


if __name__ == "__main__":
    import json
    import sys

    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"INVALID: could not parse JSON ({e})")
        sys.exit(1)

    ok, reason = validate_analysis(payload)
    if ok:
        print("VALID")
        sys.exit(0)
    else:
        print(f"INVALID: {reason}")
        sys.exit(1)
