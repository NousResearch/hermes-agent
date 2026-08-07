"""Validate the shape of a SENTINEL /v1/guard verdict response.

Pure logic, no network calls — used both by the agent (to sanity-check a
response before acting on it) and by the skill's test suite (to exercise a
real shipped artifact instead of only asserting on canned mock data).
"""
from __future__ import annotations

VALID_VERDICTS = {"SAFE", "UNSAFE", "UNKNOWN"}
VALID_GRADES = {"AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"}


def validate_verdict(data: dict) -> tuple[bool, str]:
    """Return (is_valid, reason). reason is empty when is_valid is True."""
    if not isinstance(data, dict):
        return False, "response is not a JSON object"

    verdict = data.get("verdict")
    if verdict not in VALID_VERDICTS:
        return False, f"verdict must be one of {sorted(VALID_VERDICTS)}, got {verdict!r}"

    score = data.get("sentinelScore")
    if not isinstance(score, (int, float)) or not (0 <= score <= 100):
        return False, f"sentinelScore must be a number 0-100, got {score!r}"

    grade = data.get("grade")
    if grade is not None and grade not in VALID_GRADES:
        return False, f"grade must be one of {sorted(VALID_GRADES)} or absent, got {grade!r}"

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

    ok, reason = validate_verdict(payload)
    if ok:
        print("VALID")
        sys.exit(0)
    else:
        print(f"INVALID: {reason}")
        sys.exit(1)
