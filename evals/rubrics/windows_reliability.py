"""Windows reliability rubric for Hermes Agent eval suite.

Checks structural invariants that matter on Windows: encoding correctness,
long-path handling, spaces-in-paths, and Unicode/emoji argument safety.
"""


def grade(scenario: dict, result: dict) -> dict:
    """Score a Windows reliability scenario.

    Args:
        scenario: The scenario dict from the YAML suite.
        result: The run result dict with keys: final_response, messages, error, api_calls.

    Returns:
        {pass: bool, score: float 0-1, details: dict}
    """
    sid = scenario.get("id", "?")
    final = result.get("final_response", "")
    messages = result.get("messages", [])
    error = result.get("error")

    # Hard fail on any error
    if error:
        return {
            "pass": False,
            "score": 0.0,
            "details": {"error": error, "reason": "scenario errored"},
        }

    # Check for mojibake (common Windows encoding failure)
    has_mojibake = _detect_mojibake(final)
    if has_mojibake:
        return {
            "pass": False,
            "score": 0.0,
            "details": {"mojibake_detected": True, "preview": final[:200]},
        }

    # Check tool errors
    has_tool_error = False
    for msg in messages:
        if msg.get("role") == "tool":
            content = str(msg.get("content", ""))
            lower = content.lower()
            if "error" in lower or "traceback" in lower or "failed" in lower:
                # Some "errors" are benign (e.g., "no errors found")
                if "no error" not in lower and "0 error" not in lower:
                    has_tool_error = True
                    break

    if has_tool_error:
        return {
            "pass": False,
            "score": 0.3,
            "details": {"tool_error": True, "reason": "tool execution failed"},
        }

    # Check specific pass conditions from scenario
    conditions = scenario.get("pass_conditions", [])
    checks_passed = 0
    details = {}

    for cond in conditions:
        ctype = cond.get("type", "")
        if ctype == "response_contains":
            val = cond.get("value", "")
            found = val.lower() in final.lower()
            details[f"contains_{val[:30]}"] = found
            if found:
                checks_passed += 1
        elif ctype == "no_tool_error":
            details["has_tool_error"] = has_tool_error
            if not has_tool_error:
                checks_passed += 1
        else:
            checks_passed += 1

    total = len(conditions) if conditions else 1
    score = checks_passed / total
    return {
        "pass": score >= 0.5,
        "score": round(score, 3),
        "details": details,
    }


def _detect_mojibake(text: str) -> bool:
    """Detect common Windows encoding corruption patterns."""
    if not text:
        return False
    # Replacement character
    if "\ufffd" in text:
        return True
    # Common double-encoding patterns
    mojibake_markers = [
        "Ã©", "Ã¨", "Ã¼",  # Latin-1 misinterpreted as UTF-8
        "â\u0080\u0099",    # Smart quote corruption
    ]
    for marker in mojibake_markers:
        if marker in text:
            return True
    return False
