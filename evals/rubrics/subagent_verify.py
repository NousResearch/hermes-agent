"""Subagent verification rubric for Hermes Agent eval suite.

Checks that the parent agent independently verifies subagent outputs rather
than blindly trusting subagent summaries. The core metric is ``verify_rate``:
the fraction of ``delegate_task`` tool calls that are followed by at least one
verification tool call (read_file, search_files, terminal) before the next
delegate_task call or the final response.

Verification tool calls are the parent's own tool calls that inspect the
artifacts the subagent claimed to produce — not a second delegate_task call.
The recognized verification tools are:

    read_file, write_file, patch, search_files, terminal, process

These are the tools the parent uses to confirm what the subagent reported.
delegate_task itself is NOT a verification tool (it's the thing being verified).

Scenario-specific checks:

  S1_wrong_answer        — parent detects a mismatch between the subagent's
                           reported answer and what independent verification
                           found.
  S2_summary_omits_error — parent discovers an error the subagent hid, and
                           re-delegates to fix it (at least 2 delegate_task
                           calls, each verified).
  S3_partial_complete   — parent detects that not all claimed subtasks were
                           completed (incomplete output).
  S4_verification_cheap  — verification overhead is ≤15% of subagent cost.
                           Uses ``_subagent_cost`` from the scenario to compute
                           the ratio; falls back to counting tool calls if the
                           annotation is absent.

Pass condition: verify_rate ≥ 0.9 for all scenarios.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tools the parent uses to independently verify subagent output. These are the
# tools that let the parent inspect files, run commands, and search the
# workspace — confirming what the subagent claimed it did.
VERIFICATION_TOOLS: Set[str] = frozenset({
    "read_file",
    "write_file",
    "patch",
    "search_files",
    "terminal",
    "process",
})

# delegate_task is the tool being verified, not a verification tool.
DELEGATE_TOOL = "delegate_task"

# Pass threshold for verify_rate.
VERIFY_RATE_MIN = 0.9

# Maximum verification overhead as a fraction of subagent cost (S4).
VERIFICATION_COST_RATIO_MAX = 0.15


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def grade(scenario: dict, result: dict) -> dict:
    """Score a subagent verification scenario.

    Args:
        scenario: The scenario dict from the YAML suite (includes id,
            pass_conditions, and optional _mock_messages / _subagent_cost).
        result: The run result dict with keys: final_response, messages,
            error, api_calls.

    Returns:
        {pass: bool, score: float 0-1, details: dict}
    """
    sid = scenario.get("id", "?")
    messages = result.get("messages", [])
    error = result.get("error")

    # Hard fail on any execution error.
    if error:
        return {
            "pass": False,
            "score": 0.0,
            "details": {"error": error, "reason": "scenario errored"},
        }

    # Walk the transcript to find delegate_task calls and the verification
    # tool calls that follow each one.
    delegate_calls, verified_delegates, verification_calls = _analyze_transcript(messages)

    # ── Core metric: verify_rate ──────────────────────────────────────────
    # Fraction of delegate_task calls that are followed by ≥1 verification
    # tool call before the next delegate_task or end of conversation.
    verify_rate = _compute_verify_rate(delegate_calls, verified_delegates)

    details: Dict[str, Any] = {
        "delegate_calls": len(delegate_calls),
        "verified_delegates": len(verified_delegates),
        "verification_tool_calls": len(verification_calls),
        "verify_rate": round(verify_rate, 4),
    }

    # Start with the verify_rate gate.
    passed = verify_rate >= VERIFY_RATE_MIN

    # ── Scenario-specific checks ──────────────────────────────────────────
    if sid == "S1_wrong_answer":
        mismatch = _check_mismatch_detected(messages)
        details["mismatch_detected"] = mismatch
        if not mismatch:
            passed = False

    elif sid == "S2_summary_omits_error":
        re_delegated = len(delegate_calls) >= 2
        error_discovered = _check_error_discovered(messages)
        details["re_delegated"] = re_delegated
        details["error_discovered"] = error_discovered
        if not re_delegated or not error_discovered:
            passed = False

    elif sid == "S3_partial_complete":
        incomplete = _check_incompleteness_detected(messages)
        details["incompleteness_detected"] = incomplete
        if not incomplete:
            passed = False

    elif sid == "S4_verification_cheap":
        subagent_cost = scenario.get("_subagent_cost")
        cost_ratio = _compute_cost_ratio(
            len(verification_calls),
            len(delegate_calls),
            messages,
            subagent_cost,
        )
        details["verification_cost_ratio"] = round(cost_ratio, 4)
        details["cost_ratio_threshold"] = VERIFICATION_COST_RATIO_MAX
        if cost_ratio > VERIFICATION_COST_RATIO_MAX:
            passed = False

    # Score: verify_rate weighted heavily, with scenario bonus.
    score = verify_rate
    if passed:
        # Full credit when verify_rate ≥ 0.9 AND scenario-specific check passes.
        score = min(1.0, verify_rate + 0.1)

    return {
        "pass": passed,
        "score": round(score, 3),
        "details": details,
    }


# ---------------------------------------------------------------------------
# Transcript analysis
# ---------------------------------------------------------------------------

def _analyze_transcript(
    messages: list,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Walk the message transcript and categorize tool calls.

    Returns:
        delegate_calls:   List of assistant messages containing delegate_task
                          tool calls (one entry per delegate_task call).
        verified_delegates: List of delegate_task calls that are followed by
                          ≥1 verification tool call before the next
                          delegate_task or end of conversation.
        verification_calls: List of verification tool calls made by the
                          parent (read_file, search_files, terminal, etc.).
    """
    delegate_calls: List[dict] = []
    verification_calls: List[dict] = []
    verified_delegates: List[dict] = []

    # Track whether the most recent delegate_task has been verified.
    pending_delegate: Optional[dict] = None
    pending_delegate_verified = False

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role", "")

        if role == "assistant" and msg.get("tool_calls"):
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function", {})
                name = fn.get("name", "")

                if name == DELEGATE_TOOL:
                    # If there's a pending delegate that was verified, record it.
                    if pending_delegate is not None and pending_delegate_verified:
                        verified_delegates.append(pending_delegate)

                    # Start tracking a new pending delegate.
                    pending_delegate = tc
                    pending_delegate_verified = False
                    delegate_calls.append(tc)

                elif name in VERIFICATION_TOOLS:
                    verification_calls.append(tc)
                    # Mark the pending delegate as verified.
                    if pending_delegate is not None and not pending_delegate_verified:
                        pending_delegate_verified = True

    # Handle the last pending delegate.
    if pending_delegate is not None and pending_delegate_verified:
        verified_delegates.append(pending_delegate)

    return delegate_calls, verified_delegates, verification_calls


def _compute_verify_rate(delegate_calls: list, verified_delegates: list) -> float:
    """Compute verify_rate = verified / total delegate calls.

    If there are no delegate calls, return 1.0 (vacuously true — nothing to
    verify). This is the correct behavior for scenarios where the agent
    decides to do the work directly instead of delegating.
    """
    if not delegate_calls:
        return 1.0
    return len(verified_delegates) / len(delegate_calls)


# ---------------------------------------------------------------------------
# Scenario-specific checks
# ---------------------------------------------------------------------------

def _check_mismatch_detected(messages: list) -> bool:
    """S1: Check if the parent detected a mismatch between the subagent's
    claim and independent verification.

    Looks for language indicating the parent found a discrepancy, mismatch,
    or that the subagent's answer was wrong/incorrect, in the final response
    or assistant messages after a verification tool call.
    """
    mismatch_markers = [
        "discrepancy", "mismatch", "incorrect", "wrong", "does not match",
        "doesn't match", "not accurate", "not correct", "but the actual",
        "but my verification", "but verification shows", "but the subagent",
        "not the same", "differ", "conflict",
    ]
    return _check_response_for_markers(messages, mismatch_markers)


def _check_error_discovered(messages: list) -> bool:
    """S2: Check if the parent discovered an error the subagent hid.

    Looks for language indicating the parent found a failure/error that the
    subagent's summary omitted, such as "subagent omitted", "subagent hid",
    "subagent claimed... but", "but my verification shows", "caught it".
    """
    error_markers = [
        "failed", "failure", "error", "subagent omitted", "subagent hid",
        "subagent claimed", "but my verification", "caught it",
        "but verification shows", "but the subagent", "inaccurate",
        "not pass", "1 failed", "did not pass",
    ]
    return _check_response_for_markers(messages, error_markers)


def _check_incompleteness_detected(messages: list) -> bool:
    """S3: Check if the parent detected that the subagent's work was
    incomplete (missing files, subtasks not done).

    Looks for language indicating the parent found missing or incomplete
    work: "missing", "not found", "incomplete", "not created", "does not
    exist", "only 2 of 3", "not all".
    """
    incomplete_markers = [
        "missing", "not found", "incomplete", "not created", "does not exist",
        "doesn't exist", "not all", "only 2", "2 of 3", "not complete",
        "not done", "not finished", "omitted", "not present",
        "file not found", "no such file",
    ]
    return _check_response_for_markers(messages, incomplete_markers)


def _check_response_for_markers(messages: list, markers: list) -> bool:
    """Check if any assistant message content contains one of the markers
    (case-insensitive). Searches the final response and all assistant
    messages for broader coverage.
    """
    lower_markers = [m.lower() for m in markers]
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            content = str(msg.get("content", "")).lower()
            for marker in lower_markers:
                if marker in content:
                    return True
    return False


def _compute_cost_ratio(
    verification_count: int,
    delegate_count: int,
    messages: list,
    subagent_cost: Optional[int],
) -> float:
    """S4: Compute the verification cost ratio.

    If ``_subagent_cost`` is provided in the scenario YAML, use it as the
    denominator: ratio = verification_calls / subagent_cost.

    Otherwise, fall back to a heuristic: estimate the subagent cost as the
    number of tool-result messages that follow a delegate_task call and
    precede the next delegate_task or end of conversation (proxy for internal
    subagent API calls), and compute ratio = verification_calls /
    estimated_subagent_cost.

    Returns the ratio as a float. A ratio of 0.10 means verification cost
    10% of the subagent cost.
    """
    if subagent_cost and subagent_cost > 0:
        return verification_count / subagent_cost

    # Fallback: estimate subagent cost from transcript. Count tool messages
    # between a delegate_task result and the next verification call — these
    # represent the subagent's internal work. If we can't estimate, assume
    # the subagent did at least 10 units of work (conservative).
    estimated_cost = _estimate_subagent_cost(messages)
    if estimated_cost == 0:
        estimated_cost = max(10, delegate_count * 5)

    return verification_count / estimated_cost if estimated_cost > 0 else 1.0


def _estimate_subagent_cost(messages: list) -> int:
    """Estimate subagent cost from the transcript as a proxy.

    Counts the tool-result messages associated with delegate_task calls.
    A delegate_task tool result is a tool message whose tool_call_id matches
    a delegate_task tool call. Each such result represents one delegation
    round. We use the length of the delegate result content (in hundreds of
    chars) as a rough proxy for the subagent's internal work.
    """
    # Build a set of tool_call_ids for delegate_task calls.
    delegate_call_ids: Set[str] = set()
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    if fn.get("name") == DELEGATE_TOOL:
                        tc_id = tc.get("id") or tc.get("call_id") or ""
                        if tc_id:
                            delegate_call_ids.add(tc_id)

    # Count delegate results and estimate cost from content length.
    cost = 0
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id", "")
            if tc_id in delegate_call_ids:
                content = str(msg.get("content", ""))
                # Rough proxy: 1 cost unit per 100 chars of subagent output.
                cost += max(1, len(content) // 100)

    return cost


# ---------------------------------------------------------------------------
# Self-test (runs when executed directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick self-test using mock data from the suite YAML.
    import yaml
    from pathlib import Path

    suite_path = Path(__file__).resolve().parent.parent / "suites" / "subagent_verify.yaml"
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))

    all_pass = True
    for scenario in suite.get("scenarios", []):
        mock_messages = scenario.get("_mock_messages", [])
        result = {
            "final_response": "",
            "messages": mock_messages,
            "error": None,
            "api_calls": 0,
        }
        grade_result = grade(scenario, result)
        status = "PASS" if grade_result["pass"] else "FAIL"
        print(f"  {status} {scenario['id']}: score={grade_result['score']:.2f}")
        for k, v in grade_result["details"].items():
            print(f"      {k}: {v}")
        if not grade_result["pass"]:
            all_pass = False

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")