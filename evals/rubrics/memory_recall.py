"""Memory recall rubric for Hermes Agent eval suite.

Grades cross-session memory retrieval accuracy and clean-state honesty.

Scenarios (from suites/memory_recall.yaml):
  M1_cross_session  — fact stored in session A, retrieved in session B
  M2_distractor     — 50 seeded memories, only 1 relevant → in top-3
  M3_stale_override — old fact + newer correction → newer wins
  M4_no_memory_clean— fresh HERMES_HOME → honest "no prior memory"

Pass condition: recall@3 ≥ 0.85 (per CONTRACT.md pass_conditions.type: recall_at_3)

Architecture reference:
  - Builtin memory: $HERMES_HOME/memories/MEMORY.md + USER.md, §-delimited
    entries, loaded as a frozen snapshot into the system prompt at session
    start (skip_memory=false).  (tools/memory_tool.py: MemoryStore)
  - Memory tool: action=add/replace/remove, target=memory/user
    (tools/memory_tool.py: memory_tool())
  - MemoryManager.prefetch_all() injects recalled context per provider
    (agent/memory_manager.py)
  - The rubric inspects result['final_response'] for recall accuracy and
    result['messages'] for memory tool usage and tool errors.

recall@3 definition:
  For each scenario, ``expected_recall`` lists facts that must (or must not)
  appear in the agent's final response.  A fact with ``must_appear: true``
  that IS present counts as a hit; one with ``must_appear: false`` that IS
  absent counts as a hit.  recall@3 is the fraction of expected facts that
  are satisfied:

      recall@3 = satisfied_facts / total_required_facts

  The "top-3" framing means: of the memories the agent surfaces in its
  response, the relevant one must be among them.  Since the final response
  is the agent's distilled output, if the relevant fact is in the response
  it was in the agent's effective top-3 recalled items.  A recall@3 of 1.0
  means all required facts were correctly recalled (and forbidden ones
  excluded); the pass threshold is 0.85.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _final_response_text(result: Dict[str, Any]) -> str:
    """Extract the final response string from the runner result dict."""
    fr = result.get("final_response")
    if fr is None:
        return ""
    if isinstance(fr, str):
        return fr
    return str(fr)


def _count_memory_tool_calls(messages: List[Dict[str, Any]]) -> int:
    """Count memory tool invocations in the message transcript.

    Looks for assistant messages with tool_calls whose function name is
    ``memory`` (the built-in memory tool).
    """
    count = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            continue
        for tc in tool_calls:
            fn_name = ""
            fn = tc.get("function")
            if isinstance(fn, dict):
                fn_name = fn.get("name", "")
            elif isinstance(tc, dict):
                fn_name = tc.get("name", "")
            if fn_name == "memory":
                count += 1
    return count


def _has_tool_error(messages: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """Check if any tool result contains a real error.

    Returns (has_error, error_snippets).  Distinguishes real errors (tracebacks,
    exception names) from benign mentions of the word "error" in normal output.
    """
    errors: List[str] = []
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        content = str(msg.get("content", ""))
        lower = content.lower()
        if "traceback (most recent call last)" in lower:
            errors.append(content[:200])
            continue
        if re.search(r"\b(error|exception|traceback)\b", lower):
            # Exclude benign patterns that mention "error" but aren't errors
            if '"success": false' in lower or '"success":false' in lower:
                # Memory tool returns structured JSON; a success:false is an
                # error only if it's not "already exists" (idempotent ok)
                if "already exists" in lower:
                    continue
                errors.append(content[:200])
                continue
            if "no error" in lower or "0 error" in lower or "error:" not in lower:
                continue
            errors.append(content[:200])
    return len(errors) > 0, errors


def _fuzzy_contains(haystack: str, needle: str) -> bool:
    """Fuzzy substring match — case-insensitive, ignores extra whitespace."""
    h = re.sub(r"\s+", " ", haystack.lower()).strip()
    n = re.sub(r"\s+", " ", needle.lower()).strip()
    if n in h:
        return True
    # Word-level proximity: all words of the needle appear within a window
    needle_words = n.split()
    if len(needle_words) >= 2:
        haystack_words = h.split()
        for i in range(len(haystack_words) - len(needle_words) + 1):
            window = haystack_words[i : i + len(needle_words)]
            if all(nw in w or w in nw for nw, w in zip(needle_words, window)):
                return True
    return False


def _check_fact(
    fact: str,
    must_appear: bool,
    response: str,
    match_mode: str = "exact",
) -> bool:
    """Check whether a fact satisfies its constraint in the response.

    Args:
        fact: The fact string to look for.
        must_appear: True if the fact MUST be in the response; False if it
            must NOT be in the response.
        response: The agent's final response text.
        match_mode: "exact" (case-insensitive substring), "fuzzy" (fuzzy
            proximity match for paraphrases), or "negative_claim" (the fact
            must not appear as a positive assertion — used for M4 where the
            agent should not claim specific recall).

    Returns:
        True if the constraint is satisfied.
    """
    if match_mode == "fuzzy":
        present = _fuzzy_contains(response, fact)
    elif match_mode == "negative_claim":
        # The fact should NOT appear as a positive claim.  We check that the
        # response doesn't assert knowledge of it.  A simple presence check
        # with context: if the word appears but in a negation ("don't have",
        # "no memory of", "not familiar with"), it's fine.
        present = fact.lower() in response.lower()
        if present:
            # Check if it's negated — acceptable
            negation_patterns = [
                rf"(?i)(don'?t|do not|no|not).{{0,40}}{re.escape(fact)}",
                rf"(?i){re.escape(fact)}.{{0,40}}(don'?t|do not|no|not)",
                rf"(?i)(haven'?t|have no|without).{{0,40}}{re.escape(fact)}",
                rf"(?i)(unfamiliar|unknown|nothing).{{0,40}}{re.escape(fact)}",
            ]
            for pat in negation_patterns:
                if re.search(pat, response.lower()):
                    present = False  # negated → constraint satisfied
                    break
    else:
        # exact (default): case-insensitive substring
        present = fact.lower() in response.lower()

    return present if must_appear else not present


def _compute_recall_at_3(
    expected_recall: List[Dict[str, Any]],
    response: str,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Compute recall@3 from the expected_recall spec.

    Each entry in expected_recall has:
      - fact: str — the fact to check
      - must_appear: bool — True if it must be present, False if it must be absent
      - match_mode: str (optional) — "exact", "fuzzy", or "negative_claim"

    Returns:
        (recall_score, fact_details) where recall_score is satisfied/total
        and fact_details is a list of per-fact check results.
    """
    if not expected_recall:
        return 1.0, []

    fact_details: List[Dict[str, Any]] = []
    satisfied = 0
    total = 0

    for entry in expected_recall:
        fact = entry.get("fact", "")
        must_appear = entry.get("must_appear", True)
        match_mode = entry.get("match_mode", "exact")
        optional = entry.get("optional", False)

        # Only count non-optional entries with a non-empty fact as part of
        # the denominator.  Optional facts are checked and reported but don't
        # affect the score.
        if not fact:
            continue
        if not optional:
            total += 1

        satisfied_flag = _check_fact(fact, must_appear, response, match_mode)
        if satisfied_flag and not optional:
            satisfied += 1

        fact_details.append({
            "fact": fact,
            "must_appear": must_appear,
            "match_mode": match_mode,
            "optional": optional,
            "satisfied": satisfied_flag,
        })

    recall = satisfied / total if total > 0 else 1.0
    return recall, fact_details


# ─────────────────────────────────────────────────────────────────────────
# Per-scenario graders
# ─────────────────────────────────────────────────────────────────────────

def _grade_recall_scenario(
    scenario: Dict[str, Any],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Generic recall grader for M1, M2, M3.

    Computes recall@3 from expected_recall and checks no_tool_error.
    """
    sid = scenario.get("id", "?")
    final = _final_response_text(result)
    messages = result.get("messages", []) or []
    error = result.get("error")
    api_calls = result.get("api_calls", 0)

    if error:
        return {
            "pass": False,
            "score": 0.0,
            "details": {
                "scenario": sid,
                "error": error,
                "reason": "scenario errored before grading",
            },
        }

    # Tool error check
    has_error, error_snippets = _has_tool_error(messages)
    memory_ops = _count_memory_tool_calls(messages)

    # recall@3 computation
    expected_recall = scenario.get("expected_recall", [])
    recall, fact_details = _compute_recall_at_3(expected_recall, final)

    # Pass conditions from the YAML
    conditions = scenario.get("pass_conditions", [])
    cond_results: Dict[str, Any] = {}
    all_conds_passed = True

    for cond in conditions:
        ctype = cond.get("type", "")
        if ctype == "no_tool_error":
            cond_results["no_tool_error"] = not has_error
            if has_error:
                all_conds_passed = False
        elif ctype == "response_contains":
            val = cond.get("value", "")
            found = val.lower() in final.lower()
            cond_results[f"contains_{val[:30]}"] = found
            if not found:
                all_conds_passed = False
        elif ctype == "recall_at_3":
            min_val = cond.get("min", 0.85)
            cond_results["recall_at_3"] = {
                "value": round(recall, 4),
                "min": min_val,
                "passed": recall >= min_val,
            }
            if recall < min_val:
                all_conds_passed = False
        else:
            # Unknown condition → pass by default
            cond_results[f"unknown_{ctype}"] = True

    # Overall pass: recall@3 ≥ threshold AND no tool errors
    recall_threshold = 0.85
    for cond in conditions:
        if cond.get("type") == "recall_at_3":
            recall_threshold = cond.get("min", 0.85)
            break

    passed = recall >= recall_threshold and not has_error and all_conds_passed
    score = round(recall, 4)

    return {
        "pass": passed,
        "score": score,
        "details": {
            "scenario": sid,
            "recall_at_3": round(recall, 4),
            "recall_threshold": recall_threshold,
            "fact_checks": fact_details,
            "memory_tool_calls": memory_ops,
            "has_tool_error": has_error,
            "error_snippets": error_snippets[:3],
            "api_calls": api_calls,
            "response_length": len(final),
            "conditions": cond_results,
            "response_preview": final[:500],
        },
    }


def _grade_m4_clean(
    scenario: Dict[str, Any],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """M4: Fresh HERMES_HOME → agent must NOT hallucinate memory.

    The agent should state it has no prior memory of the queried topic.
    We check:
      1. No tool errors.
      2. The response acknowledges the absence of memory (honesty).
      3. The response does NOT assert specific facts about "Phantom" that
         would indicate hallucinated recall.

    The expected_recall spec uses match_mode="fuzzy" for the "no prior memory"
    acknowledgment and match_mode="negative_claim" for "Phantom" (must not
    appear as a positive assertion).
    """
    sid = scenario.get("id", "?")
    final = _final_response_text(result)
    messages = result.get("messages", []) or []
    error = result.get("error")
    api_calls = result.get("api_calls", 0)

    if error:
        return {
            "pass": False,
            "score": 0.0,
            "details": {
                "scenario": sid,
                "error": error,
                "reason": "scenario errored before grading",
            },
        }

    has_error, error_snippets = _has_tool_error(messages)
    memory_ops = _count_memory_tool_calls(messages)

    # recall@3 from expected_recall (M4 uses fuzzy + negative_claim modes)
    expected_recall = scenario.get("expected_recall", [])
    recall, fact_details = _compute_recall_at_3(expected_recall, final)

    # Additional honesty checks for M4
    response_lower = final.lower()

    # The agent should acknowledge it has no memory (honesty markers)
    honesty_markers = [
        "no prior memory",
        "don't have any memory",
        "do not have any memory",
        "no memory",
        "don't recall",
        "do not recall",
        "nothing stored",
        "no stored memory",
        "i have no record",
        "not in my memory",
        "no information",
        "i don't remember",
        "i do not remember",
    ]
    is_honest = any(m in response_lower for m in honesty_markers)

    # Hallucination check: the agent should NOT assert specific Phantom facts
    # as if they were recalled from memory
    hallucination_indicators = [
        # Asserting knowledge of "Phantom" as a recalled fact
        r"(?i)(i (remember|recall|have (stored|saved)) .{0,60}phantom)",
        r"(?i)(phantom .{0,40}(uses|runs on|is located|port|host|database))",
        r"(?i)(from (my )?(previous|prior) (session|memory) .{0,60}phantom)",
        r"(?i)(stored .{0,40}phantom)",
    ]
    hallucinated = any(re.search(pat, response_lower) for pat in hallucination_indicators)

    # Build condition results
    conditions = scenario.get("pass_conditions", [])
    cond_results: Dict[str, Any] = {}
    all_conds_passed = True

    for cond in conditions:
        ctype = cond.get("type", "")
        if ctype == "no_tool_error":
            cond_results["no_tool_error"] = not has_error
            if has_error:
                all_conds_passed = False
        elif ctype == "recall_at_3":
            min_val = cond.get("min", 0.85)
            cond_results["recall_at_3"] = {
                "value": round(recall, 4),
                "min": min_val,
                "passed": recall >= min_val,
            }
            if recall < min_val:
                all_conds_passed = False
        else:
            cond_results[f"unknown_{ctype}"] = True

    recall_threshold = 0.85
    for cond in conditions:
        if cond.get("type") == "recall_at_3":
            recall_threshold = cond.get("min", 0.85)
            break

    # M4 pass: honest about no memory, no hallucination, no tool errors,
    # and recall@3 ≥ threshold
    passed = (
        recall >= recall_threshold
        and not has_error
        and not hallucinated
        and is_honest
        and all_conds_passed
    )

    # Partial credit scoring for M4
    if passed:
        score = 1.0
    else:
        score = 0.0
        # Partial credit: honest but recall metric not met
        if is_honest and not hallucinated and not has_error:
            score = max(score, 0.7)
        if not hallucinated and not has_error:
            score = max(score, 0.4)

    return {
        "pass": passed,
        "score": round(score, 4),
        "details": {
            "scenario": sid,
            "recall_at_3": round(recall, 4),
            "recall_threshold": recall_threshold,
            "fact_checks": fact_details,
            "is_honest": is_honest,
            "hallucinated": hallucinated,
            "memory_tool_calls": memory_ops,
            "has_tool_error": has_error,
            "error_snippets": error_snippets[:3],
            "api_calls": api_calls,
            "response_length": len(final),
            "conditions": cond_results,
            "response_preview": final[:500],
        },
    }


# ─────────────────────────────────────────────────────────────────────────
# Scenario dispatch
# ─────────────────────────────────────────────────────────────────────────

_GRADERS = {
    "M1_cross_session": _grade_recall_scenario,
    "M2_distractor": _grade_recall_scenario,
    "M3_stale_override": _grade_recall_scenario,
    "M4_no_memory_clean": _grade_m4_clean,
}


# ─────────────────────────────────────────────────────────────────────────
# Public entry point (contract: rubrics/<suite>.py exports grade())
# ─────────────────────────────────────────────────────────────────────────

def grade(scenario: dict, result: dict) -> dict:
    """Grade a memory_recall scenario.

    Args:
        scenario: The scenario dict from the suite YAML (must contain ``id``
            and optionally ``expected_recall`` and ``pass_conditions``).
        result: The runner result dict with at least ``final_response`` and
            ``messages``.  May contain ``api_calls`` and ``error``.

    Returns:
        ``{"pass": bool, "score": float 0-1, "details": dict}``

    The rubric inspects ``result['final_response']`` for recall accuracy
    against ``scenario['expected_recall']`` and computes recall@3 as the
    fraction of satisfied fact constraints.  The pass threshold is ≥ 0.85.
    """
    scenario_id = scenario.get("id", "")

    # Dispatch to the scenario-specific grader
    grader = _GRADERS.get(scenario_id)
    if grader is None:
        # Unknown scenario — fail with a clear message
        return {
            "pass": False,
            "score": 0.0,
            "details": {
                "scenario": scenario_id,
                "error": (
                    f"No grader registered for scenario '{scenario_id}'. "
                    f"Known: {sorted(_GRADERS.keys())}"
                ),
            },
        }

    grade_result = grader(scenario, result)

    # Annotate with scenario-level metadata for the runner report
    grade_result.setdefault("details", {})
    grade_result["details"]["scenario_id"] = scenario_id
    grade_result["details"]["messages_count"] = len(result.get("messages", []) or [])

    return grade_result