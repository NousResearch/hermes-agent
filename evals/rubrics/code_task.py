"""Code task rubric for Hermes Agent eval suite.

Grades end-to-end coding scenarios: bug fixes, feature implementation, refactoring.
For live-model runs only (Tier 2).

Design notes
------------
``C1_bugfix_repro`` is a *reproduce-then-fix* scenario: the agent is told to
run a test that will **intentionally fail** on the first attempt (the user
message even says "This will fail"), then fix the code and re-run until it
passes.  The rubric must therefore distinguish *expected intermediate
failures* (the first test run's traceback / assertion error) from *genuine
tool errors* (a broken file write, a crash, etc.).

The general principle: for any scenario whose description or user message
indicates a reproduce-then-fix workflow, we only fail the ``no_tool_error``
condition if the **final** terminal/test output does not show a passing state.
Intermediate test failures that are later followed by a passing run are not
penalised.
"""

import re

# ---------------------------------------------------------------------------
# Scenario IDs where an *initial* test failure is expected by design.
# The agent reproduces a bug first, then fixes it — so the first test run
# will contain a traceback / AssertionError and that is NOT a real error.
# ---------------------------------------------------------------------------
_EXPECTED_INITIAL_FAILURE_SCENARIOS = frozenset({
    "C1_bugfix_repro",
})


def grade(scenario: dict, result: dict) -> dict:
    """Score a code task scenario.

    Checks: no *real* tool errors, test output contains PASS/success
    indicators, and the final response indicates completion.
    """
    sid = scenario.get("id", "?")
    final = result.get("final_response", "")
    messages = result.get("messages", [])
    error = result.get("error")
    api_calls = result.get("api_calls", 0)

    if error:
        return {
            "pass": False,
            "score": 0.0,
            "details": {"error": error, "reason": "scenario errored"},
        }

    # ------------------------------------------------------------------
    # Collect terminal outputs in *order* so we can reason about the
    # sequence of test runs (first failure → later pass).
    # ------------------------------------------------------------------
    terminal_outputs: list[str] = []
    for msg in messages:
        if msg.get("role") == "tool" and msg.get("name") == "terminal":
            content = str(msg.get("content", ""))
            terminal_outputs.append(content)

    # Did any terminal output show a test-passing state?
    test_passed = any(_has_test_pass(t) for t in terminal_outputs)

    # Did the *last* terminal output show a test-passing state?
    # For reproduce-then-fix scenarios this is the meaningful signal.
    final_terminal = terminal_outputs[-1] if terminal_outputs else ""
    final_test_passed = _has_test_pass(final_terminal) if final_terminal else False

    # ------------------------------------------------------------------
    # Determine whether there is a *real* tool error.
    #
    # For scenarios with expected initial failures (C1_bugfix_repro), a
    # traceback / AssertionError / ZeroDivisionError in an *intermediate*
    # run is NOT a real error — it's the reproduction step.  We only flag
    # a real error if:
    #   • the error appears in a terminal output that does NOT also show
    #     a passing state, AND
    #   • there is no later terminal output that shows a pass.
    #
    # Put simply: if the agent eventually got tests to pass, intermediate
    # failures are forgiven for these scenarios.
    # ------------------------------------------------------------------
    expects_initial_failure = sid in _EXPECTED_INITIAL_FAILURE_SCENARIOS

    has_error = False
    error_details: list[str] = []
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        content = str(msg.get("content", ""))
        if not _is_real_error(content):
            continue

        # For expected-initial-failure scenarios: if the *final* test run
        # passed, then any earlier error was the expected reproduction
        # failure — skip it.
        if expects_initial_failure and final_test_passed:
            continue

        has_error = True
        error_details.append(content[:200])

    conditions = scenario.get("pass_conditions", [])
    checks_passed = 0
    details: dict = {
        "test_passed": test_passed,
        "final_test_passed": final_test_passed,
        "has_tool_error": has_error,
        "api_calls": api_calls,
        "expects_initial_failure": expects_initial_failure,
    }
    if error_details:
        # Keep details readable — truncate to first 3 snippets.
        details["error_snippets"] = error_details[:3]

    for cond in conditions:
        ctype = cond.get("type", "")
        if ctype == "no_tool_error":
            if not has_error:
                checks_passed += 1
        elif ctype == "response_contains":
            val = cond.get("value", "")
            found = val.lower() in final.lower() or any(
                val.lower() in t.lower() for t in terminal_outputs
            )
            details[f"contains_{val[:30]}"] = found
            if found:
                checks_passed += 1
        else:
            checks_passed += 1

    total = len(conditions) if conditions else 1
    if total == 0:
        total = 1

    # Bonus: test actually passed and no real errors remain.
    if test_passed and not has_error:
        checks_passed = max(checks_passed, total)

    score = min(checks_passed / total, 1.0)
    return {
        "pass": score >= 0.6 and not has_error,
        "score": round(score, 3),
        "details": details,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Patterns that indicate tests **passed**.  Ordered roughly from most
# specific to most general.
_PASS_PATTERNS: list[re.Pattern] = [
    # pytest: "===== 3 passed in 0.05s =====" (must have digit(s) before
    # "passed" to avoid matching "0 passed" — handled separately by the
    # failed-check below).
    re.compile(r"\d+\s+passed\b", re.IGNORECASE),
    # unittest: "Ran 3 tests in 0.001s\n\nOK"
    re.compile(r"\bOK\b"),
    # Plain "PASS" on its own (common in simple scripts: print("PASS"))
    re.compile(r"\bPASS\b"),
    # "All tests passed" / "all tests pass"
    re.compile(r"all\s+tests?\s+(?:passed|pass)\b", re.IGNORECASE),
    # Generic trailing "pass" / "pass." (e.g. script prints "pass")
    re.compile(r"(?:^|\n)\s*pass\.?\s*$", re.IGNORECASE),
    # "Tests: X passed" (some reporters)
    re.compile(r"tests?:\s*\d+\s+passed\b", re.IGNORECASE),
    # "✓" / "✔" check marks in mocha-style or custom runners
    re.compile(r"[✓✔]\s*\d+\s+", re.IGNORECASE),
]

# Patterns that indicate tests **failed** — if present alongside a pass
# pattern, the output is not a clean pass.
_FAIL_INDICATORS = [
    "failed",
    "failures",
    "error",
    "traceback",
    "assertionerror",
    "assertion error",
]


def _has_test_pass(output: str) -> bool:
    """Return True if *output* indicates that tests passed.

    This is intentionally lenient: it scans for common pass indicators
    across unittest, pytest, mocha, jest, and simple ``print("PASS")``
    scripts.  When a pass pattern is found, it is only accepted if the
    same output chunk does not also contain a clear failure indicator
    (e.g. "0 passed, 1 failed").

    Edge cases handled:
    - ``"0 passed"`` is NOT a pass (the ``\\d+`` before "passed" can be 0).
    - ``"OK"`` inside a larger word (e.g. "BROKEN") is not matched
      because we use ``\\bOK\\b``.
    - A line that says ``pass`` at the very end of output is accepted,
      matching simple verification scripts.
    """
    if not output:
        return False

    lower = output.lower()

    # Quick short-circuit: if there's an explicit failure with zero
    # passes, it's definitely not a pass.
    if re.search(r"\b0\s+passed\b", lower):
        return False
    if re.search(r"\b\d+\s+failed\b", lower) and not re.search(
        r"\b[1-9]\d*\s+passed\b", lower
    ):
        # "N failed" with no positive "M passed" → not a pass
        return False

    # Check each pass pattern.
    for pat in _PASS_PATTERNS:
        m = pat.search(output)
        if not m:
            continue

        # Reject if the matched region also contains a failure indicator.
        # We look at a window around the match to catch lines like
        # "1 passed, 1 failed".
        start = max(0, m.start() - 60)
        end = min(len(lower), m.end() + 60)
        window = lower[start:end]
        if any(ind in window for ind in _FAIL_INDICATORS):
            # Allow "OK" from unittest even if "error" appears elsewhere
            # in the window — unittest prints "OK" only on full success.
            if pat.pattern == r"\bOK\b":
                return True
            # For "N passed" patterns, require that failures count is 0.
            if re.search(r"\b0\s+failed\b", window):
                return True
            continue
        return True

    return False


def _is_real_error(output: str) -> bool:
    """Check if tool output contains a real error (not benign mentions).

    Distinguishes actual exceptions / crashes from informational text
    that merely *mentions* the word "error" (e.g. "0 errors", "no error
    found").
    """
    lower = output.lower()

    # Python tracebacks.
    if "traceback (most recent call last)" in lower:
        return True

    # "error:" as a prefix (compiler/linter style) — but exclude benign
    # mentions like "no error" or "0 error".
    if "error:" in lower and "no error" not in lower and "0 error" not in lower:
        return True

    # Assertion errors (test failures).
    if "assertionerror" in lower or "assertion error" in lower:
        return True

    # Syntax errors.
    if "syntaxerror" in lower:
        return True

    # Import / module errors.
    if "importerror" in lower or "modulenotfounderror" in lower:
        return True

    # Common runtime exceptions that indicate a genuine crash.
    for exc in (
        "zerodivisionerror",
        "typeerror:",
        "valueerror:",
        "keyerror:",
        "attributeerror:",
        "nameerror:",
        "runtimeerror:",
    ):
        if exc in lower:
            return True

    return False


# ---------------------------------------------------------------------------
# Self-test (per BEST_PRACTICES.md §2.4)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """Exercise each pass/fail path with synthetic result dicts."""

    def _mk_msg(role: str, content: str, name: str = None) -> dict:
        m = {"role": role, "content": content}
        if name:
            m["name"] = name
        return m

    def _mk_result(messages: list, final: str = "Done, all tests pass.") -> dict:
        return {"final_response": final, "messages": messages, "api_calls": 5, "error": None}

    # --- C1_bugfix_repro: expected initial failure then pass ----------
    c1_scenario = {
        "id": "C1_bugfix_repro",
        "pass_conditions": [
            {"type": "no_tool_error"},
            {"type": "response_contains", "value": "pass"},
        ],
    }
    c1_result = _mk_result([
        _mk_msg("assistant", "Let me create the file and run the tests."),
        _mk_msg("tool", "File created.", "file"),
        # First test run — intentional failure (ZeroDivisionError + traceback)
        _mk_msg("tool",
                "Traceback (most recent call last):\n"
                "  File \"buggy.py\", line 5, in test_divide\n"
                "    assert divide(5, 0) == \"cannot divide by zero\"\n"
                "ZeroDivisionError: division by zero\n"
                "exit code: 1",
                "terminal"),
        # Fix applied
        _mk_msg("tool", "File updated.", "file"),
        # Second test run — passes
        _mk_msg("tool",
                "....\n"
                "----------------------------------------------------------------------\n"
                "Ran 4 tests in 0.001s\n\n"
                "OK\n",
                "terminal"),
    ])

    # --- C1: final test still fails (genuine failure) ----------------
    c1_fail_result = _mk_result([
        _mk_msg("tool", "File created.", "file"),
        _mk_msg("tool",
                "Traceback (most recent call last):\n"
                "  File \"buggy.py\", line 5\n"
                "ZeroDivisionError: division by zero\n",
                "terminal"),
        _mk_msg("tool", "File updated.", "file"),
        _mk_msg("tool",
                "F...\n"
                "======================================================================\n"
                "FAIL: test_divide\n"
                "----------------------------------------------------------------------\n"
                "Ran 4 tests in 0.001s\n\n"
                "FAILED (failures=1)\n",
                "terminal"),
    ])

    # --- C2_feature_tdd: no initial failure, clean pass ---------------
    c2_scenario = {
        "id": "C2_feature_tdd",
        "pass_conditions": [
            {"type": "no_tool_error"},
            {"type": "response_contains", "value": "pass"},
        ],
    }
    c2_result = _mk_result([
        _mk_msg("tool", "File created.", "file"),
        _mk_msg("tool",
                ".....\n"
                "----------------------------------------------------------------------\n"
                "Ran 5 tests in 0.001s\n\n"
                "OK\n",
                "terminal"),
    ])

    # --- C2: genuine tool error (import error) ------------------------
    c2_err_result = _mk_result([
        _mk_msg("tool",
                "Traceback (most recent call last):\n"
                "ModuleNotFoundError: No module named 'pytest'\n",
                "terminal"),
    ])

    # --- C3_refactor: pytest-style output -----------------------------
    c3_scenario = {
        "id": "C3_refactor_extract",
        "pass_conditions": [
            {"type": "no_tool_error"},
            {"type": "response_contains", "value": "PASS"},
        ],
    }
    c3_result = _mk_result([
        _mk_msg("tool", "===== 3 passed in 0.05s =====\n", "terminal"),
    ])

    # --- Edge: "0 passed, 1 failed" should NOT be a pass --------------
    edge_0_passed = _mk_result([
        _mk_msg("tool", "===== 0 passed, 1 failed in 0.03s =====\n", "terminal"),
    ])

    # --- Edge: "1 passed, 0 failed" should be a pass ------------------
    edge_1_passed_0_failed = _mk_result([
        _mk_msg("tool", "===== 1 passed, 0 failed in 0.03s =====\n", "terminal"),
    ])

    # --- Edge: simple print("PASS") -----------------------------------
    edge_simple_pass = _mk_result([
        _mk_msg("tool", "PASS\n", "terminal"),
    ])

    # --- Edge: "pass" at end of output --------------------------------
    edge_trailing_pass = _mk_result([
        _mk_msg("tool", "Running tests...\nAll good\npass\n", "terminal"),
    ])

    # --- Run tests ----------------------------------------------------
    tests = [
        ("C1 expected-fail-then-pass → PASS", c1_scenario, c1_result, True),
        ("C1 final still fails → FAIL", c1_scenario, c1_fail_result, False),
        ("C2 clean pass → PASS", c2_scenario, c2_result, True),
        ("C2 import error → FAIL", c2_scenario, c2_err_result, False),
        ("C3 pytest passed → PASS", c3_scenario, c3_result, True),
        ("edge: 0 passed 1 failed → _has_test_pass=False", None, edge_0_passed, None),
        ("edge: 1 passed 0 failed → _has_test_pass=True", None, edge_1_passed_0_failed, None),
        ("edge: simple PASS → _has_test_pass=True", None, edge_simple_pass, None),
        ("edge: trailing pass → _has_test_pass=True", None, edge_trailing_pass, None),
    ]

    failures = 0
    for label, scenario, result, expect_pass in tests:
        if scenario is None:
            # Direct _has_test_pass test
            out = result["messages"][0]["content"]
            got = _has_test_pass(out)
            want = expect_pass if isinstance(expect_pass, bool) else (expect_pass is True)
            # For edge tests we check the expected boolean directly
            if "0 passed" in label:
                want = False
            elif "1 passed" in label or "simple PASS" in label or "trailing pass" in label:
                want = True
            status = "PASS" if got == want else "FAIL"
            if got != want:
                failures += 1
            print(f"  [{status}] {label}: _has_test_pass={got} (want {want})")
        else:
            r = grade(scenario, result)
            status = "PASS" if r["pass"] == expect_pass else "FAIL"
            if r["pass"] != expect_pass:
                failures += 1
            print(
                f"  [{status}] {label}: pass={r['pass']} score={r['score']:.2f} "
                f"details={r['details']}"
            )

    print(f"\n{len(tests) - failures}/{len(tests)} checks passed (self-test)")
    if failures:
        raise SystemExit(1)