"""Unit tests for the semantic judge's deterministic crash filter, structured
output schema, and edge-case classification (no LLM required).

Run:  python3 -m pytest tests/test_semantic_judge.py -q
or:   python3 tests/test_semantic_judge.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "semantic_regression"))

from semantic_judge import crash_filter  # noqa: E402
from semantic_judge.judge import _validate  # noqa: E402


# ---------------------------------------------------------------------------
# Crash filter: must route crash-related failures out of the semantic judge.
# ---------------------------------------------------------------------------
def test_traceback_is_crash():
    r = crash_filter.classify(
        "tool ran\nTraceback (most recent call last):\n  File x\nPermissionError: denied\n"
    )
    assert r["crash_related"] is True


def test_segfault_is_crash():
    assert crash_filter.classify("Segmentation fault (core dumped) exit 139")["crash_related"]


def test_oom_is_crash():
    assert crash_filter.classify("MemoryError: unable to allocate\nProcess killed (OOM)")["crash_related"]


def test_unhandled_exception_is_crash():
    assert crash_filter.classify(
        "Unhandled exception in agent loop: TypeError: 'NoneType' object is not callable"
    )["crash_related"]


def test_patch_multimatch_refusal_is_not_crash():
    r = crash_filter.classify(
        "tool patch returned error: multiple matches for old_string; replace_all not set; refusing"
    )
    assert r["crash_related"] is False


def test_terminal_exit128_is_not_crash():
    r = crash_filter.classify(
        'tool terminal completed: {"output": "fatal: not a git repository", "exit_code": 128}'
    )
    assert r["crash_related"] is False


def test_plugin_syntax_isolation_is_not_crash():
    r = crash_filter.classify(
        "ERROR: plugin broken __init__.py SyntaxError - plugin skipped (isolated); session continues"
    )
    assert r["crash_related"] is False


def test_skill_patch_notfound_is_not_crash():
    r = crash_filter.classify("skill_manage patch -> error 'old_string not found'")
    assert r["crash_related"] is False


def test_empty_log_not_crash():
    assert crash_filter.classify("")["crash_related"] is False


def test_clean_log_not_crash():
    assert crash_filter.classify(
        "[12:00:01] tool web_search query='x' limit=5\n[12:00:02] result ok"
    )["crash_related"] is False


# ---------------------------------------------------------------------------
# Structured output schema (AC: pass/fail verdict + confidence + reasoning).
# ---------------------------------------------------------------------------
def test_valid_pass():
    ok, err = _validate({"verdict": "pass", "crash_related": False,
                         "confidence": 0.9, "reasoning": ["ok"], "summary": "s"})
    assert ok and not err


def test_valid_fail_requires_reasoning():
    ok, err = _validate({"verdict": "fail", "crash_related": False,
                         "confidence": 0.8, "reasoning": ["reason here"], "summary": "s"})
    assert ok, err
    # fail with no reasoning must be rejected
    ok2, _ = _validate({"verdict": "fail", "crash_related": False,
                        "confidence": 0.8, "reasoning": [], "summary": "s"})
    assert not ok2


def test_bad_verdict_rejected():
    ok, _ = _validate({"verdict": "maybe", "crash_related": False,
                       "confidence": 0.5, "reasoning": ["r"], "summary": "s"})
    assert not ok


def test_confidence_range_enforced():
    ok, _ = _validate({"verdict": "pass", "crash_related": False,
                       "confidence": 1.5, "reasoning": ["r"], "summary": "s"})
    assert not ok


def test_crash_related_bool_enforced():
    ok, _ = _validate({"verdict": "skip", "crash_related": "yes",
                       "confidence": 1.0, "reasoning": ["r"], "summary": "s"})
    assert not ok


# ---------------------------------------------------------------------------
# Edge cases the AC explicitly calls out (unit-level, no LLM).
# ---------------------------------------------------------------------------
def test_ambiguous_behaviour_not_forced_pass_or_fail():
    # The schema must allow an explicit 'ambiguous' verdict rather than forcing
    # a false pass/fail.
    ok, err = _validate({"verdict": "ambiguous", "crash_related": False,
                         "confidence": 0.3, "reasoning": ["cannot decide"], "summary": "s"})
    assert ok and not err


def test_partial_failure_is_semantic_not_crash():
    # A partial skill-execution failure (network error on one step) is a
    # semantic contract, not a crash.
    r = crash_filter.classify(
        "[12:00:01] step 1: ok\n[12:00:02] step 2: tool network call failed (transient)\n"
        "[12:00:03] final: 'Steps 1 and 3 succeeded; step 2 failed'"
    )
    assert r["crash_related"] is False


def test_conflicting_plugin_outputs_is_semantic_not_crash():
    r = crash_filter.classify(
        "plugin gojo tool gojo-calendar date='2026-08-16' time='09:00' (wrong params vs intent)"
    )
    assert r["crash_related"] is False


def test_hook_never_fires_is_semantic_not_crash():
    r = crash_filter.classify(
        "plugin demo registered hook pre_user_message\nagent received: 'Hello' (no prefix, hook never fired)"
    )
    assert r["crash_related"] is False


# ---------------------------------------------------------------------------
# judge_run short-circuits crash inputs to the deterministic filter without LLM.
# ---------------------------------------------------------------------------
def test_judge_run_short_circuits_crash_without_llm():
    from semantic_judge.judge import judge_run
    class _NoLLM:
        def _call(self, *a, **k):
            raise AssertionError("LLM must not be called for crash-routed input")
    out = judge_run("some expected behaviour",
                    "Traceback (most recent call last):\nValueError: bad",
                    _NoLLM())
    assert out["verdict"] == "skip"
    assert out["crash_related"] is True


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS {name}")
            except Exception:
                fails += 1
                print(f"  FAIL {name}")
                traceback.print_exc()
    print(f"\n{fails} failures")
    sys.exit(1 if fails else 0)
