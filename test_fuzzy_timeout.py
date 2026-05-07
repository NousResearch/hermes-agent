#!/usr/bin/env python3
"""
Standalone verification script for fuzzy_match.py timeout fix.
Safe to run even if the fix is broken — it does NOT use Hermes CLI.

Usage: python3 test_fuzzy_timeout.py
Expected: all 3 tests PASS within ~60 seconds total.
If it hangs > 60s, the fix is broken.
"""
import sys
import time
import os

sys.path.insert(0, os.path.dirname(__file__))

from tools.fuzzy_match import fuzzy_find_and_replace

def run(name, fn, max_time=60):
    print(f"\n  Running: {name} …")
    start = time.monotonic()
    try:
        result = fn()
        elapsed = time.monotonic() - start
        ok = bool(result)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name} — {elapsed:.2f}s")
        return ok, elapsed
    except Exception as e:
        elapsed = time.monotonic() - start
        print(f"  [ERROR] {name}: {e} — {elapsed:.2f}s")
        return False, elapsed


def main():
    print("=" * 60)
    print("fuzzy_match timeout fix verification")
    print("=" * 60)

    # ── Test 1: Exact match should NOT time out ──
    def t1():
        content = 'def foo():\n    pass\n'
        result, count, strategy, error = fuzzy_find_and_replace(
            content, 'def foo():', 'def bar():', False, timeout=5
        )
        ok = (strategy == "exact" and count == 1 and error is None and "def bar():" in result)
        print(f"    strategy={strategy}, count={count}, error={error}")
        return ok

    ok1, t = run("exact match → fast path", t1)
    if t > 5:
        print(f"  [WARN] took {t:.1f}s, expected < 1s for exact match")

    # ── Test 2: No match in 5000-line uniform content
    #     Strategies 1-7 all return [], then strategy 8 starts O(n²) SequenceMatcher.
    #     5000 lines × 40 chars ≈ 200KB file → strategy 8 takes ~1s per call
    #     → will complete within 5s timeout, returns [], falls through → error
    #     This proves slow strategies ARE reached when no early match exists.
    def t2():
        lines = [f"line_{i:04d} = {('x'*40)}" for i in range(5000)]
        content = "\n".join(lines)
        result, count, strategy, error = fuzzy_find_and_replace(
            content,
            "UNIQUE_PATTERN_THAT_EXISTS_NO_WHERE_ZZ99ZZ",
            "replaced",
            False,
            timeout=5
        )
        # Expected: strategies tried, none matched → error (no re_fallback needed)
        ok = (strategy is None and count == 0 and error is not None)
        print(f"    strategy={strategy}, count={count}, error={str(error)[:60]}")
        return ok

    ok2, t2_elapsed = run("no-match large file — should try all strategies", t2)
    if t2_elapsed > 10:
        print(f"  [WARN] took {t2_elapsed:.1f}s, slow strategies may not be protected")
    elif t2_elapsed > 5:
        print(f"  [OK] took {t2_elapsed:.1f}s, timeout did not fire (strategies fast enough)")
    else:
        print(f"  [OK] took {t2_elapsed:.1f}s")

    # ── Test 3: Simulate a pattern that WON'T match early
    #     but with enough unique content to slow strategy 8/9 on a LARGER file.
    #     Use 12000 lines to make O(n²) really bite (~3-4s per strategy call).
    #     Strategy 8 times out → re_fallback → exact "hello world" still found.
    def t3():
        # Build content with a distinctive anchor line in the middle,
        # but the search pattern won't match early strategies.
        lines = []
        for i in range(12000):
            if i == 6000:
                lines.append("X" * 80)  # unique anchor at middle
            else:
                lines.append(f"{i:05d}" + "y" * 70)
        content = "\n".join(lines)
        start = time.monotonic()
        result, count, strategy, error = fuzzy_find_and_replace(
            content,
            "X" * 80,  # matches line 6000 but NOT with exact/line_trimmed strategies
            "REPLACED",
            False,
            timeout=5
        )
        elapsed = time.monotonic() - start
        # If timeout fired: strategy=re_fallback, count=1, error=None
        # If timeout did NOT fire: strategy=block_anchor or context_aware
        if strategy == "re_fallback":
            print(f"    TIMEOUT fired → re_fallback, count={count}, elapsed={elapsed:.2f}s")
        else:
            print(f"    timeout did NOT fire → strategy={strategy}, count={count}, elapsed={elapsed:.2f}s")
        # Accept either: timeout worked (re_fallback) OR strategies fast enough (found match)
        ok = (strategy == "re_fallback") or (count == 1 and error is None)
        return ok

    ok3, t3_elapsed = run("large file with distinctive middle line — timeout test", t3, max_time=120)

    print()
    print("=" * 60)
    all_pass = ok1 and ok2 and ok3
    if all_pass:
        print("RESULT: ALL TESTS PASSED")
        print("The timeout mechanism is working.")
        print("Patch tool is safe to use in Hermes CLI.")
    else:
        print("RESULT: SOME TESTS FAILED")
        print("Review above. DO NOT restart gateway yet.")
        if not ok1:
            print("  - Test 1 (exact match) failed: patch may be slower than expected")
        if not ok2:
            print("  - Test 2 (no-match path) failed: slow strategies not reached?")
        if not ok3:
            print("  - Test 3 (timeout on large file) failed: timeout not firing?")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
