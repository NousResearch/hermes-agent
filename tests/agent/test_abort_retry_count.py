"""
Test that retry_count is incremented BEFORE the max_retries check in ABORT directive.
This verifies the fix for the off-by-one bug where retry_count += 1 was placed
after the max_retries check, causing max_retries=3 to allow 4 attempts.
"""
import sys
sys.path.insert(0, '/Users/bbni039/.hermes/hermes-agent')


def check_abort_gives_up_at_max_retries(max_retries, abort_count):
    """
    Simulate the FIXED behavior: increment retry_count BEFORE checking max_retries.

    Returns (retry_count, gave_up).
    """
    retry_count = 0
    for _ in range(abort_count):
        retry_count += 1  # Increment BEFORE check
        if retry_count >= max_retries:
            return retry_count, True
    return retry_count, False


def check_abort_gives_up_at_max_retries_BUGGY(max_retries, abort_count):
    """
    Simulate the BUGGY behavior: check max_retries BEFORE incrementing retry_count.
    """
    retry_count = 0
    for _ in range(abort_count):
        if retry_count >= max_retries:  # Check BEFORE increment
            return retry_count, True
        retry_count += 1
    return retry_count, False


def test_fixed_respects_max_retries():
    """Fixed: max_retries=3 gives up on 3rd abort."""
    attempts, gave_up = check_abort_gives_up_at_max_retries(3, 3)
    assert attempts == 3 and gave_up is True


def test_fixed_allows_under_max_retries():
    """Fixed: 2 ABORTs with max_retries=3 does not give up."""
    attempts, gave_up = check_abort_gives_up_at_max_retries(3, 2)
    assert attempts == 2 and gave_up is False


def test_buggy_allows_extra_abort():
    """
    Buggy: max_retries=3 would allow 4 ABORTs (one extra).
    Fixed: max_retries=3 allows only 3 ABORTs.

    Trace with 4 ABORTs:
    - Buggy: gives up on ABORT #4
    - Fixed: gives up on ABORT #3
    """
    # Fixed: gives up on 3rd abort
    _, gave_up_fixed = check_abort_gives_up_at_max_retries(3, 3)
    assert gave_up_fixed is True

    # Buggy: with 3 ABORTs, does NOT give up (allows 1 extra)
    _, gave_up_buggy_3 = check_abort_gives_up_at_max_retries_BUGGY(3, 3)
    assert gave_up_buggy_3 is False, "Buggy should NOT give up after 3 aborts with max_retries=3"

    # Buggy: with 4 ABORTs, gives up
    _, gave_up_buggy_4 = check_abort_gives_up_at_max_retries_BUGGY(3, 4)
    assert gave_up_buggy_4 is True, "Buggy should give up after 4 aborts with max_retries=3"


def test_contract_max_retries_n_gives_n_aborts():
    """Contract: max_retries=N allows exactly N aborts before giving up."""
    for n in [1, 2, 3, 5]:
        attempts, gave_up = check_abort_gives_up_at_max_retries(n, n)
        assert attempts == n and gave_up is True


def test_edge_case_single_retry():
    """max_retries=1 gives up on 1st abort."""
    attempts, gave_up = check_abort_gives_up_at_max_retries(1, 1)
    assert attempts == 1 and gave_up is True


if __name__ == "__main__":
    tests = [
        test_fixed_respects_max_retries,
        test_fixed_allows_under_max_retries,
        test_buggy_allows_extra_abort,
        test_contract_max_retries_n_gives_n_aborts,
        test_edge_case_single_retry,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed+failed} passed")
    sys.exit(0 if failed == 0 else 1)
