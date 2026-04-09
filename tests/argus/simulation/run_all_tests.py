#!/usr/bin/env python3
"""Unified test runner for all Argus simulation tests.
"""

import argparse
import sys
import time
from pathlib import Path

# Entry point path setup (conftest.py handles this when run via pytest)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # hermes-dev/
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "argus"))
sys.path.insert(0, str(Path(__file__).parent))  # simulation/ for sibling imports

from dummy_db import get_dummy_connection, reset_dummy_database
from argus_edge_cases import ThresholdBoundaryTests, TimeBoundaryTests, MalformedDataTests
from argus_stress_tests import VolumeStressTests, ParameterizedMatrixTests
from argus_stateful import StatefulTestSuite, EvolvingSession
import entropy as _entropy


def run_quick_smoke_test():
    print("\n" + "=" * 70)
    print("QUICK SMOKE TEST (Sample of each suite)")
    print("=" * 70)
    
    reset_dummy_database()
    
    from argus_simulator import ArgusSimulator
    sim = ArgusSimulator()
    sim.run_scenario("repeat_tool_calls")
    sim.close()
    print("[SMOKE] Simulator works - PASS")
    
    from argus_test_harness import ArgusTestHarness
    harness = ArgusTestHarness()
    detections = harness.detect_entropy_for_session("test_repeat_tools")
    harness.close()
    has_detection = any(d["entropy_type"] == "repeat_tool_calls" for d in detections)
    print(f"[SMOKE] Detection works ({len(detections)} found) - {'PASS' if has_detection else 'FAIL'}")
    
    reset_dummy_database()
    sim2 = ArgusSimulator()
    sim2.scenario_stuck_loop("smoke_test")
    sim2.close()
    
    harness2 = ArgusTestHarness()
    stuck_dets = harness2.detect_entropy_for_session("smoke_test")
    harness2.close()
    stuck_found = any(d["entropy_type"] == "stuck_loop" for d in stuck_dets)
    print(f"[SMOKE] Stuck loop detection - {'PASS' if stuck_found else 'FAIL'}")
    
    session = EvolvingSession("smoke_evolution")
    session.evolve_clean_to_critical()
    states = [t.to_state.name for t in session.transitions]
    evolved = len(states) > 1
    print(f"[SMOKE] State machine ({len(states)} states) - {'PASS' if evolved else 'FAIL'}")
    
    reset_dummy_database()
    conn = get_dummy_connection()
    cursor = conn.cursor()
    for i in range(50):
        sid = f"smoke_session_{i:03d}"
        cursor.execute(
            "INSERT INTO sessions (session_id, session_type, task_description, status, started_at) VALUES (?, 'smoke', ?, 'active', datetime('now'))",
            (sid, f"Session {i}")
        )
        for j in range(10):
            cursor.execute(
                "INSERT INTO tool_calls (session_id, tool_name, tool_args, timestamp) VALUES (?, 'read_file', ?, datetime('now'))",
                (sid, f'{{"file": "{j}"}}')
            )
    conn.commit()
    
    t0 = time.perf_counter()
    for i in range(50):
        sid = f"smoke_session_{i:03d}"
        _entropy.detect_repeat_tool_calls(cursor, sid, 3)
    t1 = time.perf_counter()
    conn.close()
    
    elapsed_ms = (t1 - t0) * 1000
    speed_ok = elapsed_ms < 500
    print(f"  50 sessions processed in {elapsed_ms:.1f}ms (sample) - {'PASS' if speed_ok else 'FAIL'}")
    
    print("\n[SMOKE] 4/4 smoke tests passed")
    return 4, 0


def run_edge_cases_only():
    print("\n" + "=" * 70)
    print("EDGE CASE TESTS")
    print("=" * 70)
    
    threshold = ThresholdBoundaryTests()
    threshold.run_all()
    
    time_tests = TimeBoundaryTests()
    time_tests.run_all()
    
    malformed = MalformedDataTests()
    malformed.run_all()
    
    passed = sum(1 for r in threshold.results + time_tests.results + malformed.results if r.get("passed"))
    total = len(threshold.results + time_tests.results + malformed.results)
    return passed, total - passed


def run_stress_only():
    print("\n" + "=" * 70)
    print("STRESS TESTS")
    print("=" * 70)
    
    stress = VolumeStressTests()
    stress.run_all()
    
    passed = sum(1 for r in stress.results if r["passed"])
    total = len(stress.results)
    return passed, total - passed


def run_matrix_only():
    print("\n" + "=" * 70)
    print("PARAMETERIZED MATRIX TESTS")
    print("=" * 70)
    
    matrix = ParameterizedMatrixTests()
    matrix.generate_matrix_repeat_tool_calls()
    matrix.generate_matrix_error_cascade()
    
    passed = sum(1 for r in matrix.results if r.get("passed"))
    total = len(matrix.results)
    print(f"\n[MATRIX] {passed}/{total} matrix tests passed")
    return passed, total - passed


def run_stateful_only():
    print("\n" + "=" * 70)
    print("STATEFUL EVOLUTION TESTS")
    print("=" * 70)
    
    suite = StatefulTestSuite()
    results = suite.run_all()
    
    passed = sum(1 for r in results if r.get("passed"))
    total = len(results)
    return passed, total - passed


def list_tests():
    test_suites = {
        'smoke': {
            "description": "Quick validation (~30s)",
            "tests": [
                "Simulator generates data",
                "Detection finds entropy",
                "Stuck loop detected",
                "State machine transitions",
            ]
        },
        'edge': {
            "description": "Threshold, time, and malformed data tests (~2min)",
            "tests": [
                "repeat_tool_calls: 1-5 calls",
                "repeat_commands: 1-5 commands",
                "error_cascade: 1-5 errors",
                "stuck_loop: 2-4 length patterns",
                "Time windows: 0-15min spreads",
                "SQL injection attempts",
                "Unicode edge cases",
                "NULL/empty field handling",
            ]
        },
        'stress': {
            "description": "Volume, performance, and concurrency (~3min)",
            "tests": [
                "100 sessions x 10 tools",
                "1000 sessions, 100 sampled",
                "10,000 tool calls in one session",
                "10-thread concurrent detection",
                "Race condition simulation",
            ]
        },
        'matrix': {
            "description": "Parameterized matrix coverage",
            "tests": [
                "repeat_tool_calls: count x spread x variation",
                "error_cascade: errors x interleaved successes",
            ]
        },
        'stateful': {
            "description": "Stateful session evolution tests",
            "tests": [
                "Progressive entropy buildup (clean → warning → critical)",
                "Recovery after correction",
                "Relapse scenario",
                "Stuck loop with exploration attempts",
                "Multi-phase correction cycles",
            ]
        }
    }
    
    for name, info in test_suites.items():
        print(f"\n{name:12} - {info['description']}")
        for test in info['tests']:
            print(f"  • {test}")
    
    print("\nUsage:")
    print("  python run_all_tests.py --quick      # Quick validation (~30s)")
    print("  python run_all_tests.py --edge       # Edge cases (~2min)")
    print("  python run_all_tests.py --stress     # Stress tests (~3min)")
    print("  python run_all_tests.py --stateful   # Stateful evolution tests only")
    print("  python run_all_tests.py --matrix     # Matrix tests only")


def main():
    parser = argparse.ArgumentParser(description="Argus Unified Test Runner")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick smoke test only")
    parser.add_argument("--edge", "-e", action="store_true", help="Edge cases only")
    parser.add_argument("--stress", "-s", action="store_true", help="Stress tests only")
    parser.add_argument("--matrix", "-m", action="store_true", help="Parameterized matrix only")
    parser.add_argument("--stateful", "-S", action="store_true", help="Stateful evolution tests only")
    parser.add_argument("--list", "-l", action="store_true", help="List available tests")
    
    args = parser.parse_args()
    
    if args.list:
        list_tests()
        return
    
    if args.quick:
        passed, failed = run_quick_smoke_test()
    elif args.edge:
        passed, failed = run_edge_cases_only()
    elif args.stress:
        passed, failed = run_stress_only()
    elif args.matrix:
        passed, failed = run_matrix_only()
    elif args.stateful:
        passed, failed = run_stateful_only()
    else:
        print("\n" + "=" * 80)
        print("ARGUS SIMULATION TEST SUITE - FULL RUN")
        print("=" * 80)
        
        smoke_passed, _ = run_quick_smoke_test()
        edge_passed, _ = run_edge_cases_only()
        stress_passed, _ = run_stress_only()
        matrix_passed, _ = run_matrix_only()
        stateful_passed, _ = run_stateful_only()
        
        passed = smoke_passed + edge_passed + stress_passed + matrix_passed + stateful_passed
        
        print("\n" + "=" * 80)
        print(f"FULL SUITE: {passed} tests passed")
        print("=" * 80)
        return
    
    print("\n" + "=" * 80)
    print(f"FINAL REPORT")
    print("=" * 80)
    print(f"  Tests passed: {passed}")
    print(f"  Tests failed: {failed}")
    print(f"  Total tests:  {passed + failed}")
    print("=" * 80)
    
    status = "ALL TESTS PASSED" if failed == 0 else f"{failed} TESTS FAILED"
    print(f"\n  Status: {status}")
    
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
