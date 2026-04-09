#!/usr/bin/env python3
"""Argus Stress Test Suite - Volume, performance, and concurrency testing."""

import json
import time
import statistics
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from dummy_db import get_dummy_connection, reset_dummy_database, DUMMY_DB_PATH
from argus_simulator import ArgusSimulator
from argus_test_harness import ArgusTestHarness

import entropy as _entropy


class PerformanceMetrics:
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.counts: Dict[str, int] = {}
        self.errors: Dict[str, List[str]] = {}
    
    def record(self, operation: str, duration_ms: float, error: Optional[str] = None):
        if operation not in self.timings:
            self.timings[operation] = []
            self.counts[operation] = 0
        
        self.timings[operation].append(duration_ms)
        self.counts[operation] += 1
        
        if error:
            if operation not in self.errors:
                self.errors[operation] = []
            self.errors[operation].append(error)
    
    def summary(self) -> Dict:
        summary = {}
        for op, times in self.timings.items():
            if times:
                summary[op] = {
                    "count": self.counts[op],
                    "total_ms": sum(times),
                    "avg_ms": statistics.mean(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "median_ms": statistics.median(times),
                    "errors": len(self.errors.get(op, [])),
                }
        return summary
    
    def print_report(self):
        print("\n" + "=" * 70)
        print("PERFORMANCE METRICS REPORT")
        print("=" * 70)
        print(f"{'Operation':<30} {'Count':>8} {'Avg(ms)':>10} {'Max(ms)':>10} {'Errors':>8}")
        print("-" * 70)
        
        summary = self.summary()
        for op, stats in sorted(summary.items()):
            print(f"{op:<30} {stats['count']:>8} {stats['avg_ms']:>10.2f} {stats['max_ms']:>10.2f} {stats['errors']:>8}")
        
        print("=" * 70)


class VolumeStressTests:
    def __init__(self, db_path: Path = DUMMY_DB_PATH):
        self.db_path = db_path
        self.metrics = PerformanceMetrics()
        self.results: List[Dict] = []
    
    def test_sessions_100(self) -> Dict:
        print("\n[VOLUME] Testing 100 sessions...")
        
        reset_dummy_database()
        sim = ArgusSimulator(self.db_path)
        
        session_ids = []
        for i in range(100):
            sid = f"volume_100_{i:03d}"
            session_ids.append(sid)
            sim._insert_session(sid, "volume_test", f"Session {i}")
            
            for j in range(10):
                sim._insert_tool_call(sid, "read_file", f'{{"file": "{j}"}}')
        
        sim.close()
        
        harness = ArgusTestHarness(self.db_path)
        
        start = time.perf_counter()
        for sid in session_ids:
            t0 = time.perf_counter()
            detections = harness.detect_entropy_for_session(sid)
            t1 = time.perf_counter()
            self.metrics.record("detect_single_session", (t1 - t0) * 1000)
        
        total_time = (time.perf_counter() - start) * 1000
        harness.close()
        
        result = {
            "test": "sessions_100",
            "sessions": 100,
            "total_tools": 1000,
            "total_time_ms": total_time,
            "avg_per_session_ms": total_time / 100,
            "passed": total_time < 10000,
        }
        
        print(f"  100 sessions, 1000 tool calls: {total_time:.1f}ms total, {total_time/100:.2f}ms avg")
        self.results.append(result)
        return result
    
    def test_sessions_1000(self) -> Dict:
        print("\n[VOLUME] Testing 1000 sessions (100 with entropy)...")
        
        reset_dummy_database()
        sim = ArgusSimulator(self.db_path)
        
        for i in range(1000):
            sid = f"volume_1000_{i:04d}"
            sim._insert_session(sid, "volume_test", f"Session {i}")
            
            if i % 10 == 0:
                for _ in range(5):
                    sim._insert_tool_call(sid, "read_file", '{"path": "/test"}')
            else:
                for _ in range(2):
                    sim._insert_tool_call(sid, "terminal", '{"cmd": "ls"}')
        
        sim.close()
        
        try:
            import psutil
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024
            has_psutil = True
        except ImportError:
            mem_before = 0
            has_psutil = False
        
        harness = ArgusTestHarness(self.db_path)
        
        start = time.perf_counter()
        import random
        sample_sids = [f"volume_1000_{random.randint(0, 999):04d}" for _ in range(100)]
        
        for sid in sample_sids:
            t0 = time.perf_counter()
            detections = harness.detect_entropy_for_session(sid)
            t1 = time.perf_counter()
            self.metrics.record("detect_sample_1000", (t1 - t0) * 1000)
        
        total_time = (time.perf_counter() - start) * 1000
        harness.close()
        
        if has_psutil:
            mem_after = process.memory_info().rss / 1024 / 1024
            mem_used = mem_after - mem_before
        else:
            mem_used = -1
        
        result = {
            "test": "sessions_1000",
            "sessions": 1000,
            "total_tools": 100 * 5 + 900 * 2,
            "memory_mb": mem_used,
            "sample_time_ms": total_time,
            "passed": True,
        }
        
        print(f"  1000 sessions: {mem_used:.1f}MB memory, {total_time:.1f}ms for 100 samples")
        self.results.append(result)
        return result
    
    def test_tool_calls_10k(self) -> Dict:
        print("\n[VOLUME] Testing 10,000 tool calls in one session...")
        
        reset_dummy_database()
        sim = ArgusSimulator(self.db_path)
        
        session_id = "volume_10k_tools"
        sim._insert_session(session_id, "stress", "10k tool calls")
        
        batch_size = 1000
        for batch in range(10):
            conn = get_dummy_connection(self.db_path)
            cursor = conn.cursor()
            
            calls = []
            for i in range(batch_size):
                idx = batch * batch_size + i
                calls.append((
                    session_id,
                    "read_file" if idx % 2 == 0 else "write_file",
                    f'{{"idx": {idx}}}',
                    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                ))
            
            cursor.executemany(
                """
                INSERT INTO tool_calls (session_id, tool_name, tool_args, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                calls
            )
            conn.commit()
            conn.close()
        
        sim.close()
        
        harness = ArgusTestHarness(self.db_path)
        
        start = time.perf_counter()
        detections = harness.detect_entropy_for_session(session_id)
        elapsed = (time.perf_counter() - start) * 1000
        
        harness.close()
        
        result = {
            "test": "tool_calls_10k",
            "session_id": session_id,
            "tool_calls": 10000,
            "detection_time_ms": elapsed,
            "detections_found": len(detections),
            "passed": elapsed < 1000,
        }
        
        print(f"  10k tool calls: {elapsed:.1f}ms detection, {len(detections)} patterns found")
        self.results.append(result)
        return result
    
    def test_concurrent_detection(self) -> Dict:
        print("\n[VOLUME] Testing concurrent detection (10 threads)...")
        
        reset_dummy_database()
        sim = ArgusSimulator(self.db_path)
        
        session_ids = []
        for i in range(50):
            sid = f"concurrent_{i:03d}"
            session_ids.append(sid)
            sim._insert_session(sid, "concurrent", f"Session {i}")
            
            for _ in range(5):
                sim._insert_tool_call(sid, "read_file", '{"path": "/test"}')
        
        sim.close()
        
        def detect_session(sid):
            conn = get_dummy_connection(self.db_path)
            cursor = conn.cursor()
            t0 = time.perf_counter()
            
            detections = []
            detections.extend(_entropy.detect_repeat_tool_calls(cursor, sid, 3))
            
            t1 = time.perf_counter()
            conn.close()
            return (t1 - t0) * 1000, len(detections)
        
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(detect_session, sid) for sid in session_ids]
            results = [f.result() for f in as_completed(futures)]
        
        total_time = (time.perf_counter() - start) * 1000
        
        times = [r[0] for r in results]
        detect_counts = [r[1] for r in results]
        
        result = {
            "test": "concurrent_detection",
            "threads": 10,
            "sessions": 50,
            "total_time_ms": total_time,
            "avg_time_ms": statistics.mean(times),
            "max_time_ms": max(times),
            "total_detections": sum(detect_counts),
            "passed": total_time < 5000,
        }
        
        print(f"  50 sessions, 10 threads: {total_time:.1f}ms total, {result['avg_time_ms']:.2f}ms avg")
        self.results.append(result)
        return result
    
    def test_race_condition(self) -> Dict:
        print("\n[VOLUME] Testing race conditions...")
        
        reset_dummy_database()
        sim = ArgusSimulator(self.db_path)
        
        session_id = "race_test"
        sim._insert_session(session_id, "race", "Race condition test")
        
        def writer():
            conn = get_dummy_connection(self.db_path)
            cursor = conn.cursor()
            for i in range(50):
                cursor.execute(
                    """
                    INSERT INTO tool_calls (session_id, tool_name, tool_args, timestamp)
                    VALUES (?, 'read_file', ?, datetime('now'))
                    """,
                    (session_id, f'{{"idx": {i}}}')
                )
                conn.commit()
                time.sleep(0.001)
            conn.close()
            return "writer_done"
        
        def reader():
            detections_found = 0
            for _ in range(20):
                conn = get_dummy_connection(self.db_path)
                cursor = conn.cursor()
                dets = _entropy.detect_repeat_tool_calls(cursor, session_id, 3)
                detections_found += len(dets)
                conn.close()
                time.sleep(0.005)
            return detections_found
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            writer_future = executor.submit(writer)
            reader_future = executor.submit(reader)
            
            writer_result = writer_future.result()
            reader_result = reader_future.result()
        
        sim.close()
        
        result = {
            "test": "race_condition",
            "writes": 50,
            "reads": 20,
            "detections_during_race": reader_result,
            "no_crashes": True,
            "passed": True,
        }
        
        print(f"  Race test: 50 writes, 20 reads, {reader_result} detections, no crashes")
        self.results.append(result)
        return result
    
    def run_all(self) -> List[Dict]:
        print("\n" + "=" * 70)
        print("VOLUME STRESS TEST SUITE")
        print("=" * 70)
        
        self.test_sessions_100()
        self.test_sessions_1000()
        self.test_tool_calls_10k()
        self.test_concurrent_detection()
        self.test_race_condition()
        
        self.metrics.print_report()
        
        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)
        print(f"\n[SUMMARY] Volume tests: {passed}/{total} passed")
        
        return self.results


class ParameterizedMatrixTests:
    def __init__(self, db_path: Path = DUMMY_DB_PATH):
        self.db_path = db_path
        self.results: List[Dict] = []
    
    def generate_matrix_repeat_tool_calls(self) -> List[Dict]:
        print("\n[MATRIX] repeat_tool_calls parameter matrix...")
        results = []
        
        counts = [1, 2, 3, 4, 5, 10]
        time_spreads = [0, 5, 9, 10, 11]
        arg_variations = ["identical", "slightly_different", "completely_different"]
        
        test_num = 0
        for count in counts:
            for spread in time_spreads:
                for var in arg_variations:
                    test_num += 1
                    
                    reset_dummy_database()
                    sim = ArgusSimulator(self.db_path)
                    
                    session_id = f"matrix_repeat_{test_num:04d}"
                    sim._insert_session(session_id, "matrix", 
                                       f"count={count}, spread={spread}, var={var}")
                    
                    for i in range(count):
                        if var == "identical":
                            args = '{"path": "/test"}'
                        elif var == "slightly_different":
                            args = f'{{"path": "/test", "idx": {i}}}'
                        else:
                            args = f'{{"id": {i}, "random": {hash(str(i))}}}'
                        
                        mins_ago = spread * (i / max(count - 1, 1))
                        from datetime import timezone, timedelta, datetime
                        ts = (datetime.now(timezone.utc) - timedelta(minutes=mins_ago)).strftime("%Y-%m-%d %H:%M:%S")
                        
                        sim._insert_tool_call(session_id, "read_file", args, timestamp=ts)
                    
                    sim.close()
                    
                    harness = ArgusTestHarness(self.db_path)
                    detections = harness.detect_entropy_for_session(session_id)
                    harness.close()
                    
                    detected = any(d["entropy_type"] == "repeat_tool_calls" for d in detections)
                    
                    should_detect = (count >= 3 and spread < 10 and var == "identical")
                    passed = detected == should_detect
                    
                    result = {
                        "test": f"matrix_repeat_{test_num}",
                        "count": count,
                        "time_spread": spread,
                        "variation": var,
                        "detected": detected,
                        "should_detect": should_detect,
                        "passed": passed,
                    }
                    results.append(result)
                    
                    status = "PASS" if passed else "FAIL"
                    if not passed or test_num <= 5 or test_num % 20 == 0:
                        print(f"  Test {test_num:4d}: count={count:2d}, spread={spread:2d}min, var={var:20s} -> detected={detected} (expected {should_detect}) - {status}")
        
        self.results.extend(results)
        passed_count = sum(1 for r in results if r["passed"])
        print(f"\n  Matrix: {passed_count}/{len(results)} passed ({len(results)} total combinations)")
        return results
    
    def generate_matrix_error_cascade(self) -> List[Dict]:
        print("\n[MATRIX] error_cascade parameter matrix...")
        results = []
        
        error_counts = [0, 1, 2, 3, 4, 5, 10]
        interleaved_successes = [0, 1, 2, 3]
        
        test_num = 0
        for err_count in error_counts:
            for successes in interleaved_successes:
                test_num += 1
                
                reset_dummy_database()
                sim = ArgusSimulator(self.db_path)
                
                session_id = f"matrix_cascade_{test_num:04d}"
                sim._insert_session(session_id, "matrix",
                                  f"errors={err_count}, interleaved={successes}")
                
                sequence = []
                remaining_errors = err_count
                remaining_successes = successes
                
                for i in range(err_count + successes):
                    if i % (successes + 1) == 0 and remaining_errors > 0:
                        sequence.append("error")
                        remaining_errors -= 1
                    elif remaining_successes > 0:
                        sequence.append("success")
                        remaining_successes -= 1
                
                while remaining_errors > 0:
                    sequence.append("error")
                    remaining_errors -= 1
                
                for s in sequence:
                    sim._insert_tool_call(
                        session_id, "terminal", '{}',
                        success=(s == "success"),
                        error_message=None if s == "success" else "exit 1"
                    )
                
                sim.close()
                
                harness = ArgusTestHarness(self.db_path)
                detections = harness.detect_entropy_for_session(session_id)
                harness.close()
                
                cascade = [d for d in detections if d["entropy_type"] == "error_cascade"]
                detected = len(cascade) > 0
                
                max_run = 0
                current = 0
                for s in sequence:
                    if s == "error":
                        current += 1
                        max_run = max(max_run, current)
                    else:
                        current = 0
                
                should_detect = max_run >= 3
                passed = detected == should_detect
                
                result = {
                    "test": f"matrix_cascade_{test_num}",
                    "errors": sum(1 for s in sequence if s == "error"),
                    "successes": sum(1 for s in sequence if s == "success"),
                    "max_consecutive_errors": max_run,
                    "detected": detected,
                    "should_detect": should_detect,
                    "passed": passed,
                }
                results.append(result)
                
                if not passed or test_num <= 5 or test_num % 10 == 0:
                    status = "PASS" if passed else "FAIL"
                    print(f"  Test {test_num:4d}: max_run={max_run}, detected={detected} (expected {should_detect}) - {status}")
        
        self.results.extend(results)
        passed_count = sum(1 for r in results if r["passed"])
        print(f"\n  Matrix: {passed_count}/{len(results)} passed")
        return results
    
    def run_all(self) -> List[Dict]:
        print("\n" + "=" * 70)
        print("PARAMETERIZED MATRIX TEST SUITE")
        print("=" * 70)
        
        self.generate_matrix_repeat_tool_calls()
        self.generate_matrix_error_cascade()
        
        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)
        print(f"\n[SUMMARY] Matrix tests: {passed}/{total} passed")
        
        return self.results


def run_all_tests():
    print("\n" + "=" * 80)
    print("ARGUS COMPREHENSIVE STRESS & EDGE CASE TEST SUITE")
    print("=" * 80)
    
    from argus_edge_cases import run_all_edge_case_tests
    
    edge_results = run_all_edge_case_tests()
    
    stress = VolumeStressTests()
    stress.run_all()
    stress_results = stress.results
    
    matrix = ParameterizedMatrixTests()
    matrix.run_all()
    matrix_results = matrix.results
    
    all_results = edge_results + stress_results + matrix_results
    total_passed = sum(1 for r in all_results if r.get("passed", False))
    total_tests = len(all_results)
    
    print("\n" + "=" * 80)
    print(f"FINAL SUMMARY: {total_passed}/{total_tests} total tests passed")
    print("=" * 80)
    
    print(f"\n  Edge case tests:    {sum(1 for r in edge_results if r.get('passed', False))}/{len(edge_results)}")
    print(f"  Stress tests:       {sum(1 for r in stress_results if r.get('passed', False))}/{len(stress_results)}")
    print(f"  Matrix tests:       {sum(1 for r in matrix_results if r.get('passed', False))}/{len(matrix_results)}")
    
    print("\n  Performance metrics:")
    stress.metrics.print_report()
    
    return all_results


if __name__ == "__main__":
    run_all_tests()
