#!/usr/bin/env python3
"""Argus Edge Case Test Suite - Threshold, time window, and malformed data tests."""

import json
import random
import sqlite3
import string
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

from dummy_db import get_dummy_connection, reset_dummy_database, DUMMY_DB_PATH
from argus_simulator import ArgusSimulator
from argus_test_harness import ArgusTestHarness

import entropy as _entropy


class ThresholdBoundaryTests:
    def __init__(self, db_path: Path = DUMMY_DB_PATH):
        self.db_path = db_path
        self.results: List[Dict] = []
        
    def test_repeat_tool_calls_thresholds(self) -> List[Dict]:
        print("\n[THRESHOLD] repeat_tool_calls boundaries...")
        results = []
        
        for count in [1, 2, 3, 4, 5]:
            reset_dummy_database()
            sim = ArgusSimulator(self.db_path)
            
            session_id = f"threshold_repeat_{count}"
            sim._insert_session(session_id, "test", f"{count} identical calls")
            
            for i in range(count):
                sim._insert_tool_call(
                    session_id, "read_file", 
                    json.dumps({"path": "/etc/hosts"})
                )
            
            sim.close()
            
            harness = ArgusTestHarness(self.db_path)
            detections = harness.detect_entropy_for_session(session_id)
            harness.close()
            
            if count < 3:
                expected_severity = None
            elif count < 5:
                expected_severity = "warning"
            else:
                expected_severity = "critical"
            
            actual = detections[0] if detections else None
            actual_severity = actual["severity"] if actual else None
            passed = actual_severity == expected_severity
            
            result = {
                "test": f"repeat_tool_calls_{count}",
                "count": count,
                "expected": expected_severity,
                "actual": actual_severity,
                "passed": passed,
            }
            results.append(result)
            status = "PASS" if passed else "FAIL"
            print(f"  {count} calls: {actual_severity or 'none':10} (expected {expected_severity or 'none':10}) - {status}")
            
        self.results.extend(results)
        return results
    
    def test_repeat_commands_thresholds(self) -> List[Dict]:
        print("\n[THRESHOLD] repeat_commands boundaries...")
        results = []
        
        for count in [1, 2, 3, 4, 5]:
            reset_dummy_database()
            sim = ArgusSimulator(self.db_path)
            
            session_id = f"threshold_cmds_{count}"
            sim._insert_session(session_id, "test", f"{count} identical commands")
            
            for i in range(count):
                sim._insert_terminal_command(session_id, "ls -la", exit_code=0)
            
            sim.close()
            
            harness = ArgusTestHarness(self.db_path)
            detections = harness.detect_entropy_for_session(session_id)
            harness.close()
            
            if count < 3:
                expected = None
            elif count < 5:
                expected = "warning"
            else:
                expected = "critical"
            
            actual = detections[0]["severity"] if detections else None
            passed = actual == expected
            
            result = {
                "test": f"repeat_commands_{count}",
                "count": count,
                "expected": expected,
                "actual": actual,
                "passed": passed,
            }
            results.append(result)
            status = "PASS" if passed else "FAIL"
            print(f"  {count} cmds: {actual or 'none':10} (expected {expected or 'none':10}) - {status}")
            
        self.results.extend(results)
        return results
    
    def test_error_cascade_thresholds(self) -> List[Dict]:
        print("\n[THRESHOLD] error_cascade boundaries...")
        results = []
        
        for count in [1, 2, 3, 4, 5]:
            reset_dummy_database()
            sim = ArgusSimulator(self.db_path)
            
            session_id = f"threshold_cascade_{count}"
            sim._insert_session(session_id, "test", f"{count} consecutive errors")
            
            sim._insert_tool_call(session_id, "read_file", "{}", success=True)
            
            for i in range(count):
                sim._insert_tool_call(
                    session_id, "terminal", 
                    json.dumps({"cmd": f"fail_{i}"}),
                    success=False, error_message="exit 1"
                )
            
            sim._insert_tool_call(session_id, "read_file", "{}", success=True)
            
            sim.close()
            
            harness = ArgusTestHarness(self.db_path)
            detections = harness.detect_entropy_for_session(session_id)
            harness.close()
            
            cascade = [d for d in detections if d["entropy_type"] == "error_cascade"]
            
            if count < 3:
                expected = None
            elif count < 5:
                expected = "warning"
            else:
                expected = "critical"
            
            actual = cascade[0]["severity"] if cascade else None
            passed = actual == expected
            
            result = {
                "test": f"error_cascade_{count}",
                "count": count,
                "expected": expected,
                "actual": actual,
                "passed": passed,
            }
            results.append(result)
            status = "PASS" if passed else "FAIL"
            print(f"  {count} errors: {actual or 'none':10} (expected {expected or 'none':10}) - {status}")
            
        self.results.extend(results)
        return results
    
    def test_stuck_loop_thresholds(self) -> List[Dict]:
        print("\n[THRESHOLD] stuck_loop pattern boundaries...")
        results = []
        
        for pattern_len in [2, 3, 4]:
            for iterations in [1, 2, 3]:
                reset_dummy_database()
                sim = ArgusSimulator(self.db_path)
                
                session_id = f"threshold_loop_{pattern_len}x{iterations}"
                sim._insert_session(session_id, "test", f"Pattern {pattern_len} x {iterations}")
                
                pattern = [(f"tool_{i}", "{}") for i in range(pattern_len)]
                
                for _ in range(iterations):
                    for tool_name, args in pattern:
                        sim._insert_tool_call(session_id, tool_name, args)
                
                sim.close()
                
                harness = ArgusTestHarness(self.db_path)
                detections = harness.detect_entropy_for_session(session_id)
                harness.close()
                
                total_tools = pattern_len * iterations
                should_detect = iterations >= 2 and total_tools >= 6 and pattern_len <= 3
                
                loop_dets = [d for d in detections if d["entropy_type"] == "stuck_loop"]
                detected = len(loop_dets) > 0
                passed = detected == should_detect
                
                result = {
                    "test": f"stuck_loop_{pattern_len}x{iterations}",
                    "pattern_len": pattern_len,
                    "iterations": iterations,
                    "total_tools": total_tools,
                    "should_detect": should_detect,
                    "detected": detected,
                    "passed": passed,
                }
                results.append(result)
                status = "PASS" if passed else "FAIL"
                det_str = "YES" if detected else "NO"
                print(f"  Pattern {pattern_len} x {iterations} (total={total_tools}): detected={det_str} (expected {should_detect}) - {status}")
        
        self.results.extend(results)
        return results
    
    def run_all(self) -> List[Dict]:
        print("\n" + "=" * 60)
        print("THRESHOLD BOUNDARY TEST SUITE")
        print("=" * 60)
        
        self.test_repeat_tool_calls_thresholds()
        self.test_repeat_commands_thresholds()
        self.test_error_cascade_thresholds()
        self.test_stuck_loop_thresholds()
        
        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)
        print(f"\n[SUMMARY] Threshold tests: {passed}/{total} passed")
        
        return self.results


class TimeBoundaryTests:
    def __init__(self, db_path: Path = DUMMY_DB_PATH):
        self.db_path = db_path
        self.results: List[Dict] = []
    
    def _make_timestamp(self, minutes_ago: float) -> str:
        dt = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    def test_repeat_tool_calls_windows(self) -> List[Dict]:
        print("\n[TIME] repeat_tool_calls window boundaries...")
        results = []
        
        test_cases = [
            ([0, 1, 2], True, "All within 10min"),
            ([0, 4, 9], True, "Edge case: 9min apart"),
            ([0, 5, 10], False, "Edge case: exactly 10min (3rd at boundary)"),
            ([0, 5, 11], False, "3rd outside window by 1min"),
            ([0, 4, 9, 12], True, "First 3 detect, 4th outside"),
            ([15, 16, 17], False, "All outside 10min window"),
        ]
        
        for minutes_list, should_detect, desc in test_cases:
            reset_dummy_database()
            sim = ArgusSimulator(self.db_path)
            
            session_id = f"time_repeat_{desc.replace(' ', '_').replace(':', '')}"
            sim._insert_session(session_id, "test", desc)
            
            for mins in minutes_list:
                ts = self._make_timestamp(mins)
                sim._insert_tool_call(
                    session_id, "read_file", 
                    json.dumps({"path": "/test"}),
                    timestamp=ts
                )
            
            sim.close()
            
            harness = ArgusTestHarness(self.db_path)
            detections = harness.detect_entropy_for_session(session_id)
            harness.close()
            
            detected = any(d["entropy_type"] == "repeat_tool_calls" for d in detections)
            passed = detected == should_detect
            
            result = {
                "test": "repeat_time_window",
                "description": desc,
                "minutes": minutes_list,
                "should_detect": should_detect,
                "detected": detected,
                "passed": passed,
            }
            results.append(result)
            status = "PASS" if passed else "FAIL"
            print(f"  {desc:30} -> detected={detected} (expected {should_detect}) - {status}")
        
        self.results.extend(results)
        return results
    
    def test_mixed_timestamps(self) -> List[Dict]:
        print("\n[TIME] Mixed timestamp handling...")
        results = []
        
        reset_dummy_database()
        sim = ArgusSimulator(self.db_path)
        
        session_id = "time_mixed"
        sim._insert_session(session_id, "test", "Mixed timestamps")
        
        conn = get_dummy_connection(self.db_path)
        cursor = conn.cursor()
        
        test_cases = [
            ("NULL timestamp", None),
            ("Future timestamp", (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")),
            ("Ancient timestamp", (datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%d %H:%M:%S")),
            ("Valid now", self._make_timestamp(0)),
        ]
        
        for desc, ts in test_cases:
            cursor.execute(
                """
                INSERT INTO tool_calls (session_id, tool_name, tool_args, timestamp)
                VALUES (?, 'read_file', '{}', ?)
                """,
                (session_id, ts)
            )
        
        conn.commit()
        conn.close()
        
        harness = ArgusTestHarness(self.db_path)
        try:
            detections = harness.detect_entropy_for_session(session_id)
            result = {
                "test": "mixed_timestamps",
                "description": "NULL, future, ancient timestamps",
                "error": None,
                "passed": True,
            }
            print(f"  Mixed timestamps: handled without error - PASS")
        except Exception as e:
            result = {
                "test": "mixed_timestamps",
                "description": "NULL, future, ancient timestamps",
                "error": str(e),
                "passed": False,
            }
            print(f"  Mixed timestamps: ERROR - {e} - FAIL")
        harness.close()
        sim.close()
        
        self.results.append(result)
        return results
    
    def run_all(self) -> List[Dict]:
        print("\n" + "=" * 60)
        print("TIME BOUNDARY TEST SUITE")
        print("=" * 60)
        
        self.test_repeat_tool_calls_windows()
        self.test_mixed_timestamps()
        
        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)
        print(f"\n[SUMMARY] Time boundary tests: {passed}/{total} passed")
        
        return self.results


class MalformedDataTests:
    def __init__(self, db_path: Path = DUMMY_DB_PATH):
        self.db_path = db_path
        self.results: List[Dict] = []
    
    def test_invalid_json(self) -> List[Dict]:
        print("\n[MALFORMED] Invalid JSON handling...")
        results = []
        
        invalid_jsons = [
            "{not json",
            "{\"unclosed: \"string}",
            "'single quotes'",
            "",
            "null",
            "undefined",
            "{}",
        ]
        
        for i, bad_json in enumerate(invalid_jsons):
            reset_dummy_database()
            sim = ArgusSimulator(self.db_path)
            
            session_id = f"malformed_json_{i}"
            sim._insert_session(session_id, "test", f"Invalid: {bad_json[:30]}...")
            
            conn = get_dummy_connection(self.db_path)
            cursor = conn.cursor()
            
            for _ in range(3):
                cursor.execute(
                    """
                    INSERT INTO tool_calls (session_id, tool_name, tool_args, timestamp)
                    VALUES (?, 'read_file', ?, datetime('now'))
                    """,
                    (session_id, bad_json)
                )
            
            conn.commit()
            conn.close()
            
            harness = ArgusTestHarness(self.db_path)
            try:
                detections = harness.detect_entropy_for_session(session_id)
                result = {
                    "test": f"invalid_json_{i}",
                    "input": bad_json[:50],
                    "error": None,
                    "crashed": False,
                    "passed": True,
                }
                print(f"  JSON '{bad_json[:30]:30}': handled - PASS")
            except Exception as e:
                result = {
                    "test": f"invalid_json_{i}",
                    "input": bad_json[:50],
                    "error": str(e),
                    "crashed": True,
                    "passed": False,
                }
                print(f"  JSON '{bad_json[:30]:30}': CRASH - {e} - FAIL")
            
            harness.close()
            sim.close()
            results.append(result)
        
        self.results.extend(results)
        return results
    
    def test_sql_injection(self) -> List[Dict]:
        print("\n[MALFORMED] SQL injection attempts...")
        results = []
        
        injection_attempts = [
            "'; DROP TABLE sessions; --",
            "1' OR '1'='1",
            "\"; DELETE FROM tool_calls; --",
            "'; UPDATE sessions SET status='hacked'; --",
            "' UNION SELECT * FROM sessions --",
        ]
        
        for i, payload in enumerate(injection_attempts):
            reset_dummy_database()
            sim = ArgusSimulator(self.db_path)
            
            session_id = f"test_sqli_{i}"
            sim._insert_session(session_id, "test", f"SQLi test {i}")
            
            conn = get_dummy_connection(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                """
                INSERT INTO tool_calls (session_id, tool_name, tool_args, error_message)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, "terminal", payload, payload)
            )
            
            conn.commit()
            
            cursor.execute("SELECT COUNT(*) as cnt FROM sessions")
            session_count = cursor.fetchone()["cnt"]
            
            cursor.execute("SELECT COUNT(*) as cnt FROM tool_calls")
            tool_count = cursor.fetchone()["cnt"]
            
            conn.close()
            
            intact = session_count >= 1 and tool_count >= 1
            
            result = {
                "test": f"sqli_{i}",
                "payload": payload[:40],
                "tables_intact": intact,
                "passed": intact,
            }
            status = "PASS" if intact else "FAIL"
            print(f"  Payload '{payload[:35]:35}': tables intact={intact} - {status}")
            
            results.append(result)
            sim.close()
        
        self.results.extend(results)
        return results
    
    def test_unicode_edge_cases(self) -> List[Dict]:
        print("\n[MALFORMED] Unicode edge cases...")
        results = []
        
        unicode_cases = [
            ("emoji_path", "{\"path\": \"/tmp/🗑️🔥💣\"}"),
            ("cjk_session", "测试会话_中文"),
            ("arabic_error", "خطأ في الملف المحدد"),
            ("russian_tool", "инструмент_чтения"),
            ("zalgo_text", "t̷͓̖͈͔͛̈́͛͑̈́͆͠e̶̢͎͇̪̻͉̒͑͐̍͑̆͝s̶̨̛̱̜̠̭̮̈́̍͐̕͝t̷̞̭̘̺̓̉̔̒̈́͝"),
            ("null_bytes", "path\x00file"),
            ("newlines", "line1\nline2\r\nline3"),
        ]
        
        for name, value in unicode_cases:
            reset_dummy_database()
            sim = ArgusSimulator(self.db_path)
            
            session_id = f"unicode_{name}"
            sim._insert_session(session_id, "test", f"Unicode: {name}")
            
            conn = get_dummy_connection(self.db_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute(
                    """
                    INSERT INTO tool_calls (session_id, tool_name, tool_args, error_message)
                    VALUES (?, 'test_tool', ?, ?)
                    """,
                    (session_id, value, value)
                )
                conn.commit()
                
                cursor.execute(
                    "SELECT tool_args, error_message FROM tool_calls WHERE session_id=?",
                    (session_id,)
                )
                row = cursor.fetchone()
                read_back = row["tool_args"] if row else None
                
                result = {
                    "test": f"unicode_{name}",
                    "value": value[:40],
                    "stored": read_back is not None,
                    "passed": read_back is not None,
                }
                status = "PASS" if read_back else "FAIL"
                print(f"  {name:20}: stored={read_back is not None} - {status}")
                
            except Exception as e:
                result = {
                    "test": f"unicode_{name}",
                    "value": value[:40],
                    "error": str(e),
                    "passed": False,
                }
                print(f"  {name:20}: ERROR - {e} - FAIL")
            
            conn.close()
            sim.close()
            results.append(result)
        
        self.results.extend(results)
        return results
    
    def test_null_empty_fields(self) -> List[Dict]:
        print("\n[MALFORMED] NULL and empty field handling...")
        results = []
        
        reset_dummy_database()
        sim = ArgusSimulator(self.db_path)
        session_id = "null_empty_test"
        sim._insert_session(session_id, "test", "NULL/empty test")
        
        conn = get_dummy_connection(self.db_path)
        cursor = conn.cursor()
        
        null_cases = [
            ("NULL tool_name", {"tool_name": None}, True),
            ("NULL session_id", {"session_id": None}, True),
            ("NULL timestamp", {"timestamp": None}, False),
            ("NULL success", {"success": None}, False),
            ("Empty tool_args", {"tool_args": ""}, False),
        ]
        
        for desc, overrides, should_fail in null_cases:
            try:
                fields = ["session_id", "tool_name", "tool_args", "timestamp", "success"]
                values = [
                    overrides.get("session_id", session_id),
                    overrides.get("tool_name", "test_tool"),
                    overrides.get("tool_args", "{}"),
                    overrides.get("timestamp", "datetime('now')"),
                    overrides.get("success", True),
                ]
                
                placeholders = ",".join(["?" if v is not None else "NULL" for v in values])
                actual_values = [v for v in values if v is not None]
                
                cursor.execute(
                    f"INSERT INTO tool_calls ({','.join(fields)}) VALUES ({placeholders})",
                    actual_values
                )
                conn.commit()
                
                actually_failed = False
                
            except Exception as e:
                actually_failed = True
            
            passed = actually_failed == should_fail
            status = "PASS" if passed else "FAIL"
            print(f"  {desc:25}: failed={actually_failed} (expected fail={should_fail}) - {status}")
            
            results.append({
                "test": desc,
                "should_fail": should_fail,
                "actually_failed": actually_failed,
                "passed": passed,
            })
        
        conn.close()
        sim.close()
        self.results.extend(results)
        return results
    
    def run_all(self) -> List[Dict]:
        print("\n" + "=" * 60)
        print("MALFORMED DATA TEST SUITE")
        print("=" * 60)
        
        self.test_invalid_json()
        self.test_sql_injection()
        self.test_unicode_edge_cases()
        self.test_null_empty_fields()
        
        passed = sum(1 for r in self.results if r.get("passed", False))
        total = len(self.results)
        print(f"\n[SUMMARY] Malformed data tests: {passed}/{total} passed")
        
        return self.results


def run_all_edge_case_tests():
    print("\n" + "=" * 70)
    print("ARGUS COMPLETE EDGE CASE TEST SUITE")
    print("=" * 70)
    
    threshold = ThresholdBoundaryTests()
    threshold.run_all()
    
    time_tests = TimeBoundaryTests()
    time_tests.run_all()
    
    malformed = MalformedDataTests()
    malformed.run_all()
    
    all_results = threshold.results + time_tests.results + malformed.results
    total_passed = sum(1 for r in all_results if r.get("passed", False))
    total_tests = len(all_results)
    
    print("\n" + "=" * 70)
    print(f"FINAL SUMMARY: {total_passed}/{total_tests} edge case tests passed")
    print("=" * 70)
    
    failures = [r for r in all_results if not r.get("passed", False)]
    if failures:
        print("\nFAILED TESTS:")
        for f in failures:
            test_name = f.get("test", "unknown")
            print(f"  - {test_name}")
    
    return all_results


if __name__ == "__main__":
    run_all_edge_case_tests()
