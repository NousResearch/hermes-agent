#!/usr/bin/env python3
"""Argus Test Harness - Runs detection against dummy database.

Paths configured by tests/argus/conftest.py when run via pytest.
For direct execution, ensure parent directory is in path.
"""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple

from dummy_db import get_dummy_connection, reset_dummy_database, DUMMY_DB_PATH
from argus_simulator import ArgusSimulator
import entropy as _entropy


class ArgusTestHarness:
    def __init__(self, db_path: Path = DUMMY_DB_PATH):
        self.db_path = db_path
        self.conn = get_dummy_connection(db_path)
        self.cursor = self.conn.cursor()
        self.results: List[Dict] = []
        
    def close(self):
        self.conn.close()
    
    def detect_entropy_for_session(self, session_id: str) -> List[Dict]:
        self.conn.close()
        self.conn = get_dummy_connection(self.db_path)
        self.cursor = self.conn.cursor()
        
        detections = []
        detections.extend(_entropy.detect_repeat_tool_calls(self.cursor, session_id, threshold=3))
        detections.extend(_entropy.detect_repeat_commands(self.cursor, session_id, threshold=3))
        detections.extend(_entropy.detect_stuck_loops(self.cursor, session_id))
        detections.extend(_entropy.detect_no_file_changes(self.cursor, session_id))
        detections.extend(_entropy.detect_error_cascade(self.cursor, session_id))
        
        return detections
    
    def validate_scenario(self, scenario_name: str, expected_detections: List[str]) -> Dict:
        SESSION_ID_MAP = {
            "repeat_tool_calls": "test_repeat_tools",
            "repeat_commands": "test_repeat_cmds",
            "stuck_loop": "test_stuck_loop",
            "no_file_changes": "test_no_changes",
            "error_cascade": "test_error_cascade",
            "budget_pressure": "test_budget_pressure",
            "combined": "test_combined",
        }
        
        reset_dummy_database()
        sim = ArgusSimulator(self.db_path)
        
        try:
            sim.run_scenario(scenario_name)
            session_id = SESSION_ID_MAP.get(scenario_name, f"test_{scenario_name}")
            
            actual_detections = self.detect_entropy_for_session(session_id)
            actual_types = [d["entropy_type"] for d in actual_detections]
            
            found = set(actual_types)
            expected = set(expected_detections)
            
            missed = expected - found
            extra = found - expected
            
            result = {
                "scenario": scenario_name,
                "session_id": session_id,
                "passed": len(missed) == 0,
                "expected": list(expected),
                "found": list(found),
                "missed": list(missed),
                "extra": list(extra),
                "detections": actual_detections,
            }
            
            self.results.append(result)
            return result
            
        finally:
            sim.close()
    
    def run_all_validations(self) -> List[Dict]:
        test_cases = [
            ("repeat_tool_calls", ["repeat_tool_calls"]),
            ("repeat_commands", ["repeat_commands"]),
            ("stuck_loop", ["stuck_loop"]),
            ("no_file_changes", ["no_file_changes"]),
            ("error_cascade", ["error_cascade"]),
        ]
        
        print("\n" + "=" * 60)
        print("ARGUS VALIDATION SUITE")
        print("=" * 60)
        
        for scenario, expected in test_cases:
            print(f"\n[TEST] Running: {scenario}")
            result = self.validate_scenario(scenario, expected)
            
            status = "PASS" if result["passed"] else "FAIL"
            print(f"  Status: {status}")
            print(f"  Expected: {', '.join(result['expected'])}")
            print(f"  Found: {', '.join(result['found']) if result['found'] else 'None'}")
            
            if result["missed"]:
                print(f"  MISSED: {', '.join(result['missed'])}")
            if result["extra"]:
                print(f"  EXTRA: {', '.join(result['extra'])}")
        
        return self.results
    
    def get_session_summary(self, session_id: str) -> dict:
        self.cursor.execute(
            "SELECT COUNT(*) as cnt FROM tool_calls WHERE session_id = ?",
            (session_id,)
        )
        tool_count = self.cursor.fetchone()["cnt"]
        
        self.cursor.execute(
            "SELECT COUNT(*) as cnt FROM terminal_commands WHERE session_id = ?",
            (session_id,)
        )
        cmd_count = self.cursor.fetchone()["cnt"]
        
        self.cursor.execute(
            "SELECT COUNT(*) as cnt FROM entropy_detections WHERE session_id = ?",
            (session_id,)
        )
        entropy_count = self.cursor.fetchone()["cnt"]
        
        return {
            "session_id": session_id,
            "tool_calls": tool_count,
            "terminal_commands": cmd_count,
            "entropy_detections": entropy_count,
        }
    
    def inspect_session(self, session_id: str):
        print(f"\n[INSPECT] Session: {session_id}")
        print("-" * 40)
        
        self.cursor.execute(
            "SELECT tool_name, success, error_message, file_changed "
            "FROM tool_calls WHERE session_id = ? ORDER BY timestamp",
            (session_id,)
        )
        tools = self.cursor.fetchall()
        print(f"Tool calls ({len(tools)}):")
        for t in tools:
            status = "OK" if t["success"] else "ERR"
            changed = "changed" if t["file_changed"] else "no-change"
            print(f"  [{status}] {t['tool_name']:15} ({changed})")
            if t["error_message"]:
                print(f"       -> {t['error_message']}")
        
        self.cursor.execute(
            "SELECT command, exit_code FROM terminal_commands WHERE session_id = ?",
            (session_id,)
        )
        cmds = self.cursor.fetchall()
        if cmds:
            print(f"\nTerminal commands ({len(cmds)}):")
            for c in cmds:
                print(f"  [{c['exit_code']}] {c['command']}")
        
        detections = self.detect_entropy_for_session(session_id)
        print(f"\nEntropy detections ({len(detections)}):")
        for d in detections:
            details = json.loads(d["details"]) if d["details"] else {}
            print(f"  [{d['severity']:8}] {d['entropy_type']}")
            if "count" in details:
                print(f"       -> count: {details['count']}")
            if "consecutive_errors" in details:
                print(f"       -> consecutive: {details['consecutive_errors']}")
    
    def print_summary(self):
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for r in self.results if r["passed"])
        failed = sum(1 for r in self.results if not r["passed"])
        
        print(f"Total tests: {len(self.results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        
        if failed > 0:
            print("\nFailed scenarios:")
            for r in self.results:
                if not r["passed"]:
                    print(f"  - {r['scenario']}: missed {r['missed']}")
        
        return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Argus Test Harness")
    parser.add_argument("--scenario", "-s", help="Validate specific scenario")
    parser.add_argument("--all", "-a", action="store_true", help="Run all scenario validations")
    parser.add_argument("--inspect", "-i", help="Inspect specific session ID")
    
    args = parser.parse_args()
    
    harness = ArgusTestHarness()
    
    try:
        if args.all:
            harness.run_all_validations()
            success = harness.print_summary()
            sys.exit(0 if success else 1)
        
        elif args.scenario:
            expected_map = {
                "repeat_tool_calls": ["repeat_tool_calls"],
                "repeat_commands": ["repeat_commands"],
                "stuck_loop": ["stuck_loop"],
                "no_file_changes": ["no_file_changes"],
                "error_cascade": ["error_cascade"],
            }
            expected = expected_map.get(args.scenario, [])
            result = harness.validate_scenario(args.scenario, expected)
            
            print(f"\nResult: {'PASS' if result['passed'] else 'FAIL'}")
            print(f"Expected: {result['expected']}")
            print(f"Found: {result['found']}")
            
            if result["detections"]:
                print("\nDetections:")
                for d in result["detections"]:
                    print(f"  [{d['severity']}] {d['entropy_type']}")
        
        elif args.inspect:
            harness.inspect_session(args.inspect)
        
        else:
            parser.print_help()
    
    finally:
        harness.close()


if __name__ == "__main__":
    main()
