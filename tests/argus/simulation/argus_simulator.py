#!/usr/bin/env python3
"""Argus Test Simulator - Generates dummy test data for entropy detection."""

import argparse
import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from dummy_db import get_dummy_connection, reset_dummy_database, DUMMY_DB_PATH


class ArgusSimulator:
    def __init__(self, db_path: Path = DUMMY_DB_PATH):
        self.db_path = db_path
        self.conn = get_dummy_connection(db_path)
        self.cursor = self.conn.cursor()
        
    def close(self):
        self.conn.close()
        
    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        
    def _insert_session(self, session_id: str, session_type: str = "test", 
                       task_description: str = "Test session") -> None:
        self.cursor.execute(
            """
            INSERT OR REPLACE INTO sessions 
            (session_id, session_type, task_description, status, started_at)
            VALUES (?, ?, ?, 'active', ?)
            """,
            (session_id, session_type, task_description, self._utc_now())
        )
        self.conn.commit()
    
    def _insert_tool_call(self, session_id: str, tool_name: str, 
                          tool_args: str = "{}", 
                          success: bool = True,
                          error_message: Optional[str] = None,
                          file_changed: Optional[bool] = None,
                          file_path: Optional[str] = None,
                          timestamp: Optional[str] = None) -> int:
        ts = timestamp or self._utc_now()
        
        if file_changed is None and tool_name in ("write_file", "patch"):
            file_changed = success
        
        self.cursor.execute(
            """
            INSERT INTO tool_calls 
            (session_id, tool_name, tool_args, timestamp, success, error_message, 
             file_changed, file_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, tool_name, tool_args, ts, success, error_message, 
             file_changed, file_path)
        )
        self.conn.commit()
        return self.cursor.lastrowid
    
    def _insert_terminal_command(self, session_id: str, command: str,
                                  exit_code: int = 0,
                                  timestamp: Optional[str] = None) -> None:
        ts = timestamp or self._utc_now()
        self.cursor.execute(
            """
            INSERT INTO terminal_commands (session_id, command, timestamp, exit_code)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, command, ts, exit_code)
        )
        self.conn.commit()
    
    def _insert_entropy_detection(self, session_id: str, entropy_type: str,
                                   severity: str = "warning",
                                   details: Dict = None) -> None:
        self.cursor.execute(
            """
            INSERT INTO entropy_detections 
            (session_id, entropy_type, severity, timestamp, details)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, entropy_type, severity, self._utc_now(), 
             json.dumps(details or {}))
        )
        self.conn.commit()
    
    def scenario_repeat_tool_calls(self, session_id: str = "test_repeat_tools",
                                    count: int = 5) -> None:
        self._insert_session(session_id, "test", f"Repeat tool calls test ({count}x same args)")
        
        for i in range(count):
            self._insert_tool_call(
                session_id,
                "read_file",
                json.dumps({"path": "/etc/hosts"}),
                success=True
            )
        
        print(f"[SIM] Generated {count} identical read_file calls for {session_id}")
    
    def scenario_repeat_commands(self, session_id: str = "test_repeat_cmds",
                                   count: int = 4) -> None:
        self._insert_session(session_id, "test", f"Repeat commands test ({count}x same cmd)")
        
        for i in range(count):
            self._insert_terminal_command(
                session_id,
                "ls -la /var/log",
                exit_code=0
            )
        
        print(f"[SIM] Generated {count} identical 'ls -la /var/log' commands for {session_id}")
    
    def scenario_stuck_loop(self, session_id: str = "test_stuck_loop",
                             iterations: int = 2) -> None:
        self._insert_session(session_id, "test", f"Stuck loop test (pattern x{iterations})")
        
        pattern = [
            ("read_file", json.dumps({"path": "config.py"})),
            ("write_file", json.dumps({"path": "config.py"})),
            ("patch", json.dumps({"path": "main.py"})),
        ]
        
        for _ in range(iterations):
            for tool_name, tool_args in pattern:
                self._insert_tool_call(session_id, tool_name, tool_args)
        
        print(f"[SIM] Generated stuck loop pattern {iterations}x ({len(pattern)} tools) for {session_id}")
    
    def scenario_no_file_changes(self, session_id: str = "test_no_changes",
                                  count: int = 3) -> None:
        self._insert_session(session_id, "test", f"No file changes test ({count}x ineffective writes)")
        
        for i in range(count):
            self._insert_tool_call(
                session_id,
                "write_file",
                json.dumps({"path": f"file{i}.txt", "content": "same"}),
                success=True,
                file_changed=False,
                file_path=f"file{i}.txt"
            )
        
        for i in range(count):
            self._insert_tool_call(
                session_id,
                "patch",
                json.dumps({"path": f"module{i}.py", "old_string": "x", "new_string": "x"}),
                success=True,
                file_changed=False,
                file_path=f"module{i}.py"
            )
        
        print(f"[SIM] Generated {count*2} write/patch ops with no file changes for {session_id}")
    
    def scenario_error_cascade(self, session_id: str = "test_error_cascade",
                                consecutive_errors: int = 3,
                                severity: str = "warning") -> None:
        self._insert_session(session_id, "test", f"Error cascade test ({consecutive_errors} consecutive errors)")
        
        self._insert_tool_call(session_id, "read_file", "{}", success=True)
        self._insert_tool_call(session_id, "terminal", "{}", success=True)
        
        error_tools = [
            ("write_file", "FileNotFoundError: /missing/path"),
            ("patch", "old_string not found in file"),
            ("terminal", "exit 1"),
            ("web_extract", "Connection timeout"),
            ("search_files", "Permission denied"),
        ]
        
        for i in range(consecutive_errors):
            tool, error = error_tools[i % len(error_tools)]
            self._insert_tool_call(
                session_id,
                tool,
                json.dumps({"attempt": i}),
                success=False,
                error_message=error
            )
        
        self._insert_tool_call(session_id, "read_file", "{}", success=True)
        
        print(f"[SIM] Generated {consecutive_errors} consecutive errors for {session_id}")
    
    def scenario_budget_pressure(self, session_id: str = "test_budget_pressure",
                                 iterations_used: int = 80,
                                 max_budget: int = 90,
                                 has_entropy: bool = True) -> None:
        self._insert_session(session_id, "test", f"Budget pressure test ({iterations_used}/{max_budget} iterations)")
        
        if has_entropy:
            self._insert_entropy_detection(
                session_id,
                "repeat_tool_calls",
                "warning",
                {"count": 4, "tool": "read_file"}
            )
        
        for i in range(10):
            self._insert_tool_call(
                session_id,
                "terminal",
                json.dumps({"cmd": f"cmd_{i}"}),
                success=i < 2,
                error_message="exit 1" if i >= 2 else None
            )
        
        print(f"[SIM] Generated budget pressure conditions for {session_id} "
              f"({iterations_used}/{max_budget}, has_entropy={has_entropy})")
    
    def scenario_combined_entropy(self, session_id: str = "test_combined") -> None:
        self._insert_session(session_id, "test", "Combined entropy test")
        
        for i in range(4):
            self._insert_tool_call(session_id, "search_files", 
                                  json.dumps({"pattern": "*.py"}))
        
        for i in range(4):
            self._insert_terminal_command(session_id, "git status", exit_code=0)
        
        for _ in range(2):
            self._insert_tool_call(session_id, "read_file", json.dumps({"path": "a.py"}))
            self._insert_tool_call(session_id, "write_file", json.dumps({"path": "a.py"}))
        
        self._insert_tool_call(session_id, "patch", json.dumps({"path": "x.py"}),
                              success=True, file_changed=False)
        
        for tool in ["web_search", "web_extract", "terminal"]:
            self._insert_tool_call(session_id, tool, "{}", 
                                  success=False, error_message="timeout")
        
        print(f"[SIM] Generated combined entropy patterns for {session_id}")
    
    def run_scenario(self, scenario_name: str, **kwargs) -> None:
        scenarios = {
            "repeat_tool_calls": self.scenario_repeat_tool_calls,
            "repeat_commands": self.scenario_repeat_commands,
            "stuck_loop": self.scenario_stuck_loop,
            "no_file_changes": self.scenario_no_file_changes,
            "error_cascade": self.scenario_error_cascade,
            "budget_pressure": self.scenario_budget_pressure,
            "combined": self.scenario_combined_entropy,
        }
        
        if scenario_name not in scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}. "
                           f"Available: {', '.join(scenarios.keys())}")
        
        scenarios[scenario_name](**kwargs)
    
    def run_all_scenarios(self) -> List[str]:
        scenarios = [
            ("repeat_tool_calls", {"session_id": "sim_repeat_tools", "count": 5}),
            ("repeat_commands", {"session_id": "sim_repeat_cmds", "count": 4}),
            ("stuck_loop", {"session_id": "sim_stuck_loop", "iterations": 2}),
            ("no_file_changes", {"session_id": "sim_no_changes", "count": 3}),
            ("error_cascade", {"session_id": "sim_error_cascade", "consecutive_errors": 4}),
            ("budget_pressure", {"session_id": "sim_budget", "iterations_used": 85}),
            ("combined", {"session_id": "sim_combined"}),
        ]
        
        session_ids = []
        for name, kwargs in scenarios:
            self.run_scenario(name, **kwargs)
            session_ids.append(kwargs["session_id"])
        
        return session_ids
    
    def get_session_summary(self, session_id: str) -> Dict:
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


def list_scenarios():
    print("\nAvailable test scenarios:")
    print("-" * 40)
    scenarios = {
        "repeat_tool_calls": "Same tool called 3+ times with identical args",
        "repeat_commands": "Same terminal command executed 3+ times",
        "stuck_loop": "Repeating sequence of tool calls (A,B,C,A,B,C)",
        "no_file_changes": "write_file/patch with file_changed=FALSE",
        "error_cascade": "3+ consecutive tool failures",
        "budget_pressure": "High iteration burn rate with entropy",
        "combined": "Multiple entropy types in one session",
        "all": "Run all scenarios sequentially",
    }
    for name, desc in scenarios.items():
        print(f"  {name:20} - {desc}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Argus Test Simulator")
    parser.add_argument("--scenario", "-s", help="Scenario to run (use 'all' for all scenarios)")
    parser.add_argument("--list", "-l", action="store_true", help="List available scenarios")
    parser.add_argument("--reset", "-r", action="store_true", help="Reset database before running")
    parser.add_argument("--summary", action="store_true", help="Print summary of generated data")
    
    args = parser.parse_args()
    
    if args.list:
        list_scenarios()
        return
    
    if not args.scenario:
        parser.print_help()
        list_scenarios()
        return
    
    if args.reset:
        reset_dummy_database()
        print("[SIM] Database reset")
    
    sim = ArgusSimulator()
    
    try:
        if args.scenario == "all":
            session_ids = sim.run_all_scenarios()
            print(f"\n[SIM] Generated data for {len(session_ids)} sessions")
            
            if args.summary:
                print("\nSession summaries:")
                for sid in session_ids:
                    summary = sim.get_session_summary(sid)
                    print(f"  {summary['session_id']}: "
                          f"{summary['tool_calls']} tools, "
                          f"{summary['terminal_commands']} cmds, "
                          f"{summary['entropy_detections']} entropy")
        else:
            sim.run_scenario(args.scenario)
            
            if args.summary:
                SESSION_ID_MAP = {
                    "repeat_tool_calls": "test_repeat_tools",
                    "repeat_commands": "test_repeat_cmds",
                    "stuck_loop": "test_stuck_loop",
                    "no_file_changes": "test_no_changes",
                    "error_cascade": "test_error_cascade",
                    "budget_pressure": "test_budget_pressure",
                    "combined": "test_combined",
                }
                session_id = SESSION_ID_MAP.get(args.scenario, f"test_{args.scenario}")
                summary = sim.get_session_summary(session_id)
                print(f"\nSummary for {session_id}:")
                print(f"  Tool calls: {summary['tool_calls']}")
                print(f"  Terminal commands: {summary['terminal_commands']}")
                print(f"  Entropy detections: {summary['entropy_detections']}")
    finally:
        sim.close()
    
    print(f"\n[SIM] Data written to: {DUMMY_DB_PATH}")


if __name__ == "__main__":
    main()
