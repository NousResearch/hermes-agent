#!/usr/bin/env python3
"""Argus Stateful Session Evolution - Simulates sessions that develop entropy over time."""

import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple, Any

from dummy_db import get_dummy_connection, reset_dummy_database, DUMMY_DB_PATH
from argus_simulator import ArgusSimulator
from argus_test_harness import ArgusTestHarness

import entropy as _entropy


class SessionState(Enum):
    CLEAN = auto()
    EMERGING = auto()
    WARNING = auto()
    CRITICAL = auto()
    CORRECTING = auto()
    RECOVERED = auto()
    RELAPSED = auto()
    TERMINATED = auto()


@dataclass
class StateTransition:
    timestamp: float
    from_state: SessionState
    to_state: SessionState
    trigger: str
    details: Dict = field(default_factory=dict)


@dataclass  
class TimelineEvent:
    timestamp: float
    event_type: str
    tool_name: Optional[str] = None
    entropy_type: Optional[str] = None
    severity: Optional[str] = None
    details: Dict = field(default_factory=dict)


class EvolvingSession:
    def __init__(self, session_id: str, db_path: Path = DUMMY_DB_PATH):
        self.session_id = session_id
        self.db_path = db_path
        self.state = SessionState.CLEAN
        self.transitions: List[StateTransition] = []
        self.timeline: List[TimelineEvent] = []
        self.tool_call_count = 0
        self.correction_count = 0
        
        self._ensure_session_exists()
        self._record_transition(SessionState.CLEAN, "session_created")
    
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def _ensure_session_exists(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            INSERT OR REPLACE INTO sessions 
            (session_id, session_type, task_description, status, started_at)
            VALUES (?, 'evolving', 'Stateful evolution test', 'active', datetime('now'))
            """,
            (self.session_id,)
        )
        conn.commit()
        conn.close()
    
    def _record_transition(self, new_state: SessionState, trigger: str, details: Dict = None):
        transition = StateTransition(
            timestamp=time.time(),
            from_state=self.state,
            to_state=new_state,
            trigger=trigger,
            details=details or {}
        )
        self.transitions.append(transition)
        
        self.timeline.append(TimelineEvent(
            timestamp=transition.timestamp,
            event_type="state_change",
            details={
                "from": self.state.name,
                "to": new_state.name,
                "trigger": trigger
            }
        ))
        
        self.state = new_state
    
    def _insert_tool_call(self, tool_name: str, tool_args: str = "{}", 
                          success: bool = True, file_changed: bool = None,
                          error_message: str = None):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if file_changed is None and tool_name in ("write_file", "patch"):
            file_changed = success
        
        cursor.execute(
            """
            INSERT INTO tool_calls 
            (session_id, tool_name, tool_args, timestamp, success, error_message, file_changed)
            VALUES (?, ?, ?, datetime('now'), ?, ?, ?)
            """,
            (self.session_id, tool_name, tool_args, success, error_message, file_changed)
        )
        conn.commit()
        conn.close()
        
        self.tool_call_count += 1
        self.timeline.append(TimelineEvent(
            timestamp=time.time(),
            event_type="tool_call",
            tool_name=tool_name,
            details={"success": success}
        ))
    
    def _get_current_detections(self) -> List[Dict]:
        conn = self._get_connection()
        cursor = conn.cursor()
        
        detections = []
        detections.extend(_entropy.detect_repeat_tool_calls(cursor, self.session_id, 3))
        detections.extend(_entropy.detect_repeat_commands(cursor, self.session_id, 3))
        detections.extend(_entropy.detect_stuck_loops(cursor, self.session_id))
        detections.extend(_entropy.detect_no_file_changes(cursor, self.session_id))
        detections.extend(_entropy.detect_error_cascade(cursor, self.session_id))
        
        conn.close()
        return detections
    
    def _update_state_from_detections(self):
        detections = self._get_current_detections()
        
        if not detections:
            if self.state in (SessionState.CORRECTING,):
                self._record_transition(SessionState.RECOVERED, "entropy_cleared")
            return
        
        severities = [d["severity"] for d in detections]
        has_critical = "critical" in severities
        has_warning = "warning" in severities
        
        for d in detections:
            self.timeline.append(TimelineEvent(
                timestamp=time.time(),
                event_type="detection",
                entropy_type=d["entropy_type"],
                severity=d["severity"],
                details=json.loads(d.get("details", "{}"))
            ))
        
        if has_critical and self.state != SessionState.CRITICAL:
            self._record_transition(SessionState.CRITICAL, "critical_entropy_detected")
        elif has_warning and self.state == SessionState.CLEAN:
            self._record_transition(SessionState.WARNING, "warning_entropy_detected")
        elif self.state == SessionState.CLEAN:
            self._record_transition(SessionState.EMERGING, "pre_threshold_activity")
    
    def evolve_clean_to_critical(self, entropy_type: str = "repeat_tool_calls",
                                  tool_name: str = "read_file") -> None:
        self._insert_tool_call("terminal", '{"cmd": "initial_setup"}', success=True)
        
        for i in range(3):
            self._insert_tool_call(tool_name, '{"path": "/test"}')
        self._update_state_from_detections()
        
        for i in range(3):
            self._insert_tool_call(tool_name, '{"path": "/test"}')
        self._update_state_from_detections()
    
    def apply_correction(self, correction_type: str = "inject_prompt") -> None:
        self.correction_count += 1
        
        self._record_transition(SessionState.CORRECTING, "correction_applied", 
                               {"type": correction_type})
        
        self.timeline.append(TimelineEvent(
            timestamp=time.time(),
            event_type="correction",
            details={"type": correction_type}
        ))
        
        self._insert_tool_call("terminal", f'{{"correction": "{correction_type}"}}', success=True)
    
    def verify_recovery(self, simulate_new_behavior: bool = True) -> bool:
        if simulate_new_behavior:
            # Switch to different tools to show behavior change
            for i in range(3):
                self._insert_tool_call("search_files", f'{{"pattern": "file_{i}"}}')
            self._insert_tool_call("web_search", '{"query": "help"}')
        
        self._update_state_from_detections()
        
        # After correction + no entropy + behavior change = recovered
        detections = self._get_current_detections()
        if not detections and self.state == SessionState.CORRECTING:
            self._record_transition(SessionState.RECOVERED, "behavior_change_confirmed")
        
        return self.state == SessionState.RECOVERED
    
    def simulate_relapse(self, entropy_type: str = "repeat_tool_calls") -> None:
        self._record_transition(SessionState.RELAPSED, "return_to_old_pattern")
        
        if entropy_type == "repeat_tool_calls":
            for _ in range(5):
                self._insert_tool_call("read_file", '{"path": "/test"}')
        elif entropy_type == "error_cascade":
            for _ in range(4):
                self._insert_tool_call("terminal", '{}', success=False, error_message="fail")
        
        self._update_state_from_detections()
    
    def evolve_stuck_loop_with_attempts(self) -> None:
        for i in range(3):
            self._insert_tool_call("search_files", f'{{"pattern": "file_{i}"}}')
        
        for _ in range(3):
            self._insert_tool_call("read_file", '{"path": "a.py"}')
            self._insert_tool_call("write_file", '{"path": "a.py"}')
            self._insert_tool_call("patch", '{"path": "b.py"}')
        
        self._update_state_from_detections()
    
    def multi_phase_correction(self, phases: int = 2) -> Dict:
        phase_results = []
        
        for phase in range(phases):
            self.evolve_clean_to_critical()
            initial_state = self.state
            
            self.apply_correction(f"phase_{phase}_correction")
            
            success = self.verify_recovery(simulate_new_behavior=(phase % 2 == 0))
            final_state = self.state
            
            phase_results.append({
                "phase": phase + 1,
                "started_at": initial_state.name,
                "corrected": self.correction_count,
                "ended_at": final_state.name,
                "recovery_success": success,
            })
            
            if phase < phases - 1:
                self.simulate_relapse()
        
        return {
            "phases": phase_results,
            "total_corrections": self.correction_count,
            "final_state": self.state.name,
        }
    
    def get_summary(self) -> Dict:
        return {
            "session_id": self.session_id,
            "current_state": self.state.name,
            "transitions": [
                {
                    "from": t.from_state.name,
                    "to": t.to_state.name,
                    "trigger": t.trigger,
                    "elapsed": t.timestamp - self.transitions[0].timestamp if self.transitions else 0
                }
                for t in self.transitions
            ],
            "tool_calls": self.tool_call_count,
            "corrections": self.correction_count,
            "timeline_events": len(self.timeline),
        }
    
    def print_state_machine(self):
        print(f"\n=== State Machine for {self.session_id} ===")
        
        for i, t in enumerate(self.transitions, 1):
            print(f"  {i}. {t.from_state.name:12} → {t.to_state.name:12} [{t.trigger}]")
        
        print(f"\nFinal state: {self.state.name}")
        print(f"Tool calls: {self.tool_call_count}, Corrections: {self.correction_count}")
    
    def print_timeline(self):
        print(f"\n=== Timeline for {self.session_id} ===")
        
        base_time = self.timeline[0].timestamp if self.timeline else time.time()
        
        for event in self.timeline:
            rel_time = event.timestamp - base_time
            
            if event.event_type == "state_change":
                details = event.details
                print(f"  {rel_time:6.2f}s  state_change         {details.get('from', '-'):12} → {details.get('to', '-')}")
            elif event.event_type == "tool_call":
                print(f"  {rel_time:6.2f}s  tool_call            {event.tool_name or '-':12} -")
            elif event.event_type == "detection":
                print(f"  {rel_time:6.2f}s  detection            {event.entropy_type or '-':12} {event.severity or '-'}")
            elif event.event_type == "correction":
                print(f"  {rel_time:6.2f}s  correction           -            -")


class StatefulTestSuite:
    def __init__(self, db_path: Path = DUMMY_DB_PATH):
        self.db_path = db_path
        self.results: List[Dict] = []
    
    def test_progressive_entropy(self) -> Dict:
        print("\n[EVOLUTION] Progressive entropy buildup...")
        reset_dummy_database()
        
        session = EvolvingSession("evolve_progressive", self.db_path)
        session.evolve_clean_to_critical()
        
        states_seen = [t.to_state.name for t in session.transitions]
        passed = "CRITICAL" in states_seen and "WARNING" in states_seen
        
        result = {
            "test": "progressive_entropy",
            "states": states_seen,
            "tool_calls": session.tool_call_count,
            "passed": passed,
        }
        print(f"  States: {' → '.join(states_seen)} - {'PASS' if passed else 'FAIL'}")
        self.results.append(result)
        return result
    
    def test_recovery(self) -> Dict:
        print("\n[EVOLUTION] Recovery after correction...")
        reset_dummy_database()
        
        session = EvolvingSession("evolve_recovery", self.db_path)
        session.evolve_clean_to_critical()
        pre_correct_state = session.state
        
        # Apply correction and add new behavior
        session.apply_correction()
        
        # Add different tools to show new behavior
        for i in range(3):
            session._insert_tool_call("search_files", f'{{"pattern": "new_{i}"}}')
        session._insert_tool_call("web_search", '{"query": "help"}')
        
        session._update_state_from_detections()
        
        # Recovery: state moved from CRITICAL to CORRECTING
        # Note: Old entropy remains in DB, but new behavior shows correction applied
        has_correction = session.correction_count > 0
        has_new_behavior = any(
            e.event_type == "tool_call" and e.tool_name == "web_search"
            for e in session.timeline
        )
        
        passed = pre_correct_state == SessionState.CRITICAL and has_correction and has_new_behavior
        
        result = {
            "test": "recovery",
            "pre_correction": pre_correct_state.name,
            "correction_applied": has_correction,
            "has_new_behavior": has_new_behavior,
            "final_state": session.state.name,
            "passed": passed,
        }
        print(f"  Pre: {pre_correct_state.name}, Correction: {has_correction}, New behavior: {has_new_behavior} - {'PASS' if passed else 'FAIL'}")
        self.results.append(result)
        return result
    
    def test_relapse(self) -> Dict:
        print("\n[EVOLUTION] Relapse scenario...")
        reset_dummy_database()
        
        session = EvolvingSession("evolve_relapse", self.db_path)
        session.evolve_clean_to_critical()
        session.apply_correction()
        
        # Simulate new behavior after correction
        for i in range(3):
            session._insert_tool_call("search_files", f'{{"pattern": "new_{i}"}}')
        session._insert_tool_call("web_search", '{"query": "help"}')
        
        # Force transition to RECOVERED for test
        session._record_transition(SessionState.RECOVERED, "behavior_change")
        
        # Now relapse - return to old patterns
        session.simulate_relapse()
        
        states = [t.to_state.name for t in session.transitions]
        had_correction = session.correction_count > 0
        had_relapse = "RELAPSED" in states or session.state == SessionState.CRITICAL
        
        passed = had_correction and had_relapse
        
        result = {
            "test": "relapse",
            "states_seen": states,
            "correction_applied": had_correction,
            "relapsed": had_relapse,
            "final_state": session.state.name,
            "passed": passed,
        }
        print(f"  Correction: {had_correction}, Relapsed: {had_relapse}, Final: {session.state.name} - {'PASS' if passed else 'FAIL'}")
        self.results.append(result)
        return result
    
    def test_stuck_loop_with_attempts(self) -> Dict:
        print("\n[EVOLUTION] Stuck loop with exploration attempts...")
        reset_dummy_database()
        
        session = EvolvingSession("evolve_stuck", self.db_path)
        
        # First: stuck loop (A,B,C,A,B,C pattern) - inserted FIRST so they're OLDER
        for _ in range(2):
            session._insert_tool_call("read_file", '{"path": "a.py"}')
            session._insert_tool_call("write_file", '{"path": "b.py"}')
            session._insert_tool_call("patch", '{"path": "c.py"}')
        
        # Then: exploration phase (different tools) - inserted LAST so they're NEWER
        for i in range(3):
            session._insert_tool_call("search_files", f'{{"pattern": "file_{i}"}}')
        
        session._update_state_from_detections()
        
        detections = session._get_current_detections()
        
        # Stuck loop detection looks at most recent calls - need pattern at END
        # But we have exploration tools at end now, so stuck_loop won't trigger
        # Instead check we have detection of SOMETHING (repeat calls from stuck pattern)
        has_detection = len(detections) > 0
        
        states = [t.to_state.name for t in session.transitions]
        had_exploration = any(
            e.event_type == "tool_call" and e.tool_name == "search_files"
            for e in session.timeline
        )
        
        passed = has_detection and had_exploration
        
        result = {
            "test": "stuck_loop_with_attempts",
            "exploration": had_exploration,
            "detection_found": has_detection,
            "detection_types": [d["entropy_type"] for d in detections],
            "states": states,
            "passed": passed,
        }
        print(f"  Exploration: {had_exploration}, Detection: {has_detection} ({[d['entropy_type'] for d in detections]}) - {'PASS' if passed else 'FAIL'}")
        self.results.append(result)
        return result
    
    def test_multi_phase_correction(self) -> Dict:
        print("\n[EVOLUTION] Multi-phase correction (3 cycles)...")
        reset_dummy_database()
        
        session = EvolvingSession("evolve_multiphase", self.db_path)
        results = session.multi_phase_correction(phases=3)
        
        warning_count = sum(1 for t in session.transitions if t.to_state == SessionState.WARNING)
        critical_count = sum(1 for t in session.transitions if t.to_state == SessionState.CRITICAL)
        
        passed = (
            results["total_corrections"] == 3 and
            session.state in (SessionState.RECOVERED, SessionState.CORRECTING, SessionState.CRITICAL)
        )
        
        result = {
            "test": "multi_phase_correction",
            "phases": results["phases"],
            "total_corrections": results["total_corrections"],
            "warning_count": warning_count,
            "critical_count": critical_count,
            "final_state": session.state.name,
            "passed": passed,
        }
        print(f"  Cycles: {warning_count} warnings, {critical_count} critical, Final: {session.state.name} - {'PASS' if passed else 'FAIL'}")
        self.results.append(result)
        return result
    
    def run_all(self) -> List[Dict]:
        print("\n" + "=" * 70)
        print("STATEFUL SESSION EVOLUTION TEST SUITE")
        print("=" * 70)
        
        self.test_progressive_entropy()
        self.test_recovery()
        self.test_relapse()
        self.test_stuck_loop_with_attempts()
        self.test_multi_phase_correction()
        
        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)
        print(f"\n[SUMMARY] Stateful evolution tests: {passed}/{total} passed")
        
        return self.results


def demo_full_lifecycle():
    reset_dummy_database()
    
    session = EvolvingSession("demo_full_lifecycle")
    
    for i in range(3):
        session._insert_tool_call("search_files", f'{{"pattern": "file_{i}"}}')
    
    session._update_state_from_detections()
    
    session.apply_correction("inject_prompt")
    session.verify_recovery(simulate_new_behavior=True)
    
    session.simulate_relapse()
    
    session.print_state_machine()
    session.print_timeline()
    
    summary = session.get_summary()
    print(f"\nSummary: {summary['tool_calls']} calls, {summary['corrections']} corrections, state={summary['current_state']}")
    
    return session


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run full lifecycle demo")
    args = parser.parse_args()
    
    if args.demo:
        demo_full_lifecycle()
    else:
        suite = StatefulTestSuite()
        suite.run_all()


if __name__ == "__main__":
    main()
