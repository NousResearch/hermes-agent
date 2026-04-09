# Argus Test Simulation Framework

Isolated test environment for validating ARGUS entropy detection without touching production data.

## Overview

This framework provides:
- **dummy_db.py**: Isolated SQLite database (`dummy_argus.db`)
- **argus_simulator.py**: Test data generator for all 6 entropy types
- **argus_test_harness.py**: Validation harness for detection algorithms

All components work exclusively with `dummy_argus.db` — zero production data access.

## Entropy Types Tested

| Type | Detection Trigger | Severity |
|------|------------------|----------|
| `repeat_tool_calls` | 3+ identical tool calls in 10min | warning→critical at 5 |
| `repeat_commands` | 3+ identical terminal commands in 10min | warning→critical at 5 |
| `stuck_loop` | Repeating pattern (A,B,C,A,B,C) | critical |
| `no_file_changes` | write_file/patch with file_changed=FALSE | critical |
| `error_cascade` | 3+ consecutive tool failures | warning→critical at 5 |
| `budget_pressure` | 70%+ iteration burn with entropy | warning/critical |

## Quick Start

```bash
cd ~/Projects/hermes-dev/argus/tests/simulation

# Initialize dummy database
python dummy_db.py

# Quick smoke test (all suites)
python run_all_tests.py --quick

# Run specific test suites
python run_all_tests.py --edge       # Threshold & boundary tests
python run_all_tests.py --stress     # Volume & performance
python run_all_tests.py --stateful   # Session evolution
python run_all_tests.py --matrix     # Parameterized matrix

# Run everything
python run_all_tests.py

# Demo: Full session lifecycle visualization
python argus_stateful.py --demo

# Legacy: Individual test harnesses
python argus_test_harness.py --all
python argus_test_harness.py --inspect test_repeat_tools
```

## Available Scenarios

```bash
python argus_simulator.py --list

# Output:
# repeat_tool_calls  - Same tool called 3+ times with identical args
# repeat_commands    - Same terminal command executed 3+ times
# stuck_loop         - Repeating sequence of tool calls (A,B,C,A,B,C)
# no_file_changes    - write_file/patch with file_changed=FALSE
# error_cascade      - 3+ consecutive tool failures
# budget_pressure    - High iteration burn rate with entropy
# combined           - Multiple entropy types in one session
```

## Architecture

```
simulation/
├── dummy_argus.db              # Isolated test database (gitignored)
├── dummy_db.py                 # DB initialization & connection
├── argus_simulator.py          # Test data generator
├── argus_test_harness.py       # Detection validator
├── argus_edge_cases.py         # Threshold & boundary tests
├── argus_stress_tests.py       # Volume & performance tests
├── argus_stateful.py           # Stateful session evolution
├── run_all_tests.py            # Unified test runner
└── README.md                   # This file
```

### Data Flow

1. `dummy_db.py` creates `dummy_argus.db` with full Argus schema
2. `argus_simulator.py` populates it with synthetic error patterns
3. `argus_test_harness.py` runs entropy detection and validates results

### Isolation Guarantees

- Hardcoded path: `DUMMY_DB_PATH` points only to `simulation/dummy_argus.db`
- No imports from production `~/hermes` or `~/.hermes`
- No access to `holographic_memory.db` or `state.db`
- All sessions prefixed with `test_*` or `sim_*`

## Python API

```python
from argus_simulator import ArgusSimulator
from argus_test_harness import ArgusTestHarness

# Generate test data
sim = ArgusSimulator()
sim.scenario_error_cascade("my_test", consecutive_errors=5)
sim.close()

# Validate detection
harness = ArgusTestHarness()
detections = harness.detect_entropy_for_session("my_test")
for d in detections:
    print(f"[{d['severity']}] {d['entropy_type']}")
harness.close()
```

## Validation Results

```
$ python argus_test_harness.py --all

============================================================
ARGUS VALIDATION SUITE
============================================================

[TEST] Running: repeat_tool_calls
  Status: PASS
  Expected: repeat_tool_calls
  Found: repeat_tool_calls

[TEST] Running: repeat_commands
  Status: PASS
  Expected: repeat_commands
  Found: repeat_commands

[TEST] Running: stuck_loop
  Status: PASS
  Expected: stuck_loop
  Found: stuck_loop

[TEST] Running: no_file_changes
  Status: PASS
  Expected: no_file_changes
  Found: no_file_changes

[TEST] Running: error_cascade
  Status: PASS
  Expected: error_cascade
  Found: no_file_changes, error_cascade

============================================================
VALIDATION SUMMARY
============================================================
Total tests: 5
Passed: 5
Failed: 0
```

## Error Payload Samples

### repeat_tool_calls
```json
{
  "tool_name": "read_file",
  "tool_args": "{\"path\": \"/etc/hosts\"}",
  "timestamp": "2026-04-09 12:00:00",
  "success": true
}
```
Generated 5× identical → detected as `critical`

### error_cascade
```json
{
  "tool_name": "write_file",
  "success": false,
  "error_message": "FileNotFoundError: /missing/path"
}
```
3+ consecutive errors → detected as `warning` (5+ → `critical`)

### no_file_changes
```json
{
  "tool_name": "patch",
  "file_changed": false,
  "success": true
}
```
Write operations with `file_changed=FALSE` → detected as `critical`

## Stateful Session Evolution

The `argus_stateful.py` module provides realistic session lifecycle simulation:

```python
from argus_stateful import EvolvingSession

# Create evolving session
session = EvolvingSession("my_session")

# Phase 1: Clean exploration
session.evolve_clean_to_critical("repeat_tool_calls")
# States: CLEAN → WARNING → CRITICAL

# Phase 2: Correction applied
session.apply_correction("inject_prompt")
# State: CORRECTING

# Phase 3: Recovery
session.verify_recovery()
# State: RECOVERED (if behavior changed)

# Phase 4: Relapse
session.simulate_relapse("repeat_tool_calls")
# State: CRITICAL

# Multi-phase cycles
session.multi_phase_correction(phases=3)
# Multiple entropy/correction cycles

# Analysis
session.print_timeline()       # Chronological event log
session.print_state_machine()  # State transitions
summary = session.get_summary()
```

### Session States

| State | Description | Transition Trigger |
|-------|-------------|-------------------|
| `CLEAN` | No entropy detected | Initial state |
| `EMERGING` | Early signs (1-2 calls) | Pre-threshold activity |
| `WARNING` | Threshold crossed | 3+ repeat calls / errors |
| `CRITICAL` | Severe entropy | 5+ calls, stuck loops |
| `CORRECTING` | Correction in progress | Prompt injection |
| `RECOVERED` | Back to clean | Behavior change confirmed |
| `RELAPSED` | Returned to entropy | Old patterns resumed |
| `TERMINATED` | Session killed | Max restarts reached |

## Development

Add new scenarios to `ArgusSimulator`:

```python
def scenario_my_pattern(self, session_id: str = "test_my_pattern"):
    """Generate my custom entropy pattern."""
    self._insert_session(session_id, "test", "My pattern test")
    
    # Insert tool calls that trigger detection
    for i in range(5):
        self._insert_tool_call(session_id, "my_tool", "{}")
    
    print(f"[SIM] Generated my pattern for {session_id}")
```

Then register in `run_scenario()` and add validation test case.

## CI/CD Integration

```bash
#!/bin/bash
set -e
cd ~/Projects/hermes-dev/argus/tests/simulation

# Reset and validate
python dummy_db.py
python argus_test_harness.py --all

echo "All Argus detections validated"
```
