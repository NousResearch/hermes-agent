# Argus Test Simulation Package
"""
Isolated test environment for Argus monitoring system.

This package provides:
- dummy_db.py: Isolated SQLite database for testing
- argus_simulator.py: Test data generator for all entropy types
- argus_test_harness.py: Detection validator
- argus_edge_cases.py: Threshold, time window, and malformed data tests
- argus_stress_tests.py: Volume, performance, and matrix tests

All components work with DUMMY_DB_PATH only — no production data access.
"""

from .dummy_db import DUMMY_DB_PATH, init_dummy_database, get_dummy_connection, reset_dummy_database
from .argus_simulator import ArgusSimulator
from .argus_test_harness import ArgusTestHarness
from .argus_edge_cases import (
    ThresholdBoundaryTests,
    TimeBoundaryTests,
    MalformedDataTests,
    run_all_edge_case_tests
)
from .argus_stress_tests import (
    VolumeStressTests,
    ParameterizedMatrixTests,
    PerformanceMetrics
)
from .argus_stateful import (
    SessionState,
    StateTransition,
    TimelineEvent,
    EvolvingSession,
    StatefulTestSuite,
)

__all__ = [
    # Database
    "DUMMY_DB_PATH",
    "init_dummy_database",
    "get_dummy_connection",
    "reset_dummy_database",
    # Simulator
    "ArgusSimulator",
    # Harness
    "ArgusTestHarness",
    # Edge Cases
    "ThresholdBoundaryTests",
    "TimeBoundaryTests",
    "MalformedDataTests",
    "run_all_edge_case_tests",
    # Stress Tests
    "VolumeStressTests",
    "ParameterizedMatrixTests",
    "PerformanceMetrics",
    # Stateful Evolution
    "SessionState",
    "StateTransition",
    "TimelineEvent",
    "EvolvingSession",
    "StatefulTestSuite",
]
