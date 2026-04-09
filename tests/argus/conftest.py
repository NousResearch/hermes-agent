"""Pytest configuration for Argus tests.

Centralizes path setup so individual test files don't need sys.path.insert.
"""

import sys
from pathlib import Path

# Setup paths once at pytest startup
# tests/argus/ -> tests/ -> hermes-dev/
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "argus"))

# Also add tests/argus/simulation/ for simulation imports
SIMULATION_DIR = Path(__file__).parent / "simulation"
if SIMULATION_DIR.exists():
    sys.path.insert(0, str(SIMULATION_DIR))
