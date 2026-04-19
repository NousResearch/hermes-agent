"""Code-evolution fitness template.

Runs a candidate Python solution against a hidden pytest suite via the
sandbox. Each evaluation spawns a fresh ``python3`` subprocess with
rlimit caps (CPU, memory) so an infinite loop or runaway allocation
cannot stall or crash the evaluator.

Point ``TEST_SOURCE`` at the pytest test file content for your problem.
The fitness is the fraction of tests that pass; a runtime failure
outside pytest returns 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the skill's sandbox module importable without packaging.
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import sandbox  # noqa: E402
from evolver_sdk import fitness_spec  # noqa: E402


TEST_SOURCE = """
from solution import solve


def test_positive():
    assert solve(3) == 9


def test_zero():
    assert solve(0) == 0


def test_negative():
    assert solve(-4) == 16
"""


@fitness_spec(held_out_frac=0.2, timeout_s=60)
def fitness(candidate: str, context: dict) -> float:
    _, pass_frac = sandbox.run_pytest_suite(
        candidate_code=candidate,
        test_code=TEST_SOURCE,
        timeout_s=30,
        cpu_s=25,
        mem_mb=512,
    )
    return pass_frac
