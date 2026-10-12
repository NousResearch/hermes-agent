"""``known_failure`` under strict acceptance: the gap fails the run instead of XFAILing.

A real pytest process runs a cell whose final assertion is wrapped in ``known_failure``;
only the environment of that process changes between the cases.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]

_CELL = '''
from tests.e2e.core._pending_fixes import known_failure, known_gate

GAP = (r"marker survived", "upd-txn: fixed by the batch")


def test_gap_still_open():
    with known_failure(*GAP):
        assert False, "orphaned_update: marker survived the update"


def test_other_failure_is_never_excused():
    with known_failure(*GAP):
        assert False, "the update timed out"


def test_fixed_gap_passes():
    with known_failure(*GAP):
        assert True


def test_gate_table_follows_the_same_mode():
    with known_gate({"cell": GAP}, "cell"):
        assert False, "marker survived again"


def test_gap_another_owner_tracks():
    with known_failure(r"slow fetch", "#123254: tracked elsewhere"):
        assert False, "slow fetch downloaded the whole history"
'''


def _run(tmp_path: Path, strict: str | None) -> dict[str, str]:
    cell = tmp_path / "test_cell.py"
    cell.write_text(_CELL, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTEST_", "HERMES_E2E_"))}
    env["PYTHONPATH"] = str(_REPO)
    if strict is not None:
        env["HERMES_E2E_STRICT_ACCEPTANCE"] = strict
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(cell), "-rA", "-q", "-p", "no:cacheprovider",
         "-p", "no:randomly", "-o", "addopts=", "--rootdir", str(tmp_path)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    if strict and strict not in ("0",):
        assert "known gap is not excused" not in proc.stdout or f"HERMES_E2E_STRICT_ACCEPTANCE={strict.strip()}:" in proc.stdout
    outcomes = {}
    for line in proc.stdout.splitlines():
        for verdict in ("PASSED", "FAILED", "XFAIL"):
            if line.startswith(verdict + " "):
                outcomes[line.split("::", 1)[1].split()[0]] = verdict
    assert len(outcomes) == 5, proc.stdout + proc.stderr
    return outcomes


@pytest.mark.parametrize("strict", [None, "", "0", "other-batch"])
def test_default_run_xfails_only_the_known_gap(tmp_path, strict):
    assert _run(tmp_path, strict) == {
        "test_gap_still_open": "XFAIL",
        "test_other_failure_is_never_excused": "FAILED",
        "test_fixed_gap_passes": "PASSED",
        "test_gate_table_follows_the_same_mode": "XFAIL",
        "test_gap_another_owner_tracks": "XFAIL",
    }


@pytest.mark.parametrize("strict", ["upd-txn", "other-batch, upd-txn"])
def test_strict_acceptance_fails_the_batch_owned_gaps_only(tmp_path, strict):
    assert _run(tmp_path, strict) == {
        "test_gap_still_open": "FAILED",
        "test_other_failure_is_never_excused": "FAILED",
        "test_fixed_gap_passes": "PASSED",
        "test_gate_table_follows_the_same_mode": "FAILED",
        "test_gap_another_owner_tracks": "XFAIL",
    }


def test_strict_acceptance_1_fails_every_known_gap(tmp_path):
    outcomes = _run(tmp_path, "1")
    assert outcomes.pop("test_fixed_gap_passes") == "PASSED"
    assert set(outcomes.values()) == {"FAILED"}
