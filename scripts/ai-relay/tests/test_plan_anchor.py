"""Tests for plan-anchor.py — subprocess invocation for real exit codes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "plan-anchor.py"

SAMPLE_PLAN = """\
# Plan — GRD · test fixture

> **plan_id: GRD** · branch: feature/plan-guardrails

## กติกาเหล็กของแผนนี้ — fixture

1. **เลขงานต้องขึ้นต้นด้วย plan_id** เช่น `GRD-P1-I1`
2. **verified = มีแถว gate-run เท่านั้น**

## GRD-P1 — fixture phase

- **GRD-P1-I1** plan-anchor script
  - allowed: `scripts/ai-relay/plan-anchor.py`
  - forbidden: `relay-call.py`
  - verify: `python3 -m pytest scripts/ai-relay/tests/test_plan_anchor.py -q` → exit 0
- **GRD-P1-I2** relay integration
  - allowed: `scripts/ai-relay/relay-call.py`
"""


def run_anchor(
    task_id: str,
    plan_path: Path,
    *,
    emit_brief: bool = False,
) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(SCRIPT), "--task-id", task_id, "--plan", str(plan_path)]
    if emit_brief:
        cmd.append("--emit-brief")
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def first_json_line(stdout: str) -> dict:
    line = stdout.splitlines()[0]
    return json.loads(line)


@pytest.fixture
def plan_file(tmp_path: Path) -> Path:
    path = tmp_path / "plan.md"
    path.write_text(SAMPLE_PLAN, encoding="utf-8")
    return path


def test_ok_exit_zero(plan_file: Path) -> None:
    result = run_anchor("GRD-P1-I1", plan_file)
    payload = first_json_line(result.stdout)

    assert result.returncode == 0
    assert payload["status"] == "ok"
    assert payload["plan_id"] == "GRD"
    assert payload["task_id"] == "GRD-P1-I1"


def test_unknown_task_off_plan(plan_file: Path) -> None:
    result = run_anchor("GRD-P9-Z9", plan_file)
    payload = first_json_line(result.stdout)

    assert result.returncode == 1
    assert payload["status"] == "off_plan"
    assert payload["plan_id"] == "GRD"


def test_wrong_prefix_off_plan(plan_file: Path) -> None:
    result = run_anchor("XYZ-P1-I1", plan_file)
    payload = first_json_line(result.stdout)

    assert result.returncode == 1
    assert payload["status"] == "off_plan"


def test_missing_plan_no_plan(tmp_path: Path) -> None:
    missing = tmp_path / "missing.md"
    result = run_anchor("GRD-P1-I1", missing)
    payload = first_json_line(result.stdout)

    assert result.returncode == 2
    assert payload["status"] == "no_plan"
    assert payload["plan_id"] is None


def test_plan_without_plan_id_no_plan(tmp_path: Path) -> None:
    path = tmp_path / "plan.md"
    path.write_text("# Plan\n\nno plan id here\n", encoding="utf-8")
    result = run_anchor("GRD-P1-I1", path)
    payload = first_json_line(result.stdout)

    assert result.returncode == 2
    assert payload["status"] == "no_plan"


def test_i1_does_not_match_inside_i10(tmp_path: Path) -> None:
    path = tmp_path / "plan.md"
    path.write_text(
        """\
# Plan — GRD · word-boundary fixture

> **plan_id: GRD**

## GRD-P1

- **GRD-P1-I10** unrelated issue only
""",
        encoding="utf-8",
    )
    result = run_anchor("GRD-P1-I1", path)
    payload = first_json_line(result.stdout)

    assert result.returncode == 1
    assert payload["status"] == "off_plan"
    assert payload["plan_id"] == "GRD"


def test_i1_ok_when_i10_also_present(tmp_path: Path) -> None:
    path = tmp_path / "plan.md"
    path.write_text(
        """\
# Plan — GRD · both-issues fixture

> **plan_id: GRD**

## GRD-P1

- **GRD-P1-I1** plan-anchor script
- **GRD-P1-I10** unrelated issue
""",
        encoding="utf-8",
    )
    result = run_anchor("GRD-P1-I1", path)
    payload = first_json_line(result.stdout)

    assert result.returncode == 0
    assert payload["status"] == "ok"
    assert payload["plan_id"] == "GRD"
    assert payload["task_id"] == "GRD-P1-I1"


def test_emit_brief_contains_rules_and_verify(tmp_path: Path) -> None:
    path = tmp_path / "plan.md"
    path.write_text(
        """\
# Plan — GRD · emit-brief fixture

> **plan_id: GRD** · branch: feature/plan-guardrails

## กติกาเหล็กของแผนนี้ — fixture

1. **เลขงานต้องขึ้นต้นด้วย plan_id** เช่น `GRD-P1-I1`

## GRD-P1 — fixture phase

- **GRD-P1-I1** plan-anchor script
  - allowed: `scripts/ai-relay/plan-anchor.py`
  - forbidden: `relay-call.py`
  - verify: `python3 -m pytest scripts/ai-relay/tests/test_plan_anchor.py -q` → exit 0
- **GRD-P1-I2** relay integration
  - allowed: `scripts/ai-relay/relay-call.py`
""",
        encoding="utf-8",
    )
    result = run_anchor("GRD-P1-I1", path, emit_brief=True)
    payload = first_json_line(result.stdout)
    brief = "\n".join(result.stdout.splitlines()[1:])

    assert result.returncode == 0
    assert payload["status"] == "ok"
    assert "กติกาเหล็ก" in brief
    assert "allowed: `scripts/ai-relay/plan-anchor.py`" in brief
    assert "verify: `python3 -m pytest scripts/ai-relay/tests/test_plan_anchor.py -q`" in brief
    assert "forbidden: `relay-call.py`" in brief
    assert "GRD-P1-I2" not in brief


def test_crlf_plan_ok(tmp_path: Path) -> None:
    path = tmp_path / "plan.md"
    path.write_bytes(SAMPLE_PLAN.replace("\n", "\r\n").encode("utf-8"))
    result = run_anchor("GRD-P1-I1", path)

    assert result.returncode == 0
    assert first_json_line(result.stdout)["status"] == "ok"