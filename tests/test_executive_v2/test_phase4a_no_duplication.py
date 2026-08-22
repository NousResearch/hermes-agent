"""Phase 4A non-duplication gates (12 tests).

Verifies that Phase 4A's new modules DO NOT:

* modify protected modules (hermes_cli/goals.py, orchestrator/*,
  orchestrator_interface.py, hermes_cli/kanban*.py,
  completion_observation_trace.py);
* import or call Kanban command/swarms;
* import or call delegate_task, worker_runner, pilot_bridge,
  batch_runner;
* use LLM client libraries, subprocess, network libs;
* import GBrain, Obsidian, NotebookLM;
* create new DB tables.

Plus cross-file footprint check + consolidation gate.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RISK_PATH = REPO_ROOT / "agent" / "executive" / "risk.py"
APPROVAL_GATES_PATH = REPO_ROOT / "agent" / "executive" / "approval_gates.py"
POLICY_PATH = REPO_ROOT / "agent" / "executive" / "policy.py"
TYPES_PATH = REPO_ROOT / "agent" / "executive" / "types.py"
STATE_STORAGE_PATH = REPO_ROOT / "agent" / "executive" / "state_storage.py"

# Pre-Phase-4A SHA256 of the protected modules (from Phase 3 promotion).
PRE_PHASE4A_HASHES = {
    "hermes_cli/goals.py": "ac757e82fc6cbfa0215ad695007f0d77c22d89fa5ef3f02f05a498e771d733bc",  # BASELINE_AUTHORITY=SYNTHETIC_CURRENT_MAIN_PLUS_PR80271_DELTA@0b33ee88 (E6C3 forward-port; Batch17 goals.py rebind: legitimate evolution, protects current-main against modification)
    "agent/orchestrator_interface.py": "ad55af1bbfb9e44359e4efad8ff41d21726d72dad2ab3ba9ca2189eeab5f8081",
    "agent/completion_observation_trace.py": "7cd3319e4c3cbf5c3245ae8bff58d8ad85647ce27995e07504acb1b7539da751",
    "agent/orchestrator/__init__.py": "619800da4587505b5128d02cefe00791e5ce233e7d940fe944e2ac19ffa3e604",
    "agent/orchestrator/batch_runner.py": "29407c9597325a1ebd33c06c3c1f7a8613183bbb30c45e35625257bdebdbe8a0",
    "agent/orchestrator/dispatcher.py": "e2cf31f46e5cfd469f871b9b243ef13ab6b2b698ed41888ce4684e5d9461ac6e",
    "agent/orchestrator/handlers.py": "e5911a7785b2e971cf813df7c46a872c29c62656c7657faa3df7d67f1d66f8f3",
    "agent/orchestrator/kanban_adapter.py": "e6dedfc2264a397cf1bbbae6ff906afdaee5b6f2d740bb7789563d65da97b273",
    "agent/orchestrator/pilot_bridge.py": "e85ee2802ae8eda8951b50094ba9c87383054e43117372738cb56fdcb512c604",
    "agent/orchestrator/worker_runner.py": "fe0d250197a9887f79b41df57e088ea1defbb2d26bfbfec302417489fa99fbdb",
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ──────────────────────────────────────────────────────────────────────
# Gates 1-4: protected modules byte-intact
# ──────────────────────────────────────────────────────────────────────

def test_gate1_goals_py_byte_intact():
    actual = _sha256(REPO_ROOT / "hermes_cli" / "goals.py")
    assert actual == PRE_PHASE4A_HASHES["hermes_cli/goals.py"]


def test_gate2_orchestrator_interface_byte_intact():
    actual = _sha256(REPO_ROOT / "agent" / "orchestrator_interface.py")
    assert actual == PRE_PHASE4A_HASHES["agent/orchestrator_interface.py"]


def test_gate3_completion_observation_trace_byte_intact():
    actual = _sha256(REPO_ROOT / "agent" / "completion_observation_trace.py")
    assert actual == PRE_PHASE4A_HASHES["agent/completion_observation_trace.py"]


def test_gate4_orchestrator_dir_byte_intact():
    for name, expected in PRE_PHASE4A_HASHES.items():
        if not name.startswith("agent/orchestrator/"):
            continue
        path = REPO_ROOT / name
        actual = _sha256(path)
        assert actual == expected, f"{name} drifted: {actual} != {expected}"


# ──────────────────────────────────────────────────────────────────────
# Gate 5: hermes_cli/kanban*.py byte-intact (all 6 files)
# ──────────────────────────────────────────────────────────────────────

def test_gate5_kanban_cli_byte_intact():
    """All hermes_cli/kanban*.py modules byte-intact."""
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", "hermes_cli/kanban*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    changed = [line for line in result.stdout.splitlines() if line.strip()]
    assert not changed, f"kanban*.py files modified: {changed}"


# ──────────────────────────────────────────────────────────────────────
# Gate 6: no kanban_command references in new modules
# ──────────────────────────────────────────────────────────────────────

KANBAN_FORBIDDEN_TOKENS = (
    "kanban_command",
    "_cmd_create",
    "_cmd_swarm",
    "create_swarm",
    "kanban_db.create_task",
    "kanban_decompose",
    "kanban_swarm",
)


def test_gate6_no_kanban_command_in_new_modules():
    for path in (RISK_PATH, APPROVAL_GATES_PATH, POLICY_PATH):
        src = _read(path)
        for token in KANBAN_FORBIDDEN_TOKENS:
            assert token not in src, f"{path.name} contains forbidden token: {token}"


# ──────────────────────────────────────────────────────────────────────
# Gate 7: no delegate_task, worker_runner, pilot_bridge, batch_runner
# ──────────────────────────────────────────────────────────────────────

WORKER_FORBIDDEN_TOKENS = (
    "delegate_task",
    "worker_runner",
    "pilot_bridge",
    "batch_runner",
)


def test_gate7_no_worker_invocation_in_new_modules():
    for path in (RISK_PATH, APPROVAL_GATES_PATH, POLICY_PATH):
        src = _read(path)
        for token in WORKER_FORBIDDEN_TOKENS:
            assert token not in src, f"{path.name} contains forbidden token: {token}"


# ──────────────────────────────────────────────────────────────────────
# Gate 8: no LLM client, subprocess, network
# ──────────────────────────────────────────────────────────────────────

EXTERNAL_FORBIDDEN_TOKENS = (
    "from anthropic",
    "import openai",
    "import urllib",
    "import requests",
    "import httpx",
    "auxiliary_client",
    "os.system",
    "subprocess.run",
    "subprocess.Popen",
    "subprocess.call",
)


def test_gate8_no_external_calls_in_new_modules():
    for path in (RISK_PATH, APPROVAL_GATES_PATH, POLICY_PATH):
        src = _read(path)
        for token in EXTERNAL_FORBIDDEN_TOKENS:
            assert token not in src, f"{path.name} contains forbidden token: {token}"


# ──────────────────────────────────────────────────────────────────────
# Gate 9: no GBrain, Obsidian, NotebookLM
# ──────────────────────────────────────────────────────────────────────

KNOWLEDGE_FORBIDDEN_TOKENS = ("gbrain", "obsidian", "notebooklm")


def test_gate9_no_external_knowledge_in_new_modules():
    for path in (RISK_PATH, APPROVAL_GATES_PATH, POLICY_PATH):
        src = _read(path).lower()
        for token in KNOWLEDGE_FORBIDDEN_TOKENS:
            assert token not in src, f"{path.name} references forbidden knowledge source: {token}"


# ──────────────────────────────────────────────────────────────────────
# Gate 10: no DB schema mutations in new modules
# ──────────────────────────────────────────────────────────────────────

DB_FORBIDDEN_PATTERNS = ("CREATE TABLE", "ALTER TABLE", "CREATE INDEX")


def test_gate10_no_db_schema_mutations_in_new_modules():
    for path in (RISK_PATH, APPROVAL_GATES_PATH, POLICY_PATH):
        src = _read(path)
        for pattern in DB_FORBIDDEN_PATTERNS:
            assert pattern not in src, f"{path.name} contains forbidden DDL pattern: {pattern}"


# ──────────────────────────────────────────────────────────────────────
# Gate 11: cross-file footprint — new modules don't reference
# other protected paths
# ──────────────────────────────────────────────────────────────────────

def test_gate11_policy_does_not_reference_protected_paths():
    """policy.py / approval_gates.py must not reference other
    protected paths (no path imports outside Phase 4A scope)."""
    for path in (RISK_PATH, APPROVAL_GATES_PATH, POLICY_PATH):
        src = _read(path)
        forbidden_paths = (
            "hermes_cli/goals.py",
            "agent/orchestrator/",
            "agent/orchestrator_interface",
            "hermes_cli/kanban",
        )
        for path_token in forbidden_paths:
            assert path_token not in src, (
                f"{path.name} references forbidden path: {path_token}"
            )


# ──────────────────────────────────────────────────────────────────────
# Gate 12: ApprovalGateEvaluator consolidates Phase 1+2+3's 4 layers
# ──────────────────────────────────────────────────────────────────────

def test_gate12_approval_gates_consolidate_phase_1_2_3_layers():
    """The 8-layer ApprovalGateEvaluator must include the 4 Phase
    1+2+3 layers (default, STRATEGIC, HIGH_RISK, cross-session)."""
    src = _read(APPROVAL_GATES_PATH)
    assert "approver_id" in src, "Layer 1 (default) missing"
    assert "STRATEGIC" in src, "Layer 2 (STRATEGIC) missing"
    assert "HIGH_RISK" in src or "risk_score" in src, "Layer 3 (HIGH_RISK) missing"
    assert "cross_session" in src, "Layer 4 (cross-session) missing"